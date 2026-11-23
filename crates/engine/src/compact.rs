//! Fast branchless arithmetic primitives, precomputed lookup tables, and the
//! 256-byte fixed-size `CompactBattle` search kernel for `[Gen 2] NC 2000`
//! (`gen2stadium2`) and Gym Leader's Castle PBS (`PS.cgi`).
//!
//! Key design principles:
//! 1. **Branchless arithmetic over control-flow branching**:
//!    - Turn action ordering (`ActionSortKey` for PS, `action_sort_key_exact` for Castle)
//!      packs phase, priority, Quick Claw activation, effective speed (exact `x3360` rational
//!      speed in Castle matching Perl `f64`), and tie-break bit into a single integer.
//!    - Stat computation (`get_stat_branchless` / `get_stat_castle`) uses lookup tables
//!      and integer arithmetic for paralysis, burn, screens, species items, and `ovfl`.
//!    - Accuracy computation (`ACC_LUT` / `CASTLE_ACC_LUT`) uses precomputed 101 × 13 × 13
//!      byte tables with Foresight evasion masking and BrightPowder `-20` saturation.
//!    - Damage computation (`get_damage_base_branchless` / `get_damage_base_castle`) handles
//!      PS and Castle rollover/Explosion ordering, critical hits, 1.1x items, weather, STAB,
//!      type effectiveness, and `/ 255` damage rolls.
//!    - Substitute targeting uses a 2-element HP pool `[hp, sub_hp]` indexed by
//!      `idx = (has_sub & !bypasses_sub) as usize`.
//! 2. **256-byte cache-line-aligned `Copy` state (`CompactBattle`)**:
//!    - Immutable match constants (base stats, stored stats, move metadata,
//!      1/8 and 1/16 residual quanta, ruleset) live in `MatchTable`.
//!    - Mutable state for the 6 picked Pokemon (32B × 6 = 192B), 2 sides
//!      (12B × 2 = 24B), and field/turn/damage/Future Sight/Castle/RNG state (40B)
//!      fits in exactly 256 bytes (4 cache lines, 64-byte aligned, zero padding bytes).

use std::sync::OnceLock;

use crate::dex::{Accuracy, Category, Dex, ItemId, MoveId, SpeciesId};
use crate::prng::Prng;
use crate::state::{stat_idx, Battle, Gender, RequestKind, Status, DK, QUICK_CLAW_ACTION_SPEED};

/// Sentinel `fixed_damage` indicating level-based fixed damage (Seismic Toss / Night Shade).
pub const FIXED_DAMAGE_LEVEL: i16 = -1;
/// Sentinel move/slot index meaning no move slot.
pub const NO_MOVE_SLOT: u8 = 0xFF;
/// Sentinel stat index meaning no stat slot (`sec_boost.0` / `boosts[i].0`).
pub const NO_STAT_IDX: u8 = 0xFF;
/// Sentinel type id meaning typeless (`Struggle` / unknown).
pub const NO_TYPE_ID: u8 = 0xFF;
/// Flag bit OR'd onto `CompactMoveSpec::sec_boost.0` for secondary self-boosts (AncientPower etc.).
pub const SELF_BOOST_FLAG: u8 = 0x80;
/// Bit 7 on `sides[0].side_flags`: disables the mid-turn Baton Pass fix when set (test/ablation toggle).
pub const SIDE0_DISABLE_BP_FIX: u8 = 0x80;
/// Stat-index sentinel (`7` with `SELF_BOOST_FLAG`) encoding AncientPower's all-5-stat raise.
pub const ANCIENTPOWER_ALL_STATS: u8 = 7;
/// Saturated 10-bit speed field when Quick Claw triggers in [`ActionSortKey`].
pub const COMPACT_QC_SATURATED_SPEED: u16 = 1023;
/// Exact common denominator for Castle non-truncated speed (`LCM(1..=8) * 4 = 3360`).
pub const CASTLE_SPEED_DENOM: u32 = 3360;
/// Saturated 22-bit speed field when Quick Claw triggers in Castle exact sort keys.
pub const CASTLE_QC_SATURATED_SPEED: u32 = 0x3F_FFFF;

/// Moves explicitly NOT faithfully modeled in `CompactBattle` Castle mode (Gate A contract §6-2).
pub const COMPACT_CASTLE_UNSUPPORTED_MOVES: &[&str] = &[
    "bide",
    "metronome",
    "sketch",
    "mimic",
    "transform",
    "mirrormove",
    "conversion",
    "conversion2",
    "skullbash",
    "razorwind",
    "skyattack",
    "solarbeam",
    "teleport",
];

/// Returns `true` when `move_id` is faithfully supported by `CompactBattle` in Castle PBS mode
/// (Gate A contract §6-2; unsupported niche moves are listed in [`COMPACT_CASTLE_UNSUPPORTED_MOVES`]).
#[inline]
pub fn is_supported_move_castle(dex: &Dex, move_id: MoveId) -> bool {
    let key = dex.moves.key(move_id);
    !COMPACT_CASTLE_UNSUPPORTED_MOVES.contains(&key)
}

/// Returns `true` when `battle` is eligible to run the search tree core natively on
/// [`CompactBattle`] (3v3 in-battle state without Team Preview, transformed mons, exotic
/// mid-turn queue entries, or any of the 13 unsupported Gen 2 moves).
pub fn is_compact_tree_supported(battle: &Battle, dex: &Dex) -> bool {
    use crate::state::{ActionKind, RequestState};
    if battle.ended
        || !battle.started
        || battle.turn == 0
        || battle.request_state == RequestState::TeamPreview
        || !battle.faint_queue.is_empty()
        || !battle.event_stack.is_empty()
        || !battle.effect_stack.is_empty()
        || battle.active_move.is_some()
        || battle.pending_boosts.is_some()
    {
        return false;
    }
    if battle.mid_turn && battle.queue.is_empty() && battle.request_state != RequestState::Switch {
        return false;
    }
    if let Some(scm) = dex.conds_id("sleepclausemod") {
        if battle.field.has_pseudo_weather(scm) {
            return false;
        }
    }
    for side in &battle.sides {
        if side.request_state() == Some(RequestKind::TeamPreview)
            || side.party.is_empty()
            || side.party.len() > 3
            || side.active != Some(side.party[0])
            || side.choice.forced_passes_left > 0
            || !side.choice.actions.is_empty()
            || !side.choice.switch_ins.is_empty()
        {
            return false;
        }
        for &slot in &side.party {
            let Some(p) = side.roster.get(slot as usize) else {
                return false;
            };
            if p.transformed
                || p.move_slots.len() > 4
                || p.item.is_some_and(|i| (i.0 as usize) >= 127)
            {
                return false;
            }
            for ms in p.move_slots.iter().chain(p.base_move_slots.iter()) {
                if !is_supported_move_castle(dex, ms.id) {
                    return false;
                }
            }
        }
    }
    if !battle.queue.is_empty() {
        if !battle.mid_turn || battle.request_state != RequestState::Switch {
            return false;
        }
        for act in &battle.queue {
            match act.choice {
                ActionKind::Residual => {}
                ActionKind::Move { move_id, .. } => {
                    if !is_supported_move_castle(dex, move_id) {
                        return false;
                    }
                    let dcb = dex.move_static(move_id).damage_cb;
                    if matches!(
                        dcb,
                        crate::dex::DamageCbKind::Counter | crate::dex::DamageCbKind::MirrorCoat
                    ) {
                        return false;
                    }
                    let Some(pid) = act.pokemon else {
                        return false;
                    };
                    let s = pid.side as usize;
                    if s >= 2
                        || battle.sides[s].request_state().is_some()
                        || battle.sides[s].active != Some(pid.slot)
                    {
                        return false;
                    }
                    let active = battle.poke(pid);
                    let valid_move = Some(move_id) == dex.known_moves.struggle
                        || Some(move_id) == dex.known_moves.recharge
                        || active.move_slots.iter().any(|ms| ms.id == move_id);
                    if !valid_move {
                        return false;
                    }
                }
                _ => return false,
            }
        }
    }
    true
}

/// Returns `true` when `battle` is a fresh Team Preview state eligible to run
/// ply 1+ of the search tree natively on [`CompactBattle`] via [`CompactPreviewRoster`].
/// Supports Turn-0 Berserk Gene natively and excludes `sleepclausemod`, unsupported moves,
/// transformed/damaged/statused mons, `turn_limit == Some(0)`, non-mid-turn synthetic roots,
/// and items with `onStart` handlers so those cases fall back transparently to `Battle`.
pub fn is_compact_preview_supported(battle: &Battle, dex: &Dex) -> bool {
    use crate::state::{ActionKind, RequestState};
    if battle.ended
        || !battle.started
        || !battle.mid_turn
        || battle.turn != 0
        || battle.request_state != RequestState::TeamPreview
        || !battle.faint_queue.is_empty()
        || !battle.event_stack.is_empty()
        || !battle.effect_stack.is_empty()
        || battle.active_move.is_some()
        || battle.pending_boosts.is_some()
        || battle.field.weather.is_some()
        || battle.triple_kick_mult != 0
        || battle.recoil_ko_side.is_some()
        || battle.bp_switched_this_turn != [false, false]
        || battle.queue.len() != 1
        || !matches!(battle.queue[0].choice, ActionKind::Start)
        || battle.ruleset.turn_limit().is_some_and(|lim| lim == 0)
    {
        return false;
    }
    if let Some(scm) = dex.conds_id("sleepclausemod") {
        if battle.field.has_pseudo_weather(scm) {
            return false;
        }
    }
    for side in &battle.sides {
        if side.request_state() != Some(RequestKind::TeamPreview)
            || side.party.is_empty()
            || side.party.len() > 6
            || side.roster.len() > 6
            || side.pokemon_left <= 0
            || side.active.is_some()
            || !side.side_conditions.is_empty()
            || !side.slot_conditions.is_empty()
            || side.choice.forced_passes_left > 0
            || !side.choice.actions.is_empty()
            || !side.choice.switch_ins.is_empty()
        {
            return false;
        }
        for &slot in &side.party {
            let Some(p) = side.roster.get(slot as usize) else {
                return false;
            };
            if p.transformed
                || p.fainted
                || p.hp <= 0
                || p.hp != p.maxhp
                || p.status != Status::None
                || !p.volatiles.is_empty()
                || p.boosts != [0; 7]
                || p.move_slots.len() > 4
                || p.item.is_some_and(|i| {
                    (i.0 as usize) >= 127 || dex.items.get(i).mask.has(dex.known.on_start)
                })
            {
                return false;
            }
            for ms in &p.move_slots {
                if ms.pp != ms.maxpp || !is_supported_move_castle(dex, ms.id) {
                    return false;
                }
            }
            for ms in &p.base_move_slots {
                if !is_supported_move_castle(dex, ms.id) {
                    return false;
                }
            }
        }
    }
    true
}

/// Precompiled 6-mon-per-side Team Preview snapshot enabling O(1) Turn-1
/// `(MatchTable, CompactBattle)` synthesis for any legal `(pick0, pick1)` pair
/// without touching `Battle` or allocating on the heap.
#[derive(Clone)]
pub struct CompactPreviewRoster {
    mons: [[StaticMonSpec; 6]; 2],
    move_specs: [[[CompactMoveSpec; 4]; 6]; 2],
    init_mons: [[CompactMon; 6]; 2],
    party_map: [[u8; 6]; 2],
    party_len: [u8; 2],
    picked_len: [u8; 2],
    roster_len: [u8; 2],
    quick_claw_mask: [u8; 2],
    berserk_gene_mask: [u8; 2],
    clause_flags: u8,
    pub ruleset: crate::rules::Ruleset,
    dt: &'static CompactDexTables,
}

impl CompactPreviewRoster {
    /// Extract a reusable Team Preview roster snapshot from a supported Team Preview `Battle`.
    pub fn from_battle(
        battle: &Battle,
        dex: &Dex,
        roster_cache: Option<&RosterCompactCache>,
    ) -> Self {
        let dt = dex_tables(dex);
        let mut mons = [[StaticMonSpec::default(); 6]; 2];
        let mut move_specs = [[[CompactMoveSpec::default(); 4]; 6]; 2];
        let mut init_mons = [[CompactMon::default(); 6]; 2];
        let mut party_map = [[0u8; 6]; 2];
        let mut party_len = [0u8; 2];
        let mut picked_len = [0u8; 2];
        let mut roster_len = [0u8; 2];
        let mut quick_claw_mask = [0u8; 2];
        let mut berserk_gene_mask = [0u8; 2];

        for side_n in 0..2usize {
            let side = &battle.sides[side_n];
            let plen = side.party.len().min(6);
            let rlen = side.roster.len().min(6);
            party_len[side_n] = plen as u8;
            picked_len[side_n] = plen.min(3) as u8;
            roster_len[side_n] = rlen as u8;
            for (pos, &r_slot) in side.party.iter().take(6).enumerate() {
                party_map[side_n][pos] = r_slot;
            }
            for r_slot in 0..rlen {
                let p = &side.roster[r_slot];
                let num_moves = p.move_slots.len().min(4) as u8;
                let cached_hit = roster_cache
                    .and_then(|rc| rc.entries[side_n].get(r_slot))
                    .and_then(|opt| opt.as_ref())
                    .filter(|cm| {
                        !p.transformed
                            && p.species == cm.spec.species
                            && p.level == cm.spec.level
                            && p.maxhp == cm.guard_maxhp
                            && p.stored_stats == cm.guard_stored_stats
                            && p.types == cm.guard_types
                            && p.item == cm.guard_item
                            && p.hp_type == cm.guard_hp_type
                            && p.hp_power == cm.guard_hp_power
                            && p.happiness == cm.guard_happiness
                            && num_moves == cm.spec.num_moves
                            && (0..num_moves as usize).all(|m_i| {
                                let sl = p.move_slots[m_i];
                                sl.id == cm.spec.moves[m_i] && sl.maxpp == cm.guard_maxpp[m_i]
                            })
                    });
                let (spec, ms) = if let Some(cm) = cached_hit {
                    (cm.spec, cm.move_specs)
                } else {
                    build_static_mon_spec(p, dex, dt)
                };
                mons[side_n][r_slot] = spec;
                move_specs[side_n][r_slot] = ms;

                let item = p.item.map(|i| (i.0 & 0x7F) as u8 + 1).unwrap_or(0);
                if item != 0 {
                    if let Some(ispec) = dt.item_specs.get((item - 1) as usize) {
                        if ispec.is_quick_claw {
                            quick_claw_mask[side_n] |= 1u8 << r_slot;
                        }
                        if ispec.is_berserk_gene {
                            berserk_gene_mask[side_n] |= 1u8 << r_slot;
                        }
                    }
                }
                let gender_byte = match p.gender {
                    Gender::M => 1,
                    Gender::F => 2,
                    Gender::N => 0,
                };
                let mut c_mon = CompactMon {
                    hp_pool: [p.hp.clamp(0, i16::MAX as i32) as i16, 0],
                    pp: [0u8; 4],
                    status_packed: 0,
                    item,
                    vol_flags: 0,
                    timers: [0u8; 4],
                    boosts: [0; 7],
                    slot_refs: 0,
                    ext_flags: 0,
                    ext_timers: [0u8; 4],
                    gender: gender_byte,
                    rollout_rage: 0,
                };
                for m_i in 0..num_moves as usize {
                    c_mon.pp[m_i] = p.move_slots[m_i].pp.clamp(0, 64) as u8;
                }
                init_mons[side_n][r_slot] = c_mon;
            }
        }

        let mut clause_flags = 0u8;
        if !battle.field.pseudo_weather.is_empty() {
            for &(cid, _) in &battle.field.pseudo_weather {
                if Some(cid) == dt.sleep_clause_cond {
                    clause_flags |= 1;
                } else if Some(cid) == dt.freeze_clause_cond {
                    clause_flags |= 4;
                }
            }
        }

        Self {
            mons,
            move_specs,
            init_mons,
            party_map,
            party_len,
            picked_len,
            roster_len,
            quick_claw_mask,
            berserk_gene_mask,
            clause_flags,
            ruleset: battle.ruleset,
            dt,
        }
    }

    /// Synthesize the Turn-1 `(MatchTable, CompactBattle)` resulting from both sides'
    /// 1-based `SearchChoice::Team` picks (`0` for unused slots when `picked_len < 3`).
    #[inline]
    pub fn build_turn1(&self, picks: [[u8; 3]; 2], seed: u64) -> (MatchTable, CompactBattle) {
        let mut static_mons = [StaticMonSpec::default(); 6];
        let mut move_specs = [[CompactMoveSpec::default(); 4]; 6];
        let mut compact_mons = [CompactMon::default(); 6];
        let mut compact_sides = [CompactSide::default(); 2];
        let mut any_quick_claw = false;

        for side_n in 0..2usize {
            let picked = self.picked_len[side_n] as usize;
            let plen = self.party_len[side_n] as usize;
            let rlen = self.roster_len[side_n];
            let mut party_roster = [0u8; 3];
            let mut alive_mask = 0u8;
            for p_idx in 0..3usize {
                let c_idx = side_n * 3 + p_idx;
                if p_idx < picked {
                    let disp = (picks[side_n][p_idx].saturating_sub(1) as usize)
                        .min(plen.saturating_sub(1));
                    let r_slot = self.party_map[side_n][disp];
                    party_roster[p_idx] = r_slot;
                    alive_mask |= 1u8 << p_idx;
                    static_mons[c_idx] = self.mons[side_n][r_slot as usize];
                    move_specs[c_idx] = self.move_specs[side_n][r_slot as usize];
                    compact_mons[c_idx] = self.init_mons[side_n][r_slot as usize];
                    if (self.quick_claw_mask[side_n] & (1u8 << r_slot)) != 0 {
                        any_quick_claw = true;
                    }
                } else {
                    party_roster[p_idx] = (p_idx as u8).min(rlen.saturating_sub(1));
                    compact_mons[c_idx].mark_fainted();
                }
            }
            compact_sides[side_n] = CompactSide {
                active_idx: 0,
                party_roster,
                alive_mask,
                reflect_turns: 0,
                lightscreen_turns: 0,
                safeguard_turns: 0,
                side_flags: 0,
                last_move_selfko: u8::MAX,
                request: 1,
                pad: 0,
            };
        }

        let mut fp = 0xcbf29ce484222325u64;
        for m in &static_mons {
            fp = fp.wrapping_add(m.static_fp).wrapping_mul(0xf1357aea2e62a9c5);
        }
        let rs_tag = self.ruleset.fingerprint_tag();
        if rs_tag != 0 {
            fp = fp.wrapping_add(rs_tag).wrapping_mul(0xf1357aea2e62a9c5);
        }
        compact_sides[0].pad = (fp >> 16) as u8;
        compact_sides[1].pad = (fp >> 24) as u8;

        let mut prng = Prng::new(seed);

        let dt = self.dt;
        let table = MatchTable {
            mons: static_mons,
            move_specs,
            type_chart: &dt.type_chart,
            item_specs: &dt.item_specs,
            normal_type: dt.normal_type,
            fighting_type: dt.fighting_type,
            flying_type: dt.flying_type,
            poison_type: dt.poison_type,
            ground_type: dt.ground_type,
            ghost_type: dt.ghost_type,
            grass_type: dt.grass_type,
            fire_type: dt.fire_type,
            electric_type: dt.electric_type,
            ice_type: dt.ice_type,
            sandstorm_immune_mask: dt.sandstorm_immune_mask,
            poison_immune_mask: dt.poison_immune_mask,
            match_fingerprint: fp,
            ruleset: self.ruleset,
            any_quick_claw,
        };
        let mut cb = CompactBattle {
            mons: compact_mons,
            sides: compact_sides,
            turn: 1,
            weather_kind: CompactWeather::None,
            weather_turns: 0,
            quick_claw_roll: 0,
            outcome: 0,
            speed_order: 0,
            clause_flags: self.clause_flags,
            hurt_damage_this_turn: [0, 0],
            hurt_cat_this_turn: [0, 0],
            match_fp_lo: fp as u16,
            future_sight_dmg: [0, 0],
            future_sight_turns: [0, 0],
            future_sight_src: [0, 0],
            bp_switch_flags: 0,
            triple_kick_mult: 0,
            turn_extra_flags: 0,
            reserved_castle: [0; 5],
            prng_seed: 0,
        };
        if (self.berserk_gene_mask[0] | self.berserk_gene_mask[1]) != 0 {
            for s in 0..2usize {
                let lead_r = cb.sides[s].party_roster[0];
                if (self.berserk_gene_mask[s] & (1u8 << lead_r)) != 0 {
                    cb.trigger_berserk_gene(&table, s, &mut prng);
                }
            }
        }
        let qc = Battle::roll_quick_claw_with(self.ruleset, &mut prng);
        cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
        cb.prng_seed = prng.seed();
        (table, cb)
    }
}

/// Indices into [`CompactMon::timers`] (`[confusion, encore, disable, perish/stall]`).
pub mod timer_idx {
    pub const CONFUSION: usize = 0;
    pub const ENCORE: usize = 1;
    pub const DISABLE: usize = 2;
    pub const PERISH_STALL: usize = 3;
}

/// Indices into [`CompactMon::ext_timers`] (`[locked_turns, partial_trap_turns, fury_cutter, rage_counter]`).
pub mod ext_timer_idx {
    pub const LOCKED_TURNS: usize = 0;
    pub const PARTIAL_TRAP_TURNS: usize = 1;
    pub const FURY_CUTTER: usize = 2;
    pub const RAGE_COUNTER: usize = 3;
}

/// Pre-classified special move mechanics tag on [`CompactMoveSpec`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SpecialMoveKind {
    #[default]
    None = 0,
    Reflect = 1,
    LightScreen = 2,
    Safeguard = 3,
    Spikes = 4,
    Curse = 5,
    BellyDrum = 6,
    Phaze = 7,
    Encore = 8,
    Disable = 9,
    SleepTalk = 10,
    Counter = 11,
    MirrorCoat = 12,
    FlailReversal = 13,
    LevelDamage = 14,
    SuperFang = 15,
    Thief = 16,
    PainSplit = 17,
    BatonPass = 18,
    PerishSong = 19,
    DestinyBond = 20,
    HealBell = 21,
    PsychUp = 22,
    RainDance = 23,
    SunnyDay = 24,
    Sandstorm = 25,
    MeanLook = 26,
    Haze = 27,
    RapidSpin = 28,
    WeatherHeal = 29,
    Swagger = 30,
    Pursuit = 31,
    Present = 32,
    Rollout = 33,
    DefenseCurl = 34,
    Minimize = 35,
    OutrageLock = 36,
    Attract = 37,
    Mist = 38,
    FocusEnergy = 39,
    CrashKick = 40,
    FutureSight = 41,
    LockOn = 42,
    BeatUp = 43,
    FuryCutter = 44,
    Rage = 45,
    TripleKick = 46,
    Twineedle = 47,
    TriAttack = 48,
    PartialTrap = 49,
    Dig = 50,
    Fly = 51,
    FalseSwipe = 52,
    Psywave = 53,
    Spite = 54,
    Ohko = 55,
    DreamEater = 56,
    Magnitude = 57,
    Thunder = 58,
    Stomp = 59,
    Earthquake = 60,
    GustTwister = 61,
}

/// Variable base power formula tag on [`CompactMoveSpec`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum VarBpKind {
    #[default]
    Static = 0,
    Return = 1,
    Frustration = 2,
    HiddenPower = 3,
    FlailReversal = 4,
    Magnitude = 5,
    Present = 6,
}

// =========================================================================
// 1. ActionSortKey: Single-u32 Branchless Turn Ordering
// =========================================================================

/// Action phase priority (larger value executes earlier in the turn).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ActionPhase {
    Move = 1,
    Switch = 2,
}

/// Bit-packed 32-bit sort key for turn action ordering:
/// `[ phase (3b: bits 16..18) | (priority + 8) (4b: bits 12..15) | speed_or_qc (10b: bits 1..10) | tie_bit (1b: bit 0) ]`
///
/// Note on Quick Claw (`qc_active`):
/// In `gen2stadium2`, triggering Quick Claw sets a Pokemon's action speed to
/// `65535` (`pokemon.rs:139`). By saturating the 10-bit speed field (`0..=999`)
/// to `1023` via `eff_speed | (-(qc_active as i32) as u32 & 1023)`, a Quick Claw
/// activation beats every natural speed (`<= 999`) AND causes two simultaneous
/// Quick Claw activations at the same priority to tie (`1023 == 1023`),
/// preserving exact equivalence with `65535`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ActionSortKey(pub u32);

impl ActionSortKey {
    #[inline(always)]
    pub const fn new(
        phase: ActionPhase,
        priority: i8,
        qc_active: bool,
        eff_speed: i32,
        tie_bit: u8,
    ) -> Self {
        let phase_bits = (phase as u32 & 0x7) << 15;
        let prio_bits = (((priority as i32 + 8) as u32) & 0xF) << 11;
        let clamped_speed = if eff_speed < 0 {
            0u32
        } else if eff_speed > 999 {
            999u32
        } else {
            eff_speed as u32
        };
        let qc_mask = (-(qc_active as i32) as u32) & (COMPACT_QC_SATURATED_SPEED as u32);
        let speed_bits = (clamped_speed | qc_mask) << 1;
        let tie = (tie_bit & 1) as u32;
        Self(phase_bits | prio_bits | speed_bits | tie)
    }

    /// Compare two keys ignoring the tie-break bit (bit 0).
    #[inline(always)]
    pub const fn ties_without_rng(self, other: Self) -> bool {
        (self.0 >> 1) == (other.0 >> 1)
    }
}

// =========================================================================
// 2. Branchless Stat Calculation (13-entry LUT + Bitshifts)
// =========================================================================

/// Numerators for Gen 2 stat stages `-6..=+6` (divided by 100).
/// Matches `pokemon.rs` `BOOST_TABLE` (`[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]`)
/// and `NUMERATORS` (`[100, 66, 50, 40, 33, 28, 25]`) bit-for-bit on all integers.
pub const STAT_NUM: [i32; 13] = [25, 28, 33, 40, 50, 66, 100, 150, 200, 250, 300, 350, 400];

/// Species-specific stat-boosting held item classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ItemStatKind {
    #[default]
    None = 0,
    /// Thick Club held by Cubone or Marowak (doubles Attack, `stat == 0`).
    ThickClub = 1,
    /// Light Ball held by Pikachu (doubles Special Attack, `stat == 2`).
    LightBall = 2,
    /// Metal Powder held by Ditto (1.5x Defense and Special Defense, `stat == 1 | 3`).
    MetalPowder = 3,
}

impl ItemStatKind {
    #[inline(always)]
    pub fn classify(dex: &Dex, species: SpeciesId, item: Option<ItemId>) -> Self {
        let Some(item_id) = item else {
            return Self::None;
        };
        let ki = &dex.known_items;
        let ks = &dex.known_species;
        let sp = Some(species);
        let it = Some(item_id);
        if it == ki.thickclub && (sp == ks.cubone || sp == ks.marowak) {
            Self::ThickClub
        } else if it == ki.lightball && sp == ks.pikachu {
            Self::LightBall
        } else if it == ki.metalpowder && sp == ks.ditto {
            Self::MetalPowder
        } else {
            Self::None
        }
    }
}

/// Branchless integer implementation of `gen2stadium2` `Battle::get_stat`.
/// Guaranteed bit-for-bit identical to `Battle::get_stat` across all inputs.
#[inline(always)]
pub const fn get_stat_branchless(
    stored_stat: i32,
    boost: i8,
    stat_idx: usize,
    unboosted: bool,
    unmodified: bool,
    fast_return: bool,
    has_par_speed_drop: bool,
    has_brn_atk_drop: bool,
    has_reflect: bool,
    has_light_screen: bool,
    item_stat_kind: ItemStatKind,
) -> i32 {
    let clamped_boost = if boost < -6 {
        -6
    } else if boost > 6 {
        6
    } else {
        boost
    };
    // Zero the boost when `unboosted` is true via arithmetic mask.
    let keep_boost_mask = -(!unboosted as i8);
    let eff_boost = clamped_boost & keep_boost_mask;
    let mut val = (stored_stat * STAT_NUM[(eff_boost + 6) as usize]) / 100;

    // Paralysis speed drop (÷4 = >> 2) and Burn attack drop (÷2 = >> 1).
    let par_shift = ((stat_idx == stat_idx::SPE) & has_par_speed_drop) as u32 * 2;
    let brn_shift = ((stat_idx == stat_idx::ATK) & (!unmodified) & has_brn_atk_drop) as u32;
    val >>= par_shift + brn_shift;

    // Clamp to [1, 999].
    if val < 1 {
        val = 1;
    } else if val > 999 {
        val = 999;
    }
    if fast_return {
        return val;
    }

    // Reflect (stat 1) / Light Screen (stat 3) and Thick Club / Light Ball (×2 = << 1).
    let screen_shift = (!unboosted
        & (((stat_idx == stat_idx::DEF) & has_reflect)
            | ((stat_idx == stat_idx::SPD) & has_light_screen))) as u32;
    let kind = item_stat_kind as u8;
    let item_double_shift =
        (((kind == 1) & (stat_idx == stat_idx::ATK)) | ((kind == 2) & (stat_idx == stat_idx::SPA)))
            as u32;
    val <<= screen_shift + item_double_shift;

    // Metal Powder on Ditto (×1.5 = val + (val >> 1)).
    let is_metal = ((kind == 3) & ((stat_idx == stat_idx::DEF) | (stat_idx == stat_idx::SPD)))
        as i32;
    val += (val >> 1) & -is_metal;

    val
}

/// Gym Leader's Castle PBS (`PS.cgi`) stat computation (spec #17, #18, #20, #35).
///
/// Key Castle diffs vs PS:
/// - Negative stage uses exact rational `int(stat * 2 / (2 - stage))` (not GB `%` table).
/// - Burn attack halving and Paralysis speed quartering are NEVER cleared by stat boosts/Haze.
/// - Thick Club / Light Ball / Metal Powder / Screens apply before `999` clamp (or `% 1024` wrap when `overflow_mode`),
///   then Burn `/ 2` applies (matching `DamageCalc.java` and `Probe 12B` / `Probe 3A-3B`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub const fn get_stat_castle(
    stored_stat: i32,
    boost: i8,
    stat_idx: usize,
    unboosted: bool,
    unmodified: bool,
    is_paralyzed: bool,
    is_burned: bool,
    has_reflect: bool,
    has_light_screen: bool,
    item_stat_kind: ItemStatKind,
    overflow_mode: bool,
) -> i32 {
    let clamped_boost = if boost < -6 {
        -6
    } else if boost > 6 {
        6
    } else {
        boost
    };
    let eff_boost = if unboosted { 0 } else { clamped_boost };
    let mut val = if eff_boost >= 0 {
        (stored_stat * (2 + eff_boost as i32)) / 2
    } else {
        (stored_stat * 2) / (2 - eff_boost as i32)
    };

    // Species-specific items (Thick Club / Light Ball / Metal Powder).
    let kind = item_stat_kind as u8;
    if (kind == 1 && stat_idx == stat_idx::ATK) || (kind == 2 && stat_idx == stat_idx::SPA) {
        val <<= 1;
    } else if kind == 3 && (stat_idx == stat_idx::DEF || stat_idx == stat_idx::SPD) {
        val += val >> 1;
    }

    // Reflect / Light Screen (skipped when unboosted, e.g. crit ignoring defender boosts).
    if !unboosted
        && ((stat_idx == stat_idx::DEF && has_reflect)
            || (stat_idx == stat_idx::SPD && has_light_screen))
    {
        val <<= 1;
    }

    // Cap at 999 (standard) or wrap `% 1024` (`ovfl=1` mode), min 1 before par/burn.
    if overflow_mode {
        if val >= 1024 {
            val %= 1024;
        }
    } else if val > 999 {
        val = 999;
    }
    if val < 1 {
        val = 1;
    }

    // Burn / Paralysis penalties (never cleared by boosts/Haze in Castle).
    if stat_idx == stat_idx::SPE && is_paralyzed {
        val >>= 2;
    }
    if stat_idx == stat_idx::ATK && !unmodified && is_burned {
        val >>= 1;
    }

    if val < 1 {
        1
    } else {
        val
    }
}

/// Exact non-truncated Castle speed scaled by [`CASTLE_SPEED_DENOM`] (`3360`),
/// matching Perl `f64` speed comparison without truncation (`Probe 5B/8E/9B/12H`).
#[inline(always)]
pub const fn get_castle_speed_x3360(stored_spe: i32, boost: i8, is_paralyzed: bool) -> u32 {
    let clamped_boost = if boost < -6 {
        -6
    } else if boost > 6 {
        6
    } else {
        boost
    };
    let spe = if stored_spe < 1 { 1u32 } else { stored_spe as u32 };
    let (num, den): (u32, u32) = if clamped_boost >= 0 {
        (spe * (2 + clamped_boost as u32), 2)
    } else {
        (spe * 2, (2 - clamped_boost) as u32)
    };
    // Clamp pre-paralysis speed to 999.0 (`DamageCalc.java:57`).
    let pre_par_x3360 = if num >= 999 * den {
        999 * CASTLE_SPEED_DENOM
    } else {
        num * (CASTLE_SPEED_DENOM / den)
    };
    if is_paralyzed {
        pre_par_x3360 >> 2
    } else {
        pre_par_x3360
    }
}

// =========================================================================
// 3. O(1) Accuracy Lookup Table (101 × 13 × 13 = 17,069 bytes)
// =========================================================================

/// Full precomputed Gen 2 accuracy table `[base_pct 0..=100][acc_boost + 6][eva_boost + 6]`.
/// Each entry stores the exact floored & clamped `[1, 255]` threshold from
/// `try_move_hit` / `hit_probability`.
pub type AccTable = [[[u8; 13]; 13]; 101];

static ACC_LUT_CELL: OnceLock<AccTable> = OnceLock::new();
static CASTLE_ACC_LUT_CELL: OnceLock<AccTable> = OnceLock::new();

fn build_acc_lut() -> AccTable {
    const POS_BOOST: [f64; 7] = [1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0];
    const NEG_BOOST: [f64; 7] = [1.0, 0.75, 0.6, 0.5, 0.43, 0.36, 0.33];
    let mut table = [[[1u8; 13]; 13]; 101];
    for pct in 0..=100usize {
        let base = (pct as f64 * 255.0 / 100.0).floor();
        for acc_idx in 0..13usize {
            let acc_b = acc_idx as i32 - 6;
            let after_acc = if acc_b > 0 {
                base * POS_BOOST[acc_b as usize]
            } else {
                base * NEG_BOOST[(-acc_b) as usize]
            };
            for eva_idx in 0..13usize {
                let eva_b = eva_idx as i32 - 6;
                let mut acc = after_acc;
                if eva_b > 0 {
                    acc *= NEG_BOOST[eva_b as usize];
                } else if eva_b < 0 {
                    acc *= POS_BOOST[(-eva_b) as usize];
                }
                acc = acc.floor().min(255.0).max(1.0);
                table[pct][acc_idx][eva_idx] = acc as u8;
            }
        }
    }
    table
}

/// Precomputed Castle PBS accuracy table using exact rationals `3/9..=9/3` (spec #21).
fn build_castle_acc_lut() -> AccTable {
    let mut table = [[[1u8; 13]; 13]; 101];
    for pct in 0..=100usize {
        let base = (pct as f64 * 255.0 / 100.0).floor();
        for acc_idx in 0..13usize {
            let acc_b = acc_idx as i32 - 6;
            let acc_mult = if acc_b >= 0 {
                (3 + acc_b) as f64 / 3.0
            } else {
                3.0 / (3 - acc_b) as f64
            };
            let after_acc = base * acc_mult;
            for eva_idx in 0..13usize {
                let eva_b = eva_idx as i32 - 6;
                let eva_mult = if eva_b >= 0 {
                    3.0 / (3 + eva_b) as f64
                } else {
                    (3 - eva_b) as f64 / 3.0
                };
                let acc = (after_acc * eva_mult).floor().min(255.0).max(1.0);
                table[pct][acc_idx][eva_idx] = acc as u8;
            }
        }
    }
    table
}

/// Access the static 101 × 13 × 13 Gen 2 accuracy threshold table.
#[inline(always)]
pub fn acc_lut() -> &'static AccTable {
    ACC_LUT_CELL.get_or_init(build_acc_lut)
}

/// Access the static 101 × 13 × 13 Castle PBS exact-rational accuracy table (`3/9..=9/3`).
#[inline(always)]
pub fn castle_acc_lut() -> &'static AccTable {
    CASTLE_ACC_LUT_CELL.get_or_init(build_castle_acc_lut)
}

/// Branchless O(1) lookup of the Gen 2 `[0, 255]` accuracy threshold out of 256.
/// Returns `255` when a move has 255/256 (or guaranteed hit in `hit_probability`).
#[inline(always)]
pub fn lookup_accuracy_threshold(
    base_pct: u8,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    ignore_positive_evasion: bool,
    brightpowder: bool,
) -> u8 {
    let pct = base_pct.min(100) as usize;
    let a_clamped = acc_boost.clamp(-6, 6);
    let e_clamped = eva_boost.clamp(-6, 6);

    let eff_acc = a_clamped & -(!ignore_accuracy as i8);
    let drop_pos_eva = ignore_evasion | (ignore_positive_evasion & (e_clamped > 0));
    let eff_eva = e_clamped & -(!drop_pos_eva as i8);

    let raw = acc_lut()[pct][(eff_acc + 6) as usize][(eff_eva + 6) as usize];
    let bp_sub = (-(brightpowder as i8) as u8) & 20;
    raw.saturating_sub(bp_sub)
}

/// Castle PBS accuracy threshold lookup using exact `3/9..=9/3` rationals,
/// Foresight `acc < eva` both-zero rule (spec #54③), and BrightPowder `-20/256`.
#[inline(always)]
pub fn lookup_accuracy_threshold_castle(
    base_pct: u8,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    foresight_active: bool,
    brightpowder: bool,
) -> u8 {
    let pct = base_pct.min(100) as usize;
    let mut eff_acc = if ignore_accuracy { 0 } else { acc_boost.clamp(-6, 6) };
    let mut eff_eva = if ignore_evasion { 0 } else { eva_boost.clamp(-6, 6) };
    if foresight_active {
        if eff_acc < eff_eva {
            eff_acc = 0;
            eff_eva = 0;
        } else if eff_eva > 0 {
            eff_eva = 0;
        }
    }
    let raw = castle_acc_lut()[pct][(eff_acc + 6) as usize][(eff_eva + 6) as usize];
    let bp_sub = (-(brightpowder as i8) as u8) & 20;
    raw.saturating_sub(bp_sub)
}

/// Branchless implementation of `Battle::hit_probability`.
#[inline(always)]
pub fn hit_probability_branchless(
    base_pct: u8,
    always_hit: bool,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    ignore_positive_evasion: bool,
) -> f64 {
    if always_hit {
        return 1.0;
    }
    let acc = lookup_accuracy_threshold(
        base_pct,
        acc_boost,
        eva_boost,
        ignore_accuracy,
        ignore_evasion,
        ignore_positive_evasion,
        false,
    );
    if acc == 255 {
        1.0
    } else {
        acc as f64 / 256.0
    }
}

// =========================================================================
// 4. Branchless Damage & SIMD 16-Roll / 39-Roll Evaluators
// =========================================================================

/// Weather kind in `CompactBattle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CompactWeather {
    #[default]
    None = 0,
    Rain = 1,
    Sun = 2,
    Sandstorm = 3,
}

impl CompactWeather {
    /// HP healed by Morning Sun / Moonlight / Synthesis under this active weather.
    #[inline(always)]
    pub const fn weather_heal_amount(self, maxhp: i16) -> i16 {
        match self {
            CompactWeather::Sun => maxhp,
            CompactWeather::Rain | CompactWeather::Sandstorm => {
                let q = maxhp / 4;
                if q < 1 { 1 } else { q }
            }
            CompactWeather::None => {
                let q = maxhp / 2;
                if q < 1 { 1 } else { q }
            }
        }
    }
}

/// Branchless computation of the pre-variance Gen 2 Stadium 2 damage (`[0, 3996]`).
///
/// Mirrors `Battle::get_damage` (`moveexec.rs:2865-2945`) using pure integer and
/// bitmask operations:
/// - Stadium 2 `>= 256` rollover (`>> 2` on both attack and defense simultaneously)
/// - Self-Destruct / Explosion defense halving (`>> 1`)
/// - Critical hit doubling (`<< 1`)
/// - 1.1x type-boost item (`dmg + dmg / 10`, exact for all IEEE-754 integers)
/// - Clamp `[1, 997] + 2`
/// - Weather boost (`+ (dmg >> 1)`) and nerf (`>> 1`)
/// - STAB (`+ (dmg >> 1)` via `attacker_type_mask` bit test)
/// - Type effectiveness (`<< super_shift >> resist_shift`, masked by `!is_immune`)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub const fn get_damage_base_branchless(
    level: i32,
    base_power: i32,
    atk: i32,
    def: i32,
    is_physical: bool,
    is_selfdestruct: bool,
    is_crit: bool,
    has_type_boost_item: bool,
    weather: CompactWeather,
    is_water_move: bool,
    is_fire_move: bool,
    is_solarbeam: bool,
    move_type: u8,
    attacker_type_mask: u32,
    total_type_mod: i8,
    is_immune: bool,
) -> i32 {
    get_damage_base_branchless_ext(
        level,
        base_power,
        atk,
        def,
        is_physical,
        is_selfdestruct,
        is_crit,
        has_type_boost_item,
        weather,
        is_water_move,
        is_fire_move,
        is_solarbeam,
        move_type,
        attacker_type_mask,
        total_type_mod,
        is_immune,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
const fn get_damage_base_branchless_ext(
    level: i32,
    base_power: i32,
    mut atk: i32,
    mut def: i32,
    is_physical: bool,
    is_selfdestruct: bool,
    is_crit: bool,
    has_type_boost_item: bool,
    weather: CompactWeather,
    is_water_move: bool,
    is_fire_move: bool,
    is_solarbeam: bool,
    move_type: u8,
    attacker_type_mask: u32,
    total_type_mod: i8,
    is_immune: bool,
    pre_plus2_mult: i32,
) -> i32 {
    if is_immune || base_power <= 0 {
        return 0;
    }

    // Stadium 2 rollover fix: clamp atk/def to [1, 999], then if atk >= 256 || def >= 256
    // divide both by 4 (min 1), matching `Battle::get_damage` (`moveexec.rs:3440-3451`).
    if atk < 1 {
        atk = 1;
    } else if atk > 999 {
        atk = 999;
    }
    if def < 1 {
        def = 1;
    } else if def > 999 {
        def = 999;
    }
    let rollover_shift = (((atk | def) >= 256) as u32) << 1;
    atk >>= rollover_shift;
    if atk < 1 {
        atk = 1;
    }
    def >>= rollover_shift;
    if def < 1 {
        def = 1;
    }

    // Self-Destruct / Explosion halves physical defense (min 1).
    let selfdestruct_shift = (is_selfdestruct & is_physical) as u32;
    def >>= selfdestruct_shift;
    if def < 1 {
        def = 1;
    }

    // Core Gen 2 damage formula.
    let mut dmg = ((level * 2 / 5 + 2) * base_power * atk / def) / 50;

    // Critical hit doubles before ModifyDamage.
    dmg <<= is_crit as u32;

    // 1.1x type-boost held item: floor(dmg * 1.1) == dmg + dmg / 10.
    let boost_mask = -(has_type_boost_item as i32);
    dmg += (dmg / 10) & boost_mask;

    // ModifyDamage multiplier (e.g. Stomp vs Minimize = 2x) applies BEFORE clamp(997) + 2.
    if pre_plus2_mult > 1 {
        dmg *= pre_plus2_mult;
    }

    // Clamp to [1, 997], then + 2.
    if dmg < 1 {
        dmg = 1;
    } else if dmg > 997 {
        dmg = 997;
    }
    dmg += 2;

    // Weather modifier.
    let rain = (weather as u8) == (CompactWeather::Rain as u8);
    let sun = (weather as u8) == (CompactWeather::Sun as u8);
    let w_boost = ((is_water_move & rain) | (is_fire_move & sun)) as i32;
    let w_nerf = (((is_fire_move | is_solarbeam) & rain) | (is_water_move & sun)) as u32;
    dmg += (dmg >> 1) & -w_boost;
    dmg >>= w_nerf;

    // STAB (Same-Type Attack Bonus): +50% when move_type bit is set in attacker_type_mask.
    let is_stab = (((move_type < 32) as i32) & ((attacker_type_mask >> (move_type & 31)) as i32)) & 1;
    dmg += (dmg >> 1) & -is_stab;

    // Type effectiveness: super-effective shifts left (1 or 2), resist shifts right (1 or 2).
    let super_shift = if total_type_mod > 0 {
        if total_type_mod >= 2 { 2u32 } else { 1u32 }
    } else {
        0u32
    };
    let resist_shift = if total_type_mod < 0 {
        if total_type_mod <= -2 { 2u32 } else { 1u32 }
    } else {
        0u32
    };
    dmg = (dmg << super_shift) >> resist_shift;

    // PS / Gen 2 Stadium 2: any non-immune hit with base_power > 0 deals at least 1 damage
    // (`moveexec.rs:3687`, even on 0.25x resisted weak hits where integer division floors to 0).
    if dmg < 1 {
        1
    } else {
        dmg
    }
}

/// Branchless application of a single Gen 2 damage roll (`roll in 217..=255`).
/// Uses reciprocal multiplication `(x * 0x808081) >> 31` instead of integer
/// division `/ 255` (exact for all `x < 2^24`).
#[inline(always)]
pub const fn apply_roll_branchless(
    base_dmg: i32,
    roll_217_to_255: u32,
    no_damage_variance: bool,
) -> i32 {
    if base_dmg <= 0 {
        return 0;
    }
    let apply_var = (!no_damage_variance & (base_dmg > 1)) as i32;
    let num = (base_dmg as u64) * (roll_217_to_255 as u64);
    let varied = ((num * 0x808081) >> 31) as i32;
    let mask = -apply_var;
    let raw = (varied & mask) | (base_dmg & !mask);
    if raw < 1 { 1 } else { raw }
}

/// Gym Leader's Castle PBS (`PS.cgi`) pre-variance damage computation (spec #19, #26, #30).
///
/// Key Castle diffs vs PS:
/// - **Explosion / Self-Destruct halves physical Defense BEFORE the `>= 256` `/4` rollover check** (`Probe 12E2`).
/// - **Rollout multiplier (`2^(k-1) * (2 if Defense Curl)`) is applied AFTER `+2` and BEFORE STAB / type / roll** (`Probe 12E2`).
/// - **`extra_mult`** supports Stomp vs Minimize (`2`), Earthquake/Magnitude vs Dig (`2`), Gust/Twister vs Fly (`2`),
///   and Castle Triple Kick multiplier leak (`2` or `3`).
/// - **Minimum damage `0` is allowed** (`Probe 5A`, no `+2` floor clamp to 1 if `base_dmg` floors to 0).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub const fn get_damage_base_castle(
    level: i32,
    base_power: i32,
    mut atk: i32,
    mut def: i32,
    is_physical: bool,
    is_selfdestruct: bool,
    is_crit: bool,
    has_type_boost_item: bool,
    weather: CompactWeather,
    is_water_move: bool,
    is_fire_move: bool,
    is_solarbeam: bool,
    move_type: u8,
    attacker_type_mask: u32,
    total_type_mod: i8,
    is_immune: bool,
    rollout_mult: i32,
    extra_mult: i32,
) -> i32 {
    if is_immune || base_power <= 0 {
        return 0;
    }

    // Castle #19 (`Probe 12E2`): Self-Destruct / Explosion halves physical defense BEFORE >= 256 / 4 check!
    if is_selfdestruct && is_physical {
        def >>= 1;
        if def < 1 {
            def = 1;
        }
    }

    // >= 256 rollover: divide both atk and def by 4.
    if (atk | def) >= 256 {
        atk >>= 2;
        if atk < 1 {
            atk = 1;
        }
        def >>= 2;
        if def < 1 {
            def = 1;
        }
    }

    // Core Gen 2 damage formula (`int(lv*2/5)+2`).
    let mut dmg = ((level * 2 / 5 + 2) * base_power * atk / def) / 50;

    // Critical hit doubles before 1.1x item / 997 clamp.
    if is_crit {
        dmg <<= 1;
    }

    // 1.1x type-boost held item (before +2).
    if has_type_boost_item {
        dmg += dmg / 10;
    }

    // ModifyDamage multiplier (e.g. Stomp vs Minimize = 2x) applies BEFORE clamp(997) + 2.
    if extra_mult > 1 {
        dmg *= extra_mult;
    }

    // Clamp to [0, 997] (note: Castle allows 0 before +2; +2 makes >= 2 before resist/roll), then +2.
    if dmg < 0 {
        dmg = 0;
    } else if dmg > 997 {
        dmg = 997;
    }
    dmg += 2;

    // Castle #26 (`Probe 12E2`): Rollout multiplier applied AFTER +2 and BEFORE STAB/type/roll.
    if rollout_mult > 1 {
        dmg *= rollout_mult;
    }

    // Weather modifier.
    let rain = (weather as u8) == (CompactWeather::Rain as u8);
    let sun = (weather as u8) == (CompactWeather::Sun as u8);
    let w_boost = ((is_water_move & rain) | (is_fire_move & sun)) as i32;
    let w_nerf = (((is_fire_move | is_solarbeam) & rain) | (is_water_move & sun)) as u32;
    dmg += (dmg >> 1) & -w_boost;
    dmg >>= w_nerf;

    // STAB (+50%).
    let is_stab =
        (((move_type < 32) as i32) & ((attacker_type_mask >> (move_type & 31)) as i32)) & 1;
    dmg += (dmg >> 1) & -is_stab;

    // Type effectiveness.
    if total_type_mod > 0 {
        dmg <<= if total_type_mod >= 2 { 2 } else { 1 };
    } else if total_type_mod < 0 {
        dmg >>= if total_type_mod <= -2 { 2 } else { 1 };
    }

    dmg
}

/// Castle PBS damage roll (`217..=255 / 255`) allowing final `0` damage (spec #30, `Probe 5A`).
#[inline(always)]
pub const fn apply_roll_castle(
    base_dmg: i32,
    roll_217_to_255: u32,
    no_damage_variance: bool,
) -> i32 {
    if base_dmg <= 0 {
        return 0;
    }
    if no_damage_variance {
        return base_dmg;
    }
    let num = (base_dmg as u64) * (roll_217_to_255 as u64);
    ((num * 0x808081) >> 31) as i32
}

// =========================================================================
// 5. CompactBattle (256-Byte Fixed-Size, 64B-Aligned Copy Kernel)
// =========================================================================

/// Volatile status bitflags stored in `CompactMon::vol_flags` (`u16`).
pub mod vol_bit {
    pub const CONFUSION: u16 = 1 << 0;
    pub const FLINCH: u16 = 1 << 1;
    pub const DESTINY_BOND: u16 = 1 << 2;
    pub const SUBSTITUTE: u16 = 1 << 3;
    pub const PAR_SPEED_DROP: u16 = 1 << 4;
    pub const BRN_ATK_DROP: u16 = 1 << 5;
    pub const LEECH_SEED: u16 = 1 << 6;
    pub const NIGHTMARE: u16 = 1 << 7;
    pub const CURSE: u16 = 1 << 8;
    pub const FORESIGHT: u16 = 1 << 9;
    pub const PROTECT: u16 = 1 << 10;
    pub const ENDURE: u16 = 1 << 11;
    pub const ENCORE: u16 = 1 << 12;
    pub const DISABLE: u16 = 1 << 13;
    pub const PERISH_SONG: u16 = 1 << 14;
    pub const TRAPPED_OR_LOCKED: u16 = 1 << 15;
}

/// Extended volatile bitflags stored in `CompactMon::ext_flags` (`u16`).
pub mod ext_vol_bit {
    pub const ATTRACT: u16 = 1 << 0;
    pub const MIST: u16 = 1 << 1;
    pub const FOCUS_ENERGY: u16 = 1 << 2;
    pub const LOCK_ON: u16 = 1 << 3;
    pub const DEFENSE_CURL: u16 = 1 << 4;
    pub const MINIMIZE: u16 = 1 << 5;
    pub const RAGE: u16 = 1 << 6;
    pub const TRANSFORMED: u16 = 1 << 7;
    /// Castle #13 (`Probe 11B`): Miracle Berry cured Sleep while Nightmare was active;
    /// Nightmare keeps dealing 1/4 maxhp even while awake.
    pub const MIRACLE_NIGHTMARE: u16 = 1 << 8;
    pub const LOCKED_OUTRAGE: u16 = 1 << 9;
    pub const LOCKED_ROLLOUT: u16 = 1 << 10;
    pub const DIGGING: u16 = 1 << 11;
    pub const FLYING: u16 = 1 << 12;
    /// Castle #53: Target received 1st hit of foe's locked move (Outrage/Thrash/Petal Dance/Rollout),
    /// enabling Counter/Mirror Coat against subsequent hits.
    pub const LOCKED_FIRST_HIT_RECV: u16 = 1 << 13;
    /// Castle #54③: This Pokemon cast Foresight (cleared if caster switches out).
    pub const FORESIGHT_CASTER: u16 = 1 << 14;
    /// Castle #52: Mon used Rage as its last action this turn (eligible for Rage counter increment on hit).
    pub const RAGE_ARMED: u16 = 1 << 15;
}

/// Mutable battle state of a single picked Pokemon (32 bytes, `Copy`, zero padding).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CompactMon {
    /// `[0]` = main HP, `[1]` = Substitute HP (`0` when no Substitute).
    pub hp_pool: [i16; 2],
    /// Current PP of the 4 move slots (`0..=64`).
    pub pp: [u8; 4],
    /// Low 3 bits = `Status` discriminant (`0..=7`), high 5 bits = sleep turns or toxic counter.
    pub status_packed: u8,
    /// Held item slot (`0` = None, `1..=127` = compact item id).
    pub item: u8,
    /// Bitmask of active volatile statuses (see [`vol_bit`]).
    pub vol_flags: u16,
    /// `[0]` = confusion turns, `[1]` = encore turns, `[2]` = disable turns, `[3]` = perish/stall/partial-trap.
    pub timers: [u8; 4],
    /// Stat stages `[atk, def, spa, spd, spe, acc, eva]` (`-6..=+6`).
    pub boosts: [i8; 7],
    /// Packed slot references:
    /// - bits 0..1 (`0x03`): `last_move` slot (`0..3`)
    /// - bits 2..3 (`0x0C`): `encore_slot` (`0..3`)
    /// - bits 4..5 (`0x30`): `disable_slot` (`0..3`)
    /// - bit 6 (`0x40`): `last_move` valid flag
    /// - bit 7 (`0x80`): `mustrecharge` flag (Hyper Beam recharge turn)
    pub slot_refs: u8,
    /// Extended volatile flags (Attract, Mist, Focus Energy, Lock-On, Defense Curl, Minimize,
    /// Outrage/Rollout/Dig/Fly locks, Miracle Nightmare, etc. — see [`ext_vol_bit`]).
    pub ext_flags: u16,
    /// Extended timers (`[locked_turns, partial_trap_turns, fury_cutter_streak, rage_counter]`).
    pub ext_timers: [u8; 4],
    /// Pokemon gender (`0` = Genderless `N`, `1` = Male `M`, `2` = Female `F`).
    pub gender: u8,
    /// Packed Rollout/Lock-On/locked-move metadata:
    /// - bits 0..2 (`0x07`): `rollout_count` (`0..=5`)
    /// - bits 3..4 (`0x18`): `locked_move_slot` (`0..=3`, for Outrage/Thrash/Petal Dance/Rollout/Dig/Fly)
    /// - bits 5..6 (`0x60`): `lock_on_turns` (`0..=2`; PS `2->1->0`, Castle `1` = persistent until consumed)
    /// - bit 7 (`0x80`): `two_turn_locked` (active `twoturnmove` charge lock, distinct from Castle persisted `DIGGING`/`FLYING`)
    pub rollout_rage: u8,
}

const _: () = assert!(std::mem::size_of::<CompactMon>() == 32);

impl CompactMon {
    #[inline(always)]
    pub const fn hp(&self) -> i16 {
        self.hp_pool[0]
    }

    #[inline(always)]
    pub const fn sub_hp(&self) -> i16 {
        self.hp_pool[1]
    }

    #[inline(always)]
    pub const fn status_raw(&self) -> u8 {
        self.status_packed & 0x07
    }

    #[inline(always)]
    pub const fn status_counter(&self) -> u8 {
        self.status_packed >> 3
    }

    #[inline(always)]
    pub fn set_status(&mut self, status_disc: u8, counter: u8) {
        self.status_packed = (status_disc & 0x07) | ((counter & 0x1F) << 3);
    }

    #[inline(always)]
    pub const fn perish_turns(&self) -> u8 {
        self.timers[timer_idx::PERISH_STALL] & 0x07
    }

    #[inline(always)]
    pub fn set_perish_turns(&mut self, turns: u8) {
        self.timers[timer_idx::PERISH_STALL] =
            (self.timers[timer_idx::PERISH_STALL] & !0x07) | (turns.min(7) & 0x07);
    }

    #[inline(always)]
    pub const fn stall_counter(&self) -> u8 {
        (self.timers[timer_idx::PERISH_STALL] >> 3) & 0x07
    }

    #[inline(always)]
    pub fn set_stall_counter(&mut self, count: u8) {
        self.timers[timer_idx::PERISH_STALL] =
            (self.timers[timer_idx::PERISH_STALL] & !0xB8) | ((count.min(7) & 0x07) << 3);
    }

    /// Extended 4-bit stall counter (`0..=8`) using bit 7 of `timers[PERISH_STALL]` for Castle `sc == 8` (`0/256`).
    #[inline(always)]
    pub const fn stall_counter_ext(&self) -> u8 {
        let lo = (self.timers[timer_idx::PERISH_STALL] >> 3) & 0x07;
        let hi = (self.timers[timer_idx::PERISH_STALL] >> 7) & 0x01;
        lo | (hi << 3)
    }

    #[inline(always)]
    pub fn set_stall_counter_ext(&mut self, count: u8) {
        let c = count.min(8);
        self.timers[timer_idx::PERISH_STALL] = (self.timers[timer_idx::PERISH_STALL] & !0xB8)
            | ((c & 0x07) << 3)
            | (((c >> 3) & 0x01) << 7);
    }

    /// Partial-trapping volatile (Wrap / Bind / Fire Spin / Clamp / Whirlpool),
    /// stored in bit 6 of `timers[PERISH_STALL]` so `Rapid Spin` / `Substitute`
    /// clear partial traps without colliding with Mean Look / Spider Web (`vol_bit::TRAPPED_OR_LOCKED`).
    #[inline(always)]
    pub const fn partially_trapped(&self) -> bool {
        (self.timers[timer_idx::PERISH_STALL] & 0x40) != 0
    }

    #[inline(always)]
    pub fn set_partially_trapped(&mut self, on: bool) {
        if on {
            self.timers[timer_idx::PERISH_STALL] |= 0x40;
        } else {
            self.timers[timer_idx::PERISH_STALL] &= !0x40;
            self.ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] = 0;
        }
    }

    #[inline(always)]
    pub const fn is_trapped(&self) -> bool {
        (self.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0 || self.partially_trapped()
    }

    #[inline(always)]
    pub const fn rollout_count(&self) -> u8 {
        self.rollout_rage & 0x07
    }

    #[inline(always)]
    pub fn set_rollout_count(&mut self, count: u8) {
        self.rollout_rage = (self.rollout_rage & !0x07) | (count.min(5) & 0x07);
    }

    #[inline(always)]
    pub const fn locked_move_slot(&self) -> u8 {
        (self.rollout_rage >> 3) & 0x03
    }

    #[inline(always)]
    pub fn set_locked_move_slot(&mut self, slot: u8) {
        self.rollout_rage = (self.rollout_rage & !0x18) | ((slot & 0x03) << 3);
    }

    #[inline(always)]
    pub const fn lock_on_turns(&self) -> u8 {
        (self.rollout_rage >> 5) & 0x03
    }

    #[inline(always)]
    pub fn set_lock_on_turns(&mut self, turns: u8) {
        self.rollout_rage = (self.rollout_rage & !0x60) | ((turns.min(3) & 0x03) << 5);
    }

    #[inline(always)]
    pub const fn two_turn_locked(&self) -> bool {
        (self.rollout_rage & 0x80) != 0
    }

    #[inline(always)]
    pub fn set_two_turn_locked(&mut self, on: bool) {
        if on {
            self.rollout_rage |= 0x80;
        } else {
            self.rollout_rage &= !0x80;
        }
    }

    /// True if forced into a multi-turn locked move (Outrage/Thrash/Petal Dance, Rollout, or active Dig/Fly `twoturnmove` lock).
    /// Distinct from Castle's persisted `DIGGING`/`FLYING` semi-invulnerability after Turn-2 abort.
    #[inline(always)]
    pub const fn is_multi_turn_locked(&self) -> bool {
        (self.ext_flags & (ext_vol_bit::LOCKED_OUTRAGE | ext_vol_bit::LOCKED_ROLLOUT)) != 0
            || self.two_turn_locked()
    }

    #[inline(always)]
    pub fn clear_multi_turn_locks(&mut self) {
        self.ext_flags &= !(ext_vol_bit::LOCKED_OUTRAGE
            | ext_vol_bit::LOCKED_ROLLOUT
            | ext_vol_bit::DIGGING
            | ext_vol_bit::FLYING);
        self.set_two_turn_locked(false);
        self.set_locked_move_slot(0);
        self.ext_timers[ext_timer_idx::LOCKED_TURNS] = 0;
        self.set_rollout_count(0);
    }

    #[inline(always)]
    pub const fn last_move_slot(&self) -> Option<u8> {
        if (self.slot_refs & 0x40) != 0 {
            Some(self.slot_refs & 0x03)
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn set_last_move_slot(&mut self, slot: u8) {
        self.slot_refs = (self.slot_refs & !0x43) | (slot & 0x03) | 0x40;
    }

    #[inline(always)]
    pub fn clear_last_move_slot(&mut self) {
        self.slot_refs &= !0x43;
    }

    #[inline(always)]
    pub const fn encore_slot(&self) -> u8 {
        (self.slot_refs >> 2) & 0x03
    }

    #[inline(always)]
    pub fn set_encore_slot(&mut self, slot: u8) {
        self.slot_refs = (self.slot_refs & !(0x03 << 2)) | ((slot & 0x03) << 2);
    }

    #[inline(always)]
    pub const fn disable_slot(&self) -> u8 {
        (self.slot_refs >> 4) & 0x03
    }

    #[inline(always)]
    pub fn set_disable_slot(&mut self, slot: u8) {
        self.slot_refs = (self.slot_refs & !(0x03 << 4)) | ((slot & 0x03) << 4);
    }

    #[inline(always)]
    pub const fn is_slot_disabled(&self, slot: usize) -> bool {
        (self.vol_flags & vol_bit::DISABLE) != 0 && (self.disable_slot() as usize) == slot
    }

    #[inline(always)]
    pub const fn must_recharge(&self) -> bool {
        (self.slot_refs & 0x80) != 0
    }

    #[inline(always)]
    pub fn set_must_recharge(&mut self, on: bool) {
        if on {
            self.slot_refs |= 0x80;
        } else {
            self.slot_refs &= !0x80;
        }
    }

    /// Canonicalize a fainted Pokemon's mutable fields so `search_hash()` and `PartialEq`
    /// never retain stale boosts, timers, slot refs, or Substitute HP after KO.
    /// Preserves `pp`, `item`, and `gender`.
    #[inline(always)]
    pub fn mark_fainted(&mut self) {
        self.hp_pool = [0, 0];
        self.status_packed = Status::Fnt as u8;
        self.vol_flags = 0;
        self.timers = [0; 4];
        self.boosts = [0; 7];
        self.slot_refs = 0;
        self.ext_flags = 0;
        self.ext_timers = [0; 4];
        self.rollout_rage = 0;
    }

    /// Apply damage to either Substitute (`hp_pool[1]`) or main HP (`hp_pool[0]`)
    /// using branchless array indexing. Returns actual damage dealt.
    #[inline(always)]
    pub fn apply_damage_branchless(&mut self, dmg: i16, bypasses_sub: bool) -> i16 {
        let has_sub = (self.hp_pool[1] > 0) & (!bypasses_sub);
        let idx = has_sub as usize;
        let cur = self.hp_pool[idx];
        let dealt = if dmg < cur { dmg.max(0) } else { cur.max(0) };
        self.hp_pool[idx] = cur - dealt;
        if self.hp_pool[1] <= 0 {
            self.vol_flags &= !vol_bit::SUBSTITUTE;
        }
        if self.hp_pool[0] <= 0 {
            self.mark_fainted();
        }
        dealt
    }
}

/// Mutable state of one side (12 bytes, `Copy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CompactSide {
    /// Active party index (`0..=2`).
    pub active_idx: u8,
    /// Original roster indices (`0..=5`) for the 3 picked party members.
    pub party_roster: [u8; 3],
    /// Bitmask of alive party indices (`bit 0..2`, `count_ones() == pokemon_left`).
    pub alive_mask: u8,
    /// Reflect remaining turns (`0..=5`).
    pub reflect_turns: u8,
    /// Light Screen remaining turns (`0..=5`).
    pub lightscreen_turns: u8,
    /// Safeguard remaining turns (`0..=5`).
    pub safeguard_turns: u8,
    /// Packed side flags:
    /// - bits 0..1 (`0x03`): Spikes layers (`0..3`)
    /// - bit 2 (`0x04`): reserved (`0`, never set)
    /// - bit 3 (`0x08`, `sides[0]` only): `RESIDUALS_DONE` (prevents double residuals on post-residual switch)
    /// - bit 4 (`0x10`) + bits 5..6 (`0x60`): pending mid-turn move slot after Baton Pass
    /// - bit 7 (`0x80`, `sides[0]` only): Baton Pass fix disabled (`0` = Baton Pass fix active)
    pub side_flags: u8,
    /// Last move used for Self-KO clause tracking (`u8::MAX` = None).
    pub last_move_selfko: u8,
    /// Current request kind (`0` = Wait/None, `1` = Move, `2` = Switch, `3` = TeamPreview).
    pub request: u8,
    /// High static match fingerprint byte (`(fp >> 16)` on side 0, `(fp >> 24)` on side 1).
    pub pad: u8,
}

const _: () = assert!(std::mem::size_of::<CompactSide>() == 12);

/// Complete mutable battle state in exactly 256 bytes (4 cache lines, 64-byte aligned, zero padding).
/// Zero heap allocations: `Clone` is a 4-cache-line register copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct CompactBattle {
    /// 6 picked Pokemon (`0..3` = Side 0 party, `3..6` = Side 1 party): 192 bytes (offset 0..192).
    pub mons: [CompactMon; 6],
    /// 2 sides: 24 bytes (offset 192..216).
    pub sides: [CompactSide; 2],
    /// Current turn number (`1..=1001`): 2 bytes (offset 216..218).
    pub turn: u16,
    /// Weather kind (`None`, `Rain`, `Sun`, `Sandstorm`): 1 byte (offset 218).
    pub weather_kind: CompactWeather,
    /// Weather remaining turns (`0..=5`): 1 byte (offset 219).
    pub weather_turns: u8,
    /// Pre-rolled Quick Claw activation mask (`bit 0 = P1, bit 1 = P2`): 1 byte (offset 220).
    pub quick_claw_roll: u8,
    /// Terminal state (`0` = Ongoing, `1` = P1Win, `2` = P2Win, `3` = Tie): 1 byte (offset 221).
    pub outcome: u8,
    /// Speed order bit (`0` = P1 faster/won tie, `1` = P2 faster/won tie): 1 byte (offset 222).
    pub speed_order: u8,
    /// Clause flags (`bit 0 = sleep_clause`, `bit 2 = freeze_clause`): 1 byte (offset 223).
    pub clause_flags: u8,
    /// Last damage dealt to `[P1, P2]` this turn (for Counter / Mirror Coat): 4 bytes (offset 224..228).
    pub hurt_damage_this_turn: [u16; 2],
    /// Category of last move that damaged `[P1, P2]` (`0` = None, `1` = Physical, `2` = Special): 2 bytes (offset 228..230).
    pub hurt_cat_this_turn: [u8; 2],
    /// Low 16 bits of the static match fingerprint (bits 16..32 stored in `sides[0..2].pad`): 2 bytes (offset 230..232).
    pub match_fp_lo: u16,
    /// Precomputed Future Sight damage targeting `[P1 side, P2 side]`: 4 bytes (offset 232..236).
    pub future_sight_dmg: [u16; 2],
    /// Remaining turns before Future Sight strikes `[P1 side, P2 side]` (`0` = none, `3->2->1->0`): 2 bytes (offset 236..238).
    pub future_sight_turns: [u8; 2],
    /// Party slot (`0..=2`) of caster on opposite side for Future Sight (`0` when inactive): 2 bytes (offset 238..240).
    pub future_sight_src: [u8; 2],
    /// Explicit Baton Pass / mid-turn switch flags (1 byte, offset 240):
    /// - bit 0 (`0x01`): P1 `request == 2` is an explicit `BATONN` Baton Pass switch
    /// - bit 1 (`0x02`): P2 `request == 2` is an explicit `BATONN` Baton Pass switch
    /// - bit 2 (`0x04`): P1 arrived via first-mover Baton Pass this turn (Castle 2x Pursuit target)
    /// - bit 3 (`0x08`): P2 arrived via first-mover Baton Pass this turn (Castle 2x Pursuit target)
    /// - bit 4 (`0x10`): P1 pending mid-turn move is forced Struggle
    /// - bit 5 (`0x20`): P2 pending mid-turn move is forced Struggle
    /// - bit 6 (`0x40`): P1 used Rest first this turn (deferred post-second-mover Mint/Miracle check)
    /// - bit 7 (`0x80`): P2 used Rest first this turn (deferred post-second-mover Mint/Miracle check)
    pub bp_switch_flags: u8,
    /// Castle #51: Triple Kick 2x/3x damage multiplier leak to second mover this turn (`0` = none, `2` or `3`): 1 byte (offset 241).
    pub triple_kick_mult: u8,
    /// Extra turn / overlay tracking flags (1 byte, offset 242):
    /// - bit 0 (`0x01`): recoil attacker side valid this turn
    /// - bit 1 (`0x02`): recoil attacker side (`0` = P1, `1` = P2)
    /// - bit 2 (`0x04`): P1 cast active Perish Song (for `perish_dbond_lastmon_forfeit`)
    /// - bit 3 (`0x08`): P2 cast active Perish Song (for `perish_dbond_lastmon_forfeit`)
    /// - bit 4 (`0x10`): P1 fainted foe via Destiny Bond this turn
    /// - bit 5 (`0x20`): P2 fainted foe via Destiny Bond this turn
    /// - bit 6 (`0x40`): P1 missed Fissure/Horn Drill this turn (`ovfl=1` instant-KO Counter)
    /// - bit 7 (`0x80`): P2 missed Fissure/Horn Drill this turn (`ovfl=1` instant-KO Counter)
    pub turn_extra_flags: u8,
    /// Byte `0..2`: `Battle::side.party` permutation code (`0..=5`, default `0` = `[0, 1, 2]`)
    /// matching `Battle::switch_in` swap semantics so bench ordering, forced-switch tie-breaking,
    /// and `sync_to_battle` roundtrips match `Battle` 1:1.
    /// Bytes `2..5`: zero-initialized reserved bytes (offset 243..248).
    pub reserved_castle: [u8; 5],
    /// 64-bit PRNG state (Gen5 LCG seed), word 31 (skipped by `search_hash`): 8 bytes (offset 248..256).
    pub prng_seed: u64,
}

const PARTY_PERMS: [[u8; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Transition table `PARTY_SWAP_ACTIVE[perm_code][target_slot]` matching `Battle::switch_in`
/// (`side.party.swap(0, pos)` where `party[pos] == target_slot`).
const PARTY_SWAP_ACTIVE: [[u8; 3]; 6] = [
    [0, 2, 5],
    [1, 3, 4],
    [0, 2, 4],
    [1, 3, 5],
    [1, 2, 4],
    [0, 3, 5],
];

const _: () = assert!(std::mem::size_of::<CompactBattle>() == 256);
const _: () = assert!(std::mem::align_of::<CompactBattle>() == 64);

/// Precomputed per-Pokemon move-role bitflags on [`StaticMonSpec::role_flags`].
pub mod role_bit {
    pub const HALF_HEAL: u8 = 1 << 0;
    pub const REST: u8 = 1 << 1;
    pub const SLEEP_USABLE: u8 = 1 << 2;
    pub const PERISH_PHAZE_TRAP: u8 = 1 << 3;
    pub const BATON_PASS: u8 = 1 << 4;
    pub const PROTECT: u8 = 1 << 5;
    pub const SWAGGER: u8 = 1 << 6;
    pub const PHAZE_OR_HAZE: u8 = 1 << 7;
}

/// Immutable per-Pokemon reference data stored in [`MatchTable`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticMonSpec {
    /// Precomputed 64-bit static fingerprint of this Pokemon's species, stats, item, and moves.
    pub static_fp: u64,
    /// Bitmask of the Pokemon's 1–2 types (`(1 << t0) | (1 << t1)`).
    pub type_mask: u32,
    pub species: SpeciesId,
    pub maxhp: i16,
    /// 1/8 of maxhp (min 1) for standard poison/burn/leech seed slip damage.
    pub quantum_1_8: i16,
    /// 1/16 of maxhp (min 1) for Leftovers and Toxic base slip damage.
    pub quantum_1_16: i16,
    /// Stored stats `[atk, def, spa, spd, spe]`.
    pub stored_stats: [i16; 5],
    pub moves: [MoveId; 4],
    pub max_pp: [u8; 4],
    pub types: [u8; 2],
    pub num_types: u8,
    pub level: u8,
    pub num_moves: u8,
    pub role_flags: u8,
    /// Species-eligible stat-boost item kind (Cubone/Marowak = ThickClub, Pikachu = LightBall, Ditto = MetalPowder),
    /// independent of initial held item so stolen items work dynamically.
    pub species_stat_kind: ItemStatKind,
    /// Species base Attack stat (for Beat Up).
    pub base_atk: u8,
    /// Species base Defense stat (for Beat Up).
    pub base_def: u8,
    /// Gen 2 Present type number of the Pokemon's last type (Normal/Fighting = 1, ..., Dark = 27).
    pub present_type_num: u8,
    /// Packed species flags (`bit 0 = is_farfetchd`, `bit 1 = is_chansey`, `bit 2 = is_ditto_base`).
    pub species_flags: u8,
}

/// Immutable per-move compiled metadata for branchless execution.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactMoveSpec {
    pub id: MoveId,
    pub base_power: i16,
    /// Fixed damage (`-1` = Level, `> 0` = Amount(n), `0` = None).
    pub fixed_damage: i16,
    /// Primary volatile inflicted on target (`vol_bit` mask).
    pub primary_vol: u16,
    /// Secondary volatile inflicted on target (`vol_bit` mask).
    pub sec_vol: u16,
    pub move_type: u8,
    pub is_physical: bool,
    pub is_status: bool,
    pub priority: i8,
    pub base_acc_pct: u8,
    pub always_hit: bool,
    pub ignore_accuracy: bool,
    pub ignore_evasion: bool,
    pub ignore_positive_evasion: bool,
    pub crit_ratio: u8,
    pub is_selfdestruct: bool,
    pub is_water: bool,
    pub is_fire: bool,
    pub is_solarbeam: bool,
    pub bypasses_sub: bool,
    pub is_rest: bool,
    pub is_half_heal: bool,
    pub is_recharge: bool,
    pub is_defrost: bool,
    pub is_heal_cycle: bool,
    pub is_sleep_usable: bool,
    pub can_sleep_talk: bool,
    pub is_protect: bool,
    pub is_endure: bool,
    pub blocked_by_protect: bool,
    pub is_substitute: bool,
    pub drain_half: bool,
    pub recoil_quarter: bool,
    /// Primary status inflicted on target (`0` = None, else `Status as u8`).
    pub primary_status: u8,
    /// Secondary chance out of 256 (`0` = None; `255` = 100% secondary).
    pub sec_chance_256: u8,
    /// Secondary status inflicted on target (`0` = None, else `Status as u8`).
    pub sec_status: u8,
    /// Secondary stat stage delta (`(stat_idx, delta)` or `(0xFF, 0)`).
    pub sec_boost: (u8, i8),
    /// Self-targeted or target-applied stat boosts (`[(stat_idx_or_ff, delta); 3]`).
    pub boosts: [(u8, i8); 3],
    pub boosts_self: bool,
    /// Special move effect tag (`SpecialMoveKind`).
    pub special_kind: SpecialMoveKind,
    /// Variable base power kind (`VarBpKind`).
    pub var_bp_kind: VarBpKind,
    /// Twice the expected hit count (`2` = single hit, `6` = 2..5 multihit, `2*n` = Fixed(n)).
    pub multihit_mean_x2: u8,
    /// Minimum hits (`1`, `2`).
    pub multihit_min: u8,
    /// Maximum hits (`1`, `2`, `3`, `5`).
    pub multihit_max: u8,
    /// Whether King's Rock can apply a 30/256 flinch secondary on this move.
    pub kingsrock: bool,
    /// Whether Encore fails against this move (`failencore` flag).
    pub fail_encore: bool,
    /// Whether this move is Fissure or Horn Drill (`ovfl=1` instant-KO Counter).
    pub is_fissure_or_horndrill: bool,
    /// Whether this move is specifically Fissure (hits underground Dig).
    pub is_fissure: bool,
    /// Whether this move is specifically Whirlwind (hits airborne Fly).
    pub is_whirlwind: bool,
    /// Whether this status/non-damaging move ignores type immunities (e.g. Lovely Kiss/Sing = true; Glare/Thunder Wave = false).
    pub ignore_immunity: bool,
}

/// Precompiled per-item metadata for O(1) lookup in `CompactBattle`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactItemSpec {
    pub boost_type: u8,
    /// Type boosted by this item in Castle PBS (`dragonscale` = Dragon, `dragonfang` = inert).
    pub castle_boost_type: u8,
    pub stat_kind: ItemStatKind,
    pub is_quick_claw: bool,
    pub is_leftovers: bool,
    pub is_brightpowder: bool,
    pub is_mintberry: bool,
    pub is_miracleberry: bool,
    pub is_goldberry: bool,
    pub is_berry_juice: bool,
    pub is_berry: bool,
    pub is_scope_lens: bool,
    pub is_focus_band: bool,
    pub is_kings_rock: bool,
    pub is_mystery_berry: bool,
    pub is_ice_berry: bool,
    pub is_bitter_berry: bool,
    pub is_przcure_berry: bool,
    pub is_psncure_berry: bool,
    pub is_burnt_berry: bool,
    pub is_stick: bool,
    pub is_lucky_punch: bool,
    pub is_berserk_gene: bool,
}

static DEFAULT_ITEM_SPEC: CompactItemSpec = CompactItemSpec {
    boost_type: NO_TYPE_ID,
    castle_boost_type: NO_TYPE_ID,
    stat_kind: ItemStatKind::None,
    is_quick_claw: false,
    is_leftovers: false,
    is_brightpowder: false,
    is_mintberry: false,
    is_miracleberry: false,
    is_goldberry: false,
    is_berry_juice: false,
    is_berry: false,
    is_scope_lens: false,
    is_focus_band: false,
    is_kings_rock: false,
    is_mystery_berry: false,
    is_ice_berry: false,
    is_bitter_berry: false,
    is_przcure_berry: false,
    is_psncure_berry: false,
    is_burnt_berry: false,
    is_stick: false,
    is_lucky_punch: false,
    is_berserk_gene: false,
};

/// Process-wide precompiled Dex tables for `CompactBattle` and fast evaluation.
pub struct CompactDexTables {
    pub type_chart: [[i8; 20]; 20],
    pub special_type_mask: u32,
    pub normal_type: u8,
    pub fighting_type: u8,
    pub flying_type: u8,
    pub poison_type: u8,
    pub ground_type: u8,
    pub rock_type: u8,
    pub ghost_type: u8,
    pub steel_type: u8,
    pub fire_type: u8,
    pub water_type: u8,
    pub grass_type: u8,
    pub electric_type: u8,
    pub ice_type: u8,
    pub dragon_type: u8,
    pub sandstorm_immune_mask: u32,
    pub poison_immune_mask: u32,
    pub move_specs: Vec<CompactMoveSpec>,
    pub item_specs: Vec<CompactItemSpec>,
    pub sub_cond: Option<crate::dex::CondId>,
    pub conf_cond: Option<crate::dex::CondId>,
    pub flinch_cond: Option<crate::dex::CondId>,
    pub dbond_cond: Option<crate::dex::CondId>,
    pub par_cond: Option<crate::dex::CondId>,
    pub brn_cond: Option<crate::dex::CondId>,
    pub seed_cond: Option<crate::dex::CondId>,
    pub night_cond: Option<crate::dex::CondId>,
    pub curse_cond: Option<crate::dex::CondId>,
    pub fore_cond: Option<crate::dex::CondId>,
    pub prot_cond: Option<crate::dex::CondId>,
    pub end_cond: Option<crate::dex::CondId>,
    pub stall_cond: Option<crate::dex::CondId>,
    pub enc_cond: Option<crate::dex::CondId>,
    pub dis_cond: Option<crate::dex::CondId>,
    pub perish_cond: Option<crate::dex::CondId>,
    pub res_cond: Option<crate::dex::CondId>,
    pub recharge_cond: Option<crate::dex::CondId>,
    pub trapped_cond: Option<crate::dex::CondId>,
    pub partiallytrapped_cond: Option<crate::dex::CondId>,
    pub attract_cond: Option<crate::dex::CondId>,
    pub mist_cond: Option<crate::dex::CondId>,
    pub focusenergy_cond: Option<crate::dex::CondId>,
    pub lockon_cond: Option<crate::dex::CondId>,
    pub defensecurl_cond: Option<crate::dex::CondId>,
    pub minimize_cond: Option<crate::dex::CondId>,
    pub rollout_cond: Option<crate::dex::CondId>,
    pub lockedmove_cond: Option<crate::dex::CondId>,
    pub furycutter_cond: Option<crate::dex::CondId>,
    pub rage_cond: Option<crate::dex::CondId>,
    pub dig_cond: Option<crate::dex::CondId>,
    pub fly_cond: Option<crate::dex::CondId>,
    pub twoturnmove_cond: Option<crate::dex::CondId>,
    pub futuremove_cond: Option<crate::dex::CondId>,
    pub refl_cond: Option<crate::dex::CondId>,
    pub ls_cond: Option<crate::dex::CondId>,
    pub safe_cond: Option<crate::dex::CondId>,
    pub spikes_cond: Option<crate::dex::CondId>,
    pub rain_cond: Option<crate::dex::CondId>,
    pub sun_cond: Option<crate::dex::CondId>,
    pub sand_cond: Option<crate::dex::CondId>,
    pub sleep_clause_cond: Option<crate::dex::CondId>,
    pub freeze_clause_cond: Option<crate::dex::CondId>,
}

static COMPACT_DEX_TABLES: OnceLock<CompactDexTables> = OnceLock::new();

/// Access the process-wide precompiled `CompactDexTables`.
#[inline]
pub fn dex_tables(dex: &Dex) -> &'static CompactDexTables {
    COMPACT_DEX_TABLES.get_or_init(|| build_compact_dex_tables(dex))
}

fn build_compact_dex_tables(dex: &Dex) -> CompactDexTables {
    let mut type_chart = [[0i8; 20]; 20];
    let n_types = dex.type_names.len().min(19);
    for atk_t in 0..n_types {
        for def_t in 0..n_types {
            let atk_id = crate::dex::TypeId(atk_t as u8);
            let def_id = crate::dex::TypeId(def_t as u8);
            if dex.type_immune(atk_id, def_id) {
                type_chart[atk_t][def_t] = -9;
            } else {
                type_chart[atk_t][def_t] = dex.eff(atk_id, def_id) as i8;
            }
        }
    }

    let mut special_type_mask = 0u32;
    for t_name in [
        "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
    ] {
        if let Some(tid) = dex.type_id(t_name) {
            special_type_mask |= 1u32 << (tid.0 & 31);
        }
    }

    let tid_or = |name: &str, fallback: u8| dex.type_id(name).map(|t| t.0).unwrap_or(fallback);
    let normal_type = dex.known_types.normal.0;
    let fighting_type = dex.known_types.fighting.0;
    let flying_type = dex.known_types.flying.0;
    let poison_type = tid_or("Poison", NO_TYPE_ID);
    let ground_type = dex.known_types.ground.0;
    let rock_type = tid_or("Rock", NO_TYPE_ID);
    let ghost_type = dex.known_types.ghost.0;
    let steel_type = tid_or("Steel", NO_TYPE_ID);
    let fire_type = dex.known_types.fire.0;
    let water_type = dex.known_types.water.0;
    let grass_type = dex.known_types.grass.0;
    let electric_type = tid_or("Electric", NO_TYPE_ID);
    let ice_type = tid_or("Ice", NO_TYPE_ID);
    let dragon_type = tid_or("Dragon", NO_TYPE_ID);

    let mut sandstorm_immune_mask = 0u32;
    for &t in &[rock_type, ground_type, steel_type] {
        if t < 32 {
            sandstorm_immune_mask |= 1u32 << t;
        }
    }
    let mut poison_immune_mask = 0u32;
    for &t in &[poison_type, steel_type] {
        if t < 32 {
            poison_immune_mask |= 1u32 << t;
        }
    }

    let parse_status_code = |s: Option<&str>| -> u8 {
        match s {
            Some("slp") => Status::Slp as u8,
            Some("psn") => Status::Psn as u8,
            Some("brn") => Status::Brn as u8,
            Some("frz") => Status::Frz as u8,
            Some("par") => Status::Par as u8,
            Some("tox") => Status::Tox as u8,
            _ => 0,
        }
    };
    let parse_vol_code = |s: Option<&str>| -> u16 {
        match s {
            Some("confusion") => vol_bit::CONFUSION,
            Some("flinch") => vol_bit::FLINCH,
            Some("leechseed") => vol_bit::LEECH_SEED,
            Some("nightmare") => vol_bit::NIGHTMARE,
            Some("curse") => vol_bit::CURSE,
            Some("foresight") => vol_bit::FORESIGHT,
            Some("encore") => vol_bit::ENCORE,
            Some("disable") => vol_bit::DISABLE,
            Some("perishsong") => vol_bit::PERISH_SONG,
            Some("destinybond") => vol_bit::DESTINY_BOND,
            _ => 0,
        }
    };

    let bypasssub_bit = dex.flag_bit("bypasssub");
    let nosleeptalk_bit = dex.flag_bit("nosleeptalk");
    let charge_bit = dex.flag_bit("charge");
    let protect_bit = dex.flag_bit("protect");
    let failencore_bit = dex.flag_bit("failencore");

    let km = &dex.known_moves;
    let ki = &dex.known_items;
    let mut move_specs = Vec::with_capacity(dex.moves.len());
    for m_raw in 0..dex.moves.len() {
        let mid = MoveId(m_raw as u16);
        let ms = dex.move_static(mid);
        let mkey = dex.moves.key(mid);
        let var_bp_kind = match mkey {
            "return" => VarBpKind::Return,
            "frustration" => VarBpKind::Frustration,
            "hiddenpower" => VarBpKind::HiddenPower,
            "flail" | "reversal" => VarBpKind::FlailReversal,
            "magnitude" => VarBpKind::Magnitude,
            "present" => VarBpKind::Present,
            _ => VarBpKind::Static,
        };
        let bp = match var_bp_kind {
            VarBpKind::Magnitude => 71,
            VarBpKind::Present => 40,
            _ => ms.base_power as i16,
        };
        let (base_acc_pct, always_hit) = match ms.accuracy {
            Accuracy::AlwaysHits => (100, true),
            Accuracy::Pct(a) => (a.clamp(0, 100) as u8, ms.always_hit),
        };
        let primary_status = parse_status_code(ms.status);
        let primary_vol = parse_vol_code(ms.volatile_status);
        let mut sec_chance_256 = 0u8;
        let mut sec_status = 0u8;
        let mut sec_vol = 0u16;
        let mut sec_boost = (NO_STAT_IDX, 0i8);
        if let Some(sec) = ms.secondaries.first() {
            let ch = sec.chance.unwrap_or(100).clamp(0, 100);
            sec_chance_256 = (ch * 255 / 100).clamp(1, 255) as u8;
            sec_status = parse_status_code(sec.status);
            sec_vol = parse_vol_code(sec.volatile_status);
            if let Some(&(st_i, d)) = sec.boosts.first() {
                sec_boost = (st_i as u8, d);
            } else if let Some(se) = sec.self_effect.as_ref() {
                if mkey == "ancientpower" {
                    sec_boost = (SELF_BOOST_FLAG | ANCIENTPOWER_ALL_STATS, 1);
                } else if let Some(&(st_i, d)) = se.boosts.first() {
                    sec_boost = ((st_i as u8) | SELF_BOOST_FLAG, d);
                }
            }
        }
        let mut boosts = [(NO_STAT_IDX, 0i8); 3];
        let mut boosts_self = matches!(ms.target, "self" | "allySide" | "adjacentAllyOrSelf");
        let src_boosts = if !ms.boosts.is_empty() {
            &ms.boosts[..]
        } else if let Some(se) = ms.self_effect.as_ref() {
            boosts_self = true;
            &se.boosts[..]
        } else {
            &[][..]
        };
        for (b_i, &(st_i, d)) in src_boosts.iter().take(3).enumerate() {
            boosts[b_i] = (st_i as u8, d);
        }
        let special_kind = match mkey {
            "reflect" => SpecialMoveKind::Reflect,
            "lightscreen" => SpecialMoveKind::LightScreen,
            "safeguard" => SpecialMoveKind::Safeguard,
            "spikes" => SpecialMoveKind::Spikes,
            "curse" => SpecialMoveKind::Curse,
            "bellydrum" => SpecialMoveKind::BellyDrum,
            "roar" | "whirlwind" => SpecialMoveKind::Phaze,
            "encore" => SpecialMoveKind::Encore,
            "disable" => SpecialMoveKind::Disable,
            "sleeptalk" => SpecialMoveKind::SleepTalk,
            "counter" => SpecialMoveKind::Counter,
            "mirrorcoat" => SpecialMoveKind::MirrorCoat,
            "flail" | "reversal" => SpecialMoveKind::FlailReversal,
            "seismictoss" | "nightshade" => SpecialMoveKind::LevelDamage,
            "superfang" => SpecialMoveKind::SuperFang,
            "thief" => SpecialMoveKind::Thief,
            "painsplit" => SpecialMoveKind::PainSplit,
            "batonpass" => SpecialMoveKind::BatonPass,
            "perishsong" => SpecialMoveKind::PerishSong,
            "destinybond" => SpecialMoveKind::DestinyBond,
            "healbell" => SpecialMoveKind::HealBell,
            "psychup" => SpecialMoveKind::PsychUp,
            "raindance" => SpecialMoveKind::RainDance,
            "sunnyday" => SpecialMoveKind::SunnyDay,
            "sandstorm" => SpecialMoveKind::Sandstorm,
            "meanlook" | "spiderweb" => SpecialMoveKind::MeanLook,
            "haze" => SpecialMoveKind::Haze,
            "rapidspin" => SpecialMoveKind::RapidSpin,
            "moonlight" | "morningsun" | "synthesis" => SpecialMoveKind::WeatherHeal,
            "swagger" => SpecialMoveKind::Swagger,
            "pursuit" => SpecialMoveKind::Pursuit,
            "present" => SpecialMoveKind::Present,
            "rollout" => SpecialMoveKind::Rollout,
            "defensecurl" => SpecialMoveKind::DefenseCurl,
            "minimize" => SpecialMoveKind::Minimize,
            "outrage" | "thrash" | "petaldance" => SpecialMoveKind::OutrageLock,
            "attract" => SpecialMoveKind::Attract,
            "mist" => SpecialMoveKind::Mist,
            "focusenergy" => SpecialMoveKind::FocusEnergy,
            "highjumpkick" | "jumpkick" => SpecialMoveKind::CrashKick,
            "futuresight" => SpecialMoveKind::FutureSight,
            "lockon" | "mindreader" => SpecialMoveKind::LockOn,
            "beatup" => SpecialMoveKind::BeatUp,
            "furycutter" => SpecialMoveKind::FuryCutter,
            "rage" => SpecialMoveKind::Rage,
            "triplekick" => SpecialMoveKind::TripleKick,
            "twineedle" => SpecialMoveKind::Twineedle,
            "triattack" => SpecialMoveKind::TriAttack,
            "wrap" | "bind" | "firespin" | "clamp" | "whirlpool" => SpecialMoveKind::PartialTrap,
            "dig" => SpecialMoveKind::Dig,
            "fly" => SpecialMoveKind::Fly,
            "falseswipe" => SpecialMoveKind::FalseSwipe,
            "psywave" => SpecialMoveKind::Psywave,
            "spite" => SpecialMoveKind::Spite,
            "fissure" | "horndrill" | "guillotine" => SpecialMoveKind::Ohko,
            "dreameater" => SpecialMoveKind::DreamEater,
            "magnitude" => SpecialMoveKind::Magnitude,
            "thunder" => SpecialMoveKind::Thunder,
            "stomp" => SpecialMoveKind::Stomp,
            "earthquake" => SpecialMoveKind::Earthquake,
            "gust" | "twister" => SpecialMoveKind::GustTwister,
            _ => SpecialMoveKind::None,
        };
        let fixed_damage = match ms.damage {
            Some(crate::dex::FixedDamage::Level) => FIXED_DAMAGE_LEVEL,
            Some(crate::dex::FixedDamage::Amount(n)) => n.clamp(1, i16::MAX as i32) as i16,
            None => 0,
        };
        let (multihit_mean_x2, multihit_min, multihit_max) = match ms.multihit {
            Some(crate::dex::Multihit::Fixed(n)) => {
                let nc = n.clamp(1, 10) as u8;
                ((nc * 2).clamp(2, 20), nc, nc)
            }
            Some(crate::dex::Multihit::Range(2, 5)) => (6, 2, 5),
            Some(crate::dex::Multihit::Range(lo, hi)) => {
                let l = lo.clamp(1, 10) as u8;
                let h = hi.clamp(l as i32, 10) as u8;
                ((l + h).clamp(2, 20), l, h)
            }
            None => (2, 1, 1),
        };

        let smid = Some(mid);
        let is_half_heal = smid == km.recover || smid == km.softboiled || smid == km.milkdrink;
        let is_rest = smid == km.rest;
        move_specs.push(CompactMoveSpec {
            id: mid,
            base_power: bp,
            move_type: ms.move_type.0,
            is_physical: ms.category == Category::Physical,
            is_status: ms.category == Category::Status,
            priority: ms.priority,
            base_acc_pct,
            always_hit,
            ignore_accuracy: ms.ignore_accuracy,
            ignore_evasion: ms.ignore_evasion,
            ignore_positive_evasion: ms.ignore_positive_evasion,
            crit_ratio: ms.crit_ratio.clamp(0, 5) as u8,
            is_selfdestruct: ms.selfdestruct,
            is_water: ms.move_type.0 == water_type,
            is_fire: ms.move_type.0 == fire_type,
            is_solarbeam: smid == km.solarbeam,
            bypasses_sub: (ms.flags & bypasssub_bit) != 0,
            is_rest,
            is_half_heal,
            is_recharge: ms.has_flag(dex, "recharge"),
            is_defrost: ms.has_flag(dex, "defrost"),
            is_heal_cycle: is_rest || is_half_heal,
            is_sleep_usable: ms.sleep_usable,
            can_sleep_talk: (ms.flags & (nosleeptalk_bit | charge_bit)) == 0,
            is_protect: smid == km.protect || smid == km.detect,
            is_endure: smid == km.endure,
            blocked_by_protect: (ms.flags & protect_bit) != 0,
            is_substitute: smid == km.substitute,
            drain_half: ms.drain.is_some(),
            recoil_quarter: ms.recoil.is_some(),
            primary_status,
            primary_vol: if ms.target == "self"
                || matches!(
                    special_kind,
                    SpecialMoveKind::Curse
                        | SpecialMoveKind::Encore
                        | SpecialMoveKind::Disable
                        | SpecialMoveKind::PerishSong
                        | SpecialMoveKind::DestinyBond
                        | SpecialMoveKind::Attract
                        | SpecialMoveKind::Mist
                        | SpecialMoveKind::FocusEnergy
                        | SpecialMoveKind::DefenseCurl
                        | SpecialMoveKind::Minimize
                        | SpecialMoveKind::LockOn
                        | SpecialMoveKind::OutrageLock
                        | SpecialMoveKind::Rollout
                        | SpecialMoveKind::Rage
                        | SpecialMoveKind::PartialTrap
                        | SpecialMoveKind::Dig
                        | SpecialMoveKind::Fly
                        | SpecialMoveKind::FutureSight
                ) {
                0
            } else {
                primary_vol
            },
            sec_chance_256,
            sec_status,
            sec_vol,
            sec_boost,
            boosts,
            boosts_self,
            special_kind,
            var_bp_kind,
            fixed_damage,
            multihit_mean_x2,
            multihit_min,
            multihit_max,
            kingsrock: ms.kingsrock,
            fail_encore: (ms.flags & failencore_bit) != 0,
            is_fissure_or_horndrill: matches!(mkey, "fissure" | "horndrill"),
            is_fissure: mkey == "fissure",
            is_whirlwind: mkey == "whirlwind",
            ignore_immunity: ms.ignore_immunity,
        });
    }

    debug_assert!(dex.items.len() < 128);
    let mut item_specs = Vec::with_capacity(dex.items.len());
    for i_raw in 0..dex.items.len() {
        let iid = ItemId(i_raw as u16);
        let siid = Some(iid);
        let ikey = dex.items.key(iid);
        let boost_type = COMPACT_TYPE_BOOST_ITEMS
            .iter()
            .find(|&&(k, _)| k == ikey)
            .and_then(|&(_, t_name)| dex.type_id(t_name))
            .map(|t| t.0)
            .unwrap_or(NO_TYPE_ID);
        // Castle #34 (`Probe 11D`): Dragon Scale (ID 152) boosts Dragon 1.1x; Dragon Fang (ID 145) is inert!
        let castle_boost_type = if ikey == "dragonscale" {
            dragon_type
        } else if ikey == "dragonfang" {
            NO_TYPE_ID
        } else {
            boost_type
        };
        let stat_kind = if siid == ki.thickclub {
            ItemStatKind::ThickClub
        } else if siid == ki.lightball {
            ItemStatKind::LightBall
        } else if siid == ki.metalpowder {
            ItemStatKind::MetalPowder
        } else {
            ItemStatKind::None
        };
        item_specs.push(CompactItemSpec {
            boost_type,
            castle_boost_type,
            stat_kind,
            is_quick_claw: siid == ki.quickclaw,
            is_leftovers: siid == ki.leftovers,
            is_brightpowder: siid == ki.brightpowder,
            is_mintberry: siid == ki.mintberry,
            is_miracleberry: siid == ki.miracleberry,
            is_goldberry: siid == ki.goldberry,
            is_berry_juice: siid == ki.berryjuice,
            is_berry: siid == ki.berry,
            is_scope_lens: siid == ki.scopelens,
            is_focus_band: ikey == "focusband",
            is_kings_rock: ikey == "kingsrock",
            is_mystery_berry: siid == ki.mysteryberry || ikey == "mysteryberry",
            is_ice_berry: ikey == "iceberry",
            is_bitter_berry: ikey == "bitterberry",
            is_przcure_berry: ikey == "przcureberry",
            is_psncure_berry: ikey == "psncureberry",
            is_burnt_berry: ikey == "burntberry",
            is_stick: ikey == "stick",
            is_lucky_punch: ikey == "luckypunch",
            is_berserk_gene: siid == ki.berserkgene || ikey == "berserkgene",
        });
    }

    CompactDexTables {
        type_chart,
        special_type_mask,
        normal_type,
        fighting_type,
        flying_type,
        poison_type,
        ground_type,
        rock_type,
        ghost_type,
        steel_type,
        fire_type,
        water_type,
        grass_type,
        electric_type,
        ice_type,
        dragon_type,
        sandstorm_immune_mask,
        poison_immune_mask,
        move_specs,
        item_specs,
        sub_cond: dex.conds_id("substitute"),
        conf_cond: dex.conds_id("confusion"),
        flinch_cond: dex.conds_id("flinch"),
        dbond_cond: dex.conds_id("destinybond"),
        par_cond: dex.conds_id("parspeeddrop"),
        brn_cond: dex.conds_id("brnattackdrop"),
        seed_cond: dex.conds_id("leechseed"),
        night_cond: dex.conds_id("nightmare"),
        curse_cond: dex.conds_id("curse"),
        fore_cond: dex.conds_id("foresight"),
        prot_cond: dex.conds_id("protect"),
        end_cond: dex.conds_id("endure"),
        stall_cond: dex.conds_id("stall"),
        enc_cond: dex.conds_id("encore"),
        dis_cond: dex.conds_id("disable"),
        perish_cond: dex.conds_id("perishsong"),
        res_cond: dex.conds_id("residualdmg"),
        recharge_cond: dex.conds_id("mustrecharge"),
        trapped_cond: dex.conds_id("trapped"),
        partiallytrapped_cond: dex.conds_id("partiallytrapped"),
        attract_cond: dex.conds_id("attract"),
        mist_cond: dex.conds_id("mist"),
        focusenergy_cond: dex.conds_id("focusenergy"),
        lockon_cond: dex.conds_id("lockon"),
        defensecurl_cond: dex.conds_id("defensecurl"),
        minimize_cond: dex.conds_id("minimize"),
        rollout_cond: dex.conds_id("rollout"),
        lockedmove_cond: dex.conds_id("lockedmove"),
        furycutter_cond: dex.conds_id("furycutter"),
        rage_cond: dex.conds_id("rage"),
        dig_cond: dex.conds_id("dig"),
        fly_cond: dex.conds_id("fly"),
        twoturnmove_cond: dex.conds_id("twoturnmove"),
        futuremove_cond: dex.conds_id("futuremove"),
        refl_cond: dex.conds_id("reflect"),
        ls_cond: dex.conds_id("lightscreen"),
        safe_cond: dex.conds_id("safeguard"),
        spikes_cond: dex.conds_id("spikes"),
        rain_cond: dex.conds_id("raindance"),
        sun_cond: dex.conds_id("sunnyday"),
        sand_cond: dex.conds_id("sandstorm"),
        sleep_clause_cond: dex.conds_id("stadiumsleepclause"),
        freeze_clause_cond: dex.conds_id("freezeclausemod"),
    }
}

static DEFAULT_TYPE_CHART: [[i8; 20]; 20] = [[0i8; 20]; 20];

/// Static match context shared across all `CompactBattle` nodes in a search tree.
#[derive(Clone, Debug)]
pub struct MatchTable {
    /// 6 picked Pokemon specs (`0..3` = Side 0, `3..6` = Side 1).
    pub mons: [StaticMonSpec; 6],
    /// Precompiled move specs for each of the 6 Pokemon's 4 move slots.
    pub move_specs: [[CompactMoveSpec; 4]; 6],
    /// Reference to the process-wide 20 × 20 type effectiveness matrix (`-1` = resist, `0` = neutral, `+1` = super, `-9` = immune).
    pub type_chart: &'static [[i8; 20]; 20],
    /// Reference to the process-wide precompiled per-item metadata slice (indexed by `compact_item - 1`).
    pub item_specs: &'static [CompactItemSpec],
    pub normal_type: u8,
    pub fighting_type: u8,
    pub flying_type: u8,
    pub poison_type: u8,
    pub ground_type: u8,
    pub ghost_type: u8,
    pub grass_type: u8,
    pub fire_type: u8,
    pub electric_type: u8,
    pub ice_type: u8,
    pub sandstorm_immune_mask: u32,
    pub poison_immune_mask: u32,
    pub match_fingerprint: u64,
    pub ruleset: crate::rules::Ruleset,
    /// True if any of the 6 picked Pokemon starts holding Quick Claw (cannot be created out of thin air).
    pub any_quick_claw: bool,
}

impl Default for MatchTable {
    fn default() -> Self {
        Self {
            mons: [StaticMonSpec::default(); 6],
            move_specs: [[CompactMoveSpec::default(); 4]; 6],
            type_chart: &DEFAULT_TYPE_CHART,
            item_specs: &[],
            normal_type: 0,
            fighting_type: 0,
            flying_type: 0,
            poison_type: 0,
            ground_type: 0,
            ghost_type: 0,
            grass_type: 0,
            fire_type: 0,
            electric_type: 0,
            ice_type: 0,
            sandstorm_immune_mask: 0,
            poison_immune_mask: 0,
            match_fingerprint: 0,
            ruleset: crate::rules::Ruleset::PsStadium2,
            any_quick_claw: false,
        }
    }
}

impl MatchTable {
    /// O(1) dynamic item metadata lookup by `CompactMon::item` (`0` = None, `1..=N` = `ItemId.0 + 1`).
    #[inline(always)]
    pub fn item_spec(&self, compact_item: u8) -> &'static CompactItemSpec {
        if compact_item == 0 {
            return &DEFAULT_ITEM_SPEC;
        }
        self.item_specs
            .get((compact_item - 1) as usize)
            .unwrap_or(&DEFAULT_ITEM_SPEC)
    }

    /// Compute the net type effectiveness (`-2..=+2`) and immunity flag against a target,
    /// accounting for Foresight (`vol_bit::FORESIGHT`) removing Ghost's immunity to Normal/Fighting.
    #[inline(always)]
    pub const fn effectiveness_ext(
        &self,
        move_type: u8,
        def_mon: &StaticMonSpec,
        foresight: bool,
    ) -> (i8, bool) {
        if move_type == NO_TYPE_ID {
            return (0, false);
        }
        let mt = if (move_type as usize) < 20 {
            move_type as usize
        } else {
            19
        };
        let t0 = if (def_mon.types[0] as usize) < 20 {
            def_mon.types[0] as usize
        } else {
            19
        };
        let mut e0 = self.type_chart[mt][t0];
        if e0 == -9 {
            if foresight
                && (move_type == self.normal_type || move_type == self.fighting_type)
                && def_mon.types[0] == self.ghost_type
            {
                e0 = 0;
            } else {
                return (0, true);
            }
        }
        if def_mon.num_types > 1 && def_mon.types[1] != def_mon.types[0] {
            let t1 = if (def_mon.types[1] as usize) < 20 {
                def_mon.types[1] as usize
            } else {
                19
            };
            let mut e1 = self.type_chart[mt][t1];
            if e1 == -9 {
                if foresight
                    && (move_type == self.normal_type || move_type == self.fighting_type)
                    && def_mon.types[1] == self.ghost_type
                {
                    e1 = 0;
                } else {
                    return (0, true);
                }
            }
            (e0 + e1, false)
        } else {
            (e0, false)
        }
    }
}

const COMPACT_TYPE_BOOST_ITEMS: [(&str, &str); 18] = [
    ("blackbelt", "Fighting"),
    ("blackglasses", "Dark"),
    ("charcoal", "Fire"),
    ("dragonfang", "Dragon"),
    ("hardstone", "Rock"),
    ("magnet", "Electric"),
    ("metalcoat", "Steel"),
    ("miracleseed", "Grass"),
    ("mysticwater", "Water"),
    ("nevermeltice", "Ice"),
    ("pinkbow", "Normal"),
    ("poisonbarb", "Poison"),
    ("polkadotbow", "Normal"),
    ("sharpbeak", "Flying"),
    ("silverpowder", "Bug"),
    ("softsand", "Ground"),
    ("spelltag", "Ghost"),
    ("twistedspoon", "Psychic"),
];

#[derive(Clone, Copy, Debug)]
pub struct CachedRosterMon {
    pub spec: StaticMonSpec,
    pub move_specs: [CompactMoveSpec; 4],
    guard_maxhp: i32,
    guard_stored_stats: [i32; 5],
    guard_types: crate::dex::TypeList,
    guard_item: Option<ItemId>,
    guard_hp_type: crate::dex::TypeId,
    guard_hp_power: i32,
    guard_happiness: u8,
    guard_maxpp: [i32; 4],
}

#[derive(Clone, Debug, Default)]
pub struct RosterCompactCache {
    pub entries: [[Option<CachedRosterMon>; 6]; 2],
}

impl RosterCompactCache {
    fn build_side_entries(
        side: &crate::state::Side,
        dex: &Dex,
        dt: &CompactDexTables,
    ) -> [Option<CachedRosterMon>; 6] {
        let mut out = [None; 6];
        for (slot, p) in side.roster.iter().take(6).enumerate() {
            if p.transformed {
                continue;
            }
            let (spec, move_specs) = build_static_mon_spec(p, dex, dt);
            let mut guard_maxpp = [0i32; 4];
            for m_i in 0..(spec.num_moves as usize) {
                guard_maxpp[m_i] = p.move_slots[m_i].maxpp;
            }
            out[slot] = Some(CachedRosterMon {
                spec,
                move_specs,
                guard_maxhp: p.maxhp,
                guard_stored_stats: p.stored_stats,
                guard_types: p.types,
                guard_item: p.item,
                guard_hp_type: p.hp_type,
                guard_hp_power: p.hp_power,
                guard_happiness: p.happiness,
                guard_maxpp,
            });
        }
        out
    }

    pub fn from_battle(battle: &Battle, dex: &Dex) -> Self {
        let dt = dex_tables(dex);
        let entries = [
            Self::build_side_entries(&battle.sides[0], dex, dt),
            Self::build_side_entries(&battle.sides[1], dex, dt),
        ];
        Self { entries }
    }

    pub fn from_battle_with_side_template(
        battle: &Battle,
        dex: &Dex,
        template_side: usize,
        template: &Self,
    ) -> Self {
        if template_side >= 2 {
            return Self::from_battle(battle, dex);
        }
        let dt = dex_tables(dex);
        let mut entries = [[None; 6]; 2];
        entries[template_side] = template.entries[template_side];
        let other = 1 - template_side;
        entries[other] = Self::build_side_entries(&battle.sides[other], dex, dt);
        Self { entries }
    }
}

#[inline]
fn build_static_mon_spec(
    p: &crate::state::Pokemon,
    dex: &Dex,
    dt: &CompactDexTables,
) -> (StaticMonSpec, [CompactMoveSpec; 4]) {
    let maxhp = p.maxhp.max(1) as i16;
    let mut type_mask = 0u32;
    for ty in p.types.iter() {
        type_mask |= 1u32 << (ty.0 & 31);
    }
    let item_spec = p
        .item
        .and_then(|i| dt.item_specs.get(i.0 as usize).copied())
        .unwrap_or(CompactItemSpec {
            boost_type: NO_TYPE_ID,
            ..CompactItemSpec::default()
        });

    let return_bp = (p.happiness as i32 * 10 / 25) as i16;
    let mut moves = [MoveId(0); 4];
    let mut max_pp = [0u8; 4];
    let mut move_specs = [CompactMoveSpec::default(); 4];
    let mut role_flags = 0u8;
    let num_moves = p.move_slots.len().min(4) as u8;
    for m_i in 0..num_moves as usize {
        let slot = p.move_slots[m_i];
        moves[m_i] = slot.id;
        max_pp[m_i] = slot.maxpp.clamp(0, 64) as u8;

        let mut cm = dt.move_specs[slot.id.0 as usize];
        match cm.var_bp_kind {
            VarBpKind::Return => cm.base_power = return_bp,
            VarBpKind::Frustration => cm.base_power = ((255 - p.happiness as i32) * 10 / 25) as i16,
            VarBpKind::HiddenPower => {
                let hpt = p.hp_type.0;
                cm.move_type = hpt;
                cm.base_power = p.hp_power as i16;
                cm.is_physical = (dt.special_type_mask & (1u32 << (hpt & 31))) == 0;
                cm.is_water = hpt == dt.water_type;
                cm.is_fire = hpt == dt.fire_type;
            }
            _ => {}
        }
        if cm.is_half_heal || cm.special_kind == SpecialMoveKind::WeatherHeal {
            role_flags |= role_bit::HALF_HEAL;
        }
        if cm.is_rest {
            role_flags |= role_bit::REST;
        }
        if cm.special_kind == SpecialMoveKind::SleepTalk || cm.is_sleep_usable {
            role_flags |= role_bit::SLEEP_USABLE;
        }
        if matches!(
            cm.special_kind,
            SpecialMoveKind::Phaze | SpecialMoveKind::PerishSong | SpecialMoveKind::MeanLook
        ) {
            role_flags |= role_bit::PERISH_PHAZE_TRAP;
        }
        if cm.special_kind == SpecialMoveKind::BatonPass {
            role_flags |= role_bit::BATON_PASS;
        }
        if cm.is_protect {
            role_flags |= role_bit::PROTECT;
        }
        if !cm.boosts_self
            && cm.boosts[0].0 == stat_idx::ATK as u8
            && cm.boosts[0].1 > 0
            && (cm.primary_vol & vol_bit::CONFUSION) != 0
        {
            role_flags |= role_bit::SWAGGER;
        }
        if matches!(
            cm.special_kind,
            SpecialMoveKind::Phaze | SpecialMoveKind::Haze
        ) {
            role_flags |= role_bit::PHAZE_OR_HAZE;
        }
        move_specs[m_i] = cm;
    }

    let stored_stats = [
        p.stored_stats[stat_idx::ATK] as i16,
        p.stored_stats[stat_idx::DEF] as i16,
        p.stored_stats[stat_idx::SPA] as i16,
        p.stored_stats[stat_idx::SPD] as i16,
        p.stored_stats[stat_idx::SPE] as i16,
    ];
    let item_stat_kind = ItemStatKind::classify(dex, p.species, p.item);
    let species_stat_kind = {
        let ks = &dex.known_species;
        let sp = Some(p.species);
        if sp == ks.cubone || sp == ks.marowak {
            ItemStatKind::ThickClub
        } else if sp == ks.pikachu {
            ItemStatKind::LightBall
        } else if sp == ks.ditto {
            ItemStatKind::MetalPowder
        } else {
            ItemStatKind::None
        }
    };
    let mut static_fp = 0xcbf29ce484222325u64;
    let w0 = (p.species.0 as u64)
        | ((p.level as u64) << 16)
        | ((maxhp as u16 as u64) << 24)
        | ((type_mask as u64) << 40)
        | ((item_stat_kind as u64) << 60)
        | ((item_spec.boost_type as u64) << 52);
    let w1 = (stored_stats[0] as u16 as u64)
        | ((stored_stats[1] as u16 as u64) << 12)
        | ((stored_stats[2] as u16 as u64) << 24)
        | ((stored_stats[3] as u16 as u64) << 36)
        | ((stored_stats[4] as u16 as u64) << 48)
        | ((p.hp_type.0 as u64) << 60);
    let w2 = (moves[0].0 as u64)
        | ((moves[1].0 as u64) << 16)
        | ((moves[2].0 as u64) << 32)
        | ((moves[3].0 as u64) << 48);
    let w3 = (move_specs[0].base_power as u16 as u64)
        | ((move_specs[1].base_power as u16 as u64) << 16)
        | ((move_specs[2].base_power as u16 as u64) << 32)
        | ((move_specs[3].base_power as u16 as u64) << 48);
    for w in [w0, w1, w2, w3] {
        static_fp = static_fp.wrapping_add(w).wrapping_mul(0xf1357aea2e62a9c5);
    }

    let spec = StaticMonSpec {
        static_fp,
        type_mask,
        species: p.species,
        maxhp,
        quantum_1_8: (maxhp / 8).max(1),
        quantum_1_16: (maxhp / 16).max(1),
        stored_stats,
        moves,
        max_pp,
        types: [p.types.t[0].0, p.types.t[1].0],
        num_types: p.types.n,
        level: p.level,
        num_moves,
        role_flags,
        species_stat_kind,
        base_atk: dex.species.get(p.species).base_stats.atk.clamp(1, 255) as u8,
        base_def: dex.species.get(p.species).base_stats.def.clamp(1, 255) as u8,
        present_type_num: present_type_index(
            dex.type_name(p.types.t[(p.types.n.max(1) - 1) as usize]),
        ),
        species_flags: {
            let ks = &dex.known_species;
            let sp = Some(p.species);
            let mut f = 0u8;
            if sp == ks.farfetchd {
                f |= 1 << 0;
            }
            if sp == ks.chansey {
                f |= 1 << 1;
            }
            if Some(p.base_species) == ks.ditto {
                f |= 1 << 2;
            }
            f
        },
    };
    (spec, move_specs)
}

#[inline]
const fn present_type_index(name: &str) -> u8 {
    // Match Gen 2 / PBS v0.3 type numbers (Normal/Fighting -> 1, ..., Dark -> 27).
    let b = name.as_bytes();
    if b.len() < 3 {
        return 1;
    }
    match (b[0], b[1], b[2]) {
        (b'F', b'i', b'g') => 1,  // Fighting
        (b'F', b'l', b'y') => 2,  // Flying
        (b'P', b'o', b'i') => 3,  // Poison
        (b'G', b'r', b'o') => 4,  // Ground
        (b'R', b'o', b'c') => 5,  // Rock
        (b'B', b'u', b'g') => 7,  // Bug
        (b'G', b'h', b'o') => 8,  // Ghost
        (b'S', b't', b'e') => 9,  // Steel
        (b'F', b'i', b'r') => 20, // Fire
        (b'W', b'a', b't') => 21, // Water
        (b'G', b'r', b'a') => 22, // Grass
        (b'E', b'l', b'e') => 23, // Electric
        (b'P', b's', b'y') => 24, // Psychic
        (b'I', b'c', b'e') => 25, // Ice
        (b'D', b'r', b'a') => 26, // Dragon
        (b'D', b'a', b'r') => 27, // Dark
        _ => 1,                   // Normal (0 -> 1) and unknown
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SwitchKind {
    Manual,
    BatonPass,
    FaintReplace,
    Drag,
}

impl CompactBattle {
    /// Extract a `(MatchTable, CompactBattle)` pair from an existing `Battle`.
    #[inline]
    pub fn from_battle(battle: &Battle, dex: &Dex) -> (MatchTable, CompactBattle) {
        Self::from_battle_with_roster_cache(battle, dex, None)
    }

    /// Extract a `(MatchTable, CompactBattle)` pair using an optional per-world
    /// [`RosterCompactCache`] to bypass static species/stat/item/move compilation.
    #[inline]
    pub fn from_battle_with_roster_cache(
        battle: &Battle,
        dex: &Dex,
        roster_cache: Option<&RosterCompactCache>,
    ) -> (MatchTable, CompactBattle) {
        let dt = dex_tables(dex);

        let mut static_mons = [StaticMonSpec::default(); 6];
        let mut move_specs = [[CompactMoveSpec::default(); 4]; 6];
        let mut compact_mons = [CompactMon::default(); 6];
        let mut compact_sides = [CompactSide::default(); 2];
        let mut future_sight_dmg = [0u16; 2];
        let mut future_sight_turns = [0u8; 2];
        let mut future_sight_src = [0u8; 2];
        let mut future_sight_raw: [Option<(Option<i64>, Option<i32>)>; 2] = [None; 2];
        let mut foresight_casters: [Option<crate::state::PokeId>; 2] = [None; 2];
        let mut locked_hit1_receivers: [Option<(u8, u8)>; 2] = [None; 2];
        let mut bp_switch_flags = 0u8;
        let mut turn_extra_flags = 0u8;

        for side_n in 0..2usize {
            let side = &battle.sides[side_n];
            let party_len = side.party.len().min(3);
            let mut alive_mask = 0u8;
            let mut active_idx = if side.active.is_none() { 3u8 } else { 0u8 };
            let mut party_roster = [0u8; 3];

            for p_idx in 0..3usize {
                let c_idx = side_n * 3 + p_idx;
                if p_idx >= party_len {
                    party_roster[p_idx] = (p_idx as u8).min(side.roster.len().saturating_sub(1) as u8);
                    compact_mons[c_idx].mark_fainted();
                    continue;
                }
                let roster_slot = side.party[p_idx];
                party_roster[p_idx] = roster_slot;
                if side.active == Some(roster_slot) {
                    active_idx = p_idx as u8;
                }
                let p = &side.roster[roster_slot as usize];
                if !p.fainted && p.hp > 0 {
                    alive_mask |= 1 << p_idx;
                }

                let num_moves = p.move_slots.len().min(4) as u8;
                let cached_hit = roster_cache
                    .and_then(|rc| rc.entries[side_n].get(roster_slot as usize))
                    .and_then(|opt| opt.as_ref())
                    .filter(|cm| {
                        !p.transformed
                            && p.species == cm.spec.species
                            && p.level == cm.spec.level
                            && p.maxhp == cm.guard_maxhp
                            && p.stored_stats == cm.guard_stored_stats
                            && p.types == cm.guard_types
                            && p.item == cm.guard_item
                            && p.hp_type == cm.guard_hp_type
                            && p.hp_power == cm.guard_hp_power
                            && p.happiness == cm.guard_happiness
                            && num_moves == cm.spec.num_moves
                            && (0..num_moves as usize).all(|m_i| {
                                let sl = p.move_slots[m_i];
                                sl.id == cm.spec.moves[m_i] && sl.maxpp == cm.guard_maxpp[m_i]
                            })
                    });
                if let Some(cm) = cached_hit {
                    static_mons[c_idx] = cm.spec;
                    move_specs[c_idx] = cm.move_specs;
                } else {
                    let (spec, ms) = build_static_mon_spec(p, dex, dt);
                    static_mons[c_idx] = spec;
                    move_specs[c_idx] = ms;
                }
                let moves = static_mons[c_idx].moves;

                let gender_byte = match p.gender {
                    Gender::M => 1,
                    Gender::F => 2,
                    Gender::N => 0,
                };
                let mut c_mon = CompactMon {
                    hp_pool: [p.hp.clamp(0, i16::MAX as i32) as i16, 0],
                    pp: [0u8; 4],
                    status_packed: 0,
                    item: p.item.map(|i| (i.0 & 0x7F) as u8 + 1).unwrap_or(0),
                    vol_flags: 0,
                    timers: [0u8; 4],
                    boosts: p.boosts,
                    slot_refs: 0,
                    ext_flags: if p.transformed { ext_vol_bit::TRANSFORMED } else { 0 },
                    ext_timers: [0u8; 4],
                    gender: gender_byte,
                    rollout_rage: 0,
                };
                for m_i in 0..num_moves as usize {
                    c_mon.pp[m_i] = p.move_slots[m_i].pp.clamp(0, 64) as u8;
                }

                if p.fainted || p.hp <= 0 {
                    c_mon.mark_fainted();
                    compact_mons[c_idx] = c_mon;
                    continue;
                }

                let mut sub_hp = 0i16;
                let mut perish_turns = 0u8;
                let mut stall_cnt = 0u8;
                let mut res_counter_packed = 0u8;
                if let Some(last_m) = p.last_move_encore.or(p.last_move) {
                    if let Some(m_i) = (0..num_moves as usize).find(|&i| moves[i] == last_m) {
                        c_mon.set_last_move_slot(m_i as u8);
                    }
                }

                let mut has_trapped = false;
                let mut has_partial_trap = false;
                let mut partial_trap_dur = 0u8;
                let mut ttm_move: Option<MoveId> = None;
                if !p.volatiles.is_empty() {
                    for &(cid, ref vst) in &p.volatiles {
                        let scid = Some(cid);
                        if scid == dt.sub_cond {
                            sub_hp = vst.get_int(DK::Hp).max(0) as i16;
                            if sub_hp > 0 {
                                c_mon.vol_flags |= vol_bit::SUBSTITUTE;
                            }
                        } else if scid == dt.conf_cond {
                            c_mon.vol_flags |= vol_bit::CONFUSION;
                            c_mon.timers[timer_idx::CONFUSION] =
                                vst.get_int(DK::Time).clamp(0, 255) as u8;
                        } else if scid == dt.flinch_cond {
                            c_mon.vol_flags |= vol_bit::FLINCH;
                        } else if scid == dt.dbond_cond {
                            c_mon.vol_flags |= vol_bit::DESTINY_BOND;
                        } else if scid == dt.par_cond {
                            c_mon.vol_flags |= vol_bit::PAR_SPEED_DROP;
                        } else if scid == dt.brn_cond {
                            c_mon.vol_flags |= vol_bit::BRN_ATK_DROP;
                        } else if scid == dt.seed_cond {
                            c_mon.vol_flags |= vol_bit::LEECH_SEED;
                        } else if scid == dt.night_cond {
                            c_mon.vol_flags |= vol_bit::NIGHTMARE;
                            if p.status != Status::Slp && battle.ruleset.is_castle() {
                                c_mon.ext_flags |= ext_vol_bit::MIRACLE_NIGHTMARE;
                            }
                        } else if scid == dt.curse_cond {
                            c_mon.vol_flags |= vol_bit::CURSE;
                        } else if scid == dt.fore_cond {
                            c_mon.vol_flags |= vol_bit::FORESIGHT;
                            if vst.source.is_some() {
                                foresight_casters[side_n] = vst.source;
                            }
                        } else if scid == dt.prot_cond {
                            c_mon.vol_flags |= vol_bit::PROTECT;
                        } else if scid == dt.end_cond {
                            c_mon.vol_flags |= vol_bit::ENDURE;
                        } else if scid == dt.stall_cond {
                            if battle.ruleset.is_castle() {
                                stall_cnt = vst.get_int(DK::Counter).clamp(1, 8) as u8;
                            } else {
                                let c = vst
                                    .get(DK::Counter)
                                    .map(|v| v.as_f64().floor().max(0.0) as u32)
                                    .unwrap_or(127);
                                stall_cnt = if c >= 127 {
                                    1
                                } else if c >= 63 {
                                    2
                                } else if c >= 31 {
                                    3
                                } else if c >= 15 {
                                    4
                                } else if c >= 7 {
                                    5
                                } else if c >= 3 {
                                    6
                                } else if c >= 1 {
                                    7
                                } else {
                                    8
                                };
                            }
                        } else if scid == dt.enc_cond {
                            c_mon.vol_flags |= vol_bit::ENCORE;
                            c_mon.timers[timer_idx::ENCORE] =
                                vst.duration.unwrap_or(0).clamp(0, 255) as u8;
                            if let Some(enc_mid) = vst.get_move() {
                                if let Some(m_i) =
                                    (0..num_moves as usize).find(|&i| moves[i] == enc_mid)
                                {
                                    c_mon.set_encore_slot(m_i as u8);
                                }
                            }
                        } else if scid == dt.dis_cond {
                            if let Some(dis_mid) = vst.get_move() {
                                if let Some(m_i) =
                                    (0..num_moves as usize).find(|&i| moves[i] == dis_mid)
                                {
                                    c_mon.vol_flags |= vol_bit::DISABLE;
                                    c_mon.timers[timer_idx::DISABLE] =
                                        vst.duration.unwrap_or(0).clamp(0, 255) as u8;
                                    c_mon.set_disable_slot(m_i as u8);
                                }
                            }
                        } else if scid == dt.perish_cond {
                            c_mon.vol_flags |= vol_bit::PERISH_SONG;
                            perish_turns = vst.duration.unwrap_or(0).clamp(0, 7) as u8;
                            if let Some(src) = vst.source {
                                turn_extra_flags |= 1u8 << (2 + (src.side & 1));
                            }
                        } else if scid == dt.res_cond {
                            res_counter_packed =
                                (vst.get_int(DK::Counter).clamp(0, 15) as u8) | 0x10;
                        } else if scid == dt.recharge_cond {
                            c_mon.set_must_recharge(true);
                        } else if scid == dt.trapped_cond {
                            has_trapped = true;
                        } else if scid == dt.partiallytrapped_cond {
                            let trapper_active = match vst.source {
                                Some(src) => {
                                    battle.active_id(side_n ^ 1) == Some(src)
                                        && battle.poke(src).is_active
                                        && !battle.poke(src).fainted
                                        && battle.poke(src).hp > 0
                                }
                                None => p.trapped,
                            };
                            if trapper_active {
                                has_partial_trap = true;
                                partial_trap_dur = vst.duration.unwrap_or(2).clamp(1, 5) as u8;
                            }
                        } else if scid == dt.attract_cond {
                            c_mon.ext_flags |= ext_vol_bit::ATTRACT;
                        } else if scid == dt.mist_cond {
                            c_mon.ext_flags |= ext_vol_bit::MIST;
                        } else if scid == dt.focusenergy_cond {
                            c_mon.ext_flags |= ext_vol_bit::FOCUS_ENERGY;
                        } else if scid == dt.lockon_cond {
                            c_mon.ext_flags |= ext_vol_bit::LOCK_ON;
                            c_mon.set_lock_on_turns(vst.duration.unwrap_or(1).clamp(1, 2) as u8);
                        } else if scid == dt.defensecurl_cond {
                            c_mon.ext_flags |= ext_vol_bit::DEFENSE_CURL;
                        } else if scid == dt.minimize_cond {
                            c_mon.ext_flags |= ext_vol_bit::MINIMIZE;
                        } else if scid == dt.rollout_cond {
                            c_mon.ext_flags |= ext_vol_bit::LOCKED_ROLLOUT;
                            c_mon.set_rollout_count(vst.get_int(DK::HitCount).clamp(1, 5) as u8);
                            if let Some(m_i) = (0..num_moves as usize).find(|&i| {
                                move_specs[c_idx][i].special_kind == SpecialMoveKind::Rollout
                            }) {
                                c_mon.set_locked_move_slot(m_i as u8);
                            }
                            if let Some(crate::state::Scalar::Slot(ts, tslot)) =
                                vst.get(DK::TargetSlot)
                            {
                                if (ts as usize) < 2 {
                                    locked_hit1_receivers[side_n] = Some((ts, tslot));
                                }
                            }
                        } else if scid == dt.lockedmove_cond {
                            c_mon.ext_flags |= ext_vol_bit::LOCKED_OUTRAGE;
                            c_mon.ext_timers[ext_timer_idx::LOCKED_TURNS] =
                                vst.duration.unwrap_or(1).clamp(1, 3) as u8;
                            if let Some(lm_mid) = vst.get_move() {
                                if let Some(m_i) =
                                    (0..num_moves as usize).find(|&i| moves[i] == lm_mid)
                                {
                                    c_mon.set_locked_move_slot(m_i as u8);
                                }
                            } else if let Some(m_i) = (0..num_moves as usize).find(|&i| {
                                move_specs[c_idx][i].special_kind == SpecialMoveKind::OutrageLock
                            }) {
                                c_mon.set_locked_move_slot(m_i as u8);
                            }
                            if let Some(crate::state::Scalar::Slot(ts, tslot)) =
                                vst.get(DK::TargetSlot)
                            {
                                if (ts as usize) < 2 {
                                    locked_hit1_receivers[side_n] = Some((ts, tslot));
                                }
                            }
                        } else if scid == dt.furycutter_cond {
                            let mult = vst.get_int(DK::Multiplier).max(1) as u32;
                            c_mon.ext_timers[ext_timer_idx::FURY_CUTTER] =
                                (mult.trailing_zeros().min(12) + 1) as u8;
                        } else if scid == dt.rage_cond {
                            c_mon.ext_flags |= ext_vol_bit::RAGE;
                            c_mon.ext_timers[ext_timer_idx::RAGE_COUNTER] =
                                vst.get_int(DK::Counter).clamp(0, 255) as u8;
                        } else if scid == dt.dig_cond {
                            c_mon.ext_flags |= ext_vol_bit::DIGGING;
                        } else if scid == dt.fly_cond {
                            c_mon.ext_flags |= ext_vol_bit::FLYING;
                        } else if scid == dt.twoturnmove_cond {
                            ttm_move = vst.get_move();
                        }
                    }
                }
                if let Some(ttm_mid) = ttm_move {
                    if let Some(m_i) = (0..num_moves as usize).find(|&i| moves[i] == ttm_mid) {
                        let charge_active = match move_specs[c_idx][m_i].special_kind {
                            SpecialMoveKind::Dig => (c_mon.ext_flags & ext_vol_bit::DIGGING) != 0,
                            SpecialMoveKind::Fly => (c_mon.ext_flags & ext_vol_bit::FLYING) != 0,
                            _ => false,
                        };
                        if charge_active {
                            c_mon.set_two_turn_locked(true);
                            c_mon.set_locked_move_slot(m_i as u8);
                        }
                    }
                }
                if has_trapped {
                    c_mon.vol_flags |= vol_bit::TRAPPED_OR_LOCKED;
                }
                if has_partial_trap {
                    c_mon.set_partially_trapped(true);
                    c_mon.ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] = partial_trap_dur;
                }

                let status_counter = match p.status {
                    Status::Slp => p.status_state.get_int(DK::Time).clamp(0, 31) as u8,
                    Status::Fnt => 0,
                    _ => res_counter_packed,
                };

                c_mon.hp_pool[1] = sub_hp;
                c_mon.set_perish_turns(perish_turns);
                c_mon.set_stall_counter_ext(stall_cnt);
                c_mon.set_status(p.status as u8, status_counter);
                compact_mons[c_idx] = c_mon;
            }
            if side.pokemon_left == 1 && alive_mask.count_ones() > 1 && active_idx < 3 {
                alive_mask = 1 << active_idx;
            }
            if active_idx >= 3 {
                let fainted_roster = side.fainted_this_turn.or(side.fainted_last_turn);
                active_idx = (0..party_len)
                    .find(|&i| {
                        Some(party_roster[i]) == fainted_roster && (alive_mask & (1 << i)) == 0
                    })
                    .or_else(|| (0..3usize).find(|&i| (alive_mask & (1 << i)) == 0))
                    .unwrap_or(0) as u8;
            }

            let mut reflect_turns = 0u8;
            let mut lightscreen_turns = 0u8;
            let mut safeguard_turns = 0u8;
            let mut spikes_layers = 0u8;
            if !side.side_conditions.is_empty() {
                for &(cid, ref vst) in &side.side_conditions {
                    let scid = Some(cid);
                    if scid == dt.refl_cond {
                        reflect_turns = vst.duration.unwrap_or(0).clamp(0, 15) as u8;
                    } else if scid == dt.ls_cond {
                        lightscreen_turns = vst.duration.unwrap_or(0).clamp(0, 15) as u8;
                    } else if scid == dt.safe_cond {
                        safeguard_turns = vst.duration.unwrap_or(0).clamp(0, 15) as u8;
                    } else if scid == dt.spikes_cond {
                        spikes_layers = vst.get_int(DK::Layers).clamp(1, 3) as u8;
                    }
                }
            }
            if !side.slot_conditions.is_empty() {
                for &(cid, ref vst) in &side.slot_conditions {
                    if Some(cid) == dt.futuremove_cond {
                        let ending_opt = vst.get(DK::EndingTurn).map(|v| v.as_int());
                        future_sight_raw[side_n] = Some((ending_opt, vst.duration));
                        future_sight_dmg[side_n] =
                            vst.future_damage.unwrap_or(0).max(0) as u16;
                        if let Some(src_pid) = vst.source {
                            let src_side = &battle.sides[src_pid.side as usize];
                            if let Some(p_i) = src_side
                                .party
                                .iter()
                                .take(3)
                                .position(|&r| r == src_pid.slot)
                            {
                                future_sight_src[side_n] = p_i as u8;
                            }
                        }
                    }
                }
            }
            let mut side_flags = spikes_layers & 0x03;
            let request = match side.request_state() {
                None | Some(RequestKind::Wait) => 0,
                Some(RequestKind::Move) => 1,
                Some(RequestKind::Switch) => 2,
                Some(RequestKind::TeamPreview) => 3,
            };
            if request == 2 && active_idx < 3 && compact_mons[side_n * 3 + active_idx as usize].hp() > 0
            {
                bp_switch_flags |= 1u8 << side_n;
            }
            if request == 0 && active_idx < 3 {
                let act_c_idx = side_n * 3 + active_idx as usize;
                for q_act in &battle.queue {
                    if q_act.pokemon.is_some_and(|pid| pid.side as usize == side_n) {
                        if let crate::state::ActionKind::Move { move_id, .. } = q_act.choice {
                            let num_m = static_mons[act_c_idx].num_moves as usize;
                            if Some(move_id) == dex.known_moves.struggle {
                                side_flags |= 0x10;
                                bp_switch_flags |= 1u8 << (4 + side_n);
                                break;
                            } else if Some(move_id) == dex.known_moves.recharge {
                                side_flags |= 0x10;
                                break;
                            } else if let Some(m_slot) = (0..num_m)
                                .find(|&m_i| static_mons[act_c_idx].moves[m_i] == move_id)
                            {
                                side_flags |= 0x10 | (((m_slot as u8) & 0x03) << 5);
                                break;
                            }
                        }
                    }
                }
            }

            compact_sides[side_n] = CompactSide {
                active_idx,
                party_roster,
                alive_mask,
                reflect_turns,
                lightscreen_turns,
                safeguard_turns,
                side_flags,
                last_move_selfko: side.last_move.map(|m| (m.0 & 0xFF) as u8).unwrap_or(u8::MAX),
                request,
                pad: 0,
            };
        }
        // Post-EOT forced switch (e.g. Sandstorm / Perish Song / Future Sight KO in Residual):
        // mark `RESIDUALS_DONE` (`0x08`) so `step_switch` never double-executes EOT residuals!
        let residuals_already_ran = battle.mid_turn
            && (bp_switch_flags & 0x03) == 0
            && !battle
                .queue
                .iter()
                .any(|a| matches!(a.choice, crate::state::ActionKind::Residual));
        let residuals_done = residuals_already_ran
            && (compact_sides[0].request == 2 || compact_sides[1].request == 2);
        if residuals_done {
            compact_sides[0].side_flags |= 0x08;
        }
        let outcome = match battle.outcome() {
            None => 0,
            Some(crate::battle::Outcome::P1Win) => 1,
            Some(crate::battle::Outcome::P2Win) => 2,
            Some(crate::battle::Outcome::Tie) => 3,
        };
        // Compute Future Sight countdown from `DK::EndingTurn` (rather than un-decremented `vst.duration`),
        // accounting for whether this turn's EOT residuals have already executed. Clear on terminal states
        // (matching `finish_outcome`) so Order-2 Sandstorm KOs before Order-3 Future Sight agree identically.
        if outcome == 0 {
            let fs_eff_turn = battle.turn as i64 + (residuals_already_ran as i64);
            for side_n in 0..2usize {
                if let Some((ending_opt, dur_opt)) = future_sight_raw[side_n] {
                    future_sight_turns[side_n] = if let Some(ending) = ending_opt {
                        (ending - (fs_eff_turn - 1) + 1).clamp(1, 3) as u8
                    } else {
                        dur_opt.unwrap_or(1).clamp(1, 3) as u8
                    };
                }
            }
        }
        // Apply `FORESIGHT_CASTER` (Castle #54③ only) and `LOCKED_FIRST_HIT_RECV` cross-side volatile references.
        if battle.ruleset.is_castle() {
            for caster_opt in foresight_casters {
                if let Some(src) = caster_opt {
                    let s_n = src.side as usize;
                    if s_n < 2 {
                        if let Some(p_i) = (0..3usize)
                            .find(|&i| compact_sides[s_n].party_roster[i] == src.slot)
                        {
                            if compact_sides[s_n].active_idx as usize == p_i
                                && compact_mons[s_n * 3 + p_i].hp() > 0
                            {
                                compact_mons[s_n * 3 + p_i].ext_flags |=
                                    ext_vol_bit::FORESIGHT_CASTER;
                            }
                        }
                    }
                }
            }
        }
        for recv_opt in locked_hit1_receivers {
            if let Some((ts, tslot)) = recv_opt {
                let r_side = ts as usize;
                if r_side < 2 {
                    if let Some(p_i) = (0..3usize)
                        .find(|&i| compact_sides[r_side].party_roster[i] == tslot)
                    {
                        if compact_sides[r_side].active_idx as usize == p_i
                            && compact_mons[r_side * 3 + p_i].hp() > 0
                        {
                            compact_mons[r_side * 3 + p_i].ext_flags |=
                                ext_vol_bit::LOCKED_FIRST_HIT_RECV;
                        }
                    }
                }
            }
        }
        // If side `s` Baton Passed mid-turn and foe still has a queued move (NOT cancelled by an entry faint),
        // mark side `s` as first-mover BP so Castle Pursuit gets 2x BP 80 after Spikes.
        for s in 0..2usize {
            if ((bp_switch_flags & (1u8 << s)) != 0 || battle.bp_switched_this_turn[s])
                && (compact_sides[s ^ 1].side_flags & 0x10) != 0
            {
                bp_switch_flags |= 1u8 << (2 + s);
            }
        }

        let weather_kind = match battle.field.weather {
            w if w.is_some() && w == dt.rain_cond => CompactWeather::Rain,
            w if w.is_some() && w == dt.sun_cond => CompactWeather::Sun,
            w if w.is_some() && w == dt.sand_cond => CompactWeather::Sandstorm,
            _ => CompactWeather::None,
        };
        let weather_turns = battle
            .field
            .weather_state
            .duration
            .unwrap_or(0)
            .clamp(0, 15) as u8;
        let mut clause_flags = 0u8;
        if !battle.field.pseudo_weather.is_empty() {
            for &(cid, _) in &battle.field.pseudo_weather {
                if Some(cid) == dt.sleep_clause_cond {
                    clause_flags |= 1;
                } else if Some(cid) == dt.freeze_clause_cond {
                    clause_flags |= 4;
                }
            }
        }

        // Compute static fingerprint for this match configuration.
        let mut fp = 0xcbf29ce484222325u64;
        for m in &static_mons {
            fp = fp.wrapping_add(m.static_fp).wrapping_mul(0xf1357aea2e62a9c5);
        }
        let rs_tag = battle.ruleset.fingerprint_tag();
        if rs_tag != 0 {
            fp = fp.wrapping_add(rs_tag).wrapping_mul(0xf1357aea2e62a9c5);
        }
        compact_sides[0].pad = (fp >> 16) as u8;
        compact_sides[1].pad = (fp >> 24) as u8;

        let any_quick_claw = compact_mons.iter().any(|m| {
            m.item != 0
                && dt
                    .item_specs
                    .get((m.item - 1) as usize)
                    .is_some_and(|s| s.is_quick_claw)
        });
        let table = MatchTable {
            mons: static_mons,
            move_specs,
            type_chart: &dt.type_chart,
            item_specs: &dt.item_specs,
            normal_type: dt.normal_type,
            fighting_type: dt.fighting_type,
            flying_type: dt.flying_type,
            poison_type: dt.poison_type,
            ground_type: dt.ground_type,
            ghost_type: dt.ghost_type,
            grass_type: dt.grass_type,
            fire_type: dt.fire_type,
            electric_type: dt.electric_type,
            ice_type: dt.ice_type,
            sandstorm_immune_mask: dt.sandstorm_immune_mask,
            poison_immune_mask: dt.poison_immune_mask,
            match_fingerprint: fp,
            ruleset: battle.ruleset,
            any_quick_claw,
        };
        let cb = CompactBattle {
            mons: compact_mons,
            sides: compact_sides,
            turn: battle.turn,
            weather_kind,
            weather_turns,
            quick_claw_roll: (battle.quick_claw_rolls[0] as u8)
                | ((battle.quick_claw_rolls[1] as u8) << 1),
            outcome,
            speed_order: if battle.mid_turn {
                battle.speed_order.first().copied().unwrap_or(0).clamp(0, 1) as u8
            } else {
                0
            },
            clause_flags,
            hurt_damage_this_turn: [0, 0],
            hurt_cat_this_turn: [0, 0],
            match_fp_lo: fp as u16,
            future_sight_dmg,
            future_sight_turns,
            future_sight_src,
            bp_switch_flags,
            triple_kick_mult: battle.triple_kick_mult,
            turn_extra_flags,
            reserved_castle: [0; 5],
            prng_seed: battle.prng.lcg.seed(),
        };
        if cfg!(debug_assertions) && roster_cache.is_some() {
            let (exp_t, exp_cb) = Self::from_battle_with_roster_cache(battle, dex, None);
            debug_assert_eq!(table.match_fingerprint, exp_t.match_fingerprint);
            debug_assert_eq!(cb.search_hash(), exp_cb.search_hash());
            debug_assert_eq!(cb.prng_seed, exp_cb.prng_seed);
        }
        (table, cb)
    }

    /// Current `Battle::side.party` display order of the 3 compact slots `[0..3]` for `side`.
    /// Index 0 is the active slot; indices 1..3 are the bench slots in exact `Battle` swap order.
    #[inline(always)]
    pub const fn party_order(&self, side: usize) -> [u8; 3] {
        let code = self.reserved_castle[side];
        PARTY_PERMS[if code < 6 { code as usize } else { 0 }]
    }

    /// Ultra-fast polynomial hash of the 256-byte `CompactBattle` state
    /// (hashes the first 31 `u64` words = 248 bytes, skipping word 31 `prng_seed`).
    #[inline(always)]
    pub fn search_hash(&self) -> u64 {
        // SAFETY: `CompactBattle` is `#[repr(C, align(64))]` of size 256 bytes (32 u64 words)
        // with zero padding bytes across `CompactMon` (32B), `CompactSide` (12B), and scalar fields.
        let words: &[u64; 32] = unsafe { &*(self as *const Self as *const [u64; 32]) };
        let mut h = (self.match_fp_lo as u64).wrapping_mul(0x9e3779b97f4a7c15);
        // Words 0..31 cover `mons` (24 words), `sides` (3 words), and turn/field/damage/castle (4 words).
        // Word 31 is `prng_seed` (skipped so states differing only in RNG seed merge).
        for &w in &words[..31] {
            h = h.wrapping_add(w).wrapping_mul(0xf1357aea2e62a9c5);
        }
        h.rotate_left(26)
    }

    /// True when both sides are at an idle start-of-turn Move request with no mid-turn flags.
    #[inline(always)]
    pub const fn is_idle_move_request(&self) -> bool {
        self.outcome == 0
            && self.sides[0].request == 1
            && self.sides[1].request == 1
            && (self.sides[0].side_flags & 0x78) == 0
            && (self.sides[1].side_flags & 0x70) == 0
            && self.bp_switch_flags == 0
    }

    /// Whether `quick_claw_roll` can still influence future outcomes from this state
    /// (mirrors `Battle::quick_claw_roll_observable` on picked 3v3 mons).
    #[inline(always)]
    pub fn quick_claw_observable(&self, table: &MatchTable) -> bool {
        if !table.any_quick_claw {
            return false;
        }
        let idle = self.is_idle_move_request();
        for i in 0..6usize {
            let m = &self.mons[i];
            if m.item != 0 && table.item_spec(m.item).is_quick_claw && (!idle || m.hp() > 0) {
                return true;
            }
        }
        false
    }

    /// Transposition-table search key over `CompactBattle` with optional maxhp-relative HP
    /// bucketing (`hp_buckets > 0`, matching `RmConfig::hp_buckets`) and canonical Quick Claw
    /// roll omission (`omit_quick_claw`). Seeded with the full 64-bit `table.match_fingerprint`
    /// so cross-world transposition tables never alias distinct opponent movesets or rulesets.
    #[inline]
    pub fn search_key(&self, table: &MatchTable, hp_buckets: i64, omit_quick_claw: bool) -> u64 {
        const _: () = assert!(std::mem::offset_of!(CompactBattle, quick_claw_roll) == 220);
        const _: () = assert!(std::mem::offset_of!(CompactBattle, hurt_damage_this_turn) == 224);
        const _: () = assert!(std::mem::offset_of!(CompactMon, hp_pool) == 0);

        let words: &[u64; 32] = unsafe { &*(self as *const Self as *const [u64; 32]) };
        let mut h = table.match_fingerprint.wrapping_mul(0x9e3779b97f4a7c15);
        if hp_buckets > 0 {
            for i in 0..6usize {
                let base = i * 4;
                let mon = &self.mons[i];
                let maxhp = table.mons[i].maxhp.max(1) as i64;
                let hp = mon.hp();
                let hp_b = if hp <= 0 {
                    0u16
                } else {
                    ((hp as i64) * hp_buckets / maxhp) as u16
                };
                let sub = mon.sub_hp();
                let sub_b = if sub <= 0 {
                    0u16
                } else {
                    (((sub as i64) * hp_buckets / maxhp) as u16).max(1)
                };
                let w0 = if cfg!(target_endian = "little") {
                    (words[base] & !0xFFFF_FFFFu64) | (hp_b as u64) | ((sub_b as u64) << 16)
                } else {
                    (words[base] & 0xFFFF_FFFFu64)
                        | ((hp_b as u64) << 48)
                        | ((sub_b as u64) << 32)
                };
                h = h.wrapping_add(w0).wrapping_mul(0xf1357aea2e62a9c5);
                h = h.wrapping_add(words[base + 1]).wrapping_mul(0xf1357aea2e62a9c5);
                h = h.wrapping_add(words[base + 2]).wrapping_mul(0xf1357aea2e62a9c5);
                h = h.wrapping_add(words[base + 3]).wrapping_mul(0xf1357aea2e62a9c5);
            }
        } else {
            for &w in &words[..24] {
                h = h.wrapping_add(w).wrapping_mul(0xf1357aea2e62a9c5);
            }
        }
        // Sides (words 24..27)
        h = h.wrapping_add(words[24]).wrapping_mul(0xf1357aea2e62a9c5);
        h = h.wrapping_add(words[25]).wrapping_mul(0xf1357aea2e62a9c5);
        h = h.wrapping_add(words[26]).wrapping_mul(0xf1357aea2e62a9c5);
        // Word 27: turn | weather_kind | weather_turns | quick_claw_roll | outcome | speed_order | clause_flags
        let w27 = if omit_quick_claw {
            if cfg!(target_endian = "little") {
                words[27] & !(0xFFu64 << 32)
            } else {
                words[27] & !(0xFFu64 << 24)
            }
        } else {
            words[27]
        };
        h = h.wrapping_add(w27).wrapping_mul(0xf1357aea2e62a9c5);
        // Word 28: hurt_damage_this_turn[0..2] | hurt_cat_this_turn[0..2] | match_fp_lo
        let w28 = if hp_buckets > 0 && (self.hurt_damage_this_turn[0] != 0 || self.hurt_damage_this_turn[1] != 0) {
            let d0 = (self.hurt_damage_this_turn[0] / 16) as u64;
            let d1 = (self.hurt_damage_this_turn[1] / 16) as u64;
            if cfg!(target_endian = "little") {
                (words[28] & !0xFFFF_FFFFu64) | d0 | (d1 << 16)
            } else {
                (words[28] & 0xFFFF_FFFFu64) | (d0 << 48) | (d1 << 32)
            }
        } else {
            words[28]
        };
        h = h.wrapping_add(w28).wrapping_mul(0xf1357aea2e62a9c5);
        // Words 29..31: future_sight + bp_switch_flags + triple_kick_mult + turn_extra_flags
        h = h.wrapping_add(words[29]).wrapping_mul(0xf1357aea2e62a9c5);
        h = h.wrapping_add(words[30]).wrapping_mul(0xf1357aea2e62a9c5);
        // Word 31 is `prng_seed` (skipped).
        h.rotate_left(26) ^ (h >> 33)
    }

    /// Decode a compact action code (`0..4` = move slot, `4..7` = party slot switch, `7` = pass)
    /// into a `SearchChoice` relative to `party_order` (the 0-based `Battle::sides[side].party`
    /// slice where index 0 is active and indices 1..len are bench switch positions `2..=len`).
    #[inline]
    pub fn decode_action(
        &self,
        table: &MatchTable,
        dex: &Dex,
        side: usize,
        act: u8,
        party_order: &[u8],
    ) -> Option<crate::battle::SearchChoice> {
        use crate::battle::SearchChoice;
        if act >= 7 || self.sides[side].request == 0 || self.outcome != 0 {
            return None;
        }
        if act >= 4 {
            let slot = (act - 4) as usize;
            if slot >= 3 {
                return None;
            }
            let roster_slot = self.sides[side].party_roster[slot];
            let pos = party_order.iter().position(|&r| r == roster_slot)?;
            return Some(SearchChoice::Switch(pos as u8 + 1));
        }
        let act_idx = side * 3 + self.sides[side].active_idx as usize;
        let mon = &self.mons[act_idx];
        if mon.must_recharge() {
            return dex.known_moves.recharge.map(SearchChoice::Move);
        }
        if mon.is_multi_turn_locked() {
            let locked = mon.locked_move_slot() as usize;
            return (locked < table.mons[act_idx].num_moves as usize)
                .then(|| SearchChoice::Move(table.mons[act_idx].moves[locked]));
        }
        if self.is_forced_struggle(table, side) {
            return dex.known_moves.struggle.map(SearchChoice::Move);
        }
        let m_slot = act as usize;
        if m_slot < table.mons[act_idx].num_moves as usize {
            Some(SearchChoice::Move(table.mons[act_idx].moves[m_slot]))
        } else {
            None
        }
    }

    /// Materialize this `CompactBattle` state onto an existing `Battle` template (`dst`) from the
    /// same match (preserving static species/EV/IV/name/roster metadata). Used by diagnostic/solver
    /// callback hooks (`SearchTrace`, `step_with_leaf`, `step_forced_joint_with_root_child_value`,
    /// and eligible small-endgame `EndgameSession` leaves).
    #[cold]
    pub fn sync_to_battle(&self, table: &MatchTable, dex: &Dex, dst: &mut Battle) {
        use crate::state::{
            Action, ActionKind, Choice, EffId, EffectState, PokeId, RequestState, Scalar,
            SwitchFlag,
        };
        let dt = dex_tables(dex);
        let is_castle = table.ruleset.is_castle();

        dst.turn = self.turn;
        dst.prng = crate::prng::BattleRng::seeded(Prng::new(self.prng_seed));
        dst.quick_claw_rolls = [self.quick_claw_for(0), self.quick_claw_for(1)];
        dst.triple_kick_mult = self.triple_kick_mult;
        dst.bp_switched_this_turn = [
            (self.bp_switch_flags & 0x04) != 0,
            (self.bp_switch_flags & 0x08) != 0,
        ];
        dst.recoil_ko_side = if (self.turn_extra_flags & 0x01) != 0 {
            Some((self.turn_extra_flags >> 1) & 1)
        } else {
            None
        };
        dst.speed_order = [self.speed_order as usize, (self.speed_order ^ 1) as usize];

        dst.queue.clear();
        dst.faint_queue.clear();
        dst.event_stack.clear();
        dst.effect_stack.clear();
        dst.active_move = None;
        dst.active_pokemon = None;
        dst.active_target = None;
        dst.pending_boosts = None;

        match self.outcome {
            0 => {
                dst.ended = false;
                dst.winner = None;
            }
            1 => {
                dst.ended = true;
                dst.winner = Some(dst.sides[0].name.to_string());
            }
            2 => {
                dst.ended = true;
                dst.winner = Some(dst.sides[1].name.to_string());
            }
            _ => {
                dst.ended = true;
                dst.winner = Some(String::new());
            }
        }

        // Field weather
        let weather_cid = if self.weather_turns > 0 {
            match self.weather_kind {
                CompactWeather::Rain => dt.rain_cond,
                CompactWeather::Sun => dt.sun_cond,
                CompactWeather::Sandstorm => dt.sand_cond,
                CompactWeather::None => None,
            }
        } else {
            None
        };
        dst.field.weather = weather_cid;
        if let Some(cid) = weather_cid {
            let mut st = EffectState::default();
            st.id = EffId::Cond(cid);
            st.duration = Some(self.weather_turns as i32);
            dst.field.weather_state = st;
        } else {
            dst.field.weather_state = EffectState::default();
        }

        // Field pseudo-weather: preserve non-clause rules (e.g. `maxtotallevel`) so `state_key128()`
        // matches `Battle` solver cache entries, while syncing Stadium Sleep / Freeze clauses.
        dst.field.pseudo_weather.retain(|(cid, _)| {
            Some(*cid) != dt.sleep_clause_cond && Some(*cid) != dt.freeze_clause_cond
        });
        if (self.clause_flags & 1) != 0 {
            if let Some(cid) = dt.sleep_clause_cond {
                let mut st = EffectState::default();
                st.id = EffId::Cond(cid);
                dst.field.pseudo_weather.push((cid, st));
            }
        }
        if (self.clause_flags & 4) != 0 {
            if let Some(cid) = dt.freeze_clause_cond {
                let mut st = EffectState::default();
                st.id = EffId::Cond(cid);
                dst.field.pseudo_weather.push((cid, st));
            }
        }

        // Active PokeIds for cross-side volatile sources
        let mut active_pids = [None; 2];
        for side_n in 0..2usize {
            let c_side = &self.sides[side_n];
            let party_len = dst.sides[side_n].party.len().min(3);
            let act_idx = (c_side.active_idx as usize).min(party_len.saturating_sub(1));
            if party_len > 0 {
                active_pids[side_n] = Some(PokeId {
                    side: side_n as u8,
                    slot: c_side.party_roster[act_idx],
                });
            }
        }

        // Pre-residual mid-turn states have not yet decremented 2-turn volatile durations (`2 -> 1`).
        let pre_residual = self.outcome == 0
            && (self.sides[0].request == 2 || self.sides[1].request == 2)
            && (self.sides[0].side_flags & 0x08) == 0;
        let two_turn_dur = if pre_residual { 2 } else { 1 };

        // Sync both sides and their picked Pokemon
        for side_n in 0..2usize {
            let c_side = &self.sides[side_n];
            let foe_active = active_pids[side_n ^ 1];
            let party_len = dst.sides[side_n].party.len().min(3);
            let act_idx = (c_side.active_idx as usize).min(party_len.saturating_sub(1));

            // Restore `Battle::side.party` in exact `party_order(side_n)` (matching `Battle::switch_in` swap order).
            let p_order = self.party_order(side_n);
            dst.sides[side_n].party.clear();
            if party_len > 0 {
                let act_roster = c_side.party_roster[act_idx];
                dst.sides[side_n].party.push(act_roster);
                for &slot_u8 in &p_order {
                    let p_idx = slot_u8 as usize;
                    if p_idx < party_len && p_idx != act_idx {
                        dst.sides[side_n].party.push(c_side.party_roster[p_idx]);
                    }
                }
                dst.sides[side_n].active = Some(act_roster);
            } else {
                dst.sides[side_n].active = None;
            }
            for (pos, &r_slot) in dst.sides[side_n].party.iter().enumerate() {
                dst.sides[side_n].roster[r_slot as usize].position = pos as u8;
            }

            let alive_cnt = c_side.alive_mask.count_ones() as i32;
            dst.sides[side_n].pokemon_left = alive_cnt;
            dst.sides[side_n].total_fainted = (party_len as i32 - alive_cnt).max(0);
            let is_faint_replace = self.outcome == 0
                && c_side.request == 2
                && (self.bp_switch_flags & (1u8 << side_n)) == 0
                && party_len > 0;
            dst.sides[side_n].fainted_this_turn =
                is_faint_replace.then(|| c_side.party_roster[act_idx]);
            dst.sides[side_n].fainted_last_turn = None;
            dst.sides[side_n].last_move = if c_side.last_move_selfko != u8::MAX {
                let a_cidx = side_n * 3 + act_idx;
                if (c_side.last_move_selfko as usize) < (table.mons[a_cidx].num_moves as usize) {
                    Some(table.mons[a_cidx].moves[c_side.last_move_selfko as usize])
                } else {
                    let mut found = None;
                    'find_lm: for p_idx in 0..party_len {
                        let c_idx = side_n * 3 + p_idx;
                        let nm = table.mons[c_idx].num_moves as usize;
                        for m in 0..nm {
                            let mid = table.mons[c_idx].moves[m];
                            if (mid.0 & 0xFF) as u8 == c_side.last_move_selfko {
                                found = Some(mid);
                                break 'find_lm;
                            }
                        }
                    }
                    found
                }
            } else {
                None
            };

            dst.sides[side_n].request = if self.outcome != 0 {
                None
            } else {
                match c_side.request {
                    1 => Some(RequestKind::Move),
                    2 => Some(RequestKind::Switch),
                    _ => Some(RequestKind::Wait),
                }
            };
            let mut choice = Choice::default();
            if self.outcome == 0 && c_side.request == 2 {
                choice.forced_switches_left = 1;
            }
            dst.sides[side_n].choice = choice;

            // Side conditions
            dst.sides[side_n].side_conditions.clear();
            if c_side.reflect_turns > 0 {
                if let Some(cid) = dt.refl_cond {
                    let mut st = EffectState::default();
                    st.id = EffId::Cond(cid);
                    st.duration = Some(c_side.reflect_turns as i32);
                    dst.sides[side_n].side_conditions.push((cid, st));
                }
            }
            if c_side.lightscreen_turns > 0 {
                if let Some(cid) = dt.ls_cond {
                    let mut st = EffectState::default();
                    st.id = EffId::Cond(cid);
                    st.duration = Some(c_side.lightscreen_turns as i32);
                    dst.sides[side_n].side_conditions.push((cid, st));
                }
            }
            if c_side.safeguard_turns > 0 {
                if let Some(cid) = dt.safe_cond {
                    let mut st = EffectState::default();
                    st.id = EffId::Cond(cid);
                    st.duration = Some(c_side.safeguard_turns as i32);
                    dst.sides[side_n].side_conditions.push((cid, st));
                }
            }
            let spikes = c_side.side_flags & 0x03;
            if spikes > 0 {
                if let Some(cid) = dt.spikes_cond {
                    let mut st = EffectState::default();
                    st.id = EffId::Cond(cid);
                    st.set_int(DK::Layers, spikes as i64);
                    dst.sides[side_n].side_conditions.push((cid, st));
                }
            }

            // Slot conditions (Future Sight)
            dst.sides[side_n].slot_conditions.clear();
            if self.future_sight_turns[side_n] > 0 {
                if let Some(cid) = dt.futuremove_cond {
                    let mut st = EffectState::default();
                    st.id = EffId::Cond(cid);
                    let res_done = (self.sides[0].side_flags & 0x08) != 0;
                    let fs_eff_turn = self.turn as i64 + (res_done as i64);
                    let ending = fs_eff_turn - 1 + self.future_sight_turns[side_n] as i64 - 1;
                    st.set_int(DK::EndingTurn, ending);
                    st.set(DK::TargetSlot, Scalar::Slot(side_n as u8, 0));
                    st.duration = Some(self.future_sight_turns[side_n] as i32);
                    st.future_damage = Some(self.future_sight_dmg[side_n] as i16);
                    let foe = side_n ^ 1;
                    let src_p = (self.future_sight_src[side_n] as usize).min(2);
                    st.source = Some(PokeId {
                        side: foe as u8,
                        slot: self.sides[foe].party_roster[src_p],
                    });
                    dst.sides[side_n].slot_conditions.push((cid, st));
                }
            }

            // Per-Pokemon state
            for p_idx in 0..party_len {
                let c_idx = side_n * 3 + p_idx;
                let cm = &self.mons[c_idx];
                let spec = &table.mons[c_idx];
                let r_slot = c_side.party_roster[p_idx];
                let is_active_slot = p_idx == act_idx;
                let fainted = cm.hp() <= 0 || cm.status_raw() == Status::Fnt as u8;
                let p = &mut dst.sides[side_n].roster[r_slot as usize];

                p.hp = cm.hp().max(0) as i32;
                p.fainted = fainted;
                p.faint_queued = false;
                p.is_active = is_active_slot && !fainted;
                p.is_started = is_active_slot && !fainted;
                p.active_turns = if is_active_slot && !fainted {
                    p.active_turns.max(1)
                } else {
                    0
                };
                let is_bp_switch = is_active_slot
                    && self.outcome == 0
                    && c_side.request == 2
                    && (self.bp_switch_flags & (1u8 << side_n)) != 0;
                p.switch_flag = if is_active_slot && self.outcome == 0 && c_side.request == 2 {
                    if is_bp_switch {
                        dex.moves
                            .id("batonpass")
                            .map(SwitchFlag::Move)
                            .unwrap_or(SwitchFlag::Yes)
                    } else {
                        SwitchFlag::Yes
                    }
                } else {
                    SwitchFlag::No
                };
                p.force_switch_flag = false;
                p.skip_before_switch_out = is_bp_switch;
                p.being_called_back = false;
                p.dragged_in = None;
                p.newly_switched = false;
                p.hurt_this_turn = None;
                p.attacked_by.clear();
                p.last_damage = 0;

                p.item = if cm.item == 0 {
                    None
                } else {
                    Some(ItemId((cm.item - 1) as u16))
                };
                p.item_state = match p.item {
                    Some(iid) => EffectState {
                        id: EffId::Item(iid),
                        ..EffectState::default()
                    },
                    None => EffectState::default(),
                };
                p.boosts = cm.boosts;

                let st_enum = match cm.status_raw() {
                    1 => Status::Brn,
                    2 => Status::Par,
                    3 => Status::Slp,
                    4 => Status::Frz,
                    5 => Status::Psn,
                    6 => Status::Tox,
                    7 => Status::Fnt,
                    _ => Status::None,
                };
                p.status = if fainted && !is_active_slot {
                    Status::None
                } else {
                    st_enum
                };
                p.status_state = EffectState::default();
                if st_enum == Status::Slp {
                    if let Some(cid) = dex.conds_id("slp") {
                        p.status_state.id = EffId::Cond(cid);
                    }
                    p.status_state.set_int(DK::Time, cm.status_counter() as i64);
                } else if st_enum != Status::None && st_enum != Status::Fnt {
                    if let Some(cid) = dex.conds_id(st_enum.as_str()) {
                        p.status_state.id = EffId::Cond(cid);
                    }
                }

                let num_m = (spec.num_moves as usize).min(p.move_slots.len());
                let has_encore = (cm.vol_flags & vol_bit::ENCORE) != 0;
                let enc_slot = cm.encore_slot() as usize;
                for m_i in 0..num_m {
                    p.move_slots[m_i].pp = cm.pp[m_i] as i32;
                    let slot_dis = cm.is_slot_disabled(m_i);
                    let enc_dis = has_encore && m_i != enc_slot;
                    p.move_slots[m_i].disabled = if is_castle && has_encore && m_i == enc_slot {
                        false
                    } else {
                        slot_dis || enc_dis
                    };
                    if p.move_slots[m_i].shared && m_i < p.base_move_slots.len() {
                        p.base_move_slots[m_i].pp = p.move_slots[m_i].pp;
                        p.base_move_slots[m_i].disabled = p.move_slots[m_i].disabled;
                    }
                }

                let lm = cm.last_move_slot().and_then(|s| {
                    let si = s as usize;
                    (si < num_m).then(|| spec.moves[si])
                });
                p.last_move = lm;
                p.last_move_encore = lm;
                p.last_move_used = lm;
                p.trapped = cm.is_trapped();
                p.maybe_trapped = false;

                p.volatiles.clear();
                if !fainted {
                    let push_vol = |vols: &mut Vec<(crate::dex::CondId, EffectState)>,
                                    cid_opt: Option<crate::dex::CondId>,
                                    mut st: EffectState| {
                        if let Some(cid) = cid_opt {
                            st.id = EffId::Cond(cid);
                            st.has_name = true;
                            vols.push((cid, st));
                        }
                    };
                    if (cm.vol_flags & vol_bit::SUBSTITUTE) != 0 && cm.sub_hp() > 0 {
                        let mut st = EffectState::default();
                        st.set_int(DK::Hp, cm.sub_hp() as i64);
                        push_vol(&mut p.volatiles, dt.sub_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::CONFUSION) != 0 {
                        let mut st = EffectState::default();
                        st.set_int(DK::Time, cm.timers[timer_idx::CONFUSION] as i64);
                        push_vol(&mut p.volatiles, dt.conf_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::FLINCH) != 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(1);
                        push_vol(&mut p.volatiles, dt.flinch_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::DESTINY_BOND) != 0 {
                        push_vol(&mut p.volatiles, dt.dbond_cond, EffectState::default());
                    }
                    if (cm.vol_flags & vol_bit::PAR_SPEED_DROP) != 0 {
                        push_vol(&mut p.volatiles, dt.par_cond, EffectState::default());
                    }
                    if (cm.vol_flags & vol_bit::BRN_ATK_DROP) != 0 {
                        push_vol(&mut p.volatiles, dt.brn_cond, EffectState::default());
                    }
                    if (cm.vol_flags & vol_bit::LEECH_SEED) != 0 {
                        let mut st = EffectState::default();
                        st.source = foe_active;
                        st.source_slot = Some(((side_n ^ 1) as u8, 0));
                        push_vol(&mut p.volatiles, dt.seed_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::NIGHTMARE) != 0 {
                        push_vol(&mut p.volatiles, dt.night_cond, EffectState::default());
                    }
                    if (cm.vol_flags & vol_bit::CURSE) != 0 {
                        push_vol(&mut p.volatiles, dt.curse_cond, EffectState::default());
                    }
                    if (cm.vol_flags & vol_bit::FORESIGHT) != 0 {
                        let mut st = EffectState::default();
                        if is_castle {
                            st.source = foe_active;
                        }
                        push_vol(&mut p.volatiles, dt.fore_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::PROTECT) != 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(1);
                        push_vol(&mut p.volatiles, dt.prot_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::ENDURE) != 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(1);
                        push_vol(&mut p.volatiles, dt.end_cond, st);
                    }
                    let sc = cm.stall_counter_ext();
                    if sc > 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(two_turn_dur);
                        if is_castle {
                            st.set_int(DK::Counter, sc as i64);
                        } else {
                            let val = (127u32 >> (sc - 1).min(6)).max(1) as f64;
                            st.set(DK::Counter, Scalar::Float(val));
                        }
                        push_vol(&mut p.volatiles, dt.stall_cond, st);
                    }
                    if has_encore && enc_slot < num_m {
                        let mut st = EffectState::default();
                        st.duration = Some(cm.timers[timer_idx::ENCORE] as i32);
                        st.set(DK::Move, Scalar::MoveK(spec.moves[enc_slot]));
                        push_vol(&mut p.volatiles, dt.enc_cond, st);
                    }
                    if (cm.vol_flags & vol_bit::DISABLE) != 0 {
                        let dis_slot = cm.disable_slot() as usize;
                        if dis_slot < num_m {
                            let mut st = EffectState::default();
                            st.duration = Some(cm.timers[timer_idx::DISABLE] as i32);
                            st.set(DK::Move, Scalar::MoveK(spec.moves[dis_slot]));
                            push_vol(&mut p.volatiles, dt.dis_cond, st);
                        }
                    }
                    if (cm.vol_flags & vol_bit::PERISH_SONG) != 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(cm.perish_turns() as i32);
                        if (self.turn_extra_flags & (1u8 << (2 + side_n))) != 0 {
                            st.source = active_pids[side_n];
                        } else if (self.turn_extra_flags & (1u8 << (2 + (side_n ^ 1)))) != 0 {
                            st.source = foe_active;
                        }
                        push_vol(&mut p.volatiles, dt.perish_cond, st);
                    }
                    if st_enum != Status::Slp && (cm.status_counter() & 0x10) != 0 {
                        let mut st = EffectState::default();
                        st.set_int(DK::Counter, (cm.status_counter() & 0x0F) as i64);
                        push_vol(&mut p.volatiles, dt.res_cond, st);
                    }
                    if cm.must_recharge() {
                        push_vol(&mut p.volatiles, dt.recharge_cond, EffectState::default());
                    }
                    if (cm.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0 {
                        let mut st = EffectState::default();
                        st.source = foe_active;
                        push_vol(&mut p.volatiles, dt.trapped_cond, st);
                    }
                    if cm.partially_trapped() {
                        let mut st = EffectState::default();
                        st.duration = Some(cm.ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] as i32);
                        st.source = foe_active;
                        st.set_int(DK::BoundDivisor, 16);
                        push_vol(&mut p.volatiles, dt.partiallytrapped_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::ATTRACT) != 0 {
                        let mut st = EffectState::default();
                        st.source = foe_active;
                        push_vol(&mut p.volatiles, dt.attract_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::MIST) != 0 {
                        push_vol(&mut p.volatiles, dt.mist_cond, EffectState::default());
                    }
                    if (cm.ext_flags & ext_vol_bit::FOCUS_ENERGY) != 0 {
                        push_vol(&mut p.volatiles, dt.focusenergy_cond, EffectState::default());
                    }
                    if (cm.ext_flags & ext_vol_bit::LOCK_ON) != 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(cm.lock_on_turns().max(1) as i32);
                        st.source = foe_active;
                        push_vol(&mut p.volatiles, dt.lockon_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::DEFENSE_CURL) != 0 {
                        push_vol(&mut p.volatiles, dt.defensecurl_cond, EffectState::default());
                    }
                    if (cm.ext_flags & ext_vol_bit::MINIMIZE) != 0 {
                        push_vol(&mut p.volatiles, dt.minimize_cond, EffectState::default());
                    }
                    if (cm.ext_flags & ext_vol_bit::LOCKED_ROLLOUT) != 0 {
                        let mut st = EffectState::default();
                        let rc = cm.rollout_count().max(1) as i64;
                        st.duration = Some(two_turn_dur);
                        st.set_int(DK::HitCount, rc);
                        st.set_int(DK::ContactHitCount, rc);
                        push_vol(&mut p.volatiles, dt.rollout_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::LOCKED_OUTRAGE) != 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(cm.ext_timers[ext_timer_idx::LOCKED_TURNS].max(1) as i32);
                        let ls = cm.locked_move_slot() as usize;
                        if ls < num_m {
                            st.set(DK::Move, Scalar::MoveK(spec.moves[ls]));
                        }
                        push_vol(&mut p.volatiles, dt.lockedmove_cond, st);
                    }
                    if cm.ext_timers[ext_timer_idx::FURY_CUTTER] > 0 {
                        let mut st = EffectState::default();
                        st.duration = Some(two_turn_dur);
                        let mult = 1i64 << (cm.ext_timers[ext_timer_idx::FURY_CUTTER] - 1).min(12);
                        st.set_int(DK::Multiplier, mult);
                        push_vol(&mut p.volatiles, dt.furycutter_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::RAGE) != 0 {
                        let mut st = EffectState::default();
                        st.set_int(DK::Counter, cm.ext_timers[ext_timer_idx::RAGE_COUNTER] as i64);
                        push_vol(&mut p.volatiles, dt.rage_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::DIGGING) != 0 {
                        let mut st = EffectState::default();
                        if !is_castle || cm.two_turn_locked() {
                            st.duration = Some(two_turn_dur);
                        }
                        push_vol(&mut p.volatiles, dt.dig_cond, st);
                    }
                    if (cm.ext_flags & ext_vol_bit::FLYING) != 0 {
                        let mut st = EffectState::default();
                        if !is_castle || cm.two_turn_locked() {
                            st.duration = Some(two_turn_dur);
                        }
                        push_vol(&mut p.volatiles, dt.fly_cond, st);
                    }
                    if cm.two_turn_locked() {
                        let ls = cm.locked_move_slot() as usize;
                        if ls < num_m {
                            let mut st = EffectState::default();
                            st.duration = Some(two_turn_dur);
                            st.set(DK::Move, Scalar::MoveK(spec.moves[ls]));
                            push_vol(&mut p.volatiles, dt.twoturnmove_cond, st);
                        }
                    }
                }
            }
        }

        // Request state & mid-turn queue reconstruction
        if self.outcome != 0 {
            dst.request_state = RequestState::None;
            dst.mid_turn = false;
        } else if self.sides[0].request == 2 || self.sides[1].request == 2 {
            dst.request_state = RequestState::Switch;
            dst.mid_turn = true;
            for s in 0..2usize {
                let flags = self.sides[s].side_flags;
                if (flags & 0x10) != 0 {
                    if let Some(pid) = active_pids[s] {
                        let a_cidx = s * 3 + (self.sides[s].active_idx.min(2) as usize);
                        let m_slot = ((flags >> 5) & 0x03) as usize;
                        let move_id = if (self.bp_switch_flags & (1u8 << (4 + s))) != 0 {
                            dex.known_moves.struggle
                        } else if self.mons[a_cidx].must_recharge() {
                            dex.known_moves.recharge
                        } else if m_slot < table.mons[a_cidx].num_moves as usize {
                            Some(table.mons[a_cidx].moves[m_slot])
                        } else {
                            None
                        };
                        if let Some(mid) = move_id {
                            dst.queue.push(Action {
                                choice: ActionKind::Move {
                                    move_id: mid,
                                    target_loc: 1,
                                    original_target: None,
                                    source_effect: None,
                                },
                                order: 200,
                                priority: 0.0,
                                fractional_priority: 0.0,
                                speed: 1.0,
                                pokemon: Some(pid),
                            });
                        }
                    }
                }
            }
            if (self.sides[0].side_flags & 0x08) == 0 {
                dst.queue.push(Action {
                    choice: ActionKind::Residual,
                    order: 300,
                    priority: 0.0,
                    fractional_priority: 0.0,
                    speed: 1.0,
                    pokemon: None,
                });
            }
        } else {
            dst.request_state = RequestState::Move;
            dst.mid_turn = false;
        }

        // Refresh handler masks and effective speeds so `Battle` methods work seamlessly.
        for side_n in 0..2usize {
            let party: [u8; 3] = [
                dst.sides[side_n].party.first().copied().unwrap_or(0),
                dst.sides[side_n].party.get(1).copied().unwrap_or(0),
                dst.sides[side_n].party.get(2).copied().unwrap_or(0),
            ];
            let len = dst.sides[side_n].party.len().min(3);
            for &r_slot in &party[..len] {
                dst.refresh_poke_mask(
                    dex,
                    PokeId {
                        side: side_n as u8,
                        slot: r_slot,
                    },
                );
            }
            dst.refresh_side_mask(dex, side_n as u8);
        }
        dst.refresh_battle_mask(dex);
        dst.update_all_speeds(dex);
    }

    /// Compute any party Pokemon's effective stat (`party_slot` in `0..3`) on `side` (`0` or `1`)
    /// using [`get_stat_branchless`] (PS) or [`get_stat_castle`] (Castle PBS).
    #[inline(always)]
    pub fn mon_stat(
        &self,
        table: &MatchTable,
        side: usize,
        party_slot: usize,
        stat_idx: usize,
        unboosted: bool,
        unmodified: bool,
    ) -> i32 {
        let p_idx = side * 3 + party_slot;
        let mon = &self.mons[p_idx];
        let spec = &table.mons[p_idx];
        let side_st = &self.sides[side];
        let ispec = table.item_spec(mon.item);
        if table.ruleset.is_castle() {
            let item_kind = if ispec.stat_kind == ItemStatKind::MetalPowder
                && (spec.species_stat_kind == ItemStatKind::MetalPowder
                    || (spec.species_flags & (1 << 2)) != 0)
            {
                // Castle #35: Metal Powder persists after Transform (checks base species Ditto).
                ItemStatKind::MetalPowder
            } else if ispec.stat_kind != ItemStatKind::None
                && ispec.stat_kind == spec.species_stat_kind
            {
                ispec.stat_kind
            } else {
                ItemStatKind::None
            };
            // Castle #20: Burn/Paralysis penalties are NEVER cleared by stat boosts/Haze.
            return get_stat_castle(
                spec.stored_stats[stat_idx] as i32,
                mon.boosts[stat_idx],
                stat_idx,
                unboosted,
                unmodified,
                mon.status_raw() == Status::Par as u8,
                mon.status_raw() == Status::Brn as u8,
                side_st.reflect_turns > 0,
                side_st.lightscreen_turns > 0,
                item_kind,
                table.ruleset.overflow_mode(),
            );
        }
        let has_par = (mon.status_raw() == Status::Par as u8)
            & ((mon.vol_flags & vol_bit::PAR_SPEED_DROP) != 0);
        let has_brn = (mon.status_raw() == Status::Brn as u8)
            & ((mon.vol_flags & vol_bit::BRN_ATK_DROP) != 0);
        let item_kind = if ispec.stat_kind == ItemStatKind::MetalPowder
            && (mon.ext_flags & ext_vol_bit::TRANSFORMED) != 0
        {
            ItemStatKind::None
        } else if ispec.stat_kind != ItemStatKind::None
            && ispec.stat_kind == spec.species_stat_kind
        {
            ispec.stat_kind
        } else {
            ItemStatKind::None
        };
        get_stat_branchless(
            spec.stored_stats[stat_idx] as i32,
            mon.boosts[stat_idx],
            stat_idx,
            unboosted,
            unmodified,
            false,
            has_par,
            has_brn,
            side_st.reflect_turns > 0,
            side_st.lightscreen_turns > 0,
            item_kind,
        )
    }

    /// Exact non-truncated Castle speed scaled by [`CASTLE_SPEED_DENOM`] (`3360`).
    #[inline(always)]
    pub fn castle_speed_x3360(&self, table: &MatchTable, side: usize, party_slot: usize) -> u32 {
        let p_idx = side * 3 + party_slot;
        let mon = &self.mons[p_idx];
        let spec = &table.mons[p_idx];
        get_castle_speed_x3360(
            spec.stored_stats[stat_idx::SPE] as i32,
            mon.boosts[stat_idx::SPE],
            mon.status_raw() == Status::Par as u8,
        )
    }

    /// Compute the active Pokemon's effective stat on `side` (`0` or `1`).
    #[inline(always)]
    pub fn active_stat(
        &self,
        table: &MatchTable,
        side: usize,
        stat_idx: usize,
        unboosted: bool,
        unmodified: bool,
    ) -> i32 {
        self.mon_stat(
            table,
            side,
            self.sides[side].active_idx as usize,
            stat_idx,
            unboosted,
            unmodified,
        )
    }

    /// Whether Quick Claw rolled active for `side` (`bit 0 = P1, bit 1 = P2`).
    #[inline(always)]
    pub const fn quick_claw_for(&self, side: usize) -> bool {
        (self.quick_claw_roll & (1u8 << side)) != 0
    }

    #[inline(always)]
    fn roll_quick_claw_mask(ruleset: crate::rules::Ruleset, rng: &mut Prng) -> u8 {
        if ruleset.independent_quick_claw() {
            let p1 = rng.random_chance(60, 256) as u8;
            let p2 = rng.random_chance(60, 256) as u8;
            p1 | (p2 << 1)
        } else if rng.random_chance(60, 256) {
            0b11
        } else {
            0
        }
    }

    /// Compute a party Pokemon's action speed (matching `Battle::get_pokemon_action_speed`).
    #[inline(always)]
    pub fn mon_action_speed(&self, table: &MatchTable, side: usize, party_slot: usize) -> i32 {
        let p_idx = side * 3 + party_slot;
        let mon = &self.mons[p_idx];
        if self.quick_claw_for(side) && table.item_spec(mon.item).is_quick_claw {
            return QUICK_CLAW_ACTION_SPEED;
        }
        self.mon_stat(table, side, party_slot, stat_idx::SPE, false, false)
    }

    /// Compute hit probability for `m_slot` used by `(att_side, att_slot)` against `(def_side, def_slot)`.
    #[inline(always)]
    pub fn hit_probability(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
    ) -> f64 {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let mspec = &table.move_specs[att_idx][m_slot];
        if mspec.always_hit || (self.mons[att_idx].ext_flags & ext_vol_bit::LOCK_ON) != 0 {
            return 1.0;
        }
        if table.ruleset.is_castle() {
            let acc = lookup_accuracy_threshold_castle(
                mspec.base_acc_pct,
                self.mons[att_idx].boosts[stat_idx::ACC],
                self.mons[def_idx].boosts[stat_idx::EVA],
                mspec.ignore_accuracy,
                mspec.ignore_evasion,
                (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) != 0,
                false,
            );
            return if acc == 255 { 1.0 } else { acc as f64 / 256.0 };
        }
        hit_probability_branchless(
            mspec.base_acc_pct,
            mspec.always_hit,
            self.mons[att_idx].boosts[stat_idx::ACC],
            self.mons[def_idx].boosts[stat_idx::EVA],
            mspec.ignore_accuracy,
            mspec.ignore_evasion,
            mspec.ignore_positive_evasion,
        )
    }

    /// Expected fraction of `(def_side, def_slot)`'s current HP removed by move slot `m_slot`
    /// of `(att_side, att_slot)`, paired with its effective hit probability `acc`.
    #[inline]
    pub fn expected_hit_fraction_with_acc(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
    ) -> (f64, f64) {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let mspec = &table.move_specs[att_idx][m_slot];
        let att_mon = &self.mons[att_idx];
        let att_spec = &table.mons[att_idx];
        let def_mon = &self.mons[def_idx];
        let def_spec = &table.mons[def_idx];
        let is_castle = table.ruleset.is_castle();

        let base_power = if mspec.var_bp_kind == VarBpKind::FlailReversal {
            Self::flail_reversal_bp(att_mon.hp(), att_spec.maxhp, is_castle)
        } else if is_castle
            && matches!(mspec.var_bp_kind, VarBpKind::Return | VarBpKind::Frustration)
        {
            102
        } else {
            mspec.base_power as i32
        };

        if mspec.fixed_damage == 0
            && mspec.special_kind != SpecialMoveKind::SuperFang
            && (mspec.is_status || base_power <= 0)
        {
            return (0.0, 0.0);
        }

        let foresight = (def_mon.vol_flags & vol_bit::FORESIGHT) != 0;
        let (type_mod, is_immune) =
            table.effectiveness_ext(mspec.move_type, def_spec, foresight);
        if is_immune {
            return (0.0, 0.0);
        }

        let acc = self.hit_probability(table, att_side, att_slot, def_side, def_slot, m_slot);

        let raw = if mspec.special_kind == SpecialMoveKind::SuperFang {
            (def_mon.hp() / 2).max(1) as f64
        } else if mspec.fixed_damage != 0 {
            if mspec.fixed_damage < 0 {
                att_spec.level as f64
            } else {
                mspec.fixed_damage as f64
            }
        } else {
            let (ai, di) = if mspec.is_physical {
                (stat_idx::ATK, stat_idx::DEF)
            } else {
                (stat_idx::SPA, stat_idx::SPD)
            };
            let atk = self.mon_stat(table, att_side, att_slot, ai, false, false) as f64;
            let mut defense = self.mon_stat(table, def_side, def_slot, di, false, false) as f64;
            if mspec.is_selfdestruct && di == stat_idx::DEF {
                defense = (defense / 2.0).floor().max(1.0);
            }
            let mut dmg = ((att_spec.level as f64 * 2.0 / 5.0 + 2.0).floor()
                * base_power as f64
                * atk
                / defense
                / 50.0)
                .floor();
            let item_boost_type = if is_castle {
                table.item_spec(att_mon.item).castle_boost_type
            } else {
                table.item_spec(att_mon.item).boost_type
            };
            if mspec.move_type != NO_TYPE_ID && item_boost_type == mspec.move_type {
                dmg += (dmg / 10.0).floor();
            }
            let mut core = dmg.clamp(1.0, 997.0) + 2.0;
            if self.weather_turns > 0 {
                if (mspec.is_water && self.weather_kind == CompactWeather::Rain)
                    || (mspec.is_fire && self.weather_kind == CompactWeather::Sun)
                {
                    core = (core * 1.5).floor();
                } else if ((mspec.is_fire || mspec.is_solarbeam)
                    && self.weather_kind == CompactWeather::Rain)
                    || (mspec.is_water && self.weather_kind == CompactWeather::Sun)
                {
                    core = (core / 2.0).floor();
                }
            }
            let stab = if mspec.move_type < 32
                && (att_spec.type_mask & (1u32 << (mspec.move_type & 31))) != 0
            {
                1.5
            } else {
                1.0
            };
            let eff = match type_mod {
                2 => 4.0,
                1 => 2.0,
                -1 => 0.5,
                -2 => 0.25,
                _ => 1.0,
            };
            let hits = mspec.multihit_mean_x2 as f64 * 0.5;
            core * stab * eff * (236.0 / 255.0) * hits
        };

        (raw * acc / (def_mon.hp().max(1) as f64), acc)
    }

    /// Expected fraction of `(def_side, def_slot)`'s current HP removed by move slot `m_slot`
    /// of `(att_side, att_slot)`. Mirrors `nc2000_bot::eval::expected_hit_fraction` bit-for-bit.
    #[inline]
    pub fn expected_hit_fraction(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
    ) -> f64 {
        self.expected_hit_fraction_with_acc(
            table,
            att_side,
            att_slot,
            def_side,
            def_slot,
            m_slot,
        )
        .0
    }

    /// Best expected hit fraction over `(att_side, att_slot)`'s usable move slots against `(def_side, def_slot)`.
    #[inline]
    pub fn best_hit_fraction(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
    ) -> f64 {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let att_mon = &self.mons[att_idx];
        let att_spec = &table.mons[att_idx];
        let def_mon = &self.mons[def_idx];
        let def_spec = &table.mons[def_idx];
        let is_castle = table.ruleset.is_castle();
        let num_moves = att_spec.num_moves as usize;
        let foresight = (def_mon.vol_flags & vol_bit::FORESIGHT) != 0;
        let level_factor = (att_spec.level as f64 * 2.0 / 5.0 + 2.0).floor();
        let def_hp = def_mon.hp().max(1) as f64;
        let item_boost_type = if is_castle {
            table.item_spec(att_mon.item).castle_boost_type
        } else {
            table.item_spec(att_mon.item).boost_type
        };
        let mut phys_stats: Option<(f64, f64)> = None;
        let mut spec_stats: Option<(f64, f64)> = None;
        let mut best = 0.0f64;

        for m_i in 0..num_moves {
            if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
                continue;
            }
            let mspec = &table.move_specs[att_idx][m_i];
            let base_power = if mspec.var_bp_kind == VarBpKind::FlailReversal {
                Self::flail_reversal_bp(att_mon.hp(), att_spec.maxhp, is_castle)
            } else if is_castle
                && matches!(mspec.var_bp_kind, VarBpKind::Return | VarBpKind::Frustration)
            {
                102
            } else {
                mspec.base_power as i32
            };
            if mspec.fixed_damage == 0
                && mspec.special_kind != SpecialMoveKind::SuperFang
                && (mspec.is_status || base_power <= 0)
            {
                continue;
            }
            let (type_mod, is_immune) =
                table.effectiveness_ext(mspec.move_type, def_spec, foresight);
            if is_immune {
                continue;
            }
            let acc = self.hit_probability(table, att_side, att_slot, def_side, def_slot, m_i);
            let raw = if mspec.special_kind == SpecialMoveKind::SuperFang {
                (def_mon.hp() / 2).max(1) as f64
            } else if mspec.fixed_damage != 0 {
                if mspec.fixed_damage < 0 {
                    att_spec.level as f64
                } else {
                    mspec.fixed_damage as f64
                }
            } else {
                let (atk, mut defense) = if mspec.is_physical {
                    *phys_stats.get_or_insert_with(|| {
                        (
                            self.mon_stat(table, att_side, att_slot, stat_idx::ATK, false, false)
                                as f64,
                            self.mon_stat(table, def_side, def_slot, stat_idx::DEF, false, false)
                                as f64,
                        )
                    })
                } else {
                    *spec_stats.get_or_insert_with(|| {
                        (
                            self.mon_stat(table, att_side, att_slot, stat_idx::SPA, false, false)
                                as f64,
                            self.mon_stat(table, def_side, def_slot, stat_idx::SPD, false, false)
                                as f64,
                        )
                    })
                };
                if mspec.is_selfdestruct && mspec.is_physical {
                    defense = (defense / 2.0).floor().max(1.0);
                }
                let mut dmg = (level_factor * base_power as f64 * atk / defense / 50.0).floor();
                if mspec.move_type != NO_TYPE_ID && item_boost_type == mspec.move_type {
                    dmg += (dmg / 10.0).floor();
                }
                let mut core = dmg.clamp(1.0, 997.0) + 2.0;
                if self.weather_turns > 0 {
                    if (mspec.is_water && self.weather_kind == CompactWeather::Rain)
                        || (mspec.is_fire && self.weather_kind == CompactWeather::Sun)
                    {
                        core = (core * 1.5).floor();
                    } else if ((mspec.is_fire || mspec.is_solarbeam)
                        && self.weather_kind == CompactWeather::Rain)
                        || (mspec.is_water && self.weather_kind == CompactWeather::Sun)
                    {
                        core = (core / 2.0).floor();
                    }
                }
                let stab = if mspec.move_type < 32
                    && (att_spec.type_mask & (1u32 << (mspec.move_type & 31))) != 0
                {
                    1.5
                } else {
                    1.0
                };
                let eff = match type_mod {
                    2 => 4.0,
                    1 => 2.0,
                    -1 => 0.5,
                    -2 => 0.25,
                    _ => 1.0,
                };
                let hits = mspec.multihit_mean_x2 as f64 * 0.5;
                core * stab * eff * (236.0 / 255.0) * hits
            };
            let v = raw * acc / def_hp;
            if v > best {
                best = v;
            }
        }

        if cfg!(debug_assertions) {
            let mut ref_best = 0.0f64;
            for m_i in 0..num_moves {
                if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
                    continue;
                }
                let v = self.expected_hit_fraction(
                    table,
                    att_side,
                    att_slot,
                    def_side,
                    def_slot,
                    m_i,
                );
                if v > ref_best {
                    ref_best = v;
                }
            }
            debug_assert_eq!(best.to_bits(), ref_best.to_bits());
        }
        best
    }

    /// Flail / Reversal base power (PS `floor(48 * hp / maxhp)`, Castle `< 77.5%` 20↔40 boundary per `Probe 12E1`).
    #[inline(always)]
    pub const fn flail_reversal_bp(hp: i16, maxhp: i16, is_castle: bool) -> i32 {
        let mh = if maxhp < 1 { 1i32 } else { maxhp as i32 };
        let cur = if hp < 0 { 0i32 } else { hp as i32 };
        let ratio = (cur * 48) / mh;
        if ratio <= 1 {
            200
        } else if ratio <= 4 {
            150
        } else if ratio <= 9 {
            100
        } else if ratio <= 16 {
            80
        } else if is_castle {
            if cur * 1000 < mh * 775 { 40 } else { 20 }
        } else if ratio <= 32 {
            40
        } else {
            20
        }
    }

    /// True when `side`'s active mon has no usable non-Struggle move at the start of the turn
    /// (matching `legal_actions` Struggle fallback).
    #[inline(always)]
    pub fn is_forced_struggle(&self, table: &MatchTable, side: usize) -> bool {
        let p_idx = side * 3 + (self.sides[side].active_idx as usize);
        let mon = &self.mons[p_idx];
        if mon.must_recharge() || mon.is_multi_turn_locked() {
            return false;
        }
        let num_moves = table.mons[p_idx].num_moves as usize;
        if (mon.vol_flags & vol_bit::ENCORE) != 0 {
            let enc_slot = mon.encore_slot() as usize;
            if table.ruleset.is_castle() {
                // Castle #54①: Encore + Disable does NOT force Struggle (exec_move aborts on Disable).
                !(enc_slot < num_moves && mon.pp[enc_slot] > 0)
            } else {
                !(enc_slot < num_moves && mon.pp[enc_slot] > 0 && !mon.is_slot_disabled(enc_slot))
            }
        } else {
            !(0..num_moves).any(|m| mon.pp[m] > 0 && !mon.is_slot_disabled(m))
        }
    }

    /// Compute the branchless `ActionSortKey` for `side` (`0` or `1`) given a
    /// compact action (`0..4` = move slot, `4..7` = switch to party index `0..3`).
    #[inline(always)]
    pub fn action_sort_key(
        &self,
        table: &MatchTable,
        side: usize,
        action_code: u8,
        tie_bit: u8,
    ) -> ActionSortKey {
        let p_idx = side * 3 + (self.sides[side].active_idx as usize);
        let mon = &self.mons[p_idx];
        let speed = self.active_stat(table, side, stat_idx::SPE, false, false);
        let must_recharge = mon.must_recharge();
        let is_locked = mon.is_multi_turn_locked();
        let is_switch = !must_recharge && !is_locked && action_code >= 4;
        let phase = if is_switch {
            ActionPhase::Switch
        } else {
            ActionPhase::Move
        };
        let move_slot = if is_locked {
            mon.locked_move_slot() as usize
        } else if (mon.vol_flags & vol_bit::ENCORE) != 0 {
            mon.encore_slot() as usize
        } else {
            (action_code & 3) as usize
        };
        let prio = if is_switch || must_recharge || self.is_forced_struggle(table, side) {
            0i8
        } else {
            table.move_specs[p_idx][move_slot.min(3)].priority
        };
        let qc_active = (!is_switch)
            & table.item_spec(mon.item).is_quick_claw
            & self.quick_claw_for(side);
        ActionSortKey::new(phase, prio, qc_active, speed, tie_bit)
    }

    /// Compute the exact 64-bit Castle turn-order sort key using non-truncated
    /// `x3360` speed (`Probe 5B/8E/9B/12H`) and 2-bit independent Quick Claw.
    #[inline(always)]
    pub fn action_sort_key_exact(
        &self,
        table: &MatchTable,
        side: usize,
        action_code: u8,
        tie_bit: u8,
    ) -> u64 {
        let active_slot = self.sides[side].active_idx as usize;
        let p_idx = side * 3 + active_slot;
        let mon = &self.mons[p_idx];
        let must_recharge = mon.must_recharge();
        let is_locked = mon.is_multi_turn_locked();
        let is_switch = !must_recharge && !is_locked && action_code >= 4;
        let phase = if is_switch {
            ActionPhase::Switch as u64
        } else {
            ActionPhase::Move as u64
        };
        let move_slot = if is_locked {
            mon.locked_move_slot() as usize
        } else if (mon.vol_flags & vol_bit::ENCORE) != 0 {
            mon.encore_slot() as usize
        } else {
            (action_code & 3) as usize
        };
        let prio = if is_switch || must_recharge || self.is_forced_struggle(table, side) {
            0i8
        } else {
            table.move_specs[p_idx][move_slot.min(3)].priority
        };
        let qc_active = (!is_switch)
            & table.item_spec(mon.item).is_quick_claw
            & self.quick_claw_for(side);
        let speed_bits = if qc_active {
            CASTLE_QC_SATURATED_SPEED as u64
        } else {
            (self.castle_speed_x3360(table, side, active_slot) as u64) & 0x3F_FFFF
        };
        let phase_bits = (phase & 0x7) << 28;
        let prio_bits = (((prio as i32 + 8) as u64) & 0xF) << 24;
        phase_bits | prio_bits | (speed_bits << 1) | ((tie_bit & 1) as u64)
    }

    /// Resolve effective move slot for `side` given `action_code` (`0..4`), honoring
    /// multi-turn locks (Outrage/Rollout/Dig/Fly) and Encore.
    #[inline(always)]
    fn resolve_move_slot(&self, table: &MatchTable, side: usize, action_code: u8) -> usize {
        let p_idx = side * 3 + self.sides[side].active_idx as usize;
        let mon = &self.mons[p_idx];
        if mon.is_multi_turn_locked() {
            return mon.locked_move_slot() as usize;
        }
        if (mon.vol_flags & vol_bit::ENCORE) != 0 {
            let enc_slot = mon.encore_slot() as usize;
            if enc_slot < table.mons[p_idx].num_moves as usize {
                return enc_slot;
            }
        }
        (action_code & 3) as usize
    }

    /// Execute one full 4-stage turn on `CompactBattle` with zero heap allocations:
    /// - Stage 0: Turn-order determination (`ActionSortKey` for PS, exact `x3360` for Castle)
    /// - Stage 1: First side action + `AfterMoveSelf` slip damage (with Pursuit interception & explicit `BATONN` pause)
    /// - Stage 2: Second side action + `AfterMoveSelf` slip damage (skipped on Turn-Cut faint)
    /// - Stage 3: End-of-turn `Residual` (speed order for PS, fixed `1P -> 2P` for Castle) + turn-limit judge
    ///
    /// `actions`: `[action_p1, action_p2]` where `0..4` is a move slot and `4..7`
    /// is switching to party index `action - 4`.
    pub fn step_turn(&mut self, table: &MatchTable, actions: [u8; 2]) {
        if self.outcome != 0 {
            return;
        }
        let is_castle = table.ruleset.is_castle();
        let mut rng = Prng::new(self.prng_seed);
        self.sides[0].side_flags &= !0x7C;
        self.sides[1].side_flags &= !0x74;
        self.sides[0].last_move_selfko = u8::MAX;
        self.sides[1].last_move_selfko = u8::MAX;
        self.hurt_damage_this_turn = [0, 0];
        self.hurt_cat_this_turn = [0, 0];
        self.bp_switch_flags = 0;
        self.triple_kick_mult = 0;
        // Preserve Perish Song user bits (bits 2..3) only while Perish Song is active.
        let any_perish = self.mons.iter().any(|m| (m.vol_flags & vol_bit::PERISH_SONG) != 0);
        self.turn_extra_flags = if any_perish {
            self.turn_extra_flags & 0x0C
        } else {
            0
        };

        // Clear single-turn Protect/Endure/Flinch/Rage-armed flags at start of turn,
        // and trigger lead Berserk Gene if not already consumed at battle start.
        for side in 0..2usize {
            let idx = side * 3 + self.sides[side].active_idx as usize;
            self.mons[idx].vol_flags &= !(vol_bit::PROTECT | vol_bit::ENDURE | vol_bit::FLINCH);
            self.mons[idx].ext_flags &= !ext_vol_bit::RAGE_ARMED;
            self.trigger_berserk_gene(table, side, &mut rng);
        }

        // Snapshot start-of-turn Struggle state so mid-turn Disable aborts the queued move
        // (matching Battle::DisableBeforeMove prio 7) instead of morphing it into Struggle.
        let forced_struggle = [
            self.is_forced_struggle(table, 0),
            self.is_forced_struggle(table, 1),
        ];

        // Stage 0: Turn-order comparison (exact non-truncated speed in Castle, branchless u32 in PS).
        let (first, second) = if is_castle {
            let k0 = self.action_sort_key_exact(table, 0, actions[0], 0);
            let k1 = self.action_sort_key_exact(table, 1, actions[1], 1);
            if (k0 >> 1) == (k1 >> 1) {
                let f = (rng.random(2) == 1) as usize;
                (f, f ^ 1)
            } else {
                let f = (k0 < k1) as usize;
                (f, f ^ 1)
            }
        } else {
            let k0 = self.action_sort_key(table, 0, actions[0], 0);
            let k1 = self.action_sort_key(table, 1, actions[1], 1);
            if k0.ties_without_rng(k1) {
                let f = (rng.random(2) == 1) as usize;
                (f, f ^ 1)
            } else {
                let f = (k0.0 < k1.0) as usize;
                (f, f ^ 1)
            }
        };
        self.speed_order = first as u8;

        // Stage 1 & Stage 2: Execute first action, then second action.
        let mut consumed = [false; 2];
        let mut paused_for_bp = false;
        for (step_i, &actor) in [first, second].iter().enumerate() {
            if self.outcome != 0 {
                break;
            }
            if consumed[actor] {
                continue;
            }
            let p0 = self.sides[0].active_idx as usize;
            let p1 = self.sides[1].active_idx as usize;
            let any_fainted = self.mons[p0].hp() <= 0 || self.mons[3 + p1].hp() <= 0;
            // Castle #3 (`Probe 9F`): Turn-Cut skips second mover's action EVEN IF it is a Switch!
            if is_castle && step_i == 1 && any_fainted {
                break;
            }

            let act = actions[actor];
            let act_idx = actor * 3 + self.sides[actor].active_idx as usize;
            let is_switch = act >= 4
                && act < 7
                && !self.mons[act_idx].must_recharge()
                && !self.mons[act_idx].is_multi_turn_locked();

            if is_switch {
                // Pursuit interception: if foe queued Pursuit, Pursuit strikes BEFORE switch-out at BP 80!
                let foe = actor ^ 1;
                let foe_idx = foe * 3 + self.sides[foe].active_idx as usize;
                let foe_act = actions[foe];
                let foe_can_pursuit = !consumed[foe]
                    && foe_act < 4
                    && !forced_struggle[foe]
                    && self.mons[foe_idx].hp() > 0
                    && !self.mons[foe_idx].must_recharge()
                    && !self.mons[foe_idx].is_multi_turn_locked()
                    && {
                        let slot = self.resolve_move_slot(table, foe, foe_act);
                        slot < table.mons[foe_idx].num_moves as usize
                            && table.move_specs[foe_idx][slot].special_kind
                                == SpecialMoveKind::Pursuit
                    };
                if foe_can_pursuit {
                    consumed[foe] = true;
                    let p_slot = self.resolve_move_slot(table, foe, foe_act);
                    // In Gen 2, slower Pursuit clears switching target's Destiny Bond before striking.
                    let foe_slower = if is_castle {
                        let s_foe = self.castle_speed_x3360(table, foe, self.sides[foe].active_idx as usize);
                        let s_act = self.castle_speed_x3360(table, actor, self.sides[actor].active_idx as usize);
                        s_foe < s_act || (s_foe == s_act && rng.random_chance(1, 2))
                    } else {
                        let s_foe = self.active_stat(table, foe, stat_idx::SPE, false, false);
                        let s_act = self.active_stat(table, actor, stat_idx::SPE, false, false);
                        s_foe < s_act || (s_foe == s_act && rng.random_chance(1, 2))
                    };
                    if foe_slower {
                        self.mons[act_idx].vol_flags &= !vol_bit::DESTINY_BOND;
                    }
                    self.exec_move_inner(table, foe, p_slot, false, false, true, &mut rng);
                    if self.mons[act_idx].hp() <= 0 {
                        // Target KO'd on switch-out by Pursuit!
                        if is_castle {
                            // Castle #4 (`Probe 10B`): planned switch is canceled; Turn-Cut opens fresh CHANGE!
                            break;
                        } else {
                            // PS: mark fainted mon dead in alive_mask, then execute planned switch.
                            let old_slot = self.sides[actor].active_idx as usize;
                            self.sides[actor].alive_mask &= !(1 << old_slot);
                            let target_slot = (act - 4).min(2);
                            if (self.sides[actor].alive_mask & (1 << target_slot)) != 0 {
                                self.exec_switch(
                                    table,
                                    actor,
                                    target_slot,
                                    SwitchKind::FaintReplace,
                                    &mut rng,
                                );
                            }
                            continue;
                        }
                    }
                }
                self.exec_switch(table, actor, (act - 4).min(2), SwitchKind::Manual, &mut rng);
            } else {
                // Gen 2: fainting skips any remaining queued move actions this turn.
                if step_i == 1 && any_fainted {
                    break;
                }
                let eff_slot = if act < 4 {
                    (act & 3) as usize
                } else {
                    self.mons[act_idx].locked_move_slot() as usize
                };
                self.exec_move_inner(
                    table,
                    actor,
                    eff_slot,
                    step_i == 1,
                    forced_struggle[actor],
                    false,
                    &mut rng,
                );
                // Castle #32②: First-mover `Rest` triggers Mint/Miracle Berry right after second mover acts
                // (BEFORE checking `BATONN` pause so Rest user wakes up before opponent picks BP receiver!).
                if is_castle && step_i == 1 && (self.bp_switch_flags & 0xC0) != 0 {
                    for s in 0..2usize {
                        if (self.bp_switch_flags & (1u8 << (6 + s))) != 0 {
                            self.bp_switch_flags &= !(1u8 << (6 + s));
                            self.check_status_berry_castle(table, s);
                        }
                    }
                }
                // If Baton Pass opened explicit `BATONN` switch selection (`request == 2`), pause mid-turn!
                if (self.bp_switch_flags & (1u8 << actor)) != 0 {
                    paused_for_bp = true;
                    if step_i == 0 && !consumed[second] && actions[second] < 4 {
                        self.sides[second].side_flags = (self.sides[second].side_flags & !0x70)
                            | 0x10
                            | (((actions[second] & 3) as u8) << 5);
                        if forced_struggle[second] {
                            self.bp_switch_flags |= 1u8 << (4 + second);
                        }
                        // Mark first-mover BP for Castle 2x Pursuit after Spikes (`Probe 12C`).
                        self.bp_switch_flags |= 1u8 << (2 + actor);
                    }
                    break;
                }
            }
        }

        if paused_for_bp {
            self.prng_seed = rng.seed();
            return;
        }

        // Flush any remaining deferred first-mover Rest berry if second mover didn't run (e.g., Turn-Cut or Pursuit).
        if is_castle && (self.bp_switch_flags & 0xC0) != 0 {
            for s in 0..2usize {
                if (self.bp_switch_flags & (1u8 << (6 + s))) != 0 {
                    self.bp_switch_flags &= !(1u8 << (6 + s));
                    self.check_status_berry_castle(table, s);
                }
            }
        }

        self.check_faints(table, &mut rng);

        // Stage 3: End-of-turn Residual when no forced switch is pending.
        if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
            self.sides[0].side_flags |= 0x08;
            self.exec_residuals(table, &mut rng);
            self.check_faints(table, &mut rng);
            if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
                self.sides[0].side_flags &= !0x08;
                self.finish_turn_advance(table, &mut rng);
            }
        }
        self.clear_turn_transients();
        self.prng_seed = rng.seed();
    }

    #[inline(always)]
    fn clear_turn_transients(&mut self) {
        self.sides[0].side_flags &= !0x04;
        self.sides[1].side_flags &= !0x04;
        self.hurt_damage_this_turn = [0, 0];
        self.hurt_cat_this_turn = [0, 0];
        self.speed_order = 0;
        self.triple_kick_mult = 0;
        self.bp_switch_flags &= 0x03;
        let any_perish = self.mons.iter().any(|m| (m.vol_flags & vol_bit::PERISH_SONG) != 0);
        self.turn_extra_flags = if any_perish {
            self.turn_extra_flags & 0x0C
        } else {
            0
        };
    }

    /// Judge a Castle `L_TURN` timeout (`Probe 12A1-12A3`):
    /// ① Alive count (`alive_mask.count_ones()`) -> ② sum of alive `hp / maxhp` ratios -> ③ Tie (`3`).
    #[inline]
    pub fn judge_castle_turn_limit(&self, table: &MatchTable) -> u8 {
        let c0 = self.sides[0].alive_mask.count_ones();
        let c1 = self.sides[1].alive_mask.count_ones();
        if c0 > c1 {
            return 1;
        }
        if c1 > c0 {
            return 2;
        }
        let mut r0 = 0.0f64;
        let mut r1 = 0.0f64;
        for slot in 0..3usize {
            if (self.sides[0].alive_mask & (1 << slot)) != 0 {
                let m = &self.mons[slot];
                let mh = table.mons[slot].maxhp.max(1) as f64;
                r0 += (m.hp().max(0) as f64) / mh;
            }
            if (self.sides[1].alive_mask & (1 << slot)) != 0 {
                let m = &self.mons[3 + slot];
                let mh = table.mons[3 + slot].maxhp.max(1) as f64;
                r1 += (m.hp().max(0) as f64) / mh;
            }
        }
        if (r0 - r1).abs() <= 1e-12 {
            3 // Tie / DRAW
        } else if r0 > r1 {
            1 // P1Win
        } else {
            2 // P2Win
        }
    }

    #[inline(always)]
    fn clear_stale_perish_bits(&mut self) {
        let any_perish = (0..6usize).any(|i| {
            self.mons[i].hp() > 0 && (self.mons[i].vol_flags & vol_bit::PERISH_SONG) != 0
        });
        if !any_perish {
            self.turn_extra_flags &= !0x0C;
        }
    }

    #[inline(always)]
    fn finish_outcome(&mut self, outcome: u8) {
        self.outcome = outcome;
        self.sides[0].request = 0;
        self.sides[1].request = 0;
        self.sides[0].side_flags &= !0x08;
        self.future_sight_turns = [0, 0];
        self.clear_stale_perish_bits();
    }

    #[inline]
    fn finish_turn_advance(&mut self, table: &MatchTable, rng: &mut Prng) {
        if let Some(cfg) = table.ruleset.castle_overlay() {
            self.turn = self.turn.saturating_add(1);
            if let Some(limit) = cfg.turn_limit {
                if self.turn > limit {
                    let out = self.judge_castle_turn_limit(table);
                    self.finish_outcome(out);
                    return;
                }
            }
            self.quick_claw_roll = Self::roll_quick_claw_mask(table.ruleset, rng);
        } else {
            self.turn = self.turn.saturating_add(1);
            if self.turn > 1000 {
                self.finish_outcome(3); // Tie
            } else {
                self.quick_claw_roll = Self::roll_quick_claw_mask(table.ruleset, rng);
            }
        }
    }

    /// Apply a stat stage delta (`-6..=6`) matching `Battle::boost`: under PS (`!is_castle`),
    /// any actual non-zero Attack stage change removes `BRN_ATK_DROP` (if Burned) and any actual
    /// non-zero Speed stage change removes `PAR_SPEED_DROP` (if Paralyzed). Castle PBS #20
    /// preserves Burn/Paralysis drops across boosts.
    #[inline(always)]
    fn apply_boost_stage(&mut self, idx: usize, stat: usize, delta: i8, is_castle: bool) {
        let before = self.mons[idx].boosts[stat];
        let after = (before + delta).clamp(-6, 6);
        self.mons[idx].boosts[stat] = after;
        if !is_castle && after != before {
            if stat == stat_idx::ATK && self.mons[idx].status_raw() == Status::Brn as u8 {
                self.mons[idx].vol_flags &= !vol_bit::BRN_ATK_DROP;
            } else if stat == stat_idx::SPE && self.mons[idx].status_raw() == Status::Par as u8 {
                self.mons[idx].vol_flags &= !vol_bit::PAR_SPEED_DROP;
            }
        }
    }

    /// Trigger Berserk Gene on `side`'s active mon (+2 Atk, Confusion if not Safeguarded, consume item).
    #[inline]
    fn trigger_berserk_gene(&mut self, table: &MatchTable, side: usize, rng: &mut Prng) {
        let idx = side * 3 + (self.sides[side].active_idx.min(2) as usize);
        if self.mons[idx].hp() <= 0 || self.mons[idx].item == 0 {
            return;
        }
        if !table.item_spec(self.mons[idx].item).is_berserk_gene {
            return;
        }
        self.mons[idx].item = 0;
        self.apply_boost_stage(idx, stat_idx::ATK, 2, table.ruleset.is_castle());
        if self.sides[side].safeguard_turns == 0
            && (self.mons[idx].vol_flags & vol_bit::CONFUSION) == 0
        {
            self.mons[idx].vol_flags |= vol_bit::CONFUSION;
            self.mons[idx].timers[timer_idx::CONFUSION] = (2 + rng.random(4)) as u8;
        }
    }

    #[inline]
    fn exec_switch(
        &mut self,
        table: &MatchTable,
        side: usize,
        target_party_idx: u8,
        kind: SwitchKind,
        rng: &mut Prng,
    ) {
        let is_castle = table.ruleset.is_castle();
        let baton_pass = kind == SwitchKind::BatonPass;
        let old_idx = side * 3 + (self.sides[side].active_idx.min(2) as usize);
        let saved_boosts = self.mons[old_idx].boosts;
        let saved_sub = self.mons[old_idx].hp_pool[1];
        let pass_mask = vol_bit::SUBSTITUTE
            | vol_bit::CONFUSION
            | vol_bit::LEECH_SEED
            | vol_bit::CURSE
            | vol_bit::FORESIGHT
            | vol_bit::PERISH_SONG
            | vol_bit::TRAPPED_OR_LOCKED;
        let saved_vols = self.mons[old_idx].vol_flags & pass_mask;
        let ext_pass_mask = ext_vol_bit::MIST
            | ext_vol_bit::FOCUS_ENERGY
            | if is_castle {
                // Castle #50① (`Probe 11C`): Defense Curl & Minimize are passed by Baton Pass!
                ext_vol_bit::DEFENSE_CURL | ext_vol_bit::MINIMIZE
            } else {
                // PS (`gen2stadium2.json`): lockon has `noCopy: false`, so Baton Pass copies it.
                ext_vol_bit::LOCK_ON
            };
        let saved_ext_vols = self.mons[old_idx].ext_flags & ext_pass_mask;
        let saved_lock_on_turns = if !is_castle && (saved_ext_vols & ext_vol_bit::LOCK_ON) != 0 {
            self.mons[old_idx].lock_on_turns()
        } else {
            0
        };
        let old_was_foresight_caster =
            (self.mons[old_idx].ext_flags & ext_vol_bit::FORESIGHT_CASTER) != 0;
        let saved_pt = self.mons[old_idx].partially_trapped();
        let saved_pt_turns = self.mons[old_idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS];
        let saved_conf_t = self.mons[old_idx].timers[timer_idx::CONFUSION];
        let saved_perish_t = self.mons[old_idx].perish_turns();
        let old_st = self.mons[old_idx].status_raw();
        let saved_res = if old_st != Status::Slp as u8
            && old_st != Status::Fnt as u8
            && (self.mons[old_idx].status_counter() & 0x10) != 0
        {
            self.mons[old_idx].status_counter()
        } else {
            0
        };

        // Clear volatile state and boosts on outgoing active Pokemon (preserving gender!).
        // Matches `Battle::clear_volatile`: benched mons do NOT keep `parspeeddrop`/`brnattackdrop`
        // (they are re-attached on switch-in below).
        self.mons[old_idx].boosts = [0; 7];
        self.mons[old_idx].hp_pool[1] = 0;
        if old_st != Status::Slp as u8 && old_st != Status::Fnt as u8 {
            self.mons[old_idx].set_status(old_st, 0);
        }
        self.mons[old_idx].vol_flags = 0;
        self.mons[old_idx].timers = [0; 4];
        self.mons[old_idx].slot_refs = 0;
        self.mons[old_idx].ext_flags = 0;
        self.mons[old_idx].ext_timers = [0; 4];
        self.mons[old_idx].rollout_rage = 0;

        // Opponent switch-out expires active Destiny Bond (`onFoeSwitchOut`, NOT on FaintReplace),
        // partial traps, Attract, Lock-On (on non-BP non-FaintReplace switch), and locked-move
        // 1st-hit Counter eligibility; non-Baton-Pass switching also releases Mean Look / Spider Web
        // trap (and Castle Foresight caster).
        let foe_side = side ^ 1;
        let foe_idx = foe_side * 3 + (self.sides[foe_side].active_idx.min(2) as usize);
        if kind != SwitchKind::FaintReplace {
            self.mons[foe_idx].vol_flags &= !vol_bit::DESTINY_BOND;
        }
        self.mons[foe_idx].set_partially_trapped(false);
        self.mons[foe_idx].ext_flags &=
            !(ext_vol_bit::ATTRACT | ext_vol_bit::LOCKED_FIRST_HIT_RECV);
        if is_castle && old_was_foresight_caster {
            self.mons[foe_idx].vol_flags &= !vol_bit::FORESIGHT;
        }
        if !baton_pass {
            self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
            if kind != SwitchKind::FaintReplace {
                self.mons[foe_idx].ext_flags &= !ext_vol_bit::LOCK_ON;
                self.mons[foe_idx].set_lock_on_turns(0);
            }
        }

        self.sides[side].active_idx = target_party_idx;
        let cur_perm = self.reserved_castle[side] as usize;
        if cur_perm < 6 && (target_party_idx as usize) < 3 {
            self.reserved_castle[side] = PARTY_SWAP_ACTIVE[cur_perm][target_party_idx as usize];
        }
        let new_idx = side * 3 + target_party_idx as usize;
        if baton_pass {
            self.mons[new_idx].boosts = saved_boosts;
            self.mons[new_idx].hp_pool[1] = saved_sub;
            self.mons[new_idx].vol_flags |= saved_vols;
            self.mons[new_idx].ext_flags |= saved_ext_vols;
            if saved_lock_on_turns > 0 {
                self.mons[new_idx].set_lock_on_turns(saved_lock_on_turns);
            }
            self.mons[new_idx].timers[timer_idx::CONFUSION] = saved_conf_t;
            self.mons[new_idx].set_perish_turns(saved_perish_t);
            if saved_pt {
                self.mons[new_idx].set_partially_trapped(true);
                self.mons[new_idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] = saved_pt_turns;
            }
        }
        if self.mons[new_idx].status_raw() == Status::Par as u8 {
            self.mons[new_idx].vol_flags |= vol_bit::PAR_SPEED_DROP;
        }
        if self.mons[new_idx].status_raw() == Status::Brn as u8 {
            self.mons[new_idx].vol_flags |= vol_bit::BRN_ATK_DROP;
        }
        if self.mons[new_idx].status_raw() == Status::Tox as u8 {
            self.mons[new_idx].set_status(Status::Psn as u8, 0);
        }
        if baton_pass && saved_res != 0 {
            let new_st = self.mons[new_idx].status_raw();
            // Preserve `residualdmg` volatile across Baton Pass (matching `Battle::copy_volatile_from`
            // for onward BP propagation); Castle #14 skips applying/incrementing it on Burned mons
            // in `exec_switch` and `exec_after_move_self`.
            if new_st != Status::Slp as u8 && new_st != Status::Fnt as u8 {
                self.mons[new_idx].set_status(new_st, saved_res);
            }
        }

        // Apply Spikes entry hazard to non-Flying switch-in.
        let spikes_layers = self.sides[side].side_flags & 0x03;
        if spikes_layers != 0
            && (table.mons[new_idx].type_mask & (1u32 << (table.flying_type & 31))) == 0
        {
            let spike_dmg = match spikes_layers {
                1 => table.mons[new_idx].quantum_1_8,
                2 => (table.mons[new_idx].maxhp / 6).max(1),
                _ => (table.mons[new_idx].maxhp / 4).max(1),
            };
            self.mons[new_idx].apply_damage_branchless(spike_dmg, true);
        }

        // Gen 2 `AfterSwitchInSelf` (`actions.rs:248-252`): manual switches and
        // Baton Pass (not faint replacements or Roar/Whirlwind drag-ins) incur
        // Burn/Poison slip damage immediately on switch-in.
        // Note: `PsnAfterMoveSelf` runs BEFORE `ResidualDmgAfterMoveSelf` on switch-in
        // in both PS and Castle (`conditions.rs:219, 440-455`), so damage uses pre-increment `c`.
        if matches!(kind, SwitchKind::Manual | SwitchKind::BatonPass)
            && self.mons[new_idx].hp() > 0
        {
            let st = self.mons[new_idx].status_raw();
            if st == Status::Brn as u8 || st == Status::Psn as u8 {
                let cnt = self.mons[new_idx].status_counter();
                let use_res = (cnt & 0x10) != 0 && !(is_castle && st == Status::Brn as u8);
                let slip = if use_res {
                    let c = cnt & 0x0F;
                    self.mons[new_idx].set_status(st, (c + 1).min(15) | 0x10);
                    (table.mons[new_idx].quantum_1_16 * (c as i16)).max(1)
                } else {
                    table.mons[new_idx].quantum_1_8
                };
                self.mons[new_idx].apply_damage_branchless(slip, true);
            }
        }
        self.clear_stale_perish_bits();

        // Berserk Gene triggers on switch-in (Manual, Baton Pass, FaintReplace, Drag) if alive.
        if self.mons[new_idx].hp() > 0 {
            self.trigger_berserk_gene(table, side, rng);
        }
    }

    #[inline]
    fn exec_move(
        &mut self,
        table: &MatchTable,
        side: usize,
        move_slot: usize,
        moved_second: bool,
        is_struggle: bool,
        rng: &mut Prng,
    ) {
        self.exec_move_inner(table, side, move_slot, moved_second, is_struggle, false, rng);
    }

    /// Compute scheduled crash recoil for High Jump Kick / Jump Kick (PS 1/8, Castle 1/4 ignoring target maxhp).
    #[inline]
    fn apply_crash_kick_recoil(
        &mut self,
        table: &MatchTable,
        side: usize,
        foe_side: usize,
        att_idx: usize,
        def_idx: usize,
        mspec: &CompactMoveSpec,
    ) {
        let is_castle = table.ruleset.is_castle();
        let def_foresight = (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) != 0;
        let (type_mod, is_immune) =
            table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
        if is_immune || self.mons[att_idx].hp() <= 0 {
            return;
        }
        let atk = self.active_stat(table, side, stat_idx::ATK, false, false);
        let def = self.active_stat(table, foe_side, stat_idx::DEF, false, false);
        let att_item = table.item_spec(self.mons[att_idx].item);
        let boost_type = if is_castle {
            att_item.castle_boost_type
        } else {
            att_item.boost_type
        };
        let has_type_boost = mspec.move_type != NO_TYPE_ID && boost_type == mspec.move_type;
        let weather = if self.weather_turns > 0 {
            self.weather_kind
        } else {
            CompactWeather::None
        };
        let sched = if is_castle {
            get_damage_base_castle(
                table.mons[att_idx].level as i32,
                mspec.base_power as i32,
                atk,
                def,
                true,
                false,
                false,
                has_type_boost,
                weather,
                false,
                false,
                false,
                mspec.move_type,
                table.mons[att_idx].type_mask,
                type_mod,
                false,
                1,
                1,
            )
        } else {
            get_damage_base_branchless(
                table.mons[att_idx].level as i32,
                mspec.base_power as i32,
                atk,
                def,
                true,
                false,
                false,
                has_type_boost,
                weather,
                false,
                false,
                false,
                mspec.move_type,
                table.mons[att_idx].type_mask,
                type_mod,
                false,
            )
        };
        let divisor = if is_castle { 4 } else { 8 };
        let recoil = (sched / divisor).max(1).min(9999) as i16;
        self.mons[att_idx].apply_damage_branchless(recoil, true);
    }

    #[inline]
    fn roll_crit_hit(
        &self,
        table: &MatchTable,
        att_idx: usize,
        mspec: &CompactMoveSpec,
        rng: &mut Prng,
    ) -> bool {
        if mspec.special_kind == SpecialMoveKind::FlailReversal {
            return false;
        }
        let att_item = table.item_spec(self.mons[att_idx].item);
        let spec = &table.mons[att_idx];
        let mut c_stage = mspec.crit_ratio as usize;
        if att_item.is_scope_lens {
            c_stage += 1;
        }
        if (self.mons[att_idx].ext_flags & ext_vol_bit::FOCUS_ENERGY) != 0 {
            c_stage += 1;
        }
        if ((spec.species_flags & 1) != 0 && att_item.is_stick)
            || ((spec.species_flags & 2) != 0 && att_item.is_lucky_punch)
        {
            c_stage = c_stage.max(3);
        }
        let c_stage = c_stage.min(5);
        if c_stage == 0 {
            return false;
        }
        if table.ruleset.is_castle() {
            // Castle #23: base crit stage 1 = 17/256; stages 2..5 = 32, 64, 85, 128 / 256.
            const CASTLE_CRIT_NUM: [u32; 6] = [0, 17, 32, 64, 85, 128];
            rng.random_chance(CASTLE_CRIT_NUM[c_stage], 256)
        } else {
            const CRIT_DEN: [u32; 6] = [0, 16, 8, 4, 3, 2];
            rng.random_chance(1, CRIT_DEN[c_stage])
        }
    }

    #[inline]
    fn abort_move_cleanup(&mut self, table: &MatchTable, side: usize, att_idx: usize) {
        self.mons[att_idx].vol_flags &= !vol_bit::DESTINY_BOND;
        // Outrage/Rollout locks always end on abort (without confusion).
        // Castle #49②: Dig/Fly semi-invulnerability persists if unable to act on turn 2,
        // WITHOUT keeping `two_turn_locked` (the user is free to switch or pick another move)!
        if table.ruleset.is_castle() {
            self.mons[att_idx].ext_flags &=
                !(ext_vol_bit::LOCKED_OUTRAGE | ext_vol_bit::LOCKED_ROLLOUT);
            self.mons[att_idx].ext_timers[ext_timer_idx::LOCKED_TURNS] = 0;
            self.mons[att_idx].set_rollout_count(0);
            self.mons[att_idx].set_two_turn_locked(false);
            self.mons[att_idx].set_locked_move_slot(0);
        } else {
            self.mons[att_idx].clear_multi_turn_locks();
        }
        self.exec_after_move_self(table, side);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn exec_move_inner(
        &mut self,
        table: &MatchTable,
        side: usize,
        move_slot: usize,
        moved_second: bool,
        is_struggle: bool,
        pursuit_switch_double: bool,
        rng: &mut Prng,
    ) {
        let is_castle = table.ruleset.is_castle();
        let foe_side = side ^ 1;
        let att_idx = side * 3 + self.sides[side].active_idx as usize;
        let def_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
        if self.mons[att_idx].hp() <= 0 || self.mons[def_idx].hp() <= 0 {
            return;
        }

        // Hyper Beam recharge turn wastes the turn (`onBeforeMovePriority = 11`, Condition subOrder 2).
        if self.mons[att_idx].must_recharge() {
            self.mons[att_idx].set_must_recharge(false);
            self.mons[att_idx].vol_flags &= !vol_bit::DESTINY_BOND;
            self.exec_after_move_self(table, side);
            return;
        }

        let num_moves = table.mons[att_idx].num_moves as usize;
        let is_locked_cont = self.mons[att_idx].is_multi_turn_locked();
        let mut move_slot = move_slot;
        if is_locked_cont {
            move_slot = self.mons[att_idx].locked_move_slot() as usize;
        } else if !is_struggle && (self.mons[att_idx].vol_flags & vol_bit::ENCORE) != 0 {
            let enc_slot = self.mons[att_idx].encore_slot() as usize;
            if enc_slot < num_moves {
                move_slot = enc_slot;
            }
        }
        let mut mspec = if is_struggle {
            CompactMoveSpec {
                // Castle #42: Struggle gets Normal STAB (1.5x) & Pink Bow (1.1x) while still hitting neutrally.
                move_type: if is_castle { table.normal_type } else { NO_TYPE_ID },
                base_power: 50,
                base_acc_pct: 100,
                crit_ratio: 1,
                is_physical: true,
                blocked_by_protect: true,
                recoil_quarter: true,
                multihit_mean_x2: 2,
                multihit_min: 1,
                multihit_max: 1,
                sec_boost: (NO_STAT_IDX, 0),
                boosts: [(NO_STAT_IDX, 0); 3],
                ..CompactMoveSpec::default()
            }
        } else if move_slot < num_moves {
            table.move_specs[att_idx][move_slot]
        } else {
            self.mons[att_idx].vol_flags &= !vol_bit::DESTINY_BOND;
            self.exec_after_move_self(table, side);
            return;
        };

        // Castle #36②: if the chosen move was drained to 0 PP (e.g. by faster Spite),
        // Mystery Berry triggers right before BeforeMove / PP deduction (with no maxpp clamp).
        if is_castle
            && !is_struggle
            && !is_locked_cont
            && move_slot < num_moves
            && self.mons[att_idx].pp[move_slot] == 0
            && table.item_spec(self.mons[att_idx].item).is_mystery_berry
        {
            self.mons[att_idx].pp[move_slot] = 5;
            self.mons[att_idx].item = 0;
        }

        // Status Berry `onBeforeMove` (Mint/Miracle + single-status berries).
        if is_castle {
            self.check_status_berry_castle(table, side);
        } else {
            self.check_status_berry_trigger(table, side);
        }
        let st_raw = self.mons[att_idx].status_raw();

        // Freeze check (`onBeforeMovePriority = 10`; Flame Wheel / Sacred Fire bypass freeze and thaw on hit).
        if st_raw == Status::Frz as u8 && !mspec.is_defrost {
            self.abort_move_cleanup(table, side, att_idx);
            return;
        }

        // Sleep check (`onBeforeMovePriority = 10`, BEFORE Flinch priority 8).
        let mut sleeping = false;
        if st_raw == Status::Slp as u8 {
            let rem = self.mons[att_idx].status_counter().saturating_sub(1);
            if rem == 0 {
                self.mons[att_idx].set_status(Status::None as u8, 0);
                // Natural wake-up clears Nightmare unless Castle Miracle Nightmare bug was already triggered.
                if (self.mons[att_idx].ext_flags & ext_vol_bit::MIRACLE_NIGHTMARE) == 0 {
                    self.mons[att_idx].vol_flags &= !vol_bit::NIGHTMARE;
                }
            } else {
                self.mons[att_idx].set_status(Status::Slp as u8, rem);
                if !mspec.is_sleep_usable {
                    self.abort_move_cleanup(table, side, att_idx);
                    return;
                }
                sleeping = true;
            }
        }

        // Flinch check (`onBeforeMovePriority = 8`, after Freeze/Sleep and before Disable/Confusion/Attract/Par).
        if (self.mons[att_idx].vol_flags & vol_bit::FLINCH) != 0 {
            self.abort_move_cleanup(table, side, att_idx);
            return;
        }

        // Disable check (`onBeforeMovePriority = 7`, after Flinch and before Confusion/Attract/Par).
        // A queued move disabled mid-turn (or Castle Encore+Disable) is aborted without morphing into Struggle.
        if !is_struggle && !is_locked_cont {
            let is_disabled = (self.mons[att_idx].vol_flags & vol_bit::DISABLE) != 0;
            let dis_slot = self.mons[att_idx].disable_slot() as usize;
            if is_disabled && move_slot == dis_slot {
                self.abort_move_cleanup(table, side, att_idx);
                return;
            }
        }

        // Confusion check.
        if (self.mons[att_idx].vol_flags & vol_bit::CONFUSION) != 0 {
            let rem = self.mons[att_idx].timers[timer_idx::CONFUSION].saturating_sub(1);
            self.mons[att_idx].timers[timer_idx::CONFUSION] = rem;
            if rem == 0 {
                self.mons[att_idx].vol_flags &= !vol_bit::CONFUSION;
            } else if rng.random_chance(1, 2) {
                let lvl = table.mons[att_idx].level as i32;
                let self_dmg = if is_castle {
                    // Castle #24 & #38 (`Probe 10A/11B`): random roll 217..=255, NO Thick Club,
                    // own Reflect applies, hits own Sub, Focus Band 30/256 applies on main HP!
                    let mon = &self.mons[att_idx];
                    let spec = &table.mons[att_idx];
                    let atk = get_stat_castle(
                        spec.stored_stats[stat_idx::ATK] as i32,
                        mon.boosts[stat_idx::ATK],
                        stat_idx::ATK,
                        false,
                        false,
                        false,
                        mon.status_raw() == Status::Brn as u8,
                        false,
                        false,
                        ItemStatKind::None,
                        table.ruleset.overflow_mode(),
                    );
                    let def = self.active_stat(table, side, stat_idx::DEF, false, false);
                    let base = get_damage_base_castle(
                        lvl,
                        40,
                        atk,
                        def,
                        true,
                        mspec.is_selfdestruct,
                        false,
                        false,
                        CompactWeather::None,
                        false,
                        false,
                        false,
                        NO_TYPE_ID,
                        0,
                        0,
                        false,
                        1,
                        1,
                    );
                    let roll = 217 + rng.random(39);
                    let mut d = apply_roll_castle(base, roll, false).min(9999) as i16;
                    let hitting_sub = self.mons[att_idx].sub_hp() > 0;
                    if !hitting_sub
                        && d >= self.mons[att_idx].hp()
                        && table.item_spec(self.mons[att_idx].item).is_focus_band
                        && rng.random_chance(30, 256)
                    {
                        d = (self.mons[att_idx].hp() - 1).max(0);
                    }
                    d
                } else {
                    let atk = self.active_stat(table, side, stat_idx::ATK, false, false);
                    let def = self.active_stat(table, side, stat_idx::DEF, false, false);
                    get_damage_base_branchless(
                        lvl,
                        40,
                        atk,
                        def,
                        true,
                        mspec.is_selfdestruct,
                        false,
                        false,
                        CompactWeather::None,
                        false,
                        false,
                        false,
                        NO_TYPE_ID,
                        0,
                        0,
                        false,
                    )
                    .clamp(1, 9999) as i16
                };
                self.mons[att_idx].apply_damage_branchless(self_dmg, !is_castle);
                self.abort_move_cleanup(table, side, att_idx);
                return;
            }
        }

        // Attract 50% immobilization check (`onBeforeMovePriority = 2`).
        if (self.mons[att_idx].ext_flags & ext_vol_bit::ATTRACT) != 0 && rng.random_chance(1, 2) {
            self.abort_move_cleanup(table, side, att_idx);
            return;
        }

        // Paralysis 25% full-paralyze check.
        if self.mons[att_idx].status_raw() == Status::Par as u8 && rng.random_chance(1, 4) {
            self.abort_move_cleanup(table, side, att_idx);
            return;
        }

        // Clear Destiny Bond when the user attempts a move.
        self.mons[att_idx].vol_flags &= !vol_bit::DESTINY_BOND;

        // Reset Fury Cutter streak and Rage counter when choosing a different move.
        if mspec.special_kind != SpecialMoveKind::FuryCutter {
            self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER] = 0;
        }
        if mspec.special_kind != SpecialMoveKind::Rage {
            self.mons[att_idx].ext_flags &= !(ext_vol_bit::RAGE | ext_vol_bit::RAGE_ARMED);
            self.mons[att_idx].ext_timers[ext_timer_idx::RAGE_COUNTER] = 0;
        }
        // Castle #39 (`PokeStatePlace.java:461`): choosing Sleep Talk resets Protect/Endure stall counter.
        if is_castle && mspec.special_kind == SpecialMoveKind::SleepTalk {
            self.mons[att_idx].set_stall_counter_ext(0);
        }

        // Deduct PP and record last executed move slot (Struggle and multi-turn continuations spend 0 PP).
        if is_struggle {
            self.mons[att_idx].clear_last_move_slot();
        } else if !is_locked_cont {
            if self.mons[att_idx].pp[move_slot] == 0 {
                return;
            }
            self.mons[att_idx].pp[move_slot] -= 1;
            self.mons[att_idx].set_last_move_slot(move_slot as u8);
            // PS Mystery Berry `onUpdate`: immediately restores +5 PP clamped to maxpp on reaching 0 PP.
            if !is_castle
                && self.mons[att_idx].pp[move_slot] == 0
                && table.item_spec(self.mons[att_idx].item).is_mystery_berry
            {
                let max_pp = table.mons[att_idx].max_pp[move_slot];
                self.mons[att_idx].pp[move_slot] = 5u8.min(max_pp);
                self.mons[att_idx].item = 0;
            }
        }

        // Sleep-usable moves (Sleep Talk, Snore) fail when the user is awake.
        if mspec.is_sleep_usable && !sleeping {
            self.exec_after_move_self(table, side);
            return;
        }

        // Sleep Talk: randomly call one of the user's other sleep-talkable moves.
        let mut called_by_sleeptalk = false;
        if mspec.special_kind == SpecialMoveKind::SleepTalk {
            let mut candidates = [0usize; 4];
            let mut n_cand = 0usize;
            for m_i in 0..num_moves {
                if m_i != move_slot && table.move_specs[att_idx][m_i].can_sleep_talk {
                    candidates[n_cand] = m_i;
                    n_cand += 1;
                }
            }
            if n_cand == 0 {
                self.exec_after_move_self(table, side);
                return;
            }
            let picked = candidates[rng.random(n_cand as u32) as usize];
            mspec = table.move_specs[att_idx][picked];
            called_by_sleeptalk = true;
        }

        // Two-turn semi-invulnerable moves (Dig / Fly): Turn 1 enters charge state, Turn 2 strikes.
        if mspec.special_kind == SpecialMoveKind::Dig {
            if (self.mons[att_idx].ext_flags & ext_vol_bit::DIGGING) == 0 {
                self.mons[att_idx].ext_flags |= ext_vol_bit::DIGGING;
                self.mons[att_idx].set_two_turn_locked(true);
                self.mons[att_idx].set_locked_move_slot(move_slot as u8);
                self.exec_after_move_self(table, side);
                return;
            }
            self.mons[att_idx].ext_flags &= !ext_vol_bit::DIGGING;
            self.mons[att_idx].set_two_turn_locked(false);
            if !self.mons[att_idx].is_multi_turn_locked() {
                self.mons[att_idx].set_locked_move_slot(0);
            }
        } else if mspec.special_kind == SpecialMoveKind::Fly {
            if (self.mons[att_idx].ext_flags & ext_vol_bit::FLYING) == 0 {
                self.mons[att_idx].ext_flags |= ext_vol_bit::FLYING;
                self.mons[att_idx].set_two_turn_locked(true);
                self.mons[att_idx].set_locked_move_slot(move_slot as u8);
                self.exec_after_move_self(table, side);
                return;
            }
            self.mons[att_idx].ext_flags &= !ext_vol_bit::FLYING;
            self.mons[att_idx].set_two_turn_locked(false);
            if !self.mons[att_idx].is_multi_turn_locked() {
                self.mons[att_idx].set_locked_move_slot(0);
            }
        }

        // Self-Destruct faints user upfront (HP = 0, Self-KO tracker), but preserves
        // `status_packed` / `vol_flags` / `boosts` until damage calc completes so Burn
        // Atk halving and boosts apply in both PS and Castle, then canonicalizes via `mark_fainted()`.
        if mspec.is_selfdestruct {
            self.mons[att_idx].hp_pool[0] = 0;
            self.sides[side].last_move_selfko = move_slot as u8;
            self.sides[foe_side].last_move_selfko = u8::MAX;
        }

        // Handle common status self-moves (Rest, Recover/Soft-Boiled, Weather heals, Protect, Substitute).
        if mspec.is_rest {
            let maxhp = table.mons[att_idx].maxhp;
            if self.mons[att_idx].hp() < maxhp {
                self.mons[att_idx].hp_pool[0] = maxhp;
                self.mons[att_idx].set_status(Status::Slp as u8, 3);
                self.mons[att_idx].vol_flags &=
                    !(vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP | vol_bit::NIGHTMARE);
                self.mons[att_idx].ext_flags &= !ext_vol_bit::MIRACLE_NIGHTMARE;
                // Castle #32②: first-mover Rest defers Mint/Miracle Berry until right after second mover acts.
                if is_castle && !moved_second {
                    self.bp_switch_flags |= 1u8 << (6 + side);
                }
            }
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.is_half_heal {
            let maxhp = table.mons[att_idx].maxhp;
            // Castle #27 (`Probe 9A`): strict floor(maxhp / 2) vs PS round(maxhp / 2).
            let heal = if is_castle {
                (maxhp / 2).max(1)
            } else {
                (maxhp + 1) / 2
            };
            self.mons[att_idx].hp_pool[0] = (self.mons[att_idx].hp() + heal).min(maxhp);
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.special_kind == SpecialMoveKind::WeatherHeal {
            let maxhp = table.mons[att_idx].maxhp;
            let weather = if self.weather_turns > 0 {
                self.weather_kind
            } else {
                CompactWeather::None
            };
            let heal = weather.weather_heal_amount(maxhp);
            self.mons[att_idx].hp_pool[0] = (self.mons[att_idx].hp() + heal).min(maxhp);
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.is_protect || mspec.is_endure {
            if is_castle {
                // Castle #39 & #40 (`Probe 6/8C/9F`):
                // - Endure SUCCEEDS behind own Substitute; Protect/Detect fail behind own Sub.
                // - Consecutive success table out of 256: [255, 127, 63, 31, 15, 7, 3, 1, 0].
                if !moved_second && (mspec.is_endure || self.mons[att_idx].sub_hp() == 0) {
                    const CASTLE_STALL_NUM: [u32; 9] = [255, 127, 63, 31, 15, 7, 3, 1, 0];
                    let sc = self.mons[att_idx].stall_counter_ext().min(8);
                    let num = CASTLE_STALL_NUM[sc as usize];
                    let ok = num > 0 && rng.random_chance(num, 256);
                    if ok {
                        let bit = if mspec.is_protect {
                            vol_bit::PROTECT
                        } else {
                            vol_bit::ENDURE
                        };
                        self.mons[att_idx].vol_flags |= bit;
                        self.mons[att_idx].set_stall_counter_ext((sc + 1).min(8));
                    } else {
                        self.mons[att_idx].set_stall_counter_ext(0);
                    }
                }
            } else {
                // In Gen 2 PS, Protect/Detect/Endure always fail if the user has a Substitute up.
                if !moved_second && self.mons[att_idx].sub_hp() == 0 {
                    let sc = self.mons[att_idx].stall_counter();
                    let ok =
                        sc == 0 || rng.random_chance((127u32 >> (sc - 1).min(6)).max(1), 255);
                    if ok {
                        let bit = if mspec.is_protect {
                            vol_bit::PROTECT
                        } else {
                            vol_bit::ENDURE
                        };
                        self.mons[att_idx].vol_flags |= bit;
                        self.mons[att_idx].set_stall_counter((sc + 1).min(7));
                    }
                }
            }
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.is_substitute {
            let cost = table.mons[att_idx].maxhp / 4;
            if self.mons[att_idx].hp() > cost && self.mons[att_idx].sub_hp() == 0 {
                self.mons[att_idx].hp_pool[0] -= cost;
                self.mons[att_idx].hp_pool[1] = cost;
                self.mons[att_idx].vol_flags |= vol_bit::SUBSTITUTE;
                self.mons[att_idx].set_partially_trapped(false);
            }
            self.exec_after_move_self(table, side);
            return;
        }

        // Target Protect check (only for moves flagged `protect` in Gen 2).
        if mspec.blocked_by_protect && (self.mons[def_idx].vol_flags & vol_bit::PROTECT) != 0 {
            if mspec.special_kind == SpecialMoveKind::Rollout {
                self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCKED_ROLLOUT;
                self.mons[att_idx].set_rollout_count(0);
                if !self.mons[att_idx].is_multi_turn_locked() {
                    self.mons[att_idx].set_locked_move_slot(0);
                }
            }
            if mspec.special_kind == SpecialMoveKind::FuryCutter {
                self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER] = 0;
            }
            if mspec.special_kind == SpecialMoveKind::CrashKick {
                self.apply_crash_kick_recoil(table, side, foe_side, att_idx, def_idx, &mspec);
            }
            if mspec.is_selfdestruct {
                self.mons[att_idx].mark_fainted();
            }
            self.exec_after_move_self(table, side);
            return;
        }

        let def_foresight = (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) != 0;
        let has_lock_on = (self.mons[att_idx].ext_flags & ext_vol_bit::LOCK_ON) != 0;
        let def_digging = (self.mons[def_idx].ext_flags & ext_vol_bit::DIGGING) != 0;
        let def_flying = (self.mons[def_idx].ext_flags & ext_vol_bit::FLYING) != 0;

        // Semi-invulnerable target check (Dig / Fly) for target-directed moves.
        let is_non_ghost_curse = mspec.special_kind == SpecialMoveKind::Curse
            && (table.mons[att_idx].type_mask & (1u32 << (table.ghost_type & 31))) == 0;
        let is_noop_self_status = mspec.is_status
            && mspec.special_kind == SpecialMoveKind::None
            && mspec.primary_status == 0
            && mspec.primary_vol == 0
            && mspec.boosts[0].0 == NO_STAT_IDX;
        let targets_foe = !mspec.boosts_self
            && !is_non_ghost_curse
            && !is_noop_self_status
            && !matches!(
                mspec.special_kind,
                SpecialMoveKind::Reflect
                    | SpecialMoveKind::LightScreen
                    | SpecialMoveKind::Safeguard
                    | SpecialMoveKind::Spikes
                    | SpecialMoveKind::BellyDrum
                    | SpecialMoveKind::BatonPass
                    | SpecialMoveKind::PerishSong
                    | SpecialMoveKind::DestinyBond
                    | SpecialMoveKind::HealBell
                    | SpecialMoveKind::RainDance
                    | SpecialMoveKind::SunnyDay
                    | SpecialMoveKind::Sandstorm
                    | SpecialMoveKind::Haze
                    | SpecialMoveKind::Mist
                    | SpecialMoveKind::FocusEnergy
                    | SpecialMoveKind::DefenseCurl
                    | SpecialMoveKind::Minimize
                    | SpecialMoveKind::FutureSight
            );
        if targets_foe && (def_digging || def_flying) {
            let lock_on_blocked_move = matches!(
                mspec.special_kind,
                SpecialMoveKind::Attract | SpecialMoveKind::Curse | SpecialMoveKind::MeanLook
            ) || (mspec.primary_vol & (vol_bit::FORESIGHT | vol_bit::NIGHTMARE)) != 0;
            let can_hit_invuln = if def_digging {
                if matches!(
                    mspec.special_kind,
                    SpecialMoveKind::Earthquake | SpecialMoveKind::Magnitude
                ) || mspec.is_fissure
                {
                    true
                } else if is_castle && mspec.special_kind == SpecialMoveKind::MeanLook {
                    true
                } else if lock_on_blocked_move {
                    false
                } else {
                    !is_castle && has_lock_on
                }
            } else {
                // def_flying
                if matches!(
                    mspec.special_kind,
                    SpecialMoveKind::Thunder | SpecialMoveKind::GustTwister
                ) || mspec.is_whirlwind
                {
                    true
                } else if matches!(
                    mspec.special_kind,
                    SpecialMoveKind::Earthquake | SpecialMoveKind::Magnitude
                ) || mspec.is_fissure
                {
                    false
                } else if is_castle && mspec.special_kind == SpecialMoveKind::MeanLook {
                    true
                } else if lock_on_blocked_move {
                    false
                } else {
                    has_lock_on
                }
            };
            if !can_hit_invuln {
                // Castle #43: Rapid Spin clears Spikes, Leech Seed, and partial traps even on invuln miss.
                if is_castle
                    && mspec.special_kind == SpecialMoveKind::RapidSpin
                    && self.mons[att_idx].hp() > 0
                {
                    self.sides[side].side_flags &= !0x03;
                    self.mons[att_idx].vol_flags &= !vol_bit::LEECH_SEED;
                    self.mons[att_idx].set_partially_trapped(false);
                }
                if is_castle && has_lock_on {
                    self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCK_ON;
                    self.mons[att_idx].set_lock_on_turns(0);
                }
                if mspec.special_kind == SpecialMoveKind::Rollout {
                    self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCKED_ROLLOUT;
                    self.mons[att_idx].set_rollout_count(0);
                    if !self.mons[att_idx].is_multi_turn_locked() {
                        self.mons[att_idx].set_locked_move_slot(0);
                    }
                }
                if mspec.special_kind == SpecialMoveKind::FuryCutter {
                    self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER] = 0;
                }
                if mspec.special_kind == SpecialMoveKind::CrashKick {
                    self.apply_crash_kick_recoil(table, side, foe_side, att_idx, def_idx, &mspec);
                }
                if mspec.is_selfdestruct {
                    self.mons[att_idx].mark_fainted();
                }
                self.exec_after_move_self(table, side);
                return;
            }
        }

        // Accuracy check (Triple Kick checks accuracy per hit inside its loop; Ohko uses level formula).
        if mspec.special_kind == SpecialMoveKind::Ohko {
            // Type immunity runs before accuracy / Lock-On consume / ovfl=1 Counter arming.
            let (_, is_immune) =
                table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
            if is_immune {
                self.exec_after_move_self(table, side);
                return;
            }
            let ohko_hit = if !is_castle && has_lock_on {
                true
            } else {
                let att_lvl = table.mons[att_idx].level as i32;
                let def_lvl = table.mons[def_idx].level as i32;
                if att_lvl < def_lvl {
                    // Fails before Castle Lock-On consumption or ovfl=1 Counter arming (`moveexec.rs:2308`).
                    self.exec_after_move_self(table, side);
                    return;
                }
                let mut acc = (76 + (att_lvl - def_lvl) * 2).min(255) as f64;
                if is_castle {
                    // 2025/05/27 Castle PBS spec & `moveexec.rs:2295-2345`: apply acc/eva/Foresight/BrightPowder
                    // in BOTH `ovfl=0` and `ovfl=1`!
                    let mut acc_boost = if mspec.ignore_accuracy {
                        0
                    } else {
                        self.mons[att_idx].boosts[stat_idx::ACC].clamp(-6, 6)
                    };
                    let mut eva_boost = if mspec.ignore_evasion {
                        0
                    } else {
                        self.mons[def_idx].boosts[stat_idx::EVA].clamp(-6, 6)
                    };
                    if mspec.ignore_positive_evasion && eva_boost > 0 {
                        eva_boost = 0;
                    }
                    if def_foresight {
                        if acc_boost < eva_boost {
                            acc_boost = 0;
                            eva_boost = 0;
                        } else if eva_boost > 0 {
                            eva_boost = 0;
                        }
                    }
                    if acc_boost > 0 {
                        acc *= (3 + acc_boost as i32) as f64 / 3.0;
                    } else if acc_boost < 0 {
                        acc *= 3.0 / (3 - acc_boost as i32) as f64;
                    }
                    if eva_boost > 0 {
                        acc *= 3.0 / (3 + eva_boost as i32) as f64;
                    } else if eva_boost < 0 {
                        acc *= (3 - eva_boost as i32) as f64 / 3.0;
                    }
                    // Castle #45: Lock-On does NOT make OHKO auto-hit, and IS consumed on attempt.
                    if has_lock_on {
                        self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCK_ON;
                        self.mons[att_idx].set_lock_on_turns(0);
                    }
                } else {
                    const POS_BOOST: [f64; 7] = [1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0];
                    const NEG_BOOST: [f64; 7] = [1.0, 0.75, 0.6, 0.5, 0.43, 0.36, 0.33];
                    if !mspec.ignore_accuracy {
                        let boost = self.mons[att_idx].boosts[stat_idx::ACC].clamp(-6, 6);
                        if boost > 0 {
                            acc *= POS_BOOST[boost as usize];
                        } else {
                            acc *= NEG_BOOST[(-boost) as usize];
                        }
                    }
                    if !mspec.ignore_evasion {
                        let boost = self.mons[def_idx].boosts[stat_idx::EVA].clamp(-6, 6);
                        let ignore_pos = mspec.ignore_positive_evasion || def_foresight;
                        if boost > 0 && !ignore_pos {
                            acc *= NEG_BOOST[boost as usize];
                        } else if boost < 0 {
                            acc *= POS_BOOST[(-boost) as usize];
                        }
                    }
                }
                let mut thresh = acc.floor().clamp(1.0, 255.0) as u8;
                if table.item_spec(self.mons[def_idx].item).is_brightpowder {
                    thresh = thresh.saturating_sub(20);
                }
                thresh >= 255 || rng.random_chance(thresh as u32, 256)
            };
            if !ohko_hit {
                // Castle #18 (`Probe 8D1`): ovfl=1 missed Fissure / Horn Drill arms instant-KO Counter!
                if is_castle && table.ruleset.overflow_mode() && mspec.is_fissure_or_horndrill {
                    self.turn_extra_flags |= 1u8 << (6 + side);
                }
                self.exec_after_move_self(table, side);
                return;
            }
        } else if !mspec.always_hit && mspec.special_kind != SpecialMoveKind::FutureSight {
            let thunder_rain = mspec.special_kind == SpecialMoveKind::Thunder
                && self.weather_turns > 0
                && self.weather_kind == CompactWeather::Rain;
            let eff_base_acc = if mspec.special_kind == SpecialMoveKind::Thunder
                && self.weather_turns > 0
                && self.weather_kind == CompactWeather::Sun
            {
                50
            } else {
                mspec.base_acc_pct
            };
            let hit = if has_lock_on || thunder_rain {
                true
            } else if is_castle {
                let thresh = lookup_accuracy_threshold_castle(
                    eff_base_acc,
                    self.mons[att_idx].boosts[5],
                    self.mons[def_idx].boosts[6],
                    mspec.ignore_accuracy,
                    mspec.ignore_evasion,
                    def_foresight,
                    table.item_spec(self.mons[def_idx].item).is_brightpowder,
                );
                thresh >= 255 || rng.random_chance(thresh as u32, 256)
            } else {
                let thresh = lookup_accuracy_threshold(
                    eff_base_acc,
                    self.mons[att_idx].boosts[5],
                    self.mons[def_idx].boosts[6],
                    mspec.ignore_accuracy,
                    mspec.ignore_evasion,
                    mspec.ignore_positive_evasion | def_foresight,
                    table.item_spec(self.mons[def_idx].item).is_brightpowder,
                );
                thresh >= 255 || rng.random_chance(thresh as u32, 256)
            };
            // Castle #45② (`Probe 12E1`): Lock-On is consumed by an accuracy-checked move (not Splash/status).
            if is_castle && has_lock_on {
                self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCK_ON;
                self.mons[att_idx].set_lock_on_turns(0);
            }
            if !hit {
                // Castle #43: Rapid Spin clears Spikes, Leech Seed, and partial traps even on miss!
                if is_castle
                    && mspec.special_kind == SpecialMoveKind::RapidSpin
                    && self.mons[att_idx].hp() > 0
                {
                    self.sides[side].side_flags &= !0x03;
                    self.mons[att_idx].vol_flags &= !vol_bit::LEECH_SEED;
                    self.mons[att_idx].set_partially_trapped(false);
                }
                if mspec.special_kind == SpecialMoveKind::Rollout {
                    self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCKED_ROLLOUT;
                    self.mons[att_idx].set_rollout_count(0);
                    if !self.mons[att_idx].is_multi_turn_locked() {
                        self.mons[att_idx].set_locked_move_slot(0);
                    }
                }
                if mspec.special_kind == SpecialMoveKind::FuryCutter {
                    self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER] = 0;
                }
                if mspec.special_kind == SpecialMoveKind::CrashKick {
                    self.apply_crash_kick_recoil(table, side, foe_side, att_idx, def_idx, &mspec);
                }
                if mspec.special_kind == SpecialMoveKind::OutrageLock
                    && !called_by_sleeptalk
                    && (self.mons[att_idx].ext_flags & ext_vol_bit::LOCKED_OUTRAGE) == 0
                {
                    self.mons[att_idx].ext_flags |= ext_vol_bit::LOCKED_OUTRAGE;
                    self.mons[att_idx].set_locked_move_slot(move_slot as u8);
                    self.mons[att_idx].ext_timers[ext_timer_idx::LOCKED_TURNS] = if is_castle {
                        (1 + rng.random(2)) as u8
                    } else {
                        (2 + rng.random(2)) as u8
                    };
                }
                if mspec.is_selfdestruct {
                    self.mons[att_idx].mark_fainted();
                }
                self.exec_after_move_self(table, side);
                return;
            }
        }

        if !mspec.is_status
            && mspec.special_kind != SpecialMoveKind::FutureSight
            && (mspec.base_power > 0
                || (is_castle
                    && matches!(mspec.var_bp_kind, VarBpKind::Return | VarBpKind::Frustration))
                || matches!(
                    mspec.special_kind,
                    SpecialMoveKind::Counter
                        | SpecialMoveKind::MirrorCoat
                        | SpecialMoveKind::FlailReversal
                        | SpecialMoveKind::LevelDamage
                        | SpecialMoveKind::SuperFang
                        | SpecialMoveKind::Present
                        | SpecialMoveKind::BeatUp
                        | SpecialMoveKind::Psywave
                        | SpecialMoveKind::Ohko
                        | SpecialMoveKind::Magnitude
                        | SpecialMoveKind::TripleKick
                )
                || mspec.fixed_damage != 0)
        {
            let def_had_dbond = (self.mons[def_idx].vol_flags & vol_bit::DESTINY_BOND) != 0;
            let def_had_sub = self.mons[def_idx].sub_hp() > 0 && !mspec.bypasses_sub;
            // In Gen 2, draining moves (Giga Drain, Mega Drain, Absorb, Leech Life, Dream Eater)
            // always miss against Substitute; Dream Eater also requires target to be asleep.
            if (mspec.drain_half && def_had_sub)
                || (mspec.special_kind == SpecialMoveKind::DreamEater
                    && self.mons[def_idx].status_raw() != Status::Slp as u8)
            {
                self.exec_after_move_self(table, side);
                return;
            }
            let def_endure = (self.mons[def_idx].vol_flags & vol_bit::ENDURE) != 0 && !def_had_sub;
            let def_has_focus_band = table.item_spec(self.mons[def_idx].item).is_focus_band;
            let att_item = table.item_spec(self.mons[att_idx].item);
            let mut dealt = 0i16;
            let mut main_dealt = 0i16;
            let mut false_swipe_counter_override = 0u16;
            let mut hit_landed = false;
            let mut hit_main_hp = false;
            let mut immune_hit = false;

            if mspec.special_kind == SpecialMoveKind::Counter {
                // Counter: double last physical damage taken this turn (or Castle ovfl=1 missed OHKO KO).
                let (_, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                let ovfl_ohko_counter = is_castle
                    && table.ruleset.overflow_mode()
                    && (self.turn_extra_flags & (1u8 << (6 + foe_side))) != 0;
                if !is_immune
                    && (ovfl_ohko_counter
                        || (self.hurt_cat_this_turn[side] == 1
                            && self.hurt_damage_this_turn[side] > 0))
                {
                    let mut dmg = if ovfl_ohko_counter {
                        self.mons[def_idx].hp()
                    } else {
                        (self.hurt_damage_this_turn[side] as i32 * 2).min(9999) as i16
                    };
                    if !def_had_sub {
                        if def_endure && dmg >= self.mons[def_idx].hp() {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                    hit_landed = true;
                    if !def_had_sub {
                        hit_main_hp = true;
                        main_dealt = dealt;
                    }
                }
            } else if mspec.special_kind == SpecialMoveKind::MirrorCoat {
                // Mirror Coat: double last special damage taken this turn.
                let (_, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune
                    && self.hurt_cat_this_turn[side] == 2
                    && self.hurt_damage_this_turn[side] > 0
                {
                    let mut dmg = (self.hurt_damage_this_turn[side] as i32 * 2).min(9999) as i16;
                    if !def_had_sub {
                        if def_endure && dmg >= self.mons[def_idx].hp() {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                    hit_landed = true;
                    if !def_had_sub {
                        hit_main_hp = true;
                        main_dealt = dealt;
                    }
                }
            } else if mspec.special_kind == SpecialMoveKind::Ohko {
                let (_, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let mut dmg = if def_had_sub {
                        self.mons[def_idx].sub_hp()
                    } else {
                        self.mons[def_idx].hp()
                    };
                    if !def_had_sub {
                        if def_endure && dmg >= self.mons[def_idx].hp() {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                    hit_landed = true;
                    if !def_had_sub {
                        hit_main_hp = true;
                        main_dealt = dealt;
                    }
                } else {
                    immune_hit = true;
                }
            } else if mspec.special_kind == SpecialMoveKind::LevelDamage || mspec.fixed_damage != 0
            {
                let (_, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let mut dmg = if mspec.fixed_damage > 0 {
                        mspec.fixed_damage
                    } else {
                        table.mons[att_idx].level as i16
                    };
                    if !def_had_sub {
                        if def_endure && dmg >= self.mons[def_idx].hp() {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                    hit_landed = true;
                    if !def_had_sub {
                        hit_main_hp = true;
                        main_dealt = dealt;
                    }
                } else {
                    immune_hit = true;
                }
            } else if mspec.special_kind == SpecialMoveKind::Psywave {
                let (_, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let hi = ((table.mons[att_idx].level as u32 * 3) / 2).max(2);
                    let mut dmg = (1 + rng.random(hi - 1)) as i16;
                    if !def_had_sub {
                        if def_endure && dmg >= self.mons[def_idx].hp() {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                    hit_landed = true;
                    if !def_had_sub {
                        hit_main_hp = true;
                        main_dealt = dealt;
                    }
                } else {
                    immune_hit = true;
                }
            } else if mspec.special_kind == SpecialMoveKind::SuperFang {
                let (_, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let mut dmg = (self.mons[def_idx].hp() / 2).max(1);
                    if !def_had_sub {
                        if def_endure && dmg >= self.mons[def_idx].hp() {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                    hit_landed = true;
                    if !def_had_sub {
                        hit_main_hp = true;
                        main_dealt = dealt;
                    }
                } else {
                    immune_hit = true;
                }
            } else if mspec.special_kind == SpecialMoveKind::Present {
                // Present (Castle v0.3 spec #25 vs PS Gen 2 glitch).
                let p_bp = if is_castle && def_had_sub {
                    40i32
                } else {
                    let r = rng.random(10);
                    if r < 4 {
                        40
                    } else if r < 7 {
                        80
                    } else if r < 8 {
                        120
                    } else {
                        0 // Heal target!
                    }
                };
                if p_bp == 0 {
                    // Heal target (works even on Ghost; Castle = fixed 80 HP, PS = round(maxhp/4)).
                    let def_max = table.mons[def_idx].maxhp;
                    let heal = if is_castle {
                        80i16
                    } else {
                        ((def_max + 2) / 4).max(1)
                    };
                    self.mons[def_idx].hp_pool[0] = (self.mons[def_idx].hp() + heal).min(def_max);
                } else {
                    let (type_mod, is_immune) = table.effectiveness_ext(
                        table.normal_type,
                        &table.mons[def_idx],
                        def_foresight,
                    );
                    if !is_immune {
                        let is_crit = self.roll_crit_hit(table, att_idx, &mspec, rng);
                        let ignore_boosts = is_crit
                            && self.mons[att_idx].boosts[stat_idx::ATK]
                                <= self.mons[def_idx].boosts[stat_idx::DEF];
                        let boost_type = if is_castle {
                            att_item.castle_boost_type
                        } else {
                            att_item.boost_type
                        };
                        let has_pink_bow = boost_type == table.normal_type;
                        let att_t = table.mons[att_idx].present_type_num as f64;
                        let def_t = table.mons[def_idx].present_type_num as f64;
                        let base_dmg = if is_castle {
                            // Castle Present v0.3 (`DamageCalc.java` + `majin_damagepzt.js` + spec #25):
                            // level term = def_type2 * 6; atk = 100/(200-eff) w/ rank/ThickClub/Burn;
                            // def = att_type2 w/ rank/MetalPowder/Reflect; Normal resist effectively 1/3.
                            let eff_pct = match type_mod {
                                -1 => 50.0,
                                -2 => 25.0,
                                _ => 100.0,
                            };
                            let stage_mult = |b: i8| -> f64 {
                                let c = b.clamp(-6, 6);
                                if c >= 0 {
                                    (2 + c) as f64 / 2.0
                                } else {
                                    2.0 / (2 - c) as f64
                                }
                            };
                            let mut p_atk = 100.0 / (200.0 - eff_pct);
                            if !ignore_boosts {
                                p_atk *= stage_mult(self.mons[att_idx].boosts[stat_idx::ATK]);
                            }
                            if att_item.stat_kind == ItemStatKind::ThickClub
                                && table.mons[att_idx].species_stat_kind == ItemStatKind::ThickClub
                            {
                                p_atk *= 2.0;
                            }
                            if p_atk > 999.0 {
                                p_atk = 999.0;
                            }
                            if !ignore_boosts && self.mons[att_idx].status_raw() == Status::Brn as u8
                            {
                                p_atk /= 2.0;
                            }
                            let mut p_def = att_t.max(1.0);
                            if !ignore_boosts {
                                p_def *= stage_mult(self.mons[def_idx].boosts[stat_idx::DEF]);
                            }
                            let def_item = table.item_spec(self.mons[def_idx].item);
                            if def_item.stat_kind == ItemStatKind::MetalPowder
                                && (table.mons[def_idx].species_stat_kind
                                    == ItemStatKind::MetalPowder
                                    || (table.mons[def_idx].species_flags & 4) != 0)
                            {
                                p_def *= 1.5;
                            }
                            if !ignore_boosts && self.sides[foe_side].reflect_turns > 0 {
                                p_def *= 2.0;
                            }
                            if p_def > 999.0 {
                                p_def = 999.0;
                            }
                            if p_atk >= 256.0 || p_def >= 256.0 {
                                p_atk /= 4.0;
                                p_def /= 4.0;
                            }
                            if p_def <= 0.0 {
                                p_def = 1.0;
                            }
                            let mut d =
                                ((def_t * 6.0 * (p_bp as f64) * p_atk / p_def) / 50.0).floor()
                                    as i32;
                            if is_crit {
                                d <<= 1;
                            }
                            if has_pink_bow {
                                d += d / 10;
                            }
                            d = d.clamp(0, 997) + 2;
                            if (table.mons[att_idx].type_mask & (1u32 << (table.normal_type & 31)))
                                != 0
                            {
                                d += d >> 1;
                            }
                            if type_mod < 0 {
                                d >>= if type_mod <= -2 { 2 } else { 1 };
                            }
                            d
                        } else {
                            get_damage_base_branchless(
                                def_t as i32,
                                p_bp,
                                10,
                                (att_t as i32).max(1),
                                true,
                                false,
                                is_crit,
                                has_pink_bow,
                                CompactWeather::None,
                                false,
                                false,
                                false,
                                table.normal_type,
                                table.mons[att_idx].type_mask,
                                type_mod,
                                false,
                            )
                        };
                        let roll = 217 + rng.random(39);
                        let raw_d = if is_castle {
                            apply_roll_castle(base_dmg, roll, false)
                        } else {
                            apply_roll_branchless(base_dmg, roll, false)
                        };
                        let mut hit_dmg = raw_d.min(9999) as i16;
                        if !def_had_sub {
                            if def_endure && hit_dmg >= self.mons[def_idx].hp() {
                                hit_dmg = (self.mons[def_idx].hp() - 1).max(0);
                            } else if def_has_focus_band
                                && hit_dmg >= self.mons[def_idx].hp()
                                && rng.random_chance(30, 256)
                            {
                                hit_dmg = (self.mons[def_idx].hp() - 1).max(0);
                            }
                        }
                        dealt = self.mons[def_idx].apply_damage_branchless(hit_dmg, false);
                        hit_landed = true;
                        if !def_had_sub {
                            hit_main_hp = true;
                            main_dealt = dealt;
                        }
                    } else {
                        immune_hit = true;
                    }
                }
            } else if mspec.special_kind == SpecialMoveKind::BeatUp {
                // Beat Up (Spec #53, `Probe 11D`/`12D2`):
                // PS: alive & status == None; Castle: alive & (status == None || p_i == active_idx).
                let active_slot = self.sides[side].active_idx as usize;
                let def_base_def = table.mons[def_idx].base_def.max(1) as i32;
                let has_endure = (self.mons[def_idx].vol_flags & vol_bit::ENDURE) != 0;
                for &p_u8 in &self.party_order(side) {
                    let p_i = p_u8 as usize;
                    if self.mons[def_idx].hp() <= 0 {
                        break;
                    }
                    if (self.sides[side].alive_mask & (1 << p_i)) == 0 {
                        continue;
                    }
                    let ally_idx = side * 3 + p_i;
                    if self.mons[ally_idx].hp() <= 0 {
                        continue;
                    }
                    let ally_st = self.mons[ally_idx].status_raw();
                    let eligible = if is_castle {
                        ally_st == Status::None as u8 || p_i == active_slot
                    } else {
                        ally_st == Status::None as u8
                    };
                    if !eligible {
                        continue;
                    }
                    let is_crit = self.roll_crit_hit(table, att_idx, &mspec, rng);
                    let ally_lvl = table.mons[ally_idx].level as i32;
                    let ally_atk = table.mons[ally_idx].base_atk.max(1) as i32;
                    let base = if is_castle {
                        get_damage_base_castle(
                            ally_lvl,
                            10,
                            ally_atk,
                            def_base_def,
                            false,
                            false,
                            is_crit,
                            false,
                            CompactWeather::None,
                            false,
                            false,
                            false,
                            NO_TYPE_ID,
                            0,
                            0,
                            false,
                            1,
                            1,
                        )
                    } else {
                        get_damage_base_branchless(
                            ally_lvl,
                            10,
                            ally_atk,
                            def_base_def,
                            false,
                            false,
                            is_crit,
                            false,
                            CompactWeather::None,
                            false,
                            false,
                            false,
                            NO_TYPE_ID,
                            0,
                            0,
                            false,
                        )
                    };
                    let roll = 217 + rng.random(39);
                    let raw = if is_castle {
                        apply_roll_castle(base, roll, false)
                    } else {
                        apply_roll_branchless(base, roll, false)
                    };
                    let mut h_dmg = raw.min(9999) as i16;
                    let hitting_sub = self.mons[def_idx].sub_hp() > 0 && !mspec.bypasses_sub;
                    if !hitting_sub {
                        if has_endure && h_dmg >= self.mons[def_idx].hp() {
                            h_dmg = (self.mons[def_idx].hp() - 1).max(0);
                        } else if def_has_focus_band
                            && h_dmg >= self.mons[def_idx].hp()
                            && rng.random_chance(30, 256)
                        {
                            h_dmg = (self.mons[def_idx].hp() - 1).max(0);
                        }
                    }
                    let h = self.mons[def_idx].apply_damage_branchless(h_dmg, mspec.bypasses_sub);
                    dealt = dealt.saturating_add(h);
                    hit_landed = true;
                    if !hitting_sub {
                        hit_main_hp = true;
                        main_dealt = h;
                    }
                }
            } else if mspec.special_kind == SpecialMoveKind::TripleKick {
                // Triple Kick (Spec #51①): up to 3 hits, each hit checks accuracy;
                // Castle leaks hit count (2 or 3) as damage multiplier onto second mover!
                let (type_mod, is_immune) =
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let boost_type = if is_castle {
                        att_item.castle_boost_type
                    } else {
                        att_item.boost_type
                    };
                    let has_type_boost =
                        mspec.move_type != NO_TYPE_ID && boost_type == mspec.move_type;
                    let weather = if self.weather_turns > 0 {
                        self.weather_kind
                    } else {
                        CompactWeather::None
                    };
                    let has_endure = (self.mons[def_idx].vol_flags & vol_bit::ENDURE) != 0;
                    // `gen2stadium2.json:9985` & `Battle::moveexec.rs:2546`: single outer accuracy check,
                    // then uniform 1..=3 hits!
                    let hits = (1 + rng.random(3)) as u8;
                    let mut hits_done = 0u8;
                    for hit_k in 1..=hits {
                        if self.mons[def_idx].hp() <= 0 {
                            break;
                        }
                        let is_crit = self.roll_crit_hit(table, att_idx, &mspec, rng);
                        let ignore_boosts = is_crit
                            && self.mons[att_idx].boosts[stat_idx::ATK]
                                <= self.mons[def_idx].boosts[stat_idx::DEF];
                        let atk =
                            self.active_stat(table, side, stat_idx::ATK, ignore_boosts, ignore_boosts);
                        let def =
                            self.active_stat(table, foe_side, stat_idx::DEF, ignore_boosts, false);
                        let base = if is_castle {
                            // MoveCalc.java:8238-8283: BP=10, multiply by `hit_k` AFTER type compat & BEFORE roll.
                            get_damage_base_castle(
                                table.mons[att_idx].level as i32,
                                10,
                                atk,
                                def,
                                true,
                                false,
                                is_crit,
                                has_type_boost,
                                weather,
                                false,
                                false,
                                false,
                                mspec.move_type,
                                table.mons[att_idx].type_mask,
                                type_mod,
                                false,
                                1,
                                1,
                            ) * (hit_k as i32)
                        } else {
                            get_damage_base_branchless(
                                table.mons[att_idx].level as i32,
                                10 * (hit_k as i32),
                                atk,
                                def,
                                true,
                                false,
                                is_crit,
                                has_type_boost,
                                weather,
                                false,
                                false,
                                false,
                                mspec.move_type,
                                table.mons[att_idx].type_mask,
                                type_mod,
                                false,
                            )
                        };
                        let roll = 217 + rng.random(39);
                        let raw = if is_castle {
                            apply_roll_castle(base, roll, false)
                        } else {
                            apply_roll_branchless(base, roll, false)
                        };
                        let mut h_dmg = raw.min(9999) as i16;
                        let hitting_sub = self.mons[def_idx].sub_hp() > 0 && !mspec.bypasses_sub;
                        if !hitting_sub {
                            if has_endure && h_dmg >= self.mons[def_idx].hp() {
                                h_dmg = (self.mons[def_idx].hp() - 1).max(0);
                            } else if def_has_focus_band
                                && h_dmg >= self.mons[def_idx].hp()
                                && rng.random_chance(30, 256)
                            {
                                h_dmg = (self.mons[def_idx].hp() - 1).max(0);
                            }
                        }
                        let h = self.mons[def_idx].apply_damage_branchless(h_dmg, false);
                        dealt = dealt.saturating_add(h);
                        hits_done = hit_k;
                        hit_landed = true;
                        if !hitting_sub {
                            hit_main_hp = true;
                            main_dealt = h;
                        }
                    }
                    if is_castle && !moved_second && hits_done >= 2 {
                        self.triple_kick_mult = hits_done;
                    }
                } else {
                    immune_hit = true;
                }
            } else {
                // Standard damaging move, multi-hit (Fixed(2) / Range(2..5) / Twineedle),
                // Rollout, OutrageLock, Pursuit, FuryCutter, Rage, FlailReversal, Magnitude,
                // FalseSwipe, Stomp, Earthquake, GustTwister, Struggle.
                let (atk_stat, def_stat) = if mspec.is_physical {
                    (stat_idx::ATK, stat_idx::DEF)
                } else {
                    (stat_idx::SPA, stat_idx::SPD)
                };

                let mut rollout_mult = 1i32;
                let mut pre_plus2_mult = 1i32;
                let mut post_type_mult = 1i32;
                let mut bp = match mspec.special_kind {
                    SpecialMoveKind::FlailReversal => Self::flail_reversal_bp(
                        self.mons[att_idx].hp(),
                        table.mons[att_idx].maxhp,
                        is_castle,
                    ),
                    SpecialMoveKind::Pursuit => {
                        let castle_bp_double = is_castle
                            && moved_second
                            && (self.bp_switch_flags & (1u8 << (2 + foe_side))) != 0;
                        if pursuit_switch_double || castle_bp_double {
                            mspec.base_power as i32 * 2
                        } else {
                            mspec.base_power as i32
                        }
                    }
                    SpecialMoveKind::Magnitude => {
                        let r = rng.random(20);
                        if r < 1 {
                            10
                        } else if r < 3 {
                            30
                        } else if r < 7 {
                            50
                        } else if r < 13 {
                            70
                        } else if r < 17 {
                            90
                        } else if r < 19 {
                            110
                        } else {
                            150
                        }
                    }
                    SpecialMoveKind::Rollout => {
                        let k = self.mons[att_idx].rollout_count().min(4) as u32;
                        let dcurl =
                            (self.mons[att_idx].ext_flags & ext_vol_bit::DEFENSE_CURL) != 0;
                        let mult = (1i32 << k) * if dcurl { 2 } else { 1 };
                        if is_castle {
                            // Castle #26 (`Probe 12E2`): keep base BP 30; apply mult AFTER +2 & BEFORE STAB/type/roll!
                            rollout_mult = mult;
                            30
                        } else {
                            30 * mult
                        }
                    }
                    SpecialMoveKind::FuryCutter => {
                        let streak = self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER];
                        let sh = if is_castle {
                            streak.min(12) as u32
                        } else {
                            streak.min(4) as u32
                        };
                        10i32 << sh
                    }
                    SpecialMoveKind::Rage => {
                        if is_castle {
                            // Castle #52: power multiplies 1x, 1x, 2x, 3x...
                            let cnt = self.mons[att_idx].ext_timers[ext_timer_idx::RAGE_COUNTER];
                            let mult = if cnt <= 1 { 1i32 } else { cnt as i32 };
                            20 * mult
                        } else {
                            20
                        }
                    }
                    _ => {
                        if is_castle
                            && matches!(
                                mspec.var_bp_kind,
                                VarBpKind::Return | VarBpKind::Frustration
                            )
                        {
                            102
                        } else {
                            mspec.base_power as i32
                        }
                    }
                };

                // Situational 2x multipliers matching `Battle`:
                // - Stomp vs Minimize: `ModifyDamage` (before clamp(997) + 2)
                // - EQ/Mag vs Dig & Gust/Twister vs Fly: `BasePower` (doubles BP before core formula)
                if mspec.special_kind == SpecialMoveKind::Stomp
                    && (self.mons[def_idx].ext_flags & ext_vol_bit::MINIMIZE) != 0
                {
                    pre_plus2_mult *= 2;
                }
                if matches!(
                    mspec.special_kind,
                    SpecialMoveKind::Earthquake | SpecialMoveKind::Magnitude
                ) && def_digging
                {
                    bp *= 2;
                }
                if mspec.special_kind == SpecialMoveKind::GustTwister && def_flying {
                    bp *= 2;
                }
                // Castle #51① (`moveexec.rs:3666-3672`, `MoveCalc.java:8283`):
                // Triple Kick 2x/3x leak applies AFTER weather/STAB/type-effectiveness & BEFORE roll!
                if is_castle
                    && moved_second
                    && self.triple_kick_mult >= 2
                    && !matches!(
                        mspec.special_kind,
                        SpecialMoveKind::Rollout | SpecialMoveKind::FuryCutter
                    )
                {
                    post_type_mult = self.triple_kick_mult as i32;
                }

                // Struggle always hits neutrally (even in Castle where move_type = Normal for STAB/Pink Bow!).
                let (type_mod, is_immune) = if is_struggle {
                    (0i8, false)
                } else {
                    table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight)
                };
                if is_immune {
                    immune_hit = true;
                } else {
                    let weather = if self.weather_turns > 0 {
                        self.weather_kind
                    } else {
                        CompactWeather::None
                    };
                    let boost_type = if is_castle {
                        att_item.castle_boost_type
                    } else {
                        att_item.boost_type
                    };
                    let has_type_boost =
                        mspec.move_type != NO_TYPE_ID && boost_type == mspec.move_type;
                    let no_var = mspec.special_kind == SpecialMoveKind::FlailReversal;

                    let hits = if mspec.multihit_min == 2 && mspec.multihit_max == 5 {
                        let r = rng.random(8);
                        if r < 3 {
                            2usize
                        } else if r < 6 {
                            3
                        } else if r == 6 {
                            4
                        } else {
                            5
                        }
                    } else {
                        (mspec.multihit_min.max(1)) as usize
                    };

                    let has_endure = (self.mons[def_idx].vol_flags & vol_bit::ENDURE) != 0;
                    for hit_i in 0..hits {
                        if self.mons[def_idx].hp() <= 0 {
                            break;
                        }
                        let is_crit = self.roll_crit_hit(table, att_idx, &mspec, rng);
                        let ignore_boosts = is_crit
                            & (self.mons[att_idx].boosts[atk_stat]
                                <= self.mons[def_idx].boosts[def_stat]);
                        let atk = self.active_stat(
                            table,
                            side,
                            atk_stat,
                            ignore_boosts,
                            ignore_boosts,
                        );
                        let def = self.active_stat(
                            table,
                            foe_side,
                            def_stat,
                            ignore_boosts,
                            false,
                        );
                        let base_dmg = if is_castle {
                            let b = get_damage_base_castle(
                                table.mons[att_idx].level as i32,
                                bp,
                                atk,
                                def,
                                mspec.is_physical,
                                mspec.is_selfdestruct,
                                is_crit,
                                has_type_boost,
                                weather,
                                mspec.is_water,
                                mspec.is_fire,
                                mspec.is_solarbeam,
                                mspec.move_type,
                                table.mons[att_idx].type_mask,
                                type_mod,
                                false,
                                rollout_mult,
                                pre_plus2_mult,
                            );
                            if post_type_mult > 1 {
                                b * post_type_mult
                            } else {
                                b
                            }
                        } else {
                            get_damage_base_branchless_ext(
                                table.mons[att_idx].level as i32,
                                bp,
                                atk,
                                def,
                                mspec.is_physical,
                                mspec.is_selfdestruct,
                                is_crit,
                                has_type_boost,
                                weather,
                                mspec.is_water,
                                mspec.is_fire,
                                mspec.is_solarbeam,
                                mspec.move_type,
                                table.mons[att_idx].type_mask,
                                type_mod,
                                false,
                                pre_plus2_mult,
                            )
                        };
                        let roll = if no_var { 255 } else { 217 + rng.random(39) };
                        let raw_dmg = if is_castle {
                            apply_roll_castle(base_dmg, roll, no_var)
                        } else if base_dmg > 0 {
                            apply_roll_branchless(base_dmg, roll, no_var)
                        } else {
                            0
                        };
                        let mut hit_dmg = raw_dmg.min(9999) as i16;
                        let hitting_sub =
                            self.mons[def_idx].sub_hp() > 0 && !mspec.bypasses_sub;
                        if !hitting_sub {
                            let cur_hp = self.mons[def_idx].hp();
                            if mspec.special_kind == SpecialMoveKind::FalseSwipe
                                && hit_dmg >= cur_hp
                            {
                                // Castle #53 (`Probe 11F`/`12C`): False Swipe surviving at 1 HP
                                // reflects `2 * old_hp` on Counter!
                                if is_castle {
                                    false_swipe_counter_override = cur_hp.max(0) as u16;
                                }
                                hit_dmg = (cur_hp - 1).max(0);
                            } else if has_endure && hit_dmg >= cur_hp {
                                hit_dmg = (cur_hp - 1).max(0);
                            } else if def_has_focus_band
                                && hit_dmg >= cur_hp
                                && rng.random_chance(30, 256)
                            {
                                hit_dmg = (cur_hp - 1).max(0);
                            }
                        }
                        let h = self.mons[def_idx]
                            .apply_damage_branchless(hit_dmg, mspec.bypasses_sub);
                        dealt = dealt.saturating_add(h);
                        hit_landed = true;
                        if !hitting_sub {
                            hit_main_hp = true;
                            main_dealt = h;
                            // Rage trigger on hit against surviving main HP.
                            if self.mons[def_idx].hp() > 0
                                && (self.mons[def_idx].ext_flags & ext_vol_bit::RAGE) != 0
                            {
                                if is_castle {
                                    if (self.mons[def_idx].ext_flags & ext_vol_bit::RAGE_ARMED) != 0
                                    {
                                        self.mons[def_idx].ext_timers[ext_timer_idx::RAGE_COUNTER] =
                                            self.mons[def_idx].ext_timers
                                                [ext_timer_idx::RAGE_COUNTER]
                                                .saturating_add(1);
                                    }
                                } else {
                                    self.apply_boost_stage(def_idx, stat_idx::ATK, 1, is_castle);
                                }
                            }
                            // Twineedle poison (Spec #51②):
                            // Castle: 1st hit only AND Steel immune! PS: each main hit & CAN poison Steel!
                            if mspec.special_kind == SpecialMoveKind::Twineedle
                                && self.mons[def_idx].hp() > 0
                                && self.mons[def_idx].status_raw() == 0
                                && self.sides[foe_side].safeguard_turns == 0
                            {
                                let can_try = !is_castle || hit_i == 0;
                                if can_try && rng.random_chance(mspec.sec_chance_256 as u32, 256) {
                                    if is_castle {
                                        self.try_inflict_status(
                                            table,
                                            foe_side,
                                            Status::Psn as u8,
                                            false,
                                            rng,
                                        );
                                    } else {
                                        // PS Gen 2 Twineedle poisons even Steel types, but Poison types remain immune.
                                        let is_poison_type = (table.mons[def_idx].type_mask
                                            & (1u32 << (table.poison_type & 31)))
                                            != 0;
                                        if !is_poison_type {
                                            let prev_res =
                                                if (self.mons[def_idx].status_counter() & 0x10) != 0
                                                {
                                                    self.mons[def_idx].status_counter()
                                                } else {
                                                    0
                                                };
                                            self.mons[def_idx]
                                                .set_status(Status::Psn as u8, prev_res);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if mspec.drain_half && dealt > 0 {
                        let drain = (dealt / 2).max(1);
                        let maxhp = table.mons[att_idx].maxhp;
                        self.mons[att_idx].hp_pool[0] =
                            (self.mons[att_idx].hp() + drain).min(maxhp);
                    }
                }
            }

            // Castle #53: Counter/Mirror Coat against Outrage/Thrash/Petal Dance/Rollout 2nd+ hits
            // ONLY works if the current defender received the 1st hit!
            let is_locked_multi_turn_atk = matches!(
                mspec.special_kind,
                SpecialMoveKind::OutrageLock | SpecialMoveKind::Rollout
            );
            let counter_allowed_locked = if is_castle && is_locked_multi_turn_atk && is_locked_cont
            {
                (self.mons[def_idx].ext_flags & ext_vol_bit::LOCKED_FIRST_HIT_RECV) != 0
            } else {
                true
            };

            // Record Counter / Mirror Coat turn damage only when hitting main HP (never Substitute).
            let counter_dmg = if false_swipe_counter_override > 0 {
                false_swipe_counter_override
            } else if main_dealt > 0 {
                main_dealt as u16
            } else {
                0
            };
            if counter_dmg > 0 && hit_main_hp && counter_allowed_locked {
                self.hurt_damage_this_turn[foe_side] = counter_dmg;
                // Gen 2 Counter/Mirror Coat checks static move category (Hidden Power is always Physical!).
                let counter_is_physical =
                    mspec.is_physical || mspec.var_bp_kind == VarBpKind::HiddenPower;
                self.hurt_cat_this_turn[foe_side] = if counter_is_physical { 1 } else { 2 };
            }

            // Locked-move bookkeeping (Outrage/Thrash/Petal Dance, Rollout, Fury Cutter, Rage).
            if mspec.special_kind == SpecialMoveKind::OutrageLock && !called_by_sleeptalk {
                if !is_locked_cont {
                    if self.mons[att_idx].hp() > 0 {
                        self.mons[att_idx].ext_flags |= ext_vol_bit::LOCKED_OUTRAGE;
                        self.mons[att_idx].set_locked_move_slot(move_slot as u8);
                        // PS: `duration = 2..=3`, decremented in EOT `exec_residuals` (matching `Battle`).
                        // Castle #44: `castle_residual` skips `lockedmove`; `1..=2` remaining turns decremented on act.
                        self.mons[att_idx].ext_timers[ext_timer_idx::LOCKED_TURNS] = if is_castle {
                            (1 + rng.random(2)) as u8
                        } else {
                            (2 + rng.random(2)) as u8
                        };
                        if hit_main_hp && self.mons[def_idx].hp() > 0 {
                            self.mons[def_idx].ext_flags |= ext_vol_bit::LOCKED_FIRST_HIT_RECV;
                        }
                    }
                } else if is_castle {
                    let rem = self.mons[att_idx].ext_timers[ext_timer_idx::LOCKED_TURNS]
                        .saturating_sub(1);
                    self.mons[att_idx].ext_timers[ext_timer_idx::LOCKED_TURNS] = rem;
                    if rem == 0 && self.mons[att_idx].hp() > 0 {
                        // Castle #44 (`Probe 8A`): confuse IMMEDIATELY after final locked action!
                        self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCKED_OUTRAGE;
                        if !self.mons[att_idx].is_multi_turn_locked() {
                            self.mons[att_idx].set_locked_move_slot(0);
                        }
                        if self.sides[side].safeguard_turns == 0 {
                            self.mons[att_idx].vol_flags |= vol_bit::CONFUSION;
                            self.mons[att_idx].timers[timer_idx::CONFUSION] =
                                (2 + rng.random(4)) as u8;
                            self.check_status_berry_castle(table, side);
                        }
                    }
                }
            }
            if mspec.special_kind == SpecialMoveKind::Rollout {
                if hit_landed && !immune_hit && self.mons[att_idx].hp() > 0 {
                    let prev_cnt = self.mons[att_idx].rollout_count();
                    if prev_cnt == 0 && hit_main_hp && self.mons[def_idx].hp() > 0 {
                        self.mons[def_idx].ext_flags |= ext_vol_bit::LOCKED_FIRST_HIT_RECV;
                    }
                    let next_cnt = prev_cnt + 1;
                    if next_cnt >= 5 {
                        self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCKED_ROLLOUT;
                        self.mons[att_idx].set_rollout_count(0);
                        if !self.mons[att_idx].is_multi_turn_locked() {
                            self.mons[att_idx].set_locked_move_slot(0);
                        }
                    } else {
                        self.mons[att_idx].ext_flags |= ext_vol_bit::LOCKED_ROLLOUT;
                        self.mons[att_idx].set_locked_move_slot(move_slot as u8);
                        self.mons[att_idx].set_rollout_count(next_cnt);
                    }
                } else {
                    self.mons[att_idx].ext_flags &= !ext_vol_bit::LOCKED_ROLLOUT;
                    self.mons[att_idx].set_rollout_count(0);
                    if !self.mons[att_idx].is_multi_turn_locked() {
                        self.mons[att_idx].set_locked_move_slot(0);
                    }
                }
            }
            if mspec.special_kind == SpecialMoveKind::FuryCutter {
                if hit_landed && !immune_hit && self.mons[att_idx].hp() > 0 {
                    let cur = self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER];
                    self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER] = if is_castle {
                        cur.saturating_add(1)
                    } else {
                        (cur + 1).min(4)
                    };
                } else {
                    self.mons[att_idx].ext_timers[ext_timer_idx::FURY_CUTTER] = 0;
                }
            }
            if mspec.special_kind == SpecialMoveKind::Rage
                && hit_landed
                && !immune_hit
                && self.mons[att_idx].hp() > 0
            {
                self.mons[att_idx].ext_flags |= ext_vol_bit::RAGE | ext_vol_bit::RAGE_ARMED;
            }

            // Recoil (PS: 1 HP on Sub & skip 1v1 KO recoil; Castle: 0 HP on Sub & always take 1v1 recoil!).
            if mspec.recoil_quarter && dealt > 0 && self.mons[att_idx].hp() > 0 {
                if is_castle {
                    let ghost_struggle = is_struggle
                        && (table.mons[def_idx].type_mask & (1u32 << (table.ghost_type & 31)))
                            != 0;
                    if !def_had_sub && !ghost_struggle {
                        let recoil = (dealt / 4).max(1);
                        self.mons[att_idx].apply_damage_branchless(recoil, true);
                        self.turn_extra_flags =
                            (self.turn_extra_flags & !0x03) | 0x01 | ((side as u8) << 1);
                    }
                } else if def_had_sub {
                    self.mons[att_idx].apply_damage_branchless(1, true);
                } else {
                    let is_1v1_ko = self.sides[0].alive_mask.count_ones() == 1
                        && self.sides[1].alive_mask.count_ones() == 1
                        && self.mons[def_idx].hp() <= 0;
                    if !is_1v1_ko {
                        let recoil = (dealt / 4).max(1);
                        self.mons[att_idx].apply_damage_branchless(recoil, true);
                    }
                }
            }

            // Hyper Beam sets mustrecharge on any non-immune hit (even on KO or Substitute).
            if mspec.is_recharge && hit_landed && self.mons[att_idx].hp() > 0 {
                self.mons[att_idx].set_must_recharge(true);
            }

            // Rapid Spin clears Spikes, Leech Seed, and partial traps (Wrap/Bind/Fire Spin/Clamp/Whirlpool),
            // NEVER Mean Look / Spider Web (`vol_bit::TRAPPED_OR_LOCKED`).
            // Castle #43 (`Probe 12D2`): clears even vs Ghost immunity!
            if mspec.special_kind == SpecialMoveKind::RapidSpin
                && (hit_landed || is_castle)
                && self.mons[att_idx].hp() > 0
            {
                self.sides[side].side_flags &= !0x03;
                self.mons[att_idx].vol_flags &= !vol_bit::LEECH_SEED;
                self.mons[att_idx].set_partially_trapped(false);
            }

            // Partial trapping moves (Wrap / Bind / Fire Spin / Clamp / Whirlpool) — NEVER through Sub (`Probe 12D1`)!
            if mspec.special_kind == SpecialMoveKind::PartialTrap
                && hit_landed
                && hit_main_hp
                && self.mons[def_idx].hp() > 0
                && !self.mons[def_idx].partially_trapped()
            {
                self.mons[def_idx].set_partially_trapped(true);
                self.mons[def_idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] =
                    (2 + rng.random(4)) as u8;
            }

            // Defrost user (Flame Wheel / Sacred Fire) or frozen target hit by burn-secondary move.
            if hit_landed
                && self.mons[att_idx].hp() > 0
                && mspec.is_defrost
                && self.mons[att_idx].status_raw() == Status::Frz as u8
            {
                let prev_res = if (self.mons[att_idx].status_counter() & 0x10) != 0 {
                    self.mons[att_idx].status_counter()
                } else {
                    0
                };
                self.mons[att_idx].set_status(Status::None as u8, prev_res);
            }
            if hit_landed
                && hit_main_hp
                && self.mons[def_idx].hp() > 0
                && self.mons[def_idx].status_raw() == Status::Frz as u8
                && (mspec.sec_status == Status::Brn as u8
                    || (mspec.special_kind == SpecialMoveKind::TriAttack && rng.random_chance(1, 3)))
            {
                let prev_res = if (self.mons[def_idx].status_counter() & 0x10) != 0 {
                    self.mons[def_idx].status_counter()
                } else {
                    0
                };
                self.mons[def_idx].set_status(Status::None as u8, prev_res);
            }

            // Destiny Bond: if direct attack fainted a target with Destiny Bond active, attacker faints too.
            if def_had_dbond && self.mons[def_idx].hp() <= 0 && self.mons[att_idx].hp() > 0 {
                self.mons[att_idx].mark_fainted();
                self.turn_extra_flags |= 1u8 << (4 + foe_side);
            }

            // Secondary effects on target (when target survived and hit main HP — including Endure 0 dmg at 1 HP).
            if hit_landed
                && hit_main_hp
                && self.mons[def_idx].hp() > 0
                && mspec.special_kind != SpecialMoveKind::Twineedle
            {
                // Castle #22: 100% secondaries (`sec_chance_256 == 255`) are strictly 100.0% (no 1/256 fail).
                let sec_ok = mspec.sec_chance_256 > 0
                    && ((is_castle && mspec.sec_chance_256 == 255)
                        || rng.random_chance(mspec.sec_chance_256 as u32, 256));
                if sec_ok {
                    if mspec.special_kind == SpecialMoveKind::TriAttack {
                        // Tri Attack: 20% secondary -> equal 1/3 Par / Brn / Frz.
                        // Castle #11: Tri Attack freeze bypasses Freeze Clause!
                        if self.mons[def_idx].status_raw() == 0
                            && self.sides[foe_side].safeguard_turns == 0
                        {
                            let pick = rng.random(3);
                            let st = match pick {
                                0 => Status::Par as u8,
                                1 => Status::Brn as u8,
                                _ => Status::Frz as u8,
                            };
                            self.try_inflict_status(table, foe_side, st, true, rng);
                        }
                    } else if mspec.sec_status != 0
                        && self.mons[def_idx].status_raw() == 0
                        && self.sides[foe_side].safeguard_turns == 0
                    {
                        self.try_inflict_status(table, foe_side, mspec.sec_status, false, rng);
                    }
                    if mspec.sec_vol != 0 {
                        let mut add_vol = mspec.sec_vol;
                        let def_st = self.mons[def_idx].status_raw();
                        // PS: non-King's-Rock flinch cannot flinch sleeping/frozen targets.
                        // Castle #12 (`Probe 8E`): flinch CAN be applied to sleeping/frozen targets!
                        if !is_castle
                            && (add_vol & vol_bit::FLINCH) != 0
                            && (def_st == Status::Slp as u8 || def_st == Status::Frz as u8)
                        {
                            add_vol &= !vol_bit::FLINCH;
                        }
                        if (add_vol & vol_bit::CONFUSION) != 0 {
                            if self.sides[foe_side].safeguard_turns != 0
                                || (self.mons[def_idx].vol_flags & vol_bit::CONFUSION) != 0
                            {
                                add_vol &= !vol_bit::CONFUSION;
                            } else {
                                self.mons[def_idx].timers[timer_idx::CONFUSION] =
                                    (2 + rng.random(4)) as u8;
                            }
                        }
                        self.mons[def_idx].vol_flags |= add_vol;
                    }
                    if mspec.sec_boost.0 != NO_STAT_IDX {
                        let is_self_boost = (mspec.sec_boost.0 & SELF_BOOST_FLAG) != 0;
                        let b_idx = if is_self_boost { att_idx } else { def_idx };
                        let st_i = mspec.sec_boost.0 & !SELF_BOOST_FLAG;
                        let mist_blocks = !is_self_boost
                            && mspec.sec_boost.1 < 0
                            && (self.mons[def_idx].ext_flags & ext_vol_bit::MIST) != 0;
                        if !mist_blocks {
                            if is_self_boost && st_i == ANCIENTPOWER_ALL_STATS {
                                for stat_i in 0..5usize {
                                    self.apply_boost_stage(b_idx, stat_i, mspec.sec_boost.1, is_castle);
                                }
                            } else if (st_i as usize) < 7 {
                                let st = st_i as usize;
                                self.apply_boost_stage(b_idx, st, mspec.sec_boost.1, is_castle);
                            }
                        }
                    }
                    // Thief steals via 100% secondary when hitting surviving, un-Substituted target.
                    if mspec.special_kind == SpecialMoveKind::Thief
                        && self.mons[att_idx].hp() > 0
                        && self.mons[att_idx].item == 0
                        && self.mons[def_idx].item != 0
                    {
                        self.mons[att_idx].item = self.mons[def_idx].item;
                        self.mons[def_idx].item = 0;
                        self.trigger_berserk_gene(table, side, rng);
                    }
                }
                // King's Rock 30/256 silent flinch (once on final hit, hits sleeping/frozen targets too).
                if att_item.is_kings_rock
                    && mspec.kingsrock
                    && (self.mons[def_idx].vol_flags & vol_bit::FLINCH) == 0
                    && rng.random_chance(30, 256)
                {
                    self.mons[def_idx].vol_flags |= vol_bit::FLINCH;
                }
                // Castle #32①: opponent-inflicted status/confusion triggers target's status berry IMMEDIATELY!
                if is_castle {
                    self.check_status_berry_castle(table, foe_side);
                } else {
                    self.check_single_status_berry_ps(table, foe_side);
                }
            }
        } else if mspec.is_status || mspec.special_kind == SpecialMoveKind::FutureSight {
            self.exec_status_move(
                table,
                side,
                foe_side,
                att_idx,
                def_idx,
                &mspec,
                moved_second,
                rng,
            );
            // Castle #32①: status move inflicting status/confusion triggers target's berry immediately!
            if is_castle {
                self.check_status_berry_castle(table, foe_side);
            } else {
                self.check_single_status_berry_ps(table, foe_side);
            }
        }

        if mspec.is_selfdestruct {
            self.mons[att_idx].mark_fainted();
        }
        if !mspec.is_selfdestruct
            && mspec.special_kind != SpecialMoveKind::BatonPass
            && self.mons[att_idx].hp() > 0
            && self.mons[def_idx].hp() > 0
            && self.outcome == 0
        {
            self.exec_after_move_self(table, side);
        } else if self.mons[def_idx].hp() > 0 {
            self.mons[def_idx].vol_flags &= !vol_bit::DESTINY_BOND;
        }
    }

    /// Gen 2 `onAfterMoveSelf` slip damage (Burn, Poison, Toxic, Leech Seed, Castle Partial Trap, Nightmare, Curse).
    #[inline]
    fn exec_after_move_self(&mut self, table: &MatchTable, side: usize) {
        let is_castle = table.ruleset.is_castle();
        let foe_side = side ^ 1;
        let idx = side * 3 + self.sides[side].active_idx as usize;
        let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
        if self.mons[foe_idx].hp() > 0 {
            self.mons[foe_idx].vol_flags &= !vol_bit::DESTINY_BOND;
        }
        if self.mons[idx].hp() <= 0 || self.mons[foe_idx].hp() <= 0 {
            return;
        }
        let st = self.mons[idx].status_raw();
        if st == Status::Brn as u8 || st == Status::Psn as u8 {
            let cnt = self.mons[idx].status_counter();
            let use_res = (cnt & 0x10) != 0 && !(is_castle && st == Status::Brn as u8);
            if use_res {
                let c = ((cnt & 0x0F) + 1).min(15);
                self.mons[idx].set_status(st, c | 0x10);
                let slip = table.mons[idx].quantum_1_16 * (c as i16);
                self.mons[idx].apply_damage_branchless(slip, true);
            } else {
                let slip = table.mons[idx].quantum_1_8;
                self.mons[idx].apply_damage_branchless(slip, true);
            }
        } else if st == Status::Tox as u8 {
            let c = ((self.mons[idx].status_counter() & 0x0F) + 1).min(15);
            self.mons[idx].set_status(Status::Tox as u8, c | 0x10);
            let slip = table.mons[idx].quantum_1_16 * (c as i16);
            self.mons[idx].apply_damage_branchless(slip, true);
        }
        if self.mons[idx].hp() > 0 && (self.mons[idx].vol_flags & vol_bit::LEECH_SEED) != 0 {
            let drain = table.mons[idx].quantum_1_8;
            let dealt = self.mons[idx].apply_damage_branchless(drain, true);
            if self.mons[foe_idx].hp() > 0 {
                let foe_max = table.mons[foe_idx].maxhp;
                self.mons[foe_idx].hp_pool[0] = (self.mons[foe_idx].hp() + dealt).min(foe_max);
            }
        }
        // Castle #6 (`Probe 6`/`12I`): Partial Trap (Wrap/Bind/Fire Spin/Clamp/Whirlpool)
        // ticks in `AfterMoveSelf` right after Poison/Burn/Leech Seed (`1/16 maxhp`).
        if is_castle && self.mons[idx].hp() > 0 && self.mons[idx].partially_trapped() {
            let rem = self.mons[idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS];
            if rem <= 1 {
                self.mons[idx].set_partially_trapped(false);
            } else {
                self.mons[idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] = rem - 1;
                let slip = table.mons[idx].quantum_1_16;
                self.mons[idx].apply_damage_branchless(slip, true);
            }
        }
        // Nightmare: ticks while asleep, OR in Castle while `MIRACLE_NIGHTMARE` persists (`Probe 11B`).
        if self.mons[idx].hp() > 0
            && (self.mons[idx].vol_flags & vol_bit::NIGHTMARE) != 0
            && (self.mons[idx].status_raw() == Status::Slp as u8
                || (is_castle
                    && (self.mons[idx].ext_flags & ext_vol_bit::MIRACLE_NIGHTMARE) != 0))
        {
            let slip = (table.mons[idx].maxhp / 4).max(1);
            self.mons[idx].apply_damage_branchless(slip, true);
        }
        if self.mons[idx].hp() > 0 && (self.mons[idx].vol_flags & vol_bit::CURSE) != 0 {
            let slip = (table.mons[idx].maxhp / 4).max(1);
            self.mons[idx].apply_damage_branchless(slip, true);
        }
    }

    #[inline]
    fn try_inflict_status(
        &mut self,
        table: &MatchTable,
        target_side: usize,
        status_code: u8,
        from_tri_attack: bool,
        rng: &mut Prng,
    ) {
        let is_castle = table.ruleset.is_castle();
        let idx = target_side * 3 + self.sides[target_side].active_idx as usize;
        if self.mons[idx].status_raw() != 0
            || self.mons[idx].sub_hp() > 0
            || self.sides[target_side].safeguard_turns > 0
        {
            return;
        }
        let prev_res = if (self.mons[idx].status_counter() & 0x10) != 0 {
            self.mons[idx].status_counter()
        } else {
            0
        };
        let target_tmask = table.mons[idx].type_mask;
        // Sleep Clause & Freeze Clause checks.
        if status_code == Status::Slp as u8 {
            for p_i in 0..3usize {
                if p_i != (self.sides[target_side].active_idx as usize)
                    && (self.sides[target_side].alive_mask & (1 << p_i)) != 0
                {
                    let m = &self.mons[target_side * 3 + p_i];
                    if m.hp() > 0 && m.status_raw() == Status::Slp as u8 {
                        // Castle #9: Blocking Sleep Clause (including Rest), NO forfeit!
                        // PS: Stadium Sleep Clause immediate forfeit when clause_flags & 1 != 0.
                        if !is_castle && (self.clause_flags & 1) != 0 {
                            self.finish_outcome(if target_side == 0 { 1 } else { 2 });
                        }
                        return;
                    }
                }
            }
            // Castle #10: non-Rest sleep uniform 1..=4 skipped turns (`counter = 2..=5`).
            // PS: non-Rest sleep 1..=3 skipped turns (`counter = 2..=4`).
            let turns = if is_castle {
                (2 + rng.random(4)) as u8
            } else {
                (2 + rng.random(3)) as u8
            };
            self.mons[idx].set_status(Status::Slp as u8, turns);
        } else if status_code == Status::Frz as u8 {
            if !from_tri_attack
                && ((target_tmask & (1u32 << (table.ice_type & 31))) != 0
                    || (self.weather_turns > 0 && self.weather_kind == CompactWeather::Sun))
            {
                return;
            }
            // Castle #11: Tri Attack freeze bypasses Freeze Clause!
            let check_freeze_clause = !(is_castle && from_tri_attack);
            if check_freeze_clause {
                for p_i in 0..3usize {
                    if (self.sides[target_side].alive_mask & (1 << p_i)) != 0 {
                        let m = &self.mons[target_side * 3 + p_i];
                        if m.hp() > 0 && m.status_raw() == Status::Frz as u8 {
                            return;
                        }
                    }
                }
            }
            self.mons[idx].set_status(Status::Frz as u8, prev_res);
        } else if status_code == Status::Par as u8 {
            self.mons[idx].set_status(Status::Par as u8, prev_res);
            self.mons[idx].vol_flags |= vol_bit::PAR_SPEED_DROP;
        } else if status_code == Status::Brn as u8 {
            if !from_tri_attack && (target_tmask & (1u32 << (table.fire_type & 31))) != 0 {
                return;
            }
            self.mons[idx].set_status(Status::Brn as u8, prev_res);
            self.mons[idx].vol_flags |= vol_bit::BRN_ATK_DROP;
        } else if status_code == Status::Tox as u8 {
            if (target_tmask & table.poison_immune_mask) != 0 {
                return;
            }
            self.mons[idx].set_status(Status::Tox as u8, 0x10);
        } else if status_code == Status::Psn as u8 {
            if (target_tmask & table.poison_immune_mask) != 0 {
                return;
            }
            self.mons[idx].set_status(Status::Psn as u8, prev_res);
        }
    }

    /// PS single-status berries (`onUpdate` immediate trigger: Ice=Brn, Bitter=Conf, PrzCure=Par, PsnCure=Psn/Tox, Burnt=Frz).
    #[inline]
    fn check_single_status_berry_ps(&mut self, table: &MatchTable, side: usize) {
        let idx = side * 3 + self.sides[side].active_idx as usize;
        let mon = &mut self.mons[idx];
        if mon.hp() <= 0 || mon.item == 0 {
            return;
        }
        let ispec = table.item_spec(mon.item);
        let st = mon.status_raw();
        let prev_res = if st != Status::Slp as u8 && (mon.status_counter() & 0x10) != 0 {
            mon.status_counter()
        } else {
            0
        };
        if ispec.is_ice_berry && st == Status::Brn as u8 {
            mon.set_status(Status::None as u8, prev_res);
            mon.vol_flags &= !vol_bit::BRN_ATK_DROP;
            mon.item = 0;
        } else if ispec.is_bitter_berry && (mon.vol_flags & vol_bit::CONFUSION) != 0 {
            mon.vol_flags &= !vol_bit::CONFUSION;
            mon.timers[timer_idx::CONFUSION] = 0;
            mon.item = 0;
        } else if ispec.is_przcure_berry && st == Status::Par as u8 {
            mon.set_status(Status::None as u8, prev_res);
            mon.vol_flags &= !vol_bit::PAR_SPEED_DROP;
            mon.item = 0;
        } else if ispec.is_psncure_berry && (st == Status::Psn as u8 || st == Status::Tox as u8) {
            mon.set_status(Status::None as u8, prev_res);
            mon.item = 0;
        } else if ispec.is_burnt_berry && st == Status::Frz as u8 {
            mon.set_status(Status::None as u8, prev_res);
            mon.item = 0;
        }
    }

    /// Gen 2 PS Mint Berry / Miracle Berry activation (`onBeforeMove` priority 11 + `onResidual` order 5)
    /// plus single-status berries.
    #[inline]
    fn check_status_berry_trigger(&mut self, table: &MatchTable, side: usize) {
        let idx = side * 3 + self.sides[side].active_idx as usize;
        let mon = &mut self.mons[idx];
        if mon.hp() <= 0 || mon.item == 0 {
            return;
        }
        let ispec = table.item_spec(mon.item);
        let st = mon.status_raw();
        if ispec.is_mintberry && st == Status::Slp as u8 {
            mon.set_status(Status::None as u8, 0);
            mon.vol_flags &= !vol_bit::NIGHTMARE;
            mon.item = 0;
        } else if ispec.is_miracleberry && (st != 0 || (mon.vol_flags & vol_bit::CONFUSION) != 0) {
            let prev_res = if st != Status::Slp as u8 && (mon.status_counter() & 0x10) != 0 {
                mon.status_counter()
            } else {
                0
            };
            mon.set_status(Status::None as u8, prev_res);
            mon.vol_flags &= !(vol_bit::CONFUSION
                | vol_bit::PAR_SPEED_DROP
                | vol_bit::BRN_ATK_DROP
                | vol_bit::NIGHTMARE);
            mon.timers[timer_idx::CONFUSION] = 0;
            mon.item = 0;
        } else {
            self.check_single_status_berry_ps(table, side);
        }
    }

    /// Castle PBS status berry activation (spec #13, #32, #33):
    /// - `Ice Berry` (ID 81) cures Burn AND Confusion (`Probe 6`); `Bitter Berry` (ID 84) is inert!
    /// - `Miracle Berry` curing Sleep with `Nightmare` active leaves `MIRACLE_NIGHTMARE` ticking (`Probe 11B`)!
    #[inline]
    fn check_status_berry_castle(&mut self, table: &MatchTable, side: usize) {
        let idx = side * 3 + self.sides[side].active_idx as usize;
        let mon = &mut self.mons[idx];
        if mon.hp() <= 0 || mon.item == 0 {
            return;
        }
        let ispec = table.item_spec(mon.item);
        let st = mon.status_raw();
        let has_conf = (mon.vol_flags & vol_bit::CONFUSION) != 0;
        if ispec.is_mintberry && st == Status::Slp as u8 {
            mon.set_status(Status::None as u8, 0);
            mon.vol_flags &= !vol_bit::NIGHTMARE;
            mon.ext_flags &= !ext_vol_bit::MIRACLE_NIGHTMARE;
            mon.item = 0;
        } else if ispec.is_miracleberry && (st != 0 || has_conf) {
            let was_asleep_nightmare =
                st == Status::Slp as u8 && (mon.vol_flags & vol_bit::NIGHTMARE) != 0;
            mon.set_status(Status::None as u8, 0);
            mon.vol_flags &= !(vol_bit::CONFUSION | vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP);
            if was_asleep_nightmare {
                // Castle #13 (`Probe 11B`): Miracle Berry does NOT clear Nightmare!
                mon.ext_flags |= ext_vol_bit::MIRACLE_NIGHTMARE;
            }
            mon.timers[timer_idx::CONFUSION] = 0;
            mon.item = 0;
        } else if ispec.is_ice_berry && (st == Status::Brn as u8 || has_conf) {
            // Castle #33 (`Probe 6`): Ice Berry (81) cures Burn AND Confusion!
            if st == Status::Brn as u8 {
                mon.set_status(Status::None as u8, 0);
                mon.vol_flags &= !vol_bit::BRN_ATK_DROP;
            }
            if has_conf {
                mon.vol_flags &= !vol_bit::CONFUSION;
                mon.timers[timer_idx::CONFUSION] = 0;
            }
            mon.item = 0;
        } else if ispec.is_przcure_berry && st == Status::Par as u8 {
            mon.set_status(Status::None as u8, 0);
            mon.vol_flags &= !vol_bit::PAR_SPEED_DROP;
            mon.item = 0;
        } else if ispec.is_psncure_berry && (st == Status::Psn as u8 || st == Status::Tox as u8) {
            mon.set_status(Status::None as u8, 0);
            mon.item = 0;
        } else if ispec.is_burnt_berry && st == Status::Frz as u8 {
            mon.set_status(Status::None as u8, 0);
            mon.item = 0;
        }
        // Note: `is_bitter_berry` (ID 84) is completely inert in Castle PBS!
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn exec_status_move(
        &mut self,
        table: &MatchTable,
        side: usize,
        foe_side: usize,
        att_idx: usize,
        def_idx: usize,
        mspec: &CompactMoveSpec,
        moved_second: bool,
        rng: &mut Prng,
    ) {
        let is_castle = table.ruleset.is_castle();
        match mspec.special_kind {
            SpecialMoveKind::Reflect => {
                if self.sides[side].reflect_turns == 0 {
                    self.sides[side].reflect_turns = 5;
                }
            }
            SpecialMoveKind::LightScreen => {
                if self.sides[side].lightscreen_turns == 0 {
                    self.sides[side].lightscreen_turns = 5;
                }
            }
            SpecialMoveKind::Safeguard => {
                if self.sides[side].safeguard_turns == 0 {
                    self.sides[side].safeguard_turns = 5;
                }
            }
            SpecialMoveKind::Spikes => {
                if (self.sides[foe_side].side_flags & 0x03) == 0 {
                    self.sides[foe_side].side_flags |= 1;
                }
            }
            SpecialMoveKind::Curse => {
                // Curse: non-Ghost (+1 Atk, +1 Def, -1 Spe), Ghost (-50% HP, inflicts Curse).
                let is_ghost =
                    (table.mons[att_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
                if !is_ghost {
                    self.apply_boost_stage(att_idx, stat_idx::ATK, 1, is_castle);
                    self.apply_boost_stage(att_idx, stat_idx::DEF, 1, is_castle);
                    self.apply_boost_stage(att_idx, stat_idx::SPE, -1, is_castle);
                } else if self.mons[def_idx].sub_hp() == 0
                    && (self.mons[def_idx].vol_flags & vol_bit::CURSE) == 0
                {
                    let cost = (table.mons[att_idx].maxhp / 2).max(1);
                    self.mons[att_idx].apply_damage_branchless(cost, true);
                    self.mons[def_idx].vol_flags |= vol_bit::CURSE;
                }
            }
            SpecialMoveKind::BellyDrum => {
                // Belly Drum: costs 50% maxhp and sets Attack stage to +6.
                // Castle #41 (`Probe 3A`): succeeds at `hp == floor(maxhp / 2)` leaving 1 HP!
                let cost = table.mons[att_idx].maxhp / 2;
                let hp = self.mons[att_idx].hp();
                let enough_hp = if is_castle { hp >= cost } else { hp > cost };
                if enough_hp && self.mons[att_idx].boosts[stat_idx::ATK] < 6 {
                    self.mons[att_idx].hp_pool[0] = (hp - cost).max(1);
                    self.apply_boost_stage(att_idx, stat_idx::ATK, 12, is_castle);
                }
            }
            SpecialMoveKind::Phaze => {
                // Roar / Whirlwind: fails if user moves before target; otherwise drags random alive bench mon.
                if moved_second {
                    let mask =
                        self.sides[foe_side].alive_mask & !(1 << self.sides[foe_side].active_idx);
                    if mask != 0 {
                        let slot = if mask.count_ones() == 1 || rng.random(2) == 0 {
                            mask.trailing_zeros() as u8
                        } else {
                            (7 - mask.leading_zeros()) as u8
                        };
                        self.exec_switch(table, foe_side, slot.min(2), SwitchKind::Drag, rng);
                    }
                }
            }
            SpecialMoveKind::Encore => {
                // Encore: locks target into its last move for 3..=6 turns (bypasses Substitute in Gen 2).
                // Castle #54② (`Probe 12I`): fails unconditionally vs Roar, Whirlwind, Sleep Talk, Encore, etc.!
                if (self.mons[def_idx].vol_flags & vol_bit::ENCORE) == 0 {
                    if let Some(lm) = self.mons[def_idx].last_move_slot().map(|s| s as usize) {
                        if lm < table.mons[def_idx].num_moves as usize
                            && self.mons[def_idx].pp[lm] > 0
                        {
                            let lm_spec = &table.move_specs[def_idx][lm];
                            let blocked = lm_spec.special_kind == SpecialMoveKind::Encore
                                || lm_spec.fail_encore
                                || (is_castle
                                    && matches!(
                                        lm_spec.special_kind,
                                        SpecialMoveKind::Phaze | SpecialMoveKind::SleepTalk
                                    ));
                            if !blocked {
                                self.mons[def_idx].vol_flags |= vol_bit::ENCORE;
                                self.mons[def_idx].timers[timer_idx::ENCORE] =
                                    (3 + rng.random(4)) as u8;
                                self.mons[def_idx].set_encore_slot(lm as u8);
                            }
                        }
                    }
                }
            }
            SpecialMoveKind::Disable => {
                // Disable: disables target's last move (bypasses Substitute in Gen 2).
                // PS: 2..=5 (+1 if moved_second). Castle #54①: 1..=7 (+1 if moved_second).
                if (self.mons[def_idx].vol_flags & vol_bit::DISABLE) == 0 {
                    if let Some(lm) = self.mons[def_idx].last_move_slot().map(|s| s as usize) {
                        if lm < table.mons[def_idx].num_moves as usize
                            && self.mons[def_idx].pp[lm] > 0
                        {
                            self.mons[def_idx].vol_flags |= vol_bit::DISABLE;
                            self.mons[def_idx].timers[timer_idx::DISABLE] = if is_castle {
                                (1 + rng.random(7) + (moved_second as u32)) as u8
                            } else {
                                (2 + rng.random(4) + (moved_second as u32)) as u8
                            };
                            self.mons[def_idx].set_disable_slot(lm as u8);
                        }
                    }
                }
            }
            SpecialMoveKind::PainSplit => {
                if self.mons[def_idx].sub_hp() == 0 {
                    let avg = ((self.mons[att_idx].hp() + self.mons[def_idx].hp()) / 2).max(1);
                    self.mons[att_idx].hp_pool[0] = avg.min(table.mons[att_idx].maxhp);
                    self.mons[def_idx].hp_pool[0] = avg.min(table.mons[def_idx].maxhp);
                }
            }
            SpecialMoveKind::BatonPass => {
                // Explicit Baton Pass receiver selection (`BATONN`, §7-A.2):
                // - When >= 2 bench receivers are alive (and BP fix is enabled), pause in `request = 2`
                //   with `bp_switch_flags` bit set so caller explicitly picks `4..7` via `step_switch`.
                // - When only 1 bench receiver is alive (zero branching, and preserves `mcts.rs` directed BP),
                //   switch immediately to that sole bench receiver!
                let mask = self.sides[side].alive_mask & !(1 << self.sides[side].active_idx);
                if mask != 0 {
                    let bp_fix = (self.sides[0].side_flags & SIDE0_DISABLE_BP_FIX) == 0;
                    if mask.count_ones() >= 2 && bp_fix {
                        self.sides[side].request = 2;
                        self.sides[foe_side].request = 0;
                        self.bp_switch_flags |= 1u8 << side;
                    } else if mask.count_ones() == 1 {
                        let sole = mask.trailing_zeros() as u8;
                        if !moved_second {
                            self.bp_switch_flags |= 1u8 << (2 + side);
                        }
                        self.exec_switch(table, side, sole.min(2), SwitchKind::BatonPass, rng);
                    } else {
                        // Legacy ablation (`SIDE0_DISABLE_BP_FIX`): highest HP fraction bench mon.
                        let mut best_slot = mask.trailing_zeros() as u8;
                        let mut best_hp = -1i32;
                        for s_i in 0..3u8 {
                            if (mask & (1 << s_i)) != 0 {
                                let idx = side * 3 + s_i as usize;
                                let frac = (self.mons[idx].hp() as i32 * 1024)
                                    / (table.mons[idx].maxhp.max(1) as i32);
                                if frac > best_hp {
                                    best_hp = frac;
                                    best_slot = s_i;
                                }
                            }
                        }
                        if !moved_second {
                            self.bp_switch_flags |= 1u8 << (2 + side);
                        }
                        self.exec_switch(table, side, best_slot.min(2), SwitchKind::BatonPass, rng);
                    }
                }
            }
            SpecialMoveKind::PerishSong => {
                if self.sides[side].alive_mask.count_ones() > 1 {
                    let mut any_new = false;
                    for &idx in &[att_idx, def_idx] {
                        if (self.mons[idx].vol_flags & vol_bit::PERISH_SONG) == 0 {
                            self.mons[idx].vol_flags |= vol_bit::PERISH_SONG;
                            self.mons[idx].set_perish_turns(4);
                            any_new = true;
                        }
                    }
                    if any_new {
                        self.turn_extra_flags |= 1u8 << (2 + side);
                    }
                }
            }
            SpecialMoveKind::DestinyBond => {
                if self.sides[side].alive_mask.count_ones() > 1 {
                    self.mons[att_idx].vol_flags |= vol_bit::DESTINY_BOND;
                }
            }
            SpecialMoveKind::HealBell => {
                // Castle #14 (`Probe 12B`): Heal Bell completely resets Toxic counter (`residualdmg`)!
                // PS: preserves `residualdmg` counter on active mon.
                for p_i in 0..3usize {
                    let m = &mut self.mons[side * 3 + p_i];
                    if m.hp() > 0 {
                        let prev_res = if !is_castle
                            && m.status_raw() != Status::Slp as u8
                            && (m.status_counter() & 0x10) != 0
                        {
                            m.status_counter()
                        } else {
                            0
                        };
                        m.set_status(Status::None as u8, prev_res);
                        m.vol_flags &= !(vol_bit::PAR_SPEED_DROP
                            | vol_bit::BRN_ATK_DROP
                            | vol_bit::NIGHTMARE);
                        m.ext_flags &= !ext_vol_bit::MIRACLE_NIGHTMARE;
                    }
                }
            }
            SpecialMoveKind::PsychUp => {
                // Castle #46 (`Probe 8C`): fails if all 7 target boosts are 0!
                // Psych Up copies only the 7 stat boost stages (NEVER Focus Energy, matching `Battle::PsychUpHit`).
                if !is_castle || self.mons[def_idx].boosts != [0; 7] {
                    self.mons[att_idx].boosts = self.mons[def_idx].boosts;
                }
            }
            SpecialMoveKind::RainDance => {
                self.weather_kind = CompactWeather::Rain;
                self.weather_turns = 5;
            }
            SpecialMoveKind::SunnyDay => {
                self.weather_kind = CompactWeather::Sun;
                self.weather_turns = 5;
            }
            SpecialMoveKind::Sandstorm => {
                self.weather_kind = CompactWeather::Sandstorm;
                self.weather_turns = 5;
            }
            SpecialMoveKind::MeanLook => {
                // Mean Look / Spider Web bypass Substitute in Gen 2 (and Dig/Fly in Castle).
                self.mons[def_idx].vol_flags |= vol_bit::TRAPPED_OR_LOCKED;
            }
            SpecialMoveKind::Haze => {
                self.mons[att_idx].boosts = [0; 7];
                self.mons[def_idx].boosts = [0; 7];
                // Castle #20: Haze does NOT clear Burn/Paralysis stat penalties!
                if !is_castle {
                    self.mons[att_idx].vol_flags &=
                        !(vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP);
                    self.mons[def_idx].vol_flags &=
                        !(vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP);
                }
            }
            SpecialMoveKind::Swagger => {
                if self.mons[def_idx].boosts[stat_idx::ATK] < 6
                    && (is_castle
                        || self.active_stat(table, foe_side, stat_idx::ATK, false, true) != 999)
                {
                    self.apply_boost_stage(def_idx, stat_idx::ATK, 2, is_castle);
                    if self.mons[def_idx].sub_hp() == 0
                        && self.sides[foe_side].safeguard_turns == 0
                        && (self.mons[def_idx].vol_flags & vol_bit::CONFUSION) == 0
                    {
                        self.mons[def_idx].vol_flags |= vol_bit::CONFUSION;
                        self.mons[def_idx].timers[timer_idx::CONFUSION] =
                            (2 + rng.random(4)) as u8;
                    }
                }
                return;
            }
            SpecialMoveKind::Attract => {
                let g_att = self.mons[att_idx].gender;
                let g_def = self.mons[def_idx].gender;
                let opposite = (g_att == 1 && g_def == 2) || (g_att == 2 && g_def == 1);
                if opposite && (self.mons[def_idx].ext_flags & ext_vol_bit::ATTRACT) == 0 {
                    self.mons[def_idx].ext_flags |= ext_vol_bit::ATTRACT;
                }
                return;
            }
            SpecialMoveKind::Mist => {
                self.mons[att_idx].ext_flags |= ext_vol_bit::MIST;
                return;
            }
            SpecialMoveKind::FocusEnergy => {
                self.mons[att_idx].ext_flags |= ext_vol_bit::FOCUS_ENERGY;
                return;
            }
            SpecialMoveKind::DefenseCurl => {
                self.mons[att_idx].ext_flags |= ext_vol_bit::DEFENSE_CURL;
                // Fall through so `mspec.boosts` raises Defense +1.
            }
            SpecialMoveKind::Minimize => {
                self.mons[att_idx].ext_flags |= ext_vol_bit::MINIMIZE;
                // Fall through so `mspec.boosts` raises Evasion +1.
            }
            SpecialMoveKind::LockOn => {
                // Fails behind target Sub, or if already locked on, or if target has Foresight.
                if self.mons[def_idx].sub_hp() == 0
                    && (self.mons[att_idx].ext_flags & ext_vol_bit::LOCK_ON) == 0
                    && (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) == 0
                {
                    self.mons[att_idx].ext_flags |= ext_vol_bit::LOCK_ON;
                    self.mons[att_idx].set_lock_on_turns(if is_castle { 1 } else { 2 });
                }
                return;
            }
            SpecialMoveKind::FutureSight => {
                // Precompute 80 BP typeless Special non-crit hit (including Light Screen!) for 2 turns later.
                if self.future_sight_turns[foe_side] == 0 {
                    let spa = self.active_stat(table, side, stat_idx::SPA, false, false);
                    let spd = self.active_stat(table, foe_side, stat_idx::SPD, false, false);
                    let lvl = table.mons[att_idx].level as i32;
                    let base = if is_castle {
                        get_damage_base_castle(
                            lvl,
                            80,
                            spa,
                            spd,
                            false,
                            false,
                            false,
                            false,
                            CompactWeather::None,
                            false,
                            false,
                            false,
                            NO_TYPE_ID,
                            0,
                            0,
                            false,
                            1,
                            1,
                        )
                    } else {
                        get_damage_base_branchless(
                            lvl,
                            80,
                            spa,
                            spd,
                            false,
                            false,
                            false,
                            false,
                            CompactWeather::None,
                            false,
                            false,
                            false,
                            NO_TYPE_ID,
                            0,
                            0,
                            false,
                        )
                    };
                    let roll = 217 + rng.random(39);
                    let dmg = if is_castle {
                        apply_roll_castle(base, roll, false)
                    } else {
                        apply_roll_branchless(base, roll, false)
                    };
                    self.future_sight_dmg[foe_side] = dmg.clamp(0, 9999) as u16;
                    self.future_sight_turns[foe_side] = 3;
                    self.future_sight_src[foe_side] = self.sides[side].active_idx;
                }
                return;
            }
            SpecialMoveKind::Spite => {
                if let Some(lm) = self.mons[def_idx].last_move_slot().map(|s| s as usize) {
                    if lm < table.mons[def_idx].num_moves as usize
                        && self.mons[def_idx].pp[lm] > 0
                        && table.move_specs[def_idx][lm].special_kind != SpecialMoveKind::SleepTalk
                    {
                        let cut = (2 + rng.random(4)) as u8;
                        self.mons[def_idx].pp[lm] = self.mons[def_idx].pp[lm].saturating_sub(cut);
                        // PS `onUpdate`: Mystery Berry immediately restores +5 PP (clamped to maxpp).
                        // In Castle, Mystery Berry triggers right before victim's 0-PP move in `exec_move_inner` or at EOT!
                        if !is_castle
                            && self.mons[def_idx].pp[lm] == 0
                            && table.item_spec(self.mons[def_idx].item).is_mystery_berry
                        {
                            self.mons[def_idx].pp[lm] = 5u8.min(table.mons[def_idx].max_pp[lm]);
                            self.mons[def_idx].item = 0;
                        }
                    }
                }
                return;
            }
            _ => {}
        }

        if mspec.primary_status != 0 {
            // Respect type immunities for status moves with `ignoreImmunity: false`
            // (Thunder Wave vs Ground, Glare vs Ghost unless Foresight, Poison Powder / Toxic vs Steel),
            // while allowing Lovely Kiss / Sing / Hypnosis (`ignoreImmunity: true`).
            let def_foresight = (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) != 0;
            let type_immune = !mspec.ignore_immunity
                && table
                    .effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight)
                    .1;
            if !type_immune {
                self.try_inflict_status(table, foe_side, mspec.primary_status, false, rng);
            }
        }
        if mspec.primary_vol != 0
            && !matches!(
                mspec.special_kind,
                SpecialMoveKind::Curse
                    | SpecialMoveKind::Encore
                    | SpecialMoveKind::Disable
                    | SpecialMoveKind::PerishSong
                    | SpecialMoveKind::DestinyBond
            )
            && mspec.primary_vol != vol_bit::DESTINY_BOND
            && (self.mons[def_idx].sub_hp() == 0 || mspec.bypasses_sub)
        {
            let seed_immune = (mspec.primary_vol & vol_bit::LEECH_SEED) != 0
                && (table.mons[def_idx].type_mask & (1u32 << (table.grass_type & 31))) != 0;
            let night_fail = (mspec.primary_vol & vol_bit::NIGHTMARE) != 0
                && self.mons[def_idx].status_raw() != Status::Slp as u8;
            let conf_fail = (mspec.primary_vol & vol_bit::CONFUSION) != 0
                && (self.sides[foe_side].safeguard_turns != 0
                    || (self.mons[def_idx].vol_flags & vol_bit::CONFUSION) != 0);
            if !seed_immune && !night_fail && !conf_fail {
                self.mons[def_idx].vol_flags |= mspec.primary_vol;
                if is_castle && (mspec.primary_vol & vol_bit::FORESIGHT) != 0 {
                    self.mons[att_idx].ext_flags |= ext_vol_bit::FORESIGHT_CASTER;
                }
                if (mspec.primary_vol & vol_bit::CONFUSION) != 0 {
                    self.mons[def_idx].timers[timer_idx::CONFUSION] = (2 + rng.random(4)) as u8;
                }
            }
        }
        let b_target = if mspec.boosts_self { att_idx } else { def_idx };
        if mspec.boosts_self || self.mons[def_idx].sub_hp() == 0 || mspec.bypasses_sub {
            let mist_active =
                !mspec.boosts_self && (self.mons[def_idx].ext_flags & ext_vol_bit::MIST) != 0;
            for &(st_i, delta) in &mspec.boosts {
                if (st_i as usize) < 7 && !(mist_active && delta < 0) {
                    self.apply_boost_stage(b_target, st_i as usize, delta, is_castle);
                }
            }
        }
    }

    /// Resolve a Future Sight strike on `target_side` when its countdown reaches 0 at EOT.
    #[inline]
    fn resolve_future_sight_side(&mut self, table: &MatchTable, target_side: usize, rng: &mut Prng) {
        let is_castle = table.ruleset.is_castle();
        let dmg = self.future_sight_dmg[target_side] as i16;
        let src_slot = self.future_sight_src[target_side].min(2) as usize;
        self.future_sight_dmg[target_side] = 0;
        self.future_sight_src[target_side] = 0;

        let p_idx = self.sides[target_side].active_idx as usize;
        let def_idx = target_side * 3 + p_idx;
        if self.mons[def_idx].hp() <= 0 {
            return;
        }
        // Semi-invulnerable target dodges Future Sight.
        if (self.mons[def_idx].ext_flags & (ext_vol_bit::DIGGING | ext_vol_bit::FLYING)) != 0 {
            return;
        }
        // 90% base accuracy check (using caster's current acc boost if still active, and target's eva boost).
        let src_side = target_side ^ 1;
        let acc_boost = if self.sides[src_side].active_idx as usize == src_slot {
            self.mons[src_side * 3 + src_slot].boosts[stat_idx::ACC]
        } else {
            0
        };
        let def_foresight = (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) != 0;
        let brightpowder = table.item_spec(self.mons[def_idx].item).is_brightpowder;
        let thresh = if is_castle {
            lookup_accuracy_threshold_castle(
                90,
                acc_boost,
                self.mons[def_idx].boosts[stat_idx::EVA],
                false,
                false,
                def_foresight,
                brightpowder,
            )
        } else {
            lookup_accuracy_threshold(
                90,
                acc_boost,
                self.mons[def_idx].boosts[stat_idx::EVA],
                false,
                false,
                def_foresight,
                brightpowder,
            )
        };
        if thresh < 255 && !rng.random_chance(thresh as u32, 256) {
            return;
        }
        let mut hit_dmg = dmg;
        let hitting_sub = self.mons[def_idx].sub_hp() > 0;
        if !hitting_sub
            && hit_dmg >= self.mons[def_idx].hp()
            && table.item_spec(self.mons[def_idx].item).is_focus_band
            && rng.random_chance(30, 256)
        {
            hit_dmg = (self.mons[def_idx].hp() - 1).max(0);
        }
        self.mons[def_idx].apply_damage_branchless(hit_dmg, false);
        if self.mons[def_idx].hp() <= 0 {
            self.sides[target_side].alive_mask &= !(1 << p_idx);
            let foe_idx = src_side * 3 + self.sides[src_side].active_idx as usize;
            self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
            if is_castle {
                self.mons[foe_idx].vol_flags &= !vol_bit::FORESIGHT;
            }
            self.mons[foe_idx].set_partially_trapped(false);
            self.mons[foe_idx].ext_flags &= !ext_vol_bit::ATTRACT;
        }
    }

    /// Gym Leader's Castle PBS (`PS.cgi`) end-of-turn `Residual` in STRICT `1P -> 2P` fixed order
    /// (spec #2, #5, #6, #7, #31, #32, #36, #49):
    /// 1. Future Sight (`1P -> 2P`)
    /// 2. Screens & Safeguard expiry (`1P -> 2P`)
    /// 3. Weather / Sandstorm (`1P -> 2P`; hits Digging mons; 1v1 mutual KO = DRAW)
    /// 4. Perish Song (`1P -> 2P`; 1v1 mutual KO honors `perish_dbond_lastmon_forfeit`)
    /// 5. Per-side `[Items (Leftovers -> Healing Berries hp < floor(M/2) -> Status Berries -> Mystery Berry) -> Freeze Thaw]` (`1P` then `2P`)
    /// 6. Disable / Encore countdown & Stall reset (`1P -> 2P`)
    #[inline]
    fn exec_residuals_castle(&mut self, table: &MatchTable, rng: &mut Prng) {
        // 1. Future Sight (1P -> 2P)
        for side in 0..2usize {
            if self.future_sight_turns[side] > 0 {
                self.future_sight_turns[side] -= 1;
                if self.future_sight_turns[side] == 0 {
                    self.resolve_future_sight_side(table, side, rng);
                }
            }
        }
        {
            let a0 = self.sides[0].alive_mask;
            let a1 = self.sides[1].alive_mask;
            if a0 == 0 && a1 == 0 {
                self.finish_outcome(3);
                return;
            } else if a0 == 0 {
                self.finish_outcome(2);
                return;
            } else if a1 == 0 {
                self.finish_outcome(1);
                return;
            }
        }

        // 2. Screens & Safeguard expiry (1P -> 2P)
        for side in 0..2usize {
            self.sides[side].reflect_turns = self.sides[side].reflect_turns.saturating_sub(1);
            self.sides[side].lightscreen_turns =
                self.sides[side].lightscreen_turns.saturating_sub(1);
            self.sides[side].safeguard_turns = self.sides[side].safeguard_turns.saturating_sub(1);
        }

        // 3. Weather / Sandstorm (1P -> 2P; Castle #49 `Probe 12E2`: Digging mon TAKES Sandstorm!)
        if self.weather_kind != CompactWeather::None {
            self.weather_turns = self.weather_turns.saturating_sub(1);
            if self.weather_turns == 0 {
                self.weather_kind = CompactWeather::None;
            } else if self.weather_kind == CompactWeather::Sandstorm {
                for side in 0..2usize {
                    let p_idx = self.sides[side].active_idx as usize;
                    let idx = side * 3 + p_idx;
                    if self.mons[idx].hp() > 0
                        && (table.mons[idx].type_mask & table.sandstorm_immune_mask) == 0
                    {
                        let sand_dmg = table.mons[idx].quantum_1_8;
                        self.mons[idx].apply_damage_branchless(sand_dmg, true);
                        if self.mons[idx].hp() <= 0 {
                            self.sides[side].alive_mask &= !(1 << p_idx);
                            let foe_side = side ^ 1;
                            let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                            self.mons[foe_idx].vol_flags &=
                                !(vol_bit::TRAPPED_OR_LOCKED | vol_bit::FORESIGHT);
                            self.mons[foe_idx].set_partially_trapped(false);
                            self.mons[foe_idx].ext_flags &= !ext_vol_bit::ATTRACT;
                        }
                    }
                }
                let a0 = self.sides[0].alive_mask;
                let a1 = self.sides[1].alive_mask;
                if a0 == 0 && a1 == 0 {
                    // Castle #2 (`Probe 2B`): 1v1 Sandstorm mutual KO = DRAW!
                    self.finish_outcome(3);
                    return;
                } else if a0 == 0 {
                    self.finish_outcome(2);
                    return;
                } else if a1 == 0 {
                    self.finish_outcome(1);
                    return;
                }
            }
        }

        // 4. Perish Song (1P -> 2P, Castle #7 `Probe 12I`)
        for side in 0..2usize {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() > 0 && (self.mons[idx].vol_flags & vol_bit::PERISH_SONG) != 0 {
                let rem = self.mons[idx].perish_turns().saturating_sub(1);
                self.mons[idx].set_perish_turns(rem);
                if rem == 0 {
                    self.mons[idx].mark_fainted();
                    self.sides[side].alive_mask &= !(1 << p_idx);
                    let foe_side = side ^ 1;
                    let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                    self.mons[foe_idx].vol_flags &=
                        !(vol_bit::TRAPPED_OR_LOCKED | vol_bit::FORESIGHT);
                    self.mons[foe_idx].set_partially_trapped(false);
                    self.mons[foe_idx].ext_flags &= !ext_vol_bit::ATTRACT;
                }
            }
        }
        {
            let a0 = self.sides[0].alive_mask;
            let a1 = self.sides[1].alive_mask;
            if a0 == 0 && a1 == 0 {
                let forfeit_rule = table
                    .ruleset
                    .castle_overlay()
                    .is_some_and(|c| c.perish_dbond_lastmon_forfeit);
                let p1_cast = (self.turn_extra_flags & (1 << 2)) != 0;
                let p2_cast = (self.turn_extra_flags & (1 << 3)) != 0;
                let out = if forfeit_rule && (p1_cast ^ p2_cast) {
                    if p1_cast { 2 } else { 1 }
                } else {
                    3
                };
                self.finish_outcome(out);
                return;
            } else if a0 == 0 {
                self.finish_outcome(2);
                return;
            } else if a1 == 0 {
                self.finish_outcome(1);
                return;
            }
        }
        self.clear_stale_perish_bits();

        // 5. Per-side [Items -> Freeze natural thaw] loop (1P items -> 1P thaw -> 2P items -> 2P thaw, spec #6).
        for side in 0..2usize {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() <= 0 {
                continue;
            }
            let spec = &table.mons[idx];
            let item = self.mons[idx].item;
            if item != 0 {
                let ispec = table.item_spec(item);
                // Leftovers
                if ispec.is_leftovers && self.mons[idx].hp() < spec.maxhp {
                    self.mons[idx].hp_pool[0] =
                        (self.mons[idx].hp() + spec.quantum_1_16).min(spec.maxhp);
                }
                // Healing berries: strict `hp < floor(maxhp / 2)` (spec #31, `Probe 9F` & `E_20261028`).
                if self.mons[idx].item != 0 && self.mons[idx].hp() < spec.maxhp / 2 {
                    let heal = if ispec.is_goldberry {
                        30
                    } else if ispec.is_berry_juice {
                        20
                    } else if ispec.is_berry {
                        10
                    } else {
                        0
                    };
                    if heal > 0 {
                        self.mons[idx].hp_pool[0] = (self.mons[idx].hp() + heal).min(spec.maxhp);
                        self.mons[idx].item = 0;
                    }
                }
                // Status berries (e.g., after second-mover Rest)
                if self.mons[idx].item != 0 {
                    self.check_status_berry_castle(table, side);
                }
                // Mystery Berry (spec #36, `Probe 9F`): scan slots 0..num_moves, +5 PP with NO maxpp clamp!
                if self.mons[idx].item != 0 && table.item_spec(self.mons[idx].item).is_mystery_berry
                {
                    let nm = spec.num_moves as usize;
                    if let Some(z_slot) = (0..nm).find(|&m| self.mons[idx].pp[m] == 0) {
                        self.mons[idx].pp[z_slot] = 5;
                        self.mons[idx].item = 0;
                    }
                }
            }
            // Freeze natural thaw (25/256) immediately after this side's items.
            if self.mons[idx].hp() > 0
                && self.mons[idx].status_raw() == Status::Frz as u8
                && rng.random_chance(25, 256)
            {
                self.mons[idx].set_status(Status::None as u8, 0);
            }
        }

        // 6. Disable / Encore countdown & Stall reset (1P -> 2P).
        for side in 0..2usize {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() <= 0 {
                continue;
            }
            let spec = &table.mons[idx];
            if (self.mons[idx].vol_flags & vol_bit::DISABLE) != 0 {
                let rem = self.mons[idx].timers[timer_idx::DISABLE].saturating_sub(1);
                self.mons[idx].timers[timer_idx::DISABLE] = rem;
                if rem == 0 {
                    self.mons[idx].vol_flags &= !vol_bit::DISABLE;
                    self.mons[idx].set_disable_slot(0);
                }
            }
            if (self.mons[idx].vol_flags & vol_bit::ENCORE) != 0 {
                let enc_slot = self.mons[idx].encore_slot() as usize;
                let rem = self.mons[idx].timers[timer_idx::ENCORE].saturating_sub(1);
                self.mons[idx].timers[timer_idx::ENCORE] = rem;
                if rem == 0
                    || enc_slot >= spec.num_moves as usize
                    || self.mons[idx].pp[enc_slot] == 0
                {
                    self.mons[idx].vol_flags &= !vol_bit::ENCORE;
                    self.mons[idx].timers[timer_idx::ENCORE] = 0;
                    self.mons[idx].set_encore_slot(0);
                }
            }
            if (self.mons[idx].vol_flags & (vol_bit::PROTECT | vol_bit::ENDURE)) == 0 {
                self.mons[idx].set_stall_counter_ext(0);
            }
            self.mons[idx].vol_flags &= !(vol_bit::PROTECT | vol_bit::ENDURE | vol_bit::FLINCH);
        }

        self.check_faints(table, rng);
    }

    /// Gen 2 end-of-turn `Residual`: dispatches to [`Self::exec_residuals_castle`] in Castle PBS mode
    /// (`1P -> 2P` fixed order) or executes exact PS/Stadium 2 speed-ordered residuals.
    #[inline]
    fn exec_residuals(&mut self, table: &MatchTable, rng: &mut Prng) {
        if table.ruleset.fixed_eot_order() {
            self.exec_residuals_castle(table, rng);
            return;
        }
        let s0 = self.mon_action_speed(table, 0, self.sides[0].active_idx as usize);
        let s1 = self.mon_action_speed(table, 1, self.sides[1].active_idx as usize);
        let first = if s0 == s1 {
            (rng.random(2) == 1) as usize
        } else {
            (s0 < s1) as usize
        };
        let order = [first, first ^ 1];

        // Order 2: Weather countdown and Sandstorm damage (both sides damaged before faint batch).
        if self.weather_kind != CompactWeather::None {
            self.weather_turns = self.weather_turns.saturating_sub(1);
            if self.weather_turns == 0 {
                self.weather_kind = CompactWeather::None;
            } else if self.weather_kind == CompactWeather::Sandstorm {
                for &side in &order {
                    let p_idx = self.sides[side].active_idx as usize;
                    let idx = side * 3 + p_idx;
                    // PS: underground Digging mon is immune to Sandstorm!
                    if self.mons[idx].hp() > 0
                        && (self.mons[idx].ext_flags & ext_vol_bit::DIGGING) == 0
                        && (table.mons[idx].type_mask & table.sandstorm_immune_mask) == 0
                    {
                        let sand_dmg = table.mons[idx].quantum_1_8;
                        self.mons[idx].apply_damage_branchless(sand_dmg, true);
                        if self.mons[idx].hp() <= 0 {
                            self.sides[side].alive_mask &= !(1 << p_idx);
                            let foe_side = side ^ 1;
                            let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                            self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
                            self.mons[foe_idx].set_partially_trapped(false);
                            self.mons[foe_idx].ext_flags &= !ext_vol_bit::ATTRACT;
                        }
                    }
                }
                let a0 = self.sides[0].alive_mask;
                let a1 = self.sides[1].alive_mask;
                if a0 == 0 && a1 == 0 {
                    // Matches Battle::faint_messages: faint_queue = [faster, slower],
                    // so winner = 1 - last_faint.side = faster (`order[0]`).
                    self.finish_outcome(if order[0] == 0 { 1 } else { 2 });
                    return;
                } else if a0 == 0 {
                    self.finish_outcome(2);
                    return;
                } else if a1 == 0 {
                    self.finish_outcome(1);
                    return;
                }
            }
        }

        // Order 3: Future Sight strike (if active).
        for &side in &order {
            if self.future_sight_turns[side] > 0 {
                self.future_sight_turns[side] -= 1;
                if self.future_sight_turns[side] == 0 {
                    self.resolve_future_sight_side(table, side, rng);
                }
            }
        }
        {
            let a0 = self.sides[0].alive_mask;
            let a1 = self.sides[1].alive_mask;
            if a0 == 0 && a1 == 0 {
                self.finish_outcome(if order[0] == 0 { 1 } else { 2 });
                return;
            } else if a0 == 0 {
                self.finish_outcome(2);
                return;
            } else if a1 == 0 {
                self.finish_outcome(1);
                return;
            }
        }

        // Order 4: Perish Song countdown (both sides resolve before faint batch, matching Battle::field_event).
        for &side in &order {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() > 0 && (self.mons[idx].vol_flags & vol_bit::PERISH_SONG) != 0 {
                let rem = self.mons[idx].perish_turns().saturating_sub(1);
                self.mons[idx].set_perish_turns(rem);
                if rem == 0 {
                    self.mons[idx].mark_fainted();
                    self.sides[side].alive_mask &= !(1 << p_idx);
                    let foe_side = side ^ 1;
                    let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                    self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
                    self.mons[foe_idx].set_partially_trapped(false);
                    self.mons[foe_idx].ext_flags &= !ext_vol_bit::ATTRACT;
                }
            }
        }
        {
            let a0 = self.sides[0].alive_mask;
            let a1 = self.sides[1].alive_mask;
            if a0 == 0 && a1 == 0 {
                self.finish_outcome(if order[0] == 0 { 1 } else { 2 });
                return;
            } else if a0 == 0 {
                self.finish_outcome(2);
                return;
            } else if a1 == 0 {
                self.finish_outcome(1);
                return;
            }
        }
        self.clear_stale_perish_bits();

        // Order 5: Leftovers and Mint/Miracle Berry in speed order.
        for &side in &order {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() <= 0 {
                continue;
            }
            let spec = &table.mons[idx];
            if table.item_spec(self.mons[idx].item).is_leftovers
                && self.mons[idx].hp() < spec.maxhp
            {
                self.mons[idx].hp_pool[0] =
                    (self.mons[idx].hp() + spec.quantum_1_16).min(spec.maxhp);
            }
            self.check_status_berry_trigger(table, side);
        }

        // Order 7: Freeze 25/256 thaw in speed order.
        for &side in &order {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() > 0
                && self.mons[idx].status_raw() == Status::Frz as u8
                && rng.random_chance(25, 256)
            {
                let prev_res = if (self.mons[idx].status_counter() & 0x10) != 0 {
                    self.mons[idx].status_counter()
                } else {
                    0
                };
                self.mons[idx].set_status(Status::None as u8, prev_res);
            }
        }

        // Order 10: Healing berries (Gold Berry +30, Berry Juice +20, Berry +10), PS Partial Trap, Disable countdown.
        for &side in &order {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() <= 0 {
                continue;
            }
            let spec = &table.mons[idx];
            if self.mons[idx].item != 0 && self.mons[idx].hp() <= spec.maxhp / 2 {
                let ispec = table.item_spec(self.mons[idx].item);
                let heal = if ispec.is_goldberry {
                    30
                } else if ispec.is_berry_juice {
                    20
                } else if ispec.is_berry {
                    10
                } else {
                    0
                };
                if heal > 0 {
                    self.mons[idx].hp_pool[0] = (self.mons[idx].hp() + heal).min(spec.maxhp);
                    self.mons[idx].item = 0;
                }
            }
            if self.mons[idx].partially_trapped()
                && self.mons[idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] > 0
            {
                let foe_side = side ^ 1;
                let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                if self.mons[foe_idx].hp() <= 0 {
                    self.mons[idx].set_partially_trapped(false);
                } else {
                    let rem = self.mons[idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS];
                    if rem <= 1 {
                        self.mons[idx].set_partially_trapped(false);
                    } else {
                        self.mons[idx].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] = rem - 1;
                        self.mons[idx].apply_damage_branchless(spec.quantum_1_16, true);
                        if self.mons[idx].hp() <= 0 {
                            self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
                            self.mons[foe_idx].set_partially_trapped(false);
                            self.mons[foe_idx].ext_flags &= !ext_vol_bit::ATTRACT;
                        }
                    }
                }
            }
            if (self.mons[idx].vol_flags & vol_bit::DISABLE) != 0 {
                let rem = self.mons[idx].timers[timer_idx::DISABLE].saturating_sub(1);
                self.mons[idx].timers[timer_idx::DISABLE] = rem;
                if rem == 0 {
                    self.mons[idx].vol_flags &= !vol_bit::DISABLE;
                    self.mons[idx].set_disable_slot(0);
                }
            }
        }

        // Order 13: Encore countdown, PS Outrage LockedMoveEnd confusion, PS Lock-On expiry, Stall reset, flag clear.
        for &side in &order {
            let p_idx = self.sides[side].active_idx as usize;
            let idx = side * 3 + p_idx;
            if self.mons[idx].hp() <= 0 {
                continue;
            }
            let spec = &table.mons[idx];
            if (self.mons[idx].vol_flags & vol_bit::ENCORE) != 0 {
                let enc_slot = self.mons[idx].encore_slot() as usize;
                let rem = self.mons[idx].timers[timer_idx::ENCORE].saturating_sub(1);
                self.mons[idx].timers[timer_idx::ENCORE] = rem;
                if rem == 0
                    || enc_slot >= spec.num_moves as usize
                    || self.mons[idx].pp[enc_slot] == 0
                {
                    self.mons[idx].vol_flags &= !vol_bit::ENCORE;
                    self.mons[idx].timers[timer_idx::ENCORE] = 0;
                    self.mons[idx].set_encore_slot(0);
                }
            }
            if (self.mons[idx].ext_flags & ext_vol_bit::LOCKED_OUTRAGE) != 0 {
                if self.mons[idx].status_raw() == Status::Slp as u8 {
                    // `LockedMoveResidual`: sleep silently strips `lockedmove` without confusion.
                    self.mons[idx].ext_flags &= !ext_vol_bit::LOCKED_OUTRAGE;
                    self.mons[idx].ext_timers[ext_timer_idx::LOCKED_TURNS] = 0;
                    if !self.mons[idx].is_multi_turn_locked() {
                        self.mons[idx].set_locked_move_slot(0);
                    }
                } else {
                    let rem = self.mons[idx].ext_timers[ext_timer_idx::LOCKED_TURNS]
                        .saturating_sub(1);
                    self.mons[idx].ext_timers[ext_timer_idx::LOCKED_TURNS] = rem;
                    if rem == 0 {
                        self.mons[idx].ext_flags &= !ext_vol_bit::LOCKED_OUTRAGE;
                        if !self.mons[idx].is_multi_turn_locked() {
                            self.mons[idx].set_locked_move_slot(0);
                        }
                        if self.sides[side].safeguard_turns == 0 {
                            self.mons[idx].vol_flags |= vol_bit::CONFUSION;
                            self.mons[idx].timers[timer_idx::CONFUSION] =
                                (2 + rng.random(4)) as u8;
                        }
                    }
                }
            }
            if (self.mons[idx].ext_flags & ext_vol_bit::LOCK_ON) != 0 {
                let rem = self.mons[idx].lock_on_turns().saturating_sub(1);
                self.mons[idx].set_lock_on_turns(rem);
                if rem == 0 {
                    self.mons[idx].ext_flags &= !ext_vol_bit::LOCK_ON;
                }
            }
            if (self.mons[idx].vol_flags & (vol_bit::PROTECT | vol_bit::ENDURE)) == 0 {
                self.mons[idx].set_stall_counter(0);
            }
            self.mons[idx].vol_flags &= !(vol_bit::PROTECT | vol_bit::ENDURE | vol_bit::FLINCH);
        }

        // Decrement Reflect / Light Screen / Safeguard timers.
        for side in 0..2usize {
            self.sides[side].reflect_turns = self.sides[side].reflect_turns.saturating_sub(1);
            self.sides[side].lightscreen_turns =
                self.sides[side].lightscreen_turns.saturating_sub(1);
            self.sides[side].safeguard_turns = self.sides[side].safeguard_turns.saturating_sub(1);
        }

        self.check_faints(table, rng);
    }

    /// Resolve a simultaneous dual replacement Spikes faint:
    /// - Castle (`Spec #2`): simultaneous Spikes faint -> `3` (`DRAW`).
    /// - PS: action-speed order (`Battle::RunSwitch`), faster incoming mon faints first -> slower wins.
    #[inline]
    fn resolve_dual_spikes_faint(&self, table: &MatchTable, rng: &mut Prng) -> u8 {
        if table.ruleset.is_castle() {
            return 3;
        }
        let spd0 = self.mon_action_speed(table, 0, self.sides[0].active_idx as usize);
        let spd1 = self.mon_action_speed(table, 1, self.sides[1].active_idx as usize);
        if spd0 == spd1 {
            if rng.random(2) == 0 { 1 } else { 2 }
        } else if spd0 > spd1 {
            2 // P1 incoming mon faster -> faints first -> P2Win
        } else {
            1 // P2 incoming mon faster -> faints first -> P1Win
        }
    }

    #[inline]
    fn check_faints(&mut self, table: &MatchTable, rng: &mut Prng) {
        if self.outcome != 0 {
            return;
        }
        for side in 0..2usize {
            let p_idx = self.sides[side].active_idx as usize;
            let m_idx = side * 3 + p_idx;
            if self.mons[m_idx].hp() <= 0 {
                self.mons[m_idx].mark_fainted();
                self.sides[side].alive_mask &= !(1 << p_idx);
                // Gen 2: any faint cancels remaining queued moves (`Battle::faint_messages` `queue_cancel_move`).
                self.sides[0].side_flags &= !0x70;
                self.sides[1].side_flags &= !0x70;
                self.bp_switch_flags &= !0x3F;
                // Fainting releases Mean Look / Spider Web, partial trap, Attract, Lock-On on opposing active mon
                // (and Foresight in Castle PBS).
                let foe_side = side ^ 1;
                let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
                if table.ruleset.is_castle() {
                    self.mons[foe_idx].vol_flags &= !vol_bit::FORESIGHT;
                }
                self.mons[foe_idx].set_partially_trapped(false);
                self.mons[foe_idx].ext_flags &= !(ext_vol_bit::ATTRACT
                    | ext_vol_bit::LOCK_ON
                    | ext_vol_bit::LOCKED_FIRST_HIT_RECV);
                self.mons[foe_idx].set_lock_on_turns(0);
            }
        }
        let a0 = self.sides[0].alive_mask;
        let a1 = self.sides[1].alive_mask;
        if a0 == 0 && a1 == 0 {
            let out = if let Some(cfg) = table.ruleset.castle_overlay() {
                // Castle Layer 1 + Layer 2 overlay rules (spec §2-A):
                if self.sides[0].last_move_selfko != u8::MAX {
                    2 // Self-Destruct / Explosion user loses
                } else if self.sides[1].last_move_selfko != u8::MAX {
                    1
                } else if (self.turn_extra_flags & 0x30) != 0 {
                    // 1v1 Destiny Bond mutual KO
                    if cfg.perish_dbond_lastmon_forfeit {
                        if (self.turn_extra_flags & 0x10) != 0 { 2 } else { 1 }
                    } else {
                        3
                    }
                } else if (self.turn_extra_flags & 0x01) != 0 {
                    // 1v1 recoil mutual KO (`Probe 2B`, `H_20260407`)
                    if cfg.recoil_1v1_attacker_wins {
                        let att_side = (self.turn_extra_flags >> 1) & 1;
                        if att_side == 0 { 1 } else { 2 }
                    } else {
                        3 // Raw PS.cgi DRAW
                    }
                } else {
                    3 // Tie / DRAW
                }
            } else {
                // Stadium 2 Self-KO Clause: user of Self-Destruct / Explosion loses.
                if self.sides[0].last_move_selfko != u8::MAX {
                    2 // P2Win
                } else if self.sides[1].last_move_selfko != u8::MAX {
                    1 // P1Win
                } else {
                    3 // Tie
                }
            };
            self.finish_outcome(out);
        } else if a0 == 0 {
            self.finish_outcome(2); // P2Win
        } else if a1 == 0 {
            self.finish_outcome(1); // P1Win
        } else {
            self.clear_stale_perish_bits();
            // Clear Self-KO tracker once non-terminal faints resolve.
            self.sides[0].last_move_selfko = u8::MAX;
            self.sides[1].last_move_selfko = u8::MAX;
            // Process single-bench auto-switches (and resulting Spikes faints).
            let mut any_auto_switch = false;
            for side in 0..2usize {
                let p_idx = self.sides[side].active_idx as usize;
                if (self.sides[side].alive_mask & (1 << p_idx)) == 0
                    && self.sides[side].alive_mask.count_ones() == 1
                {
                    let next_slot = (self.sides[side].alive_mask.trailing_zeros() as u8).min(2);
                    self.exec_switch(table, side, next_slot, SwitchKind::FaintReplace, rng);
                    any_auto_switch = true;
                    if self.mons[side * 3 + next_slot as usize].hp() <= 0 {
                        self.sides[side].alive_mask &= !(1 << next_slot);
                    }
                }
            }
            if any_auto_switch {
                let a0 = self.sides[0].alive_mask;
                let a1 = self.sides[1].alive_mask;
                if a0 == 0 && a1 == 0 {
                    let out = self.resolve_dual_spikes_faint(table, rng);
                    self.finish_outcome(out);
                    return;
                } else if a0 == 0 {
                    self.finish_outcome(2);
                    return;
                } else if a1 == 0 {
                    self.finish_outcome(1);
                    return;
                }
            }

            let mut any_forced_switch = false;
            for side in 0..2usize {
                let p_idx = self.sides[side].active_idx as usize;
                if (self.sides[side].alive_mask & (1 << p_idx)) == 0 {
                    self.sides[side].request = 2; // Forced Switch request
                    any_forced_switch = true;
                } else {
                    self.sides[side].request = 1;
                }
            }
            if any_forced_switch {
                for side in 0..2usize {
                    if self.sides[side].request != 2 {
                        self.sides[side].request = 0; // Wait
                    }
                }
            }
        }
    }

    /// Resolve a forced switch phase (`CHANGE` after faint OR explicit `BATONN` Baton Pass switch
    /// when `request == 2`), executing any pending mid-turn second mover action and EOT residuals.
    pub fn step_switch(&mut self, table: &MatchTable, switch_actions: [u8; 2]) {
        let is_castle = table.ruleset.is_castle();
        let mut rng = Prng::new(self.prng_seed);
        // Any forced-switch prompt occurs strictly after a non-terminal faint batch or BP pause.
        self.sides[0].last_move_selfko = u8::MAX;
        self.sides[1].last_move_selfko = u8::MAX;
        let bp_fix = (self.sides[0].side_flags & SIDE0_DISABLE_BP_FIX) == 0;
        let both_forced = self.sides[0].request == 2 && self.sides[1].request == 2;
        for side in 0..2usize {
            if self.sides[side].request == 2 {
                let act = switch_actions[side];
                let slot = if act >= 4 {
                    (act - 4).min(2)
                } else {
                    (self.sides[side].alive_mask.trailing_zeros() as u8).min(2)
                };
                let is_bp = (self.bp_switch_flags & (1u8 << side)) != 0 || {
                    let act_idx = side * 3 + (self.sides[side].active_idx.min(2) as usize);
                    bp_fix && self.mons[act_idx].hp() > 0
                };
                self.bp_switch_flags &= !(1u8 << side);
                if (self.sides[side].alive_mask & (1 << slot)) != 0 {
                    self.exec_switch(
                        table,
                        side,
                        slot,
                        if is_bp {
                            SwitchKind::BatonPass
                        } else {
                            SwitchKind::FaintReplace
                        },
                        &mut rng,
                    );
                }
            }
            self.sides[side].request = 1;
        }
        if both_forced {
            let i0 = self.sides[0].active_idx as usize;
            let i1 = self.sides[1].active_idx as usize;
            if self.mons[i0].hp() <= 0 && self.mons[3 + i1].hp() <= 0 {
                self.sides[0].alive_mask &= !(1 << i0);
                self.sides[1].alive_mask &= !(1 << i1);
                if self.sides[0].alive_mask == 0 && self.sides[1].alive_mask == 0 {
                    let out = self.resolve_dual_spikes_faint(table, &mut rng);
                    self.finish_outcome(out);
                    self.prng_seed = rng.seed();
                    return;
                }
            }
        }
        self.check_faints(table, &mut rng);
        // If this was a mid-turn self-switch (e.g. faster Baton Pass) and the other side
        // had a pending move queued for the second half of the turn, execute it now!
        if bp_fix && self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1
        {
            for s in 0..2usize {
                let flags = self.sides[s].side_flags;
                if (flags & 0x10) != 0 {
                    self.sides[s].side_flags &= !0x70;
                    let m_slot = ((flags >> 5) & 0x03) as usize;
                    let forced_str = (self.bp_switch_flags & (1u8 << (4 + s))) != 0
                        || self.is_forced_struggle(table, s);
                    self.bp_switch_flags &= !(1u8 << (4 + s));
                    self.exec_move_inner(table, s, m_slot, true, forced_str, false, &mut rng);
                    // Clear first-mover BP marker after second mover has attacked (Castle Pursuit 2x consumed).
                    self.bp_switch_flags &= !(1u8 << (2 + (s ^ 1)));
                    // Castle #32②: flush deferred first-mover Rest berry right after 2nd mover acts,
                    // BEFORE any 2nd-mover BATONN pause!
                    if is_castle && (self.bp_switch_flags & 0xC0) != 0 {
                        for rs in 0..2usize {
                            if (self.bp_switch_flags & (1u8 << (6 + rs))) != 0 {
                                self.bp_switch_flags &= !(1u8 << (6 + rs));
                                self.check_status_berry_castle(table, rs);
                            }
                        }
                    }
                    // If the second mover ITSELF used Baton Pass with >= 2 bench options, pause now!
                    if (self.bp_switch_flags & (1u8 << s)) != 0 {
                        self.prng_seed = rng.seed();
                        return;
                    }
                    self.check_faints(table, &mut rng);
                    break;
                }
            }
        }
        if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
            if (self.sides[0].side_flags & 0x08) == 0 {
                self.sides[0].side_flags |= 0x08;
                self.exec_residuals(table, &mut rng);
                self.check_faints(table, &mut rng);
            }
            if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
                self.sides[0].side_flags &= !0x08;
                self.finish_turn_advance(table, &mut rng);
                self.clear_turn_transients();
            } else {
                self.hurt_damage_this_turn = [0, 0];
                self.hurt_cat_this_turn = [0, 0];
            }
        }
        self.prng_seed = rng.seed();
    }

    /// Enumerate legal compact action codes for `side` (`0` or `1`) into a stack buffer.
    /// Action codes:
    /// - `0..4`: move slot `0..4`
    /// - `4..7`: switch to party slot `0..3` (`action - 4`)
    /// - `7`: `"pass"` (when waiting for opponent's forced switch / `BATONN`)
    #[inline]
    pub fn legal_actions(&self, table: &MatchTable, side: usize, out: &mut [u8; 7]) -> usize {
        if self.outcome != 0 {
            return 0;
        }
        let side_st = &self.sides[side];
        if side_st.request == 0 {
            out[0] = 7; // pass
            return 1;
        }
        if side_st.request == 2 {
            let mut n = 0;
            let order = self.party_order(side);
            for &slot in &order {
                if slot != side_st.active_idx && (side_st.alive_mask & (1 << slot)) != 0 {
                    out[n] = 4 + slot;
                    n += 1;
                }
            }
            return n;
        }
        let mut n = 0;
        let idx = side * 3 + side_st.active_idx as usize;
        let mon = &self.mons[idx];
        if mon.must_recharge() {
            out[0] = 0; // Forced Recharge turn cannot choose other moves or switch
            return 1;
        }
        if mon.is_multi_turn_locked() {
            // Forced Outrage / Thrash / Petal Dance / Rollout / Dig / Fly continuation cannot switch.
            out[0] = mon.locked_move_slot();
            return 1;
        }
        let num_moves = table.mons[idx].num_moves as usize;
        let is_castle = table.ruleset.is_castle();
        if (mon.vol_flags & vol_bit::ENCORE) != 0 {
            let enc_slot = mon.encore_slot() as usize;
            // Castle #54①: Encore + Disable does NOT force Struggle (still offers enc_slot).
            let enc_usable = enc_slot < num_moves
                && mon.pp[enc_slot] > 0
                && (is_castle || !mon.is_slot_disabled(enc_slot));
            if enc_usable {
                out[0] = enc_slot as u8;
                n = 1;
            }
        } else {
            for m in 0..num_moves {
                if mon.pp[m] > 0 && !mon.is_slot_disabled(m) {
                    out[n] = m as u8;
                    n += 1;
                }
            }
        }
        if n == 0 {
            out[0] = 0; // Struggle fallback
            n = 1;
        }
        if !mon.is_trapped() {
            let order = self.party_order(side);
            for &slot in &order {
                if slot != side_st.active_idx && (side_st.alive_mask & (1 << slot)) != 0 {
                    out[n] = 4 + slot;
                    n += 1;
                }
            }
        }
        n
    }

    /// Apply a pair of action codes `[p1_action, p2_action]`, handling both normal
    /// turns and forced-switch / `BATONN` phases.
    #[inline]
    pub fn apply_actions(&mut self, table: &MatchTable, actions: [u8; 2]) {
        if self.sides[0].request == 2 || self.sides[1].request == 2 {
            self.step_switch(table, actions);
        } else {
            self.step_turn(table, actions);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_battle_layout_is_exact_256_bytes_and_64b_aligned() {
        assert_eq!(std::mem::size_of::<CompactMon>(), 32);
        assert_eq!(std::mem::size_of::<CompactSide>(), 12);
        assert_eq!(std::mem::size_of::<CompactBattle>(), 256);
        assert_eq!(std::mem::align_of::<CompactBattle>(), 64);
    }

    #[test]
    fn step_turn_respects_switches_priority_and_quick_claw() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };
        let p1 = vec![
            mk("Snorlax", 55, "Quick Claw", &["Double-Edge", "Protect", "Rest", "Curse"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Ice Beam", "Spikes", "Explosion"]),
        ];
        let p2 = vec![
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Protect"]),
            mk("Jolteon", 55, "Leftovers", &["Thunderbolt", "Quick Attack", "Substitute", "Baton Pass"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();
        let (table, base) = CompactBattle::from_battle(&b, &dex);

        // Switch beats +2 Protect (slot 1 moves first onto incoming Zapdos; Marowak Protect fails because it moves second, consuming PP cleanly).
        let mut cb_sw = base;
        cb_sw.step_turn(&table, [5, 3]);
        assert_eq!(cb_sw.sides[0].active_idx, 1);
        assert_eq!(cb_sw.mons[3].stall_counter(), 0);
        assert_eq!(cb_sw.mons[3].pp[3], base.mons[3].pp[3] - 1);

        // +2 Protect blocks slower priority-0 Double-Edge (Marowak stays full HP).
        let mut cb_prot = base;
        cb_prot.quick_claw_roll = 0;
        cb_prot.step_turn(&table, [0, 3]);
        assert_eq!(cb_prot.mons[3].hp(), table.mons[3].maxhp);
        assert_eq!(cb_prot.mons[3].stall_counter(), 1);

        // Quick Claw on slow Snorlax (quick_claw_roll != 0, bit 0 = P1 triggers) moves before faster Marowak at priority 0.
        let mut cb_qc = base;
        cb_qc.quick_claw_roll = 1;
        cb_qc.step_turn(&table, [0, 0]);
        assert_eq!(cb_qc.mons[0].hp() < table.mons[0].maxhp, true);
        assert_eq!(cb_qc.mons[3].hp() < table.mons[3].maxhp, true);

        // Switch to Jolteon, then verify Quick Attack (+1) outprioritizes Quick Claw Snorlax Double-Edge (0).
        let mut cb_qa = base;
        cb_qa.step_turn(&table, [2, 5]); // Snorlax Rest (full HP noop), P2 switches Jolteon
        cb_qa.quick_claw_roll = 1;
        // Snorlax asleep? No, Rest fails at full HP; try Double-Edge (0) vs Jolteon Quick Attack (1) at low Snorlax HP
        let mut low_hp = cb_qa;
        low_hp.mons[0].hp_pool[0] = 5;
        low_hp.step_turn(&table, [0, 1]);
        assert_eq!(low_hp.mons[0].hp(), 0);
        assert_eq!(low_hp.mons[4].hp(), table.mons[4].maxhp);
    }

    #[test]
    fn stat_branchless_matches_reference_formula_exhaustively() {
        const STAT_POS: [f64; 7] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        const STAT_NEG: [f64; 7] = [100.0, 66.0, 50.0, 40.0, 33.0, 28.0, 25.0];

        for stored in [1, 15, 95, 130, 178, 255, 358, 500, 999] {
            for boost in -6..=6i8 {
                for stat_idx in 0..5usize {
                    for &unboosted in &[false, true] {
                        for &unmodified in &[false, true] {
                            for &par in &[false, true] {
                                for &brn in &[false, true] {
                                    for &screen in &[false, true] {
                                        for &item in &[
                                            ItemStatKind::None,
                                            ItemStatKind::ThickClub,
                                            ItemStatKind::LightBall,
                                            ItemStatKind::MetalPowder,
                                        ] {
                                            let mut ref_val = stored as f64;
                                            if !unboosted {
                                                if boost >= 0 {
                                                    ref_val =
                                                        (ref_val * STAT_POS[boost as usize]).floor();
                                                } else {
                                                    ref_val = (ref_val * STAT_NEG[(-boost) as usize]
                                                        / 100.0)
                                                        .floor();
                                                }
                                            }
                                            if par && stat_idx == 4 {
                                                ref_val = (ref_val / 4.0).floor();
                                            }
                                            if !unmodified && brn && stat_idx == 0 {
                                                ref_val = (ref_val / 2.0).floor();
                                            }
                                            ref_val = ref_val.clamp(1.0, 999.0);
                                            if !unboosted && screen && (stat_idx == 1 || stat_idx == 3) {
                                                ref_val *= 2.0;
                                            }
                                            if (item == ItemStatKind::ThickClub && stat_idx == 0)
                                                || (item == ItemStatKind::LightBall && stat_idx == 2)
                                            {
                                                ref_val *= 2.0;
                                            } else if item == ItemStatKind::MetalPowder
                                                && (stat_idx == 1 || stat_idx == 3)
                                            {
                                                ref_val = (ref_val * 1.5).floor();
                                            }

                                            let fast_val = get_stat_branchless(
                                                stored, boost, stat_idx, unboosted, unmodified,
                                                false, par, brn, screen, screen, item,
                                            );
                                            assert_eq!(
                                                fast_val, ref_val as i32,
                                                "mismatch at stored={stored}, boost={boost}, stat={stat_idx}, unboosted={unboosted}, unmodified={unmodified}"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn acc_lut_and_reciprocal_damage_rolls_are_bit_exact() {
        // Verify reciprocal multiplication (x * 0x808081) >> 31 matches x / 255
        // for every possible damage (1..=4000) and every roll (217..=255).
        for dmg in 1..=4000i32 {
            for roll in 217..=255u32 {
                let exact = ((dmg as f64) * (roll as f64) / 255.0).floor() as i32;
                let fast = apply_roll_branchless(dmg, roll, false);
                assert_eq!(fast, exact.max(1), "mismatch at dmg={dmg}, roll={roll}");
            }
        }

        // Verify accuracy LUT across stages, Foresight (ignore_positive_evasion), and BrightPowder (-20).
        assert_eq!(lookup_accuracy_threshold(100, 0, 0, false, false, false, false), 255);
        assert_eq!(lookup_accuracy_threshold(100, 0, 0, false, false, false, true), 235);
        assert_eq!(lookup_accuracy_threshold(75, 0, 0, false, false, false, false), 191);
        assert!(
            lookup_accuracy_threshold(100, 0, 2, false, false, false, false) < 255,
            "+2 evasion must lower threshold"
        );
        assert_eq!(
            lookup_accuracy_threshold(100, 0, 2, false, false, true, false),
            255,
            "Foresight drops positive evasion"
        );
        assert!(
            lookup_accuracy_threshold(100, 0, -2, false, false, true, false) == 255,
            "Foresight keeps negative evasion beneficial"
        );
        assert!(
            lookup_accuracy_threshold(100, -1, 0, false, false, false, false) < 255,
            "-1 accuracy lowers threshold"
        );
        assert_eq!(
            lookup_accuracy_threshold(100, -1, 0, true, false, false, false),
            255,
            "ignore_accuracy drops negative accuracy"
        );
    }

    #[test]
    fn compact_battle_from_battle_step_and_hash() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };
        let p1 = vec![
            mk("Snorlax", 55, "Leftovers", &["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
            mk("Zapdos", 50, "Mint Berry", &["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Ice Beam", "Spikes", "Explosion"]),
        ];
        let p2 = vec![
            mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Curse", "Rest"]),
            mk("Raikou", 50, "Miracle Berry", &["Thunderbolt", "Crunch", "Reflect", "Rest"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();

        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);
        let h0 = cb.search_hash();
        // Cloning CompactBattle is a 256-byte Copy.
        let mut cb_clone = cb;
        assert_eq!(cb_clone.search_hash(), h0);

        // Execute a turn in CompactBattle: Snorlax Double-Edge (0) vs Marowak Earthquake (0).
        cb.step_turn(&table, [0, 0]);
        assert_eq!(cb.turn, 2);
        assert_ne!(cb.search_hash(), h0);
        // Original copy is untouched.
        assert_eq!(cb_clone.turn, 1);
        // Switch Snorlax to party slot 1 (action 4 + 1 = 5), Marowak Swords Dance (move 2).
        cb_clone.step_turn(&table, [5, 2]);
        assert_eq!(cb_clone.sides[0].active_idx, 1);
    }

    #[test]
    fn compact_battle_forced_switch_after_ko_steps_safely() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };
        let p1 = vec![
            mk("Cloyster", 50, "Leftovers", &["Explosion", "Surf", "Ice Beam", "Spikes"]),
            mk("Snorlax", 55, "Leftovers", &["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
            mk("Zapdos", 50, "Mint Berry", &["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        ];
        let p2 = vec![
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Curse", "Rest"]),
            mk("Raikou", 50, "Miracle Berry", &["Thunderbolt", "Crunch", "Reflect", "Rest"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();

        // Cloyster uses Explosion to KO both itself (side 0) and Marowak (side 1).
        let id1 = b.active_id(1).unwrap();
        b.poke_mut(id1).hp = 10;
        b.choose(&dex, 0, "move 1").unwrap();
        b.choose(&dex, 1, "move 3").unwrap();

        assert_eq!(b.sides[0].request_state(), Some(RequestKind::Switch));
        assert_eq!(b.sides[1].request_state(), Some(RequestKind::Switch));

        // Also test the PositionSpec / protocol-import representation where `side.active` is `None`.
        for clear_active in [false, true] {
            let mut b_case = b.clone();
            if clear_active {
                b_case.sides[0].active = None;
                b_case.sides[1].active = None;
            }
            let (table, mut cb) = CompactBattle::from_battle(&b_case, &dex);
            for s in 0..2 {
                assert!(
                    cb.sides[s].active_idx < 3,
                    "side {s} active_idx={} must be < 3 (clear_active={clear_active})",
                    cb.sides[s].active_idx
                );
                assert_eq!(cb.sides[s].request, 2);
                assert_eq!(cb.sides[s].alive_mask & (1 << cb.sides[s].active_idx), 0);
                let mut acts = [0u8; 7];
                let n = cb.legal_actions(&table, s, &mut acts);
                assert_eq!(&acts[..n], &[5, 6]);
            }

            // Step forced switch on both sides (including side 1 where active_idx=3 previously panicked).
            cb.apply_actions(&table, [5, 6]);
            assert_eq!(cb.sides[0].active_idx, 1);
            assert_eq!(cb.sides[1].active_idx, 2);
            assert_eq!(cb.sides[0].request, 1);
            assert_eq!(cb.sides[1].request, 1);
            assert_eq!(cb.hurt_damage_this_turn, [0, 0]);
            assert_eq!(cb.hurt_cat_this_turn, [0, 0]);
            assert_eq!(cb.speed_order, 0);
        }
    }

    #[test]
    fn compact_protect_decay_self_moves_freeze_thaw_and_encore_lock() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };
        let p1 = vec![
            mk("Jolteon", 50, "Leftovers", &["Protect", "Thunderbolt", "Agility", "Baton Pass"]),
            mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Ice Beam", "Spikes", "Explosion"]),
        ];
        let p2 = vec![
            mk("Alakazam", 50, "Leftovers", &["Psychic", "Encore", "Disable", "Recover"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();

        // 1. Protect sets stall counter in Battle and imports into CompactBattle; consecutive Protect decays.
        let mut b_prot = b.clone();
        b_prot.choose(&dex, 0, "move 1").unwrap(); // Protect
        b_prot.choose(&dex, 1, "move 1").unwrap(); // Psychic (blocked)
        let (table, cb_imported) = CompactBattle::from_battle(&b_prot, &dex);
        assert_eq!(cb_imported.mons[0].stall_counter(), 1);
        assert_eq!(cb_imported.mons[0]. hp(), table.mons[0].maxhp);

        let mut fail_seen = false;
        for seed in 1..=64u64 {
            let mut cb_try = cb_imported;
            cb_try.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let hp_before = cb_try.mons[0].hp();
            cb_try.step_turn(&table, [0, 0]); // 2nd consecutive Protect vs Psychic
            if cb_try.mons[0].hp() < hp_before {
                fail_seen = true;
                assert_eq!(cb_try.mons[0].stall_counter(), 0);
                break;
            }
        }
        assert!(fail_seen, "2nd consecutive Protect must not succeed 100% of the time");

        // 2. Opponent Protect does NOT block self-targeted moves (Swords Dance / Baton Pass / Curse).
        let (table2, mut cb2) = CompactBattle::from_battle(&b, &dex);
        // Jolteon Protect (0) vs Alakazam Recover/Curse test: switch P2 to Snorlax first, then Jolteon Protect vs Snorlax Curse.
        cb2.step_turn(&table2, [2, 5]); // Jolteon Agility (+2 Spe), P2 switches to Snorlax (slot 1)
        assert_eq!(cb2.mons[0].boosts[4], 2);
        cb2.step_turn(&table2, [0, 2]); // Jolteon Protect vs Snorlax Curse (+1 Atk/+1 Def/-1 Spe)
        assert_eq!(cb2.mons[4].boosts[0], 1, "Snorlax Curse must bypass Jolteon Protect");
        assert_eq!(cb2.mons[4].boosts[1], 1);
        // Next turn: Jolteon Baton Pass (move 3) with 2 alive bench mons pauses in explicit BATONN
        // (`request == 2`, `bp_switch_flags & 1 != 0`), letting P1 choose Marowak (5) or Cloyster (6)
        // while preserving Snorlax's queued Curse (2)!
        cb2.step_turn(&table2, [3, 2]);
        assert_eq!(cb2.sides[0].request, 2, "Baton Pass with 2 bench mons must enter explicit switch request");
        assert_eq!(cb2.sides[1].request, 0);
        assert_ne!(cb2.bp_switch_flags & 1, 0);
        let mut bp_acts = [0u8; 7];
        let n_bp = cb2.legal_actions(&table2, 0, &mut bp_acts);
        assert_eq!(&bp_acts[..n_bp], &[5, 6]);
        cb2.step_switch(&table2, [5, 7]); // Explicitly choose Marowak (slot 1 -> action 5)
        assert_eq!(cb2.sides[0].active_idx, 1);
        assert_eq!(cb2.mons[1].boosts[4], 2, "Marowak must inherit +2 Spe from explicit Baton Pass");
        assert_eq!(cb2.mons[4].boosts[0], 2, "Snorlax queued Curse must execute after explicit Baton Pass");

        // 3. Encore imported from Battle locks legal_actions to the encored move slot and expires.
        let mut b_enc = b.clone();
        b_enc.choose(&dex, 0, "move 3").unwrap(); // Jolteon Agility (slot 2)
        b_enc.choose(&dex, 1, "move 2").unwrap(); // Alakazam Encore -> locks Jolteon into Agility
        let (t_enc, mut cb_enc) = CompactBattle::from_battle(&b_enc, &dex);
        assert_ne!(cb_enc.mons[0].vol_flags & vol_bit::ENCORE, 0);
        assert_eq!(cb_enc.mons[0].encore_slot(), 2);
        let mut acts = [0u8; 7];
        let n = cb_enc.legal_actions(&t_enc, 0, &mut acts);
        // Only move 2 (Agility) plus bench switches (5, 6) are legal!
        assert_eq!(&acts[..n], &[2, 5, 6]);
        for _ in 0..8 {
            if (cb_enc.mons[0].vol_flags & vol_bit::ENCORE) == 0 {
                break;
            }
            cb_enc.step_turn(&t_enc, [2, 3]);
        }
        assert_eq!(cb_enc.mons[0].vol_flags & vol_bit::ENCORE, 0);

        // 4. Freeze 25/256 residual thaw occurs across seeds.
        let (t_frz, cb_frz_base) = CompactBattle::from_battle(&b, &dex);
        let mut thawed = false;
        for seed in 1..=128u64 {
            let mut cb_frz = cb_frz_base;
            cb_frz.mons[0].set_status(Status::Frz as u8, 0);
            cb_frz.prng_seed = seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
            cb_frz.step_turn(&t_frz, [1, 3]); // frozen Jolteon vs Alakazam Recover
            if cb_frz.mons[0].status_raw() == Status::None as u8 {
                thawed = true;
                break;
            }
        }
        assert!(thawed, "Frozen Pokemon must have 25/256 residual thaw chance in CompactBattle");
    }

    #[test]
    fn compact_step_fixes_match_battle_mechanics() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };

        // 1. Non-Ghost Curse, Destiny Bond, Encore, Disable do not fall through via primary_vol.
        {
            let dt = dex_tables(&dex);
            for key in ["curse", "destinybond", "encore", "disable", "perishsong"] {
                let mid = dex.moves.id(key).unwrap();
                assert_eq!(dt.move_specs[mid.0 as usize].primary_vol, 0, "{key} primary_vol must be 0");
            }
            let p1 = vec![mk("Snorlax", 55, "", &["Curse", "Substitute", "Double-Edge", "Rest"])];
            let p2 = vec![mk("Gengar", 55, "", &["Destiny Bond", "Encore", "Disable", "Psychic"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, mut cb) = CompactBattle::from_battle(&b, &dex);
            cb.step_turn(&t, [1, 0]); // Snorlax Sub, Gengar Destiny Bond (both faster Gengar moves first -> Snorlax Sub)
            assert_eq!(cb.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0);
            assert_eq!(cb.mons[3].vol_flags & vol_bit::DESTINY_BOND, 0, "DBond expired after Snorlax moved");
            cb.step_turn(&t, [0, 1]); // Gengar Encore (faster) locks Snorlax into Substitute through Sub, redirecting Curse -> Substitute!
            assert_eq!(cb.mons[0].boosts[0], 0);
            assert_eq!(cb.mons[3].vol_flags & vol_bit::CURSE, 0);
            assert_ne!(cb.mons[0].vol_flags & vol_bit::ENCORE, 0, "Encore must bypass Substitute in Gen 2");
        }

        // 2. Struggle hits Ghost, takes 1/4 recoil on main HP and 1 HP recoil on Sub (Battle parity).
        {
            let p1 = vec![mk("Snorlax", 55, "", &["Tackle"])];
            let p2 = vec![mk("Gengar", 55, "", &["Substitute", "Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, mut cb) = CompactBattle::from_battle(&b, &dex);
            cb.mons[0].pp = [0; 4];
            let gengar_before = cb.mons[3].hp();
            cb.step_turn(&t, [0, 1]); // Struggle vs Splash
            let dmg = gengar_before - cb.mons[3].hp();
            assert!(dmg > 0, "Struggle must hit Ghost (Gengar) typelessly");
            assert_eq!(cb.mons[0].hp(), t.mons[0].maxhp - (dmg / 4).max(1));

            let (_, mut cb_sub) = CompactBattle::from_battle(&b, &dex);
            cb_sub.mons[0].pp = [0; 4];
            cb_sub.step_turn(&t, [0, 0]); // Gengar sets Sub first, Snorlax Struggles into Sub
            assert_eq!(cb_sub.mons[0].hp(), t.mons[0].maxhp - 1, "Struggle into Sub must deal 1 HP recoil");
        }

        // 3. Double residuals prevented on post-Sandstorm/Perish forced switch (`RESIDUALS_DONE`).
        {
            let p1 = vec![
                mk("Cloyster", 50, "", &["Surf", "Spikes"]),
                mk("Zapdos", 50, "Leftovers", &["Thunderbolt"]),
                mk("Snorlax", 55, "Leftovers", &["Body Slam"]),
            ];
            let p2 = vec![
                mk("Tyranitar", 55, "", &["Sandstorm", "Rock Slide"]),
                mk("Skarmory", 50, "Leftovers", &["Drill Peck"]),
                mk("Gengar", 50, "Leftovers", &["Thunderbolt"]),
            ];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2,3").unwrap();
            b.choose(&dex, 1, "team 1,2,3").unwrap();
            let clo_id = b.active_id(0).unwrap();
            b.poke_mut(clo_id).hp = 1;
            let mut b_ref = b.clone();
            b_ref.choose(&dex, 0, "move 2").unwrap(); // Spikes
            b_ref.choose(&dex, 1, "move 1").unwrap(); // Sandstorm -> kills 1 HP Cloyster in Residual!
            b_ref.choose(&dex, 0, "switch 2").unwrap(); // Switch to Zapdos (auto-completes Turn 1 switch and advances to Turn 2)

            let (t, mut cb) = CompactBattle::from_battle(&b, &dex);
            cb.mons[0].hp_pool[0] = 1;
            cb.step_turn(&t, [1, 0]);
            assert_eq!(cb.sides[0].request, 2);
            assert_ne!(cb.sides[0].side_flags & 0x08, 0, "RESIDUALS_DONE must be set after Sandstorm KO");
            cb.step_switch(&t, [5, 7]);
            assert_eq!(cb.turn, 2);
            assert_eq!(cb.sides[0].side_flags & 0x08, 0);
            assert_eq!(cb.mons[1].hp(), b_ref.poke(b_ref.active_id(0).unwrap()).hp as i16, "Zapdos must not take Sandstorm on residual switch-in");
        }

        // 4. Baton Pass skips AfterMoveSelf slip damage on receiver.
        {
            let p1 = vec![
                mk("Jolteon", 50, "", &["Baton Pass", "Splash"]),
                mk("Snorlax", 55, "", &["Body Slam"]),
            ];
            let p2 = vec![mk("Umbreon", 50, "", &["Toxic", "Mean Look"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, mut cb) = CompactBattle::from_battle(&b, &dex);
            cb.mons[0].set_status(Status::Psn as u8, 0);
            cb.step_turn(&t, [0, 1]); // Jolteon Baton Passes to Snorlax
            assert_eq!(cb.sides[0].active_idx, 1);
            assert_eq!(cb.mons[1].hp(), t.mons[1].maxhp, "Baton Pass receiver must not take outgoing Psn slip");
        }

        // 5. Counter/Mirror Coat Ghost/Dark immunities, Sub blocking hurt tracking, and Fixed/Super Fang arming Counter.
        {
            let p1 = vec![
                mk("Snorlax", 55, "", &["Double-Edge", "Seismic Toss", "Super Fang", "Thunderbolt"]),
                mk("Gengar", 50, "", &["Counter", "Night Shade"]),
                mk("Umbreon", 50, "", &["Mirror Coat", "Substitute"]),
            ];
            let p2 = vec![
                mk("Wobbuffet", 50, "", &["Counter", "Mirror Coat", "Substitute", "Splash"]),
            ];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2,3").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, mut cb) = CompactBattle::from_battle(&b, &dex);
            // Sub absorbs Double-Edge -> Wobbuffet Counter fails
            cb.step_turn(&t, [0, 2]);
            cb.step_turn(&t, [3, 0]); // Thunderbolt vs Counter -> hurt_cat from Sub wasn't set
            // Fresh battle: Seismic Toss on main HP arms Counter vs Snorlax
            let (_, mut cb2) = CompactBattle::from_battle(&b, &dex);
            cb2.step_turn(&t, [1, 0]); // Snorlax Seismic Toss (55), Wobbuffet Counter (110)
            assert_eq!(cb2.mons[0].hp(), t.mons[0].maxhp - 110);
            // Gengar (Ghost) immune to Wobbuffet Counter
            let (_, mut cb_g) = CompactBattle::from_battle(&b, &dex);
            cb_g.step_turn(&t, [5, 3]); // switch to Gengar
            cb_g.mons[1].hp_pool[0] = t.mons[1].maxhp;
            cb_g.step_turn(&t, [1, 0]); // Gengar Night Shade (Physical Ghost move in Gen2, though Special/Phys Ghost -> Wait: Night Shade is fixed Special Ghost!)
            // Switch Gengar into Counter after hurt_cat=1 manually to test Fighting immunity cleanly:
            cb_g.hurt_cat_this_turn[1] = 1;
            cb_g.hurt_damage_this_turn[1] = 50;
            // Direct exec via exec_move/step not exposing hurt across turns: test Umbreon vs Mirror Coat cleanly:
            let (_, mut cb_u) = CompactBattle::from_battle(&b, &dex);
            cb_u.step_turn(&t, [6, 3]); // switch to Umbreon (Dark)
            let p1b = vec![mk("Jolteon", 50, "", &["Mirror Coat"]), mk("Gengar", 50, "", &["Shadow Ball"])];
            let p2b = vec![mk("Heracross", 50, "", &["Seismic Toss"]), mk("Umbreon", 50, "", &["Thunder Wave"])];
            let mut b2 = Battle::from_fixture(&dex, "1,2,3,4", &p1b, &p2b).unwrap();
            b2.choose(&dex, 0, "team 1,2").unwrap();
            b2.choose(&dex, 1, "team 1,2").unwrap();
            // P2 Umbreon immune to Jolteon Mirror Coat (Dark) -> zero damage even if hurt_cat == 2
            let (t2, mut cbu) = CompactBattle::from_battle(&b2, &dex);
            cbu.exec_switch(&t2, 1, 1, SwitchKind::Manual, &mut Prng::new(1));
            cbu.step_turn(&t2, [0, 0]);
            assert_eq!(cbu.mons[4].hp(), t2.mons[4].maxhp);
        }

        // 6. 1v1 Perish/Sandstorm speed order, 1v1 Double-Edge KO skips recoil, Hyper Beam recharge, Haze, Defrost, Pain Split vs Sub, Rapid Spin, Moonlight, AncientPower.
        {
            let p1 = vec![mk("Zapdos", 55, "", &["Hyper Beam", "AncientPower", "Moonlight", "Double-Edge"])];
            let p2 = vec![mk("Forretress", 50, "Charcoal", &["Rapid Spin", "Spikes", "Pain Split", "Flame Wheel"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);

            // 1v1 Double-Edge KO skips recoil on attacker.
            let mut cb_1v1 = base;
            cb_1v1.mons[3].hp_pool[0] = 1;
            cb_1v1.step_turn(&t, [3, 0]);
            assert_eq!(cb_1v1.outcome, 1);
            assert_eq!(cb_1v1.mons[0].hp(), t.mons[0].maxhp, "1v1 KO recoil must be skipped");

            // 1v1 Double Perish: Battle::field_event queues [faster Zapdos, slower Forretress] into
            // faint_queue before faint_messages runs, so last faint is slower -> faster Zapdos wins (outcome = 1).
            let mut cb_perish = base;
            cb_perish.mons[0].vol_flags |= vol_bit::PERISH_SONG;
            cb_perish.mons[0].set_perish_turns(1);
            cb_perish.mons[3].vol_flags |= vol_bit::PERISH_SONG;
            cb_perish.mons[3].set_perish_turns(1);
            cb_perish.step_turn(&t, [2, 2]);
            assert_eq!(cb_perish.outcome, 1, "1v1 double Perish must match Battle (faster side wins)");

            // Moonlight weather scaling: Clear=1/2, Sun=maxhp, Rain=1/4
            let mut cb_moon = base;
            cb_moon.mons[0].hp_pool[0] = 10;
            cb_moon.weather_kind = CompactWeather::Sun;
            cb_moon.weather_turns = 3;
            cb_moon.step_turn(&t, [2, 1]);
            assert_eq!(cb_moon.mons[0].hp(), t.mons[0].maxhp);

            let mut cb_rain = base;
            cb_rain.mons[0].hp_pool[0] = 10;
            cb_rain.weather_kind = CompactWeather::Rain;
            cb_rain.weather_turns = 3;
            cb_rain.step_turn(&t, [2, 1]);
            assert_eq!(cb_rain.mons[0].hp(), 10 + (t.mons[0].maxhp / 4).max(1));

            // Rapid Spin removes P1 Spikes and Flame Wheel thaws frozen Forretress.
            let mut cb_rs = base;
            cb_rs.sides[1].side_flags |= 1; // 1 Spike layer on P2 side
            cb_rs.mons[3].vol_flags |= vol_bit::LEECH_SEED;
            cb_rs.step_turn(&t, [2, 0]); // Moonlight vs Rapid Spin
            assert_eq!(cb_rs.sides[1].side_flags & 0x03, 0);
            assert_eq!(cb_rs.mons[3].vol_flags & vol_bit::LEECH_SEED, 0);

            let mut cb_fw = base;
            cb_fw.mons[3].set_status(Status::Frz as u8, 0);
            cb_fw.step_turn(&t, [2, 3]);
            assert_eq!(cb_fw.mons[3].status_raw(), Status::None as u8, "Flame Wheel must thaw user on hit");

            // AncientPower 10% boost raises all 5 stats (0..5) when triggered.
            let mut ap_boosted = false;
            for s in 1..=256u64 {
                let mut cb_ap = base;
                cb_ap.prng_seed = s.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb_ap.step_turn(&t, [1, 1]);
                if cb_ap.mons[0].boosts[0] == 1 {
                    assert_eq!(&cb_ap.mons[0].boosts[..5], &[1, 1, 1, 1, 1]);
                    ap_boosted = true;
                    break;
                }
            }
            assert!(ap_boosted, "AncientPower must boost atk..spe together");
        }

        // 7. Super Fang, 1.1x type boost item, and Rain/Sun hit fraction parity.
        {
            let p1 = vec![mk("Raticate", 50, "Charcoal", &["Super Fang", "Flame Wheel"])];
            let p2 = vec![mk("Snorlax", 55, "Mystic Water", &["Surf", "Solar Beam"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, mut cb) = CompactBattle::from_battle(&b, &dex);
            let sf = cb.expected_hit_fraction(&t, 0, 0, 1, 0, 0);
            assert!(sf > 0.4 && sf < 0.6, "Super Fang expected hit fraction ~0.5: {sf}");
            cb.weather_kind = CompactWeather::Rain;
            cb.weather_turns = 3;
            let fw_rain = cb.expected_hit_fraction(&t, 0, 0, 1, 0, 1);
            let surf_rain = cb.expected_hit_fraction(&t, 1, 0, 0, 0, 0);
            let sb_rain = cb.expected_hit_fraction(&t, 1, 0, 0, 0, 1);
            cb.weather_kind = CompactWeather::None;
            cb.weather_turns = 0;
            let fw_clear = cb.expected_hit_fraction(&t, 0, 0, 1, 0, 1);
            let surf_clear = cb.expected_hit_fraction(&t, 1, 0, 0, 0, 0);
            let sb_clear = cb.expected_hit_fraction(&t, 1, 0, 0, 0, 1);
            assert!(fw_rain < fw_clear);
            assert!(surf_rain > surf_clear);
            assert!(sb_rain < sb_clear);
            // cfg!(debug_assertions) inside best_hit_fraction checks bit-for-bit equality with expected_hit_fraction:
            let _ = cb.best_hit_fraction(&t, 0, 0, 1, 0);
            let _ = cb.best_hit_fraction(&t, 1, 0, 0, 0);
        }

        // 8. Hyper Beam mustrecharge, Haze clearing status speed/atk drops, Pain Split vs Sub,
        //    Destiny Bond aborts (flinch/frz/slp/par/recharge) & foe switch expiry,
        //    symmetric auto-switch Spikes faint, and Fire secondary burn-chance thaw.
        {
            let p1 = vec![
                mk("Zapdos", 55, "", &["Hyper Beam", "Destiny Bond", "Thunderbolt", "Sleep Talk"]),
                mk("Vaporeon", 50, "", &["Haze", "Flamethrower", "Thunderbolt", "Rest"]),
            ];
            let p2 = vec![
                mk("Forretress", 50, "", &["Substitute", "Pain Split", "Spikes", "Baton Pass"]),
                mk("Tauros", 50, "", &["Body Slam", "Headbutt", "Hyper Beam", "Sleep Talk"]),
            ];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1,2").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);

            // Hyper Beam non-KO sets mustrecharge (slot_refs & 0x80), legal_actions == [0],
            // and recharging clears 0x80 as well as armed Destiny Bond.
            let mut cb_hb = base;
            cb_hb.prng_seed = 1;
            cb_hb.step_turn(&t, [0, 2]); // Hyper Beam vs Spikes (Steel resists Normal -> non-KO)
            assert!(cb_hb.mons[0].must_recharge(), "Hyper Beam hit must set mustrecharge");
            let mut legal = [0u8; 7];
            let n_legal = cb_hb.legal_actions(&t, 0, &mut legal);
            assert_eq!(&legal[..n_legal], &[0]);
            cb_hb.mons[0].vol_flags |= vol_bit::DESTINY_BOND;
            cb_hb.step_turn(&t, [0, 2]); // Recharge turn aborts any armed DBond via onMoveAborted
            assert!(!cb_hb.mons[0].must_recharge());
            assert_eq!(cb_hb.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0);

            // Haze clears PAR_SPEED_DROP and BRN_ATK_DROP alongside boosts.
            let mut cb_haze = base;
            cb_haze.exec_switch(&t, 0, 1, SwitchKind::Manual, &mut Prng::new(1)); // Vaporeon active
            cb_haze.mons[1].set_status(Status::Brn as u8, 0);
            cb_haze.mons[1].vol_flags |= vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP;
            cb_haze.mons[1].boosts[0] = 2;
            cb_haze.mons[3].set_status(Status::Par as u8, 0);
            cb_haze.mons[3].vol_flags |= vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP;
            cb_haze.mons[3].boosts[1] = -2;
            cb_haze.step_turn(&t, [0, 2]); // Vaporeon Haze (not asleep/frozen/paralyzed)
            assert_eq!(cb_haze.mons[1].vol_flags & (vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP), 0);
            assert_eq!(cb_haze.mons[3].vol_flags & (vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP), 0);
            assert_eq!(cb_haze.mons[1].boosts[0], 0);
            assert_eq!(cb_haze.mons[3].boosts[1], 0);

            // Pain Split fails when target has Substitute (hp & sub unchanged).
            let mut cb_ps = base;
            cb_ps.mons[0].hp_pool[0] = 50;
            cb_ps.mons[0].vol_flags |= vol_bit::SUBSTITUTE;
            cb_ps.mons[0].hp_pool[1] = 30;
            cb_ps.mons[3].hp_pool[0] = 100;
            cb_ps.step_turn(&t, [1, 1]); // Zapdos Destiny Bond, Forretress Pain Split into Sub
            assert_eq!(cb_ps.mons[0].hp(), 50);
            assert_eq!(cb_ps.mons[0].sub_hp(), 30);
            assert_eq!(cb_ps.mons[3].hp(), 100);

            // Destiny Bond clears on foe switch (onFoeSwitchOut), Sleep abort, Freeze abort, Flinch abort, Full Par abort.
            let mut cb_fsw = base;
            cb_fsw.mons[0].vol_flags |= vol_bit::DESTINY_BOND;
            cb_fsw.exec_switch(&t, 1, 1, SwitchKind::Manual, &mut Prng::new(1));
            assert_eq!(cb_fsw.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0, "Foe switch must clear Destiny Bond");

            let mut cb_slp = base;
            cb_slp.mons[0].vol_flags |= vol_bit::DESTINY_BOND;
            cb_slp.mons[0].set_status(Status::Slp as u8, 3);
            cb_slp.step_turn(&t, [2, 2]); // Sleeping Zapdos attempts Thunderbolt -> cleared before Spikes
            assert_eq!(cb_slp.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0, "Sleep abort must clear Destiny Bond");

            let mut cb_frz = base;
            cb_frz.mons[0].vol_flags |= vol_bit::DESTINY_BOND;
            cb_frz.mons[0].set_status(Status::Frz as u8, 0);
            cb_frz.step_turn(&t, [2, 2]);
            assert_eq!(cb_frz.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0, "Freeze abort must clear Destiny Bond");

            let mut cb_flinch = base;
            cb_flinch.mons[0].vol_flags |= vol_bit::DESTINY_BOND | vol_bit::FLINCH;
            cb_flinch.step_turn(&t, [2, 2]);
            assert_eq!(cb_flinch.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0, "Flinch abort must clear Destiny Bond");

            // Full paralysis abort: 100%-accurate Thunderbolt blocked only by 25% full paralysis.
            let mut par_cleared = false;
            for s in 1..=64u64 {
                let mut cb_par = base;
                cb_par.prng_seed = s.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb_par.mons[0].vol_flags |= vol_bit::DESTINY_BOND;
                cb_par.mons[0].set_status(Status::Par as u8, 0);
                cb_par.mons[3].hp_pool[0] = t.mons[3].maxhp;
                cb_par.step_turn(&t, [2, 2]); // Thunderbolt
                if cb_par.mons[3].hp() == t.mons[3].maxhp {
                    assert_eq!(cb_par.mons[0].vol_flags & vol_bit::DESTINY_BOND, 0, "Full para abort must clear DBond");
                    par_cleared = true;
                    break;
                }
            }
            assert!(par_cleared);

            // Symmetric auto-switch Spikes faint (side 0 sole remaining bench has 1 HP vs 1 layer of Spikes).
            let mut cb_autosp2 = base;
            cb_autosp2.sides[0].side_flags |= 1;
            cb_autosp2.mons[0].hp_pool[0] = 1;
            cb_autosp2.mons[1].hp_pool[0] = 1; // Vaporeon 1 HP -> Spikes deals maxhp/8 >= 1
            cb_autosp2.mons[0].boosts[4] = -6;
            cb_autosp2.exec_switch(&t, 1, 1, SwitchKind::Manual, &mut Prng::new(1));
            cb_autosp2.step_turn(&t, [0, 0]); // Tauros Body Slam KOs Zapdos -> Vaporeon auto-switches in and dies to Spikes -> outcome == 2
            assert_eq!(cb_autosp2.outcome, 2, "Auto-switch Spikes faint must end battle cleanly");
            assert_eq!(cb_autosp2.sides[0].alive_mask, 0);

            // Fire secondary-burn move (Flamethrower) thaws frozen target even when burn proc fails.
            let mut cb_thaw = base;
            cb_thaw.prng_seed = 1;
            cb_thaw.exec_switch(&t, 0, 1, SwitchKind::Manual, &mut Prng::new(1)); // Vaporeon active (has Flamethrower at move index 1)
            cb_thaw.exec_switch(&t, 1, 1, SwitchKind::Manual, &mut Prng::new(1)); // Tauros active
            cb_thaw.mons[4].set_status(Status::Frz as u8, 0);
            cb_thaw.step_turn(&t, [1, 0]); // Vaporeon Flamethrower hits Tauros -> must thaw Frz
            assert_ne!(cb_thaw.mons[4].status_raw(), Status::Frz as u8, "Flamethrower must thaw frozen target");
        }
    }

    #[test]
    fn compact_ps_parity_bugfix_suite() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };

        // 1. Non-Rest sleep duration is 2..=4 (1..=3 skipped turns), matching PS `gen2stadium2nc2000`.
        {
            let p1 = vec![mk("Parasect", 50, "", &["Spore", "Splash"])];
            let p2 = vec![mk("Snorlax", 55, "", &["Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);
            let mut seen = [false; 8];
            for seed in 1..=128u64 {
                let mut cb = base;
                cb.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb.step_turn(&t, [0, 0]);
                assert_eq!(cb.mons[3].status_raw(), Status::Slp as u8);
                let c = cb.mons[3].status_counter() as usize;
                assert!(
                    (2..=4).contains(&c),
                    "non-Rest sleep counter must be in 2..=4 (1..=3 skipped turns), got {c}"
                );
                seen[c] = true;
            }
            assert!(seen[2] && seen[3] && seen[4], "all 3 sleep counters (2, 3, 4) must occur");
        }

        // 2. Substitute bypass: Encore, Disable, Mean Look, Swagger (+2 Atk only, no confusion), Foresight.
        //    Plus Disable duration matches Battle (2..=5 moved first, 3..=6 moved second).
        {
            let dt = dex_tables(&dex);
            for key in ["encore", "disable", "foresight", "roar", "whirlwind"] {
                let mid = dex.moves.id(key).unwrap();
                assert!(dt.move_specs[mid.0 as usize].bypasses_sub, "{key} must set bypasses_sub");
            }
            let p1 = vec![mk("Alakazam", 55, "", &["Encore", "Disable", "Swagger", "Foresight"])];
            let p2 = vec![mk("Snorlax", 50, "", &["Substitute", "Splash", "Mean Look", "Disable"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            // Turn 1: Alakazam Disable (fails: Snorlax hasn't moved yet) vs Snorlax Substitute
            // -> Snorlax now has Sub & last_move = Substitute, and FORESIGHT is NOT set yet.
            b.choose(&dex, 0, "move 2").unwrap();
            b.choose(&dex, 1, "move 1").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);
            assert!(base.mons[3].sub_hp() > 0);
            assert_eq!(base.mons[3].vol_flags & vol_bit::FORESIGHT, 0);

            // Encore bypasses Sub (matches Battle)
            let mut b_enc = b.clone();
            b_enc.choose(&dex, 0, "move 1").unwrap();
            b_enc.choose(&dex, 1, "move 2").unwrap();
            assert!(b_enc.poke(b_enc.active_id(1).unwrap()).has_volatile(crate::cond_id!(&dex, "encore").unwrap()));
            let mut cb_enc = base;
            cb_enc.step_turn(&t, [0, 1]);
            assert_ne!(cb_enc.mons[3].vol_flags & vol_bit::ENCORE, 0, "Encore must bypass Sub");

            // Disable bypasses Sub & duration matches Battle:
            // Alakazam faster (moved_second=false): initial duration in 2..=5 -> after EOT decrement: 1..=4
            let mut seen_fast = [false; 8];
            for seed in 1..=256u64 {
                let mut cb_dis = base;
                cb_dis.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb_dis.step_turn(&t, [1, 1]); // Alakazam Disable (55% acc) vs Snorlax Splash
                if (cb_dis.mons[3].vol_flags & vol_bit::DISABLE) != 0 {
                    let rem = cb_dis.mons[3].timers[timer_idx::DISABLE] as usize;
                    assert!((1..=4).contains(&rem), "faster Disable post-EOT rem must be 1..=4, got {rem}");
                    seen_fast[rem] = true;
                }
            }
            assert!(seen_fast[1] && seen_fast[2] && seen_fast[3] && seen_fast[4]);

            // Snorlax slower (moved_second=true) Disables Alakazam's Foresight behind Sub:
            // initial duration in 3..=6 -> after EOT decrement: 2..=5
            let mut seen_slow = [false; 8];
            for seed in 1..=256u64 {
                let mut cb_dis = base;
                cb_dis.mons[0].vol_flags |= vol_bit::SUBSTITUTE;
                cb_dis.mons[0].hp_pool[1] = 30;
                cb_dis.prng_seed = seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
                cb_dis.step_turn(&t, [3, 3]); // Alakazam Foresight vs Snorlax Disable
                if (cb_dis.mons[0].vol_flags & vol_bit::DISABLE) != 0 {
                    let rem = cb_dis.mons[0].timers[timer_idx::DISABLE] as usize;
                    assert!((2..=5).contains(&rem), "slower Disable post-EOT rem must be 2..=5, got {rem}");
                    seen_slow[rem] = true;
                }
            }
            assert!(seen_slow[2] && seen_slow[3] && seen_slow[4] && seen_slow[5]);

            // Mean Look bypasses Sub (Snorlax uses Mean Look on Subbed Alakazam)
            let mut cb_ml = base;
            cb_ml.mons[0].vol_flags |= vol_bit::SUBSTITUTE;
            cb_ml.mons[0].hp_pool[1] = 30;
            cb_ml.step_turn(&t, [3, 2]);
            assert_ne!(cb_ml.mons[0].vol_flags & vol_bit::TRAPPED_OR_LOCKED, 0, "Mean Look must bypass Sub");

            // Swagger through Sub: raises Atk +2 WITHOUT confusing target; fails at +6 or Atk == 999
            let mut hit_swag = false;
            for seed in 1..=64u64 {
                let mut cb_sw = base;
                cb_sw.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb_sw.step_turn(&t, [2, 1]); // Alakazam Swagger vs Subbed Snorlax
                if cb_sw.mons[3].boosts[stat_idx::ATK] == 2 {
                    assert_eq!(cb_sw.mons[3].vol_flags & vol_bit::CONFUSION, 0, "Swagger must not confuse through Sub");
                    hit_swag = true;
                    break;
                }
            }
            assert!(hit_swag, "Swagger must boost Atk +2 through Sub");

            // Swagger misses when target Atk is already +6 (or unmodified Atk == 999, e.g. Lv 100 Snorlax at +5).
            for seed in 1..=32u64 {
                let mut cb_sw6 = base;
                cb_sw6.mons[3].vol_flags &= !vol_bit::SUBSTITUTE;
                cb_sw6.mons[3].hp_pool[1] = 0;
                cb_sw6.mons[3].boosts[stat_idx::ATK] = 6;
                cb_sw6.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb_sw6.step_turn(&t, [2, 1]);
                assert_eq!(cb_sw6.mons[3].boosts[stat_idx::ATK], 6, "Swagger must fail at +6 Atk");
                assert_eq!(cb_sw6.mons[3].vol_flags & vol_bit::CONFUSION, 0, "failed Swagger must not confuse");
            }
            {
                let p1_hi = vec![mk("Alakazam", 55, "", &["Swagger"])];
                let p2_hi = vec![mk("Snorlax", 100, "", &["Splash"])];
                let mut b_hi = Battle::from_fixture(&dex, "1,2,3,4", &p1_hi, &p2_hi).unwrap();
                b_hi.choose(&dex, 0, "team 1").unwrap();
                b_hi.choose(&dex, 1, "team 1").unwrap();
                let (t_hi, base_hi) = CompactBattle::from_battle(&b_hi, &dex);
                for seed in 1..=32u64 {
                    let mut cb_hi = base_hi;
                    cb_hi.mons[3].boosts[stat_idx::ATK] = 5; // Lv 100 Snorlax +5 unmodified Atk = 999
                    cb_hi.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    cb_hi.step_turn(&t_hi, [0, 0]);
                    assert_eq!(cb_hi.mons[3].boosts[stat_idx::ATK], 5, "Swagger must fail at 999 Atk");
                    assert_eq!(cb_hi.mons[3].vol_flags & vol_bit::CONFUSION, 0, "failed Swagger at 999 Atk must not confuse");
                }
            }
            // Lv 55 Thick Club Marowak at +5 Atk has unmodified Atk = 1008 (!= 999) -> Swagger SUCCEEDS (+6 & confuses), matching Battle!
            {
                let p1_tc = vec![mk("Alakazam", 55, "", &["Swagger"])];
                let p2_tc = vec![mk("Marowak", 55, "Thick Club", &["Splash"])];
                let mut b_tc = Battle::from_fixture(&dex, "1,2,3,4", &p1_tc, &p2_tc).unwrap();
                b_tc.choose(&dex, 0, "team 1").unwrap();
                b_tc.choose(&dex, 1, "team 1").unwrap();
                let wak_id = b_tc.active_id(1).unwrap();
                b_tc.poke_mut(wak_id).boosts[stat_idx::ATK] = 5;
                let (t_tc, base_tc) = CompactBattle::from_battle(&b_tc, &dex);
                let mut b_tc_hit = false;
                for seed in 1..=32u16 {
                    let mut b_try = Battle::from_fixture(&dex, &format!("{seed},2,3,4"), &p1_tc, &p2_tc).unwrap();
                    b_try.choose(&dex, 0, "team 1").unwrap();
                    b_try.choose(&dex, 1, "team 1").unwrap();
                    let wid = b_try.active_id(1).unwrap();
                    b_try.poke_mut(wid).boosts[stat_idx::ATK] = 5;
                    b_try.choose(&dex, 0, "move 1").unwrap();
                    b_try.choose(&dex, 1, "move 1").unwrap();
                    if b_try.poke(wid).boosts[stat_idx::ATK] == 6 {
                        assert!(b_try.poke(wid).has_volatile(crate::cond_id!(&dex, "confusion").unwrap()));
                        b_tc_hit = true;
                        break;
                    }
                }
                assert!(b_tc_hit, "Battle: Swagger must succeed on Lv 55 Thick Club Marowak at +5 Atk (1008 != 999)");
                let mut cb_tc_hit = false;
                for seed in 1..=64u64 {
                    let mut cb_try = base_tc;
                    cb_try.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    cb_try.step_turn(&t_tc, [0, 0]);
                    if cb_try.mons[3].boosts[stat_idx::ATK] == 6 {
                        assert_ne!(cb_try.mons[3].vol_flags & vol_bit::CONFUSION, 0);
                        cb_tc_hit = true;
                        break;
                    }
                }
                assert!(cb_tc_hit, "CompactBattle: Swagger must succeed on Lv 55 Thick Club Marowak at +5 Atk (1008 != 999)");
            }

            // Foresight bypasses Sub (starting cleanly with FORESIGHT == 0, matching Battle)
            let mut b_fs = b.clone();
            b_fs.choose(&dex, 0, "move 4").unwrap();
            b_fs.choose(&dex, 1, "move 2").unwrap();
            assert!(b_fs.poke(b_fs.active_id(1).unwrap()).has_volatile(crate::cond_id!(&dex, "foresight").unwrap()));
            let mut cb_fs = base;
            assert_eq!(cb_fs.mons[3].vol_flags & vol_bit::FORESIGHT, 0);
            cb_fs.step_turn(&t, [3, 1]);
            assert_ne!(cb_fs.mons[3].vol_flags & vol_bit::FORESIGHT, 0, "Foresight must bypass Sub");
        }

        // 3. Giga Drain misses vs Sub; Protect/Endure fail behind own Sub; Sub-break blocks secondaries.
        {
            let p1 = vec![mk("Exeggutor", 55, "", &["Giga Drain", "Protect", "Endure", "Substitute"])];
            let p2 = vec![mk("Snorlax", 50, "Leftovers", &["Substitute", "Body Slam", "Headbutt", "Thief"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            // Turn 1: Exeggutor Sub vs Snorlax Sub
            b.choose(&dex, 0, "move 4").unwrap();
            b.choose(&dex, 1, "move 1").unwrap();

            let (t, base) = CompactBattle::from_battle(&b, &dex);
            // Giga Drain into Sub: 0 Sub damage, 0 drain heal (matches Battle!)
            let mut b_gd = b.clone();
            let egg_id = b_gd.active_id(0).unwrap();
            let lax_id = b_gd.active_id(1).unwrap();
            let sub_cid = crate::cond_id!(&dex, "substitute").unwrap();
            let b_sub_before = b_gd.poke(lax_id).volatile(sub_cid).unwrap().get_int(crate::state::DK::Hp);
            let b_hp_before = b_gd.poke(egg_id).hp;
            b_gd.choose(&dex, 0, "move 1").unwrap(); // Giga Drain
            b_gd.choose(&dex, 1, "move 1").unwrap(); // Substitute (fails)
            assert_eq!(b_gd.poke(lax_id).volatile(sub_cid).unwrap().get_int(crate::state::DK::Hp), b_sub_before);
            assert_eq!(b_gd.poke(egg_id).hp, b_hp_before);

            let mut cb_gd = base;
            let cb_sub_before = cb_gd.mons[3].sub_hp();
            let cb_hp_before = cb_gd.mons[0].hp();
            cb_gd.step_turn(&t, [0, 0]);
            assert_eq!(cb_gd.mons[3].sub_hp(), cb_sub_before, "Giga Drain must not damage Sub");
            assert_eq!(cb_gd.mons[0].hp(), cb_hp_before, "Giga Drain must not heal off Sub");

            // Protect behind own Sub fails (matches Battle!)
            let mut cb_pr = base;
            let egg_sub_before = cb_pr.mons[0].sub_hp();
            cb_pr.step_turn(&t, [1, 1]); // Exeggutor Protect (fails behind own Sub) vs Snorlax Body Slam
            assert_eq!(cb_pr.mons[0].stall_counter(), 0, "Protect behind own Sub must fail");
            assert!(cb_pr.mons[0].sub_hp() < egg_sub_before, "Body Slam must hit Sub when Protect fails");

            // Endure behind own Sub fails
            let mut cb_en = base;
            cb_en.step_turn(&t, [2, 1]);
            assert_eq!(cb_en.mons[0].stall_counter(), 0, "Endure behind own Sub must fail");

            // Sub-breaking hit (1 HP Sub) must NOT inflict status/flinch/Thief secondary!
            for seed in 1..=32u64 {
                let mut cb_sb = base;
                cb_sb.mons[0].hp_pool[1] = 1; // 1 HP Sub
                cb_sb.mons[0].item = cb_sb.mons[3].item; // Give Exeggutor Leftovers
                cb_sb.mons[3].item = 0; // Snorlax itemless for Thief
                cb_sb.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                cb_sb.step_turn(&t, [1, 3]); // Snorlax Thief breaks Exeggutor's 1 HP Sub
                assert_eq!(cb_sb.mons[0].sub_hp(), 0);
                assert_eq!(cb_sb.mons[3].item, 0, "Thief breaking a Sub must NOT steal item");

                let mut cb_bs = base;
                cb_bs.mons[0].hp_pool[1] = 1;
                cb_bs.prng_seed = seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
                cb_bs.step_turn(&t, [1, 1]); // Snorlax Body Slam breaks Exeggutor's 1 HP Sub
                assert_eq!(cb_bs.mons[0].sub_hp(), 0);
                assert_eq!(cb_bs.mons[0].status_raw(), Status::None as u8, "Body Slam breaking Sub must NOT paralyze");
            }

            // Multihit Sub-break: Twineedle breaks 1-HP Sub on hit 1 and deals main-HP damage + can poison on hit 2 (matching Battle).
            {
                let p1_mh = vec![mk("Beedrill", 55, "", &["Twineedle", "Splash"])];
                let p2_mh = vec![mk("Snorlax", 55, "", &["Substitute", "Splash"])];
                let mut b_mh = Battle::from_fixture(&dex, "1,2,3,4", &p1_mh, &p2_mh).unwrap();
                b_mh.choose(&dex, 0, "team 1").unwrap();
                b_mh.choose(&dex, 1, "team 1").unwrap();
                b_mh.choose(&dex, 0, "move 2").unwrap();
                b_mh.choose(&dex, 1, "move 1").unwrap(); // Snorlax creates Sub
                let lax_id = b_mh.active_id(1).unwrap();
                b_mh.poke_mut(lax_id).volatile_mut(sub_cid).unwrap().set_int(crate::state::DK::Hp, 1);
                let hp_before = b_mh.poke(lax_id).hp as i16;
                let (t_mh, base_mh) = CompactBattle::from_battle(&b_mh, &dex);
                b_mh.choose(&dex, 0, "move 1").unwrap();
                b_mh.choose(&dex, 1, "move 2").unwrap();
                assert!((b_mh.poke(lax_id).hp as i16) < hp_before, "Battle: Twineedle hit 2 must hit main HP after hit 1 breaks Sub");

                let mut hit_main = false;
                let mut poisoned = false;
                for seed in 1..=128u64 {
                    let mut cb_mh = base_mh;
                    cb_mh.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    cb_mh.step_turn(&t_mh, [0, 1]);
                    assert_eq!(cb_mh.mons[3].sub_hp(), 0);
                    if cb_mh.mons[3].hp() < hp_before {
                        hit_main = true;
                    }
                    if cb_mh.mons[3].status_raw() == Status::Psn as u8 {
                        poisoned = true;
                    }
                }
                assert!(hit_main, "CompactBattle: Twineedle hit 2 must hit main HP after hit 1 breaks Sub");
                assert!(poisoned, "CompactBattle: Twineedle hit 2 onto main HP must be able to poison");
            }
        }

        // 4. Rapid Spin clears Spikes/Leech Seed/partiallytrapped, NEVER Mean Look;
        //    Baton Pass passes Mean Look, partiallytrapped, AND Foresight (matching Battle!).
        {
            let p1 = vec![
                mk("Starmie", 50, "", &["Rapid Spin", "Baton Pass", "Splash"]),
                mk("Snorlax", 55, "", &["Splash"]),
            ];
            let p2 = vec![mk("Umbreon", 50, "", &["Mean Look", "Wrap", "Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);

            // Both Mean Look and partially_trapped set on Starmie: Rapid Spin clears only partial trap.
            let mut cb_rs = base;
            cb_rs.sides[0].side_flags |= 1;
            cb_rs.mons[0].vol_flags |= vol_bit::TRAPPED_OR_LOCKED | vol_bit::LEECH_SEED;
            cb_rs.mons[0].set_partially_trapped(true);
            assert!(cb_rs.mons[0].is_trapped());
            cb_rs.step_turn(&t, [0, 2]); // Rapid Spin vs Splash
            assert_eq!(cb_rs.sides[0].side_flags & 0x03, 0);
            assert_eq!(cb_rs.mons[0].vol_flags & vol_bit::LEECH_SEED, 0);
            assert!(!cb_rs.mons[0].partially_trapped(), "Rapid Spin must clear partiallytrapped");
            assert_ne!(
                cb_rs.mons[0].vol_flags & vol_bit::TRAPPED_OR_LOCKED,
                0,
                "Rapid Spin must NOT clear Mean Look"
            );
            let mut acts = [0u8; 7];
            let n = cb_rs.legal_actions(&t, 0, &mut acts);
            assert!(!acts[..n].contains(&5), "Starmie must remain trapped by Mean Look");

            // Baton Pass passes Mean Look, partiallytrapped, AND Foresight (verified directly against Battle!).
            let mut b_bp = b.clone();
            let st_id = b_bp.active_id(0).unwrap();
            let um_id = b_bp.active_id(1).unwrap();
            b_bp.add_volatile(&dex, st_id, "trapped", Some(um_id), crate::battle::EffectHandle::None);
            b_bp.add_volatile(&dex, st_id, "partiallytrapped", Some(um_id), crate::battle::EffectHandle::None);
            b_bp.add_volatile(&dex, st_id, "foresight", Some(um_id), crate::battle::EffectHandle::None);
            let (t_bp, mut cb_bp) = CompactBattle::from_battle(&b_bp, &dex);
            b_bp.choose(&dex, 0, "move 2").unwrap();
            b_bp.choose(&dex, 1, "move 3").unwrap();
            b_bp.choose(&dex, 0, "switch 2").unwrap();
            let lax_bp_id = b_bp.active_id(0).unwrap();
            assert!(b_bp.poke(lax_bp_id).has_volatile(crate::cond_id!(&dex, "trapped").unwrap()));
            assert!(b_bp.poke(lax_bp_id).has_volatile(crate::cond_id!(&dex, "partiallytrapped").unwrap()));
            assert!(b_bp.poke(lax_bp_id).has_volatile(crate::cond_id!(&dex, "foresight").unwrap()));

            cb_bp.step_turn(&t_bp, [1, 2]); // Starmie Baton Passes to Snorlax
            assert_eq!(cb_bp.sides[0].active_idx, 1);
            assert_ne!(cb_bp.mons[1].vol_flags & vol_bit::TRAPPED_OR_LOCKED, 0, "Baton Pass must pass Mean Look");
            assert!(cb_bp.mons[1].partially_trapped(), "Baton Pass must pass partiallytrapped matching Battle");
            assert_ne!(cb_bp.mons[1].vol_flags & vol_bit::FORESIGHT, 0, "Baton Pass must pass Foresight matching Battle");

            // Import from Battle distinguishes meanlook vs partiallytrapped.
            let mut b_wrap = b.clone();
            b_wrap.add_volatile(&dex, b_wrap.active_id(0).unwrap(), "partiallytrapped", b_wrap.active_id(1), crate::battle::EffectHandle::None);
            let (_, cb_imp) = CompactBattle::from_battle(&b_wrap, &dex);
            assert!(cb_imp.mons[0].partially_trapped());
            assert_eq!(cb_imp.mons[0].vol_flags & vol_bit::TRAPPED_OR_LOCKED, 0);
            assert!(cb_imp.mons[0].is_trapped());
        }

        // 5. AfterSwitchInSelf: Burn/Poison damages on manual switch & Baton Pass, NOT faint replace / Roar;
        //    Baton Pass passes residualdmg with pre-increment damage (C=0 -> 1 HP) & healthy receiver retention.
        {
            let p1 = vec![
                mk("Jolteon", 55, "", &["Baton Pass", "Explosion", "Splash"]),
                mk("Snorlax", 55, "", &["Splash"]),
            ];
            let p2 = vec![mk("Skarmory", 50, "", &["Whirlwind", "Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            b.sides[0].roster[1].status = Status::Psn;
            b.refresh_poke_mask(&dex, crate::state::PokeId { side: 0, slot: 1 });

            let mut b_man = b.clone();
            b_man.choose(&dex, 0, "switch 2").unwrap();
            b_man.choose(&dex, 1, "move 2").unwrap();
            let expected_hp_man = b_man.poke(b_man.active_id(0).unwrap()).hp as i16;

            let (t, base) = CompactBattle::from_battle(&b, &dex);
            assert!(expected_hp_man < t.mons[1].maxhp);

            let mut cb_man = base;
            cb_man.step_turn(&t, [5, 1]); // Manual switch to poisoned Snorlax
            assert_eq!(cb_man.mons[1].hp(), expected_hp_man, "manual switch-in must take Gen 2 Psn slip");

            let mut b_bp = b.clone();
            b_bp.choose(&dex, 0, "move 1").unwrap();
            b_bp.choose(&dex, 1, "move 2").unwrap();
            b_bp.choose(&dex, 0, "switch 2").unwrap();
            let expected_hp_bp = b_bp.poke(b_bp.active_id(0).unwrap()).hp as i16;

            let mut cb_bp = base;
            cb_bp.step_turn(&t, [0, 1]); // Baton Pass to poisoned Snorlax
            assert_eq!(cb_bp.mons[1].hp(), expected_hp_bp, "Baton Pass switch-in must take Gen 2 Psn slip");

            // Baton Pass residualdmg pre-increment: passer has residualdmg counter=0, passes to poisoned Snorlax
            // -> Battle deals ((maxhp/16)*0).max(1) == 1 HP (NOT 1/16) and increments counter to 1.
            {
                let mut b_res = b.clone();
                let j_id = b_res.active_id(0).unwrap();
                let res_cid = crate::cond_id!(&dex, "residualdmg").unwrap();
                b_res.add_volatile(&dex, j_id, "residualdmg", None, crate::battle::EffectHandle::None);
                b_res.poke_mut(j_id).volatile_mut(res_cid).unwrap().set_int(crate::state::DK::Counter, 0);
                let (t_r, mut cb_res) = CompactBattle::from_battle(&b_res, &dex);
                b_res.choose(&dex, 0, "move 1").unwrap();
                b_res.choose(&dex, 1, "move 2").unwrap();
                b_res.choose(&dex, 0, "switch 2").unwrap();
                let lax_id = b_res.active_id(0).unwrap();
                let b_hp = b_res.poke(lax_id).hp as i16;
                let b_cnt = b_res.poke(lax_id).volatile(res_cid).unwrap().get_int(crate::state::DK::Counter);
                assert_eq!(b_hp, t_r.mons[1].maxhp - 1, "Battle BP residualdmg C=0 must deal 1 HP");
                assert_eq!(b_cnt, 1);

                cb_res.step_turn(&t_r, [0, 1]);
                assert_eq!(cb_res.mons[1].hp(), b_hp, "CompactBattle BP residualdmg C=0 must deal pre-increment 1 HP");
                assert_eq!(cb_res.mons[1].status_counter(), 0x10 | 1);
            }

            // Baton Pass residualdmg onto a HEALTHY receiver retains counter WITHOUT incrementing
            // (Battle::ResidualDmgAfterMoveSelf only increments when status is Brn|Psn|Tox).
            {
                let mut b_hlth = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
                b_hlth.choose(&dex, 0, "team 1,2").unwrap();
                b_hlth.choose(&dex, 1, "team 1").unwrap();
                let j_id = b_hlth.active_id(0).unwrap();
                let res_cid = crate::cond_id!(&dex, "residualdmg").unwrap();
                b_hlth.add_volatile(&dex, j_id, "residualdmg", None, crate::battle::EffectHandle::None);
                b_hlth.poke_mut(j_id).volatile_mut(res_cid).unwrap().set_int(crate::state::DK::Counter, 2);
                let (t_h, mut cb_hlth) = CompactBattle::from_battle(&b_hlth, &dex);
                b_hlth.choose(&dex, 0, "move 1").unwrap();
                b_hlth.choose(&dex, 1, "move 2").unwrap();
                b_hlth.choose(&dex, 0, "switch 2").unwrap();
                let lax_id = b_hlth.active_id(0).unwrap();
                assert_eq!(b_hlth.poke(lax_id).hp as i16, t_h.mons[1].maxhp);
                assert_eq!(b_hlth.poke(lax_id).volatile(res_cid).unwrap().get_int(crate::state::DK::Counter), 2);

                cb_hlth.step_turn(&t_h, [0, 1]);
                assert_eq!(cb_hlth.mons[1].hp(), t_h.mons[1].maxhp);
                assert_eq!(cb_hlth.mons[1].status_raw(), Status::None as u8);
                assert_eq!(cb_hlth.mons[1].status_counter(), 0x10 | 2, "healthy BP receiver must retain residualdmg counter unchanged");
            }

            // Faint replacement (Jolteon Explosion -> Snorlax auto-switch) skips AfterSwitchInSelf
            let mut cb_fnt = base;
            cb_fnt.step_turn(&t, [1, 1]);
            assert_eq!(cb_fnt.sides[0].active_idx, 1);
            assert_eq!(cb_fnt.mons[1].hp(), t.mons[1].maxhp, "faint replacement must NOT take AfterSwitchInSelf slip");

            // Whirlwind drag-in skips AfterSwitchInSelf
            let mut cb_drag = base;
            cb_drag.step_turn(&t, [2, 0]); // Jolteon Splash -> Skarmory Whirlwind drags Snorlax
            assert_eq!(cb_drag.sides[0].active_idx, 1);
            assert_eq!(cb_drag.mons[1].hp(), t.mons[1].maxhp, "Whirlwind drag-in must NOT take AfterSwitchInSelf slip");
        }

        // 6. Healing berries (Berry Juice +20, Gold Berry +30, Berry +10) trigger ONLY at EOT Residual,
        //    NOT mid-turn before AfterMoveSelf or second hit; Mint/Miracle Berry cure before move.
        {
            let p1 = vec![
                mk("Shuckle", 50, "Berry Juice", &["Splash"]),
                mk("Chansey", 50, "Berry", &["Splash"]),
            ];
            let p2 = vec![
                mk("Snorlax", 55, "Gold Berry", &["Seismic Toss", "Splash"]),
                mk("Tauros", 55, "Mint Berry", &["Splash"]),
            ];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1,2").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);

            // Shuckle at 60 HP (max 126): Snorlax Seismic Toss deals 55 -> 5 HP (<= 63) -> EOT Berry Juice heals +20 -> 25 HP!
            let mut b_bj = b.clone();
            b_bj.poke_mut(b_bj.active_id(0).unwrap()).hp = 60;
            b_bj.choose(&dex, 0, "move 1").unwrap();
            b_bj.choose(&dex, 1, "move 1").unwrap();
            let expected_bj = b_bj.poke(b_bj.active_id(0).unwrap()).hp as i16;
            assert_eq!(expected_bj, 25);

            let mut cb_bj = base;
            cb_bj.mons[0].hp_pool[0] = 60;
            cb_bj.step_turn(&t, [0, 0]);
            assert_eq!(cb_bj.mons[0].hp(), 25, "Berry Juice must heal +20 (not +10) at EOT");
            assert_eq!(cb_bj.mons[0].item, 0);

            // Gold Berry heals +30 at EOT when <= 50% HP; Berry heals +10 at EOT when <= 50% HP.
            let mut cb_gb = base;
            cb_gb.mons[3].hp_pool[0] = 100;
            cb_gb.step_turn(&t, [0, 1]);
            assert_eq!(cb_gb.mons[3].hp(), 130, "Gold Berry must heal +30 at EOT");
            assert_eq!(cb_gb.mons[3].item, 0);

            let mut cb_br = base;
            cb_br.exec_switch(&t, 0, 1, SwitchKind::Manual, &mut Prng::new(1)); // Chansey (Berry)
            cb_br.mons[1].hp_pool[0] = 100;
            cb_br.step_turn(&t, [0, 1]);
            assert_eq!(cb_br.mons[1].hp(), 110, "Berry must heal +10 at EOT");
            assert_eq!(cb_br.mons[1].item, 0);

            // Mint Berry cures Sleep before moving so move executes on the same turn.
            let mut cb_mb = base;
            cb_mb.exec_switch(&t, 1, 1, SwitchKind::Manual, &mut Prng::new(1)); // Tauros (Mint Berry)
            cb_mb.mons[4].set_status(Status::Slp as u8, 3);
            let pp0 = cb_mb.mons[4].pp[0];
            cb_mb.step_turn(&t, [0, 0]);
            assert_eq!(cb_mb.mons[4].status_raw(), Status::None as u8);
            assert_eq!(cb_mb.mons[4].item, 0);
            assert_eq!(cb_mb.mons[4].pp[0], pp0 - 1, "Mint Berry must cure Sleep before executing move");

            // Poisoned Shuckle at 60 HP (1v1): Snorlax Seismic Toss (55 dmg -> 5 HP), Shuckle moves AFTER Snorlax and dies to Poison
            // BEFORE EOT Residual -> Berry Juice must NOT save Shuckle mid-turn!
            let p1_1v1 = vec![mk("Shuckle", 50, "Berry Juice", &["Splash"])];
            let p2_1v1 = vec![mk("Snorlax", 55, "Gold Berry", &["Seismic Toss", "Splash"])];
            let mut b_mid = Battle::from_fixture(&dex, "1,2,3,4", &p1_1v1, &p2_1v1).unwrap();
            b_mid.choose(&dex, 0, "team 1").unwrap();
            b_mid.choose(&dex, 1, "team 1").unwrap();
            let sh_id = b_mid.active_id(0).unwrap();
            b_mid.poke_mut(sh_id).hp = 60;
            b_mid.poke_mut(sh_id).status = Status::Psn;
            b_mid.refresh_poke_mask(&dex, sh_id);
            let (t_mid, mut cb_mid) = CompactBattle::from_battle(&b_mid, &dex);
            b_mid.choose(&dex, 0, "move 1").unwrap();
            b_mid.choose(&dex, 1, "move 1").unwrap();
            assert_eq!(b_mid.outcome(), Some(crate::battle::Outcome::P2Win));
            cb_mid.step_turn(&t_mid, [0, 0]);
            assert_eq!(cb_mid.outcome, 2, "Berry Juice must NOT activate mid-turn before AfterMoveSelf Poison KO");
        }

        // 7. Thief steals item dynamically (stolen Leftovers heals, stolen Thick Club works on Marowak only),
        //    does NOT steal on KO, AND DOES steal on 1-HP Endure (0 damage hit) matching Battle.
        {
            let p1 = vec![
                mk("Gengar", 55, "", &["Thief", "Splash"]),
                mk("Marowak", 50, "", &["Thief", "Splash"]),
            ];
            let p2 = vec![
                mk("Snorlax", 55, "Leftovers", &["Splash", "Endure"]),
                mk("Cubone", 50, "Thick Club", &["Splash"]),
            ];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1,2").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);

            // Gengar steals Leftovers from surviving Snorlax -> heals at EOT!
            let mut cb_th = base;
            cb_th.prng_seed = 1;
            cb_th.mons[0].hp_pool[0] = 100;
            cb_th.step_turn(&t, [0, 0]);
            assert_ne!(cb_th.mons[0].item, 0, "Gengar must steal Leftovers");
            assert_eq!(cb_th.mons[3].item, 0);
            assert_eq!(
                cb_th.mons[0].hp(),
                100 + t.mons[0].quantum_1_16,
                "Stolen Leftovers must actually heal the thief at EOT"
            );

            // Thief on 1 HP foe KOs and must NOT steal item.
            let mut cb_ko = base;
            cb_ko.prng_seed = 1;
            cb_ko.mons[3].hp_pool[0] = 1;
            cb_ko.step_turn(&t, [0, 0]);
            assert_eq!(cb_ko.mons[0].item, 0, "Thief KOing target must NOT steal its item");

            // Thief on 1-HP Endure target deals 0 damage yet STILL steals item (matching Battle!).
            let mut b_end = b.clone();
            let lax_id = b_end.active_id(1).unwrap();
            b_end.poke_mut(lax_id).hp = 1;
            let (t_end, mut cb_end) = CompactBattle::from_battle(&b_end, &dex);
            b_end.choose(&dex, 0, "move 1").unwrap();
            b_end.choose(&dex, 1, "move 2").unwrap(); // Snorlax Endure at 1 HP
            assert!(b_end.poke(b_end.active_id(0).unwrap()).item.is_some(), "Battle: Thief on 1-HP Endure must steal");
            cb_end.step_turn(&t_end, [0, 1]);
            assert_ne!(cb_end.mons[0].item, 0, "CompactBattle: Thief on 1-HP Endure (0 dmg) must still steal");
            assert_eq!(cb_end.mons[3].item, 0);

            // Stolen Thick Club doubles Marowak's Atk, NOT Gengar's Atk.
            let mut cb_tc = base;
            let tc_item = cb_tc.mons[4].item;
            let gengar_atk_before = cb_tc.mon_stat(&t, 0, 0, stat_idx::ATK, false, false);
            let wak_atk_before = cb_tc.mon_stat(&t, 0, 1, stat_idx::ATK, false, false);
            cb_tc.mons[0].item = tc_item;
            cb_tc.mons[1].item = tc_item;
            assert_eq!(cb_tc.mon_stat(&t, 0, 0, stat_idx::ATK, false, false), gengar_atk_before);
            assert_eq!(cb_tc.mon_stat(&t, 0, 1, stat_idx::ATK, false, false), (wak_atk_before * 2).min(999));
        }

        // 8. Flinch runs AFTER Sleep (wake-up happens before flinch) and Headbutt cannot flinch sleeping/frozen foes.
        {
            let p1 = vec![mk("Tauros", 55, "", &["Headbutt", "Splash"])];
            let p2 = vec![mk("Snorlax", 55, "", &["Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            let (t, base) = CompactBattle::from_battle(&b, &dex);

            // On a wake-up turn with FLINCH set (e.g., King's Rock flinch), Sleep (prio 10) runs BEFORE Flinch (prio 8):
            // Snorlax wakes up (`Slp 1 -> None`) and then flinches instead of executing Splash.
            let mut cb_wf = base;
            cb_wf.mons[3].set_status(Status::Slp as u8, 1);
            cb_wf.mons[3].vol_flags |= vol_bit::FLINCH;
            let pp_before = cb_wf.mons[3].pp[0];
            let mut rng = Prng::new(1);
            cb_wf.exec_move(&t, 1, 0, true, false, &mut rng);
            assert_eq!(cb_wf.mons[3].status_raw(), Status::None as u8, "Sleep wake-up must precede Flinch");
            assert_eq!(cb_wf.mons[3].pp[0], pp_before, "Flinch must still block the move after waking up");

            // Non-vacuous: Headbutt against sleeping (Slp 3) or frozen target must NEVER set FLINCH,
            // and a waking target (Slp 1) hit by faster Headbutt must execute its move cleanly.
            for seed in 1..=64u64 {
                let mut cb_slp = base;
                cb_slp.mons[3].set_status(Status::Slp as u8, 3);
                let mut r1 = Prng::new(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                cb_slp.exec_move(&t, 0, 0, false, false, &mut r1);
                assert_eq!(cb_slp.mons[3].vol_flags & vol_bit::FLINCH, 0, "Headbutt must not set FLINCH on sleeping target");

                let mut cb_frz = base;
                cb_frz.mons[3].set_status(Status::Frz as u8, 0);
                let mut r2 = Prng::new(seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
                cb_frz.exec_move(&t, 0, 0, false, false, &mut r2);
                assert_eq!(cb_frz.mons[3].vol_flags & vol_bit::FLINCH, 0, "Headbutt must not set FLINCH on frozen target");

                let mut cb_wake = base;
                cb_wake.mons[3].set_status(Status::Slp as u8, 1);
                cb_wake.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let pp0 = cb_wake.mons[3].pp[0];
                cb_wake.step_turn(&t, [0, 0]);
                assert_eq!(cb_wake.mons[3].status_raw(), Status::None as u8);
                assert_eq!(cb_wake.mons[3].pp[0], pp0 - 1, "Waking target hit by Headbutt must not be flinched");
            }
        }

        // 9. 1v1 Sandstorm AND 1v1 Perish Song mutual KO winner matches Battle identically (including Quick Claw!).
        {
            let p1 = vec![mk("Zapdos", 55, "", &["Splash"])];
            let p2 = vec![mk("Snorlax", 50, "Quick Claw", &["Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();

            // 1v1 Sandstorm mutual KO (normal speed: Zapdos faster -> P1Win)
            let mut b_sand = b.clone();
            b_sand.set_weather(&dex, "sandstorm", None, crate::battle::EffectHandle::None);
            b_sand.poke_mut(b_sand.active_id(0).unwrap()).hp = 1;
            b_sand.poke_mut(b_sand.active_id(1).unwrap()).hp = 1;
            b_sand.set_quick_claw_roll(false);
            let mut b_qc = b_sand.clone();
            b_qc.set_quick_claw_roll(true);
            let (t_s, mut cb_sand) = CompactBattle::from_battle(&b_sand, &dex);
            b_sand.choose(&dex, 0, "move 1").unwrap();
            b_sand.choose(&dex, 1, "move 1").unwrap();
            cb_sand.step_turn(&t_s, [0, 0]);
            let expected_s = match b_sand.outcome() {
                Some(crate::battle::Outcome::P1Win) => 1,
                Some(crate::battle::Outcome::P2Win) => 2,
                _ => 3,
            };
            assert_eq!(expected_s, 1, "Battle 1v1 Sandstorm mutual KO: faster Zapdos wins");
            assert_eq!(cb_sand.outcome, expected_s, "CompactBattle 1v1 Sandstorm must match Battle");

            // 1v1 Sandstorm mutual KO WITH Quick Claw active: slower Snorlax gets +2000 action speed -> P2Win in BOTH Battle and CompactBattle!
            let (t_qc, mut cb_qc) = CompactBattle::from_battle(&b_qc, &dex);
            b_qc.choose(&dex, 0, "move 1").unwrap();
            b_qc.choose(&dex, 1, "move 1").unwrap();
            assert_eq!(b_qc.outcome(), Some(crate::battle::Outcome::P2Win), "Battle Quick Claw active: Snorlax wins 1v1 Sandstorm");
            cb_qc.step_turn(&t_qc, [0, 0]);
            assert_eq!(cb_qc.outcome, 2, "CompactBattle Quick Claw active: Snorlax wins 1v1 Sandstorm");

            // 1v1 Perish Song mutual KO
            let mut b_per = b.clone();
            b_per.set_quick_claw_roll(false);
            let p_cid = crate::cond_id!(&dex, "perishsong").unwrap();
            for s in 0..2usize {
                let id = b_per.active_id(s).unwrap();
                b_per.add_volatile(&dex, id, "perishsong", None, crate::battle::EffectHandle::None);
                b_per.poke_mut(id).volatile_mut(p_cid).unwrap().duration = Some(1);
            }
            let (t_p, mut cb_per) = CompactBattle::from_battle(&b_per, &dex);
            b_per.choose(&dex, 0, "move 1").unwrap();
            b_per.choose(&dex, 1, "move 1").unwrap();
            cb_per.step_turn(&t_p, [0, 0]);
            let expected_p = match b_per.outcome() {
                Some(crate::battle::Outcome::P1Win) => 1,
                Some(crate::battle::Outcome::P2Win) => 2,
                _ => 3,
            };
            assert_eq!(expected_p, 1, "Battle 1v1 Perish Song mutual KO: faster Zapdos wins");
            assert_eq!(cb_per.outcome, expected_p, "CompactBattle 1v1 Perish Song must match Battle");
        }

        // 10. Dual 1-HP replacement Spikes faints (faster incoming mon faints first -> slower wins),
        //     step_switch clearing stale last_move_selfko, and BP receiver Spikes faint canceling foe queued move.
        {
            let p1 = vec![
                mk("Gengar", 55, "", &["Explosion", "Splash"]),
                mk("Jolteon", 55, "", &["Splash"]),
            ];
            let p2 = vec![
                mk("Snorlax", 55, "", &["Splash"]),
                mk("Shuckle", 50, "", &["Splash"]),
            ];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1,2").unwrap();
            b.choose(&dex, 1, "team 1,2").unwrap();
            let p0_act = b.active_id(0);
            let p1_act = b.active_id(1);
            b.add_side_condition(&dex, 0, "spikes", p1_act, crate::battle::EffectHandle::None);
            b.add_side_condition(&dex, 1, "spikes", p0_act, crate::battle::EffectHandle::None);
            b.sides[0].roster[1].hp = 1; // Faster Jolteon (sole grounded bench) at 1 HP
            b.sides[1].roster[1].hp = 1; // Slower Shuckle (sole grounded bench) at 1 HP
            b.sides[1].roster[0].hp = 1; // Snorlax dies to Explosion
            let (t_sp, mut cb_sp) = CompactBattle::from_battle(&b, &dex);
            b.choose(&dex, 0, "move 1").unwrap();
            b.choose(&dex, 1, "move 1").unwrap();
            b.choose(&dex, 0, "switch 2").unwrap();
            b.choose(&dex, 1, "switch 2").unwrap();
            let expected_sp = match b.outcome() {
                Some(crate::battle::Outcome::P1Win) => 1,
                Some(crate::battle::Outcome::P2Win) => 2,
                _ => 3,
            };
            assert_eq!(expected_sp, 2, "Battle dual Spikes faint: faster Jolteon faints first -> P2Win");
            cb_sp.step_turn(&t_sp, [0, 0]);
            assert_eq!(cb_sp.outcome, expected_sp, "CompactBattle dual auto-switch Spikes faint must match Battle");

            // Dual forced `step_switch` into 1-HP last mons on Spikes -> faster incoming mon faints first (not Tie),
            // and stale `last_move_selfko` is cleared at `step_switch` entry.
            let mut cb_ss = cb_sp;
            cb_ss.outcome = 0;
            cb_ss.sides[0].alive_mask = 0b010;
            cb_ss.sides[1].alive_mask = 0b010;
            cb_ss.sides[0].active_idx = 0;
            cb_ss.sides[1].active_idx = 0;
            cb_ss.sides[0].request = 2;
            cb_ss.sides[1].request = 2;
            cb_ss.sides[0].last_move_selfko = 1;
            cb_ss.sides[1].last_move_selfko = 1;
            cb_ss.mons[1].hp_pool[0] = 1; // Faster Jolteon
            cb_ss.mons[4].hp_pool[0] = 1; // Slower Shuckle
            cb_ss.step_switch(&t_sp, [5, 5]);
            assert_eq!(cb_ss.sides[0].last_move_selfko, u8::MAX);
            assert_eq!(cb_ss.sides[1].last_move_selfko, u8::MAX);
            assert_eq!(cb_ss.outcome, 2, "step_switch dual Spikes faint: faster Jolteon faints first -> P2Win");

            // Faster Baton Pass receiver fainting to Spikes in `step_switch` (2 bench options)
            // cancels slower opponent's pending queued move, matching Battle!
            let p1_bp = vec![
                mk("Jolteon", 55, "", &["Baton Pass", "Splash"]),
                mk("Gengar", 50, "", &["Splash"]),
                mk("Snorlax", 50, "", &["Splash"]),
            ];
            let p2_bp = vec![mk("Skarmory", 50, "", &["Tackle", "Splash"])];
            let mut b_bp = Battle::from_fixture(&dex, "1,2,3,4", &p1_bp, &p2_bp).unwrap();
            b_bp.choose(&dex, 0, "team 1,2,3").unwrap();
            b_bp.choose(&dex, 1, "team 1").unwrap();
            let foe_act = b_bp.active_id(1);
            b_bp.add_side_condition(&dex, 0, "spikes", foe_act, crate::battle::EffectHandle::None);
            b_bp.sides[0].roster[1].hp = 1; // Gengar 1 HP -> dies to Spikes on Baton Pass
            let skarm_pp_before = b_bp.poke(b_bp.active_id(1).unwrap()).move_slots[0].pp;
            b_bp.choose(&dex, 0, "move 1").unwrap();
            b_bp.choose(&dex, 1, "move 1").unwrap(); // Skarmory queues Tackle; Battle pauses for BP receiver choice
            let (t_bp, mut cb_bp) = CompactBattle::from_battle(&b_bp, &dex);
            assert_eq!(cb_bp.sides[0].request, 2);
            assert_ne!(cb_bp.sides[1].side_flags & 0x10, 0);

            b_bp.choose(&dex, 0, "switch 2").unwrap(); // Baton Pass to 1-HP Gengar -> dies to Spikes
            // After Gengar faints, Jolteon & Snorlax remain -> forced switch; Skarmory's Tackle must be canceled!
            b_bp.choose(&dex, 0, "switch 3").unwrap();
            let b_lax_hp = b_bp.poke(b_bp.active_id(0).unwrap()).hp as i16;
            let b_skarm_pp = b_bp.poke(b_bp.active_id(1).unwrap()).move_slots[0].pp;
            assert_eq!(b_skarm_pp, skarm_pp_before, "Battle: Skarmory Tackle must be canceled when BP receiver dies to Spikes");

            cb_bp.step_switch(&t_bp, [5, 0]); // BP into 1-HP Gengar -> dies to Spikes
            assert_eq!(cb_bp.sides[1].side_flags & 0x70, 0, "BP receiver faint must clear foe pending move");
            assert_eq!(cb_bp.sides[0].request, 2);
            cb_bp.step_switch(&t_bp, [6, 0]); // Switch to Snorlax
            assert_eq!(cb_bp.mons[3].pp[0], skarm_pp_before as u8, "CompactBattle: Skarmory Tackle must NOT execute after BP receiver faint");
            assert_eq!(cb_bp.mons[2].hp(), b_lax_hp);
        }

        // 11. Mid-turn Disable aborts target's queued move (NOT morphing into Struggle) and runs at prio 7
        //     after Mint Berry (prio 11) and Sleep (prio 10), matching Battle.
        {
            let p1 = vec![mk("Alakazam", 55, "", &["Disable", "Splash"])];
            let p2 = vec![mk("Snorlax", 50, "Mint Berry", &["Tackle", "Splash"])];
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
            b.choose(&dex, 0, "team 1").unwrap();
            b.choose(&dex, 1, "team 1").unwrap();
            // Turn 1: Alakazam Splash, Snorlax Tackle (sets Snorlax last_move = Tackle)
            b.choose(&dex, 0, "move 2").unwrap();
            b.choose(&dex, 1, "move 1").unwrap();
            // Drain Snorlax's Splash PP to 0 so Tackle is its SOLE remaining move
            let lax_id = b.active_id(1).unwrap();
            b.poke_mut(lax_id).move_slots[1].pp = 0;
            let (t_d, base_d) = CompactBattle::from_battle(&b, &dex);

            // Find a seed where Alakazam's Disable hits Snorlax's Tackle mid-turn:
            // Snorlax must abort Tackle (no Struggle damage to Alakazam, no Struggle recoil, Tackle PP unchanged).
            let mut verified = false;
            for seed in 1..=64u64 {
                let mut cb_d = base_d;
                cb_d.prng_seed = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let zam_hp0 = cb_d.mons[0].hp();
                let lax_hp0 = cb_d.mons[3].hp();
                let lax_pp0 = cb_d.mons[3].pp[0];
                cb_d.step_turn(&t_d, [0, 0]);
                if (cb_d.mons[3].vol_flags & vol_bit::DISABLE) != 0 {
                    assert_eq!(cb_d.mons[0].hp(), zam_hp0, "Mid-turn Disabled move must abort (no Struggle damage)");
                    assert_eq!(cb_d.mons[3].hp(), lax_hp0, "Mid-turn Disabled move must abort (no Struggle recoil)");
                    assert_eq!(cb_d.mons[3].pp[0], lax_pp0, "Mid-turn Disabled move must not deduct PP");
                    verified = true;
                    break;
                }
            }
            assert!(verified, "Disable must hit and abort target's sole remaining move");

            // Direct `exec_move` (no EOT residuals masking it!):
            // (a) Mint Berry (prio 11) cures Sleep BEFORE Disable (prio 7) aborts the queued move;
            // (b) Sleep (prio 10) decrements BEFORE Disable (prio 7) aborts the queued move.
            {
                let mut cb_mb = base_d;
                cb_mb.mons[3].set_status(Status::Slp as u8, 3);
                cb_mb.mons[3].vol_flags |= vol_bit::DISABLE;
                cb_mb.mons[3].set_disable_slot(0);
                cb_mb.mons[3].timers[timer_idx::DISABLE] = 3;
                let lax_pp0 = cb_mb.mons[3].pp[0];
                let mut r_mb = Prng::new(1);
                cb_mb.exec_move(&t_d, 1, 0, true, false, &mut r_mb);
                assert_eq!(cb_mb.mons[3].status_raw(), Status::None as u8, "Mint Berry (prio 11) must cure Sleep before Disable (prio 7)");
                assert_eq!(cb_mb.mons[3].item, 0, "Mint Berry must be consumed inside exec_move before Disable aborts");
                assert_eq!(cb_mb.mons[3].pp[0], lax_pp0, "Disable (prio 7) must still abort move after Mint Berry cure");

                let mut cb_slp_dis = base_d;
                cb_slp_dis.mons[3].item = 0;
                cb_slp_dis.mons[3].set_status(Status::Slp as u8, 3);
                cb_slp_dis.mons[3].vol_flags |= vol_bit::DISABLE;
                cb_slp_dis.mons[3].set_disable_slot(0);
                cb_slp_dis.mons[3].timers[timer_idx::DISABLE] = 3;
                let mut r_sd = Prng::new(1);
                cb_slp_dis.exec_move(&t_d, 1, 0, true, false, &mut r_sd);
                assert_eq!(cb_slp_dis.mons[3].status_counter(), 2, "Sleep (prio 10) must decrement before Disable (prio 7)");
                assert_eq!(cb_slp_dis.mons[3].pp[0], lax_pp0, "Disabled sleeping move must not deduct PP");
            }
        }
    }

    #[test]
    fn compact_tree_support_search_key_party_order_and_sync_roundtrip() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |sp: &str, lvl: u8, item: &str, moves: &[&str]| crate::battle::PokemonSet {
            name: sp.to_string(),
            species: sp.to_string(),
            item: item.to_string(),
            ability: String::new(),
            moves: moves.iter().map(|m| (*m).to_string()).collect(),
            level: lvl,
            evs: None,
            ivs: None,
            happiness: None,
            gender: None,
        };
        let p1 = vec![
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Toxic"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mk("Snorlax", 55, "Quick Claw", &["Body Slam", "Earthquake", "Rest", "Curse"]),
        ];
        let p2 = vec![
            mk("Jolteon", 50, "Leftovers", &["Thunderbolt", "Substitute", "Agility", "Baton Pass"]),
            mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
        ];

        // 1. `is_compact_tree_supported`: rejects TeamPreview, accepts started PS & Castle, rejects unsupported moves & transformed.
        let mut b_ps = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        assert!(!is_compact_tree_supported(&b_ps, &dex), "TeamPreview must fall back to Battle");
        b_ps.choose(&dex, 0, "team 1,2,3").unwrap();
        b_ps.choose(&dex, 1, "team 1,2,3").unwrap();
        assert!(is_compact_tree_supported(&b_ps, &dex));

        let mut b_castle = Battle::from_fixture_with_ruleset(
            &dex,
            "1,2,3,4",
            &p1,
            &p2,
            crate::rules::Ruleset::castle_tournament(),
        )
        .unwrap();
        b_castle.choose(&dex, 0, "team 1,2,3").unwrap();
        b_castle.choose(&dex, 1, "team 1,2,3").unwrap();
        assert!(is_compact_tree_supported(&b_castle, &dex));

        let mut b_tr = b_ps.clone();
        let a0 = b_tr.active_id(0).unwrap();
        b_tr.poke_mut(a0).transformed = true;
        assert!(!is_compact_tree_supported(&b_tr, &dex));

        for &unsup in COMPACT_CASTLE_UNSUPPORTED_MOVES {
            let mut p1_u = p1.clone();
            p1_u[0].moves[3] = unsup.to_string();
            let mut bu = Battle::from_fixture(&dex, "1,2,3,4", &p1_u, &p2).unwrap();
            bu.choose(&dex, 0, "team 1,2,3").unwrap();
            bu.choose(&dex, 1, "team 1,2,3").unwrap();
            assert!(
                !is_compact_tree_supported(&bu, &dex),
                "unsupported move `{unsup}` must trigger Battle fallback"
            );
        }

        // 2. `search_key`: HP bucketing, Sub > 0 vs Sub == 0, Quick Claw canonicalization, prng_seed exclusion, ruleset separation.
        let (t_ps, cb_ps) = CompactBattle::from_battle(&b_ps, &dex);
        let (t_castle, cb_castle) = CompactBattle::from_battle(&b_castle, &dex);
        assert!(t_ps.any_quick_claw);
        assert!(cb_ps.quick_claw_observable(&t_ps));
        assert_ne!(
            cb_ps.search_key(&t_ps, 16, true),
            cb_castle.search_key(&t_castle, 16, true),
            "PS and Castle search keys must never collide"
        );

        let mut cb_seed = cb_ps;
        cb_seed.prng_seed ^= 0x1234_5678_9ABC_DEF0;
        assert_eq!(cb_ps.search_key(&t_ps, 16, false), cb_seed.search_key(&t_ps, 16, false));

        let mut cb_qc = cb_ps;
        cb_qc.quick_claw_roll = 0b11;
        assert_eq!(cb_ps.search_key(&t_ps, 16, true), cb_qc.search_key(&t_ps, 16, true));
        assert_ne!(cb_ps.search_key(&t_ps, 16, false), cb_qc.search_key(&t_ps, 16, false));

        let mut cb_hp_same = cb_ps;
        cb_hp_same.mons[0].hp_pool[0] = t_ps.mons[0].maxhp - 2; // stays in same 16-bucket as maxhp-1? Check exact bucket!
        let mut cb_hp_adj = cb_ps;
        cb_hp_adj.mons[0].hp_pool[0] = t_ps.mons[0].maxhp - 3;
        assert_eq!(
            cb_hp_same.search_key(&t_ps, 16, true),
            cb_hp_adj.search_key(&t_ps, 16, true),
            "adjacent HP values in the same 1/16 bucket must merge"
        );
        assert_ne!(
            cb_hp_same.search_key(&t_ps, 0, true),
            cb_hp_adj.search_key(&t_ps, 0, true),
            "exact mode (hp_buckets=0) must distinguish 1 HP diffs"
        );
        let mut cb_sub1 = cb_ps;
        cb_sub1.mons[3].vol_flags |= vol_bit::SUBSTITUTE;
        cb_sub1.mons[3].hp_pool[1] = 1;
        let mut cb_sub0 = cb_ps;
        cb_sub0.mons[3].vol_flags |= vol_bit::SUBSTITUTE;
        cb_sub0.mons[3].hp_pool[1] = 0;
        assert_ne!(cb_sub1.search_key(&t_ps, 16, true), cb_sub0.search_key(&t_ps, 16, true));

        // 3. `party_order` matches `Battle::switch_in` swap semantics & `sync_to_battle` roundtrips cleanly.
        let mut b_sw = b_ps.clone();
        let mut cb_sw = cb_ps;
        // Turn 1: P0 switch 3 (Snorlax, slot 2), P1 move substitute (slot 1)
        b_sw.choose(&dex, 0, "switch 3").unwrap();
        b_sw.choose(&dex, 1, "move substitute").unwrap();
        cb_sw.apply_actions(&t_ps, [6, 1]);
        assert_eq!(cb_sw.party_order(0), [2, 1, 0]);
        let mut synced = b_ps.clone();
        cb_sw.sync_to_battle(&t_ps, &dex, &mut synced);
        assert_eq!(synced.sides[0].party.as_slice(), b_sw.sides[0].party.as_slice());
        assert_eq!(synced.legal_choices(&dex, 0), b_sw.legal_choices(&dex, 0));
        assert_eq!(synced.legal_choices(&dex, 1), b_sw.legal_choices(&dex, 1));

        // Turn 2: P0 switch 2 (Skarmory, pos 1 -> slot 1), P1 move batonpass -> BATONN pause!
        b_sw.choose(&dex, 0, "switch 2").unwrap();
        b_sw.choose(&dex, 1, "move batonpass").unwrap();
        cb_sw.apply_actions(&t_ps, [5, 3]);
        assert_eq!(cb_sw.party_order(0), [1, 2, 0]);
        cb_sw.sync_to_battle(&t_ps, &dex, &mut synced);
        assert_eq!(synced.sides[0].party.as_slice(), b_sw.sides[0].party.as_slice());
        assert_eq!(synced.legal_choices(&dex, 0), b_sw.legal_choices(&dex, 0));
        assert_eq!(synced.legal_choices(&dex, 1), b_sw.legal_choices(&dex, 1));

        // Resolve BATONN to Marowak (switch 2) and verify roundtrip
        b_sw.choose(&dex, 1, "switch 2").unwrap();
        cb_sw.apply_actions(&t_ps, [7, 5]);
        assert_eq!(cb_sw.party_order(1), [1, 0, 2]);
        cb_sw.sync_to_battle(&t_ps, &dex, &mut synced);
        assert_eq!(synced.sides[1].party.as_slice(), b_sw.sides[1].party.as_slice());
        let (_, cb_rt) = CompactBattle::from_battle(&synced, &dex);
        assert_eq!(cb_rt.turn, cb_sw.turn);
        assert_eq!(cb_rt.mons[3].sub_hp(), cb_sw.mons[4].sub_hp());
        assert_eq!(cb_rt.mons[3].hp(), cb_sw.mons[4].hp());

        // 4. PS vs Castle Burn/Paralysis drop clearing on boosts (`Curse` / `Belly Drum`).
        let lax_set = vec![mk("Snorlax", 55, "", &["Curse", "Belly Drum", "Splash"])];
        let foe_set = vec![mk("Blissey", 50, "", &["Splash"])];
        for is_castle in [false, true] {
            let rs = if is_castle {
                crate::rules::Ruleset::castle_tournament()
            } else {
                crate::rules::Ruleset::PsStadium2
            };
            let mut bm = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &lax_set, &foe_set, rs).unwrap();
            bm.choose(&dex, 0, "team 1").unwrap();
            bm.choose(&dex, 1, "team 1").unwrap();
            let (tm, base_cb) = CompactBattle::from_battle(&bm, &dex);

            // Paralyzed Curselax: PS clears PAR_SPEED_DROP on Curse (-1 Spe); Castle preserves it.
            let mut cb_par = base_cb;
            cb_par.mons[0].set_status(Status::Par as u8, 0);
            cb_par.mons[0].vol_flags |= vol_bit::PAR_SPEED_DROP;
            cb_par.step_turn(&tm, [0, 0]);
            if is_castle {
                assert_ne!(cb_par.mons[0].vol_flags & vol_bit::PAR_SPEED_DROP, 0);
            } else {
                assert_eq!(cb_par.mons[0].vol_flags & vol_bit::PAR_SPEED_DROP, 0);
            }

            // Burned Curselax: PS clears BRN_ATK_DROP on Curse (+1 Atk); Castle preserves it.
            let mut cb_brn = base_cb;
            cb_brn.mons[0].set_status(Status::Brn as u8, 0);
            cb_brn.mons[0].vol_flags |= vol_bit::BRN_ATK_DROP;
            cb_brn.step_turn(&tm, [0, 0]);
            if is_castle {
                assert_ne!(cb_brn.mons[0].vol_flags & vol_bit::BRN_ATK_DROP, 0);
            } else {
                assert_eq!(cb_brn.mons[0].vol_flags & vol_bit::BRN_ATK_DROP, 0);
            }
        }

        // 5. `sync_to_battle` preserves `maxtotallevel`, Leech Seed, `partiallytrapped` (1/16, not OHKO),
        //    Future Sight (`DK::TargetSlot`), faint-replacement (`fainted_this_turn` -> no entry slip),
        //    and volatile durations (`stall` expires).
        let p_sync1 = vec![
            mk("Snorlax", 55, "", &["Protect", "Splash"]),
            mk("Suicune", 50, "", &["Splash"]),
        ];
        let p_sync2 = vec![mk("Exeggutor", 50, "", &["Leech Seed", "Splash"])];
        let mut b_s = Battle::from_fixture(&dex, "1,2,3,4", &p_sync1, &p_sync2).unwrap();
        b_s.choose(&dex, 0, "team 1,2").unwrap();
        b_s.choose(&dex, 1, "team 1").unwrap();
        let (ts, mut cbs) = CompactBattle::from_battle(&b_s, &dex);
        cbs.mons[0].vol_flags |= vol_bit::LEECH_SEED;
        cbs.mons[0].set_partially_trapped(true);
        cbs.mons[0].ext_timers[ext_timer_idx::PARTIAL_TRAP_TURNS] = 3;
        cbs.mons[0].set_stall_counter_ext(1);
        cbs.future_sight_turns[0] = 1;
        cbs.future_sight_dmg[0] = 40;
        cbs.future_sight_src[0] = 0;

        let mut synced_mech = b_s.clone();
        cbs.sync_to_battle(&ts, &dex, &mut synced_mech);
        // `maxtotallevel` preserved
        assert!(synced_mech.field.has_pseudo_weather(dex.conds_id("maxtotallevel").unwrap()));
        let hp_before = synced_mech.sides[0].roster[0].hp;
        let foe_before = synced_mech.sides[1].roster[0].hp;
        synced_mech.sides[1].roster[0].hp = foe_before - 50;
        synced_mech.choose(&dex, 0, "move 2").unwrap(); // Splash
        synced_mech.choose(&dex, 1, "move 2").unwrap(); // Splash
        // Leech Seed drained (foe healed), partiallytrapped dealt 1/16 (NOT OHKO!), Future Sight fired, stall expired
        assert!(synced_mech.sides[1].roster[0].hp > foe_before - 50, "Leech Seed must heal foe");
        assert!(synced_mech.sides[0].roster[0].hp > 0, "partiallytrapped must deal 1/16, not 100% OHKO");
        assert!(synced_mech.sides[0].roster[0].hp < hp_before - 40, "Future Sight + Leech Seed + Wrap must deal damage");
        assert!(synced_mech.sides[0].slot_conditions.is_empty(), "Future Sight must fire and clear");
        assert!(
            !synced_mech.sides[0].roster[0].has_volatile(dex.conds_id("stall").unwrap()),
            "stall duration=2 must expire when Protect is not reused"
        );

        // Faint-replacement sync sets `fainted_this_turn` so Burned bench replacement skips AfterSwitchInSelf slip
        let mut cbs_fnt = cbs;
        cbs_fnt.mons[0].mark_fainted();
        cbs_fnt.mons[1].set_status(Status::Brn as u8, 0);
        cbs_fnt.sides[0].alive_mask = 0b10;
        cbs_fnt.sides[0].request = 2;
        cbs_fnt.sides[1].request = 0;
        cbs_fnt.future_sight_turns[0] = 0;
        let mut synced_fnt = b_s.clone();
        cbs_fnt.sync_to_battle(&ts, &dex, &mut synced_fnt);
        assert_eq!(synced_fnt.sides[0].fainted_this_turn, Some(0));
        let cune_max = synced_fnt.sides[0].roster[1].maxhp;
        synced_fnt.choose(&dex, 0, "switch 2").unwrap();
        assert_eq!(
            synced_fnt.sides[0].roster[1].hp, cune_max,
            "faint replacement via sync_to_battle must NOT incur Burn entry slip"
        );

        // `is_compact_tree_supported` rejects `sleepclausemod` and misaligned `active`
        let mut b_scm = b_ps.clone();
        if let Some(scm) = dex.conds_id("sleepclausemod") {
            let mut st = crate::state::EffectState::default();
            st.id = crate::state::EffId::Cond(scm);
            b_scm.field.pseudo_weather.push((scm, st));
            assert!(!is_compact_tree_supported(&b_scm, &dex));
        }
        let mut b_mis = b_ps.clone();
        b_mis.sides[0].active = Some(b_mis.sides[0].party[1]);
        assert!(!is_compact_tree_supported(&b_mis, &dex));
    }

    #[test]
    fn compact_preview_roster_matches_battle_team_preview_transition_bit_exact() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            crate::battle::PokemonSet {
                name: species.to_string(),
                species: species.to_string(),
                item: item.to_string(),
                ability: String::new(),
                moves: moves.iter().map(|s| s.to_string()).collect(),
                gender: None,
                evs: None,
                ivs: None,
                level,
                happiness: Some(255),
            }
        };
        let p1_6 = vec![
            mk("Suicune", 55, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Skarmory", 50, "Mint Berry", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mk("Snorlax", 55, "Quick Claw", &["Body Slam", "Earthquake", "Rest", "Curse"]),
            mk("Zapdos", 50, "Bright Powder", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Gold Berry", &["Surf", "Ice Beam", "Spikes", "Explosion"]),
            mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
        ];
        let p2_6 = vec![
            mk("Jolteon", 50, "Miracle Berry", &["Thunderbolt", "Substitute", "Agility", "Baton Pass"]),
            mk("Raikou", 55, "Magnet", &["Thunder", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 55, "Scope Lens", &["Rock Slide", "Earthquake", "Ice Beam", "Dynamic Punch"]),
            mk("Steelix", 50, "Polkadot Bow", &["Earthquake", "Sandstorm", "Roar", "Explosion"]),
            mk("Blissey", 50, "PRZ Cure Berry", &["Double-Edge", "Curse", "Counter", "Soft-Boiled"]),
        ];

        for rs in [
            crate::rules::Ruleset::PsStadium2,
            crate::rules::Ruleset::castle_tournament(),
        ] {
            for (t1, t2, picks_list) in [
                (
                    &p1_6[..6],
                    &p2_6[..6],
                    vec![
                        ([1u8, 2, 5], [1u8, 2, 3]),
                        ([6u8, 4, 2], [5u8, 3, 1]),
                        ([3u8, 5, 4], [2u8, 6, 5]),
                    ],
                ),
                (
                    &p1_6[..2],
                    &p2_6[..2],
                    vec![([2u8, 1, 0], [1u8, 2, 0])],
                ),
                (
                    &p1_6[..1],
                    &p2_6[..1],
                    vec![([1u8, 0, 0], [1u8, 0, 0])],
                ),
            ] {
                let mut b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", t1, t2, rs).unwrap();
                assert!(is_compact_preview_supported(&b, &dex));
                assert!(!is_compact_tree_supported(&b, &dex));

                let rc = RosterCompactCache::from_battle(&b, &dex);
                let roster = CompactPreviewRoster::from_battle(&b, &dex, Some(&rc));

                for (p0, p1) in picks_list {
                    let mut b_ref = b.clone();
                    b_ref
                        .apply_choices(
                            &dex,
                            [
                                Some(crate::battle::SearchChoice::Team(p0)),
                                Some(crate::battle::SearchChoice::Team(p1)),
                            ],
                        )
                        .unwrap();
                    let (exp_mt, exp_cb) = CompactBattle::from_battle(&b_ref, &dex);
                    let (got_mt, got_cb) = roster.build_turn1([p0, p1], 0x1234_5678);

                    assert_eq!(got_mt.match_fingerprint, exp_mt.match_fingerprint);
                    assert_eq!(got_mt.any_quick_claw, exp_mt.any_quick_claw);
                    assert_eq!(got_mt.ruleset, exp_mt.ruleset);
                    assert_eq!(got_cb.turn, exp_cb.turn);
                    assert_eq!(got_cb.clause_flags, exp_cb.clause_flags);
                    assert_eq!(got_cb.outcome, exp_cb.outcome);
                    assert_eq!(got_cb.sides, exp_cb.sides);
                    assert_eq!(got_cb.mons, exp_cb.mons);
                    assert_eq!(
                        got_cb.search_key(&got_mt, 16, true),
                        exp_cb.search_key(&exp_mt, 16, true)
                    );
                    assert_eq!(
                        got_cb.search_key(&got_mt, 0, true),
                        exp_cb.search_key(&exp_mt, 0, true)
                    );
                    assert_eq!(
                        got_cb.quick_claw_observable(&got_mt),
                        exp_cb.quick_claw_observable(&exp_mt)
                    );

                    // `sync_to_battle` from a TeamPreview Battle template must roundtrip Turn-1 state cleanly.
                    let mut synced = b.clone();
                    let mut got_sync_cb = got_cb;
                    got_sync_cb.quick_claw_roll = exp_cb.quick_claw_roll;
                    got_sync_cb.prng_seed = exp_cb.prng_seed;
                    got_sync_cb.sync_to_battle(&got_mt, &dex, &mut synced);
                    assert_eq!(synced.turn, 1);
                    assert_eq!(synced.request_state, crate::state::RequestState::Move);
                    assert_eq!(synced.sides[0].party.as_slice(), b_ref.sides[0].party.as_slice());
                    assert_eq!(synced.sides[1].party.as_slice(), b_ref.sides[1].party.as_slice());
                    assert_eq!(synced.legal_choices(&dex, 0), b_ref.legal_choices(&dex, 0));
                    assert_eq!(synced.legal_choices(&dex, 1), b_ref.legal_choices(&dex, 1));
                }
                let _ = &mut b;
            }
        }

        // Berserk Gene is natively supported on Team Preview: triggers on lead (+2 Atk, Confusion, item consumed)
        // and stays unconsumed on bench, matching `Battle::apply_choices(Team, Team)`.
        let mut p1_bg = p1_6.clone();
        p1_bg[1].item = "Berserk Gene".to_string(); // L50 Skarmory
        let b_bg = Battle::from_fixture(&dex, "1,2,3,4", &p1_bg, &p2_6).unwrap();
        assert!(is_compact_preview_supported(&b_bg, &dex));
        let roster_bg = CompactPreviewRoster::from_battle(&b_bg, &dex, None);
        // Lead Berserk Gene (`[2, 1, 5]` -> Skarmory leads)
        let (mt_bg_lead, cb_bg_lead) = roster_bg.build_turn1([[2, 1, 5], [1, 2, 3]], 0x9999);
        let mut b_bg_lead = b_bg.clone();
        b_bg_lead
            .apply_choices(
                &dex,
                [
                    Some(crate::battle::SearchChoice::Team([2, 1, 5])),
                    Some(crate::battle::SearchChoice::Team([1, 2, 3])),
                ],
            )
            .unwrap();
        let (exp_mt_bg, exp_cb_bg) = CompactBattle::from_battle(&b_bg_lead, &dex);
        assert_eq!(mt_bg_lead.match_fingerprint, exp_mt_bg.match_fingerprint);
        assert_eq!(cb_bg_lead.mons[0].item, 0);
        assert_eq!(cb_bg_lead.mons[0].boosts[stat_idx::ATK], 2);
        assert_ne!(cb_bg_lead.mons[0].vol_flags & vol_bit::CONFUSION, 0);
        assert!((2..=5).contains(&cb_bg_lead.mons[0].timers[timer_idx::CONFUSION]));
        assert_eq!(exp_cb_bg.mons[0].item, 0);
        assert_eq!(exp_cb_bg.mons[0].boosts[stat_idx::ATK], 2);
        assert_ne!(exp_cb_bg.mons[0].vol_flags & vol_bit::CONFUSION, 0);
        // Bench Berserk Gene (`[1, 2, 5]` -> Skarmory stays on bench unconsumed)
        let (_, cb_bg_bench) = roster_bg.build_turn1([[1, 2, 5], [1, 2, 3]], 0x9999);
        assert_ne!(cb_bg_bench.mons[1].item, 0);
        assert_eq!(cb_bg_bench.mons[1].boosts[stat_idx::ATK], 0);
        assert_eq!(cb_bg_bench.mons[1].vol_flags & vol_bit::CONFUSION, 0);

        // Unsupported move (`Solar Beam`) must trigger Battle fallback on Team Preview.
        let mut p1_sb = p1_6.clone();
        p1_sb[5].moves[0] = "Solar Beam".to_string();
        let b_sb = Battle::from_fixture(&dex, "1,2,3,4", &p1_sb, &p2_6).unwrap();
        assert!(!is_compact_preview_supported(&b_sb, &dex));

        // Synthetic TeamPreview battle with `mid_turn == false` must fall back to Battle.
        let mut b_nmt = Battle::from_fixture(&dex, "1,2,3,4", &p1_6, &p2_6).unwrap();
        b_nmt.mid_turn = false;
        assert!(!is_compact_preview_supported(&b_nmt, &dex));

        // `sleepclausemod` must trigger Battle fallback on Team Preview.
        let mut b_scm = Battle::from_fixture(&dex, "1,2,3,4", &p1_6, &p2_6).unwrap();
        if let Some(scm) = dex.conds_id("sleepclausemod") {
            let mut st = crate::state::EffectState::default();
            st.id = crate::state::EffId::Cond(scm);
            b_scm.field.pseudo_weather.push((scm, st));
            assert!(!is_compact_preview_supported(&b_scm, &dex));
        }
    }
}
