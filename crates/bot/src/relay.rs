//! Setup-Relay simultaneous-move Bellman subgame solver and equilibrium-gated
//! rollout continuation (`Problem A`: multi-turn `Substitute` + boost + `Baton Pass` chains).
//!
//! Solves a compact zero-sum discounted simultaneous-move Markov game (`<= 192` states)
//! over `(ell, h_R, sigma, k1, k2)` purely from [`CompactBattle`] and [`MatchTable`],
//! cached in a zero-allocation thread-local 2-way set-associative table.

use std::cell::RefCell;
use std::cmp::Ordering;

use nc2000_engine::compact::{
    get_damage_base_branchless, get_stat_branchless, vol_bit, CompactBattle, CompactMoveSpec,
    CompactWeather, ItemStatKind, MatchTable, SpecialMoveKind, VarBpKind, NO_MOVE_SLOT, NO_TYPE_ID,
};
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::Status;

use crate::eval::EvalWeights;
use crate::rng::SplitMix64;

pub const MAX_RELAY_STATES: usize = 192; // 2 * 4 * 2 * 4 * 3
pub const MAX_R_ACTIONS: usize = 5; // 0: Attack, 1: Sub, 2: Boost1, 3: Boost2, 4: Pass
pub const MAX_D_ACTIONS: usize = 3; // 0: Break, 1: Disrupt, 2: Pivot
pub const RELAY_GAMMA: f32 = 0.985;
pub const RELAY_COMMIT_MARGIN: f32 = 0.04;

const RELAY_CACHE_SETS: usize = 512;
const RELAY_CACHE_WAYS: usize = 4;
const MAX_RELAY_POOL: usize = RELAY_CACHE_SETS * RELAY_CACHE_WAYS; // 2,048 slots

const ACT_R_ATTACK: usize = 0;
const ACT_R_SUB: usize = 1;
const ACT_R_BOOST1: usize = 2;
const ACT_R_BOOST2: usize = 3;
const ACT_R_PASS: usize = 4;

const ACT_D_BREAK: usize = 0;
const ACT_D_DISRUPT: usize = 1;
const ACT_D_PIVOT: usize = 2;

const AXIS_FLAG_ATK: u8 = 1 << 0;
const AXIS_FLAG_SPA: u8 = 1 << 1;
const AXIS_FLAG_CURSE: u8 = 1 << 2;
const AXIS_FLAG_DEF: u8 = 1 << 3;
const AXIS_FLAG_SPD: u8 = 1 << 4;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum BoostAxisKind {
    #[default]
    None = 0,
    Evasion = 1,
    Offense = 2,
    Speed = 3,
    Defense = 4,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DisruptKind {
    #[default]
    None,
    Phaze { acc_pct: u8 },
    Encore { acc_pct: u8 },
    Haze,
    PerishSong,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct AxisSpec {
    pub kind: BoostAxisKind,
    pub len: u8,
    /// Move slot `0..4` on `u_R`, or `NO_MOVE_SLOT` when representing an already-held boost with no move.
    pub move_slot: u8,
    /// HP cost in 25% buckets (`0` normally, `2` for `Belly Drum`).
    pub hp_cost: u8,
    pub flags: u8,
    pub stages: [i8; 4],
}

impl AxisSpec {
    #[inline]
    const fn none() -> Self {
        Self {
            kind: BoostAxisKind::None,
            len: 1,
            move_slot: NO_MOVE_SLOT,
            hp_cost: 0,
            flags: 0,
            stages: [0; 4],
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
#[repr(C)]
pub struct RelayCacheKey {
    pub key_lo: u64,
    pub key_hi: u64,
}

#[derive(Clone, Copy)]
pub struct SolvedRelaySubgame {
    pub key: RelayCacheKey,
    pub axis1: AxisSpec,
    pub axis2: AxisSpec,
    pub r_move_slots: [u8; MAX_R_ACTIONS],
    pub d0_brk_slot: u8,
    pub d0_dis_slot: u8,
    pub d0_atk_frac_u8: u8,
    pub k1_len: u8,
    pub k2_len: u8,
    pub has_counter_pivot: bool,
    /// Canonical sorted index (`0` or `1`) of defender's best counter bench slot.
    pub counter_canon_idx: u8,
    pub v_star: [f32; MAX_RELAY_STATES],
    pub v_no_relay: [f32; MAX_RELAY_STATES],
    pub q_attack: [f32; MAX_RELAY_STATES],
    pub pi_r: [[u8; MAX_R_ACTIONS]; MAX_RELAY_STATES],
    pub pi_d: [[u8; MAX_D_ACTIONS]; MAX_RELAY_STATES],
    /// Canonical sorted receiver index (`0` or `1`) for each `(ell, sigma, k1, k2)` in `0..48`.
    pub best_receiver_canon: [u8; 48],
}

impl SolvedRelaySubgame {
    const fn empty() -> Self {
        Self {
            key: RelayCacheKey { key_lo: 0, key_hi: 0 },
            axis1: AxisSpec::none(),
            axis2: AxisSpec::none(),
            r_move_slots: [NO_MOVE_SLOT; MAX_R_ACTIONS],
            d0_brk_slot: NO_MOVE_SLOT,
            d0_dis_slot: NO_MOVE_SLOT,
            d0_atk_frac_u8: 0,
            k1_len: 1,
            k2_len: 1,
            has_counter_pivot: false,
            counter_canon_idx: 0,
            v_star: [0.5; MAX_RELAY_STATES],
            v_no_relay: [0.5; MAX_RELAY_STATES],
            q_attack: [0.5; MAX_RELAY_STATES],
            pi_r: [[0; MAX_R_ACTIONS]; MAX_RELAY_STATES],
            pi_d: [[0; MAX_D_ACTIONS]; MAX_RELAY_STATES],
            best_receiver_canon: [0; 48],
        }
    }
}

struct RelayCacheSet {
    keys: [RelayCacheKey; RELAY_CACHE_WAYS],
    slots: [u16; RELAY_CACHE_WAYS],
}

struct RelayCache {
    sets: [RelayCacheSet; RELAY_CACHE_SETS],
    pool: Vec<SolvedRelaySubgame>,
}

impl RelayCache {
    fn new() -> Self {
        Self {
            sets: std::array::from_fn(|_| RelayCacheSet {
                keys: [RelayCacheKey::default(); RELAY_CACHE_WAYS],
                slots: [0; RELAY_CACHE_WAYS],
            }),
            pool: Vec::with_capacity(MAX_RELAY_POOL),
        }
    }
}

thread_local! {
    static RELAY_CACHE: RefCell<RelayCache> = RefCell::new(RelayCache::new());
}

#[inline]
const fn state_idx(
    ell: usize,
    h_r: usize,
    sigma: usize,
    k1: usize,
    k2: usize,
    k1_len: usize,
    k2_len: usize,
) -> usize {
    ((((ell * 4 + (h_r - 1)) * 2 + sigma) * k1_len + k1) * k2_len) + k2
}

#[inline]
const fn recv_idx(
    ell: usize,
    sigma: usize,
    k1: usize,
    k2: usize,
    k1_len: usize,
    k2_len: usize,
) -> usize {
    (((ell * 2 + sigma) * k1_len + k1) * k2_len) + k2
}

/// Map living HP to `1..=4` such that `hp <= maxhp / 4` maps strictly to `1`
/// (preserving the exact Gen 2 `Substitute` legality condition `h_R >= 2`).
#[inline]
fn hp_to_b4(hp: i16, maxhp: i16) -> u8 {
    let m = maxhp.max(1);
    let q1 = m / 4;
    if hp <= q1 {
        1
    } else if hp <= m / 2 {
        2
    } else if hp <= (m * 3) / 4 {
        3
    } else {
        4
    }
}

/// Map living HP to `1..=8` (`12.5%` = exact Gen 2 `Spikes` quantum).
#[inline]
fn hp_to_b8(hp: i16, maxhp: i16) -> u8 {
    let m = maxhp.max(1) as i32;
    let h = hp.max(1) as i32;
    (((h * 8 + m - 1) / m).clamp(1, 8)) as u8
}

#[inline]
fn spikes_entry_dmg(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> i16 {
    let layers = (cb.sides[side].side_flags & 0x03) as usize;
    if layers == 0 {
        return 0;
    }
    let idx = side * 3 + slot;
    let flying_mask = 1u32 << (table.flying_type & 31);
    if (table.mons[idx].type_mask & flying_mask) != 0 {
        return 0;
    }
    let maxhp = table.mons[idx].maxhp.max(1);
    match layers {
        1 => (maxhp / 8).max(1),
        2 => (maxhp / 6).max(1),
        _ => (maxhp / 4).max(1),
    }
}

/// Total Gen 2 switch-in HP loss: `Spikes` + `AfterSwitchInSelf` burn/poison slip
/// (including Baton-Passed `residualdmg` counter when `passer_slot` is `Some`;
/// in Castle PBS, passed `residualdmg` applies to `Psn` only, not `Brn`).
#[inline]
fn switch_entry_dmg(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
    passer_slot: Option<usize>,
) -> i16 {
    let mut dmg = spikes_entry_dmg(cb, table, side, slot);
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let st = mon.status_raw();
    if st != Status::Brn as u8 && st != Status::Psn as u8 && st != Status::Tox as u8 {
        return dmg;
    }
    let maxhp = table.mons[idx].maxhp.max(1);
    let slip = if let Some(p_slot) = passer_slot {
        let passer = &cb.mons[side * 3 + p_slot];
        let p_cnt = passer.status_counter();
        let passer_has_res = passer.status_raw() == Status::Tox as u8 || (p_cnt & 0x10) != 0;
        let res_applies = passer_has_res
            && (st == Status::Tox as u8
                || st == Status::Psn as u8
                || (!table.ruleset.is_castle() && st == Status::Brn as u8));
        if res_applies {
            // Gen 2 `AfterSwitchInSelf` deals pre-increment `max(1, floor(maxhp/16) * c)`.
            let c = (p_cnt & 0x0F) as i16;
            ((maxhp / 16) * c).max(1)
        } else {
            (maxhp / 8).max(1)
        }
    } else {
        (maxhp / 8).max(1)
    };
    dmg = dmg.saturating_add(slip);
    dmg
}

/// Identifies Gen 2 `Pursuit` via `SpecialMoveKind::Pursuit`.
#[inline]
fn is_pursuit_move(mspec: &CompactMoveSpec) -> bool {
    mspec.special_kind == SpecialMoveKind::Pursuit
}

#[inline]
fn mon_guaranteed_incapacitated(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
) -> (bool, u8, bool) {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];
    let st = mon.status_raw();
    let ispec = table.item_spec(mon.item);
    let has_berry = ispec.is_mintberry || ispec.is_miracleberry;
    if st == Status::Frz as u8 && !ispec.is_miracleberry {
        return (true, 4, false);
    }
    if st == Status::Slp as u8 && !has_berry {
        let rem = mon.status_counter().saturating_sub(1);
        if rem >= 1 {
            let has_st = (0..spec.num_moves as usize).any(|m_i| {
                mon.pp[m_i] > 0
                    && !mon.is_slot_disabled(m_i)
                    && table.move_specs[idx][m_i].special_kind == SpecialMoveKind::SleepTalk
            });
            return (rem >= 1 && !has_st, rem.min(4), has_st);
        }
    }
    (false, 0, false)
}

#[inline]
fn classify_self_boost_move(
    table: &MatchTable,
    mon_idx: usize,
    m_slot: usize,
) -> Option<AxisSpec> {
    let ms = &table.move_specs[mon_idx][m_slot];
    if !ms.is_status {
        return None;
    }
    if ms.special_kind == SpecialMoveKind::Curse {
        let is_ghost =
            (table.mons[mon_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
        if !is_ghost {
            return Some(AxisSpec {
                kind: BoostAxisKind::Offense,
                len: 3,
                move_slot: m_slot as u8,
                hp_cost: 0,
                flags: AXIS_FLAG_ATK | AXIS_FLAG_DEF | AXIS_FLAG_CURSE,
                stages: [0, 1, 2, 2],
            });
        }
        return None;
    }
    if ms.special_kind == SpecialMoveKind::BellyDrum {
        return Some(AxisSpec {
            kind: BoostAxisKind::Offense,
            len: 2,
            move_slot: m_slot as u8,
            hp_cost: 2,
            flags: AXIS_FLAG_ATK,
            stages: [0, 6, 6, 6],
        });
    }
    if !ms.boosts_self {
        return None;
    }
    let (st_i, delta) = ms.boosts[0];
    if delta <= 0 {
        return None;
    }
    match st_i {
        6 => Some(AxisSpec {
            kind: BoostAxisKind::Evasion,
            len: 4,
            move_slot: m_slot as u8,
            hp_cost: 0,
            flags: 0,
            stages: [0, 1, 2, 3],
        }),
        0 => {
            let stages = if delta >= 2 { [0, 2, 4, 4] } else { [0, 1, 2, 2] };
            Some(AxisSpec {
                kind: BoostAxisKind::Offense,
                len: 3,
                move_slot: m_slot as u8,
                hp_cost: 0,
                flags: AXIS_FLAG_ATK,
                stages,
            })
        }
        2 => {
            let stages = if delta >= 2 { [0, 2, 4, 4] } else { [0, 1, 2, 2] };
            Some(AxisSpec {
                kind: BoostAxisKind::Offense,
                len: 3,
                move_slot: m_slot as u8,
                hp_cost: 0,
                flags: AXIS_FLAG_SPA,
                stages,
            })
        }
        4 => Some(AxisSpec {
            kind: BoostAxisKind::Speed,
            len: 2,
            move_slot: m_slot as u8,
            hp_cost: 0,
            flags: 0,
            stages: [0, delta.min(6), delta.min(6), delta.min(6)],
        }),
        1 => {
            let stages = if delta >= 2 { [0, 2, 4, 4] } else { [0, 1, 2, 2] };
            Some(AxisSpec {
                kind: BoostAxisKind::Defense,
                len: 3,
                move_slot: m_slot as u8,
                hp_cost: 0,
                flags: AXIS_FLAG_DEF,
                stages,
            })
        }
        3 => {
            let stages = if delta >= 2 { [0, 2, 4, 4] } else { [0, 1, 2, 2] };
            Some(AxisSpec {
                kind: BoostAxisKind::Defense,
                len: 3,
                move_slot: m_slot as u8,
                hp_cost: 0,
                flags: AXIS_FLAG_SPD,
                stages,
            })
        }
        _ => None,
    }
}

/// Bind up to 2 boost axes in priority order `Evasion > Offense > Speed > Defense`,
/// using canonical static move identity for deterministic ordering across move-slot permutations.
fn bind_boost_axes(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> (AxisSpec, AxisSpec) {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];

    let mut cands = [AxisSpec::none(); 6];
    let mut n_cand = 0usize;

    for m_i in 0..spec.num_moves as usize {
        if mon.pp[m_i] == 0 || mon.is_slot_disabled(m_i) {
            continue;
        }
        if let Some(ax) = classify_self_boost_move(table, idx, m_i) {
            // Deduplicate same BoostAxisKind by keeping wider delta / earlier canonical move id
            if let Some(existing) = cands[..n_cand].iter_mut().find(|c| c.kind == ax.kind) {
                let cur_id = spec.moves[existing.move_slot as usize].0;
                let new_id = spec.moves[m_i].0;
                if ax.stages[1] > existing.stages[1]
                    || (ax.stages[1] == existing.stages[1] && new_id < cur_id)
                {
                    *existing = ax;
                }
            } else if n_cand < cands.len() {
                cands[n_cand] = ax;
                n_cand += 1;
            }
        }
    }

    // Include already-held positive boosts if no move of that axis kind is present
    let held_checks: [(BoostAxisKind, i8, u8); 4] = [
        (BoostAxisKind::Evasion, mon.boosts[6], 0),
        (
            BoostAxisKind::Offense,
            mon.boosts[0].max(mon.boosts[2]),
            (if mon.boosts[0] > 0 { AXIS_FLAG_ATK } else { 0 })
                | (if mon.boosts[2] > 0 { AXIS_FLAG_SPA } else { 0 }),
        ),
        (BoostAxisKind::Speed, mon.boosts[4], 0),
        (
            BoostAxisKind::Defense,
            mon.boosts[1].max(mon.boosts[3]),
            (if mon.boosts[1] > 0 { AXIS_FLAG_DEF } else { 0 })
                | (if mon.boosts[3] > 0 { AXIS_FLAG_SPD } else { 0 }),
        ),
    ];
    for &(kind, stage, flags) in &held_checks {
        if stage > 0 && !cands[..n_cand].iter().any(|c| c.kind == kind) && n_cand < cands.len() {
            let capped = stage.min(6);
            cands[n_cand] = AxisSpec {
                kind,
                len: 2,
                move_slot: NO_MOVE_SLOT,
                hp_cost: 0,
                flags,
                stages: [0, capped, capped, capped],
            };
            n_cand += 1;
        }
    }

    // Sort by priority `Evasion (1) < Offense (2) < Speed (3) < Defense (4)`
    for i in 0..n_cand {
        for j in (i + 1)..n_cand {
            if (cands[j].kind as u8) < (cands[i].kind as u8) {
                cands.swap(i, j);
            }
        }
    }

    let ax1 = if n_cand >= 1 { cands[0] } else { AxisSpec::none() };
    let mut ax2 = if n_cand >= 2 { cands[1] } else { AxisSpec::none() };
    if ax2.len > 3 {
        ax2.len = 3;
    }
    (ax1, ax2)
}

#[inline]
fn map_current_boost_to_k(mon_boosts: &[i8; 7], ax: &AxisSpec) -> usize {
    if ax.kind == BoostAxisKind::None || ax.len <= 1 {
        return 0;
    }
    let raw = match ax.kind {
        BoostAxisKind::Evasion => mon_boosts[6],
        BoostAxisKind::Offense => {
            let a = if (ax.flags & AXIS_FLAG_ATK) != 0 { mon_boosts[0] } else { 0 };
            let s = if (ax.flags & AXIS_FLAG_SPA) != 0 { mon_boosts[2] } else { 0 };
            a.max(s)
        }
        BoostAxisKind::Speed => mon_boosts[4],
        BoostAxisKind::Defense => {
            let d = if (ax.flags & AXIS_FLAG_DEF) != 0 { mon_boosts[1] } else { 0 };
            let sd = if (ax.flags & AXIS_FLAG_SPD) != 0 { mon_boosts[3] } else { 0 };
            d.max(sd)
        }
        BoostAxisKind::None => 0,
    };
    if raw <= 0 {
        return 0;
    }
    let mut best_k = 0usize;
    for k in 1..(ax.len as usize) {
        if raw >= ax.stages[k] {
            best_k = k;
        } else if k == 1 && raw > 0 {
            best_k = 1;
        }
    }
    best_k
}

#[inline]
fn compose_boosts(ax1: &AxisSpec, k1: usize, ax2: &AxisSpec, k2: usize) -> [i8; 7] {
    let mut b = [0i8; 7];
    for &(ax, k) in &[(ax1, k1), (ax2, k2)] {
        if ax.kind == BoostAxisKind::None || k == 0 {
            continue;
        }
        let st = ax.stages[k.min(3)];
        match ax.kind {
            BoostAxisKind::Evasion => b[6] = (b[6] + st).clamp(-6, 6),
            BoostAxisKind::Offense => {
                if (ax.flags & AXIS_FLAG_ATK) != 0 {
                    b[0] = (b[0] + st).clamp(-6, 6);
                }
                if (ax.flags & AXIS_FLAG_SPA) != 0 {
                    b[2] = (b[2] + st).clamp(-6, 6);
                }
                if (ax.flags & AXIS_FLAG_DEF) != 0 {
                    b[1] = (b[1] + st).clamp(-6, 6);
                }
                if (ax.flags & AXIS_FLAG_CURSE) != 0 {
                    b[4] = (b[4] - st).clamp(-6, 6);
                }
            }
            BoostAxisKind::Speed => b[4] = (b[4] + st).clamp(-6, 6),
            BoostAxisKind::Defense => {
                if (ax.flags & AXIS_FLAG_DEF) != 0 {
                    b[1] = (b[1] + st).clamp(-6, 6);
                }
                if (ax.flags & AXIS_FLAG_SPD) != 0 {
                    b[3] = (b[3] + st).clamp(-6, 6);
                }
            }
            BoostAxisKind::None => {}
        }
    }
    b
}

/// Relay eligibility predicate (§2.1).
#[inline]
pub fn is_relay_eligible(cb: &CompactBattle, table: &MatchTable, side: usize) -> bool {
    let side_st = &cb.sides[side];
    if side_st.request == 2 || side_st.alive_mask.count_ones() < 2 {
        return false;
    }
    let act_slot = side_st.active_idx as usize;
    if act_slot >= 3 || (side_st.alive_mask & (1 << act_slot)) == 0 {
        return false;
    }
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    if def_slot >= 3
        || (cb.sides[def_side].alive_mask & (1 << def_slot)) == 0
        || cb.mons[def_side * 3 + def_slot].hp() <= 0
    {
        return false;
    }
    let act_idx = side * 3 + act_slot;
    let act_mon = &cb.mons[act_idx];
    if act_mon.hp() <= 0 || act_mon.must_recharge() {
        return false;
    }
    // Lock checks: Ghost Curse, imminent Perish Song, non-BP Encore >= 2 turns, Freeze/Sleep
    if (act_mon.vol_flags & vol_bit::CURSE) != 0 {
        return false;
    }
    if (act_mon.vol_flags & vol_bit::PERISH_SONG) != 0 && act_mon.perish_turns() <= 2 {
        return false;
    }
    let (incap, slp_rem, _) = mon_guaranteed_incapacitated(cb, table, side, act_slot);
    if incap && slp_rem >= 2 {
        return false;
    }

    let spec = &table.mons[act_idx];
    let mut bp_slot: Option<usize> = None;
    let mut has_sub_move = false;
    let mut has_boost_move = false;

    for m_i in 0..spec.num_moves as usize {
        if act_mon.pp[m_i] == 0 || act_mon.is_slot_disabled(m_i) {
            continue;
        }
        let ms = &table.move_specs[act_idx][m_i];
        if ms.special_kind == SpecialMoveKind::BatonPass {
            bp_slot = Some(m_i);
        } else if ms.is_substitute {
            has_sub_move = true;
        } else if classify_self_boost_move(table, act_idx, m_i).is_some() {
            has_boost_move = true;
        }
    }
    let Some(bp_m) = bp_slot else {
        return false;
    };
    if (act_mon.vol_flags & vol_bit::ENCORE) != 0 {
        let enc_slot = act_mon.encore_slot() as usize;
        if enc_slot != bp_m {
            return false;
        }
    }

    // Must have >= 1 bench mon surviving entry Spikes + Gen 2 AfterSwitchInSelf burn/poison slip
    let mut has_surviving_bench = false;
    for slot in 0..3usize {
        if slot == act_slot || (side_st.alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let b_idx = side * 3 + slot;
        let d_entry = switch_entry_dmg(cb, table, side, slot, Some(act_slot));
        if cb.mons[b_idx].hp() > d_entry {
            has_surviving_bench = true;
            break;
        }
    }
    if !has_surviving_bench {
        return false;
    }

    let has_pos_asset = (act_mon.vol_flags & vol_bit::SUBSTITUTE) != 0
        || act_mon.boosts[0] > 0
        || act_mon.boosts[1] > 0
        || act_mon.boosts[2] > 0
        || act_mon.boosts[3] > 0
        || act_mon.boosts[4] > 0
        || act_mon.boosts[6] > 0;

    has_pos_asset || has_sub_move || has_boost_move
}

#[derive(Clone, Copy)]
struct CanonBench {
    party_slot: u8,
    b8_hp: u8,
    survives_spikes: bool,
    status_raw: u8,
    slp_rem: u8,
    has_sleeptalk: bool,
    static_fp: u64,
}

#[derive(Clone, Copy)]
struct ActiveRelayLookup {
    act_slot: u8,
    bp_move_slot: u8,
    sub_move_slot: u8, // 0xFF if none
    atk_move_slot: u8, // 0xFF if none
    axis1: AxisSpec,
    axis2: AxisSpec,
    x0: usize,
    r0: usize,
    recv_party_slots: [u8; 2],
    n_recv: u8,
    n_def_bench: u8,
    def_break_slot: [u8; 2],   // for ell = 0, 1
    def_disrupt_slot: [u8; 2], // for ell = 0, 1 (0xFF if none)
    key: RelayCacheKey,
}

#[inline]
pub fn stat_with_boost(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
    stat_idx: usize,
    boost: i8,
) -> i32 {
    let p_idx = side * 3 + slot;
    let mon = &cb.mons[p_idx];
    let spec = &table.mons[p_idx];
    let side_st = &cb.sides[side];
    let item_kind = {
        let held = table.item_spec(mon.item).stat_kind;
        if held != ItemStatKind::None && held == spec.species_stat_kind {
            held
        } else {
            ItemStatKind::None
        }
    };
    if table.ruleset.is_castle() {
        return nc2000_engine::compact::get_stat_castle(
            spec.stored_stats[stat_idx] as i32,
            boost,
            stat_idx,
            false,
            false,
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
    get_stat_branchless(
        spec.stored_stats[stat_idx] as i32,
        boost,
        stat_idx,
        false,
        false,
        false,
        has_par,
        has_brn,
        side_st.reflect_turns > 0,
        side_st.lightscreen_turns > 0,
        item_kind,
    )
}

/// Conditional damage fraction of `def`'s `maxhp` IF the move hits, paired with hit probability
/// and move priority, evaluated under explicit stat boosts `att_boosts` and `def_boosts`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn best_move_profile(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    att_hp_frac: f32,
    att_boosts: &[i8; 7],
    def_side: usize,
    def_slot: usize,
    def_hp_frac: f32,
    def_boosts: &[i8; 7],
    allow_selfdestruct: bool,
    def_has_sub: bool,
    pursuit_2x: bool,
) -> (u8, f32, f32, i8) {
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let att_mon = &cb.mons[att_idx];
    let def_mon = &cb.mons[def_idx];
    let att_spec = &table.mons[att_idx];
    let def_spec = &table.mons[def_idx];
    let def_maxhp = def_spec.maxhp.max(1) as f32;
    let def_bp = table.item_spec(def_mon.item).is_brightpowder;
    let is_castle = table.ruleset.is_castle();
    let weather = if cb.weather_turns > 0 {
        cb.weather_kind
    } else {
        CompactWeather::None
    };

    let mut best_slot = NO_MOVE_SLOT;
    let mut best_ev = 0.0f32;
    let mut best_dmg_if_hit = 0.0f32;
    let mut best_acc = 1.0f32;
    let mut best_prio = 0i8;

    for m_i in 0..att_spec.num_moves as usize {
        if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
            continue;
        }
        let mspec = &table.move_specs[att_idx][m_i];
        if mspec.is_selfdestruct && !allow_selfdestruct {
            continue;
        }
        // Gen 2: HP-draining moves (Giga Drain, Mega Drain, Absorb, Leech Life, Dream Eater)
        // fail completely against an active Substitute.
        if def_has_sub && mspec.drain_half && !mspec.bypasses_sub {
            continue;
        }
        let bp = if mspec.var_bp_kind == VarBpKind::FlailReversal {
            let att_hp = (att_hp_frac * (att_spec.maxhp.max(1) as f32)).round() as i32;
            crate::eval::ruleset_flail_bp(table.ruleset, att_hp, att_spec.maxhp as i32)
        } else if is_castle && mspec.var_bp_kind == VarBpKind::Frustration {
            102
        } else if pursuit_2x && is_pursuit_move(mspec) {
            80
        } else {
            mspec.base_power as i32
        };
        if mspec.fixed_damage == 0
            && mspec.special_kind != SpecialMoveKind::SuperFang
            && (mspec.is_status || bp <= 0)
        {
            continue;
        }
        let def_foresight = (def_mon.vol_flags & vol_bit::FORESIGHT) != 0;
        let (type_mod, is_immune) =
            table.effectiveness_ext(mspec.move_type, def_spec, def_foresight);
        if is_immune {
            continue;
        }
        let acc = crate::eval::ruleset_hit_probability(
            table.ruleset,
            mspec.base_acc_pct,
            mspec.always_hit,
            att_boosts[5],
            def_boosts[6],
            mspec.ignore_accuracy,
            mspec.ignore_evasion,
            mspec.ignore_positive_evasion,
            def_foresight,
            def_bp,
        ) as f32;

        let raw = if mspec.special_kind == SpecialMoveKind::SuperFang {
            (0.5 * def_hp_frac * def_maxhp).max(1.0)
        } else if mspec.fixed_damage != 0 {
            if mspec.fixed_damage < 0 {
                att_spec.level as f32
            } else {
                mspec.fixed_damage as f32
            }
        } else {
            let (ai, di) = if mspec.is_physical { (0, 1) } else { (2, 3) };
            let mut atk = stat_with_boost(cb, table, att_side, att_slot, ai, att_boosts[ai]);
            let mut def = stat_with_boost(cb, table, def_side, def_slot, di, def_boosts[di]);
            let item_boost_type = if is_castle {
                table.item_spec(att_mon.item).castle_boost_type
            } else {
                table.item_spec(att_mon.item).boost_type
            };
            let has_tb_item =
                mspec.move_type != NO_TYPE_ID && item_boost_type == mspec.move_type;
            let base = if is_castle {
                // Castle PBS: Explosion Def/2 before >= 256 / 4; allow 0 minimum damage.
                if mspec.is_selfdestruct && mspec.is_physical {
                    def = (def / 2).max(1);
                }
                if (atk | def) >= 256 {
                    atk = (atk >> 2).max(1);
                    def = (def >> 2).max(1);
                }
                let mut d =
                    ((att_spec.level as i32 * 2 / 5 + 2) * bp * atk / def.max(1)) / 50;
                if has_tb_item {
                    d = d * 110 / 100;
                }
                d = d.min(997) + 2;
                match weather {
                    CompactWeather::Rain => {
                        if mspec.is_water {
                            d = d * 3 / 2;
                        } else if mspec.is_fire || mspec.is_solarbeam {
                            d /= 2;
                        }
                    }
                    CompactWeather::Sun => {
                        if mspec.is_fire {
                            d = d * 3 / 2;
                        } else if mspec.is_water {
                            d /= 2;
                        }
                    }
                    _ => {}
                }
                if (att_spec.type_mask & (1u32 << (mspec.move_type & 31))) != 0 {
                    d += d / 2;
                }
                if type_mod > 0 {
                    d <<= type_mod as u32;
                } else if type_mod < 0 {
                    d >>= (-type_mod) as u32;
                }
                d.max(0)
            } else {
                get_damage_base_branchless(
                    att_spec.level as i32,
                    bp,
                    atk,
                    def,
                    mspec.is_physical,
                    mspec.is_selfdestruct,
                    false,
                    has_tb_item,
                    weather,
                    mspec.is_water,
                    mspec.is_fire,
                    mspec.is_solarbeam,
                    mspec.move_type,
                    att_spec.type_mask,
                    type_mod,
                    false,
                )
            };
            let hits = (mspec.multihit_mean_x2 as f32) * 0.5;
            (base as f32) * (236.0 / 255.0) * hits
        };

        let dmg_frac = raw / def_maxhp;
        let ev = dmg_frac * acc;
        if ev > best_ev {
            best_ev = ev;
            best_slot = m_i as u8;
            best_dmg_if_hit = dmg_frac;
            best_acc = acc;
            best_prio = mspec.priority;
        }
    }

    (best_slot, best_dmg_if_hit, best_acc, best_prio)
}

#[inline]
fn primary_disrupt_of(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
) -> (DisruptKind, u8) {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];

    let mut best_kind = DisruptKind::None;
    let mut best_rank = 0u8;
    let mut best_slot = NO_MOVE_SLOT;

    for m_i in 0..spec.num_moves as usize {
        if mon.pp[m_i] == 0 || mon.is_slot_disabled(m_i) {
            continue;
        }
        let ms = &table.move_specs[idx][m_i];
        let (kind, rank) = match ms.special_kind {
            SpecialMoveKind::Haze => (DisruptKind::Haze, 4),
            SpecialMoveKind::Encore => (
                DisruptKind::Encore {
                    acc_pct: ms.base_acc_pct,
                },
                3,
            ),
            SpecialMoveKind::Phaze => (
                DisruptKind::Phaze {
                    acc_pct: ms.base_acc_pct,
                },
                2,
            ),
            _ => continue,
        };
        if rank > best_rank {
            best_rank = rank;
            best_kind = kind;
            best_slot = m_i as u8;
        }
    }
    (best_kind, best_slot)
}

#[inline]
fn disrupt_hit_prob(
    ruleset: Ruleset,
    kind: DisruptKind,
    target_eva: i8,
    brightpowder: bool,
) -> f32 {
    match kind {
        DisruptKind::None => 0.0,
        DisruptKind::Haze | DisruptKind::PerishSong => 1.0,
        DisruptKind::Phaze { acc_pct } | DisruptKind::Encore { acc_pct } => {
            let thr = crate::eval::ruleset_accuracy_threshold(
                ruleset,
                acc_pct,
                0,
                target_eva,
                false,
                false,
                false,
                false,
                brightpowder,
            );
            if thr == 255 {
                1.0
            } else {
                (thr as f32) / 256.0
            }
        }
    }
}

#[inline]
fn hash_mix(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(23)
}

#[inline]
fn canonical_mon_key(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
    include_b8: bool,
    passer_slot: Option<usize>,
) -> u64 {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];
    let mut usable_mask = 0u8;
    let mut own_axis_mask = 0u8;
    for m_i in 0..spec.num_moves as usize {
        if mon.pp[m_i] > 0 && !mon.is_slot_disabled(m_i) {
            usable_mask |= 1 << m_i;
            if !include_b8 {
                if let Some(ax) = classify_self_boost_move(table, idx, m_i) {
                    own_axis_mask |= 1 << (ax.kind as u8);
                }
            }
        }
    }
    let b4 = if include_b8 { hp_to_b4(mon.hp(), spec.maxhp) } else { 0 };
    let (_, slp_rem, _) = mon_guaranteed_incapacitated(cb, table, side, slot);
    let spk_ok = (mon.hp() > switch_entry_dmg(cb, table, side, slot, passer_slot)) as u8;
    let boost_hash = if include_b8 {
        let mut bh = 0u64;
        for (i, &b) in mon.boosts.iter().enumerate() {
            bh |= (((b.clamp(-6, 6) + 6) as u64) & 0xF) << (i * 4);
        }
        bh
    } else {
        // For passer `u_R`, include any negative boost or any positive boost on a non-move axis
        // so `bind_boost_axes` (`held_checks`) is 100% determined by `RelayCacheKey`.
        let mut bh = 0u64;
        for (i, &b) in mon.boosts.iter().enumerate() {
            let ax_k = match i {
                6 => BoostAxisKind::Evasion as u8,
                0 | 2 => BoostAxisKind::Offense as u8,
                4 => BoostAxisKind::Speed as u8,
                1 | 3 => BoostAxisKind::Defense as u8,
                _ => 0,
            };
            let include_this = b < 0 || (b > 0 && (ax_k == 0 || (own_axis_mask & (1 << ax_k)) == 0));
            let enc = if include_this { (b.clamp(-6, 6) + 6) as u64 } else { 6u64 };
            bh |= (enc & 0xF) << (i * 4);
        }
        bh
    };
    let (item_bits, dyn_item_extra) = if mon.item != 0 {
        let ispec = table.item_spec(mon.item);
        let active_stat = (ispec.stat_kind != ItemStatKind::None
            && ispec.stat_kind == spec.species_stat_kind) as u64;
        // `boost_type` / `castle_boost_type` is `0..=17` for type boosters or `0xFF` (`& 0x1F == 31`) for none.
        let eff_boost_type = if table.ruleset.is_castle() {
            ispec.castle_boost_type
        } else {
            ispec.boost_type
        };
        let boost_5b = (eff_boost_type & 0x1F) as u64;
        (
            1u64
                | ((ispec.is_leftovers as u64) << 1)
                | ((ispec.is_brightpowder as u64) << 2)
                | ((ispec.is_mintberry as u64) << 3)
                | ((ispec.is_miracleberry as u64) << 4),
            active_stat | (boost_5b << 1),
        )
    } else {
        (0u64, 31u64 << 1)
    };
    let vol_bits = (mon.is_trapped() as u64)
        | ((((mon.vol_flags & vol_bit::PAR_SPEED_DROP) != 0) as u64) << 1)
        | ((((mon.vol_flags & vol_bit::BRN_ATK_DROP) != 0) as u64) << 2);
    let packed = (usable_mask as u64)
        | ((b4 as u64) << 4)
        | ((mon.status_raw() as u64) << 8)
        | ((slp_rem as u64) << 12)
        | (item_bits << 16)
        | ((spk_ok as u64) << 21)
        | (vol_bits << 22)
        | (boost_hash << 25)
        | (dyn_item_extra << 53);
    let mut h = hash_mix(spec.static_fp, packed);
    if !include_b8 && (mon.status_raw() == Status::Tox as u8 || (mon.status_counter() & 0x10) != 0) {
        h = hash_mix(h, (mon.status_counter() & 0x1F) as u64);
    }
    h
}

#[derive(Clone, Copy)]
struct FastRelayProbe {
    act_slot: u8,
    h_r0: u8,
    sigma0: u8,
    recv_party_slots: [u8; 2],
    n_recv: u8,
    n_def_bench: u8,
    key: RelayCacheKey,
}

#[inline]
fn probe_fast_relay(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
) -> Option<(FastRelayProbe, [CanonBench; 2], [CanonBench; 2])> {
    if !is_relay_eligible(cb, table, side) {
        return None;
    }
    let act_slot = cb.sides[side].active_idx as usize;
    let act_idx = side * 3 + act_slot;
    let act_mon = &cb.mons[act_idx];
    let act_spec = &table.mons[act_idx];
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;

    let mut recv_arr = [CanonBench {
        party_slot: 0,
        b8_hp: 0,
        survives_spikes: false,
        status_raw: 0,
        slp_rem: 0,
        has_sleeptalk: false,
        static_fp: 0,
    }; 2];
    let mut n_recv = 0usize;
    for s_i in 0..3usize {
        if s_i == act_slot || (cb.sides[side].alive_mask & (1 << s_i)) == 0 {
            continue;
        }
        let b_idx = side * 3 + s_i;
        let b_mon = &cb.mons[b_idx];
        if b_mon.hp() <= 0 {
            continue;
        }
        let (_, slp_rem, has_st) = mon_guaranteed_incapacitated(cb, table, side, s_i);
        recv_arr[n_recv] = CanonBench {
            party_slot: s_i as u8,
            b8_hp: hp_to_b4(b_mon.hp(), table.mons[b_idx].maxhp) * 2,
            survives_spikes: b_mon.hp() > switch_entry_dmg(cb, table, side, s_i, Some(act_slot)),
            status_raw: b_mon.status_raw(),
            slp_rem,
            has_sleeptalk: has_st,
            static_fp: canonical_mon_key(cb, table, side, s_i, true, Some(act_slot)),
        };
        n_recv += 1;
    }
    if n_recv == 0 {
        return None;
    }
    if n_recv == 2 && recv_arr[1].static_fp < recv_arr[0].static_fp {
        recv_arr.swap(0, 1);
    }

    let mut def_arr = [recv_arr[0]; 2];
    let mut n_def_bench = 0usize;
    for s_i in 0..3usize {
        if s_i == def_slot || (cb.sides[def_side].alive_mask & (1 << s_i)) == 0 {
            continue;
        }
        let d_idx = def_side * 3 + s_i;
        let d_mon = &cb.mons[d_idx];
        if d_mon.hp() <= 0 {
            continue;
        }
        let (_, slp_rem, has_st) = mon_guaranteed_incapacitated(cb, table, def_side, s_i);
        def_arr[n_def_bench] = CanonBench {
            party_slot: s_i as u8,
            b8_hp: hp_to_b4(d_mon.hp(), table.mons[d_idx].maxhp) * 2,
            survives_spikes: d_mon.hp() > switch_entry_dmg(cb, table, def_side, s_i, None),
            status_raw: d_mon.status_raw(),
            slp_rem,
            has_sleeptalk: has_st,
            static_fp: canonical_mon_key(cb, table, def_side, s_i, true, None),
        };
        n_def_bench += 1;
    }
    if n_def_bench == 2 && def_arr[1].static_fp < def_arr[0].static_fp {
        def_arr.swap(0, 1);
    }

    let mut h_lo = 0x517c_c1b7_2722_0a95u64;
    let mut h_hi = 0x9e37_79b9_7f4a_7c15u64;
    let r_key = canonical_mon_key(cb, table, side, act_slot, false, None);
    let d_key = canonical_mon_key(cb, table, def_side, def_slot, true, None);
    h_lo = hash_mix(h_lo, r_key);
    h_hi = hash_mix(h_hi, d_key);
    let field_bits = ((cb.sides[side].side_flags & 0x03) as u64)
        | (((cb.sides[def_side].side_flags & 0x03) as u64) << 2)
        | (((cb.sides[side].reflect_turns > 0) as u64) << 4)
        | (((cb.sides[side].lightscreen_turns > 0) as u64) << 5)
        | (((cb.sides[def_side].reflect_turns > 0) as u64) << 6)
        | (((cb.sides[def_side].lightscreen_turns > 0) as u64) << 7)
        | (((cb.weather_turns > 0) as u64) * ((cb.weather_kind as u64) << 8))
        | ((n_recv as u64) << 12)
        | ((n_def_bench as u64) << 14);
    h_lo = hash_mix(h_lo, field_bits);
    let ruleset_tag = table.ruleset.fingerprint_tag();
    if ruleset_tag != 0 {
        h_lo = hash_mix(h_lo, ruleset_tag);
        h_hi = hash_mix(h_hi, ruleset_tag.rotate_left(19));
        if !table.ruleset.is_seat_antisymmetric() {
            h_lo = hash_mix(h_lo, (side as u64) + 1);
        }
    }
    for r in &recv_arr[..n_recv] {
        h_lo = hash_mix(h_lo, r.static_fp);
        h_hi = hash_mix(h_hi, r.static_fp.rotate_left(17));
    }
    for d in &def_arr[..n_def_bench] {
        h_hi = hash_mix(h_hi, d.static_fp);
        h_lo = hash_mix(h_lo, d.static_fp.rotate_left(29));
    }
    if h_lo == 0 && h_hi == 0 {
        h_lo = 1;
    }

    let h_r0 = hp_to_b4(act_mon.hp(), act_spec.maxhp);
    let sigma0 = ((act_mon.vol_flags & vol_bit::SUBSTITUTE) != 0) as u8;

    Some((
        FastRelayProbe {
            act_slot: act_slot as u8,
            h_r0,
            sigma0,
            recv_party_slots: [recv_arr[0].party_slot, recv_arr[1].party_slot],
            n_recv: n_recv as u8,
            n_def_bench: n_def_bench as u8,
            key: RelayCacheKey { key_lo: h_lo, key_hi: h_hi },
        },
        recv_arr,
        def_arr,
    ))
}

fn build_cold_lookup(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    probe: &FastRelayProbe,
) -> Option<(ActiveRelayLookup, u8)> {
    let act_slot = probe.act_slot as usize;
    let act_idx = side * 3 + act_slot;
    let act_mon = &cb.mons[act_idx];
    let act_spec = &table.mons[act_idx];
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;

    let mut bp_move_slot = NO_MOVE_SLOT;
    let mut sub_move_slot = NO_MOVE_SLOT;
    for m_i in 0..act_spec.num_moves as usize {
        if act_mon.pp[m_i] == 0 || act_mon.is_slot_disabled(m_i) {
            continue;
        }
        let ms = &table.move_specs[act_idx][m_i];
        if ms.special_kind == SpecialMoveKind::BatonPass && bp_move_slot == NO_MOVE_SLOT {
            bp_move_slot = m_i as u8;
        } else if ms.is_substitute && sub_move_slot == NO_MOVE_SLOT {
            sub_move_slot = m_i as u8;
        }
    }
    if bp_move_slot == NO_MOVE_SLOT {
        return None;
    }

    let (axis1, axis2) = bind_boost_axes(cb, table, side, act_slot);
    let zero_b = [0i8; 7];
    let d_boosts = &cb.mons[def_side * 3 + def_slot].boosts;
    let (atk_move_slot, dmg_r_d0, acc_r_d0, _) = best_move_profile(
        cb, table, side, act_slot, 1.0, &zero_b, def_side, def_slot, 1.0, d_boosts, false, false,
        false,
    );
    let d0_atk_frac_u8 = ((dmg_r_d0 * acc_r_d0 * 100.0).round().clamp(0.0, 250.0)) as u8;

    let h_r0 = probe.h_r0 as usize;
    let sigma0 = probe.sigma0 as usize;
    let k1_0 = map_current_boost_to_k(&act_mon.boosts, &axis1);
    let k2_0 = map_current_boost_to_k(&act_mon.boosts, &axis2);
    let k1_len = axis1.len as usize;
    let k2_len = axis2.len as usize;
    let x0 = state_idx(0, h_r0, sigma0, k1_0, k2_0, k1_len, k2_len);
    let r0 = recv_idx(0, sigma0, k1_0, k2_0, k1_len, k2_len);

    let (d0_brk, _, _, _) = best_move_profile(
        cb, table, def_side, def_slot, 1.0, d_boosts, side, act_slot, 1.0, &zero_b, false, false,
        false,
    );
    let (_, d0_dis) = primary_disrupt_of(cb, table, def_side, def_slot);

    Some((
        ActiveRelayLookup {
            act_slot: probe.act_slot,
            bp_move_slot,
            sub_move_slot,
            atk_move_slot,
            axis1,
            axis2,
            x0,
            r0,
            recv_party_slots: probe.recv_party_slots,
            n_recv: probe.n_recv,
            n_def_bench: probe.n_def_bench,
            def_break_slot: [d0_brk, NO_MOVE_SLOT],
            def_disrupt_slot: [d0_dis, NO_MOVE_SLOT],
            key: probe.key,
        },
        d0_atk_frac_u8,
    ))
}

/// Select the defender's best counter pivot `c*` (§2.3) in bench-order-invariant fashion.
fn select_best_counter(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    act_slot: usize,
    def_side: usize,
    def_slot: usize,
    def_bench: &[CanonBench],
) -> Option<usize> {
    let def_idx = def_side * 3 + def_slot;
    if cb.mons[def_idx].is_trapped() || def_bench.is_empty() {
        return None;
    }
    let zero_b = [0i8; 7];

    let counter_score = |d_slot: usize, awake_factor: f32| -> (f32, f32, f32, bool) {
        let d_boosts = &cb.mons[def_side * 3 + d_slot].boosts;
        let (dk, _) = primary_disrupt_of(cb, table, def_side, d_slot);
        let has_disrupt = if dk != DisruptKind::None { awake_factor } else { 0.0 };
        let (_, dmg_d_to_r, acc_d_to_r, _) = best_move_profile(
            cb, table, def_side, d_slot, 1.0, d_boosts, side, act_slot, 1.0, &zero_b, false, false,
            false,
        );
        let (_, dmg_r_to_d, acc_r_to_d, _) = best_move_profile(
            cb, table, side, act_slot, 1.0, &zero_b, def_side, d_slot, 1.0, d_boosts, false, false,
            false,
        );
        let eff_d_to_r = dmg_d_to_r * acc_d_to_r * awake_factor;
        let eff_r_to_d = dmg_r_to_d * acc_r_to_d;
        let faster = if crate::eval::compare_speed_with_boost(
            cb,
            table,
            def_side,
            d_slot,
            d_boosts[4],
            side,
            act_slot,
            0,
        ) == Ordering::Greater
        {
            1.0
        } else {
            0.0
        };
        let s = 1.6 * has_disrupt + 1.2 * (4.0 * eff_d_to_r).min(1.0) + 0.25 * faster - 1.5 * eff_r_to_d;
        (s, eff_r_to_d, eff_d_to_r, dk != DisruptKind::None)
    };

    let (incap_u, slp_u, st_u) = mon_guaranteed_incapacitated(cb, table, def_side, def_slot);
    let awake_u = if incap_u && slp_u >= 2 {
        0.15
    } else if slp_u >= 1 && st_u {
        0.35
    } else if incap_u {
        0.45
    } else {
        1.0
    };
    let (s_active, _, active_d_to_r, active_has_disrupt) = counter_score(def_slot, awake_u);
    // If active defender already threatens a 2HKO (`>= 0.48`) or already carries awake Disrupt,
    // it does not need to surrender a turn pivoting.
    if active_d_to_r >= 0.48 || (active_has_disrupt && awake_u >= 0.9) {
        return None;
    }

    let mut best_c: Option<(usize, f32)> = None;
    for (ci, cand) in def_bench.iter().enumerate() {
        if !cand.survives_spikes {
            continue;
        }
        let d_slot = cand.party_slot as usize;
        let d_idx = def_side * 3 + d_slot;
        let maxhp = table.mons[d_idx].maxhp.max(1) as f32;
        let spk_frac = (switch_entry_dmg(cb, table, def_side, d_slot, None) as f32) / maxhp;
        let hp_frac = (cand.b8_hp as f32) / 8.0;
        let awake_c = if cand.status_raw == Status::Frz as u8 {
            0.0
        } else if cand.slp_rem >= 2 && !cand.has_sleeptalk {
            0.15
        } else if cand.slp_rem >= 1 {
            0.40
        } else {
            1.0
        };
        let (sc, r_to_c, _, _) = counter_score(d_slot, awake_c);
        // Must not be 2HKO'd by u_R (`r_to_c < 0.48`) and must comfortably survive Spikes + switch-in hit
        if r_to_c >= 0.48 || hp_frac <= spk_frac + r_to_c * 1.35 {
            continue;
        }
        if sc > s_active + 0.25 && best_c.map_or(true, |(_, bs)| sc > bs + 1e-6) {
            best_c = Some((ci, sc));
        }
    }
    best_c.map(|(ci, _)| ci)
}

#[inline]
fn mon_status_pen(w: &EvalWeights, ruleset: Ruleset, status_raw: u8, slp_rem: u8) -> f32 {
    let p = if status_raw == Status::Brn as u8 {
        w.brn
    } else if status_raw == Status::Par as u8 {
        w.par
    } else if status_raw == Status::Slp as u8 {
        let max_slp = if ruleset.is_castle() { 4.0 } else { 3.0 };
        w.slp * ((slp_rem as f64 + 1.0).min(max_slp)) / max_slp
    } else if status_raw == Status::Frz as u8 {
        w.frz
    } else if status_raw == Status::Psn as u8 {
        w.psn
    } else if status_raw == Status::Tox as u8 {
        w.tox
    } else {
        0.0
    };
    p as f32
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Evaluate post-pass lineup sweep value `W_pass` in `[0.01, 0.97]` (§3.3) for receiver `b_slot`.
#[allow(clippy::too_many_arguments)]
fn evaluate_single_receiver_sweep(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    b_slot: usize,
    b_hp_frac_entry: f32,
    b_slp_rem: u8,
    b_status: u8,
    sigma_in: bool,
    passed_boosts: &[i8; 7],
    def_side: usize,
    foe_slots: &[u8],
    foe_hp_fracs: &[f32],
    incoming_atk_from_lead: bool,
    base_logit_anchor: f32,
) -> f32 {
    evaluate_single_receiver_sweep_ext(
        cb,
        table,
        w,
        side,
        b_slot,
        b_hp_frac_entry,
        b_slp_rem,
        b_status,
        sigma_in,
        passed_boosts,
        def_side,
        foe_slots,
        foe_hp_fracs,
        incoming_atk_from_lead,
        base_logit_anchor,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_single_receiver_sweep_ext(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    b_slot: usize,
    b_hp_frac_entry: f32,
    b_slp_rem: u8,
    b_status: u8,
    sigma_in: bool,
    passed_boosts: &[i8; 7],
    def_side: usize,
    foe_slots: &[u8],
    foe_hp_fracs: &[f32],
    incoming_atk_from_lead: bool,
    base_logit_anchor: f32,
    cached: Option<&[CachedFoeProfile; 3]>,
) -> f32 {
    if b_hp_frac_entry <= 0.001 {
        return sigmoid_f32(base_logit_anchor - (w.scale as f32) * ((w.alive + w.hp * 0.5) as f32))
            .clamp(0.01, 0.97);
    }
    if b_status == Status::Frz as u8 || (b_status == Status::Slp as u8 && b_slp_rem >= 2) {
        return sigmoid_f32(base_logit_anchor).clamp(0.01, 0.97);
    }

    let mut hp_eff = b_hp_frac_entry;
    let mut cur_sub = if sigma_in { 1.0f32 } else { 0.0f32 };

    // Handle in-turn hit from lead defender when Pass moved BEFORE Break
    if incoming_atk_from_lead && !foe_slots.is_empty() {
        let d0_slot = foe_slots[0] as usize;
        let d0_idx = def_side * 3 + d0_slot;
        let (incap_d, slp_d, st_d) = mon_guaranteed_incapacitated(cb, table, def_side, d0_slot);
        let d_act_mult = if incap_d {
            0.0
        } else if slp_d >= 1 && st_d {
            0.33
        } else {
            1.0
        };
        if d_act_mult > 0.0 {
            // Castle PBS (`Probe 12C`): slower Pursuit retains 2x BP (80) against a faster Baton Pass receiver!
            let castle_pursuit = table.ruleset.is_castle()
                && (0..table.mons[d0_idx].num_moves as usize).any(|mi| {
                    cb.mons[d0_idx].pp[mi] > 0
                        && !cb.mons[d0_idx].is_slot_disabled(mi)
                        && is_pursuit_move(&table.move_specs[d0_idx][mi])
                });
            let (dmg_d, acc_d) = if !castle_pursuit {
                if let Some(cp) = cached {
                    (cp[0].dmg_m_if_hit, cp[0].acc_m)
                } else {
                    let d0_boosts = &cb.mons[d0_idx].boosts;
                    let (_, dd, ad, _) = best_move_profile(
                        cb,
                        table,
                        def_side,
                        d0_slot,
                        foe_hp_fracs[0],
                        d0_boosts,
                        side,
                        b_slot,
                        hp_eff,
                        passed_boosts,
                        false,
                        cur_sub > 0.0,
                        false,
                    );
                    (dd, ad)
                }
            } else {
                let d0_boosts = &cb.mons[d0_idx].boosts;
                let (_, dd, ad, _) = best_move_profile(
                    cb,
                    table,
                    def_side,
                    d0_slot,
                    foe_hp_fracs[0],
                    d0_boosts,
                    side,
                    b_slot,
                    hp_eff,
                    passed_boosts,
                    false,
                    cur_sub > 0.0,
                    true,
                );
                (dd, ad)
            };
            let p_land = acc_d * d_act_mult;
            if dmg_d > 0.0 && p_land > 0.0 {
                if cur_sub > 0.0 {
                    let p_break = (4.0 * dmg_d).min(1.0) * p_land;
                    cur_sub = (cur_sub - p_break).max(0.0);
                } else {
                    hp_eff -= p_land * dmg_d;
                    if hp_eff <= 0.001 {
                        return sigmoid_f32(
                            base_logit_anchor - (w.scale as f32) * ((w.alive + w.hp * 0.5) as f32),
                        )
                        .clamp(0.01, 0.97);
                    }
                }
            }
        }
    }

    let base_sweep = simulate_lineup_sweep_ext(
        cb,
        table,
        w,
        side,
        b_slot,
        hp_eff,
        cur_sub,
        passed_boosts,
        def_side,
        foe_slots,
        foe_hp_fracs,
        base_logit_anchor,
        cached,
    );

    // If receiver enters behind a strong Substitute (`cur_sub >= 0.70`) and carries its own
    // offensive setup move (e.g. Thick Club Marowak with Swords Dance) against a lead foe
    // without Sub-bypassing disrupt, evaluate spending the Substitute to boost once.
    if cur_sub >= 0.70 && !foe_slots.is_empty() {
        let d0_slot = foe_slots[0] as usize;
        let (d0_disrupt, _) = primary_disrupt_of(cb, table, def_side, d0_slot);
        if d0_disrupt == DisruptKind::None {
            let b_idx = side * 3 + b_slot;
            let b_mon = &cb.mons[b_idx];
            let b_spec = &table.mons[b_idx];
            let d0_boosts = &cb.mons[def_side * 3 + d0_slot].boosts;
            let b_at_least_as_fast = crate::eval::compare_speed_with_boost(
                cb,
                table,
                side,
                b_slot,
                passed_boosts[4],
                def_side,
                d0_slot,
                d0_boosts[4],
            ) != Ordering::Less;
            for m_i in 0..b_spec.num_moves as usize {
                if b_mon.pp[m_i] == 0 || b_mon.is_slot_disabled(m_i) {
                    continue;
                }
                if let Some(ax) = classify_self_boost_move(table, b_idx, m_i) {
                    if ax.kind == BoostAxisKind::Offense && ax.hp_cost == 0 {
                        let (_, dmg_d0, acc_d0, _) = best_move_profile(
                            cb,
                            table,
                            def_side,
                            d0_slot,
                            foe_hp_fracs[0],
                            d0_boosts,
                            side,
                            b_slot,
                            hp_eff,
                            passed_boosts,
                            false,
                            cur_sub > 0.0,
                            false,
                        );
                        // Spending a turn to self-boost consumes the Substitute shield (either d0 breaks it
                        // or opponent pivots to their fast check on the boost turn).
                        if b_at_least_as_fast || dmg_d0 * acc_d0 < 0.45 * hp_eff {
                            let mut boosted = *passed_boosts;
                            let add = ax.stages[1];
                            if (ax.flags & AXIS_FLAG_ATK) != 0 {
                                boosted[0] = (boosted[0] + add).clamp(-6, 6);
                            }
                            if (ax.flags & AXIS_FLAG_SPA) != 0 {
                                boosted[2] = (boosted[2] + add).clamp(-6, 6);
                            }
                            if (ax.flags & AXIS_FLAG_DEF) != 0 {
                                boosted[1] = (boosted[1] + add).clamp(-6, 6);
                            }
                            if (ax.flags & AXIS_FLAG_CURSE) != 0 {
                                boosted[4] = (boosted[4] - add).clamp(-6, 6);
                            }
                            let sd_val = RELAY_GAMMA
                                * simulate_lineup_sweep(
                                    cb,
                                    table,
                                    w,
                                    side,
                                    b_slot,
                                    hp_eff,
                                    0.0,
                                    &boosted,
                                    def_side,
                                    foe_slots,
                                    foe_hp_fracs,
                                    base_logit_anchor,
                                );
                            if sd_val > base_sweep {
                                return sd_val;
                            }
                        }
                    }
                }
            }
        }
    }

    base_sweep
}

#[derive(Clone, Copy, Default)]
struct CachedFoeProfile {
    dmg_b_if_hit: f32,
    acc_b: f32,
    prio_b: i8,
    dmg_m_if_hit: f32,
    acc_m: f32,
    prio_m: i8,
}

#[inline]
fn build_cached_foe_profiles(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    b_slot: usize,
    boosts_b: &[i8; 7],
    def_side: usize,
    foe_slots: &[u8],
    foe_hp_fracs: &[f32],
) -> Option<[CachedFoeProfile; 3]> {
    let b_idx = side * 3 + b_slot;
    for mi in 0..table.mons[b_idx].num_moves as usize {
        let ms = &table.move_specs[b_idx][mi];
        if ms.var_bp_kind == VarBpKind::FlailReversal
            || ms.special_kind == SpecialMoveKind::SuperFang
        {
            return None;
        }
    }
    let mut out = [CachedFoeProfile::default(); 3];
    for (i, &m_slot_u8) in foe_slots.iter().enumerate() {
        let m_slot = m_slot_u8 as usize;
        let m_idx = def_side * 3 + m_slot;
        for mi in 0..table.mons[m_idx].num_moves as usize {
            let ms = &table.move_specs[m_idx][mi];
            if ms.special_kind == SpecialMoveKind::SuperFang
                || (ms.drain_half && !ms.bypasses_sub)
            {
                return None;
            }
        }
        let m_boosts = &cb.mons[m_idx].boosts;
        let hp_m = foe_hp_fracs[i].max(0.05);
        let (_, dmg_b_if_hit, acc_b, prio_b) = best_move_profile(
            cb, table, side, b_slot, 1.0, boosts_b, def_side, m_slot, hp_m, m_boosts, false, false,
            false,
        );
        let (_, dmg_m_if_hit, acc_m, prio_m) = best_move_profile(
            cb, table, def_side, m_slot, hp_m, m_boosts, side, b_slot, 1.0, boosts_b, false, false,
            false,
        );
        out[i] = CachedFoeProfile {
            dmg_b_if_hit,
            acc_b,
            prio_b,
            dmg_m_if_hit,
            acc_m,
            prio_m,
        };
    }
    Some(out)
}

#[allow(clippy::too_many_arguments)]
fn simulate_lineup_sweep(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    b_slot: usize,
    cur_hp: f32,
    cur_sub: f32,
    boosts_b: &[i8; 7],
    def_side: usize,
    foe_slots: &[u8],
    foe_hp_fracs: &[f32],
    base_logit_anchor: f32,
) -> f32 {
    simulate_lineup_sweep_ext(
        cb,
        table,
        w,
        side,
        b_slot,
        cur_hp,
        cur_sub,
        boosts_b,
        def_side,
        foe_slots,
        foe_hp_fracs,
        base_logit_anchor,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_lineup_sweep_ext(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    b_slot: usize,
    mut cur_hp: f32,
    mut cur_sub: f32,
    boosts_b: &[i8; 7],
    def_side: usize,
    foe_slots: &[u8],
    foe_hp_fracs: &[f32],
    base_logit_anchor: f32,
    cached: Option<&[CachedFoeProfile; 3]>,
) -> f32 {
    let b_idx = side * 3 + b_slot;
    let b_ispec = table.item_spec(cb.mons[b_idx].item);
    let b_bp = b_ispec.is_brightpowder;
    let b_lefties = b_ispec.is_leftovers;
    let scale = w.scale as f32;
    let unit_ko_logit = scale * ((w.alive + 0.85 * w.hp) as f32);

    let mut survive_prob = 1.0f32;
    let mut acc_val = 0.0f32;
    let mut kos_done = 0.0f32;

    // Order bench replacement foes `1..n` so the opponent sends out their strongest
    // revenge-killer / wall against `(b_slot, boosts_b)` first (100% bench-order-invariant!).
    let mut idx_order = [0usize, 1, 2];
    if foe_slots.len() == 3 {
        let threat_rank = |k: usize| -> f32 {
            let ms = foe_slots[k] as usize;
            let mi = def_side * 3 + ms;
            let mb = &cb.mons[mi].boosts;
            let hm = foe_hp_fracs[k].max(0.05);
            let (db, ab, dm, am) = if let Some(cp) = cached {
                (cp[k].dmg_b_if_hit, cp[k].acc_b, cp[k].dmg_m_if_hit, cp[k].acc_m)
            } else {
                let (_, db, ab, _) = best_move_profile(
                    cb, table, side, b_slot, cur_hp, boosts_b, def_side, ms, hm, mb, false, false,
                    false,
                );
                let (_, dm, am, _) = best_move_profile(
                    cb,
                    table,
                    def_side,
                    ms,
                    hm,
                    mb,
                    side,
                    b_slot,
                    cur_hp,
                    boosts_b,
                    false,
                    cur_sub > 0.0,
                    false,
                );
                (db, ab, dm, am)
            };
            let faster = if crate::eval::compare_speed_with_boost(
                cb,
                table,
                def_side,
                ms,
                mb[4],
                side,
                b_slot,
                boosts_b[4],
            ) == Ordering::Greater
            {
                1.0
            } else {
                0.0
            };
            let walls = if db * ab < 0.25 { 1.0 } else { 0.0 };
            1.5 * faster * (dm * am) + 0.8 * (dm * am) + 1.2 * walls - 0.3 * (db * ab).min(1.5)
        };
        let r1 = threat_rank(1);
        let r2 = threat_rank(2);
        if r2 > r1 + 1e-5 {
            idx_order.swap(1, 2);
        }
    }

    for &i in &idx_order[..foe_slots.len()] {
        let m_slot_u8 = foe_slots[i];
        if survive_prob <= 1e-4 {
            break;
        }
        let m_slot = m_slot_u8 as usize;
        let m_idx = def_side * 3 + m_slot;
        let m_boosts = &cb.mons[m_idx].boosts;
        let hp_m = foe_hp_fracs[i].max(0.05);
        let (dmg_b_if_hit, acc_b, prio_b, pre_m) = if let Some(cp) = cached {
            let p = &cp[i];
            (p.dmg_b_if_hit, p.acc_b, p.prio_b, Some((p.dmg_m_if_hit, p.acc_m, p.prio_m)))
        } else {
            let (_, db, ab, pb) = best_move_profile(
                cb, table, side, b_slot, cur_hp, boosts_b, def_side, m_slot, hp_m, m_boosts, false,
                false, false,
            );
            (db, ab, pb, None)
        };
        let m_roles = table.mons[m_idx].role_flags;
        let m_has_half_heal = (m_roles & nc2000_engine::compact::role_bit::HALF_HEAL) != 0
            && (0..table.mons[m_idx].num_moves as usize)
                .any(|mi| cb.mons[m_idx].pp[mi] > 0 && table.move_specs[m_idx][mi].is_half_heal);
        let m_has_rest = (m_roles & nc2000_engine::compact::role_bit::REST) != 0
            && (0..table.mons[m_idx].num_moves as usize)
                .any(|mi| cb.mons[m_idx].pp[mi] > 0 && table.move_specs[m_idx][mi].is_rest);
        let eff_dpt = dmg_b_if_hit * acc_b;
        let half_heal_frac = {
            let mx = table.mons[m_idx].maxhp.max(1);
            let amt = if table.ruleset.is_castle() {
                mx / 2
            } else {
                (mx + 1) / 2
            };
            (amt as f32) / (mx as f32)
        };
        let walled_by_heal = (m_has_half_heal
            && eff_dpt < (half_heal_frac - 0.02)
            && dmg_b_if_hit < hp_m)
            || (m_has_rest && eff_dpt < 0.34 && dmg_b_if_hit < hp_m);
        if dmg_b_if_hit <= 1e-4 || acc_b <= 1e-4 || walled_by_heal {
            // Walled by m: penalize only when walled on the very first foe (`kos_done == 0`),
            // not after already claiming >= 1 KO.
            let wall_pen = if kos_done < 0.5 {
                0.35 * scale
            } else {
                scale * (1.0 - cur_hp) * 0.4
            };
            let stop_v =
                sigmoid_f32(base_logit_anchor + kos_done * unit_ko_logit - wall_pen).min(0.89);
            acc_val += survive_prob * stop_v;
            survive_prob = 0.0;
            break;
        }

        let ceil_hits = (hp_m / dmg_b_if_hit).ceil().max(1.0);
        let cont_hits = (hp_m / dmg_b_if_hit).max(1.0);
        let mut eff_hits = 0.85 * ceil_hits + 0.15 * cont_hits;
        if m_has_rest && eff_dpt < 0.50 && ceil_hits >= 3.0 {
            eff_hits += 1.5;
        }
        let t_attempts = eff_hits / acc_b.max(0.20);

        let (dmg_m_if_hit, acc_m, prio_m) = if let Some(pm) = pre_m {
            pm
        } else {
            let (_, dm, am, pm) = best_move_profile(
                cb,
                table,
                def_side,
                m_slot,
                hp_m,
                m_boosts,
                side,
                b_slot,
                cur_hp,
                boosts_b,
                false,
                cur_sub > 0.0,
                false,
            );
            (dm, am, pm)
        };
        let faster_adv = if prio_b > prio_m {
            1.0
        } else if prio_b < prio_m {
            0.0
        } else {
            match crate::eval::compare_speed_with_boost(
                cb,
                table,
                side,
                b_slot,
                boosts_b[4],
                def_side,
                m_slot,
                m_boosts[4],
            ) {
                Ordering::Greater => 1.0,
                Ordering::Equal => 0.5,
                Ordering::Less => 0.0,
            }
        };

        let mut t_exposed = (t_attempts - faster_adv).max(0.0);
        let (incap_m, slp_m, st_m) = mon_guaranteed_incapacitated(cb, table, def_side, m_slot);
        if incap_m {
            t_exposed = (t_exposed - (slp_m as f32)).max(0.0);
        } else if slp_m >= 1 && st_m {
            t_exposed *= 0.35;
        }

        if t_exposed > 0.01 {
            let (m_disrupt, _) = primary_disrupt_of(cb, table, def_side, m_slot);
            // Phaze has priority -1, so even if b is slower, on the final KO turn b kills m BEFORE Phaze fires!
            let t_phaze_exp = match m_disrupt {
                DisruptKind::Phaze { .. } => (t_attempts - 1.0).max(0.0),
                DisruptKind::Haze | DisruptKind::PerishSong => t_exposed,
                DisruptKind::Encore { .. } => (t_exposed - 0.5).max(0.0),
                DisruptKind::None => 0.0,
            };
            if t_phaze_exp > 0.01 && m_disrupt != DisruptKind::None {
                let p_dhit = disrupt_hit_prob(table.ruleset, m_disrupt, boosts_b[6], b_bp);
                let p_stop = (1.0 - (1.0 - p_dhit).max(0.0).powf(t_phaze_exp)).clamp(0.0, 1.0);
                if p_stop > 0.0 {
                    let chip_frac = (1.0 / eff_hits).clamp(0.0, 0.75);
                    let stop_v = sigmoid_f32(
                        base_logit_anchor
                            + (kos_done + 0.35 * chip_frac) * unit_ko_logit
                            - scale * (1.0 - cur_hp) * 0.5,
                    );
                    acc_val += survive_prob * p_stop * stop_v;
                    survive_prob *= 1.0 - p_stop;
                }
            }

            if survive_prob > 1e-4 && dmg_m_if_hit > 0.0 && acc_m > 0.0 {
                let n_landed = t_exposed * acc_m;
                if cur_sub > 0.0 {
                    let hits_for_sub = (0.25 / dmg_m_if_hit).max(1.0);
                    let cap_hits = cur_sub * hits_for_sub;
                    if n_landed <= cap_hits {
                        cur_sub = (cur_sub - n_landed / hits_for_sub).max(0.0);
                    } else {
                        let overflow_hits = n_landed - cap_hits;
                        cur_sub = 0.0;
                        cur_hp -= overflow_hits * dmg_m_if_hit;
                    }
                } else {
                    cur_hp -= n_landed * dmg_m_if_hit;
                }

                if cur_hp <= 0.0 {
                    let dealt_before_ko = ((cur_hp + n_landed * dmg_m_if_hit).max(0.0)
                        / (n_landed * dmg_m_if_hit).max(1e-4))
                    .clamp(0.0, 0.95);
                    let trade_v = sigmoid_f32(
                        base_logit_anchor
                            + (kos_done + 0.65 * dealt_before_ko) * unit_ko_logit
                            - scale * ((w.alive + 0.75 * w.hp) as f32),
                    )
                    .min(0.89);
                    acc_val += survive_prob * trade_v;
                    survive_prob = 0.0;
                    break;
                }
            }
        }

        if b_lefties {
            cur_hp = (cur_hp + 0.0625 * t_attempts.min(3.0)).min(1.0);
        }
        kos_done += 1.0;
    }

    if survive_prob > 0.0 {
        let v_sweep = (0.88 + 0.09 * cur_hp.clamp(0.0, 1.0)).min(0.97);
        acc_val += survive_prob * v_sweep;
    }

    acc_val.clamp(0.01, 0.97)
}

// =========================================================================
// Closed-Form Zero-Sum Matrix Game Solver (m <= 5, n <= 3)
// =========================================================================

/// Solve 1D upper-envelope minimax $\min_{t \in [0, 1]} \max_{0 \le i < m} (t c_0[i] + (1 - t) c_1[i])$
/// and extract exact primal-dual strategies `(v*, p*, [t*, 1 - t*])`.
#[inline]
fn solve_m_by_2(m: usize, col0: &[f32; MAX_R_ACTIONS], col1: &[f32; MAX_R_ACTIONS]) -> (f32, [f32; MAX_R_ACTIONS], [f32; 2]) {
    let mut cand_t = [0.0f32; 12];
    cand_t[0] = 0.0;
    cand_t[1] = 1.0;
    let mut n_t = 2usize;

    for i in 0..m {
        let si = col0[i] - col1[i];
        for j in (i + 1)..m {
            let sj = col0[j] - col1[j];
            let denom = si - sj;
            if denom.abs() > 1e-6 {
                let t = (col1[j] - col1[i]) / denom;
                if t > 1e-6 && t < 1.0 - 1e-6 {
                    cand_t[n_t] = t;
                    n_t += 1;
                }
            }
        }
    }

    let mut best_v = f32::INFINITY;
    let mut best_t = 0.0f32;
    for &t in &cand_t[..n_t] {
        let mut g = f32::NEG_INFINITY;
        for i in 0..m {
            let val = t * col0[i] + (1.0 - t) * col1[i];
            if val > g {
                g = val;
            }
        }
        if g < best_v {
            best_v = g;
            best_t = t;
        }
    }

    let mut p = [0.0f32; MAX_R_ACTIONS];
    let tol = 1e-4f32;
    if best_t <= 1e-5 {
        let mut chosen = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for i in 0..m {
            if col1[i] >= best_v - tol {
                let s = col0[i] - col1[i];
                if s > best_s {
                    best_s = s;
                    chosen = i;
                }
            }
        }
        p[chosen] = 1.0;
    } else if best_t >= 1.0 - 1e-5 {
        let mut chosen = 0usize;
        let mut best_neg_s = f32::INFINITY;
        for i in 0..m {
            if col0[i] >= best_v - tol {
                let s = col0[i] - col1[i];
                if s < best_neg_s {
                    best_neg_s = s;
                    chosen = i;
                }
            }
        }
        p[chosen] = 1.0;
    } else {
        let mut i_pos: Option<(usize, f32)> = None;
        let mut i_neg: Option<(usize, f32)> = None;
        for i in 0..m {
            let val = best_t * col0[i] + (1.0 - best_t) * col1[i];
            if val >= best_v - tol {
                let s = col0[i] - col1[i];
                if s >= -1e-6 && i_pos.map_or(true, |(_, bs)| s > bs) {
                    i_pos = Some((i, s));
                }
                if s <= 1e-6 && i_neg.map_or(true, |(_, bs)| s < bs) {
                    i_neg = Some((i, s));
                }
            }
        }
        match (i_pos, i_neg) {
            (Some((ip, sp)), Some((ineg, sn))) => {
                if ip == ineg || (sp - sn).abs() <= 1e-6 {
                    p[ip] = 1.0;
                } else {
                    let alpha = ((-sn) / (sp - sn)).clamp(0.0, 1.0);
                    p[ip] = alpha;
                    p[ineg] = 1.0 - alpha;
                }
            }
            (Some((ip, _)), None) => p[ip] = 1.0,
            (None, Some((ineg, _))) => p[ineg] = 1.0,
            (None, None) => p[0] = 1.0,
        }
    }

    (best_v, p, [best_t, 1.0 - best_t])
}

/// Solve 2-row x 3-col lower-envelope maximin $\max_{\alpha \in [0, 1]} \min_{0 \le j < 3} (\alpha r_0[j] + (1 - \alpha) r_1[j])$.
#[inline]
fn solve_2_by_3_row(r0: &[f32; 3], r1: &[f32; 3]) -> (f32, f32) {
    let mut cand_a = [0.0f32; 5];
    cand_a[0] = 0.0;
    cand_a[1] = 1.0;
    let mut n_a = 2usize;
    for j1 in 0..3 {
        let s1 = r0[j1] - r1[j1];
        for j2 in (j1 + 1)..3 {
            let s2 = r0[j2] - r1[j2];
            let d = s1 - s2;
            if d.abs() > 1e-6 {
                let a = (r1[j2] - r1[j1]) / d;
                if a > 1e-6 && a < 1.0 - 1e-6 {
                    cand_a[n_a] = a;
                    n_a += 1;
                }
            }
        }
    }
    let mut best_v = f32::NEG_INFINITY;
    let mut best_a = 0.0f32;
    for &a in &cand_a[..n_a] {
        let mut mn = f32::INFINITY;
        for j in 0..3 {
            let val = a * r0[j] + (1.0 - a) * r1[j];
            if val < mn {
                mn = val;
            }
        }
        if mn > best_v {
            best_v = mn;
            best_a = a;
        }
    }
    (best_v, best_a)
}

/// Exact closed-form zero-sum matrix game solver for $1 \le m \le 5, 1 \le n \le 3$ (§4.2).
pub fn solve_zero_sum_small(
    mat: &[[f32; MAX_D_ACTIONS]; MAX_R_ACTIONS],
    m: usize,
    n: usize,
) -> (f32, [f32; MAX_R_ACTIONS], [f32; MAX_D_ACTIONS]) {
    let mut p_out = [0.0f32; MAX_R_ACTIONS];
    let mut q_out = [0.0f32; MAX_D_ACTIONS];
    if m == 0 || n == 0 {
        return (0.5, p_out, q_out);
    }
    if n == 1 {
        let mut best_i = 0usize;
        let mut best_v = mat[0][0];
        for i in 1..m {
            if mat[i][0] > best_v {
                best_v = mat[i][0];
                best_i = i;
            }
        }
        p_out[best_i] = 1.0;
        q_out[0] = 1.0;
        return (best_v, p_out, q_out);
    }
    if m == 1 {
        let mut best_j = 0usize;
        let mut best_v = mat[0][0];
        for j in 1..n {
            if mat[0][j] < best_v {
                best_v = mat[0][j];
                best_j = j;
            }
        }
        p_out[0] = 1.0;
        q_out[best_j] = 1.0;
        return (best_v, p_out, q_out);
    }
    if n == 2 {
        let mut c0 = [0.0f32; MAX_R_ACTIONS];
        let mut c1 = [0.0f32; MAX_R_ACTIONS];
        for i in 0..m {
            c0[i] = mat[i][0];
            c1[i] = mat[i][1];
        }
        let (v, p, q2) = solve_m_by_2(m, &c0, &c1);
        q_out[0] = q2[0];
        q_out[1] = q2[1];
        return (v, p, q_out);
    }

    // n == 3:
    // 1. Primal (q*): minimize max_i (M q)_i over the 3 boundary edges + <= 10 interior 3x3 Cramer points
    let eval_q = |q: [f32; 3]| -> f32 {
        let mut mx = f32::NEG_INFINITY;
        for i in 0..m {
            let v = mat[i][0] * q[0] + mat[i][1] * q[1] + mat[i][2] * q[2];
            if v > mx {
                mx = v;
            }
        }
        mx
    };
    let eval_p = |p: &[f32; MAX_R_ACTIONS]| -> f32 {
        let mut mn = f32::INFINITY;
        for j in 0..3 {
            let mut s = 0.0f32;
            for i in 0..m {
                s += p[i] * mat[i][j];
            }
            if s < mn {
                mn = s;
            }
        }
        mn
    };

    let mut best_v_q = f32::INFINITY;
    let mut best_q = [1.0f32, 0.0, 0.0];
    for &(j0, j1) in &[(0usize, 1usize), (1, 2), (0, 2)] {
        let mut c0 = [0.0f32; MAX_R_ACTIONS];
        let mut c1 = [0.0f32; MAX_R_ACTIONS];
        for i in 0..m {
            c0[i] = mat[i][j0];
            c1[i] = mat[i][j1];
        }
        let (_, _, q2) = solve_m_by_2(m, &c0, &c1);
        let mut q3 = [0.0f32; 3];
        q3[j0] = q2[0];
        q3[j1] = q2[1];
        let vq = eval_q(q3);
        if vq < best_v_q {
            best_v_q = vq;
            best_q = q3;
        }
    }

    // Dual (p*): maximize min_j (p^T M)_j over singletons + row-pairs + 3x3 Cramer triples
    let mut best_v_p = f32::NEG_INFINITY;
    for i in 0..m {
        let mut p_cand = [0.0f32; MAX_R_ACTIONS];
        p_cand[i] = 1.0;
        let vp = eval_p(&p_cand);
        if vp > best_v_p {
            best_v_p = vp;
            p_out = p_cand;
        }
    }
    for i1 in 0..m {
        for i2 in (i1 + 1)..m {
            let (_, alpha) = solve_2_by_3_row(&mat[i1], &mat[i2]);
            let mut p_cand = [0.0f32; MAX_R_ACTIONS];
            p_cand[i1] = alpha;
            p_cand[i2] = 1.0 - alpha;
            let vp = eval_p(&p_cand);
            if vp > best_v_p {
                best_v_p = vp;
                p_out = p_cand;
            }
        }
    }

    // 3x3 interior equalizing solutions via Cramer's rule on differences (M_sub q = v 1, 1^T q = 1)
    if m >= 3 {
        for i1 in 0..m {
            for i2 in (i1 + 1)..m {
                for i3 in (i2 + 1)..m {
                    // System for q = (q0, q1, 1 - q0 - q1):
                    // (a00)*q0 + (a01)*q1 = b0  for row i1 - row i3
                    // (a10)*q0 + (a11)*q1 = b1  for row i2 - row i3
                    let r1 = &mat[i1];
                    let r2 = &mat[i2];
                    let r3 = &mat[i3];
                    let a00 = (r1[0] - r1[2]) - (r3[0] - r3[2]);
                    let a01 = (r1[1] - r1[2]) - (r3[1] - r3[2]);
                    let b0 = r3[2] - r1[2];
                    let a10 = (r2[0] - r2[2]) - (r3[0] - r3[2]);
                    let a11 = (r2[1] - r2[2]) - (r3[1] - r3[2]);
                    let b1 = r3[2] - r2[2];
                    let det = a00 * a11 - a01 * a10;
                    if det.abs() > 1e-7 {
                        let q0 = (b0 * a11 - a01 * b1) / det;
                        let q1 = (a00 * b1 - b0 * a10) / det;
                        let q2 = 1.0 - q0 - q1;
                        if q0 >= -1e-5 && q1 >= -1e-5 && q2 >= -1e-5 {
                            let s = (q0.max(0.0) + q1.max(0.0) + q2.max(0.0)).max(1e-6);
                            let q3 = [q0.max(0.0) / s, q1.max(0.0) / s, q2.max(0.0) / s];
                            let vq = eval_q(q3);
                            if vq < best_v_q {
                                best_v_q = vq;
                                best_q = q3;
                            }
                        }
                        // Transpose system for p = (p_i1, p_i2, 1 - p_i1 - p_i2):
                        // Col 0 - Col 2 and Col 1 - Col 2
                        let t00 = (r1[0] - r3[0]) - (r1[2] - r3[2]);
                        let t01 = (r2[0] - r3[0]) - (r2[2] - r3[2]);
                        let tb0 = r3[2] - r3[0];
                        let t10 = (r1[1] - r3[1]) - (r1[2] - r3[2]);
                        let t11 = (r2[1] - r3[1]) - (r2[2] - r3[2]);
                        let tb1 = r3[2] - r3[1];
                        let tdet = t00 * t11 - t01 * t10;
                        if tdet.abs() > 1e-7 {
                            let p0 = (tb0 * t11 - t01 * tb1) / tdet;
                            let p1 = (t00 * tb1 - tb0 * t10) / tdet;
                            let p2 = 1.0 - p0 - p1;
                            if p0 >= -1e-5 && p1 >= -1e-5 && p2 >= -1e-5 {
                                let sp = (p0.max(0.0) + p1.max(0.0) + p2.max(0.0)).max(1e-6);
                                let mut p_cand = [0.0f32; MAX_R_ACTIONS];
                                p_cand[i1] = p0.max(0.0) / sp;
                                p_cand[i2] = p1.max(0.0) / sp;
                                p_cand[i3] = p2.max(0.0) / sp;
                                let vp = eval_p(&p_cand);
                                if vp > best_v_p {
                                    best_v_p = vp;
                                    p_out = p_cand;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    q_out = best_q;
    (0.5 * (best_v_p + best_v_q), p_out, q_out)
}

#[inline]
fn quantize_policy_u8<const N: usize>(probs: &[f32; N]) -> [u8; N] {
    let mut out = [0u8; N];
    let mut rem = 255i32;
    let mut max_i = 0usize;
    let mut max_p = -1.0f32;
    for i in 0..N {
        let p = probs[i].clamp(0.0, 1.0);
        if p > max_p {
            max_p = p;
            max_i = i;
        }
        let q = (p * 255.0).floor() as i32;
        out[i] = q.clamp(0, 255) as u8;
        rem -= out[i] as i32;
    }
    if rem > 0 {
        out[max_i] = (out[max_i] as i32 + rem).clamp(0, 255) as u8;
    }
    out
}

// =========================================================================
// Backward-Induction Bellman Subgame Solver
// =========================================================================

fn solve_relay_subgame_into(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    lookup: &ActiveRelayLookup,
    recv_arr: &[CanonBench; 2],
    def_arr: &[CanonBench; 2],
    out: &mut SolvedRelaySubgame,
) {
    let act_slot = lookup.act_slot as usize;
    let act_idx = side * 3 + act_slot;
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    let n_recv = lookup.n_recv as usize;
    let n_def_bench = lookup.n_def_bench as usize;

    let ax1 = &lookup.axis1;
    let ax2 = &lookup.axis2;
    let k1_len = ax1.len as usize;
    let k2_len = ax2.len as usize;

    let counter_opt = select_best_counter(
        cb,
        table,
        side,
        act_slot,
        def_side,
        def_slot,
        &def_arr[..n_def_bench],
    );
    let has_counter_pivot = counter_opt.is_some();
    let counter_canon_idx = counter_opt.unwrap_or(0) as u8;

    out.key = lookup.key;
    out.axis1 = *ax1;
    out.axis2 = *ax2;
    out.r_move_slots = [
        lookup.atk_move_slot,
        lookup.sub_move_slot,
        ax1.move_slot,
        ax2.move_slot,
        lookup.bp_move_slot,
    ];
    out.d0_brk_slot = lookup.def_break_slot[0];
    out.d0_dis_slot = lookup.def_disrupt_slot[0];
    out.k1_len = ax1.len;
    out.k2_len = ax2.len;
    out.has_counter_pivot = has_counter_pivot;
    out.counter_canon_idx = counter_canon_idx;

    // Canonical baseline score differential (u_R at reference h_R = 4, sigma = 0, boosts = 0)
    let scale = w.scale as f32;
    let w_hp = w.hp as f32;
    let w_alive = w.alive as f32;
    let r_act_st = cb.mons[act_idx].status_raw();
    let (_, r_act_slp, _) = mon_guaranteed_incapacitated(cb, table, side, act_slot);
    let mut score_r_ref4 =
        w_alive + w_hp - mon_status_pen(w, table.ruleset, r_act_st, r_act_slp);
    let flying_mask = 1u32 << (table.flying_type & 31);
    let mut r_grounded_bench = 0u32;
    for r in &recv_arr[..n_recv] {
        let r_idx = side * 3 + r.party_slot as usize;
        if (table.mons[r_idx].type_mask & flying_mask) == 0 {
            r_grounded_bench += 1;
        }
        score_r_ref4 += w_alive
            + w_hp * ((r.b8_hp as f32) / 8.0)
            - mon_status_pen(w, table.ruleset, r.status_raw, r.slp_rem);
    }
    let r_spk_layers = (cb.sides[side].side_flags & 0x03) as usize;
    if r_spk_layers > 0 && r_grounded_bench > 0 {
        const AMOUNTS: [f32; 4] = [0.0, 3.0, 4.0, 6.0];
        score_r_ref4 -= (w.spikes as f32) * (AMOUNTS[r_spk_layers] / 24.0) * (r_grounded_bench as f32);
    }

    let d_act_idx = def_side * 3 + def_slot;
    let d_act_b8 = hp_to_b4(cb.mons[d_act_idx].hp(), table.mons[d_act_idx].maxhp) * 2;
    let (_, d_act_slp, _) = mon_guaranteed_incapacitated(cb, table, def_side, def_slot);
    let mut score_d_base = w_alive
        + w_hp * ((d_act_b8 as f32) / 8.0)
        - mon_status_pen(w, table.ruleset, cb.mons[d_act_idx].status_raw(), d_act_slp);
    let mut d_grounded_bench = 0u32;
    for d in &def_arr[..n_def_bench] {
        let d_idx = def_side * 3 + d.party_slot as usize;
        if (table.mons[d_idx].type_mask & flying_mask) == 0 {
            d_grounded_bench += 1;
        }
        score_d_base += w_alive
            + w_hp * ((d.b8_hp as f32) / 8.0)
            - mon_status_pen(w, table.ruleset, d.status_raw, d.slp_rem);
    }
    let d_spk_layers = (cb.sides[def_side].side_flags & 0x03) as usize;
    if d_spk_layers > 0 && d_grounded_bench > 0 {
        const AMOUNTS: [f32; 4] = [0.0, 3.0, 4.0, 6.0];
        score_d_base -= (w.spikes as f32) * (AMOUNTS[d_spk_layers] / 24.0) * (d_grounded_bench as f32);
    }

    let base_logit_ref4 = scale * (score_r_ref4 - score_d_base);

    // Precompute W_pass(ell, sigma, k1, k2, incoming_atk) AND V_direct(ell, h_R, sigma, k1, k2) for u_R itself
    let mut w_pass_no_atk = [0.5f32; 48];
    let mut w_pass_in_atk = [0.5f32; 48];
    let mut v_direct_tbl = [0.5f32; MAX_RELAY_STATES];
    let max_ell = if has_counter_pivot { 1usize } else { 0usize };

    for ell in 0..=max_ell {
        // Lineup M_D for this ell
        let mut foe_slots = [0u8; 3];
        let mut foe_hps = [0.0f32; 3];
        let mut n_foes = 0usize;
        let d_lead_slot = if ell == 0 {
            def_slot as u8
        } else {
            def_arr[counter_canon_idx as usize].party_slot
        };
        let d_lead_hp = if ell == 0 {
            (d_act_b8 as f32) / 8.0
        } else {
            let c = &def_arr[counter_canon_idx as usize];
            let c_idx = def_side * 3 + c.party_slot as usize;
            let spk = (switch_entry_dmg(cb, table, def_side, c.party_slot as usize, None) as f32)
                / (table.mons[c_idx].maxhp.max(1) as f32);
            ((c.b8_hp as f32) / 8.0 - spk).max(0.05)
        };
        foe_slots[0] = d_lead_slot;
        foe_hps[0] = d_lead_hp;
        n_foes += 1;
        if ell == 1 {
            foe_slots[n_foes] = def_slot as u8;
            foe_hps[n_foes] = (d_act_b8 as f32) / 8.0;
            n_foes += 1;
        }
        for (di, d) in def_arr[..n_def_bench].iter().enumerate() {
            if ell == 1 && di == counter_canon_idx as usize {
                continue;
            }
            foe_slots[n_foes] = d.party_slot;
            foe_hps[n_foes] = (d.b8_hp as f32) / 8.0;
            n_foes += 1;
        }

        let pivot_spk_bonus = if ell == 1 {
            let c = &def_arr[counter_canon_idx as usize];
            let c_idx = def_side * 3 + c.party_slot as usize;
            let spk = (switch_entry_dmg(cb, table, def_side, c.party_slot as usize, None) as f32)
                / (table.mons[c_idx].maxhp.max(1) as f32);
            scale * w_hp * spk
        } else {
            0.0
        };

        for k1 in 0..k1_len {
            for k2 in 0..k2_len {
                let passed_b = compose_boosts(ax1, k1, ax2, k2);
                let recv_profiles: [Option<[CachedFoeProfile; 3]>; 2] = [
                    if n_recv >= 1 && recv_arr[0].survives_spikes {
                        build_cached_foe_profiles(
                            cb,
                            table,
                            side,
                            recv_arr[0].party_slot as usize,
                            &passed_b,
                            def_side,
                            &foe_slots[..n_foes],
                            &foe_hps[..n_foes],
                        )
                    } else {
                        None
                    },
                    if n_recv >= 2 && recv_arr[1].survives_spikes {
                        build_cached_foe_profiles(
                            cb,
                            table,
                            side,
                            recv_arr[1].party_slot as usize,
                            &passed_b,
                            def_side,
                            &foe_slots[..n_foes],
                            &foe_hps[..n_foes],
                        )
                    } else {
                        None
                    },
                ];
                let r_profiles = build_cached_foe_profiles(
                    cb,
                    table,
                    side,
                    act_slot,
                    &passed_b,
                    def_side,
                    &foe_slots[..n_foes],
                    &foe_hps[..n_foes],
                );

                for sigma in 0..2usize {
                    let r_i = recv_idx(ell, sigma, k1, k2, k1_len, k2_len);
                    let mut best_no = 0.01f32;
                    let mut best_in = 0.01f32;
                    let mut best_rc = 0u8;

                    for (rc_i, r_cand) in recv_arr[..n_recv].iter().enumerate() {
                        if !r_cand.survives_spikes {
                            continue;
                        }
                        let b_slot = r_cand.party_slot as usize;
                        let b_idx = side * 3 + b_slot;
                        let entry_r = (switch_entry_dmg(cb, table, side, b_slot, Some(act_slot))
                            as f32)
                            / (table.mons[b_idx].maxhp.max(1) as f32);
                        let hp_entry = ((r_cand.b8_hp as f32) / 8.0 - entry_r).max(0.0);
                        let anchor_here = base_logit_ref4 + pivot_spk_bonus - scale * w_hp * entry_r;
                        let rc_prof = recv_profiles[rc_i].as_ref();

                        let v_no = evaluate_single_receiver_sweep_ext(
                            cb,
                            table,
                            w,
                            side,
                            b_slot,
                            hp_entry,
                            r_cand.slp_rem,
                            r_cand.status_raw,
                            sigma == 1,
                            &passed_b,
                            def_side,
                            &foe_slots[..n_foes],
                            &foe_hps[..n_foes],
                            false,
                            anchor_here,
                            rc_prof,
                        );
                        let v_in = evaluate_single_receiver_sweep_ext(
                            cb,
                            table,
                            w,
                            side,
                            b_slot,
                            hp_entry,
                            r_cand.slp_rem,
                            r_cand.status_raw,
                            sigma == 1,
                            &passed_b,
                            def_side,
                            &foe_slots[..n_foes],
                            &foe_hps[..n_foes],
                            true,
                            anchor_here,
                            rc_prof,
                        );
                        if v_no > best_no {
                            best_no = v_no;
                            best_rc = rc_i as u8;
                        }
                        if v_in > best_in {
                            best_in = v_in;
                        }
                    }
                    w_pass_no_atk[r_i] = best_no;
                    w_pass_in_atk[r_i] = best_in;
                    out.best_receiver_canon[r_i] = best_rc;

                    // Precompute direct-combat value for u_R itself attacking without Baton Passing
                    for h_r in 1..=4usize {
                        let r_hp = (h_r as f32) * 0.25;
                        let anchor_r =
                            base_logit_ref4 + pivot_spk_bonus + scale * w_hp * (r_hp - 1.0);
                        let vd = simulate_lineup_sweep_ext(
                            cb,
                            table,
                            w,
                            side,
                            act_slot,
                            r_hp,
                            if sigma == 1 { 1.0 } else { 0.0 },
                            &passed_b,
                            def_side,
                            &foe_slots[..n_foes],
                            &foe_hps[..n_foes],
                            anchor_r,
                            r_profiles.as_ref(),
                        );
                        v_direct_tbl[state_idx(ell, h_r, sigma, k1, k2, k1_len, k2_len)] = vd;
                    }
                }
            }
        }
    }

    // Helper for non-relay reset / faint value at (h_R, ell, foe_chip)
    let v_reset_fn = |h_r: i32, ell: usize, foe_chip: f32, foe_hp_cur: f32| -> f32 {
        let c_spk = if ell == 1 {
            let c = &def_arr[counter_canon_idx as usize];
            let c_idx = def_side * 3 + c.party_slot as usize;
            (switch_entry_dmg(cb, table, def_side, c.party_slot as usize, None) as f32)
                / (table.mons[c_idx].maxhp.max(1) as f32)
        } else {
            0.0
        };
        let r_delta = if h_r <= 0 {
            -(w_alive + w_hp)
        } else {
            w_hp * ((h_r as f32) * 0.25 - 1.0)
        };
        let actual_chip = foe_chip.min(foe_hp_cur);
        let ko_bonus = if foe_chip >= foe_hp_cur { w_alive } else { 0.0 };
        let d_delta = -(w_hp * (c_spk + actual_chip) + ko_bonus);
        sigmoid_f32(base_logit_ref4 + scale * (r_delta - d_delta)).clamp(0.01, 0.99)
    };

    let (d0_disrupt, _) = primary_disrupt_of(cb, table, def_side, def_slot);
    let (d1_disrupt, _) = if has_counter_pivot {
        primary_disrupt_of(
            cb,
            table,
            def_side,
            def_arr[counter_canon_idx as usize].party_slot as usize,
        )
    } else {
        (DisruptKind::None, NO_MOVE_SLOT)
    };
    let n_outer_passes = if matches!(d0_disrupt, DisruptKind::Haze)
        || matches!(d1_disrupt, DisruptKind::Haze)
    {
        2usize
    } else {
        1usize
    };

    let zero_b = [0i8; 7];
    let r_bp_item = table.item_spec(cb.mons[act_idx].item).is_brightpowder;

    for pass_i in 0..n_outer_passes {
        let ell_start = if has_counter_pivot { 1usize } else { 0usize };
        for ell_rev in 0..=ell_start {
            let ell = ell_start - ell_rev; // 1 -> 0 so ell=1 is finalized before ell=0 Pivot reads it
            let d_slot = if ell == 0 {
                def_slot
            } else {
                def_arr[counter_canon_idx as usize].party_slot as usize
            };
            let d_hp_cur = if ell == 0 {
                (d_act_b8 as f32) / 8.0
            } else {
                let c = &def_arr[counter_canon_idx as usize];
                let c_idx = def_side * 3 + c.party_slot as usize;
                let spk = (switch_entry_dmg(cb, table, def_side, c.party_slot as usize, None)
                    as f32)
                    / (table.mons[c_idx].maxhp.max(1) as f32);
                ((c.b8_hp as f32) / 8.0 - spk).max(0.05)
            };
            let d_disrupt = if ell == 0 { d0_disrupt } else { d1_disrupt };
            let (incap_d, slp_d, st_d) = mon_guaranteed_incapacitated(cb, table, def_side, d_slot);
            let d_awake_mult = if incap_d && slp_d >= 2 {
                0.0f32
            } else if slp_d >= 1 && st_d {
                0.33f32
            } else if incap_d {
                0.35f32
            } else {
                1.0f32
            };
            let d_idx_cur = def_side * 3 + d_slot;
            let d_inflicts_status = (0..table.mons[d_idx_cur].num_moves as usize).any(|mi| {
                cb.mons[d_idx_cur].pp[mi] > 0
                    && (table.move_specs[d_idx_cur][mi].primary_status != 0
                        || (table.move_specs[d_idx_cur][mi].sec_status == Status::Par as u8
                            && table.move_specs[d_idx_cur][mi].sec_chance_256 >= 60))
            });
            let unsubbed_status_pen = if d_inflicts_status && d_awake_mult > 0.0 {
                0.035f32 * d_awake_mult
            } else {
                0.0f32
            };
            let d_boosts = &cb.mons[def_side * 3 + d_slot].boosts;
            let d_has_drain = (0..table.mons[d_idx_cur].num_moves as usize).any(|mi| {
                cb.mons[d_idx_cur].pp[mi] > 0
                    && !cb.mons[d_idx_cur].is_slot_disabled(mi)
                    && table.move_specs[d_idx_cur][mi].drain_half
                    && !table.move_specs[d_idx_cur][mi].bypasses_sub
            });

            let r_has_var_bp = (0..table.mons[act_idx].num_moves as usize).any(|mi| {
                table.move_specs[act_idx][mi].var_bp_kind == VarBpKind::FlailReversal
                    || table.move_specs[act_idx][mi].special_kind == SpecialMoveKind::SuperFang
            }) || (0..table.mons[d_idx_cur].num_moves as usize).any(|mi| {
                table.move_specs[d_idx_cur][mi].special_kind == SpecialMoveKind::SuperFang
            });
            let (piv_c_slot, piv_c_hp) = if ell == 0 && has_counter_pivot {
                let c_cand = &def_arr[counter_canon_idx as usize];
                let c_slot = c_cand.party_slot as usize;
                let c_idx = def_side * 3 + c_slot;
                let c_spk = (switch_entry_dmg(cb, table, def_side, c_slot, None) as f32)
                    / (table.mons[c_idx].maxhp.max(1) as f32);
                (c_slot, ((c_cand.b8_hp as f32) / 8.0 - c_spk).max(0.05))
            } else {
                (0usize, 1.0f32)
            };

            for k1_rev in 0..k1_len {
                let k1 = k1_len - 1 - k1_rev;
                for k2_rev in 0..k2_len {
                    let k2 = k2_len - 1 - k2_rev;

                    let boosts_r = compose_boosts(ax1, k1, ax2, k2);
                    let p_first_neutral = match crate::eval::compare_speed_with_boost(
                        cb,
                        table,
                        side,
                        act_slot,
                        boosts_r[4],
                        def_side,
                        d_slot,
                        d_boosts[4],
                    ) {
                        Ordering::Greater => 1.0f32,
                        Ordering::Less => 0.0f32,
                        Ordering::Equal => 0.5f32,
                    };

                    let (_, dmg_d_if_hit_base, acc_d_raw_base, prio_d_brk_base) = if !r_has_var_bp {
                        best_move_profile(
                            cb, table, def_side, d_slot, d_hp_cur, d_boosts, side, act_slot, 1.0,
                            &boosts_r, false, false, false,
                        )
                    } else {
                        (NO_MOVE_SLOT, 0.0, 1.0, 0)
                    };
                    let (dmg_d_sub_base, acc_d_sub_base) = if d_has_drain && !r_has_var_bp {
                        let (_, dm, ac, _) = best_move_profile(
                            cb, table, def_side, d_slot, d_hp_cur, d_boosts, side, act_slot, 1.0,
                            &boosts_r, false, true, false,
                        );
                        (dm, ac)
                    } else {
                        (dmg_d_if_hit_base, acc_d_raw_base)
                    };

                    // Hoist post-boost D-vs-R Break profiles once per (ell, k1, k2)
                    let (dmg_d_post1, acc_d_post1, dmg_d_post1_sub, acc_d_post1_sub, b_post1_eva) =
                        if ax1.move_slot < 4 && k1 + 1 < k1_len {
                            let bp1 = compose_boosts(ax1, k1 + 1, ax2, k2);
                            let (_, dm, ac, _) = best_move_profile(
                                cb, table, def_side, d_slot, d_hp_cur, d_boosts, side, act_slot,
                                1.0, &bp1, false, false, false,
                            );
                            let (dms, acs) = if d_has_drain {
                                let (_, d2, a2, _) = best_move_profile(
                                    cb, table, def_side, d_slot, d_hp_cur, d_boosts, side,
                                    act_slot, 1.0, &bp1, false, true, false,
                                );
                                (d2, a2 * d_awake_mult)
                            } else {
                                (dm, ac * d_awake_mult)
                            };
                            (dm, ac * d_awake_mult, dms, acs, bp1[6])
                        } else {
                            (0.0f32, 0.0f32, 0.0f32, 0.0f32, boosts_r[6])
                        };
                    let (dmg_d_post2, acc_d_post2, dmg_d_post2_sub, acc_d_post2_sub, b_post2_eva) =
                        if ax2.move_slot < 4 && k2 + 1 < k2_len {
                            let bp2 = compose_boosts(ax1, k1, ax2, k2 + 1);
                            let (_, dm, ac, _) = best_move_profile(
                                cb, table, def_side, d_slot, d_hp_cur, d_boosts, side, act_slot,
                                1.0, &bp2, false, false, false,
                            );
                            let (dms, acs) = if d_has_drain {
                                let (_, d2, a2, _) = best_move_profile(
                                    cb, table, def_side, d_slot, d_hp_cur, d_boosts, side,
                                    act_slot, 1.0, &bp2, false, true, false,
                                );
                                (d2, a2 * d_awake_mult)
                            } else {
                                (dm, ac * d_awake_mult)
                            };
                            (dm, ac * d_awake_mult, dms, acs, bp2[6])
                        } else {
                            (0.0f32, 0.0f32, 0.0f32, 0.0f32, boosts_r[6])
                        };

                    let (_, dmg_r_d_base, acc_r_d_base, prio_r_base) = if !r_has_var_bp {
                        best_move_profile(
                            cb, table, side, act_slot, 1.0, &boosts_r, def_side, d_slot, d_hp_cur,
                            d_boosts, false, false, false,
                        )
                    } else {
                        (NO_MOVE_SLOT, 0.0, 1.0, 0)
                    };
                    let chip_c_base = if ell == 0 && has_counter_pivot && !r_has_var_bp {
                        let (_, dm_c, ac_c, _) = best_move_profile(
                            cb, table, side, act_slot, 1.0, &boosts_r, def_side, piv_c_slot,
                            piv_c_hp, &zero_b, false, false, false,
                        );
                        dm_c * ac_c
                    } else {
                        0.0
                    };

                    for h_r in 1..=4usize {
                        let r_hp_frac = (h_r as f32) * 0.25;
                        let (dmg_d_if_hit, acc_d_raw, prio_d_brk) = if r_has_var_bp {
                            let (_, dm, ac, pr) = best_move_profile(
                                cb, table, def_side, d_slot, d_hp_cur, d_boosts, side, act_slot,
                                r_hp_frac, &boosts_r, false, false, false,
                            );
                            (dm, ac, pr)
                        } else {
                            (dmg_d_if_hit_base, acc_d_raw_base, prio_d_brk_base)
                        };
                        let (dmg_d_sub_if_hit, acc_d_sub_raw) = if d_has_drain && r_has_var_bp {
                            let (_, dm, ac, _) = best_move_profile(
                                cb, table, def_side, d_slot, d_hp_cur, d_boosts, side, act_slot,
                                r_hp_frac, &boosts_r, false, true, false,
                            );
                            (dm, ac)
                        } else if d_has_drain {
                            (dmg_d_sub_base, acc_d_sub_base)
                        } else {
                            (dmg_d_if_hit, acc_d_raw)
                        };
                        let acc_d = acc_d_raw * d_awake_mult;
                        let acc_d_sub = acc_d_sub_raw * d_awake_mult;
                        let p_status_first_vs_brk = if 0 > prio_d_brk {
                            1.0f32
                        } else if 0 < prio_d_brk {
                            0.0f32
                        } else {
                            p_first_neutral
                        };
                        let (dmg_r_to_d, acc_r_to_d, prio_r_atk) = if r_has_var_bp {
                            let (_, dm, ac, pr) = best_move_profile(
                                cb, table, side, act_slot, r_hp_frac, &boosts_r, def_side, d_slot,
                                d_hp_cur, d_boosts, false, false, false,
                            );
                            (dm, ac, pr)
                        } else {
                            (dmg_r_d_base, acc_r_d_base, prio_r_base)
                        };
                        let ev_r_to_d = dmg_r_to_d * acc_r_to_d;
                        let p_r_ko_d = if dmg_r_to_d >= d_hp_cur {
                            acc_r_to_d
                        } else if dmg_r_to_d >= 0.85 * d_hp_cur {
                            0.4 * acc_r_to_d
                        } else {
                            0.0
                        };
                        let chip_c_here = if ell == 0 && has_counter_pivot {
                            if r_has_var_bp {
                                let (_, dm_c, ac_c, _) = best_move_profile(
                                    cb, table, side, act_slot, r_hp_frac, &boosts_r, def_side,
                                    piv_c_slot, piv_c_hp, &zero_b, false, false, false,
                                );
                                dm_c * ac_c
                            } else {
                                chip_c_base
                            }
                        } else {
                            0.0
                        };

                        // Initialize (sigma=0, sigma=1) pair and run 2 exact Bellman sweeps
                        let idx_s0 = state_idx(ell, h_r, 0, k1, k2, k1_len, k2_len);
                        let idx_s1 = state_idx(ell, h_r, 1, k1, k2, k1_len, k2_len);
                        if pass_i == 0 {
                            out.v_star[idx_s0] = v_direct_tbl[idx_s0];
                            out.v_star[idx_s1] = v_direct_tbl[idx_s1];
                            out.v_no_relay[idx_s0] = v_direct_tbl[idx_s0];
                            out.v_no_relay[idx_s1] = v_direct_tbl[idx_s1];
                        }

                        for _vi in 0..3usize {
                            for sigma in 0..2usize {
                                let cur_idx = if sigma == 0 { idx_s0 } else { idx_s1 };
                                let v_dir_here = v_direct_tbl[cur_idx];

                                // Enumerate active R actions: Pass requires holding >= 1 asset (sigma==1 or boost > 0)
                                let mut r_acts = [0usize; MAX_R_ACTIONS];
                                let mut m = 0usize;
                                r_acts[m] = ACT_R_ATTACK;
                                m += 1;
                                if lookup.sub_move_slot < 4 && sigma == 0 && h_r >= 2 {
                                    r_acts[m] = ACT_R_SUB;
                                    m += 1;
                                }
                                if ax1.move_slot < 4 && k1 + 1 < k1_len && h_r > ax1.hp_cost as usize {
                                    r_acts[m] = ACT_R_BOOST1;
                                    m += 1;
                                }
                                if ax2.move_slot < 4 && k2 + 1 < k2_len && h_r > ax2.hp_cost as usize {
                                    r_acts[m] = ACT_R_BOOST2;
                                    m += 1;
                                }
                                if sigma == 1 || k1 > 0 || k2 > 0 {
                                    r_acts[m] = ACT_R_PASS;
                                    m += 1;
                                }

                                // Enumerate active D actions and map back to 0..3
                                let mut d_acts = [0usize; MAX_D_ACTIONS];
                                let mut n = 0usize;
                                d_acts[n] = ACT_D_BREAK;
                                n += 1;
                                if d_disrupt != DisruptKind::None && d_awake_mult > 0.0 {
                                    d_acts[n] = ACT_D_DISRUPT;
                                    n += 1;
                                }
                                if ell == 0 && has_counter_pivot {
                                    d_acts[n] = ACT_D_PIVOT;
                                    n += 1;
                                }

                                let lookup_v = |tbl: &[f32; MAX_RELAY_STATES],
                                                l_n: usize,
                                                h_n: i32,
                                                s_n: usize,
                                                k1_n: usize,
                                                k2_n: usize,
                                                foe_chip: f32|
                                 -> f32 {
                                    if h_n <= 0 {
                                        return RELAY_GAMMA * v_reset_fn(0, l_n, foe_chip, d_hp_cur);
                                    }
                                    let base = tbl[state_idx(l_n, h_n as usize, s_n, k1_n, k2_n, k1_len, k2_len)];
                                    let st_pen = if s_n == 0 && l_n == ell { unsubbed_status_pen } else { 0.0 };
                                    let chip_lift = if foe_chip > 0.0 {
                                        let ko_b = if foe_chip >= d_hp_cur { 0.14 } else { 0.0 };
                                        0.16 * foe_chip.min(d_hp_cur) + ko_b
                                    } else {
                                        0.0
                                    };
                                    (RELAY_GAMMA * base + chip_lift - st_pen).clamp(0.01, 0.98)
                                };

                                // Stochastic bucket damage helper on u_R from Break
                                let apply_brk_to_state =
                                    |tbl: &[f32; MAX_RELAY_STATES],
                                     l_n: usize,
                                     h_cur: i32,
                                     s_cur: usize,
                                     k1_n: usize,
                                     k2_n: usize,
                                     foe_chip: f32|
                                     -> f32 {
                                        if h_cur <= 0 {
                                            return lookup_v(tbl, l_n, 0, 0, k1_n, k2_n, foe_chip);
                                        }
                                        let v_miss = lookup_v(tbl, l_n, h_cur, s_cur, k1_n, k2_n, foe_chip);
                                        let (dmg_here, acc_here) = if s_cur == 1 {
                                            (dmg_d_sub_if_hit, acc_d_sub)
                                        } else {
                                            (dmg_d_if_hit, acc_d)
                                        };
                                        if acc_here <= 0.0 || dmg_here <= 0.0 {
                                            return v_miss;
                                        }
                                        let v_hit = if s_cur == 1 {
                                            let p_b = (4.0 * dmg_here).min(1.0);
                                            let v0 = lookup_v(tbl, l_n, h_cur, 0, k1_n, k2_n, foe_chip);
                                            let v1 = lookup_v(tbl, l_n, h_cur, 1, k1_n, k2_n, foe_chip);
                                            p_b * v0 + (1.0 - p_b) * v1
                                        } else {
                                            let z = 4.0 * dmg_here;
                                            let dh_lo = (z.floor() as i32).min(4);
                                            let dh_hi = (dh_lo + 1).min(4);
                                            let p_hi = (z - (dh_lo as f32)).clamp(0.0, 1.0);
                                            let vlo = lookup_v(tbl, l_n, h_cur - dh_lo, 0, k1_n, k2_n, foe_chip);
                                            let vhi = lookup_v(tbl, l_n, h_cur - dh_hi, 0, k1_n, k2_n, foe_chip);
                                            (1.0 - p_hi) * vlo + p_hi * vhi
                                        };
                                        (1.0 - acc_here) * v_miss + acc_here * v_hit
                                    };

                                let mut mat = [[0.5f32; MAX_D_ACTIONS]; MAX_R_ACTIONS];
                                let mut mat_norelay = [[0.5f32; MAX_D_ACTIONS]; MAX_R_ACTIONS];

                                for (ri, &a_r) in r_acts[..m].iter().enumerate() {
                                    for (dj, &a_d) in d_acts[..n].iter().enumerate() {
                                        let cell = match a_d {
                                            ACT_D_PIVOT => match a_r {
                                                ACT_R_SUB => {
                                                    lookup_v(&out.v_star, 1, h_r as i32 - 1, 1, k1, k2, 0.0)
                                                }
                                                ACT_R_BOOST1 => lookup_v(
                                                    &out.v_star,
                                                    1,
                                                    h_r as i32 - ax1.hp_cost as i32,
                                                    sigma,
                                                    k1 + 1,
                                                    k2,
                                                    0.0,
                                                ),
                                                ACT_R_BOOST2 => lookup_v(
                                                    &out.v_star,
                                                    1,
                                                    h_r as i32 - ax2.hp_cost as i32,
                                                    sigma,
                                                    k1,
                                                    k2 + 1,
                                                    0.0,
                                                ),
                                                ACT_R_PASS => {
                                                    RELAY_GAMMA
                                                        * w_pass_no_atk[recv_idx(1, sigma, k1, k2, k1_len, k2_len)]
                                                }
                                                _ => {
                                                    let base1 = out.v_star[state_idx(1, h_r, sigma, k1, k2, k1_len, k2_len)]
                                                        .max(v_direct_tbl[state_idx(1, h_r, sigma, k1, k2, k1_len, k2_len)]);
                                                    let ko_c = if chip_c_here >= piv_c_hp { 0.18 } else { 0.0 };
                                                    (RELAY_GAMMA * base1 + 0.18 * chip_c_here.min(piv_c_hp) + ko_c)
                                                        .clamp(0.01, 0.98)
                                                }
                                            },
                                            ACT_D_DISRUPT => match d_disrupt {
                                                DisruptKind::Phaze { .. } => {
                                                    match a_r {
                                                        ACT_R_ATTACK => {
                                                            let p_phz = disrupt_hit_prob(table.ruleset, d_disrupt, boosts_r[6], r_bp_item) * d_awake_mult;
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32, ell, ev_r_to_d, d_hp_cur);
                                                            p_r_ko_d * v_dir_here + (1.0 - p_r_ko_d) * (p_phz * v_rst + (1.0 - p_phz) * v_dir_here)
                                                        }
                                                        ACT_R_SUB => {
                                                            let p_phz = disrupt_hit_prob(table.ruleset, d_disrupt, boosts_r[6], r_bp_item) * d_awake_mult;
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32 - 1, ell, 0.0, d_hp_cur);
                                                            let v_keep = lookup_v(&out.v_star, ell, h_r as i32 - 1, 1, k1, k2, 0.0);
                                                            p_phz * v_rst + (1.0 - p_phz) * v_keep
                                                        }
                                                        ACT_R_BOOST1 | ACT_R_BOOST2 => {
                                                            let (k1_n, k2_n, hc, post_eva) = if a_r == ACT_R_BOOST1 {
                                                                (k1 + 1, k2, ax1.hp_cost as i32, b_post1_eva)
                                                            } else {
                                                                (k1, k2 + 1, ax2.hp_cost as i32, b_post2_eva)
                                                            };
                                                            let p_phz = disrupt_hit_prob(table.ruleset, d_disrupt, post_eva, r_bp_item) * d_awake_mult;
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32 - hc, ell, 0.0, d_hp_cur);
                                                            let v_keep = lookup_v(&out.v_star, ell, h_r as i32 - hc, sigma, k1_n, k2_n, 0.0);
                                                            p_phz * v_rst + (1.0 - p_phz) * v_keep
                                                        }
                                                        ACT_R_PASS => {
                                                            let rc_slot = recv_arr[out.best_receiver_canon[recv_idx(ell, sigma, k1, k2, k1_len, k2_len)] as usize].party_slot as usize;
                                                            let rc_idx = side * 3 + rc_slot;
                                                            let rc_bp = table.item_spec(cb.mons[rc_idx].item).is_brightpowder;
                                                            let p_phz = disrupt_hit_prob(table.ruleset, d_disrupt, boosts_r[6], rc_bp) * d_awake_mult;
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32, ell, 0.0, d_hp_cur);
                                                            let v_pass = RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, sigma, k1, k2, k1_len, k2_len)];
                                                            p_phz * v_rst + (1.0 - p_phz) * v_pass
                                                        }
                                                        _ => 0.5,
                                                    }
                                                }
                                                DisruptKind::Encore { .. } => {
                                                    let p_enc = disrupt_hit_prob(table.ruleset, d_disrupt, boosts_r[6], r_bp_item) * d_awake_mult;
                                                    let v_rst_cur = RELAY_GAMMA * v_reset_fn(h_r as i32, ell, 0.0, d_hp_cur);
                                                    let v_r_first = match a_r {
                                                        ACT_R_PASS => {
                                                            RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, sigma, k1, k2, k1_len, k2_len)]
                                                        }
                                                        ACT_R_SUB => {
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32 - 1, ell, 0.0, d_hp_cur);
                                                            let v_keep = lookup_v(&out.v_star, ell, h_r as i32 - 1, 1, k1, k2, 0.0);
                                                            p_enc * v_rst + (1.0 - p_enc) * v_keep
                                                        }
                                                        ACT_R_BOOST1 | ACT_R_BOOST2 => {
                                                            let (k1_n, k2_n, hc, post_eva) = if a_r == ACT_R_BOOST1 {
                                                                (k1 + 1, k2, ax1.hp_cost as i32, b_post1_eva)
                                                            } else {
                                                                (k1, k2 + 1, ax2.hp_cost as i32, b_post2_eva)
                                                            };
                                                            let p_enc_post = disrupt_hit_prob(table.ruleset, d_disrupt, post_eva, r_bp_item) * d_awake_mult;
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32 - hc, ell, 0.0, d_hp_cur);
                                                            let v_keep = lookup_v(&out.v_star, ell, h_r as i32 - hc, sigma, k1_n, k2_n, 0.0);
                                                            p_enc_post * v_rst + (1.0 - p_enc_post) * v_keep
                                                        }
                                                        _ => {
                                                            let v_rst = RELAY_GAMMA * v_reset_fn(h_r as i32, ell, ev_r_to_d, d_hp_cur);
                                                            p_r_ko_d * v_dir_here + (1.0 - p_r_ko_d) * (p_enc * v_rst + (1.0 - p_enc) * v_dir_here)
                                                        }
                                                    };
                                                    let v_unhindered = match a_r {
                                                        ACT_R_PASS => RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, sigma, k1, k2, k1_len, k2_len)],
                                                        ACT_R_SUB => lookup_v(&out.v_star, ell, h_r as i32 - 1, 1, k1, k2, 0.0),
                                                        ACT_R_BOOST1 => lookup_v(&out.v_star, ell, h_r as i32 - ax1.hp_cost as i32, sigma, k1 + 1, k2, 0.0),
                                                        ACT_R_BOOST2 => lookup_v(&out.v_star, ell, h_r as i32 - ax2.hp_cost as i32, sigma, k1, k2 + 1, 0.0),
                                                        _ => v_dir_here,
                                                    };
                                                    let v_d_first = p_enc * v_rst_cur + (1.0 - p_enc) * v_unhindered;
                                                    let pf = if a_r == ACT_R_ATTACK && prio_r_atk != 0 {
                                                        if prio_r_atk > 0 { 1.0 } else { 0.0 }
                                                    } else {
                                                        p_first_neutral
                                                    };
                                                    pf * v_r_first + (1.0 - pf) * v_d_first
                                                }
                                                DisruptKind::Haze => {
                                                    let v_r_first = match a_r {
                                                        ACT_R_PASS => {
                                                            RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, sigma, 0, 0, k1_len, k2_len)]
                                                        }
                                                        ACT_R_SUB => lookup_v(&out.v_star, ell, h_r as i32 - 1, 1, 0, 0, 0.0),
                                                        ACT_R_BOOST1 => lookup_v(&out.v_star, ell, h_r as i32 - ax1.hp_cost as i32, sigma, 0, 0, 0.0),
                                                        ACT_R_BOOST2 => lookup_v(&out.v_star, ell, h_r as i32 - ax2.hp_cost as i32, sigma, 0, 0, 0.0),
                                                        _ => {
                                                            let v_hazed = v_direct_tbl[state_idx(ell, h_r, sigma, 0, 0, k1_len, k2_len)];
                                                            p_r_ko_d * v_dir_here + (1.0 - p_r_ko_d) * v_hazed
                                                        }
                                                    };
                                                    let v_d_first = match a_r {
                                                        ACT_R_PASS => RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, sigma, 0, 0, k1_len, k2_len)],
                                                        ACT_R_SUB => lookup_v(&out.v_star, ell, h_r as i32 - 1, 1, 0, 0, 0.0),
                                                        ACT_R_BOOST1 => lookup_v(&out.v_star, ell, h_r as i32 - ax1.hp_cost as i32, sigma, 1.min(k1_len - 1), 0, 0.0),
                                                        ACT_R_BOOST2 => lookup_v(&out.v_star, ell, h_r as i32 - ax2.hp_cost as i32, sigma, 0, 1.min(k2_len - 1), 0.0),
                                                        _ => v_direct_tbl[state_idx(ell, h_r, sigma, 0, 0, k1_len, k2_len)],
                                                    };
                                                    p_first_neutral * v_r_first + (1.0 - p_first_neutral) * v_d_first
                                                }
                                                DisruptKind::PerishSong => {
                                                    match a_r {
                                                        ACT_R_ATTACK => {
                                                            let v_ps = RELAY_GAMMA * v_reset_fn(h_r as i32, ell, ev_r_to_d, d_hp_cur);
                                                            p_first_neutral * p_r_ko_d * v_dir_here + (1.0 - p_first_neutral * p_r_ko_d) * v_ps
                                                        }
                                                        ACT_R_SUB => RELAY_GAMMA * v_reset_fn(h_r as i32 - 1, ell, 0.0, d_hp_cur),
                                                        _ => RELAY_GAMMA * v_reset_fn(h_r as i32, ell, 0.0, d_hp_cur),
                                                    }
                                                }
                                                DisruptKind::None => 0.5,
                                            },
                                            _ => {
                                                // ACT_D_BREAK
                                                match a_r {
                                                    ACT_R_ATTACK => v_dir_here,
                                                    ACT_R_SUB => {
                                                        let v_r_first = apply_brk_to_state(&out.v_star, ell, h_r as i32 - 1, 1, k1, k2, 0.0);
                                                        let v_d_first = {
                                                            let v_miss = lookup_v(&out.v_star, ell, h_r as i32 - 1, 1, k1, k2, 0.0);
                                                            if acc_d <= 0.0 || dmg_d_if_hit <= 0.0 {
                                                                v_miss
                                                            } else {
                                                                let z = 4.0 * dmg_d_if_hit;
                                                                let dh_lo = (z.floor() as i32).min(4);
                                                                let dh_hi = (dh_lo + 1).min(4);
                                                                let p_hi = (z - (dh_lo as f32)).clamp(0.0, 1.0);
                                                                let post_sub = |h_after: i32| -> f32 {
                                                                    if h_after <= 0 {
                                                                        lookup_v(&out.v_star, ell, 0, 0, k1, k2, 0.0)
                                                                    } else if h_after >= 2 {
                                                                        lookup_v(&out.v_star, ell, h_after - 1, 1, k1, k2, 0.0)
                                                                    } else {
                                                                        lookup_v(&out.v_star, ell, 1, 0, k1, k2, 0.0)
                                                                    }
                                                                };
                                                                let v_hit = (1.0 - p_hi) * post_sub(h_r as i32 - dh_lo)
                                                                    + p_hi * post_sub(h_r as i32 - dh_hi);
                                                                (1.0 - acc_d) * v_miss + acc_d * v_hit
                                                            }
                                                        };
                                                        p_status_first_vs_brk * v_r_first + (1.0 - p_status_first_vs_brk) * v_d_first
                                                    }
                                                    ACT_R_BOOST1 | ACT_R_BOOST2 => {
                                                        let (k1_n, k2_n, hc, dmg_d_post, acc_d_post, dmg_d_post_s, acc_d_post_s) = if a_r == ACT_R_BOOST1 {
                                                            (k1 + 1, k2, ax1.hp_cost as i32, dmg_d_post1, acc_d_post1, dmg_d_post1_sub, acc_d_post1_sub)
                                                        } else {
                                                            (k1, k2 + 1, ax2.hp_cost as i32, dmg_d_post2, acc_d_post2, dmg_d_post2_sub, acc_d_post2_sub)
                                                        };
                                                        let v_r_first = {
                                                            let h_cur = h_r as i32 - hc;
                                                            let v_miss = lookup_v(&out.v_star, ell, h_cur, sigma, k1_n, k2_n, 0.0);
                                                            if sigma == 1 {
                                                                if acc_d_post_s <= 0.0 || dmg_d_post_s <= 0.0 {
                                                                    v_miss
                                                                } else {
                                                                    let p_b = (4.0 * dmg_d_post_s).min(1.0);
                                                                    let v0 = lookup_v(&out.v_star, ell, h_cur, 0, k1_n, k2_n, 0.0);
                                                                    let v1 = lookup_v(&out.v_star, ell, h_cur, 1, k1_n, k2_n, 0.0);
                                                                    (1.0 - acc_d_post_s) * v_miss + acc_d_post_s * (p_b * v0 + (1.0 - p_b) * v1)
                                                                }
                                                            } else if acc_d_post <= 0.0 || dmg_d_post <= 0.0 {
                                                                v_miss
                                                            } else {
                                                                let z = 4.0 * dmg_d_post;
                                                                let dh_lo = (z.floor() as i32).min(4);
                                                                let dh_hi = (dh_lo + 1).min(4);
                                                                let p_hi = (z - (dh_lo as f32)).clamp(0.0, 1.0);
                                                                let vlo = lookup_v(&out.v_star, ell, h_cur - dh_lo, 0, k1_n, k2_n, 0.0);
                                                                let vhi = lookup_v(&out.v_star, ell, h_cur - dh_hi, 0, k1_n, k2_n, 0.0);
                                                                (1.0 - acc_d_post) * v_miss + acc_d_post * ((1.0 - p_hi) * vlo + p_hi * vhi)
                                                            }
                                                        };
                                                        let v_d_first = {
                                                            let v_miss = lookup_v(&out.v_star, ell, h_r as i32 - hc, sigma, k1_n, k2_n, 0.0);
                                                            if sigma == 1 {
                                                                if acc_d_sub <= 0.0 || dmg_d_sub_if_hit <= 0.0 {
                                                                    v_miss
                                                                } else {
                                                                    let p_b = (4.0 * dmg_d_sub_if_hit).min(1.0);
                                                                    let v0 = lookup_v(&out.v_star, ell, h_r as i32 - hc, 0, k1_n, k2_n, 0.0);
                                                                    let v1 = lookup_v(&out.v_star, ell, h_r as i32 - hc, 1, k1_n, k2_n, 0.0);
                                                                    (1.0 - acc_d_sub) * v_miss + acc_d_sub * (p_b * v0 + (1.0 - p_b) * v1)
                                                                }
                                                            } else if acc_d <= 0.0 || dmg_d_if_hit <= 0.0 {
                                                                v_miss
                                                            } else {
                                                                let z = 4.0 * dmg_d_if_hit;
                                                                let dh_lo = (z.floor() as i32).min(4);
                                                                let dh_hi = (dh_lo + 1).min(4);
                                                                let p_hi = (z - (dh_lo as f32)).clamp(0.0, 1.0);
                                                                let post_b = |h_after: i32| -> f32 {
                                                                    if h_after <= 0 {
                                                                        lookup_v(&out.v_star, ell, 0, 0, k1, k2, 0.0)
                                                                    } else if h_after > hc {
                                                                        lookup_v(&out.v_star, ell, h_after - hc, 0, k1_n, k2_n, 0.0)
                                                                    } else {
                                                                        lookup_v(&out.v_star, ell, h_after, 0, k1, k2, 0.0)
                                                                    }
                                                                };
                                                                let v_hit = (1.0 - p_hi) * post_b(h_r as i32 - dh_lo)
                                                                    + p_hi * post_b(h_r as i32 - dh_hi);
                                                                (1.0 - acc_d) * v_miss + acc_d * v_hit
                                                            }
                                                        };
                                                        p_status_first_vs_brk * v_r_first + (1.0 - p_status_first_vs_brk) * v_d_first
                                                    }
                                                    ACT_R_PASS => {
                                                        let v_r_first = RELAY_GAMMA * w_pass_in_atk[recv_idx(ell, sigma, k1, k2, k1_len, k2_len)];
                                                        let v_clean = RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, sigma, k1, k2, k1_len, k2_len)];
                                                        let v_d_first = if sigma == 1 {
                                                            if acc_d_sub <= 0.0 || dmg_d_sub_if_hit <= 0.0 {
                                                                v_clean
                                                            } else {
                                                                let p_b = (4.0 * dmg_d_sub_if_hit).min(1.0);
                                                                let v_sub0 = RELAY_GAMMA * w_pass_no_atk[recv_idx(ell, 0, k1, k2, k1_len, k2_len)];
                                                                (1.0 - acc_d_sub) * v_clean + acc_d_sub * (p_b * v_sub0 + (1.0 - p_b) * v_clean)
                                                            }
                                                        } else if acc_d <= 0.0 || dmg_d_if_hit <= 0.0 {
                                                            v_clean
                                                        } else {
                                                            let z = 4.0 * dmg_d_if_hit;
                                                            let dh_lo = (z.floor() as i32).min(4);
                                                            let dh_hi = (dh_lo + 1).min(4);
                                                            let p_hi = (z - (dh_lo as f32)).clamp(0.0, 1.0);
                                                            let v_fnt = RELAY_GAMMA * v_reset_fn(0, ell, 0.0, d_hp_cur);
                                                            let vlo = if h_r as i32 - dh_lo <= 0 { v_fnt } else { v_clean };
                                                            let vhi = if h_r as i32 - dh_hi <= 0 { v_fnt } else { v_clean };
                                                            (1.0 - acc_d) * v_clean + acc_d * ((1.0 - p_hi) * vlo + p_hi * vhi)
                                                        };
                                                        p_status_first_vs_brk * v_r_first + (1.0 - p_status_first_vs_brk) * v_d_first
                                                    }
                                                    _ => 0.5,
                                                }
                                            }
                                        };
                                        mat[ri][dj] = cell;

                                        // Compute matching Attack-only no-relay cell for ri == 0 (ACT_R_ATTACK)
                                        if ri == 0 {
                                            let nr_cell = match a_d {
                                                ACT_D_PIVOT => {
                                                    let base1 = out.v_no_relay[state_idx(1, h_r, sigma, k1, k2, k1_len, k2_len)];
                                                    let ko_c = if chip_c_here >= piv_c_hp { 0.18 } else { 0.0 };
                                                    (RELAY_GAMMA * base1 + 0.18 * chip_c_here.min(piv_c_hp) + ko_c).clamp(0.01, 0.98)
                                                }
                                                ACT_D_DISRUPT => {
                                                    RELAY_GAMMA * v_reset_fn(h_r as i32, ell, ev_r_to_d, d_hp_cur)
                                                }
                                                _ => v_dir_here,
                                            };
                                            mat_norelay[0][dj] = nr_cell;
                                        }
                                    }
                                }

                                let (v_eq, p_sub, q_sub) = solve_zero_sum_small(&mat, m, n);
                                let (v_nr, _, _) = solve_zero_sum_small(&mat_norelay, 1, n);

                                // Expected Attack Q-value under D's equilibrium response q*
                                let mut q_atk = 0.0f32;
                                for dj in 0..n {
                                    q_atk += mat[0][dj] * q_sub[dj];
                                }

                                let mut full_p = [0.0f32; MAX_R_ACTIONS];
                                for (ri, &a_r) in r_acts[..m].iter().enumerate() {
                                    full_p[a_r] = p_sub[ri];
                                }
                                let mut full_q = [0.0f32; MAX_D_ACTIONS];
                                for (dj, &a_d) in d_acts[..n].iter().enumerate() {
                                    full_q[a_d] = q_sub[dj];
                                }

                                out.v_star[cur_idx] = v_eq;
                                out.v_no_relay[cur_idx] = v_nr;
                                out.q_attack[cur_idx] = q_atk;
                                out.pi_r[cur_idx] = quantize_policy_u8(&full_p);
                                out.pi_d[cur_idx] = quantize_policy_u8(&full_q);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn with_solved_relay<R, F: FnOnce(&SolvedRelaySubgame, &ActiveRelayLookup) -> R>(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    f: F,
) -> Option<R> {
    let (probe, recv_arr, def_arr) = probe_fast_relay(cb, table, side)?;
    let set_idx = (probe.key.key_lo as usize) & (RELAY_CACHE_SETS - 1);
    let act_idx = side * 3 + (probe.act_slot as usize);
    let act_boosts = &cb.mons[act_idx].boosts;

    let mut f_opt = Some(f);
    if let Some(res) = RELAY_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let set = &mut cache.sets[set_idx];
        for way in 0..RELAY_CACHE_WAYS {
            if set.keys[way] == probe.key {
                let slot = set.slots[way] as usize;
                if way > 0 {
                    set.keys.swap(0, way);
                    set.slots.swap(0, way);
                }
                let solved = &cache.pool[slot];
                let k1_0 = map_current_boost_to_k(act_boosts, &solved.axis1);
                let k2_0 = map_current_boost_to_k(act_boosts, &solved.axis2);
                let k1_len = solved.k1_len as usize;
                let k2_len = solved.k2_len as usize;
                let h_r0 = probe.h_r0 as usize;
                let sigma0 = probe.sigma0 as usize;
                let lookup = ActiveRelayLookup {
                    act_slot: probe.act_slot,
                    bp_move_slot: solved.r_move_slots[ACT_R_PASS],
                    sub_move_slot: solved.r_move_slots[ACT_R_SUB],
                    atk_move_slot: solved.r_move_slots[ACT_R_ATTACK],
                    axis1: solved.axis1,
                    axis2: solved.axis2,
                    x0: state_idx(0, h_r0, sigma0, k1_0, k2_0, k1_len, k2_len),
                    r0: recv_idx(0, sigma0, k1_0, k2_0, k1_len, k2_len),
                    recv_party_slots: probe.recv_party_slots,
                    n_recv: probe.n_recv,
                    n_def_bench: probe.n_def_bench,
                    def_break_slot: [solved.d0_brk_slot, NO_MOVE_SLOT],
                    def_disrupt_slot: [solved.d0_dis_slot, NO_MOVE_SLOT],
                    key: probe.key,
                };
                return Some((f_opt.take().unwrap())(solved, &lookup));
            }
        }
        None
    }) {
        return Some(res);
    }

    let (lookup, d0_atk_frac_u8) = build_cold_lookup(cb, table, side, &probe)?;
    let f = f_opt.unwrap();
    let res = RELAY_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        let evict_empty = cache.sets[set_idx].keys[RELAY_CACHE_WAYS - 1] == RelayCacheKey::default();
        let slot_idx = if evict_empty && cache.pool.len() < MAX_RELAY_POOL {
            let idx = cache.pool.len();
            cache.pool.push(SolvedRelaySubgame::empty());
            idx
        } else {
            cache.sets[set_idx].slots[RELAY_CACHE_WAYS - 1] as usize
        };
        {
            let set = &mut cache.sets[set_idx];
            for w_i in (1..RELAY_CACHE_WAYS).rev() {
                set.keys[w_i] = set.keys[w_i - 1];
                set.slots[w_i] = set.slots[w_i - 1];
            }
            set.keys[0] = lookup.key;
            set.slots[0] = slot_idx as u16;
        }
        solve_relay_subgame_into(
            cb,
            table,
            w,
            side,
            &lookup,
            &recv_arr,
            &def_arr,
            &mut cache.pool[slot_idx],
        );
        cache.pool[slot_idx].d0_atk_frac_u8 = d0_atk_frac_u8;
        let solved = &cache.pool[slot_idx];
        f(solved, &lookup)
    });
    Some(res)
}

/// Post-pass Bellman continuity for an already-landed Baton Pass receiver (`!has_bp && team_has_bp`).
/// Uses canonical quantized party scores for `base_logit_anchor` so side-swap antisymmetry holds to `< 1e-12`.
fn landed_receiver_surplus(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    v_base_side: f64,
) -> f64 {
    let side_st = &cb.sides[side];
    if side_st.request == 2 {
        return 0.0;
    }
    let act_slot = side_st.active_idx as usize;
    if act_slot >= 3 || (side_st.alive_mask & (1 << act_slot)) == 0 {
        return 0.0;
    }
    let act_idx = side * 3 + act_slot;
    let act_mon = &cb.mons[act_idx];
    if act_mon.hp() <= 0
        || (act_mon.vol_flags & (vol_bit::PERISH_SONG | vol_bit::CURSE | vol_bit::ENCORE)) != 0
    {
        return 0.0;
    }
    if (table.mons[act_idx].role_flags & nc2000_engine::compact::role_bit::BATON_PASS) != 0 {
        return 0.0;
    }
    let has_sub = (act_mon.vol_flags & vol_bit::SUBSTITUTE) != 0;
    // Verify at least one teammate actually carries Setup/Substitute + Baton Pass (not pure Mean Look + BP)
    let mut team_has_setup_bp = false;
    for s_i in 0..3usize {
        if s_i == act_slot {
            continue;
        }
        let t_idx = side * 3 + s_i;
        if (table.mons[t_idx].role_flags & nc2000_engine::compact::role_bit::BATON_PASS) != 0 {
            let has_passable = (0..table.mons[t_idx].num_moves as usize).any(|mi| {
                let ms = &table.move_specs[t_idx][mi];
                ms.is_substitute || classify_self_boost_move(table, t_idx, mi).is_some()
            });
            if has_passable {
                team_has_setup_bp = true;
                break;
            }
        }
    }
    if !team_has_setup_bp {
        return 0.0;
    }

    // Exclude negative-Speed self-Curse walls (`boosts[4] < 0` without Substitute or Evasion) while
    // preserving `+2 Atk` / `+2 SpA` / `+Spe` / `+Eva` / `Substitute` receivers (including SD Scizor -> SD Marowak).
    let has_relay_boost = act_mon.boosts[4] > 0
        || act_mon.boosts[6] > 0
        || ((act_mon.boosts[0] >= 2 || act_mon.boosts[2] >= 2) && act_mon.boosts[4] >= 0);
    if !has_sub && !has_relay_boost {
        return 0.0;
    }

    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    if def_slot >= 3
        || (cb.sides[def_side].alive_mask & (1 << def_slot)) == 0
        || cb.mons[def_side * 3 + def_slot].hp() <= 0
    {
        return 0.0;
    }

    let scale = w.scale as f32;
    let w_hp = w.hp as f32;
    let w_alive = w.alive as f32;
    let flying_mask = 1u32 << (table.flying_type & 31);

    let side_q_score = |s: usize| -> f32 {
        let st = &cb.sides[s];
        let a_sl = st.active_idx as usize;
        let mut sc = 0.0f32;
        let mut gr_bench = 0u32;
        for sl in 0..3usize {
            if (st.alive_mask & (1 << sl)) == 0 {
                continue;
            }
            let idx = s * 3 + sl;
            if cb.mons[idx].hp() <= 0 {
                continue;
            }
            if sl != a_sl && (table.mons[idx].type_mask & flying_mask) == 0 {
                gr_bench += 1;
            }
            let b8 = hp_to_b8(cb.mons[idx].hp(), table.mons[idx].maxhp);
            let (_, slp_r, _) = mon_guaranteed_incapacitated(cb, table, s, sl);
            sc += w_alive
                + w_hp * ((b8 as f32) / 8.0)
                - mon_status_pen(w, table.ruleset, cb.mons[idx].status_raw(), slp_r);
        }
        let layers = (st.side_flags & 0x03) as usize;
        if layers > 0 && gr_bench > 0 {
            const AMOUNTS: [f32; 4] = [0.0, 3.0, 4.0, 6.0];
            sc -= (w.spikes as f32) * (AMOUNTS[layers] / 24.0) * (gr_bench as f32);
        }
        sc
    };
    let anchor = scale * (side_q_score(side) - side_q_score(def_side));

    let mut foe_slots = [0u8; 3];
    let mut foe_hps = [0.0f32; 3];
    let mut n_foes = 0usize;
    let d0_idx = def_side * 3 + def_slot;
    foe_slots[0] = def_slot as u8;
    foe_hps[0] = (hp_to_b8(cb.mons[d0_idx].hp(), table.mons[d0_idx].maxhp) as f32) / 8.0;
    n_foes += 1;

    let mut rem_foes = [(0u64, 0u8, 0.0f32); 2];
    let mut n_rem = 0usize;
    for s_i in 0..3usize {
        if s_i == def_slot || (cb.sides[def_side].alive_mask & (1 << s_i)) == 0 {
            continue;
        }
        let idx = def_side * 3 + s_i;
        if cb.mons[idx].hp() > 0 {
            rem_foes[n_rem] = (
                table.mons[idx].static_fp,
                s_i as u8,
                (hp_to_b8(cb.mons[idx].hp(), table.mons[idx].maxhp) as f32) / 8.0,
            );
            n_rem += 1;
        }
    }
    if n_rem == 2 && rem_foes[1].0 < rem_foes[0].0 {
        rem_foes.swap(0, 1);
    }
    for i in 0..n_rem {
        foe_slots[n_foes] = rem_foes[i].1;
        foe_hps[n_foes] = rem_foes[i].2;
        n_foes += 1;
    }

    let hp_frac = (hp_to_b8(act_mon.hp(), table.mons[act_idx].maxhp) as f32) / 8.0;
    let (_, slp_rem, _) = mon_guaranteed_incapacitated(cb, table, side, act_slot);
    let v_sweep = evaluate_single_receiver_sweep(
        cb,
        table,
        w,
        side,
        act_slot,
        hp_frac,
        slp_rem,
        act_mon.status_raw(),
        has_sub,
        &act_mon.boosts,
        def_side,
        &foe_slots[..n_foes],
        &foe_hps[..n_foes],
        false,
        anchor,
    ) as f64;
    (v_sweep - v_base_side).max(0.0)
}

#[inline]
pub fn relay_surplus_for_side(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    v_base_side: f64,
) -> f64 {
    let base_i = side * 3;
    if ((table.mons[base_i].role_flags
        | table.mons[base_i + 1].role_flags
        | table.mons[base_i + 2].role_flags)
        & nc2000_engine::compact::role_bit::BATON_PASS)
        == 0
    {
        return 0.0;
    }
    if let Some(delta) = with_solved_relay(cb, table, w, side, |solved, lookup| {
        let v_star = solved.v_star[lookup.x0] as f64;
        let v_nr = solved.v_no_relay[lookup.x0] as f64;
        (v_star - v_nr.max(v_base_side)).max(0.0)
    }) {
        return delta;
    }
    landed_receiver_surplus(cb, table, w, side, v_base_side)
}

/// Strictly antisymmetric leaf lift (§5.2).
/// Identically returns `v_base_p0` (`0.0 ULP` change) on all `1v1` states (`alive_0 <= 1 && alive_1 <= 1`).
#[inline]
pub fn apply_relay_leaf_lift(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    v_base_p0: f64,
) -> f64 {
    let alive_0 = cb.sides[0].alive_mask.count_ones();
    let alive_1 = cb.sides[1].alive_mask.count_ones();
    if alive_0 <= 1 && alive_1 <= 1 {
        return v_base_p0;
    }

    let delta_0 = relay_surplus_for_side(cb, table, w, 0, v_base_p0);
    let delta_1 = relay_surplus_for_side(cb, table, w, 1, 1.0 - v_base_p0);

    if delta_0 == 0.0 && delta_1 == 0.0 {
        return v_base_p0;
    }
    (v_base_p0 + delta_0 - delta_1).clamp(0.001, 0.999)
}

/// Outcome of querying the Bellman equilibrium rollout gate for `side` (§5.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelayRolloutDecision {
    /// Gate did not open and side is not a gated Baton Passer: proceed with standard rollout.
    NotApplicable,
    /// Side IS an eligible Baton Passer, but `V*(x) - Q*(x, Attack) <= 0.04`:
    /// proceed with standard rollout while suppressing legacy `0.42` premature Baton Pass.
    GateClosedSuppressLegacyBp,
    /// Equilibrium gate opened: execute this compact action code (`0..4`, `4..7`, or `16..` directed BP).
    Chosen(u8),
}

#[inline]
fn sample_or_argmax<const N: usize>(
    weights: &[u8; N],
    allowed: &[bool; N],
    deterministic: bool,
    rng: &mut SplitMix64,
) -> Option<usize> {
    let mut sum = 0u32;
    let mut best_i = None;
    let mut best_w = 0u8;
    for i in 0..N {
        if allowed[i] && weights[i] > 0 {
            sum += weights[i] as u32;
            if weights[i] > best_w {
                best_w = weights[i];
                best_i = Some(i);
            }
        }
    }
    if sum == 0 {
        return None;
    }
    if deterministic {
        return best_i;
    }
    let mut draw = rng.below(sum as usize) as u32;
    for i in 0..N {
        if allowed[i] && weights[i] > 0 {
            let w_i = weights[i] as u32;
            if draw < w_i {
                return Some(i);
            }
            draw -= w_i;
        }
    }
    best_i
}

/// Select the optimal Baton Pass receiver slot (`4 + party_slot`) when `cb.sides[side].request == 2`
/// mid-turn on full `Battle` (`playout_pick`).
pub fn best_midturn_bp_switch(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    cs: &[u8],
) -> Option<u8> {
    let att_slot = cb.sides[side].active_idx as usize;
    if att_slot >= 3 {
        return None;
    }
    let att_idx = side * 3 + att_slot;
    let att_mon = &cb.mons[att_idx];
    let has_sub = (att_mon.vol_flags & vol_bit::SUBSTITUTE) != 0;
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    if def_slot >= 3 || (cb.sides[def_side].alive_mask & (1 << def_slot)) == 0 {
        return None;
    }

    let mut foe_slots = [0u8; 3];
    let mut foe_hps = [0.0f32; 3];
    let mut n_foes = 0usize;
    let d0_idx = def_side * 3 + def_slot;
    foe_slots[0] = def_slot as u8;
    foe_hps[0] = (hp_to_b8(cb.mons[d0_idx].hp(), table.mons[d0_idx].maxhp) as f32) / 8.0;
    n_foes += 1;
    for s_i in 0..3usize {
        if s_i != def_slot
            && (cb.sides[def_side].alive_mask & (1 << s_i)) != 0
            && cb.mons[def_side * 3 + s_i].hp() > 0
        {
            let idx = def_side * 3 + s_i;
            foe_slots[n_foes] = s_i as u8;
            foe_hps[n_foes] = (hp_to_b8(cb.mons[idx].hp(), table.mons[idx].maxhp) as f32) / 8.0;
            n_foes += 1;
        }
    }

    let incoming_atk_from_lead = (cb.bp_switch_flags & (1u8 << (2 + side))) != 0;
    let mut best_act = None;
    let mut best_val = -1.0f32;
    for &c in cs {
        if (4..7).contains(&c) {
            let b_slot = (c - 4) as usize;
            let b_idx = side * 3 + b_slot;
            let maxhp = table.mons[b_idx].maxhp.max(1) as f32;
            let entry =
                (switch_entry_dmg(cb, table, side, b_slot, Some(att_slot)) as f32) / maxhp;
            let hp_entry = ((cb.mons[b_idx].hp() as f32) / maxhp - entry).max(0.0);
            let (_, slp_rem, _) = mon_guaranteed_incapacitated(cb, table, side, b_slot);
            let val = evaluate_single_receiver_sweep(
                cb,
                table,
                w,
                side,
                b_slot,
                hp_entry,
                slp_rem,
                cb.mons[b_idx].status_raw(),
                has_sub,
                &att_mon.boosts,
                def_side,
                &foe_slots[..n_foes],
                &foe_hps[..n_foes],
                incoming_atk_from_lead,
                0.0,
            );
            if val > best_val {
                best_val = val;
                best_act = Some(c);
            }
        }
    }
    best_act
}

/// Evaluate the Bellman equilibrium gate for `side` at rollout step `turn_offset` (§5.3).
#[allow(clippy::too_many_arguments)]
pub fn rollout_relay_pick(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    side: usize,
    cs: &[u8],
    turn_offset: u8,
    pivot_token: &mut [u8; 2],
    deterministic: bool,
    rng: &mut SplitMix64,
) -> RelayRolloutDecision {
    if cb.sides[side].request != 1 {
        return RelayRolloutDecision::NotApplicable;
    }
    let att_slot = cb.sides[side].active_idx as usize;
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    if att_slot >= 3
        || def_slot >= 3
        || (cb.sides[side].alive_mask & (1 << att_slot)) == 0
        || (cb.sides[def_side].alive_mask & (1 << def_slot)) == 0
    {
        return RelayRolloutDecision::NotApplicable;
    }

    let att_idx = side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    // Fast-path: if neither active Pokemon carries Baton Pass, neither relay gate can apply.
    if ((table.mons[att_idx].role_flags | table.mons[def_idx].role_flags)
        & nc2000_engine::compact::role_bit::BATON_PASS)
        == 0
    {
        return RelayRolloutDecision::NotApplicable;
    }

    let immediate_ko_pick = || -> Option<u8> {
        let faster_by_speed = crate::eval::compare_action_speed(
            cb, table, side, att_slot, def_side, def_slot,
        ) == Ordering::Greater;
        let foe_last = cb.sides[def_side].alive_mask.count_ones() <= 1;
        for &c in cs {
            if c < 4 {
                let ms = &table.move_specs[att_idx][c as usize];
                if ms.is_selfdestruct && cb.sides[side].alive_mask.count_ones() <= 1 {
                    continue;
                }
                let (hit_frac, acc) = cb.expected_hit_fraction_with_acc(
                    table,
                    side,
                    att_slot,
                    def_side,
                    def_slot,
                    c as usize,
                );
                let faster = ms.priority > 0 || (ms.priority == 0 && faster_by_speed);
                if acc >= 0.995 && hit_frac >= 1.05 && (faster || foe_last) {
                    return Some(c);
                }
            }
        }
        None
    };

    let _ = (turn_offset, pivot_token);

    // 2. Own side is relay-eligible (R = side):
    //    Tree search initiates the first setup/Substitute action; rollout continuation activates
    //    once an active Substitute or positive boost is on the field (`me_has_active_setup`),
    //    preventing uninitiated rollout status over-firing while completing multi-turn chains!
    if (table.mons[att_idx].role_flags & nc2000_engine::compact::role_bit::BATON_PASS) != 0 {
        let me_mon = &cb.mons[att_idx];
        let me_has_active_setup = (me_mon.vol_flags & vol_bit::SUBSTITUTE) != 0
            || me_mon.boosts[0] > 0
            || me_mon.boosts[1] > 0
            || me_mon.boosts[2] > 0
            || me_mon.boosts[3] > 0
            || me_mon.boosts[4] > 0
            || me_mon.boosts[6] > 0;
        if !me_has_active_setup {
            return RelayRolloutDecision::GateClosedSuppressLegacyBp;
        }
        if let Some(own_res) = with_solved_relay(cb, table, w, side, |solved, lookup| {
            // 1. Immediate guaranteed faster KO priority (§5.3 #1)
            if let Some(ko_c) = immediate_ko_pick() {
                return RelayRolloutDecision::Chosen(ko_c);
            }
            let v_star = solved.v_star[lookup.x0];
            let v_nr = solved.v_no_relay[lookup.x0];
            let non_atk_mass = 255u16.saturating_sub(solved.pi_r[lookup.x0][ACT_R_ATTACK] as u16);
            if v_star - v_nr <= RELAY_COMMIT_MARGIN || non_atk_mass < 26 {
                return RelayRolloutDecision::GateClosedSuppressLegacyBp;
            }
            let has_code = |code: u8| cs.contains(&code);
            let map_act = solved.r_move_slots;
            let allowed: [bool; MAX_R_ACTIONS] = [
                map_act[0] < 4 && has_code(map_act[0]),
                map_act[1] < 4 && has_code(map_act[1]),
                map_act[2] < 4 && has_code(map_act[2]),
                map_act[3] < 4 && has_code(map_act[3]),
                map_act[4] < 4 && has_code(map_act[4]),
            ];
            if let Some(a_i) = sample_or_argmax(&solved.pi_r[lookup.x0], &allowed, deterministic, rng) {
                let mv = map_act[a_i];
                if a_i == ACT_R_PASS {
                    let rc_canon = solved.best_receiver_canon[lookup.r0] as usize;
                    let recv_slot = lookup.recv_party_slots
                        [rc_canon.min((lookup.n_recv as usize).saturating_sub(1))];
                    return RelayRolloutDecision::Chosen(((recv_slot + 1) << 4) | (mv & 3));
                }
                return RelayRolloutDecision::Chosen(mv);
            }
            RelayRolloutDecision::GateClosedSuppressLegacyBp
        }) {
            return own_res;
        }
    }

    // 3. Opponent holds active setup/Substitute on a Baton Passer: activate on-field Disrupt
    //    (`Roar`/`Whirlwind`/`Haze`/`Encore`) at Bellman minimax rate without forcing switches.
    if (table.mons[def_idx].role_flags & nc2000_engine::compact::role_bit::BATON_PASS) != 0 {
        let foe_mon = &cb.mons[def_idx];
        let foe_has_active_setup = (foe_mon.vol_flags & vol_bit::SUBSTITUTE) != 0
            || foe_mon.boosts[0] > 0
            || foe_mon.boosts[1] > 0
            || foe_mon.boosts[2] > 0
            || foe_mon.boosts[3] > 0
            || foe_mon.boosts[4] > 0
            || foe_mon.boosts[6] > 0;
        if foe_has_active_setup {
            let (_, own_dis_mv) = primary_disrupt_of(cb, table, side, att_slot);
            if own_dis_mv < 4 && cs.contains(&own_dis_mv) {
                if let Some(def_res) = with_solved_relay(cb, table, w, def_side, |solved, foe_lookup| {
                    let pi_d_row = &solved.pi_d[foe_lookup.x0];
                    if pi_d_row[ACT_D_DISRUPT] == 0 {
                        return None;
                    }
                    if let Some(ko_c) = immediate_ko_pick() {
                        return Some(ko_c);
                    }
                    let brk_mv = solved.d0_brk_slot;
                    let dis_mv = solved.d0_dis_slot;
                    let allowed: [bool; MAX_D_ACTIONS] = [
                        brk_mv < 4 && cs.contains(&brk_mv),
                        dis_mv < 4 && cs.contains(&dis_mv),
                        false,
                    ];
                    let picked = sample_or_argmax(pi_d_row, &allowed, deterministic, rng)?;
                    match picked {
                        ACT_D_BREAK => Some(brk_mv),
                        ACT_D_DISRUPT => Some(dis_mv),
                        _ => None,
                    }
                }) {
                    if let Some(act) = def_res {
                        return RelayRolloutDecision::Chosen(act);
                    }
                }
            }
        }
    }

    RelayRolloutDecision::NotApplicable
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use nc2000_engine::state::Battle;

    fn make_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
            "gender": "M"
        }))
        .unwrap()
    }

    #[test]
    fn closed_form_minimax_zero_duality_gap_property() {
        let mut rng = SplitMix64::new(0x1234_5678_9abc_def0);
        for case in 0..5_000usize {
            let m = 1 + rng.below(5);
            let n = 1 + rng.below(3);
            let mut mat = [[0.0f32; MAX_D_ACTIONS]; MAX_R_ACTIONS];
            for i in 0..m {
                for j in 0..n {
                    // Mix continuous random matrices and degenerate tied matrices
                    mat[i][j] = if case % 5 == 0 {
                        (rng.below(5) as f32) * 0.25
                    } else {
                        rng.next_f64() as f32
                    };
                }
            }
            let (v, p, q) = solve_zero_sum_small(&mat, m, n);
            let sum_p: f32 = p[..m].iter().sum();
            let sum_q: f32 = q[..n].iter().sum();
            assert!((sum_p - 1.0).abs() < 1e-4);
            assert!((sum_q - 1.0).abs() < 1e-4);

            let mut min_col = f32::INFINITY;
            for j in 0..n {
                let mut s = 0.0f32;
                for i in 0..m {
                    s += p[i] * mat[i][j];
                }
                min_col = min_col.min(s);
            }
            let mut max_row = f32::NEG_INFINITY;
            for i in 0..m {
                let mut s = 0.0f32;
                for j in 0..n {
                    s += mat[i][j] * q[j];
                }
                max_row = max_row.max(s);
            }
            assert!(
                (max_row - min_col).abs() < 2e-4 && (v - min_col).abs() < 2e-4,
                "duality gap at case {case} ({m}x{n}): max_row={max_row}, min_col={min_col}, v={v}, mat={mat:?}"
            );
        }
    }

    #[test]
    fn evasion_roar_monotonicity_and_encore_haze_suppression() {
        let dex = conformance::load_dex();
        let w = EvalWeights::default();
        let zap = make_set(
            "Zapdos",
            55,
            "Leftovers",
            &["Thunderbolt", "Double Team", "Substitute", "Baton Pass"],
        );
        let wak = make_set(
            "Marowak",
            50,
            "Thick Club",
            &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"],
        );
        let sui = make_set(
            "Suicune",
            50,
            "Leftovers",
            &["Surf", "Ice Beam", "Rest", "Sleep Talk"],
        );
        let rai_roar = make_set(
            "Raikou",
            55,
            "Leftovers",
            &["Thunderbolt", "Roar", "Rest", "Sleep Talk"],
        );
        let lax = make_set(
            "Snorlax",
            50,
            "Leftovers",
            &["Body Slam", "Rest", "Curse", "Sleep Talk"],
        );

        let mut b = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[zap.clone(), wak.clone(), sui.clone()],
            &[rai_roar, lax.clone(), sui.clone()],
        )
        .unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();
        let (table, cb) = CompactBattle::from_battle(&b, &dex);

        // Check monotonicity of V*(x) over k_eva in 0..=3 against active Roar Raikou
        let vals: [f32; 4] = with_solved_relay(&cb, &table, &w, 0, |solved, lookup| {
            assert_eq!(lookup.axis1.kind, BoostAxisKind::Evasion);
            let k2_len = solved.k2_len as usize;
            let k1_len = solved.k1_len as usize;
            std::array::from_fn(|k_eva| {
                solved.v_star[state_idx(0, 4, 0, k_eva, 0, k1_len, k2_len)]
            })
        })
        .unwrap();
        for k in 0..3 {
            assert!(
                vals[k + 1] >= vals[k] - 1e-5,
                "V* must be non-decreasing in evasion stage vs Roar: {vals:?}"
            );
        }
        assert!(
            vals[3] > vals[0] + 0.05,
            "V* at +3 evasion must strictly exceed +0 evasion vs Roar: {vals:?}"
        );

        // Check fast Encore / Haze strictly reduces V*(x) - Q*(x, Attack) vs same-typing passive wall
        // (Compare Vaporeon with Haze vs Vaporeon with Tail Whip so typing/stats are identical!)
        let zam_encore = make_set(
            "Alakazam",
            55,
            "Miracle Berry",
            &["Encore", "Recover", "Reflect", "Light Screen"],
        );
        let vap_haze = make_set(
            "Vaporeon",
            55,
            "Leftovers",
            &["Haze", "Rest", "Acid Armor", "Tail Whip"],
        );
        let vap_passive = make_set(
            "Vaporeon",
            55,
            "Leftovers",
            &["Water Gun", "Rest", "Acid Armor", "Tail Whip"],
        );
        let scizor_bp = make_set(
            "Scizor",
            50,
            "Leftovers",
            &["Steel Wing", "Swords Dance", "Agility", "Baton Pass"],
        );

        let margin_vs = |foe_lead: PokemonSet| -> f32 {
            let mut bm = Battle::from_fixture(
                &dex,
                "1,2,3,4",
                &[scizor_bp.clone(), wak.clone(), sui.clone()],
                &[foe_lead, lax.clone(), sui.clone()],
            )
            .unwrap();
            bm.choose(&dex, 0, "team 1,2,3").unwrap();
            bm.choose(&dex, 1, "team 1,2,3").unwrap();
            let (tm, cbm) = CompactBattle::from_battle(&bm, &dex);
            with_solved_relay(&cbm, &tm, &w, 0, |solved, lookup| {
                solved.v_star[lookup.x0] - solved.q_attack[lookup.x0]
            })
            .unwrap()
        };

        let m_wall = margin_vs(vap_passive);
        let m_encore = margin_vs(zam_encore);
        let m_haze = margin_vs(vap_haze);
        assert!(
            m_wall > 0.06 && m_encore < m_wall - 0.04 && m_haze < m_wall - 0.04,
            "Fast Encore ({m_encore}) and Haze ({m_haze}) must strictly suppress setup margin vs passive wall ({m_wall})"
        );
    }

    #[test]
    fn strict_antisymmetry_10000_boards_and_1v1_bit_exact_invariance() {
        use crate::eval::{
            compact_eval01, compact_eval01_blindspot, compact_eval01_blindspot_norelay,
            compact_eval_leaf, compact_eval_leaf_blindspot,
        };

        let dex = conformance::load_dex();
        let w = EvalWeights::default();
        let roster = [
            make_set("Jolteon", 50, "Leftovers", &["Thunderbolt", "Agility", "Substitute", "Baton Pass"]),
            make_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Double Team", "Substitute", "Baton Pass"]),
            make_set("Scizor", 50, "Leftovers", &["Steel Wing", "Swords Dance", "Agility", "Baton Pass"]),
            make_set("Umbreon", 50, "Leftovers", &["Feint Attack", "Curse", "Mean Look", "Baton Pass"]),
            make_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]),
            make_set("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            make_set("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            make_set("Raikou", 50, "Leftovers", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
            make_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            make_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Thunderbolt", "Recover"]),
            make_set("Alakazam", 50, "Miracle Berry", &["Psychic", "Encore", "Recover", "Thunder Wave"]),
            make_set("Quagsire", 50, "Leftovers", &["Earthquake", "Haze", "Rest", "Sludge Bomb"]),
        ];

        let mut rng = SplitMix64::new(0x2026_1117_beef_cafe);

        // 1. 10,000 boards (including active Baton Pass chains, Substitute, Spikes, statuses, boosts)
        for case in 0..10_000usize {
            let pick3 = |r: &mut SplitMix64| -> [PokemonSet; 3] {
                // Ensure <= 1 Lv55 mon so total level <= 155
                let i0 = r.below(roster.len());
                let mut i1 = r.below(roster.len());
                while i1 == i0 || (roster[i0].level == 55 && roster[i1].level == 55) {
                    i1 = r.below(roster.len());
                }
                let mut i2 = r.below(roster.len());
                while i2 == i0
                    || i2 == i1
                    || ((roster[i0].level == 55 || roster[i1].level == 55) && roster[i2].level == 55)
                {
                    i2 = r.below(roster.len());
                }
                [roster[i0].clone(), roster[i1].clone(), roster[i2].clone()]
            };
            let t0 = pick3(&mut rng);
            let t1 = pick3(&mut rng);
            let mut b_ab = Battle::from_fixture(&dex, "1,2,3,4", &t0, &t1).unwrap();
            b_ab.choose(&dex, 0, "team 1,2,3").unwrap();
            b_ab.choose(&dex, 1, "team 1,2,3").unwrap();

            let mut b_ba = Battle::from_fixture(&dex, "1,2,3,4", &t1, &t0).unwrap();
            b_ba.choose(&dex, 0, "team 1,2,3").unwrap();
            b_ba.choose(&dex, 1, "team 1,2,3").unwrap();

            let (tbl_ab, mut cb_ab) = CompactBattle::from_battle(&b_ab, &dex);
            let (tbl_ba, mut cb_ba) = CompactBattle::from_battle(&b_ba, &dex);

            // Symmetrically mutate active slots, alive masks, HP, Substitute, Spikes, statuses, and boosts
            let act0 = (rng.below(3)) as u8;
            let act1 = (rng.below(3)) as u8;
            cb_ab.sides[0].active_idx = act0;
            cb_ab.sides[1].active_idx = act1;
            cb_ba.sides[1].active_idx = act0;
            cb_ba.sides[0].active_idx = act1;

            let mask0 = (1u8 << act0) | ((rng.below(8) as u8) & 0b111);
            let mask1 = (1u8 << act1) | ((rng.below(8) as u8) & 0b111);
            cb_ab.sides[0].alive_mask = mask0;
            cb_ab.sides[1].alive_mask = mask1;
            cb_ba.sides[1].alive_mask = mask0;
            cb_ba.sides[0].alive_mask = mask1;

            let spk0 = (rng.below(2)) as u8;
            let spk1 = (rng.below(2)) as u8;
            cb_ab.sides[0].side_flags = (cb_ab.sides[0].side_flags & !3) | spk0;
            cb_ab.sides[1].side_flags = (cb_ab.sides[1].side_flags & !3) | spk1;
            cb_ba.sides[1].side_flags = (cb_ba.sides[1].side_flags & !3) | spk0;
            cb_ba.sides[0].side_flags = (cb_ba.sides[0].side_flags & !3) | spk1;

            for slot in 0..3usize {
                for s in 0..2usize {
                    let idx_ab = s * 3 + slot;
                    let idx_ba = (1 - s) * 3 + slot;
                    let maxhp = tbl_ab.mons[idx_ab].maxhp.max(4);
                    let alive = (cb_ab.sides[s].alive_mask & (1 << slot)) != 0;
                    let hp = if alive {
                        1 + (rng.below(maxhp as usize) as i16)
                    } else {
                        0
                    };
                    cb_ab.mons[idx_ab].hp_pool[0] = hp;
                    cb_ba.mons[idx_ba].hp_pool[0] = hp;

                    if alive && slot as u8 == cb_ab.sides[s].active_idx {
                        if rng.below(3) == 0 {
                            let sub_hp = (maxhp / 4).max(1);
                            cb_ab.mons[idx_ab].hp_pool[1] = sub_hp;
                            cb_ab.mons[idx_ab].vol_flags |= vol_bit::SUBSTITUTE;
                            cb_ba.mons[idx_ba].hp_pool[1] = sub_hp;
                            cb_ba.mons[idx_ba].vol_flags |= vol_bit::SUBSTITUTE;
                        }
                        let mut b_arr = [0i8; 7];
                        for bi in [0usize, 1, 2, 3, 4, 6] {
                            if rng.below(3) == 0 {
                                b_arr[bi] = (rng.below(5) as i8) - 1;
                            }
                        }
                        cb_ab.mons[idx_ab].boosts = b_arr;
                        cb_ba.mons[idx_ba].boosts = b_arr;
                    }
                }
            }

            // Alternate evaluation order across odd/even cases to verify RelayCache order-independence
            let (v_ab, v_ba) = if case % 2 == 0 {
                let a = compact_eval_leaf_blindspot(&cb_ab, &tbl_ab, &w);
                let b = compact_eval_leaf_blindspot(&cb_ba, &tbl_ba, &w);
                (a, b)
            } else {
                let b = compact_eval_leaf_blindspot(&cb_ba, &tbl_ba, &w);
                let a = compact_eval_leaf_blindspot(&cb_ab, &tbl_ab, &w);
                (a, b)
            };

            // For n0 + n1 >= 4, strict 1e-6 antisymmetry holds end-to-end;
            // on 1v1/2v1, linear.rs's f32 SVD is within 5e-5.
            let n_tot = mask0.count_ones() + mask1.count_ones();
            let tol = if n_tot >= 4 { 1e-6 } else { 5e-5 };
            assert!(
                (v_ab + v_ba - 1.0).abs() <= tol,
                "antisymmetry violated at case {case} (n_tot={n_tot}): v_ab={v_ab}, v_ba={v_ba}, sum={}",
                v_ab + v_ba
            );

            // 2. When 1v1 (mask0.count_ones() == 1 && mask1.count_ones() == 1), check 0-ULP bit identity!
            if mask0.count_ones() == 1 && mask1.count_ones() == 1 {
                assert_eq!(
                    compact_eval01_blindspot(&cb_ab, &tbl_ab, &w).to_bits(),
                    compact_eval01_blindspot_norelay(&cb_ab, &tbl_ab, &w).to_bits(),
                    "1v1 relay lift must be 0-ULP identical to norelay at case {case}"
                );
                assert_eq!(
                    compact_eval01_blindspot(&cb_ab, &tbl_ab, &w).to_bits(),
                    compact_eval01(&cb_ab, &tbl_ab, &w).to_bits(),
                    "1v1 compact_eval01_blindspot must be 0-ULP identical to compact_eval01 at case {case}"
                );
                assert_eq!(
                    compact_eval_leaf_blindspot(&cb_ab, &tbl_ab, &w).to_bits(),
                    compact_eval_leaf(&cb_ab, &tbl_ab, &w).to_bits(),
                    "1v1 compact_eval_leaf_blindspot must be 0-ULP identical to compact_eval_leaf at case {case}"
                );
            }
        }
    }

    #[test]
    fn premature_pass_suppression_and_winning_attack_gate_closure() {
        let dex = conformance::load_dex();
        let w = EvalWeights::default();

        // Scenario 1 (Premature Pass Suppression):
        // Jolteon behind intact Substitute (sigma=1, k_spe=0) vs slow Porygon2 (Tackle, Thunder Wave, Recover),
        // bench Thick Club Marowak (Lv50, 100% HP) + Forretress, opponent bench fast Starmie (Lv55 Mystic Water,
        // Surf deals >100% OHKO to Lv50 Marowak!) + Zapdos (prevents Jolteon solo sweep).
        // Immediate Pass at k_spe=0 fails because Lv55 Starmie (Spe 183 > Marowak 96) outspeeds & OHKOs Marowak;
        // Agility (+2 Spe -> 192 > 183) first lets Marowak outspeed both Starmie and Zapdos and sweep!
        let jolt = make_set("Jolteon", 55, "Leftovers", &["Thunderbolt", "Agility", "Substitute", "Baton Pass"]);
        let wak = make_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Fire Blast"]);
        let forre = make_set("Forretress", 50, "Leftovers", &["Spikes", "Rapid Spin", "Pin Missile", "Reflect"]);
        let pory2 = make_set("Porygon2", 50, "Leftovers", &["Tackle", "Recover", "Thunder Wave", "Sharpen"]);
        let starmie = make_set("Starmie", 55, "Mystic Water", &["Surf", "Psychic", "Thunderbolt", "Recover"]);
        let quag = make_set("Quagsire", 50, "Leftovers", &["Sludge Bomb", "Body Slam", "Rest", "Amnesia"]);

        let mut b = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[jolt.clone(), wak.clone(), forre.clone()],
            &[pory2, starmie, quag],
        )
        .unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();
        b.choose(&dex, 0, "move substitute").unwrap();
        b.choose(&dex, 1, "move thunderwave").unwrap();
        let (table, cb) = CompactBattle::from_battle(&b, &dex);

        let mut piv = [1u8; 2];
        let mut rng = SplitMix64::new(42);
        let cs = [0u8, 1, 2, 3];
        let dec = rollout_relay_pick(&cb, &table, &w, 0, &cs, 0, &mut piv, true, &mut rng);
        // Move slot 1 is Agility (`&["Thunderbolt", "Agility", "Substitute", "Baton Pass"]`)
        assert_eq!(
            dec,
            RelayRolloutDecision::Chosen(1),
            "Jolteon behind Sub facing fast Water bench counter must click Agility (slot 1) before Baton Pass, got {dec:?}"
        );

        // Scenario 2 (Over-Setup Gate Closure):
        // Jolteon Thunderbolt already 1-2HKOs entire Water/Flying opponent team -> gate closes!
        let starmie50 = make_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Thunderbolt", "Recover"]);
        let cloyster = make_set("Cloyster", 50, "Leftovers", &["Surf", "Ice Beam", "Spikes", "Explosion"]);
        let skarm = make_set("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]);
        let mut b_win = Battle::from_fixture(
            &dex,
            "1,2,3,4",
            &[jolt, wak, forre],
            &[starmie50, cloyster, skarm],
        )
        .unwrap();
        b_win.choose(&dex, 0, "team 1,2,3").unwrap();
        b_win.choose(&dex, 1, "team 1,2,3").unwrap();
        let (t_win, cb_win) = CompactBattle::from_battle(&b_win, &dex);
        let dec_win = rollout_relay_pick(&cb_win, &t_win, &w, 0, &cs, 0, &mut [1, 1], true, &mut rng);
        assert!(
            matches!(
                dec_win,
                RelayRolloutDecision::GateClosedSuppressLegacyBp | RelayRolloutDecision::Chosen(0)
            ),
            "Winning Jolteon vs 3 electric-weak foes must NOT waste turns setting up: got {dec_win:?}"
        );
    }

    #[test]
    fn canonical_mon_key_tracks_dynamic_stolen_stat_and_type_items() {
        let dex = conformance::load_dex();
        let wak = make_set("Marowak", 50, "No Item", &["Earthquake", "Rock Slide", "Thief", "Splash"]);
        let foe_tc = make_set("Cubone", 50, "Thick Club", &["Splash"]);
        let foe_ch = make_set("Typhlosion", 50, "Charcoal", &["Fire Blast"]);
        let foe_mw = make_set("Starmie", 50, "Mystic Water", &["Surf"]);
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[wak], &[foe_tc, foe_ch, foe_mw]).unwrap();
        b.choose(&dex, 0, "team 1").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();
        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);

        let tc_item = cb.mons[3].item;
        let ch_item = cb.mons[4].item;
        let mw_item = cb.mons[5].item;
        let k_none = canonical_mon_key(&cb, &table, 0, 0, true, None);
        cb.mons[0].item = tc_item;
        let k_tc = canonical_mon_key(&cb, &table, 0, 0, true, None);
        cb.mons[0].item = ch_item;
        let k_ch = canonical_mon_key(&cb, &table, 0, 0, true, None);
        cb.mons[0].item = mw_item;
        let k_mw = canonical_mon_key(&cb, &table, 0, 0, true, None);
        assert_ne!(k_none, k_tc, "Relay key must distinguish stolen Thick Club");
        assert_ne!(k_tc, k_ch, "Relay key must distinguish Thick Club from Charcoal");
        assert_ne!(k_ch, k_mw, "Relay key must distinguish Charcoal from Mystic Water");
    }
}
