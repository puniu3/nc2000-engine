//! M6 static eval: a hand-written, weight-parameterized position evaluation
//! plus the expected-damage estimate shared by the eval's threat feature and
//! the heavy rollout policy.
//!
//! The eval is linear in `EvalWeights` (features are O(1)-bounded per side),
//! then a sigmoid maps the side-0-minus-side-1 differential into (0, 1).
//! Linearity keeps SPSA well-behaved; the sigmoid scale is itself a weight.
//!
//! Damage model: the gen-2 core formula on *effective* stats
//! (`Battle::get_stat` — boosts, burn/par drops, screens, boosting items) with
//! STAB, type effectiveness, mean damage roll (236/255), mean multi-hit count,
//! and accuracy folded in. Callback-powered base powers (flail/magnitude/
//! return/...) score their static value — same caveat as MaxDamage; hidden
//! power uses the mon's real DV-derived type and power.

use nc2000_engine::battle::Outcome;
use nc2000_engine::dex::{Dex, MoveId, TypeId};
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::{Battle, PokeId, Status};

/// Castle PBS `L_TURN` judge value from side 0's perspective (`1.0` = P1Win, `0.5` = Tie, `0.0` = P2Win):
/// compares living Pokemon count first, then sum of living `hp / maxhp` ratios (f64, seat-unbiased).
/// Verified by live `PS.cgi` probes `12a1`, `12a2`, `12a3`.
pub fn castle_judge_value_battle(b: &Battle) -> f64 {
    let mut alive = [0u32; 2];
    let mut hp_sum = [0.0f64; 2];
    for side in 0..2 {
        for &slot in &b.sides[side].party {
            let p = &b.sides[side].roster[slot as usize];
            if !p.fainted && p.hp > 0 {
                alive[side] += 1;
                hp_sum[side] += (p.hp as f64) / (p.maxhp.max(1) as f64);
            }
        }
    }
    if alive[0] != alive[1] {
        return if alive[0] > alive[1] { 1.0 } else { 0.0 };
    }
    if (hp_sum[0] - hp_sum[1]).abs() <= 1e-12 {
        0.5
    } else if hp_sum[0] > hp_sum[1] {
        1.0
    } else {
        0.0
    }
}

/// Terminal outcome value from side 0's perspective on `Battle`, honoring both `Battle::outcome()`
/// and `b.ruleset.turn_limit()` (`1000`-turn tie on PS; `L_TURN` alive-count -> HP-ratio judge on Castle).
#[inline]
pub fn turn_limit_outcome_value(b: &Battle) -> Option<f64> {
    if let Some(o) = b.outcome() {
        return Some(match o {
            Outcome::P1Win => 1.0,
            Outcome::Tie => 0.5,
            Outcome::P2Win => 0.0,
        });
    }
    if let Some(lim) = b.ruleset.turn_limit() {
        if b.turn > lim {
            return Some(if b.ruleset.is_castle() {
                castle_judge_value_battle(b)
            } else {
                0.5
            });
        }
    }
    None
}

/// Castle PBS `L_TURN` judge value from side 0's perspective on `CompactBattle`.
#[inline]
pub fn compact_castle_judge_value(cb: &CompactBattle, table: &MatchTable) -> f64 {
    let mut alive = [0u32; 2];
    let mut hp_sum = [0.0f64; 2];
    for side in 0..2 {
        let mask = cb.sides[side].alive_mask;
        for &slot_u8 in &cb.party_order(side) {
            let slot = slot_u8 as usize;
            if (mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = side * 3 + slot;
            let hp = cb.mons[idx].hp();
            if hp > 0 {
                alive[side] += 1;
                hp_sum[side] += (hp as f64) / (table.mons[idx].maxhp.max(1) as f64);
            }
        }
    }
    if alive[0] != alive[1] {
        return if alive[0] > alive[1] { 1.0 } else { 0.0 };
    }
    if (hp_sum[0] - hp_sum[1]).abs() <= 1e-12 {
        0.5
    } else if hp_sum[0] > hp_sum[1] {
        1.0
    } else {
        0.0
    }
}

/// Terminal outcome value from side 0's perspective on `CompactBattle`, honoring `cb.outcome`
/// and `table.ruleset.turn_limit()`.
#[inline]
pub fn compact_turn_limit_outcome(cb: &CompactBattle, table: &MatchTable) -> Option<f64> {
    match cb.outcome {
        1 => return Some(1.0),
        2 => return Some(0.0),
        3 => return Some(0.5),
        _ => {}
    }
    if let Some(lim) = table.ruleset.turn_limit() {
        if cb.turn > lim {
            return Some(if table.ruleset.is_castle() {
                compact_castle_judge_value(cb, table)
            } else {
                0.5
            });
        }
    }
    None
}

/// Ruleset-aware Flail/Reversal base power (`20 <-> 40` boundary is `< 77.5%` on Castle vs `floor(48*hp/maxhp) < 33` on PS).
#[inline]
pub(crate) fn ruleset_flail_bp(ruleset: Ruleset, hp: i32, maxhp: i32) -> i32 {
    let mx = maxhp.max(1);
    let ratio = ((hp as f64 * 48.0 / mx as f64).floor() as i32).max(1);
    if ratio < 2 {
        200
    } else if ratio < 5 {
        150
    } else if ratio < 10 {
        100
    } else if ratio < 17 {
        80
    } else if ruleset.is_castle() {
        if (hp as i64) * 40 < (mx as i64) * 31 {
            40
        } else {
            20
        }
    } else if ratio < 33 {
        40
    } else {
        20
    }
}

/// Ruleset-aware Gen 2 accuracy threshold out of 256 (exact `-20/256` BrightPowder subtraction;
/// PS GB integer tables vs Castle exact `3/9..9/3` fractions with Foresight `acc < eva` rule).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ruleset_accuracy_threshold(
    ruleset: Ruleset,
    base_pct: u8,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    ignore_positive_evasion: bool,
    foresight_active: bool,
    brightpowder: bool,
) -> u8 {
    if ruleset.is_ps() {
        return nc2000_engine::compact::lookup_accuracy_threshold(
            base_pct,
            acc_boost,
            eva_boost,
            ignore_accuracy,
            ignore_evasion,
            ignore_positive_evasion,
            brightpowder,
        );
    }
    nc2000_engine::compact::lookup_accuracy_threshold_castle(
        base_pct,
        acc_boost,
        eva_boost,
        ignore_accuracy,
        ignore_evasion,
        foresight_active,
        brightpowder,
    )
}

/// Ruleset-aware hit probability in `[0.0, 1.0]` with exact `-20/256` BrightPowder subtraction.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ruleset_hit_probability(
    ruleset: Ruleset,
    base_pct: u8,
    always_hit: bool,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    ignore_positive_evasion: bool,
    foresight_active: bool,
    brightpowder: bool,
) -> f64 {
    if always_hit {
        return 1.0;
    }
    let thr = ruleset_accuracy_threshold(
        ruleset,
        base_pct,
        acc_boost,
        eva_boost,
        ignore_accuracy,
        ignore_evasion,
        ignore_positive_evasion,
        foresight_active,
        brightpowder,
    );
    if thr == 255 {
        1.0
    } else {
        (thr as f64) / 256.0
    }
}

/// Ruleset-aware effective stat calculation on `CompactBattle`:
/// - PS: exact `get_stat_branchless` / `cb.mon_stat` parity.
/// - Castle: delegates directly to `nc2000_engine::compact::get_stat_castle`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn ruleset_mon_stat(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
    stat_i: usize,
    boost_override: Option<i8>,
    unboosted: bool,
    unmodified: bool,
) -> i32 {
    let p_idx = side * 3 + slot;
    let mon = &cb.mons[p_idx];
    let spec = &table.mons[p_idx];
    let side_st = &cb.sides[side];
    let boost = boost_override.unwrap_or(mon.boosts[stat_i]);
    if table.ruleset.is_ps() {
        if boost_override.is_none() {
            return cb.mon_stat(table, side, slot, stat_i, unboosted, unmodified);
        }
        let has_par = (mon.status_raw() == Status::Par as u8)
            & ((mon.vol_flags & vol_bit::PAR_SPEED_DROP) != 0);
        let has_brn = (mon.status_raw() == Status::Brn as u8)
            & ((mon.vol_flags & vol_bit::BRN_ATK_DROP) != 0);
        let ispec = table.item_spec(mon.item);
        let item_kind = if ispec.stat_kind != nc2000_engine::compact::ItemStatKind::None
            && ispec.stat_kind == spec.species_stat_kind
        {
            ispec.stat_kind
        } else {
            nc2000_engine::compact::ItemStatKind::None
        };
        return nc2000_engine::compact::get_stat_branchless(
            spec.stored_stats[stat_i] as i32,
            boost,
            stat_i,
            unboosted,
            unmodified,
            false,
            has_par,
            has_brn,
            side_st.reflect_turns > 0,
            side_st.lightscreen_turns > 0,
            item_kind,
        );
    }
    let ispec = table.item_spec(mon.item);
    let item_kind = if ispec.stat_kind != nc2000_engine::compact::ItemStatKind::None
        && ispec.stat_kind == spec.species_stat_kind
    {
        ispec.stat_kind
    } else {
        nc2000_engine::compact::ItemStatKind::None
    };
    nc2000_engine::compact::get_stat_castle(
        spec.stored_stats[stat_i] as i32,
        boost,
        stat_i,
        unboosted,
        unmodified,
        mon.status_raw() == Status::Par as u8,
        mon.status_raw() == Status::Brn as u8,
        side_st.reflect_turns > 0,
        side_st.lightscreen_turns > 0,
        item_kind,
        table.ruleset.overflow_mode(),
    )
}

#[inline]
pub(crate) fn castle_speed_rational(stored_spe: i32, boost: i8, is_par: bool) -> (u128, u128) {
    let b = boost.clamp(-6, 6);
    let (s_num, s_den) = if b >= 0 {
        ((2 + b) as u128, 2u128)
    } else {
        (2u128, (2 - b) as u128)
    };
    let num = (stored_spe.max(1) as u128) * s_num;
    let den = s_den * if is_par { 4 } else { 1 };
    (num, den)
}

/// Exact ruleset-aware action speed comparison (`Ordering` of side `s0` vs side `s1`).
/// On Castle PBS, compares non-truncated Perl `f64` speeds (`Probe 12H`) via exact rational
/// cross-multiplication while honoring 2-side `quick_claw_for`.
#[inline]
pub(crate) fn compare_action_speed(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
) -> std::cmp::Ordering {
    if table.ruleset.is_ps() {
        return cb
            .mon_action_speed(table, s0, slot0)
            .cmp(&cb.mon_action_speed(table, s1, slot1));
    }
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let m0 = &cb.mons[idx0];
    let m1 = &cb.mons[idx1];
    let qc0 = cb.quick_claw_for(s0) && table.item_spec(m0.item).is_quick_claw;
    let qc1 = cb.quick_claw_for(s1) && table.item_spec(m1.item).is_quick_claw;
    match (qc0, qc1) {
        (true, false) => return std::cmp::Ordering::Greater,
        (false, true) => return std::cmp::Ordering::Less,
        (true, true) => return std::cmp::Ordering::Equal,
        (false, false) => {}
    }
    let (n0, d0) = castle_speed_rational(
        table.mons[idx0].stored_stats[4] as i32,
        m0.boosts[4],
        m0.status_raw() == Status::Par as u8,
    );
    let (n1, d1) = castle_speed_rational(
        table.mons[idx1].stored_stats[4] as i32,
        m1.boosts[4],
        m1.status_raw() == Status::Par as u8,
    );
    (n0 * d1).cmp(&(n1 * d0))
}

/// Exact ruleset-aware speed comparison with explicit speed-stage overrides (used by `relay.rs`).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_speed_with_boost(
    cb: &CompactBattle,
    table: &MatchTable,
    s0: usize,
    slot0: usize,
    boost0: i8,
    s1: usize,
    slot1: usize,
    boost1: i8,
) -> std::cmp::Ordering {
    if table.ruleset.is_ps() {
        let sp0 = ruleset_mon_stat(cb, table, s0, slot0, 4, Some(boost0), false, false);
        let sp1 = ruleset_mon_stat(cb, table, s1, slot1, 4, Some(boost1), false, false);
        return sp0.cmp(&sp1);
    }
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let (n0, d0) = castle_speed_rational(
        table.mons[idx0].stored_stats[4] as i32,
        boost0,
        cb.mons[idx0].status_raw() == Status::Par as u8,
    );
    let (n1, d1) = castle_speed_rational(
        table.mons[idx1].stored_stats[4] as i32,
        boost1,
        cb.mons[idx1].status_raw() == Status::Par as u8,
    );
    (n0 * d1).cmp(&(n1 * d0))
}

/// Tunable weights. All features are from the owning side's perspective
/// (penalties are stored positive and subtracted).
#[derive(Clone, Debug, PartialEq)]
pub struct EvalWeights {
    /// Per living party mon: HP fraction.
    pub hp: f64,
    /// Per living party mon: flat existence bonus.
    pub alive: f64,
    /// Status penalties on living mons (subtracted).
    pub brn: f64,
    pub par: f64,
    pub slp: f64,
    pub frz: f64,
    pub psn: f64,
    pub tox: f64,
    /// Per-stage boost values on the (living) active mon:
    /// atk, def, spa, spd, spe, accuracy, evasion.
    pub boost: [f64; 7],
    /// Best expected hit fraction vs the opposing active (clamped to [0,1]).
    pub threat: f64,
    /// Mean PP fraction over living mons' move slots.
    pub pp: f64,
    /// Sigmoid sharpness on the side differential.
    pub scale: f64,
    /// Flat bonus while the active has a Substitute up. 0.0 = off.
    pub substitute: f64,
    /// KO-race term (M17c): adds `race × margin` to the side differential on 1v1 last-mon races.
    pub race: f64,
    /// Search-cutoff backup around 0.5 (1.0 retains probability semantics).
    pub leaf_alpha: f64,
    /// Entry-hazard term (M17 tail): Spikes on grounded living benched mons.
    pub spikes: f64,
}

impl Default for EvalWeights {
    /// M6 hand-written starting point, revised 2026-07-21 by the M16a/M17
    /// paired calibration: slp/frz/tox x0.7 + sleep-clock scaling +
    /// substitute bonus + evasion coupling + Spikes.
    fn default() -> Self {
        EvalWeights {
            hp: 1.0,
            alive: 0.5,
            brn: 0.35,
            par: 0.35,
            slp: 0.42,
            frz: 0.56,
            psn: 0.25,
            tox: 0.35,
            boost: [0.15, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10],
            threat: 0.5,
            pp: 0.2,
            scale: 1.5,
            substitute: 0.5,
            race: 3.0,
            leaf_alpha: 1.0,
            spikes: 1.5,
        }
    }
}

impl EvalWeights {
    pub const N: usize = 17;

    pub const NAMES: [&'static str; Self::N] = [
        "hp", "alive", "brn", "par", "slp", "frz", "psn", "tox", "boost_atk", "boost_def",
        "boost_spa", "boost_spd", "boost_spe", "boost_acc", "boost_eva", "threat", "pp",
    ];

    /// Vector form for tuning. `scale` is deliberately NOT in the vector:
    /// scaling all weights uniformly is the same knob, so tuning it separately
    /// only adds a redundant dimension.
    pub fn to_vec(&self) -> [f64; Self::N] {
        [
            self.hp, self.alive, self.brn, self.par, self.slp, self.frz, self.psn, self.tox,
            self.boost[0], self.boost[1], self.boost[2], self.boost[3], self.boost[4],
            self.boost[5], self.boost[6], self.threat, self.pp,
        ]
    }

    pub fn from_vec(v: &[f64; Self::N], scale: f64) -> Self {
        EvalWeights {
            hp: v[0],
            alive: v[1],
            brn: v[2],
            par: v[3],
            slp: v[4],
            frz: v[5],
            psn: v[6],
            tox: v[7],
            boost: [v[8], v[9], v[10], v[11], v[12], v[13], v[14]],
            threat: v[15],
            pp: v[16],
            scale,
            substitute: 0.5,
            race: 3.0,
            leaf_alpha: 1.0,
            spikes: 1.5,
        }
    }
}

/// Win-probability-shaped eval in (0, 1) from side 0's perspective.
pub fn eval01(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    compact_eval01(&cb, &table, w)
}

/// KO-race margin from side 0's perspective, in [-1, 1]; nonzero only in
/// last-mon-vs-last-mon states (no switches — a pure race).
pub fn race_margin(b: &Battle, dex: &Dex) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    compact_race_margin(&cb, &table)
}

/// Leaf value for search cutoffs. The calibrated probability is backed up
/// directly at alpha 1; smaller alpha shrinks it toward 0.5.
#[inline]
pub fn eval_leaf(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    eval_leaf_cached(b, dex, w, None)
}

#[inline]
pub fn eval_leaf_cached(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
    roster_cache: Option<&nc2000_engine::compact::RosterCompactCache>,
) -> f64 {
    let (table, cb) = CompactBattle::from_battle_with_roster_cache(b, dex, roster_cache);
    compact_eval_leaf(&cb, &table, w)
}

/// Leaf value with surgical balance-sheet blind-spot accounting (setup/boost
/// liabilities, frozen asset impairment, healer poison liability).
#[inline]
pub fn eval_leaf_blindspot(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    eval_leaf_blindspot_cached(b, dex, w, None)
}

#[inline]
pub fn eval_leaf_blindspot_cached(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
    roster_cache: Option<&nc2000_engine::compact::RosterCompactCache>,
) -> f64 {
    let (table, cb) = CompactBattle::from_battle_with_roster_cache(b, dex, roster_cache);
    compact_eval_leaf_blindspot(&cb, &table, w)
}

#[inline]
pub fn eval_leaf_blindspot_norelay_cached(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
    roster_cache: Option<&nc2000_engine::compact::RosterCompactCache>,
) -> f64 {
    let (table, cb) = CompactBattle::from_battle_with_roster_cache(b, dex, roster_cache);
    compact_eval_leaf_blindspot_norelay(&cb, &table, w)
}


/// Best expected hit fraction over the attacker's usable move slots
/// (unclamped: >1 = expected OHKO with margin), folding Gen 2 accuracy×evasion stages.
pub fn best_hit_fraction(b: &Battle, dex: &Dex, att: PokeId, def: PokeId) -> f64 {
    let dt = nc2000_engine::compact::dex_tables(dex);
    let a = b.poke(att);
    let d = b.poke(def);
    let def_hp = d.hp.max(1) as f64;
    let level_factor = (a.level as f64 * 2.0 / 5.0 + 2.0).floor();
    let def_has_ghost = d.has_type(dex.known_types.ghost);
    let def_has_foresight = dt.fore_cond.is_some_and(|id| d.has_volatile(id));
    let def_foresight = def_has_ghost && def_has_foresight;
    let def_has_flying = d.has_type(dex.known_types.flying);

    let mut phys_stats: Option<(f64, f64)> = None;
    let mut spec_stats: Option<(f64, f64)> = None;
    let mut best = 0.0f64;

    for ms in a.move_slots.iter() {
        if ms.pp <= 0 || ms.disabled {
            continue;
        }
        let mspec = &dt.move_specs[ms.id.0 as usize];
        let (move_type, is_physical, is_status, base_power) = match mspec.var_bp_kind {
            nc2000_engine::compact::VarBpKind::Return => (
                TypeId(mspec.move_type),
                mspec.is_physical,
                mspec.is_status,
                a.happiness as i32 * 10 / 25,
            ),
            nc2000_engine::compact::VarBpKind::Frustration => (
                TypeId(mspec.move_type),
                mspec.is_physical,
                mspec.is_status,
                if b.is_castle() {
                    102
                } else {
                    (255 - a.happiness as i32) * 10 / 25
                },
            ),
            nc2000_engine::compact::VarBpKind::HiddenPower => {
                let hpt = a.hp_type;
                let is_phys = (dt.special_type_mask & (1u32 << (hpt.0 & 31))) == 0;
                (hpt, is_phys, false, a.hp_power)
            }
            nc2000_engine::compact::VarBpKind::FlailReversal => {
                let bp = ruleset_flail_bp(b.ruleset, a.hp, a.maxhp);
                (TypeId(mspec.move_type), mspec.is_physical, mspec.is_status, bp)
            }
            _ => (
                TypeId(mspec.move_type),
                mspec.is_physical,
                mspec.is_status,
                mspec.base_power as i32,
            ),
        };

        if mspec.fixed_damage == 0
            && mspec.special_kind != nc2000_engine::compact::SpecialMoveKind::SuperFang
            && (is_status || base_power <= 0)
        {
            continue;
        }

        let mt = move_type.0;
        let row = &dt.type_chart[(mt as usize).min(19)];
        let immune = if (mt == dt.normal_type || mt == dt.fighting_type) && def_foresight {
            false
        } else if mt == dt.ground_type {
            def_has_flying
        } else {
            d.types.iter().any(|t| row[(t.0 as usize).min(19)] == -9)
        };
        if immune {
            continue;
        }

        let mut eff = 1.0f64;
        for t in d.types.iter() {
            match row[(t.0 as usize).min(19)] {
                1 => eff *= 2.0,
                -1 => eff *= 0.5,
                _ => {}
            }
        }

        let acc = ruleset_hit_probability(
            b.ruleset,
            mspec.base_acc_pct,
            mspec.always_hit,
            a.boosts[5],
            d.boosts[6],
            mspec.ignore_accuracy,
            mspec.ignore_evasion,
            mspec.ignore_positive_evasion,
            def_has_foresight,
            false,
        );

        let raw = if mspec.special_kind == nc2000_engine::compact::SpecialMoveKind::SuperFang {
            (d.hp / 2).max(1) as f64
        } else if mspec.fixed_damage != 0 {
            if mspec.fixed_damage < 0 {
                a.level as f64
            } else {
                mspec.fixed_damage as f64
            }
        } else {
            let (atk, mut defense) = if is_physical {
                *phys_stats.get_or_insert_with(|| {
                    (
                        b.get_stat(dex, att, 0, false, false, false) as f64,
                        b.get_stat(dex, def, 1, false, false, false) as f64,
                    )
                })
            } else {
                *spec_stats.get_or_insert_with(|| {
                    (
                        b.get_stat(dex, att, 2, false, false, false) as f64,
                        b.get_stat(dex, def, 3, false, false, false) as f64,
                    )
                })
            };
            if mspec.is_selfdestruct && is_physical {
                defense = (defense / 2.0).floor().max(1.0);
            }
            let mut dmg = (level_factor * base_power as f64 * atk / defense / 50.0).floor();
            if let Some(iid) = a.item {
                let ispec = &dt.item_specs[iid.0 as usize];
                let boost_type = if b.is_castle() {
                    ispec.castle_boost_type
                } else {
                    ispec.boost_type
                };
                if mt != 0xFF && boost_type == mt {
                    dmg += (dmg / 10.0).floor();
                }
            }
            let mut core = dmg.clamp(1.0, 997.0) + 2.0;
            let w_turns = b.field.weather_state.duration.unwrap_or(0);
            if w_turns > 0 {
                let rain = b.field.weather.is_some() && b.field.weather == dt.rain_cond;
                let sun = b.field.weather.is_some() && b.field.weather == dt.sun_cond;
                let is_water = move_type == dex.known_types.water;
                let is_fire = move_type == dex.known_types.fire;
                if (is_water && rain) || (is_fire && sun) {
                    core = (core * 1.5).floor();
                } else if ((is_fire || mspec.is_solarbeam) && rain) || (is_water && sun) {
                    core = (core / 2.0).floor();
                }
            }
            let stab = if a.types.has(move_type) { 1.5 } else { 1.0 };
            let hits = mspec.multihit_mean_x2 as f64 * 0.5;
            core * stab * eff * (236.0 / 255.0) * hits
        };

        best = best.max(raw * acc / def_hp);
    }

    if cfg!(debug_assertions) {
        let mut ref_best = 0.0f64;
        for ms in a.move_slots.iter() {
            if ms.pp > 0 && !ms.disabled {
                ref_best = ref_best.max(expected_hit_fraction(b, dex, att, def, ms.id));
            }
        }
        debug_assert_eq!(best.to_bits(), ref_best.to_bits());
    }
    best
}

pub(crate) fn type_immune_to(b: &Battle, dex: &Dex, move_type: TypeId, def: PokeId) -> bool {
    let dt = nc2000_engine::compact::dex_tables(dex);
    let d = b.poke(def);
    let mt = move_type.0;
    if (mt == dt.normal_type || mt == dt.fighting_type)
        && d.has_type(dex.known_types.ghost)
        && dt.fore_cond.is_some_and(|id| d.has_volatile(id))
    {
        return false;
    }
    if mt == dt.ground_type {
        d.has_type(dex.known_types.flying)
    } else {
        let row = &dt.type_chart[(mt as usize).min(19)];
        d.types.iter().any(|t| row[(t.0 as usize).min(19)] == -9)
    }
}

/// Expected fraction of the defender's *current* HP removed by one use of
/// `move_id`: gen-2 damage core on effective stats x STAB x effectiveness x
/// mean roll x mean hits x Gen 2 accuracy×evasion stage probability.
pub fn expected_hit_fraction(
    b: &Battle,
    dex: &Dex,
    att: PokeId,
    def: PokeId,
    move_id: MoveId,
) -> f64 {
    let dt = nc2000_engine::compact::dex_tables(dex);
    let mspec = &dt.move_specs[move_id.0 as usize];
    let a = b.poke(att);
    let d = b.poke(def);

    let (move_type, is_physical, is_status, base_power) = match mspec.var_bp_kind {
        nc2000_engine::compact::VarBpKind::Return => (
            TypeId(mspec.move_type),
            mspec.is_physical,
            mspec.is_status,
            a.happiness as i32 * 10 / 25,
        ),
        nc2000_engine::compact::VarBpKind::Frustration => (
            TypeId(mspec.move_type),
            mspec.is_physical,
            mspec.is_status,
            if b.is_castle() {
                102
            } else {
                (255 - a.happiness as i32) * 10 / 25
            },
        ),
        nc2000_engine::compact::VarBpKind::HiddenPower => {
            let hpt = a.hp_type;
            let is_phys = (dt.special_type_mask & (1u32 << (hpt.0 & 31))) == 0;
            (hpt, is_phys, false, a.hp_power)
        }
        nc2000_engine::compact::VarBpKind::FlailReversal => {
            let bp = ruleset_flail_bp(b.ruleset, a.hp, a.maxhp);
            (TypeId(mspec.move_type), mspec.is_physical, mspec.is_status, bp)
        }
        _ => (
            TypeId(mspec.move_type),
            mspec.is_physical,
            mspec.is_status,
            mspec.base_power as i32,
        ),
    };

    if mspec.fixed_damage == 0
        && mspec.special_kind != nc2000_engine::compact::SpecialMoveKind::SuperFang
        && (is_status || base_power <= 0)
    {
        return 0.0;
    }

    if type_immune_to(b, dex, move_type, def) {
        return 0.0;
    }
    let row = &dt.type_chart[(move_type.0 as usize).min(19)];
    let mut eff = 1.0f64;
    for t in d.types.iter() {
        match row[(t.0 as usize).min(19)] {
            1 => eff *= 2.0,
            -1 => eff *= 0.5,
            _ => {}
        }
    }

    let def_has_foresight = dt.fore_cond.is_some_and(|id| d.has_volatile(id));
    let acc = ruleset_hit_probability(
        b.ruleset,
        mspec.base_acc_pct,
        mspec.always_hit,
        a.boosts[5],
        d.boosts[6],
        mspec.ignore_accuracy,
        mspec.ignore_evasion,
        mspec.ignore_positive_evasion,
        def_has_foresight,
        false,
    );

    let raw = if mspec.special_kind == nc2000_engine::compact::SpecialMoveKind::SuperFang {
        (d.hp / 2).max(1) as f64
    } else if mspec.fixed_damage != 0 {
        if mspec.fixed_damage < 0 {
            a.level as f64
        } else {
            mspec.fixed_damage as f64
        }
    } else {
        let (ai, di) = if is_physical { (0, 1) } else { (2, 3) };
        let atk = b.get_stat(dex, att, ai, false, false, false) as f64;
        let mut defense = b.get_stat(dex, def, di, false, false, false) as f64;
        if mspec.is_selfdestruct && di == 1 {
            defense = (defense / 2.0).floor().max(1.0);
        }
        let mut dmg = ((a.level as f64 * 2.0 / 5.0 + 2.0).floor() * base_power as f64 * atk
            / defense
            / 50.0)
            .floor();
        if let Some(iid) = a.item {
            let ispec = &dt.item_specs[iid.0 as usize];
            let boost_type = if b.is_castle() {
                ispec.castle_boost_type
            } else {
                ispec.boost_type
            };
            if move_type.0 != nc2000_engine::compact::NO_TYPE_ID && boost_type == move_type.0 {
                dmg += (dmg / 10.0).floor();
            }
        }
        let mut core = dmg.clamp(1.0, 997.0) + 2.0;
        let w_turns = b.field.weather_state.duration.unwrap_or(0);
        if w_turns > 0 {
            let rain = b.field.weather.is_some() && b.field.weather == dt.rain_cond;
            let sun = b.field.weather.is_some() && b.field.weather == dt.sun_cond;
            let is_water = move_type == dex.known_types.water;
            let is_fire = move_type == dex.known_types.fire;
            if (is_water && rain) || (is_fire && sun) {
                core = (core * 1.5).floor();
            } else if ((is_fire || mspec.is_solarbeam) && rain) || (is_water && sun) {
                core = (core / 2.0).floor();
            }
        }
        let stab = if a.types.has(move_type) { 1.5 } else { 1.0 };
        let hits = mspec.multihit_mean_x2 as f64 * 0.5;
        core * stab * eff * (236.0 / 255.0) * hits
    };

    raw * acc / d.hp.max(1) as f64
}

// =========================================================================
// CompactBattle Leaf Evaluation (mirrors eval_leaf / eval01 on 256B kernel)
// =========================================================================

use nc2000_engine::compact::{vol_bit, CompactBattle, MatchTable};

/// Leaf evaluation on `CompactBattle` mirroring [`eval_leaf`].
#[inline]
pub fn compact_eval_leaf(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    assert!((0.0..=1.0).contains(&w.leaf_alpha), "leaf alpha must be in [0,1]");
    0.5 + w.leaf_alpha * (compact_eval01(cb, table, w) - 0.5)
}

/// Leaf evaluation on `CompactBattle` with surgical blind-spot balance-sheet accounting
/// and Setup-Relay simultaneous-move Bellman subgame lift.
#[inline]
pub fn compact_eval_leaf_blindspot(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    assert!((0.0..=1.0).contains(&w.leaf_alpha), "leaf alpha must be in [0,1]");
    0.5 + w.leaf_alpha * (compact_eval01_blindspot(cb, table, w) - 0.5)
}

/// Ablation leaf evaluation without Setup-Relay Bellman lift (`-relay`).
#[inline]
pub fn compact_eval_leaf_blindspot_norelay(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
) -> f64 {
    assert!((0.0..=1.0).contains(&w.leaf_alpha), "leaf alpha must be in [0,1]");
    0.5 + w.leaf_alpha * (compact_eval01_blindspot_norelay(cb, table, w) - 0.5)
}

/// Win-probability-shaped eval in `(0, 1)` from side 0's perspective on `CompactBattle`,
/// adding `compact_false_boost_refund_diff` and `relay::apply_relay_leaf_lift`
/// (strictly antisymmetric, `0.0` on `1v1` so exact Schur-Bellman / Toeplitz M17e anchors
/// remain 0-ULP invariant).
#[inline]
pub fn compact_eval01_blindspot(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    let v_base = compact_eval01_blindspot_norelay(cb, table, w);
    crate::relay::apply_relay_leaf_lift(cb, table, w, v_base)
}

/// Static blind-spot win probability before the Setup-Relay Bellman subgame lift.
#[inline]
pub fn compact_eval01_blindspot_norelay(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
) -> f64 {
    if let Some(v) = compact_turn_limit_outcome(cb, table) {
        return v;
    }
    let s0 = compact_side_score(cb, table, w, 0);
    let s1 = compact_side_score(cb, table, w, 1);
    let mut diff = s0 - s1;
    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    if n0 > 1 || n1 > 1 {
        diff += compact_false_boost_refund_diff(cb, table, w);
    }
    let mut old_rm = 0.0;
    if w.race != 0.0 {
        old_rm = compact_race_margin(cb, table);
        diff += w.race * old_rm;
    }
    diff += crate::linear::compact_linear_diff(cb, table, diff, old_rm, w.race, w.scale);
    1.0 / (1.0 + (-(w.scale * diff)).exp())
}

/// Strictly antisymmetric balance-sheet adjustment in `compact_eval01_blindspot`:
/// 1. `Swagger` false `+Atk` credit refund on a mon with no self-setup/sustain move (`!has_self_setup && !team_has_bp`).
/// 2. `Encore` lock refund on a boosted setup mon without Baton Pass (`vol_bit::ENCORE && !has_bp`).
/// 3. `Substitute` + `Baton Pass` receiver credit and overvalued idle/doomed `Substitute` write-down.
/// Returns identically `0.0` (`0.0 ULP`) on all states with no positive stat stages and no active `Substitute`.
#[inline]
pub fn compact_false_boost_refund_diff(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
) -> f64 {
    if cb.sides[0].alive_mask.count_ones() <= 1 && cb.sides[1].alive_mask.count_ones() <= 1 {
        return 0.0;
    }
    compact_side_false_boost_refund(cb, table, w, 0)
        - compact_side_false_boost_refund(cb, table, w, 1)
}

#[inline]
fn compact_side_false_boost_refund(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    s: usize,
) -> f64 {
    let side_st = &cb.sides[s];
    let act_slot = side_st.active_idx as usize;
    if (side_st.alive_mask & (1 << act_slot)) == 0 {
        return 0.0;
    }
    let act_idx = s * 3 + act_slot;
    let act_mon = &cb.mons[act_idx];
    if act_mon.hp() <= 0 {
        return 0.0;
    }
    let has_sub = (act_mon.vol_flags & vol_bit::SUBSTITUTE) != 0;
    // Fast path: if no active Substitute and no positive stat stages are present, adjustment is identically 0.0.
    if !has_sub
        && act_mon.boosts[0] <= 0
        && act_mon.boosts[1] <= 0
        && act_mon.boosts[2] <= 0
        && act_mon.boosts[3] <= 0
        && act_mon.boosts[4] <= 0
        && act_mon.boosts[5] <= 0
        && act_mon.boosts[6] <= 0
    {
        return 0.0;
    }

    let act_spec = &table.mons[act_idx];
    let mut delta = 0.0;
    let mut has_self_setup = false;
    let mut has_swagger = false;
    let mut has_any_phys = false;
    let mut has_bp = false;

    for m_i in 0..act_spec.num_moves as usize {
        let ms = &table.move_specs[act_idx][m_i];
        if ms.boosts_self
            || matches!(
                ms.special_kind,
                nc2000_engine::compact::SpecialMoveKind::Curse
                    | nc2000_engine::compact::SpecialMoveKind::BellyDrum
            )
        {
            has_self_setup = true;
        }
        if !ms.boosts_self
            && ms.boosts[0].0 == 0
            && ms.boosts[0].1 > 0
            && (ms.primary_vol & vol_bit::CONFUSION) != 0
        {
            has_swagger = true;
        }
        if act_mon.pp[m_i] == 0 || act_mon.is_slot_disabled(m_i) {
            continue;
        }
        let deals_dmg = !ms.is_status
            && (ms.base_power > 0
                || ms.var_bp_kind != nc2000_engine::compact::VarBpKind::Static
                || ms.fixed_damage != 0
                || ms.special_kind == nc2000_engine::compact::SpecialMoveKind::SuperFang);
        if deals_dmg && ms.is_physical {
            has_any_phys = true;
        }
        if ms.special_kind == nc2000_engine::compact::SpecialMoveKind::BatonPass
            && side_st.alive_mask.count_ones() >= 2
        {
            has_bp = true;
        }
    }

    let mut team_has_bp = has_bp;
    if !team_has_bp {
        for slot in 0..3usize {
            if (side_st.alive_mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = s * 3 + slot;
            if cb.mons[idx].hp() <= 0 {
                continue;
            }
            for m_i in 0..table.mons[idx].num_moves as usize {
                if table.move_specs[idx][m_i].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::BatonPass
                {
                    team_has_bp = true;
                    break;
                }
            }
        }
    }

    let mut atk_refunded = false;

    // 1. Swagger false-boost refund: only when `act_mon` has no self-setup/sustain move and no team Baton Pass,
    // so `boosts[0] > 0` came from an opponent's `Swagger`.
    if !has_self_setup && !team_has_bp && act_mon.boosts[0] > 0 {
        if !has_any_phys {
            delta -= w.boost[0] * (act_mon.boosts[0] as f64);
            atk_refunded = true;
        } else if (act_mon.vol_flags & vol_bit::CONFUSION) != 0 {
            delta -= 0.70 * w.boost[0] * (act_mon.boosts[0] as f64);
            atk_refunded = true;
        }
    }

    // 2. Encore lock on a boosted setup mon: positive boosts cannot be leveraged while locked
    // into a non-Baton-Pass move (prevents rewarding Encore-locked Double Team / Agility spam).
    if (act_mon.vol_flags & vol_bit::ENCORE) != 0 {
        let enc_slot = act_mon.encore_slot() as usize;
        let enc_is_bp = has_bp
            && enc_slot < (table.mons[act_idx].num_moves as usize)
            && table.move_specs[act_idx][enc_slot].special_kind
                == nc2000_engine::compact::SpecialMoveKind::BatonPass;
        if !enc_is_bp {
            let refund_mult = if has_bp { 0.80 } else { 0.50 };
            for i in 0..7 {
                if act_mon.boosts[i] > 0 {
                    if i == 0 && atk_refunded {
                        continue;
                    }
                    delta -= refund_mult * w.boost[i] * (act_mon.boosts[i] as f64);
                }
            }
        }
    }

    // 3. Active Substitute valuation adjustment:
    //    - Scale by remaining Substitute HP fraction (`sub_frac`).
    //    - Credit Baton Pass + Substitute chains (`Passer with Sub < Receiver with Sub`).
    //    - Discount doomed Substitutes (slower against a foe that breaks the Sub) and idle
    //      Substitutes on unboosted non-passers against passive/asleep targets so flat
    //      `w.substitute = 0.50` never creates a fake static profit over attacking/KOing.
    if has_sub && w.substitute > 0.0 {
        let sub_max = (act_spec.maxhp / 4).max(1) as f64;
        let sub_frac = (act_mon.sub_hp().max(1) as f64 / sub_max).clamp(0.25, 1.0);
        let foe_s = 1 - s;
        let foe_slot = cb.sides[foe_s].active_idx as usize;
        let foe_alive = (cb.sides[foe_s].alive_mask & (1 << foe_slot)) != 0
            && cb.mons[foe_s * 3 + foe_slot].hp() > 0;
        let (me_faster, foe_hit, foe_confused_or_par, foe_asleep) = if foe_alive {
            let f_idx = foe_s * 3 + foe_slot;
            let f_mon = &cb.mons[f_idx];
            let f_st = f_mon.status_raw();
            let f_hit = cb.best_hit_fraction(table, foe_s, foe_slot, s, act_slot);
            let f_cp = (f_mon.vol_flags & vol_bit::CONFUSION) != 0 || f_st == Status::Par as u8;
            let f_slp = f_st == Status::Slp as u8 && f_mon.status_counter() >= 2;
            (
                compare_action_speed(cb, table, s, act_slot, foe_s, foe_slot)
                    == std::cmp::Ordering::Greater,
                f_hit,
                f_cp,
                f_slp,
            )
        } else {
            (true, 0.0, false, false)
        };

        let mut best_bench_hit = 0.0f64;
        if team_has_bp && foe_alive {
            for slot in 0..3usize {
                if slot == act_slot || (side_st.alive_mask & (1 << slot)) == 0 {
                    continue;
                }
                let b_idx = s * 3 + slot;
                if cb.mons[b_idx].hp() <= 0 || cb.mons[b_idx].status_raw() == Status::Slp as u8 {
                    continue;
                }
                let bh = cb.best_hit_fraction(table, s, slot, foe_s, foe_slot);
                if bh > best_bench_hit {
                    best_bench_hit = bh;
                }
            }
        }
        let my_hit = if foe_alive {
            cb.best_hit_fraction(table, s, act_slot, foe_s, foe_slot)
        } else {
            0.5
        };

        if has_bp && best_bench_hit > 0.35 {
            // Passer holding an active Substitute with a ready offensive bench receiver:
            delta += 0.08 * sub_frac * best_bench_hit.clamp(0.35, 1.0);
        } else if !has_bp && team_has_bp && my_hit >= 0.45 {
            // Offensive receiver holding a Baton-Passed Substitute: strictly higher than on the passer!
            delta += 0.22 * sub_frac * my_hit.clamp(0.45, 1.1);
        } else if !has_bp && !has_self_setup {
            if !me_faster && !foe_asleep && !foe_confused_or_par && foe_hit >= 0.25 * sub_frac {
                // Slower than an unencumbered foe whose hit breaks the Sub before we move:
                delta -= 0.50 * w.substitute;
            } else if !(has_swagger && foe_confused_or_par) {
                // Unboosted non-passer without SwagSub lock: write down excess static bonus so
                // Substitute is valued near its 25% HP cost plus a small status-buffer margin.
                let chip_pen = (1.0 - sub_frac) * 0.30 * w.substitute;
                let idle_pen = if foe_asleep || foe_hit < 0.22 {
                    0.42 * w.substitute
                } else {
                    0.24 * w.substitute
                };
                delta -= (chip_pen + idle_pen).min(0.48 * w.substitute);
            }
        }
    }

    delta
}

/// Win-probability-shaped eval in `(0, 1)` from side 0's perspective on `CompactBattle`.
#[inline]
pub fn compact_eval01(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    if let Some(v) = compact_turn_limit_outcome(cb, table) {
        return v;
    }
    let s0 = compact_side_score(cb, table, w, 0);
    let s1 = compact_side_score(cb, table, w, 1);
    let mut diff = s0 - s1;
    let mut old_rm = 0.0;
    if w.race != 0.0 {
        old_rm = compact_race_margin(cb, table);
        diff += w.race * old_rm;
    }
    diff += crate::linear::compact_linear_diff(cb, table, diff, old_rm, w.race, w.scale);
    1.0 / (1.0 + (-(w.scale * diff)).exp())
}

#[inline]
fn compact_side_score(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    s: usize,
) -> f64 {
    let side_st = &cb.sides[s];
    let mut score = 0.0;
    let mut pp_num = 0.0;
    let mut pp_den = 0.0;
    let active_slot = side_st.active_idx;
    let mut spikes_exposed = 0u32;
    let flying_mask = 1u32 << (table.flying_type & 31);

    for &slot_u8 in &cb.party_order(s) {
        let slot = slot_u8 as usize;
        if (side_st.alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let idx = s * 3 + slot;
        let mon = &cb.mons[idx];
        let spec = &table.mons[idx];
        if mon.hp() <= 0 {
            continue;
        }
        if (slot as u8) != active_slot && (spec.type_mask & flying_mask) == 0 {
            spikes_exposed += 1;
        }
        score += w.alive + w.hp * (mon.hp() as f64) / (spec.maxhp.max(1) as f64);
        let st_raw = mon.status_raw();
        score -= if st_raw == Status::Brn as u8 {
            w.brn
        } else if st_raw == Status::Par as u8 {
            w.par
        } else if st_raw == Status::Slp as u8 {
            if table.ruleset.is_castle() {
                let t = mon.status_counter().clamp(0, 5);
                w.slp * (t as f64) / 4.0
            } else {
                let t = mon.status_counter().clamp(0, 4);
                w.slp * (t as f64) / 3.0
            }
        } else if st_raw == Status::Frz as u8 {
            w.frz
        } else if st_raw == Status::Psn as u8 {
            w.psn
        } else if st_raw == Status::Tox as u8 {
            w.tox
        } else {
            0.0
        };
        for m_i in 0..spec.num_moves as usize {
            pp_num += mon.pp[m_i] as f64;
            pp_den += spec.max_pp[m_i] as f64;
        }
    }
    if pp_den > 0.0 {
        score += w.pp * pp_num / pp_den;
    }
    if w.spikes != 0.0 && spikes_exposed > 0 {
        let layers = (side_st.side_flags & 0x03) as usize;
        if layers > 0 {
            const AMOUNTS: [f64; 4] = [0.0, 3.0, 4.0, 6.0];
            let frac = AMOUNTS[layers] / 24.0;
            score -= w.spikes * frac * (spikes_exposed as f64);
        }
    }
    let act_slot = active_slot as usize;
    if (side_st.alive_mask & (1 << act_slot)) != 0 {
        let act_idx = s * 3 + act_slot;
        let act_mon = &cb.mons[act_idx];
        if act_mon.hp() > 0 {
            for i in 0..7 {
                score += w.boost[i] * (act_mon.boosts[i] as f64);
            }
            if w.substitute != 0.0 && (act_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 {
                score += w.substitute;
            }
            let foe_s = 1 - s;
            let foe_slot = cb.sides[foe_s].active_idx as usize;
            if (cb.sides[foe_s].alive_mask & (1 << foe_slot)) != 0
                && cb.mons[foe_s * 3 + foe_slot].hp() > 0
            {
                let act_eff = cb.best_hit_fraction(table, s, act_slot, foe_s, foe_slot);
                score += w.threat * act_eff.min(1.0);
            }
        }
    }
    score
}

#[inline]
pub fn compact_race_margin(cb: &CompactBattle, table: &MatchTable) -> f64 {
    if cb.sides[0].alive_mask.count_ones() != 1 || cb.sides[1].alive_mask.count_ones() != 1 {
        return 0.0;
    }
    let slot0 = cb.sides[0].active_idx as usize;
    let slot1 = cb.sides[1].active_idx as usize;
    let idx0 = slot0;
    let idx1 = 3 + slot1;
    let p0 = &cb.mons[idx0];
    let p1 = &cb.mons[idx1];
    if p0.hp() <= 0 || p1.hp() <= 0 {
        return 0.0;
    }

    let spd_ord = compare_action_speed(cb, table, 0, slot0, 1, slot1);

    let turns = |att_s: usize, att_slot: usize, def_s: usize, def_slot: usize, att_faster: bool| -> f64 {
        let e = cb.best_hit_fraction(table, att_s, att_slot, def_s, def_slot);
        let att_idx = att_s * 3 + att_slot;
        let def_idx = def_s * 3 + def_slot;
        let att_mon = &cb.mons[att_idx];
        let def_mon = &cb.mons[def_idx];
        let def_spec = &table.mons[def_idx];
        if e <= 1e-9 {
            return f64::INFINITY;
        }
        let mut has_heal = false;
        for m_i in 0..def_spec.num_moves as usize {
            if def_mon.pp[m_i] > 0
                && !def_mon.is_slot_disabled(m_i)
                && (table.move_specs[def_idx][m_i].is_heal_cycle
                    || table.move_specs[def_idx][m_i].is_rest)
            {
                has_heal = true;
                break;
            }
        }
        if has_heal {
            let dmg_frac_max = e * (def_mon.hp() as f64) / (def_spec.maxhp.max(1) as f64);
            let kill_now = e >= 1.0 && att_faster;
            if dmg_frac_max < 0.5 && !kill_now {
                return f64::INFINITY;
            }
        }
        let mut t = (1.0 / e).ceil();
        if (def_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 {
            t += 1.0;
        }
        if att_mon.must_recharge() {
            t += 1.0;
        }
        let st_raw = att_mon.status_raw();
        if st_raw == Status::Slp as u8 {
            let att_spec = &table.mons[att_idx];
            let att_ispec = table.item_spec(att_mon.item);
            if !(att_ispec.is_mintberry || att_ispec.is_miracleberry) {
                let mut talks = false;
                for m_i in 0..att_spec.num_moves as usize {
                    if att_mon.pp[m_i] > 0
                        && table.move_specs[att_idx][m_i].special_kind
                            == nc2000_engine::compact::SpecialMoveKind::SleepTalk
                    {
                        talks = true;
                        break;
                    }
                }
                if !talks {
                    let max_slp = if table.ruleset.is_castle() { 5 } else { 4 };
                    t += att_mon.status_counter().clamp(1, max_slp) as f64;
                }
            }
        } else if st_raw == Status::Frz as u8 {
            t += 4.0;
        }
        t
    };

    let surv = |s: usize, slot: usize| -> f64 {
        let idx = s * 3 + slot;
        let p = &cb.mons[idx];
        let spec = &table.mons[idx];
        let st = p.status_raw();
        if st != Status::Psn as u8 && st != Status::Tox as u8 && st != Status::Brn as u8 {
            return f64::INFINITY;
        }
        let hp = p.hp() as f64;
        let tick = (spec.maxhp as f64 / 8.0).max(1.0);
        (hp / tick).ceil()
    };

    let (k0, k1) = (
        turns(0, slot0, 1, slot1, spd_ord == std::cmp::Ordering::Greater),
        turns(1, slot1, 0, slot0, spd_ord == std::cmp::Ordering::Less),
    );
    let (v0, v1) = (surv(0, slot0), surv(1, slot1));
    let k0 = if k0 <= v0 { k0 } else { f64::INFINITY };
    let k1 = if k1 <= v1 { k1 } else { f64::INFINITY };
    let t0 = k0.min(v1);
    let t1 = k1.min(v0);
    if t0.min(t1) > 3.0 {
        return 0.0;
    }
    let diff = match (t0.is_finite(), t1.is_finite()) {
        (true, false) => 2.0,
        (false, true) => -2.0,
        _ => {
            let edge = match spd_ord {
                std::cmp::Ordering::Greater => 0.5,
                std::cmp::Ordering::Less => -0.5,
                std::cmp::Ordering::Equal => 0.0,
            };
            (t1 - t0) + edge
        }
    };
    diff.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;

    fn make_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "name": species, "species": species, "item": item, "ability": "No Ability",
            "moves": moves, "level": level,
            "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
            "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
            "happiness": 255
        }))
        .unwrap()
    }

    #[test]
    fn baton_pass_step_switch_transfers_boosts() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let p1_team = vec![
            make_set(
                "Scizor",
                50,
                "Miracle Berry",
                &["Swords Dance", "Baton Pass", "Agility", "Hidden Power"],
            ),
            make_set(
                "Marowak",
                55,
                "Thick Club",
                &["Earthquake", "Rock Slide", "Swords Dance", "Rest"],
            ),
            make_set(
                "Starmie",
                50,
                "Leftovers",
                &["Surf", "Psychic", "Recover", "Thunder Wave"],
            ),
        ];
        let p2_team = vec![
            make_set(
                "Snorlax",
                55,
                "Leftovers",
                &["Double-Edge", "Earthquake", "Rest", "Sleep Talk"],
            ),
            make_set(
                "Skarmory",
                50,
                "Leftovers",
                &["Drill Peck", "Whirlwind", "Rest", "Curse"],
            ),
            make_set(
                "Misdreavus",
                50,
                "Leftovers",
                &["Mean Look", "Perish Song", "Protect", "Thunder"],
            ),
        ];

        let mut b_ab = Battle::from_fixture(&dex, "1,2,3,4", &p1_team, &p2_team).unwrap();
        b_ab.choose(&dex, 0, "team 1,2,3").unwrap();
        b_ab.choose(&dex, 1, "team 1,2,3").unwrap();
        b_ab.choose(&dex, 0, "move 1").unwrap(); // Scizor Swords Dance (+2 Atk)
        b_ab.choose(&dex, 1, "move 3").unwrap(); // Snorlax Rest

        // Verify mid-turn Baton Pass switch_flag in CompactBattle::step_switch transfers boosts.
        b_ab.choose(&dex, 0, "move 2").unwrap(); // Scizor Baton Pass -> switch_flag request
        b_ab.choose(&dex, 1, "move 4").unwrap();
        let (tbl_bp, mut cb_bp) = CompactBattle::from_battle(&b_ab, &dex);
        assert_eq!(cb_bp.sides[0].request, 2);
        cb_bp.step_switch(&tbl_bp, [5, 7]); // switch 2 (Marowak)
        assert_eq!(cb_bp.sides[0].active_idx, 1);
        assert_eq!(
            cb_bp.mons[1].boosts[0], 2,
            "Baton Pass step_switch must transfer +2 Attack boost to Marowak"
        );
    }

    #[test]
    fn blindspot_leaf_antisymmetry_1v1_invariance_and_ledger_signs() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let w = EvalWeights::default();
        let p1_team = vec![
            make_set(
                "Snorlax",
                55,
                "Leftovers",
                &["Double-Edge", "Curse", "Rest", "Sleep Talk"],
            ),
            make_set(
                "Starmie",
                50,
                "Leftovers",
                &["Surf", "Psychic", "Recover", "Thunder Wave"],
            ),
            make_set(
                "Miltank",
                50,
                "Leftovers",
                &["Body Slam", "Milk Drink", "Heal Bell", "Growl"],
            ),
        ];
        let p2_team = vec![
            make_set(
                "Zapdos",
                55,
                "Leftovers",
                &["Thunderbolt", "Hidden Power", "Rest", "Sleep Talk"],
            ),
            make_set(
                "Skarmory",
                50,
                "Leftovers",
                &["Drill Peck", "Whirlwind", "Rest", "Curse"],
            ),
            make_set(
                "Misdreavus",
                50,
                "Leftovers",
                &["Mean Look", "Perish Song", "Protect", "Thunder"],
            ),
        ];

        let mut b_ab = Battle::from_fixture(&dex, "1,2,3,4", &p1_team, &p2_team).unwrap();
        b_ab.choose(&dex, 0, "team 1,2,3").unwrap();
        b_ab.choose(&dex, 1, "team 1,2,3").unwrap();
        b_ab.choose(&dex, 0, "move 2").unwrap(); // Snorlax Curse (+1 Atk, +1 Def, -1 Spe)
        b_ab.choose(&dex, 1, "move 3").unwrap(); // Zapdos Rest

        let mut b_ba = Battle::from_fixture(&dex, "1,2,3,4", &p2_team, &p1_team).unwrap();
        b_ba.choose(&dex, 0, "team 1,2,3").unwrap();
        b_ba.choose(&dex, 1, "team 1,2,3").unwrap();
        b_ba.choose(&dex, 0, "move 3").unwrap();
        b_ba.choose(&dex, 1, "move 2").unwrap();

        let (tbl_ab, mut cb_ab) = CompactBattle::from_battle(&b_ab, &dex);
        let (tbl_ba, cb_ba) = CompactBattle::from_battle(&b_ba, &dex);

        // 1. Exact side-swap antisymmetry on compact_eval01_blindspot
        let v_ab = compact_eval01_blindspot(&cb_ab, &tbl_ab, &w);
        let v_ba = compact_eval01_blindspot(&cb_ba, &tbl_ba, &w);
        assert!(
            (v_ab + v_ba - 1.0).abs() < 1e-12,
            "compact_eval01_blindspot must sum to 1.0 across side swap: v_ab={v_ab}, v_ba={v_ba}"
        );

        // 2. Encore lock false-boost refund on boosted setup mon without Baton Pass
        cb_ab.sides[0].alive_mask = 0b011;
        cb_ab.sides[1].alive_mask = 0b011;
        cb_ab.mons[0].boosts = [0; 7];
        cb_ab.mons[0].boosts[0] = 2;
        cb_ab.mons[0].vol_flags |= vol_bit::ENCORE;
        let refund_encore = compact_false_boost_refund_diff(&cb_ab, &tbl_ab, &w);
        assert!(
            (refund_encore - (-0.50 * 2.0 * w.boost[0])).abs() < 1e-12,
            "Encored boosted Snorlax without Baton Pass must refund 50% of +2 Atk: {refund_encore}"
        );
        cb_ab.mons[0].vol_flags &= !vol_bit::ENCORE;
        cb_ab.mons[0].boosts[0] = 0;

        // 3. Exact 1v1 invariance (`compact_eval01_blindspot == compact_eval01`)
        cb_ab.sides[0].alive_mask = 0b001;
        cb_ab.sides[1].alive_mask = 0b001;
        cb_ab.mons[0].boosts[0] = 2;
        cb_ab.mons[0].boosts[1] = 2;
        assert_eq!(
            compact_false_boost_refund_diff(&cb_ab, &tbl_ab, &w),
            0.0,
            "1v1 states must have zero blindspot adjustment"
        );
        assert_eq!(
            compact_eval01_blindspot(&cb_ab, &tbl_ab, &w),
            compact_eval01(&cb_ab, &tbl_ab, &w),
            "compact_eval01_blindspot must equal compact_eval01 in 1v1"
        );
    }
}

