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

use nc2000_engine::dex::{Dex, MoveId, TypeId};
use nc2000_engine::state::{Battle, PokeId, Status};

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
    /// Fold the gen-2 accuracy×evasion stage multipliers into the threat
    /// feature (`Battle::hit_probability`) so a boosted-evasion foe collapses
    /// the bot's estimated hit chance — the physically-correct danger channel
    /// (Double Team / Baton Pass). Always true in shipped play; the tests flip
    /// it to reproduce the pre-fix eval (base accuracy only, blind to evasion).
    pub couple_evasion: bool,
    /// M17 candidates (measured leads from the M16a calibration; not in the
    /// SPSA vector — gate via eval_calibration --ab before changing defaults):
    /// scale the slp penalty by the engine's remaining sleep clock
    /// (`DK::Time`/3) so a 2-turn Rest nap costs less than a full enemy
    /// sleep (slp slice bias −0.17 = flat penalty too heavy)...
    pub slp_time_scale: bool,
    /// ...and a flat bonus while the active has a Substitute up (sub slice
    /// bias −0.31 = the paid-for sub is invisible to the eval). 0.0 = off.
    pub substitute: f64,
    /// KO-race term (M17c). When BOTH sides are on their last living mon —
    /// a pure race, no switch outs — the side that KOs first wins
    /// regardless of remaining HP, which the material terms cannot see:
    /// both threat features saturate at 1.0 and cancel while the HP terms
    /// vote for the fatter side. Motivated by the 570-corpus certified
    /// anchors (47 zero-playout proven violations; worst = ordering
    /// inversions like eval 0.127 at a certified 1.000 win). Adds
    /// `race × margin` to the side differential, where margin ∈ [-1, 1]
    /// grades turns-to-KO advantage + speed order, and only short races
    /// (≤3 turns to the faster kill) count — longer races are stall/heal
    /// territory where the estimate lies. 0.0 = off.
    pub race: f64,
    /// Search-cutoff backup around 0.5. M6 used 0.5, compressing `eval01`
    /// into (0.25, 0.75); M17c uses 1.0 to retain probability semantics.
    pub leaf_alpha: f64,
    /// Phase A exchange term: `exchange_margin` over all living pairs. 0.0 =
    /// off (shipped default until it is calibrated and duelled).
    pub exchange: f64,
    /// Entry-hazard term (M17 tail): Spikes on the owning side. `eval01` has
    /// no side-condition channel at all today, and Spikes is the most common
    /// condition in the 570-battle spectator corpus (21.4% turn-weighted), so
    /// every position it decides currently reads as material-neutral.
    ///
    /// The cost is not paid by the mon already in — it is paid once by each
    /// living BENCHED mon when it next switches in, and Flying types never
    /// pay it. Mirrors `conditions.rs` exactly: Flying immune, damage =
    /// `AMOUNTS[layers] / 24` of max HP with `AMOUNTS = [0, 3, 4, 6]`. This
    /// format never stacks past one layer (a re-cast is rejected), so the
    /// live value is always 3/24 = 1/8, but the layer count is read rather
    /// than assumed so the term stays correct if that ever changes.
    ///
    /// Scaled like `hp`, so the weight reads as **the expected number of
    /// Spikes triggers per exposed benched mon**. 1.0 = each benched mon
    /// eats the hazard exactly once. Values above 1.0 are the expected case,
    /// not an overshoot: one Spikes pays out on *every* switch-in for the
    /// rest of the game, and mons re-enter. `party` holds the picked three
    /// after preview (`turn.rs` rebuilds it at the Team action), so at most
    /// two mons are ever exposed — a weight of 2.8 is "~5.6 triggers per
    /// game", which is unremarkable across the corpus's 21.4-turn average.
    /// Shipped at **1.5** (2026-07-25). The corpus calibration put the bias
    /// zero-crossing there and r/Brier/MSE optimise at the same weight, and
    /// the seed-paired arena gate came back parity — 0.530±0.046 at 300
    /// iters, 0.498±0.047 at 1000, i.e. the 300-iter edge did not replicate,
    /// at identical think time. It ships on the Rev-1 bar (better calibration
    /// + strength parity + no cost) plus an explicit product judgement: at
    /// equal strength, a bot that visibly understands the hazard reads as
    /// competent to the human across the board, and behaving the way a human
    /// expects is worth more to this product than a strength delta it cannot
    /// measure.
    pub spikes: f64,
    /// Confusion (M17 tail): the second condition the corpus named and the
    /// eval had no channel for at all (8.6% turn-weighted, behind Spikes at
    /// 21.4% which now has one). Active-only, because the volatile dies on
    /// switch-out.
    ///
    /// Mechanically (`conditions.rs` "confusion"): the clock is drawn
    /// `random_range(2, 6)` and decremented *before* each move attempt; when
    /// it reaches 0 the volatile is removed and that move goes through
    /// unhindered. So a stored clock of `t` buys the foe `t - 1` more
    /// hindered attempts, each a 1-in-2 coin for a 40-BP typeless self-hit
    /// instead of the move. The penalty is therefore
    /// `confusion × 0.5 × (t - 1)` and the weight reads as **the cost of one
    /// expected lost turn** — at onset (t ∈ 2..=5, mean 3.5) that is
    /// `1.25 × confusion`. Deliberately not a flat constant: the flat status
    /// penalties are exactly what the corpus calibration measured as
    /// mispriced. 0.0 = off (shipped default until calibrated and duelled).
    pub confusion: f64,
    /// **Phase B of the exchange eval — "the new scheme".** Off by default;
    /// `true` re-reads the whole exchange with the mechanics the additive
    /// terms flatten, and takes the *game value* of the pair matrix instead
    /// of Phase A's maximin/minimax midpoint:
    ///
    /// - the matrix is solved with `smmcts::solve_rm_plus` (already in the
    ///   tree, microseconds at 3×3) rather than proxied, so a matchup that
    ///   only pays off under a mixed choice is priced instead of being
    ///   rounded to the committed side;
    /// - **entry costs enter the matrix**: a pair using a *benched* mon is
    ///   charged the switch turn and its Spikes entry damage, so an
    ///   off-diagonal cell is worth what the switch actually costs, and a mon
    ///   that cannot survive its own entry loses the pairing outright. This
    ///   is the first channel through which switching reaches `eval01` at
    ///   all;
    /// - **residual clocks follow the engine**: `conditions.rs::residualdmg`
    ///   ticks `floor(maxhp/16) × counter` with the counter incrementing every
    ///   turn once the toxic volatile is on the mon (gen 2 keeps it for
    ///   psn/brn too), not the flat `maxhp/8` Phase A assumed. Toxic's value
    ///   then grows with its counter by construction — the corpus
    ///   calibration measured the flat `tox` weight as the eval's largest
    ///   oriented bias (−0.094);
    /// - **confusion enters as lost offense**: a confused attacker's expected
    ///   damage per turn halves, so it needs about twice as many turns to
    ///   close the same race.
    ///
    /// Because `race_margin` is the 1×1 case of the same function, this flag
    /// changes the shipped race term too even when `exchange` is 0.0 — which
    /// is why it defaults to `false` and ships inert until the standard gate
    /// (`eval_calibration --corpus` → candidate screen → seed-paired duel)
    /// clears it. `EvalWeights::exchange_scheme` assembles the full candidate.
    pub exchange_v2: bool,
}

impl Default for EvalWeights {
    /// M6 hand-written starting point, revised 2026-07-21 by the M16a/M17
    /// paired calibration (eval_calibration --ab, 600 positions x 32
    /// playouts, same GT): slp/frz/tox x0.7 + sleep-clock scaling +
    /// substitute bonus improved r 0.681->0.709, Brier 0.194->0.189 and
    /// halved the slp/sub/frz oriented biases, at strength parity in
    /// seed-paired duels (0.485+/-0.069 @300, 0.500+/-0.098 @1000).
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
            couple_evasion: true,
            slp_time_scale: true,
            substitute: 0.5,
            race: 3.0,
            leaf_alpha: 1.0,
            spikes: 1.5,
            exchange: 0.0,
            confusion: 0.0,
            exchange_v2: false,
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
            // Non-vector features follow the shipped defaults (the SPSA
            // vector carries only the linear weights).
            couple_evasion: true,
            slp_time_scale: true,
            substitute: 0.5,
            race: 3.0,
            leaf_alpha: 1.0,
            spikes: 1.5,
            exchange: 0.0,
            confusion: 0.0,
            exchange_v2: false,
        }
    }

    /// The Phase B candidate as one object, so every harness arm means the
    /// same thing. `exchange` is the weight on the matrix game value; `damp`
    /// scales the additive terms the exchange now carries itself (the status
    /// penalties, Substitute, Spikes) — `0.0` moves them out entirely, `1.0`
    /// leaves them double-counted, and the sweep looks for the middle.
    ///
    /// `race` is set to 0.0 unconditionally: the exchange matrix is a strict
    /// generalization of the race term (identical at 1×1), so keeping both
    /// pays for the same last-mon race twice. The material terms (`hp`,
    /// `alive`, `pp`, `boost`, `threat`) stay: the exchange returns 0 — "no
    /// claim" — on stall/heal positions, and they are what is left when it
    /// does.
    pub fn exchange_scheme(exchange: f64, damp: f64) -> Self {
        let d = EvalWeights::default();
        EvalWeights {
            exchange,
            exchange_v2: true,
            race: 0.0,
            brn: d.brn * damp,
            par: d.par * damp,
            slp: d.slp * damp,
            frz: d.frz * damp,
            psn: d.psn * damp,
            tox: d.tox * damp,
            substitute: d.substitute * damp,
            spikes: d.spikes * damp,
            ..d
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
pub fn race_margin(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    compact_race_margin(&cb, &table, w)
}

/// The exchange between ONE pair of mons, from `id0`'s perspective, in
/// [-1, 1]: who wins the straight trade and by how much, through the visible
/// mechanics (heal cycles, Substitute, recharge, sleep/freeze downtime,
/// residual self-death clocks, speed order). Returns 0 — "no claim" — when the
/// race runs past three turns, which is stall/heal territory where a
/// turns-to-KO estimate lies.
pub fn pair_margin(b: &Battle, dex: &Dex, w: &EvalWeights, id0: PokeId, id1: PokeId) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    let s0 = id0.side as usize;
    let s1 = id1.side as usize;
    let Some(slot0) = (0..3usize).find(|&i| cb.sides[s0].party_roster[i] == id0.slot) else {
        return 0.0;
    };
    let Some(slot1) = (0..3usize).find(|&i| cb.sides[s1].party_roster[i] == id1.slot) else {
        return 0.0;
    };
    compact_pair_margin_ctx(&cb, &table, w, s0, slot0, s1, slot1, [1.0, 1.0], [0.0, 0.0])
}

/// Phase A / B of the exchange eval: the same trade computation as `race_margin`,
/// read over EVERY living pair instead of only a last-mon-vs-last-mon race.
pub fn exchange_margin(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    compact_exchange_margin(&cb, &table, w)
}

/// The exchange matrix behind `exchange_margin`: side 0's living mons as rows,
/// side 1's as columns, cells row-major in margin space [-1, 1]. `None` when
/// either side has nothing living.
pub fn exchange_matrix(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
) -> Option<(Vec<PokeId>, Vec<PokeId>, Vec<f64>)> {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    let mut mine = Vec::with_capacity(3);
    let mut mine_slots = Vec::with_capacity(3);
    let mut theirs = Vec::with_capacity(3);
    let mut their_slots = Vec::with_capacity(3);
    for slot in 0..3usize {
        if (cb.sides[0].alive_mask & (1 << slot)) != 0 && cb.mons[slot].hp() > 0 {
            mine.push(PokeId { side: 0, slot: cb.sides[0].party_roster[slot] });
            mine_slots.push(slot);
        }
        if (cb.sides[1].alive_mask & (1 << slot)) != 0 && cb.mons[3 + slot].hp() > 0 {
            theirs.push(PokeId { side: 1, slot: cb.sides[1].party_roster[slot] });
            their_slots.push(slot);
        }
    }
    if mine.is_empty() || theirs.is_empty() {
        return None;
    }
    let flying_mask = 1u32 << (table.flying_type & 31);
    let entry_left = |s: usize, slot: usize| -> f64 {
        if !w.exchange_v2 || (slot as u8) == cb.sides[s].active_idx {
            return 1.0;
        }
        let idx = s * 3 + slot;
        if (table.mons[idx].type_mask & flying_mask) != 0 {
            return 1.0;
        }
        let layers = (cb.sides[s].side_flags & 0x03) as usize;
        if layers == 0 {
            return 1.0;
        }
        const AMOUNTS: [f64; 4] = [0.0, 3.0, 4.0, 6.0];
        let dmg = (AMOUNTS[layers] * (table.mons[idx].maxhp as f64) / 24.0)
            .floor()
            .max(1.0);
        1.0 - dmg / (cb.mons[idx].hp().max(1) as f64)
    };
    let extra = |s: usize, slot: usize| -> f64 {
        if w.exchange_v2 && (slot as u8) != cb.sides[s].active_idx {
            1.0
        } else {
            0.0
        }
    };
    let mut cells = Vec::with_capacity(mine.len() * theirs.len());
    for &a in &mine_slots {
        let la = entry_left(0, a);
        let ea = extra(0, a);
        for &d in &their_slots {
            cells.push(compact_pair_margin_ctx(
                &cb,
                &table,
                w,
                0,
                a,
                1,
                d,
                [la, entry_left(1, d)],
                [ea, extra(1, d)],
            ));
        }
    }
    Some((mine, theirs, cells))
}

/// RM+ sweeps for the exchange matrix.
const EXCHANGE_SWEEPS: u32 = 128;

/// Phase B's aggregation: the zero-sum **game value** of the pair matrix,
/// instead of Phase A's maximin/minimax midpoint.
fn matrix_game_value(cells: &[f64], k: [usize; 2]) -> f64 {
    let m: Vec<f64> = cells.iter().map(|v| 0.5 * (v + 1.0)).collect();
    let (s0, s1) = crate::smmcts::solve_rm_plus(&m, k, EXCHANGE_SWEEPS);
    let mut u = 0.0;
    for i in 0..k[0] {
        for j in 0..k[1] {
            u += s0[i] * s1[j] * m[i * k[1] + j];
        }
    }
    2.0 * u - 1.0
}

/// Leaf value for search cutoffs. The calibrated probability is backed up
/// directly at alpha 1; smaller alpha shrinks it toward 0.5.
pub fn eval_leaf(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    compact_eval_leaf(&cb, &table, w)
}

/// Best expected hit fraction over the attacker's usable move slots
/// (unclamped: >1 = expected OHKO with margin). `couple_evasion` folds the
/// gen-2 accuracy/evasion stage multipliers into each move's hit chance.
pub fn best_hit_fraction(
    b: &Battle,
    dex: &Dex,
    att: PokeId,
    def: PokeId,
    couple_evasion: bool,
) -> f64 {
    let mut best = 0.0f64;
    for ms in b.poke(att).move_slots.iter() {
        if ms.pp <= 0 || ms.disabled {
            continue;
        }
        best = best.max(expected_hit_fraction(b, dex, att, def, ms.id, couple_evasion));
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
/// mean roll x mean hits x accuracy. 0 for status moves and unknowable
/// callback damage (counter/mirrorcoat/bide/ohko score 0).
///
/// `couple_evasion` picks the accuracy channel: `true` (shipped) uses
/// `Battle::hit_probability` — the real gen-2 accuracy×evasion stage roll, so
/// a boosted-evasion foe collapses the estimate; `false` reproduces the
/// pre-fix behavior (base move accuracy only, blind to evasion) for the tests'
/// before/after contrast.
pub fn expected_hit_fraction(
    b: &Battle,
    dex: &Dex,
    att: PokeId,
    def: PokeId,
    move_id: MoveId,
    couple_evasion: bool,
) -> f64 {
    let dt = nc2000_engine::compact::dex_tables(dex);
    let mspec = &dt.move_specs[move_id.0 as usize];
    let a = b.poke(att);
    let d = b.poke(def);

    let (move_type, is_physical, is_status, base_power) = match mspec.var_bp_kind {
        1 => (
            TypeId(mspec.move_type),
            mspec.is_physical,
            mspec.is_status,
            a.happiness as i32 * 10 / 25,
        ),
        2 => (
            TypeId(mspec.move_type),
            mspec.is_physical,
            mspec.is_status,
            (255 - a.happiness as i32) * 10 / 25,
        ),
        3 => {
            let hpt = a.hp_type;
            let is_phys = (dt.special_type_mask & (1u32 << (hpt.0 & 31))) == 0;
            (hpt, is_phys, false, a.hp_power)
        }
        4 => {
            let ratio = ((a.hp as f64 * 48.0 / a.maxhp as f64).floor() as i32).max(1);
            let bp = if ratio < 2 {
                200
            } else if ratio < 5 {
                150
            } else if ratio < 10 {
                100
            } else if ratio < 17 {
                80
            } else if ratio < 33 {
                40
            } else {
                20
            };
            (TypeId(mspec.move_type), mspec.is_physical, mspec.is_status, bp)
        }
        _ => (
            TypeId(mspec.move_type),
            mspec.is_physical,
            mspec.is_status,
            mspec.base_power as i32,
        ),
    };

    if mspec.fixed_damage == 0 && (is_status || base_power <= 0) {
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

    let acc = if couple_evasion {
        nc2000_engine::compact::hit_probability_branchless(
            mspec.base_acc_pct,
            mspec.always_hit,
            a.boosts[5],
            d.boosts[6],
            mspec.ignore_accuracy,
            mspec.ignore_evasion,
            mspec.ignore_positive_evasion,
        )
    } else if mspec.always_hit {
        1.0
    } else {
        mspec.base_acc_pct as f64 / 100.0
    };

    let raw = if mspec.fixed_damage != 0 {
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
        let core = ((a.level as f64 * 2.0 / 5.0 + 2.0).floor() * base_power as f64 * atk
            / defense
            / 50.0)
            .floor()
            + 2.0;
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

/// Win-probability-shaped eval in `(0, 1)` from side 0's perspective on `CompactBattle`.
#[inline]
pub fn compact_eval01(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    let mut diff = compact_side_score(cb, table, w, 0) - compact_side_score(cb, table, w, 1);
    if w.race != 0.0 {
        diff += w.race * compact_race_margin(cb, table, w);
    }
    if w.exchange != 0.0 {
        diff += w.exchange * compact_exchange_margin(cb, table, w);
    }
    1.0 / (1.0 + (-w.scale * diff).exp())
}

#[inline]
fn compact_side_score(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights, s: usize) -> f64 {
    let side_st = &cb.sides[s];
    let mut score = 0.0;
    let mut pp_num = 0.0;
    let mut pp_den = 0.0;
    let active_slot = side_st.active_idx;
    let mut spikes_exposed = 0u32;
    let flying_mask = 1u32 << (table.flying_type & 31);

    for slot in 0..3usize {
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
            if w.slp_time_scale {
                let t = mon.status_counter().clamp(0, 4);
                w.slp * (t as f64) / 3.0
            } else {
                w.slp
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
            if w.confusion != 0.0 && (act_mon.vol_flags & vol_bit::CONFUSION) != 0 {
                let att = (act_mon.timers[0].clamp(0, 5) as f64 - 1.0).max(0.0);
                score -= w.confusion * 0.5 * att;
            }
            let foe_s = 1 - s;
            let foe_slot = cb.sides[foe_s].active_idx as usize;
            if (cb.sides[foe_s].alive_mask & (1 << foe_slot)) != 0
                && cb.mons[foe_s * 3 + foe_slot].hp() > 0
            {
                score += w.threat
                    * cb.best_hit_fraction(table, s, act_slot, foe_s, foe_slot, w.couple_evasion)
                        .min(1.0);
            }
        }
    }
    score
}

#[inline]
pub fn compact_race_margin(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    if cb.sides[0].alive_mask.count_ones() != 1 || cb.sides[1].alive_mask.count_ones() != 1 {
        return 0.0;
    }
    let slot0 = cb.sides[0].active_idx as usize;
    let slot1 = cb.sides[1].active_idx as usize;
    compact_pair_margin_ctx(cb, table, w, 0, slot0, 1, slot1, [1.0, 1.0], [0.0, 0.0])
}

#[allow(clippy::too_many_arguments)]
fn compact_pair_margin_ctx(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    s0: usize,
    slot0: usize,
    s1: usize,
    slot1: usize,
    hp_left: [f64; 2],
    extra: [f64; 2],
) -> f64 {
    let idx0 = s0 * 3 + slot0;
    let idx1 = s1 * 3 + slot1;
    let p0 = &cb.mons[idx0];
    let p1 = &cb.mons[idx1];
    if p0.hp() <= 0 || p1.hp() <= 0 {
        return 0.0;
    }
    match (hp_left[0] <= 0.0, hp_left[1] <= 0.0) {
        (true, true) => return 0.0,
        (true, false) => return -1.0,
        (false, true) => return 1.0,
        _ => {}
    }

    let spd0 = cb.mon_action_speed(table, s0, slot0);
    let spd1 = cb.mon_action_speed(table, s1, slot1);

    let turns = |att_s: usize,
                 att_slot: usize,
                 def_s: usize,
                 def_slot: usize,
                 att_faster: bool,
                 def_left: f64,
                 att_extra: f64|
     -> f64 {
        let mut e =
            cb.best_hit_fraction(table, att_s, att_slot, def_s, def_slot, w.couple_evasion);
        let att_idx = att_s * 3 + att_slot;
        let def_idx = def_s * 3 + def_slot;
        let att_mon = &cb.mons[att_idx];
        let def_mon = &cb.mons[def_idx];
        let def_spec = &table.mons[def_idx];
        if w.exchange_v2
            && (att_mon.vol_flags & vol_bit::CONFUSION) != 0
            && att_mon.timers[0] > 1
        {
            e *= 0.5;
        }
        if e <= 1e-9 {
            return f64::INFINITY;
        }
        let def_dis = (def_mon.vol_flags & vol_bit::DISABLE) != 0;
        let def_dis_slot = ((def_mon.slot_refs >> 4) & 0x03) as usize;
        let mut has_heal = false;
        for m_i in 0..def_spec.num_moves as usize {
            if def_mon.pp[m_i] > 0
                && !(def_dis && m_i == def_dis_slot)
                && table.move_specs[def_idx][m_i].is_heal_cycle
            {
                has_heal = true;
                break;
            }
        }
        if has_heal {
            let dmg_frac_max = e * (def_mon.hp() as f64) / (def_spec.maxhp.max(1) as f64);
            let kill_now = e >= def_left && att_faster;
            if dmg_frac_max < 0.5 && !kill_now {
                return f64::INFINITY;
            }
        }
        let mut t = (def_left / e).ceil() + att_extra;
        if (def_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 {
            t += 1.0;
        }
        if (att_mon.slot_refs & 0x80) != 0 {
            t += 1.0;
        }
        let st_raw = att_mon.status_raw();
        if st_raw == Status::Slp as u8 {
            let att_spec = &table.mons[att_idx];
            let mut talks = false;
            for m_i in 0..att_spec.num_moves as usize {
                if att_mon.pp[m_i] > 0 && table.move_specs[att_idx][m_i].special_kind == 10 {
                    talks = true;
                    break;
                }
            }
            if !talks {
                t += att_mon.status_counter().clamp(1, 4) as f64;
            }
        } else if st_raw == Status::Frz as u8 {
            t += 4.0;
        }
        t
    };

    let surv = |s: usize, slot: usize, left: f64| -> f64 {
        let idx = s * 3 + slot;
        let p = &cb.mons[idx];
        let spec = &table.mons[idx];
        let st = p.status_raw();
        if st != Status::Psn as u8 && st != Status::Tox as u8 && st != Status::Brn as u8 {
            return f64::INFINITY;
        }
        let hp = (p.hp() as f64) * left;
        if !w.exchange_v2 {
            let tick = (spec.maxhp as f64 / 8.0).max(1.0);
            return (hp / tick).ceil();
        }
        let cnt_packed = p.status_counter();
        if (cnt_packed & 0x10) != 0 {
            let c = (cnt_packed & 0x0F) as i64;
            let unit = (spec.maxhp as f64 / 16.0).floor().max(1.0);
            let mut acc = 0.0;
            for k in 1..=8u32 {
                acc += unit * (c + k as i64) as f64;
                if acc >= hp {
                    return k as f64;
                }
            }
            f64::INFINITY
        } else {
            let tick = (spec.maxhp as f64 / 8.0).floor().max(1.0);
            (hp / tick).ceil()
        }
    };

    let (k0, k1) = (
        turns(s0, slot0, s1, slot1, spd0 > spd1, hp_left[1], extra[0]),
        turns(s1, slot1, s0, slot0, spd1 > spd0, hp_left[0], extra[1]),
    );
    let (v0, v1) = (surv(s0, slot0, hp_left[0]), surv(s1, slot1, hp_left[1]));
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
            let edge = match spd0.cmp(&spd1) {
                std::cmp::Ordering::Greater => 0.5,
                std::cmp::Ordering::Less => -0.5,
                std::cmp::Ordering::Equal => 0.0,
            };
            (t1 - t0) + edge
        }
    };
    diff.clamp(-1.0, 1.0)
}

fn compact_exchange_margin(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    let mut mine = [0usize; 3];
    let mut k0 = 0usize;
    let mut theirs = [0usize; 3];
    let mut k1 = 0usize;
    for slot in 0..3usize {
        if (cb.sides[0].alive_mask & (1 << slot)) != 0 && cb.mons[slot].hp() > 0 {
            mine[k0] = slot;
            k0 += 1;
        }
        if (cb.sides[1].alive_mask & (1 << slot)) != 0 && cb.mons[3 + slot].hp() > 0 {
            theirs[k1] = slot;
            k1 += 1;
        }
    }
    if k0 == 0 || k1 == 0 {
        return 0.0;
    }
    let flying_mask = 1u32 << (table.flying_type & 31);
    let entry_left = |s: usize, slot: usize| -> f64 {
        if !w.exchange_v2 || (slot as u8) == cb.sides[s].active_idx {
            return 1.0;
        }
        let idx = s * 3 + slot;
        if (table.mons[idx].type_mask & flying_mask) != 0 {
            return 1.0;
        }
        let layers = (cb.sides[s].side_flags & 0x03) as usize;
        if layers == 0 {
            return 1.0;
        }
        const AMOUNTS: [f64; 4] = [0.0, 3.0, 4.0, 6.0];
        let dmg = (AMOUNTS[layers] * (table.mons[idx].maxhp as f64) / 24.0)
            .floor()
            .max(1.0);
        1.0 - dmg / (cb.mons[idx].hp().max(1) as f64)
    };
    let extra = |s: usize, slot: usize| -> f64 {
        if w.exchange_v2 && (slot as u8) != cb.sides[s].active_idx {
            1.0
        } else {
            0.0
        }
    };
    let mut cells = [0.0f64; 9];
    for i in 0..k0 {
        let a = mine[i];
        let la = entry_left(0, a);
        let ea = extra(0, a);
        for j in 0..k1 {
            let d = theirs[j];
            cells[i * k1 + j] = compact_pair_margin_ctx(
                cb,
                table,
                w,
                0,
                a,
                1,
                d,
                [la, entry_left(1, d)],
                [ea, extra(1, d)],
            );
        }
    }
    let slice = &cells[..k0 * k1];
    if w.exchange_v2 {
        return matrix_game_value(slice, [k0, k1]);
    }
    let maximin = (0..k0)
        .map(|i| {
            slice[i * k1..(i + 1) * k1]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let minimax = (0..k1)
        .map(|j| {
            (0..k0)
                .map(|i| slice[i * k1 + j])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .fold(f64::INFINITY, f64::min);
    0.5 * (maximin + minimax)
}
