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
    /// Scale factor on the antisymmetric leaf-evaluation residual MLP
    /// $\Delta_{\text{sym}}(s) = \Delta_\theta(x(s, 0)) - \Delta_\theta(x(s, 1))$.
    /// `0.0` disables the residual network; `1.0` applies the calibrated logit residual.
    pub residual_scale: f64,
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
            residual_scale: 0.0,
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
            residual_scale: 0.0,
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

/// Leaf value with surgical balance-sheet blind-spot accounting (setup/boost
/// liabilities, frozen asset impairment, healer poison liability).
pub fn eval_leaf_blindspot(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let (table, cb) = CompactBattle::from_battle(b, dex);
    compact_eval_leaf_blindspot(&cb, &table, w)
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

/// Leaf evaluation on `CompactBattle` with surgical blind-spot balance-sheet accounting.
#[inline]
pub fn compact_eval_leaf_blindspot(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    assert!((0.0..=1.0).contains(&w.leaf_alpha), "leaf alpha must be in [0,1]");
    0.5 + w.leaf_alpha * (compact_eval01_blindspot(cb, table, w) - 0.5)
}

/// Win-probability-shaped eval in `(0, 1)` from side 0's perspective on `CompactBattle`,
/// adding `compact_blindspot_diff` (strictly antisymmetric, `0.0` on `1v1` so exact
/// Schur-Bellman / Toeplitz M17e anchors remain 0-ULP invariant).
#[inline]
pub fn compact_eval01_blindspot(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    let (s0, eff0) = compact_side_score_with_eff(cb, table, w, 0);
    let (s1, eff1) = compact_side_score_with_eff(cb, table, w, 1);
    let mut diff = s0 - s1;
    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    if n0 > 1 || n1 > 1 {
        diff += compact_blindspot_diff(cb, table, w);
    }
    let mut old_rm = 0.0;
    if w.race != 0.0 {
        old_rm = compact_race_margin(cb, table, w);
        diff += w.race * old_rm;
    }
    if w.exchange != 0.0 {
        diff += w.exchange * compact_exchange_margin(cb, table, w);
    }
    diff += crate::linear::compact_linear_diff(cb, table, diff, old_rm, w.race, w.scale);
    let mut logit = w.scale * diff;
    if w.residual_scale != 0.0 {
        logit += w.residual_scale * compact_eval_residual_sym(cb, table, [eff0, eff1]);
    }
    1.0 / (1.0 + (-logit).exp())
}

/// Strictly antisymmetric balance-sheet correction for `CompactBattle`'s three structural
/// multi-mon blind spots:
/// 1. **Setup / Stat-Boost Liability (`積み負債計上`)**: refunds false `Curse` speed penalties
///    when speed control was not lost, and prices sustainable compounding setup (`Curse`,
///    `Swords Dance`, `Belly Drum`, `Amnesia`, `Growth`, `Double Team`, `Baton Pass`) with
///    concave saturation and active-phazer / Explosion counterplay discounts.
/// 2. **Frozen Asset Impairment (`凍結資産減損`)**: writes down frozen non-MiracleBerry Pokemon
///    to monotone positive scrap value `(0.10 + 0.20 * hp_frac) * w.hp` instead of keeping
///    `+0.94` phantom material at full HP.
/// 3. **Healer Poison / Toxic Liability (`回復壁の毒減損`)**: prices permanent stall-breaking
///    poison on 50%-heal walls (`Recover`/`Milk Drink`/`Moonlight`/`Soft-Boiled`) and naked
///    `Rest` walls lacking `Sleep Talk`.
#[inline]
pub fn compact_blindspot_diff(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    if cb.sides[0].alive_mask.count_ones() <= 1 && cb.sides[1].alive_mask.count_ones() <= 1 {
        return 0.0;
    }
    compact_side_blindspot_ledger(cb, table, w, 0) - compact_side_blindspot_ledger(cb, table, w, 1)
}

#[inline]
fn compact_side_blindspot_ledger(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    s: usize,
) -> f64 {
    let side_st = &cb.sides[s];
    if side_st.alive_mask == 0 {
        return 0.0;
    }
    let mut delta = 0.0;

    // Check whether side `s` has a living non-frozen Heal Bell cleric (`special_kind == 21`)
    // or a Baton Pass source (`special_kind == 18`).
    let mut has_cleric = false;
    let mut team_has_bp = false;
    for slot in 0..3usize {
        if (side_st.alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let idx = s * 3 + slot;
        let mon = &cb.mons[idx];
        if mon.hp() <= 0 {
            continue;
        }
        for m_i in 0..table.mons[idx].num_moves as usize {
            if table.move_specs[idx][m_i].special_kind == 18 {
                team_has_bp = true;
            }
        }
        if mon.status_raw() == Status::Frz as u8 {
            continue;
        }
        let dis = (mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((mon.slot_refs >> 4) & 0x03) as usize;
        for m_i in 0..table.mons[idx].num_moves as usize {
            if mon.pp[m_i] > 0
                && !(dis && m_i == dis_slot)
                && table.move_specs[idx][m_i].special_kind == 21
            {
                has_cleric = true;
                break;
            }
        }
    }
    let cleric_mult = if has_cleric { 0.45 } else { 1.0 };

    // 1. Party-wide Frozen Asset Impairment & Healer Poison Liability
    for slot in 0..3usize {
        if (side_st.alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let idx = s * 3 + slot;
        let mon = &cb.mons[idx];
        let spec = &table.mons[idx];
        let hp = mon.hp();
        if hp <= 0 {
            continue;
        }
        let hp_frac = (hp as f64) / (spec.maxhp.max(1) as f64);
        let st = mon.status_raw();

        if st == Status::Frz as u8 && !(mon.item != 0 && spec.is_miracleberry) {
            // In Gen 2, a frozen Pokemon has a 0% passive thaw chance while on the bench and only
            // 25/256 (~9.8%) per turn while active. Base eval only subtracts `w.frz = 0.56` from
            // `w.alive + w.hp * hp_frac` (`+0.94` at 100% HP). Write down its book value uniformly
            // across active and bench slots so switching out a frozen active mon never suffers a
            // false bench-impairment cliff.
            let base_val = w.alive + w.hp * hp_frac - w.frz;
            let scrap_val = (0.10 + 0.20 * hp_frac) * w.hp;
            if base_val > scrap_val {
                delta -= (base_val - scrap_val) * cleric_mult;
            }
        } else if st == Status::Psn as u8 || st == Status::Tox as u8 {
            let dis = (mon.vol_flags & vol_bit::DISABLE) != 0;
            let dis_slot = ((mon.slot_refs >> 4) & 0x03) as usize;
            let mut has_half_heal = false;
            let mut has_rest = false;
            let mut has_sleeptalk = false;
            for m_i in 0..spec.num_moves as usize {
                if mon.pp[m_i] > 0 && !(dis && m_i == dis_slot) {
                    let ms = &table.move_specs[idx][m_i];
                    has_half_heal |= ms.is_half_heal;
                    has_rest |= ms.is_rest;
                    has_sleeptalk |= ms.special_kind == 10;
                }
            }
            let hp_scale = 0.35 + 0.65 * hp_frac;
            if has_half_heal && !has_rest {
                let pen = if st == Status::Tox as u8 { 0.38 } else { 0.24 };
                delta -= pen * hp_scale * cleric_mult;
            } else if has_rest && !has_sleeptalk {
                let pen = if st == Status::Tox as u8 { 0.30 } else { 0.18 };
                delta -= pen * hp_scale * cleric_mult;
            }
        }
    }

    // 2. Active Setup / Stat-Boost / Confusion / Sleep-Threat Accounting
    let act_slot = side_st.active_idx as usize;
    if (side_st.alive_mask & (1 << act_slot)) == 0 {
        return delta;
    }
    let act_idx = s * 3 + act_slot;
    let act_mon = &cb.mons[act_idx];
    let act_spec = &table.mons[act_idx];
    if act_mon.hp() <= 0 {
        return delta;
    }

    let foe_s = 1 - s;
    let foe_slot = cb.sides[foe_s].active_idx as usize;
    let foe_alive = (cb.sides[foe_s].alive_mask & (1 << foe_slot)) != 0
        && cb.mons[foe_s * 3 + foe_slot].hp() > 0;
    let foe_idx = foe_s * 3 + foe_slot;
    let my_hit = if foe_alive {
        cb.best_hit_fraction(table, s, act_slot, foe_s, foe_slot, w.couple_evasion)
    } else {
        0.0
    };

    if act_mon.status_raw() == Status::Frz as u8 {
        if !(act_mon.item != 0 && act_spec.is_miracleberry) {
            // Base eval added full `w.threat * act_eff` and positive `w.boost` for the frozen active mon.
            delta -= 0.85 * w.threat * my_hit.min(1.0);
            for i in 0..7 {
                if act_mon.boosts[i] > 0 {
                    delta -= w.boost[i] * (act_mon.boosts[i] as f64);
                }
            }
        }
        return delta;
    }

    let dis = (act_mon.vol_flags & vol_bit::DISABLE) != 0;
    let dis_slot = ((act_mon.slot_refs >> 4) & 0x03) as usize;
    let mut has_phys = false;
    let mut has_any_phys = false;
    let mut has_spec = false;
    let mut has_heal = false;
    let mut has_bp = false;
    let mut has_self_setup = false;
    let mut has_sleeptalk = false;
    for m_i in 0..act_spec.num_moves as usize {
        let ms = &table.move_specs[act_idx][m_i];
        if ms.boosts_self || matches!(ms.pseudo_kind, 5 | 6 | 7) {
            has_self_setup = true;
        }
        if act_mon.pp[m_i] == 0 || (dis && m_i == dis_slot) {
            continue;
        }
        let deals_dmg = !ms.is_status
            && (ms.base_power > 0
                || ms.var_bp_kind != 0
                || ms.fixed_damage != 0
                || ms.special_kind == 15);
        if deals_dmg && ms.is_physical {
            has_any_phys = true;
            if !ms.is_selfdestruct {
                has_phys = true;
            }
        } else if deals_dmg && !ms.is_physical {
            has_spec = true;
        }
        if ms.is_rest || ms.is_half_heal {
            has_heal = true;
        }
        if ms.special_kind == 18 && side_st.alive_mask.count_ones() >= 2 {
            has_bp = true;
        }
        if ms.special_kind == 10 || ms.is_sleep_usable {
            has_sleeptalk = true;
        }
    }

    // Neutralize positive offensive stat stages on a mon with no corresponding attacks (`Swagger` on a
    // pure Special attacker or wall without physical moves / Baton Pass).
    if !has_any_phys && !has_bp && act_mon.boosts[0] > 0 {
        delta -= w.boost[0] * (act_mon.boosts[0] as f64);
    }
    if !has_spec && !has_bp && act_mon.boosts[2] > 0 {
        delta -= w.boost[2] * (act_mon.boosts[2] as f64);
    }

    let hp_frac = (act_mon.hp() as f64) / (act_spec.maxhp.max(1) as f64);
    let st = act_mon.status_raw();
    let has_cure_berry =
        act_mon.item != 0 && (act_spec.is_mintberry || act_spec.is_miracleberry);
    let has_any_pos_boost = act_mon.boosts.iter().any(|&b| b > 0);
    if st == Status::Slp as u8
        && !has_sleeptalk
        && !has_heal
        && act_mon.status_counter() >= 2
        && !has_cure_berry
        && (has_any_pos_boost || has_bp || team_has_bp)
    {
        // Active non-Rest/non-SleepTalk setup/Baton-Pass mon is guaranteed asleep;
        // discount immediate active threat.
        delta -= 0.55 * w.threat * my_hit.min(1.0);
    }

    // Confusion / Swagger liability: base `w.confusion` is 0.0, so without this adjustment a
    // Swaggered target receives `+0.30` from `+2 Atk` and 0 confusion penalty.
    if (act_mon.vol_flags & vol_bit::CONFUSION) != 0 {
        if !has_self_setup && !team_has_bp && act_mon.boosts[0] > 0 && (has_any_phys || has_bp) {
            delta -= 0.70 * w.boost[0] * (act_mon.boosts[0] as f64);
        }
        if st != Status::Slp as u8 {
            let atk_mult = if act_mon.boosts[0] > 0 {
                1.0 + 0.35 * (act_mon.boosts[0] as f64)
            } else {
                1.0
            };
            delta -= 0.12 * atk_mult * (0.4 + 0.6 * hp_frac);
        }
        return delta;
    }

    // Never credit setup capital from an opponent's Swagger (`!has_self_setup && !team_has_bp`)
    // or when under Perish Song / Encore.
    if !has_self_setup && !team_has_bp {
        return delta;
    }
    if (act_mon.vol_flags & (vol_bit::PERISH_SONG | vol_bit::ENCORE)) != 0 {
        return delta;
    }

    let my_spd = cb.mon_action_speed(table, s, act_slot);
    let (foe_spd, foe_hit) = if foe_alive {
        (
            cb.mon_action_speed(table, foe_s, foe_slot),
            cb.best_hit_fraction(table, foe_s, foe_slot, s, act_slot, w.couple_evasion),
        )
    } else {
        (0, 0.0)
    };
    let me_faster = my_spd > foe_spd;

    // Discount setup capital if the opposing active mon holds an active phazer / Perish / Explosion deterrent,
    // or if our active sweeper is completely walled by the opposing active mon (e.g. mono-Normal vs Ghost).
    let mut counter_mult: f64 = 1.0;
    let mut walled_without_bp = false;
    if foe_alive {
        let foe_mon = &cb.mons[foe_idx];
        let foe_st = foe_mon.status_raw();
        let foe_can_act_soon = foe_st != Status::Frz as u8
            && !(foe_st == Status::Slp as u8 && foe_mon.status_counter() >= 2);
        if my_hit < 0.05 && !has_bp {
            walled_without_bp = true;
        }
        if foe_can_act_soon {
            let f_dis = (foe_mon.vol_flags & vol_bit::DISABLE) != 0;
            let f_dis_slot = ((foe_mon.slot_refs >> 4) & 0x03) as usize;
            let foe_hp_frac =
                (foe_mon.hp() as f64) / (table.mons[foe_idx].maxhp.max(1) as f64);
            let foe_survives_first = my_hit < 0.90 * foe_hp_frac;
            let mut foe_has_wiper = false;
            let mut foe_has_perish = false;
            let mut foe_has_lethal_boom = false;
            for m_i in 0..table.mons[foe_idx].num_moves as usize {
                if foe_mon.pp[m_i] == 0 || (f_dis && m_i == f_dis_slot) {
                    continue;
                }
                let fms = &table.move_specs[foe_idx][m_i];
                if fms.special_kind == 19 && !has_bp && (foe_survives_first || !me_faster) {
                    foe_has_perish = true;
                } else if (fms.special_kind == 7 && side_st.alive_mask.count_ones() >= 2)
                    && foe_survives_first
                {
                    foe_has_wiper = true;
                } else if matches!(fms.special_kind, 8 | 27) && (foe_survives_first || !me_faster)
                {
                    foe_has_wiper = true;
                } else if fms.is_selfdestruct
                    && cb.sides[foe_s].alive_mask.count_ones() >= 2
                    && (foe_survives_first || !me_faster)
                {
                    let boom_dmg = cb.expected_hit_fraction(
                        table,
                        foe_s,
                        foe_slot,
                        s,
                        act_slot,
                        m_i,
                        w.couple_evasion,
                    );
                    if boom_dmg >= 0.65 * hp_frac {
                        foe_has_lethal_boom = true;
                    }
                }
            }
            if foe_has_perish {
                return delta;
            } else if foe_has_wiper {
                counter_mult = 0.45;
            } else if foe_has_lethal_boom {
                counter_mult = 0.50;
            }
        }
    }
    if walled_without_bp {
        counter_mult *= 0.25;
    }

    // 2a. Refund false Curse speed-drop deduction when speed was already <= foe (or bulky Curse)
    if act_mon.boosts[4] < 0 && act_mon.boosts[0] > 0 && act_mon.boosts[1] > 0 {
        let k_curse = (-act_mon.boosts[4])
            .min(act_mon.boosts[0])
            .min(act_mon.boosts[1]) as f64;
        let unboosted_spe = (act_spec.stored_stats[4] as i32)
            >> (if act_mon.status_raw() == Status::Par as u8 { 2 } else { 0 });
        let already_slower =
            foe_alive && unboosted_spe <= cb.mon_action_speed(table, foe_s, foe_slot);
        let refund_frac = if already_slower { 1.0 } else { 0.65 };
        delta += refund_frac * counter_mult * w.boost[4] * k_curse;
    }

    // 2b. Sustainable compounding setup capital (Offense x Defense / Speed / Evasion / Baton Pass)
    let b_atk = if has_phys || has_bp {
        act_mon.boosts[0].max(0) as f64
    } else {
        0.0
    };
    let b_spa = if has_spec || has_bp {
        act_mon.boosts[2].max(0) as f64
    } else {
        0.0
    };
    let b_off = b_atk.max(b_spa);
    let b_def = act_mon.boosts[1].max(0) as f64;
    let b_spd = act_mon.boosts[3].max(0) as f64;
    let b_spe = act_mon.boosts[4].max(0) as f64;
    let b_eva = if w.couple_evasion {
        act_mon.boosts[6].max(0) as f64
    } else {
        0.0
    };

    if b_off >= 2.0
        || (b_off + b_def + b_spd) >= 3.0
        || (b_def + b_spd) >= 2.0
        || b_eva > 0.0
        || (has_bp && (b_off > 0.0 || b_spe > 0.0))
    {
        let bp_term = if has_bp {
            0.16 * (b_off + 0.8 * b_spe + b_eva)
        } else {
            0.0
        };
        let raw_intensity = 0.28 * b_off
            + 0.14 * (b_def + b_spd)
            + 0.32 * b_eva
            + 0.12 * b_off * (b_def + 0.7 * b_spe + b_eva)
            + bp_term;
        let concave_cap = raw_intensity / (1.0 + 0.55 * raw_intensity);

        let mut sust = if has_heal {
            if st == Status::Slp as u8 && !has_sleeptalk {
                let rem = act_mon.status_counter().saturating_sub(1) as f64;
                if (hp_frac - rem * foe_hit.max(0.15)).max(0.0) > 0.25 {
                    0.75 * hp_frac
                } else {
                    0.30 * hp_frac
                }
            } else if hp_frac > foe_hit || me_faster {
                hp_frac.max(0.85)
            } else {
                0.40 * hp_frac
            }
        } else if st == Status::Slp as u8 && !has_sleeptalk {
            let rem = act_mon.status_counter().saturating_sub(1) as f64;
            let hp_wake = (hp_frac - rem * foe_hit.max(0.15) * 0.85).clamp(0.0, 1.0);
            if rem == 0.0 && (hp_frac > foe_hit || me_faster) {
                0.65 * hp_frac
            } else {
                0.25 * hp_wake
            }
        } else if hp_frac > foe_hit || me_faster {
            hp_frac
        } else {
            0.30 * hp_frac
        };
        if (act_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 {
            sust = (sust + 0.25).min(1.15);
        }

        delta += concave_cap * sust * counter_mult;
    }

    delta
}

/// Win-probability-shaped eval in `(0, 1)` from side 0's perspective on `CompactBattle`.
#[inline]
pub fn compact_eval01(cb: &CompactBattle, table: &MatchTable, w: &EvalWeights) -> f64 {
    let (s0, eff0) = compact_side_score_with_eff(cb, table, w, 0);
    let (s1, eff1) = compact_side_score_with_eff(cb, table, w, 1);
    let mut diff = s0 - s1;
    let mut old_rm = 0.0;
    if w.race != 0.0 {
        old_rm = compact_race_margin(cb, table, w);
        diff += w.race * old_rm;
    }
    if w.exchange != 0.0 {
        diff += w.exchange * compact_exchange_margin(cb, table, w);
    }
    diff += crate::linear::compact_linear_diff(cb, table, diff, old_rm, w.race, w.scale);
    let mut logit = w.scale * diff;
    if w.residual_scale != 0.0 {
        logit += w.residual_scale * compact_eval_residual_sym(cb, table, [eff0, eff1]);
    }
    1.0 / (1.0 + (-logit).exp())
}

/// Compute the analytical base logit $L_{\text{base}}(s) = w_{\text{scale}} \cdot \text{diff}$
/// and active expected hit fractions `[eff0, eff1]` without the residual network.
#[inline]
pub fn compact_base_logit_and_eff(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
) -> (f64, [f64; 2]) {
    let (s0, eff0) = compact_side_score_with_eff(cb, table, w, 0);
    let (s1, eff1) = compact_side_score_with_eff(cb, table, w, 1);
    let mut diff = s0 - s1;
    let mut old_rm = 0.0;
    if w.race != 0.0 {
        old_rm = compact_race_margin(cb, table, w);
        diff += w.race * old_rm;
    }
    if w.exchange != 0.0 {
        diff += w.exchange * compact_exchange_margin(cb, table, w);
    }
    diff += crate::linear::compact_linear_diff(cb, table, diff, old_rm, w.race, w.scale);
    (w.scale * diff, [eff0, eff1])
}

#[inline]
fn compact_side_score_with_eff(
    cb: &CompactBattle,
    table: &MatchTable,
    w: &EvalWeights,
    s: usize,
) -> (f64, f64) {
    let side_st = &cb.sides[s];
    let mut score = 0.0;
    let mut act_eff = 0.0;
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
                act_eff =
                    cb.best_hit_fraction(table, s, act_slot, foe_s, foe_slot, w.couple_evasion);
                score += w.threat * act_eff.min(1.0);
            }
        }
    }
    (score, act_eff)
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

// =========================================================================
// Residual Evaluation MLP & Rollout Action-Residual Calibration
// =========================================================================

use std::sync::OnceLock;

pub const EVAL_FEAT_DIM: usize = 32;
pub const EVAL_HIDDEN_DIM: usize = 16;

pub const ROLLOUT_STATE_DIM: usize = 12;
pub const ROLLOUT_ACT_DIM: usize = 6;
pub const ROLLOUT_RANK: usize = 4;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvalResidualWeights {
    pub active: bool,
    pub w1: [[f32; EVAL_FEAT_DIM]; EVAL_HIDDEN_DIM],
    pub b1: [f32; EVAL_HIDDEN_DIM],
    pub w2: [f32; EVAL_HIDDEN_DIM],
}

impl EvalResidualWeights {
    pub const fn zero() -> Self {
        Self {
            active: false,
            w1: [[0.0; EVAL_FEAT_DIM]; EVAL_HIDDEN_DIM],
            b1: [0.0; EVAL_HIDDEN_DIM],
            w2: [0.0; EVAL_HIDDEN_DIM],
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct RolloutResidualWeights {
    pub active: bool,
    pub u_proj: [[f32; ROLLOUT_STATE_DIM]; ROLLOUT_RANK],
    pub v_proj: [[f32; ROLLOUT_ACT_DIM]; ROLLOUT_RANK],
    pub act_bias: [f32; ROLLOUT_ACT_DIM],
}

impl RolloutResidualWeights {
    pub const fn zero() -> Self {
        Self {
            active: false,
            u_proj: [[0.0; ROLLOUT_STATE_DIM]; ROLLOUT_RANK],
            v_proj: [[0.0; ROLLOUT_ACT_DIM]; ROLLOUT_RANK],
            act_bias: [0.0; ROLLOUT_ACT_DIM],
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResidualBundle {
    pub eval: EvalResidualWeights,
    pub rollout: RolloutResidualWeights,
}

impl ResidualBundle {
    pub const fn zero() -> Self {
        Self {
            eval: EvalResidualWeights::zero(),
            rollout: RolloutResidualWeights::zero(),
        }
    }
}

/// Default compiled residual weights (Gate 1 calibrated `eval_residual` active; `rollout_residual` inactive).
pub const DEFAULT_RESIDUAL_BUNDLE: ResidualBundle = ResidualBundle {
    eval: EvalResidualWeights {
        active: true,
        w1: [
            [
                0.1592635,
                0.12946095,
                0.1194611,
                0.0003951387,
                0.107727565,
                0.0,
                0.10432918,
                0.13698481,
                0.05285247,
                0.11506298,
                0.0,
                0.097680226,
                0.13070732,
                0.11265245,
                0.12788852,
                0.097194225,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.101189956,
                0.1694334,
                0.08344528,
                0.08816895,
                0.048799418,
                0.10238286,
                0.2126568,
                0.0,
                0.07133045,
                0.10579651,
                0.09185278,
                0.07481665,
                0.028079467,
                0.12867014,
                0.1693474,
                0.08667781,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.4301937,
                0.17532508,
                0.44588163,
                0.0,
                0.0,
                0.0,
                0.40826002,
                0.09990849,
                0.0,
                0.2664039,
                0.0,
                0.30883932,
                0.012687975,
                0.032227036,
                0.16722296,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.4180432,
                0.03375988,
                0.36683336,
                0.0,
                0.0,
                0.0,
                0.4056989,
                0.0,
                0.0,
                0.299519,
                0.0,
                0.34608608,
                0.15768145,
                0.0,
                0.24105823,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.107165806,
                0.19891012,
                0.18452336,
                0.027144747,
                0.0,
                0.074771635,
                0.123442635,
                0.0,
                0.06029057,
                0.23902446,
                0.0,
                0.18970487,
                0.18352276,
                0.1174403,
                0.2622345,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.41934305,
                0.0065284097,
                0.33491388,
                0.0,
                0.0,
                0.0,
                0.31453362,
                0.09637944,
                0.0,
                0.33386287,
                0.0,
                0.23820871,
                0.07165454,
                0.12357317,
                0.06650649,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.23348774,
                0.19378921,
                0.15567504,
                0.042202853,
                0.042812735,
                0.074472606,
                0.1357919,
                0.06553856,
                0.07969123,
                0.089954264,
                0.0,
                0.16780518,
                0.0040651164,
                0.18930735,
                0.083820105,
                0.021488046,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.49159515,
                0.036362596,
                0.48515597,
                0.0,
                0.0,
                0.0,
                0.4541916,
                0.0,
                0.0,
                0.3482166,
                0.0,
                0.39139307,
                0.06412866,
                0.09684929,
                0.11244748,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.22959538,
                0.0,
                0.21355996,
                0.12527631,
                0.16826937,
                0.23940413,
                0.05320776,
                0.107458405,
                0.08296397,
                0.0,
                0.07929242,
                0.07782938,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.22885,
                0.39440846,
                0.0,
                0.23830053,
                0.185397,
                0.35957575,
                0.0,
                0.0,
                0.39405563,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.19171354,
                0.019145502,
                0.19165786,
                0.12039654,
                0.19017553,
                0.2244097,
                0.05693801,
                0.1494015,
                0.0,
                0.052529648,
                0.017231552,
                0.0,
                0.0,
                0.0,
                0.04571331,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.180914,
                0.34676903,
                0.0,
                0.28007132,
                0.1692566,
                0.22470985,
                0.06791634,
                0.0,
                0.28711328,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.234466,
                0.50558555,
                0.0,
                0.2485967,
                0.19141845,
                0.35332057,
                0.01311815,
                0.0,
                0.37858465,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.20037,
                0.43232366,
                0.0,
                0.4189235,
                0.22831236,
                0.37606534,
                0.11593896,
                0.0,
                0.35738957,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.16187,
                0.353993,
                0.0,
                0.29652765,
                0.27149695,
                0.24860895,
                0.13369489,
                0.0,
                0.3844487,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.16410282,
                0.0,
                0.14905632,
                0.13443504,
                0.28700623,
                0.15320116,
                0.060230475,
                0.14842583,
                0.07988839,
                0.08262625,
                0.047094699,
                0.09009234,
                0.0,
                0.0,
                0.0,
            ],
        ],
        b1: [0.0; EVAL_HIDDEN_DIM],
        w2: [
            0.0,
            0.0,
            0.20958547,
            0.17367375,
            0.0,
            0.15544592,
            0.0,
            0.19376242,
            0.0,
            -0.27715674,
            0.0,
            -0.21910277,
            -0.28395805,
            -0.2426653,
            -0.19603783,
            0.0,
        ],
    },
    rollout: RolloutResidualWeights::zero(),
};

static RESIDUAL_BUNDLE_CELL: OnceLock<ResidualBundle> = OnceLock::new();

/// Initialize the process-wide residual bundle explicitly before search starts.
pub fn init_residual_bundle(bundle: ResidualBundle) {
    let _ = RESIDUAL_BUNDLE_CELL.set(bundle);
}

/// Load a `ResidualBundle` from a JSON file path and install it process-wide.
pub fn init_residual_bundle_from_file(path: &std::path::Path) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bundle: ResidualBundle =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let _ = RESIDUAL_BUNDLE_CELL.set(bundle);
    Ok(())
}

#[inline(always)]
pub fn active_residual_bundle() -> &'static ResidualBundle {
    RESIDUAL_BUNDLE_CELL.get_or_init(|| {
        if let Ok(path) = std::env::var("NC2000_RESIDUAL_JSON") {
            if !path.is_empty() {
                let text = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("NC2000_RESIDUAL_JSON read {path}: {e}"));
                let bundle = serde_json::from_str::<ResidualBundle>(&text)
                    .unwrap_or_else(|e| panic!("NC2000_RESIDUAL_JSON parse {path}: {e}"));
                return bundle;
            }
        }
        DEFAULT_RESIDUAL_BUNDLE
    })
}

/// Forward pass of the 2-layer residual MLP $\Delta_\theta(x) = W_2 \text{ReLU}(W_1 x + b_1)$.
#[inline(always)]
pub fn eval_residual_forward(x: &[f32; EVAL_FEAT_DIM], m: &EvalResidualWeights) -> f32 {
    if !m.active {
        return 0.0;
    }
    let mut out = 0.0f32;
    for h in 0..EVAL_HIDDEN_DIM {
        let row = &m.w1[h];
        let mut s0 = m.b1[h];
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut j = 0;
        while j < EVAL_FEAT_DIM {
            s0 += row[j] * x[j];
            s1 += row[j + 1] * x[j + 1];
            s2 += row[j + 2] * x[j + 2];
            s3 += row[j + 3] * x[j + 3];
            j += 4;
        }
        let z = (s0 + s1) + (s2 + s3);
        if z > 0.0 {
            out += m.w2[h] * z;
        }
    }
    out
}

/// Strictly antisymmetric residual logit $\Delta_{\text{sym}}(s) = \Delta_\theta(x(s, 0)) - \Delta_\theta(x(s, 1))$.
#[inline(always)]
pub fn compact_eval_residual_sym(
    cb: &CompactBattle,
    table: &MatchTable,
    act_eff: [f64; 2],
) -> f64 {
    let m = &active_residual_bundle().eval;
    if !m.active {
        return 0.0;
    }
    if cb.sides[0].alive_mask == 0 || cb.sides[1].alive_mask == 0 {
        return 0.0;
    }
    let x0 = extract_eval_features_with_eff(cb, table, 0, act_eff);
    let x1 = extract_eval_features_with_eff(cb, table, 1, act_eff);
    (eval_residual_forward(&x0, m) - eval_residual_forward(&x1, m)) as f64
}

/// Extract the 32-dimensional continuous physical & generic move-capability feature vector
/// $x(s, \text{side}) \in \mathbb{R}^{32}$ from `side`'s perspective.
#[inline]
pub fn extract_eval_features(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
) -> [f32; EVAL_FEAT_DIM] {
    let eff = [0, 1].map(|s| {
        let act_slot = (cb.sides[s].active_idx as usize).min(2);
        let foe_s = 1 - s;
        let foe_slot = (cb.sides[foe_s].active_idx as usize).min(2);
        if (cb.sides[s].alive_mask & (1 << act_slot)) != 0
            && cb.mons[s * 3 + act_slot].hp() > 0
            && (cb.sides[foe_s].alive_mask & (1 << foe_slot)) != 0
            && cb.mons[foe_s * 3 + foe_slot].hp() > 0
        {
            cb.best_hit_fraction(table, s, act_slot, foe_s, foe_slot, true)
        } else {
            0.0
        }
    });
    extract_eval_features_with_eff(cb, table, side, eff)
}

#[inline]
pub fn extract_eval_features_with_eff(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    act_eff: [f64; 2],
) -> [f32; EVAL_FEAT_DIM] {
    let mut x = [0.0f32; EVAL_FEAT_DIM];
    let foe = 1 - side;
    let own_side = &cb.sides[side];
    let foe_side = &cb.sides[foe];
    let own_alive_cnt = own_side.alive_mask.count_ones();
    let foe_alive_cnt = foe_side.alive_mask.count_ones();
    if own_alive_cnt == 0 || foe_alive_cnt == 0 {
        return x;
    }

    let own_act_slot = (own_side.active_idx as usize).min(2);
    let foe_act_slot = (foe_side.active_idx as usize).min(2);
    let own_act_idx = side * 3 + own_act_slot;
    let foe_act_idx = foe * 3 + foe_act_slot;
    let own_mon = &cb.mons[own_act_idx];
    let foe_mon = &cb.mons[foe_act_idx];
    let own_spec = &table.mons[own_act_idx];
    let foe_spec = &table.mons[foe_act_idx];

    let own_act_alive = (own_side.alive_mask & (1 << own_act_slot)) != 0 && own_mon.hp() > 0;
    let foe_act_alive = (foe_side.alive_mask & (1 << foe_act_slot)) != 0 && foe_mon.hp() > 0;
    if !own_act_alive || !foe_act_alive {
        return x;
    }

    let own_hp_frac = (own_mon.hp() as f32) / (own_spec.maxhp.max(1) as f32);
    let foe_hp_frac = (foe_mon.hp() as f32) / (foe_spec.maxhp.max(1) as f32);
    let own_eff = (act_eff[side] as f32).min(1.5);
    let foe_eff = (act_eff[foe] as f32).min(1.5);

    let spd_own = cb.mon_action_speed(table, side, own_act_slot);
    let spd_foe = cb.mon_action_speed(table, foe, foe_act_slot);
    let faster = if spd_own > spd_foe {
        1.0f32
    } else if spd_own == spd_foe {
        0.5f32
    } else {
        0.0f32
    };

    // Active positive and negative boosts (scaled to [0, 1])
    let pos_atk = (own_mon.boosts[0].max(0) as f32) / 6.0;
    let pos_def = (own_mon.boosts[1].max(0) as f32) / 6.0;
    let pos_spa = (own_mon.boosts[2].max(0) as f32) / 6.0;
    let pos_spd = (own_mon.boosts[3].max(0) as f32) / 6.0;
    let pos_spe = (own_mon.boosts[4].max(0) as f32) / 6.0;
    let pos_eva = (own_mon.boosts[6].max(0) as f32) / 6.0;
    let pos_off_boost = (pos_atk + pos_spa).min(1.5);
    let pos_bulk_eva_boost = (pos_def + pos_spd + pos_eva).min(1.5);
    let pos_boost_sum = (pos_off_boost + pos_spe + pos_bulk_eva_boost).min(2.0);

    let neg_def_spd = ((-own_mon.boosts[1]).max(0) + (-own_mon.boosts[3]).max(0)) as f32 / 6.0;
    let neg_off_spe_acc = ((-own_mon.boosts[0]).max(0)
        + (-own_mon.boosts[2]).max(0)
        + (-own_mon.boosts[4]).max(0)
        + (-own_mon.boosts[5]).max(0)) as f32
        / 6.0;

    let mut foe_pos_boosts_i = 0i32;
    for i in 0..7 {
        if foe_mon.boosts[i] > 0 {
            foe_pos_boosts_i += foe_mon.boosts[i] as i32;
        }
    }
    let foe_pos_boost_sum = (foe_pos_boosts_i as f32 / 6.0).min(1.5);

    let sub_hp_frac = if (own_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 {
        let sub_max = (own_spec.maxhp as f32 * 0.25).max(1.0);
        (own_mon.sub_hp() as f32 / sub_max).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let foe_sub_hp_frac = if (foe_mon.vol_flags & vol_bit::SUBSTITUTE) != 0 {
        let sub_max = (foe_spec.maxhp as f32 * 0.25).max(1.0);
        (foe_mon.sub_hp() as f32 / sub_max).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let own_st = own_mon.status_raw();
    let is_asleep_or_frozen = if own_st == Status::Slp as u8 || own_st == Status::Frz as u8 {
        1.0f32
    } else {
        0.0f32
    };
    let awake_factor = 1.0 - is_asleep_or_frozen;

    // Generic move capabilities on own and foe active
    let mut has_self_switch = 0.0f32;
    let mut has_force_switch = 0.0f32;
    let mut has_setup_off = 0.0f32;
    let mut has_setup_spd_def = 0.0f32;
    let mut has_sub_move = 0.0f32;
    let mut has_heal_move = 0.0f32;
    let mut has_sleep_move = 0.0f32;

    let is_dis = (own_mon.vol_flags & vol_bit::DISABLE) != 0;
    let dis_slot = ((own_mon.slot_refs >> 4) & 0x03) as usize;
    let is_ghost = (own_spec.type_mask & (1u32 << (table.ghost_type & 31))) != 0;
    for m_i in 0..own_spec.num_moves as usize {
        if own_mon.pp[m_i] == 0 || (is_dis && m_i == dis_slot) {
            continue;
        }
        let ms = &table.move_specs[own_act_idx][m_i];
        if ms.special_kind == 18 && own_alive_cnt > 1 {
            has_self_switch = 1.0;
        }
        if ms.special_kind == 7 && foe_alive_cnt > 1 {
            has_force_switch = 1.0;
        }
        if ms.boosts_self {
            for &(st_i, d) in &ms.boosts {
                if d > 0 {
                    let b_val = (d as f32 * 0.5).min(1.0);
                    if st_i == 0 || st_i == 2 {
                        has_setup_off = has_setup_off.max(b_val);
                    } else if st_i < 7 {
                        has_setup_spd_def = has_setup_spd_def.max(b_val);
                    }
                }
            }
        } else if ms.pseudo_kind == 5 && !is_ghost {
            has_setup_off = has_setup_off.max(0.5);
            has_setup_spd_def = has_setup_spd_def.max(0.5);
        } else if ms.pseudo_kind == 8 && own_hp_frac > 0.5 {
            has_setup_off = has_setup_off.max(1.0);
        }
        if ms.is_substitute && own_hp_frac > 0.25 {
            has_sub_move = 1.0;
        }
        if ms.is_heal_cycle || ms.is_half_heal {
            has_heal_move = 1.0;
        }
        if ms.primary_status == Status::Slp as u8 {
            has_sleep_move = has_sleep_move.max(ms.base_acc_pct as f32 / 100.0);
        }
    }
    let has_setup_any = has_setup_off.max(has_setup_spd_def);

    let mut foe_has_phaze = 0.0f32;
    let foe_dis = (foe_mon.vol_flags & vol_bit::DISABLE) != 0;
    let foe_dis_slot = ((foe_mon.slot_refs >> 4) & 0x03) as usize;
    for m_i in 0..foe_spec.num_moves as usize {
        if foe_mon.pp[m_i] == 0 || (foe_dis && m_i == foe_dis_slot) {
            continue;
        }
        let ms = &table.move_specs[foe_act_idx][m_i];
        if ms.special_kind == 7 && own_alive_cnt > 1 {
            foe_has_phaze = 1.0;
        }
    }

    // Bench metrics for own and foe
    let flying_mask = 1u32 << (table.flying_type & 31);
    let mut bench_hp_sum = 0.0f32;
    let mut bench_best_eff = 0.0f32;
    let mut bench_max_atk = 0.0f32;
    let mut bench_max_spa = 0.0f32;
    let mut bench_grounded_cnt = 0.0f32;
    for slot in 0..3usize {
        if slot == own_act_slot || (own_side.alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let b_idx = side * 3 + slot;
        let b_mon = &cb.mons[b_idx];
        if b_mon.hp() <= 0 {
            continue;
        }
        let b_spec = &table.mons[b_idx];
        let b_hp = (b_mon.hp() as f32) / (b_spec.maxhp.max(1) as f32);
        bench_hp_sum += b_hp;
        if (b_spec.type_mask & flying_mask) == 0 {
            bench_grounded_cnt += 0.5;
        }
        let eff = cb
            .best_hit_fraction(table, side, slot, foe, foe_act_slot, true)
            .min(1.5) as f32;
        if eff > bench_best_eff {
            bench_best_eff = eff;
        }
        let atk_stat = cb.mon_stat(table, side, slot, 0, true, false) as f32 / 200.0;
        let spa_stat = cb.mon_stat(table, side, slot, 2, true, false) as f32 / 200.0;
        bench_max_atk = bench_max_atk.max((b_hp * atk_stat).min(1.5));
        bench_max_spa = bench_max_spa.max((b_hp * spa_stat).min(1.5));
    }

    let mut foe_bench_hp_sum = 0.0f32;
    for slot in 0..3usize {
        if slot == foe_act_slot || (foe_side.alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let b_idx = foe * 3 + slot;
        let b_mon = &cb.mons[b_idx];
        if b_mon.hp() > 0 {
            let b_spec = &table.mons[b_idx];
            foe_bench_hp_sum += (b_mon.hp() as f32) / (b_spec.maxhp.max(1) as f32);
        }
    }

    let pass_viability = (own_hp_frac - foe_eff * (1.0 - 0.5 * faster) + 0.25 * sub_hp_frac)
        .clamp(0.0, 1.0)
        * awake_factor;
    let safe_window = (own_hp_frac - 1.2 * foe_eff + 0.3 * sub_hp_frac).clamp(0.0, 1.0) * awake_factor;

    let own_screens_frac =
        ((own_side.reflect_turns.max(own_side.lightscreen_turns)) as f32 / 5.0).min(1.0);
    let own_safeguard_frac = (own_side.safeguard_turns as f32 / 5.0).min(1.0);
    let own_spikes_frac = (own_side.side_flags & 0x03) as f32 / 3.0;
    let foe_spikes_frac = (foe_side.side_flags & 0x03) as f32 / 3.0;

    let foe_status_any = if foe_mon.status_raw() != 0 { 1.0f32 } else { 0.0f32 };
    let foe_safeguard = if foe_side.safeguard_turns > 0 { 1.0f32 } else { 0.0f32 };
    let foe_seeded_or_cursed =
        if (foe_mon.vol_flags & (vol_bit::LEECH_SEED | vol_bit::CURSE)) != 0 {
            1.0f32
        } else {
            0.0f32
        };

    // =========================================================================
    // x[0..16]: Non-negative Multi-Body Synergy Features (monotonic >= 0 for `side`)
    // =========================================================================
    // 0: Baton Pass physical boost x healthy physical bench receiver
    x[0] = has_self_switch * pos_atk * bench_max_atk * pass_viability;
    // 1: Baton Pass special boost x healthy special bench receiver
    x[1] = has_self_switch * pos_spa * bench_max_spa * pass_viability;
    // 2: Baton Pass speed/bulk/evasion boost x bench best threat
    x[2] = has_self_switch * (pos_spe + 0.7 * pos_bulk_eva_boost) * bench_best_eff * pass_viability;
    // 3: Baton Pass active Substitute transfer x bench best threat
    x[3] = has_self_switch * sub_hp_frac * bench_best_eff * awake_factor;
    // 4: Unboosted Baton Pass + setup/Sub move + bench receiver readiness in safe window
    x[4] = has_self_switch
        * (has_setup_off * bench_max_atk.max(bench_max_spa)
            + (has_setup_spd_def + 0.5 * has_sub_move) * bench_best_eff)
        * safe_window;
    // 5: Active offensive boost x own offense x (speed or Sub protection) when foe cannot phaze
    x[5] = pos_off_boost
        * own_eff
        * (0.4 + 0.6 * faster + 0.5 * sub_hp_frac)
        * (1.0 - 0.7 * foe_has_phaze)
        * awake_factor;
    // 6: Active bulk/evasion boost x own HP x recovery/Sub sustain
    x[6] = pos_bulk_eva_boost
        * own_hp_frac
        * (0.4 + 0.6 * has_heal_move + 0.4 * sub_hp_frac)
        * (1.0 - 0.7 * foe_has_phaze)
        * awake_factor;
    // 7: Active Substitute protection x offensive/setup threat
    x[7] = sub_hp_frac * (own_eff + 0.5 * pos_boost_sum + 0.4 * has_setup_any) * awake_factor;
    // 8: Safe setup window x sweep potential when not yet maxed
    x[8] = has_setup_any
        * safe_window
        * own_eff.max(0.6 * bench_best_eff * has_self_switch)
        * (1.0 - pos_off_boost.min(1.0))
        * (1.0 - 0.7 * foe_has_phaze);
    // 9: Phazing (`force_switch`) countering foe boosts or Substitute
    x[9] = has_force_switch
        * (foe_pos_boost_sum + foe_sub_hp_frac)
        * (own_hp_frac - foe_eff + 0.35).clamp(0.0, 1.0)
        * awake_factor;
    // 10: Phazing (`force_switch`) + Spikes on foe side x foe bench HP
    x[10] = has_force_switch
        * foe_spikes_frac
        * (0.5 * foe_bench_hp_sum).min(1.0)
        * (own_hp_frac - foe_eff + 0.25).clamp(0.0, 1.0)
        * awake_factor;
    // 11: Screens (`Reflect`/`Light Screen`) x alive team bulk
    x[11] = own_screens_frac * (0.5 * own_hp_frac + 0.35 * bench_hp_sum).min(1.2);
    // 12: Safeguard x healthy setup/active bulk
    x[12] = own_safeguard_frac * (0.5 * own_hp_frac + 0.5 * has_setup_any);
    // 13: Sleep infliction window against unstatused, unprotected foe
    x[13] = has_sleep_move
        * (1.0 - foe_status_any)
        * (1.0 - foe_safeguard)
        * (1.0 - foe_sub_hp_frac.ceil().min(1.0))
        * (0.4 + 0.6 * faster)
        * awake_factor;
    // 14: Leech Seed / Curse drain on foe x own survival/sustain
    x[14] = foe_seeded_or_cursed * own_hp_frac * (0.6 + 0.4 * sub_hp_frac + 0.4 * has_heal_move);
    // 15: Fast KO / high-initiative offensive壓制 (`faster * own_eff` when surviving foe hit)
    x[15] = faster * (own_eff - 0.45).clamp(0.0, 1.0) * (own_hp_frac - foe_eff + 0.3).clamp(0.0, 1.0);

    // =========================================================================
    // x[16..32]: Non-negative Multi-Body Vulnerability / Discount Features (monotonic <= 0 for `side`)
    // =========================================================================
    let perish_turns = own_mon.timers[3];
    let perish_urgency = if (own_mon.vol_flags & vol_bit::PERISH_SONG) != 0 && perish_turns > 0 {
        (4.0 - (perish_turns.min(3) as f32)) / 3.0
    } else {
        0.0
    };
    let is_trapped = if (own_mon.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0 {
        1.0f32
    } else {
        0.0f32
    };
    let is_encored = if (own_mon.vol_flags & vol_bit::ENCORE) != 0 {
        1.0f32
    } else {
        0.0f32
    };
    let is_disabled = if is_dis { 1.0f32 } else { 0.0f32 };
    let is_confused = if (own_mon.vol_flags & vol_bit::CONFUSION) != 0 {
        1.0f32
    } else {
        0.0f32
    };
    let is_paralyzed = if own_st == Status::Par as u8 { 1.0f32 } else { 0.0f32 };
    let has_escalating_residual = if own_st == Status::Tox as u8
        || (own_mon.vol_flags & (vol_bit::LEECH_SEED | vol_bit::CURSE | vol_bit::NIGHTMARE)) != 0
    {
        1.0f32
    } else {
        0.0f32
    };

    let foe_is_trapped = if (foe_mon.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0 {
        1.0f32
    } else {
        0.0f32
    };

    // 16: Perish Song + Trapped impending faint penalty (only when trapped!)
    x[16] = perish_urgency * is_trapped;
    // 17: Perish Song forcing out accumulated boosts or Substitute (unless trapping foe)
    x[17] = perish_urgency
        * (1.0 - foe_is_trapped * (1.0 - is_trapped))
        * (0.6 * pos_boost_sum + 0.5 * sub_hp_frac);
    // 18: Mean Look / Spider Web trapped in losing active matchup when bench exists
    x[18] = is_trapped * (foe_eff - own_eff).clamp(0.0, 1.0) * (bench_hp_sum.min(1.0));
    // 19: Encore / Disable lock neutralizing active boosts/setup/offense
    x[19] = (is_encored + 0.6 * is_disabled).min(1.0)
        * (0.3 * own_eff + 0.7 * pos_boost_sum + 0.3 * has_setup_any);
    // 20: Active boosts nullified by sleep/freeze or slower imminent KO
    let slower_ko_risk = (1.0 - faster) * (foe_eff - own_hp_frac + 0.15).clamp(0.0, 1.0);
    x[20] = pos_boost_sum
        * (is_asleep_or_frozen * (1.0 - 0.5 * has_heal_move) + slower_ko_risk).min(1.2)
        * (1.0 - sub_hp_frac);
    // 21: Active boosts / Substitute exposed to healthy foe phazer
    x[21] = (pos_boost_sum + sub_hp_frac)
        * foe_has_phaze
        * (foe_hp_frac - own_eff + 0.35).clamp(0.0, 1.0);
    // 22: Active `threat` overcount when slower and in immediate KO range before moving
    x[22] = own_eff * slower_ko_risk * (1.0 - sub_hp_frac);
    // 23: Spikes hazard vulnerability when active is forced out by damage/phaze/Perish
    let forced_out_pressure =
        ((foe_eff - own_hp_frac + 0.2).clamp(0.0, 1.0) + 0.5 * foe_has_phaze + 0.5 * perish_urgency)
            .min(1.5);
    x[23] = own_spikes_frac * bench_grounded_cnt * forced_out_pressure;
    // 24: Confusion + paralysis / high-boost self-hit risk
    x[24] = is_confused * (0.35 + 0.45 * is_paralyzed + 0.4 * pos_atk);
    // 25: Escalating residual clock (Toxic / Leech Seed / Curse / Nightmare)
    x[25] = has_escalating_residual * (1.0 - 0.5 * own_eff).clamp(0.2, 1.0);
    // 26: Negative defense/spdef drops exposed to foe offense
    x[26] = neg_def_spd * foe_eff;
    // 27: Negative offense/speed/accuracy drops discounting active threat
    x[27] = neg_off_spe_acc * (0.3 + 0.7 * own_eff);
    // 28: Foe Substitute blocking weak attacks / status
    x[28] = foe_sub_hp_frac * (1.0 - (own_eff - 0.25).clamp(0.0, 1.0));
    // 29..=30: Reserved (zero; direct-damage bench deficit and PP penalties are inactive)
    x[29] = 0.0;
    x[30] = 0.0;
    // 31: Sleep / Freeze downtime exposed to foe setup or high damage
    x[31] = is_asleep_or_frozen * (0.5 * foe_eff + 0.5 * foe_pos_boost_sum).min(1.2);

    x
}

/// Extract rollout state vector $\phi_{\text{state}}(s) \in \mathbb{R}^{12}$ and per-choice
/// action capability vectors $\phi_{\text{act}}(s, c) \in \mathbb{R}^6$ for $c \in \text{cs}$.
/// Returns `None` if no legal action has non-zero residual capability features.
#[inline]
pub fn extract_rollout_state_and_actions(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    cs: &[u8],
    stay_best_dmg: f32,
) -> Option<([f32; ROLLOUT_STATE_DIM], [[f32; ROLLOUT_ACT_DIM]; 7])> {
    let att_slot = (cb.sides[side].active_idx as usize).min(2);
    let att_idx = side * 3 + att_slot;
    let def_side = 1 - side;
    let def_slot = (cb.sides[def_side].active_idx as usize).min(2);
    let def_idx = def_side * 3 + def_slot;
    let me = &cb.mons[att_idx];
    let foe = &cb.mons[def_idx];
    if me.hp() <= 0 || foe.hp() <= 0 {
        return None;
    }

    let maxhp = table.mons[att_idx].maxhp.max(1) as f32;
    let foe_maxhp = table.mons[def_idx].maxhp.max(1) as f32;
    let own_hp_frac = (me.hp() as f32) / maxhp;
    let foe_hp_frac = (foe.hp() as f32) / foe_maxhp;
    let bench_alive = cb.sides[side].alive_mask.count_ones() > 1;
    let foe_bench_alive = cb.sides[def_side].alive_mask.count_ones() > 1;

    let mut own_pos_boosts_i = 0i32;
    let mut foe_pos_boosts_i = 0i32;
    for i in 0..7 {
        if me.boosts[i] > 0 {
            own_pos_boosts_i += me.boosts[i] as i32;
        }
        if foe.boosts[i] > 0 {
            foe_pos_boosts_i += foe.boosts[i] as i32;
        }
    }
    let own_pos_boosts = (own_pos_boosts_i as f32 / 4.0).min(1.5);
    let foe_pos_boosts = (foe_pos_boosts_i as f32 / 4.0).min(1.5);
    let own_has_sub = if (me.vol_flags & vol_bit::SUBSTITUTE) != 0 {
        1.0f32
    } else {
        0.0f32
    };
    let foe_has_sub = if (foe.vol_flags & vol_bit::SUBSTITUTE) != 0 {
        1.0f32
    } else {
        0.0f32
    };

    // Quick scan: does any legal move have a non-damaging capability or is switch viable?
    let mut has_cap_move = false;
    let mut own_has_self_switch = 0.0f32;
    for &c in cs {
        if c < 4 {
            let ms = &table.move_specs[att_idx][c as usize];
            if ms.special_kind == 18 && bench_alive {
                own_has_self_switch = 1.0;
                has_cap_move = true;
            } else if (ms.special_kind == 7 && foe_bench_alive)
                || ms.boosts_self
                || matches!(ms.pseudo_kind, 5..=8)
                || ms.is_substitute
                || (ms.is_status && ms.primary_status != 0)
            {
                has_cap_move = true;
            }
        }
    }
    let can_vol_switch = stay_best_dmg < 0.25 && own_pos_boosts == 0.0 && own_has_sub == 0.0 && bench_alive;
    if !has_cap_move && !can_vol_switch {
        return None;
    }

    let foe_best_dmg = cb
        .best_hit_fraction(table, def_side, def_slot, side, att_slot, true)
        .min(1.5) as f32;
    let own_spd = cb.mon_action_speed(table, side, att_slot);
    let foe_spd = cb.mon_action_speed(table, def_side, def_slot);
    let speed_adv = match own_spd.cmp(&foe_spd) {
        std::cmp::Ordering::Greater => 1.0f32,
        std::cmp::Ordering::Less => -1.0f32,
        std::cmp::Ordering::Equal => 0.0f32,
    };

    let mut bench_best_stat = 0.0f32;
    let mut bench_best_eff = 0.0f32;
    if bench_alive {
        for slot in 0..3usize {
            if slot == att_slot || (cb.sides[side].alive_mask & (1 << slot)) == 0 {
                continue;
            }
            let b_idx = side * 3 + slot;
            if cb.mons[b_idx].hp() <= 0 {
                continue;
            }
            let b_hp = (cb.mons[b_idx].hp() as f32) / (table.mons[b_idx].maxhp.max(1) as f32);
            let atk_s = cb.mon_stat(table, side, slot, 0, true, false) as f32;
            let spa_s = cb.mon_stat(table, side, slot, 2, true, false) as f32;
            bench_best_stat = bench_best_stat.max((b_hp * atk_s.max(spa_s) / 200.0).min(1.5));
            let eff = cb
                .best_hit_fraction(table, side, slot, def_side, def_slot, true)
                .min(1.5) as f32;
            bench_best_eff = bench_best_eff.max(eff);
        }
    }

    let phi_state = [
        1.0,
        own_hp_frac,
        foe_hp_frac,
        stay_best_dmg.min(1.5),
        foe_best_dmg,
        speed_adv,
        own_pos_boosts,
        foe_pos_boosts,
        own_has_sub,
        foe_has_sub,
        own_has_self_switch,
        bench_best_stat,
    ];

    let is_ghost = (table.mons[att_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
    let mut foe_has_slp = false;
    for p_i in 0..3usize {
        if (cb.sides[def_side].alive_mask & (1 << p_i)) != 0 {
            let m = &cb.mons[def_side * 3 + p_i];
            if m.hp() > 0 && m.status_raw() == Status::Slp as u8 {
                foe_has_slp = true;
                break;
            }
        }
    }

    let mut act_feats = [[0.0f32; ROLLOUT_ACT_DIM]; 7];
    let mut any_nonzero = false;
    for (i, &c) in cs.iter().take(7).enumerate() {
        let a = &mut act_feats[i];
        if c < 4 {
            let ms = &table.move_specs[att_idx][c as usize];
            // a[0]: self_switch transfer
            if ms.special_kind == 18 && bench_alive && (own_pos_boosts > 0.0 || own_has_sub > 0.0) {
                let receiver_edge =
                    (bench_best_stat.max(bench_best_eff) - stay_best_dmg + 0.5).clamp(0.1, 1.5);
                a[0] = (own_pos_boosts + 0.6 * own_has_sub) * receiver_edge;
                any_nonzero = true;
            }
            // a[1]: force_switch phaze against boosted/Substitute foe
            if ms.special_kind == 7
                && foe_bench_alive
                && own_hp_frac > foe_best_dmg
                && (foe_pos_boosts > 0.0 || foe_has_sub > 0.0)
            {
                a[1] = foe_pos_boosts + 0.75 * foe_has_sub;
                any_nonzero = true;
            }
            // a[2]: self-boost setup when healthy and un-maxed (< +2)
            if own_hp_frac > foe_best_dmg + 0.15 && stay_best_dmg < 0.85 {
                let mut b_gain = 0.0f32;
                if ms.boosts_self {
                    for &(st_i, d) in &ms.boosts {
                        if (st_i as usize) < 7 && d > 0 && me.boosts[st_i as usize] < 2 {
                            b_gain += d as f32 * 0.5;
                        }
                    }
                } else if ms.pseudo_kind == 5 && !is_ghost && me.boosts[0] < 2 {
                    b_gain = 0.75;
                } else if ms.pseudo_kind == 8 && me.boosts[0] < 2 && own_hp_frac > 0.6 {
                    b_gain = 1.5;
                }
                if b_gain > 0.0 {
                    a[2] = b_gain.min(1.5);
                    any_nonzero = true;
                }
            }
            // a[3]: Substitute creation when faster or safe
            if ms.is_substitute
                && own_has_sub == 0.0
                && own_hp_frac > 0.35
                && (speed_adv > 0.0 || foe_best_dmg < 0.25)
                && stay_best_dmg < 0.85
            {
                a[3] = 1.0;
                any_nonzero = true;
            }
            // a[4]: Status infliction (Sleep / Paralysis / Toxic) on healthy foe
            if ms.is_status
                && ms.primary_status != 0
                && foe.status_raw() == 0
                && foe_has_sub == 0.0
                && cb.sides[def_side].safeguard_turns == 0
                && stay_best_dmg < 0.85
            {
                let slp_blocked = ms.primary_status == Status::Slp as u8 && foe_has_slp;
                let ground_blocked = ms.move_type == table.electric_type
                    && (table.mons[def_idx].type_mask & (1u32 << (table.ground_type & 31))) != 0;
                if !slp_blocked && !ground_blocked {
                    let w_st = if ms.primary_status == Status::Slp as u8 {
                        1.0
                    } else {
                        0.6
                    };
                    a[4] = (ms.base_acc_pct as f32 / 100.0) * w_st;
                    any_nonzero = true;
                }
            }
        } else if (4..7).contains(&c) && can_vol_switch {
            let cand_slot = (c - 4) as usize;
            let cand_idx = side * 3 + cand_slot;
            let cand_hp =
                (cb.mons[cand_idx].hp() as f32) / (table.mons[cand_idx].maxhp.max(1) as f32);
            let cand_off = cb
                .best_hit_fraction(table, side, cand_slot, def_side, def_slot, true)
                .min(1.2) as f32;
            let foe_vs_cand = cb
                .best_hit_fraction(table, def_side, def_slot, side, cand_slot, true)
                .min(1.2) as f32;
            if cand_hp > foe_vs_cand + 0.2 && cand_off > stay_best_dmg + 0.25 {
                a[5] = (cand_off - stay_best_dmg - 0.5 * foe_vs_cand).max(0.0);
                any_nonzero = true;
            }
        }
    }

    if !any_nonzero {
        return None;
    }
    Some((phi_state, act_feats))
}

/// Evaluate the rank-4 bilinear rollout action-residual $\Delta Q_\psi(s, c)$ for all $c \in \text{cs}$.
#[inline]
pub fn compute_rollout_action_residuals(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    cs: &[u8],
    stay_best_dmg: f32,
) -> Option<[f32; 7]> {
    let m = &active_residual_bundle().rollout;
    if !m.active {
        return None;
    }
    let (phi_s, act_feats) =
        extract_rollout_state_and_actions(cb, table, side, cs, stay_best_dmg)?;
    let mut u = [0.0f32; ROLLOUT_RANK];
    for r in 0..ROLLOUT_RANK {
        let row = &m.u_proj[r];
        let mut s = 0.0f32;
        for j in 0..ROLLOUT_STATE_DIM {
            s += row[j] * phi_s[j];
        }
        u[r] = s;
    }
    let mut w_eff = m.act_bias;
    for r in 0..ROLLOUT_RANK {
        let ur = u[r];
        let v_row = &m.v_proj[r];
        for a in 0..ROLLOUT_ACT_DIM {
            w_eff[a] += ur * v_row[a];
        }
    }
    let mut deltas = [0.0f32; 7];
    for i in 0..cs.len().min(7) {
        let af = &act_feats[i];
        let mut dq = 0.0f32;
        for a in 0..ROLLOUT_ACT_DIM {
            dq += w_eff[a] * af[a];
        }
        deltas[i] = dq;
    }
    Some(deltas)
}

#[cfg(test)]
mod residual_tests {
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
    fn test_default_residual_zero_identity_and_monotonicity() {
        let m = &DEFAULT_RESIDUAL_BUNDLE.eval;
        assert!(m.active, "DEFAULT_RESIDUAL_BUNDLE.eval must be active");
        assert!(
            !DEFAULT_RESIDUAL_BUNDLE.rollout.active,
            "DEFAULT_RESIDUAL_BUNDLE.rollout must be inactive"
        );
        let z = [0.0f32; EVAL_FEAT_DIM];
        assert_eq!(
            eval_residual_forward(&z, m),
            0.0,
            "f(0) must be identically 0.0"
        );

        // Synergy features x[0..16] must be non-decreasing (>= 0).
        for j in 0..16 {
            let mut x = [0.0f32; EVAL_FEAT_DIM];
            x[j] = 1.0;
            let val = eval_residual_forward(&x, m);
            assert!(
                val >= 0.0,
                "synergy feature x[{j}] produced negative residual {val}"
            );
        }
        // Vulnerability features x[16..32] must be non-increasing (<= 0).
        for j in 16..32 {
            let mut x = [0.0f32; EVAL_FEAT_DIM];
            x[j] = 1.0;
            let val = eval_residual_forward(&x, m);
            assert!(
                val <= 0.0,
                "vulnerability feature x[{j}] produced positive residual {val}"
            );
        }
    }

    #[test]
    fn test_residual_exact_antisymmetry_and_baton_pass_transfer() {
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

        let mut b_ba = Battle::from_fixture(&dex, "1,2,3,4", &p2_team, &p1_team).unwrap();
        b_ba.choose(&dex, 0, "team 1,2,3").unwrap();
        b_ba.choose(&dex, 1, "team 1,2,3").unwrap();
        b_ba.choose(&dex, 0, "move 3").unwrap();
        b_ba.choose(&dex, 1, "move 1").unwrap();

        let (tbl_ab, cb_ab) = CompactBattle::from_battle(&b_ab, &dex);
        let (tbl_ba, cb_ba) = CompactBattle::from_battle(&b_ba, &dex);
        let (_, eff_ab) = compact_base_logit_and_eff(&cb_ab, &tbl_ab, &EvalWeights::default());
        let (_, eff_ba) = compact_base_logit_and_eff(&cb_ba, &tbl_ba, &EvalWeights::default());
        let sym_ab = compact_eval_residual_sym(&cb_ab, &tbl_ab, eff_ab);
        let sym_ba = compact_eval_residual_sym(&cb_ba, &tbl_ba, eff_ba);
        assert!(
            (sym_ab + sym_ba).abs() < 1e-6,
            "antisymmetry violated: sym_ab={sym_ab}, sym_ba={sym_ba}"
        );
        assert!(
            sym_ab > 0.0,
            "Scizor +2 Atk with Baton Pass + Marowak bench vs resting Snorlax should have positive residual sym_ab={sym_ab}"
        );

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

        // 1. Exact side-swap antisymmetry on Curse (+1 Atk, +1 Def, -1 Spe)
        let d_ab = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        let d_ba = compact_blindspot_diff(&cb_ba, &tbl_ba, &w);
        assert!(
            (d_ab + d_ba).abs() < 1e-12,
            "blindspot diff must be antisymmetric: d_ab={d_ab}, d_ba={d_ba}"
        );
        assert!(
            d_ab > 0.0,
            "Snorlax Curse [A+1,D+1,S-1] vs resting Zapdos must have positive blindspot diff: {d_ab}"
        );
        let v_ab = compact_eval01_blindspot(&cb_ab, &tbl_ab, &w);
        let v_ba = compact_eval01_blindspot(&cb_ba, &tbl_ba, &w);
        assert!(
            (v_ab + v_ba - 1.0).abs() < 1e-12,
            "compact_eval01_blindspot must sum to 1.0 across side swap: v_ab={v_ab}, v_ba={v_ba}"
        );

        // 2. Stack a second Curse ([A+2, D+2, S-2]): capital increases, discounted against active Whirlwind phazer,
        // and zeroed under Perish Song or Confusion.
        cb_ab.mons[0].boosts[0] = 2;
        cb_ab.mons[0].boosts[1] = 2;
        cb_ab.mons[0].boosts[4] = -2;
        let d_curse2 = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        assert!(
            d_curse2 > d_ab,
            "[A+2,D+2,S-2] must accrue more setup capital than [A+1,D+1,S-1]: {d_curse2} vs {d_ab}"
        );

        cb_ab.sides[1].active_idx = 1; // Skarmory with Whirlwind
        let d_vs_phazer = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        assert!(
            d_vs_phazer > 0.0 && d_vs_phazer < d_curse2,
            "active Whirlwind phazer must discount setup capital: {d_vs_phazer} vs {d_curse2}"
        );

        cb_ab.sides[1].active_idx = 2; // Misdreavus with Perish Song
        let d_vs_perish = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        assert_eq!(
            d_vs_perish, 0.0,
            "facing active Perish Song user without Baton Pass must zero setup capital"
        );
        cb_ab.sides[1].active_idx = 0;

        cb_ab.mons[0].vol_flags |= vol_bit::CONFUSION;
        assert!(
            compact_blindspot_diff(&cb_ab, &tbl_ab, &w) < 0.0,
            "confused mon must not receive setup capital and must incur confusion liability"
        );
        cb_ab.mons[0].vol_flags &= !vol_bit::CONFUSION;

        // 3. Frozen asset write-down (both active and bench; switching out a frozen mon improves equity)
        // & Healer Toxic impairment (discounted by Heal Bell cleric)
        cb_ab.mons[0].boosts = [0; 7];
        cb_ab.mons[1].set_status(Status::Frz as u8, 0); // Starmie frozen
        let d_frz_with_cleric = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        assert!(
            d_frz_with_cleric < 0.0,
            "benched Frz must write down asset value: {d_frz_with_cleric}"
        );
        cb_ab.sides[0].active_idx = 1; // Frozen Starmie active
        let v_frz_active = compact_eval01_blindspot(&cb_ab, &tbl_ab, &w);
        cb_ab.sides[0].active_idx = 0; // Switch to healthy Snorlax
        let v_frz_benched = compact_eval01_blindspot(&cb_ab, &tbl_ab, &w);
        assert!(
            v_frz_benched > v_frz_active,
            "switching out a frozen active mon to a healthy teammate must improve equity: {v_frz_benched} vs {v_frz_active}"
        );
        cb_ab.sides[0].alive_mask &= !(1 << 2); // Faint Miltank (Heal Bell cleric)
        let d_frz_no_cleric = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        assert!(
            d_frz_no_cleric < d_frz_with_cleric,
            "losing Heal Bell cleric must deepen Frz write-down: {d_frz_no_cleric} vs {d_frz_with_cleric}"
        );

        cb_ab.mons[1].set_status(Status::Tox as u8, 1); // Recover Starmie badly poisoned
        let d_tox_healer = compact_blindspot_diff(&cb_ab, &tbl_ab, &w);
        assert!(
            d_tox_healer < 0.0,
            "Recover wall with Toxic must incur poison liability: {d_tox_healer}"
        );

        // 4. Exact 1v1 invariance (`compact_eval01_blindspot == compact_eval01`)
        cb_ab.sides[0].alive_mask = 0b001;
        cb_ab.sides[1].alive_mask = 0b001;
        cb_ab.mons[0].boosts[0] = 2;
        cb_ab.mons[0].boosts[1] = 2;
        assert_eq!(
            compact_blindspot_diff(&cb_ab, &tbl_ab, &w),
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

