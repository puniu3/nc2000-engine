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

use std::sync::OnceLock;

use nc2000_engine::dex::{Accuracy, Category, Dex, FixedDamage, MoveId, Multihit};
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
    let mut diff = side_score(b, dex, w, 0) - side_score(b, dex, w, 1);
    if w.race != 0.0 {
        diff += w.race * race_margin(b, dex, w);
    }
    if w.exchange != 0.0 {
        diff += w.exchange * exchange_margin(b, dex, w);
    }
    1.0 / (1.0 + (-w.scale * diff).exp())
}

/// KO-race margin from side 0's perspective, in [-1, 1]; nonzero only in
/// last-mon-vs-last-mon states (no switches — a pure race).
///
/// Turns-to-KO come from `best_hit_fraction` (expected per-use fraction of
/// the foe's current HP, accuracy folded in), adjusted by the mechanical
/// state the certified anchors proved decisive (the v1 estimate misfired
/// on exactly these): a recharge lock wastes the locked side's turn, a
/// Substitute absorbs one hit, sleep sidelines the sleeper for its
/// remaining clock UNLESS it carries usable Sleep Talk, freeze (25/256
/// thaw) pushes past the race window, and poison/burn residual caps a
/// side's survival regardless of the foe's attacks. Speed order breaks
/// ties: ±0.5 turn; a full one-turn advantage saturates the margin.
pub fn race_margin(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let alive = |s: usize| {
        b.sides[s]
            .party
            .iter()
            .filter(|&&sl| {
                let p = &b.sides[s].roster[sl as usize];
                !p.fainted && p.hp > 0
            })
            .count()
    };
    if alive(0) != 1 || alive(1) != 1 {
        return 0.0;
    }
    let (Some(id0), Some(id1)) = (b.active_id(0), b.active_id(1)) else {
        return 0.0;
    };
    pair_margin(b, dex, w, id0, id1)
}

/// The exchange between ONE pair of mons, from `id0`'s perspective, in
/// [-1, 1]: who wins the straight trade and by how much, through the visible
/// mechanics (heal cycles, Substitute, recharge, sleep/freeze downtime,
/// residual self-death clocks, speed order). Returns 0 — "no claim" — when the
/// race runs past three turns, which is stall/heal territory where a
/// turns-to-KO estimate lies.
///
/// Split out of `race_margin` so the same exchange can be read for pairs that
/// are not both on the field: that is what `exchange_margin` needs.
pub fn pair_margin(b: &Battle, dex: &Dex, w: &EvalWeights, id0: PokeId, id1: PokeId) -> f64 {
    pair_margin_ctx(b, dex, w, id0, id1, [1.0, 1.0], [0.0, 0.0])
}

/// `pair_margin` for a pair that may not be on the field yet.
///
/// `hp_left[i]` is the fraction of its *current* HP that mon `i` brings to the
/// exchange (1.0 on the field; less for a benched mon that owes entry damage),
/// and `extra[i]` is the turns it spends getting there (1.0 for the switch).
/// Both default to the on-field values through `pair_margin`, and at those
/// values every expression below reduces to the original arithmetic — the
/// Phase A / race term is unchanged bit-for-bit unless `exchange_v2` is set.
fn pair_margin_ctx(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
    id0: PokeId,
    id1: PokeId,
    hp_left: [f64; 2],
    extra: [f64; 2],
) -> f64 {
    let (p0, p1) = (b.poke(id0), b.poke(id1));
    if p0.fainted || p0.hp <= 0 || p1.fainted || p1.hp <= 0 {
        return 0.0;
    }
    // A mon that cannot survive its own entry has lost the pairing before it
    // gets a turn; both dying is no claim either way.
    match (hp_left[0] <= 0.0, hp_left[1] <= 0.0) {
        (true, true) => return 0.0,
        (true, false) => return -1.0,
        (false, true) => return 1.0,
        _ => {}
    }

    let recharge = dex.conds_id("mustrecharge");
    let sleeptalk = dex.moves.id("sleeptalk");
    let heal_ids: Vec<MoveId> = ["rest", "recover", "softboiled", "milkdrink"]
        .iter()
        .filter_map(|k| dex.moves.id(k))
        .collect();
    let spd0 = b.get_pokemon_action_speed(dex, id0);
    let spd1 = b.get_pokemon_action_speed(dex, id1);
    // expected turns for `att` to KO `def` through the visible mechanics
    let turns = |att: PokeId, def: PokeId, att_faster: bool, def_left: f64, att_extra: f64| -> f64 {
        let mut e = best_hit_fraction(b, dex, att, def, w.couple_evasion);
        // A confused attacker spends half its turns hitting itself, so the same
        // race takes about twice as long (`conditions.rs`: the clock is
        // decremented before the move and the self-hit is a 1-in-2 coin).
        if w.exchange_v2 && confusion_attempts(b, dex, att) > 0.0 {
            e *= 0.5;
        }
        if e <= 1e-9 {
            return f64::INFINITY;
        }
        // A defender with a usable self-heal cannot be raced down unless
        // the attacker out-damages the heal cycle (~half max HP per turn:
        // Rest refills everything but donates two free turns) — the duel
        // gate measured the heal-blind version losing 0.39, this format
        // Rests everywhere. Exemption: a FASTER attacker with an expected
        // one-shot kills before any heal resolves (threshold mining caught
        // the un-exempted rule voiding Kadabra-vs-3HP-Articuno).
        let d = b.poke(def);
        if d.move_slots.iter().any(|m| m.pp > 0 && !m.disabled && heal_ids.contains(&m.id)) {
            // Per-hit damage as a fraction of MAX hp, i.e. what the heal cycle
            // is being out-raced. Deliberately NOT scaled by `def_left`: entry
            // damage takes HP off the defender, it does not weaken the
            // attacker, and scaling here made the term non-monotone (a Spikes
            // tick could push the attacker under the heal threshold and hand it
            // an INFINITY, so hazards read as *good* for the side that owns
            // them — caught by `spikes_charges_the_bench_in_phase_b_only`).
            let dmg_frac_max = e * d.hp as f64 / d.maxhp as f64;
            // One-shot means the hit covers what is left after entry damage.
            let kill_now = e >= def_left && att_faster;
            if dmg_frac_max < 0.5 && !kill_now {
                return f64::INFINITY;
            }
        }
        let mut t = (def_left / e).ceil() + att_extra;
        let a = b.poke(att);
        // a standing Substitute eats one hit
        if let Some(sub) = substitute_id(dex) {
            if b.poke(def).has_volatile(sub) {
                t += 1.0;
            }
        }
        // recharge lock: the locked side spends a turn doing nothing
        if let Some(rc) = recharge {
            if a.has_volatile(rc) {
                t += 1.0;
            }
        }
        t += match a.status {
            Status::Slp => {
                let talks = sleeptalk
                    .is_some_and(|st| a.move_slots.iter().any(|m| m.id == st && m.pp > 0));
                if talks {
                    0.0
                } else {
                    a.status_state.get_int(nc2000_engine::state::DK::Time).clamp(1, 4) as f64
                }
            }
            Status::Frz => 4.0,
            _ => 0.0,
        };
        t
    };
    // residual (psn/tox/brn) self-death clock: 1/8 maxhp per turn (tox floor)
    let surv = |x: PokeId, left: f64| -> f64 {
        let p = b.poke(x);
        if !matches!(p.status, Status::Psn | Status::Tox | Status::Brn) {
            return f64::INFINITY;
        }
        let hp = p.hp as f64 * left;
        if !w.exchange_v2 {
            let tick = (p.maxhp as f64 / 8.0).max(1.0);
            return (hp / tick).ceil();
        }
        // Engine parity (`conditions.rs::residualdmg`): once the toxic
        // volatile is on the mon the tick is `floor(maxhp/16) × counter` with
        // the counter incrementing every turn — and gen 2 keeps that counter
        // for psn/brn too. So the clock ACCELERATES, which is the whole
        // difference between "Toxic just landed" and "Toxic is on turn six"
        // that the flat `tox` weight cannot express. Without the volatile it
        // is the flat `floor(maxhp/8)` tick.
        match residual_counter(b, dex, x) {
            Some(c) => {
                let unit = (p.maxhp as f64 / 16.0).floor().max(1.0);
                let mut acc = 0.0;
                for k in 1..=8u32 {
                    acc += unit * (c + k as i64) as f64;
                    if acc >= hp {
                        return k as f64;
                    }
                }
                f64::INFINITY // past the horizon this term is allowed to claim
            }
            None => {
                let tick = (p.maxhp as f64 / 8.0).floor().max(1.0);
                (hp / tick).ceil()
            }
        }
    };
    // A side's effective "foe down" time: its own kill plan — void if its
    // residual kills it first (the b293 case: a toxed racer that dies on
    // its own clock never finishes a 2-turn plan) — or the foe rotting on
    // the foe's residual with no hit needed at all.
    let (k0, k1) = (
        turns(id0, id1, spd0 > spd1, hp_left[1], extra[0]),
        turns(id1, id0, spd1 > spd0, hp_left[0], extra[1]),
    );
    let (v0, v1) = (surv(id0, hp_left[0]), surv(id1, hp_left[1]));
    let k0 = if k0 <= v0 { k0 } else { f64::INFINITY };
    let k1 = if k1 <= v1 { k1 } else { f64::INFINITY };
    let t0 = k0.min(v1);
    let t1 = k1.min(v0);
    if t0.min(t1) > 3.0 {
        return 0.0; // long race: healing/stall dominates, no claim
    }
    let diff = match (t0.is_finite(), t1.is_finite()) {
        (true, false) => 2.0,
        (false, true) => -2.0,
        _ => {
            let (s0, s1) = (spd0, spd1);
            let edge = match s0.cmp(&s1) {
                std::cmp::Ordering::Greater => 0.5,
                std::cmp::Ordering::Less => -0.5,
                std::cmp::Ordering::Equal => 0.0,
            };
            (t1 - t0) + edge
        }
    };
    diff.clamp(-1.0, 1.0)
}

/// Phase A of the exchange eval: the same trade computation as `race_margin`,
/// read over EVERY living pair instead of only a last-mon-vs-last-mon race.
///
/// The additive material terms cannot see a matchup at all — which is why the
/// eval has no opinion about switching (measured: the bot's root switch mass
/// is at or below uniform, and its agreement on switch decisions matches
/// winners and losers equally). A matchup is not a property of either mon; it
/// is a property of the pair, and the position's value is what the two sides
/// can force in the matrix of pairs.
///
/// Aggregation depends on `exchange_v2`. Phase A (default) takes the cheap
/// proxy: the mean of maximin and minimax, exact when the matrix has a saddle
/// point and erring toward the committed side otherwise. Phase B takes the
/// matrix's real game value and charges entry costs — see `exchange_matrix`
/// and `matrix_game_value`.
///
/// Pairs whose race is undecidable return 0 from `pair_margin`, so a position
/// full of stall matchups scores neutral rather than inventing a claim.
pub fn exchange_margin(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let Some((mine, theirs, cells)) = exchange_matrix(b, dex, w) else {
        return 0.0;
    };
    let (k0, k1) = (mine.len(), theirs.len());
    let row = |i: usize| &cells[i * k1..(i + 1) * k1];
    if w.exchange_v2 {
        // Phase B: the actual game value.
        return matrix_game_value(&cells, [k0, k1]);
    }
    // Phase A: maximin/minimax midpoint.
    let maximin = (0..k0)
        .map(|i| row(i).iter().cloned().fold(f64::INFINITY, f64::min))
        .fold(f64::NEG_INFINITY, f64::max);
    let minimax = (0..k1)
        .map(|j| (0..k0).map(|i| cells[i * k1 + j]).fold(f64::NEG_INFINITY, f64::max))
        .fold(f64::INFINITY, f64::min);
    0.5 * (maximin + minimax)
}

/// The exchange matrix behind `exchange_margin`: side 0's living mons as rows,
/// side 1's as columns, cells row-major in margin space [-1, 1]. `None` when
/// either side has nothing living.
///
/// Under `exchange_v2` a cell for a mon that is not already on the field is
/// charged what getting there costs — the switch turn, and the Spikes it eats
/// on arrival — so an off-diagonal cell states the value of a switch rather
/// than of a free rearrangement. That charge is the first channel through which
/// switching reaches `eval01` at all.
///
/// Public because it is the term's diagnostic surface: a single number cannot
/// be argued with, and the calibration work needs to see which pairing the eval
/// thinks it is being paid for.
pub fn exchange_matrix(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
) -> Option<(Vec<PokeId>, Vec<PokeId>, Vec<f64>)> {
    let living = |s: usize| -> Vec<PokeId> {
        b.sides[s]
            .party
            .iter()
            .filter_map(|&sl| {
                let p = &b.sides[s].roster[sl as usize];
                (!p.fainted && p.hp > 0).then_some(PokeId { side: s as u8, slot: sl })
            })
            .collect()
    };
    let (mine, theirs) = (living(0), living(1));
    if mine.is_empty() || theirs.is_empty() {
        return None;
    }
    let active = [b.active_id(0).map(|i| i.slot), b.active_id(1).map(|i| i.slot)];
    let on_field = |id: PokeId| active[id.side as usize] == Some(id.slot);
    // Phase A read every pair as if it were already the matchup on the field.
    let left = |id: PokeId| {
        if !w.exchange_v2 || on_field(id) {
            1.0
        } else {
            1.0 - entry_loss(b, dex, id)
        }
    };
    let extra = |id: PokeId| if w.exchange_v2 && !on_field(id) { 1.0 } else { 0.0 };
    let mut cells = Vec::with_capacity(mine.len() * theirs.len());
    for &a in mine.iter() {
        for &d in theirs.iter() {
            cells.push(pair_margin_ctx(b, dex, w, a, d, [left(a), left(d)], [extra(a), extra(d)]));
        }
    }
    Some((mine, theirs, cells))
}

/// RM+ sweeps for the exchange matrix. It is at most 3×3 and the payoffs are
/// bounded in [0, 1], where RM+ converges in tens of iterations; the search
/// spends orders of magnitude more per leaf inside `best_hit_fraction`, and the
/// Phase A duel measured the 9-pair read at identical think time to the
/// 1-pair one (278 vs 280 ms/move), so this is not the term's cost centre.
const EXCHANGE_SWEEPS: u32 = 128;

/// Phase B's aggregation: the zero-sum **game value** of the pair matrix,
/// instead of Phase A's maximin/minimax midpoint. Identical to the midpoint
/// wherever the matrix has a saddle point (and to the single cell at 1×1);
/// where it does not, the midpoint splits the difference between two pure
/// commitments that neither side has to make, while the value is what the
/// position is actually worth under mixing.
///
/// `solve_rm_plus` wants side-0 payoffs in [0, 1] (it reads side 1's payoff as
/// `1 − u`), so margins are mapped in and the equilibrium value mapped back.
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

/// Fraction of a benched mon's *current* HP that switching in costs it: the
/// Spikes tick on its own side, mirroring `conditions.rs` exactly (Flying
/// immune, `AMOUNTS[layers]/24` of max HP). >= 1.0 means it faints on entry.
fn entry_loss(b: &Battle, dex: &Dex, id: PokeId) -> f64 {
    let p = b.poke(id);
    if p.has_type(dex.known_types.flying) {
        return 0.0;
    }
    let Some(sp) = spikes_id(dex) else { return 0.0 };
    let Some(st) = b.sides[id.side as usize].side_condition(sp) else { return 0.0 };
    const AMOUNTS: [f64; 4] = [0.0, 3.0, 4.0, 6.0];
    let layers = st.get_int(nc2000_engine::state::DK::Layers).clamp(0, 3) as usize;
    let dmg = (AMOUNTS[layers] * p.maxhp as f64 / 24.0).floor().max(1.0);
    dmg / p.hp.max(1) as f64
}

/// Remaining *hindered* move attempts for a confused mon (0.0 when it is not
/// confused, or when the clock runs out on its next move — `conditions.rs`
/// decrements before the move and lets that move through at 0).
fn confusion_attempts(b: &Battle, dex: &Dex, id: PokeId) -> f64 {
    let Some(cf) = confusion_id(dex) else { return 0.0 };
    let Some(vs) = b.poke(id).volatile(cf) else { return 0.0 };
    (vs.get_int(nc2000_engine::state::DK::Time).clamp(0, 5) as f64 - 1.0).max(0.0)
}

/// The gen-2 toxic counter, when this mon carries the `residualdmg` volatile
/// that makes its psn/brn/tox tick accelerate.
fn residual_counter(b: &Battle, dex: &Dex, id: PokeId) -> Option<i64> {
    let rd = residualdmg_id(dex)?;
    b.poke(id).volatile(rd).map(|vs| vs.get_int(nc2000_engine::state::DK::Counter))
}

/// Leaf value for search cutoffs. The calibrated probability is backed up
/// directly at alpha 1; smaller alpha shrinks it toward 0.5.
pub fn eval_leaf(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    assert!((0.0..=1.0).contains(&w.leaf_alpha), "leaf alpha must be in [0,1]");
    0.5 + w.leaf_alpha * (eval01(b, dex, w) - 0.5)
}

fn side_score(b: &Battle, dex: &Dex, w: &EvalWeights, s: usize) -> f64 {
    let side = &b.sides[s];
    let mut score = 0.0;
    let mut pp_num = 0.0;
    let mut pp_den = 0.0;
    // Spikes is paid on switch-IN, so only living benched non-Flying mons owe
    // it; the mon already on the field has paid or was never charged.
    let active_slot = b.active_id(s).map(|id| id.slot);
    let mut spikes_exposed = 0u32;
    for &slot in side.party.iter() {
        let p = &side.roster[slot as usize];
        if p.fainted || p.hp <= 0 {
            continue;
        }
        if active_slot != Some(slot) && !p.has_type(dex.known_types.flying) {
            spikes_exposed += 1;
        }
        score += w.alive + w.hp * p.hp as f64 / p.maxhp as f64;
        score -= match p.status {
            Status::Brn => w.brn,
            Status::Par => w.par,
            Status::Slp => {
                if w.slp_time_scale {
                    // remaining sleep clock (engine DK::Time, decremented per
                    // wake attempt): Rest = 3, natural = 2..=4 at onset
                    let t = p.status_state.get_int(nc2000_engine::state::DK::Time).clamp(0, 4);
                    w.slp * t as f64 / 3.0
                } else {
                    w.slp
                }
            }
            Status::Frz => w.frz,
            Status::Psn => w.psn,
            Status::Tox => w.tox,
            _ => 0.0,
        };
        for ms in p.move_slots.iter() {
            pp_num += ms.pp as f64;
            pp_den += ms.maxpp as f64;
        }
    }
    if pp_den > 0.0 {
        score += w.pp * pp_num / pp_den;
    }
    if w.spikes != 0.0 && spikes_exposed > 0 {
        if let Some(sp) = spikes_id(dex) {
            if let Some(st) = side.side_condition(sp) {
                // Engine parity (`conditions.rs` "spikes"/"onEntryHazard").
                const AMOUNTS: [f64; 4] = [0.0, 3.0, 4.0, 6.0];
                let layers = st.get_int(nc2000_engine::state::DK::Layers).clamp(0, 3);
                let frac = AMOUNTS[layers as usize] / 24.0;
                score -= w.spikes * frac * spikes_exposed as f64;
            }
        }
    }
    if let Some(id) = b.active_id(s) {
        let p = b.poke(id);
        if !p.fainted && p.hp > 0 {
            for i in 0..7 {
                score += w.boost[i] * p.boosts[i] as f64;
            }
            if w.substitute != 0.0 {
                if let Some(sub) = substitute_id(dex) {
                    if p.has_volatile(sub) {
                        score += w.substitute;
                    }
                }
            }
            // Confusion: the clock is what costs, not the label. See the
            // `confusion` weight — `0.5 × remaining hindered attempts` is the
            // expected number of turns the mon loses.
            if w.confusion != 0.0 {
                score -= w.confusion * 0.5 * confusion_attempts(b, dex, id);
            }
            if let Some(foe) = b.active_id(1 - s) {
                if !b.poke(foe).fainted && b.poke(foe).hp > 0 {
                    score +=
                        w.threat * best_hit_fraction(b, dex, id, foe, w.couple_evasion).min(1.0);
                }
            }
        }
    }
    score
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

fn hiddenpower_id(dex: &Dex) -> Option<MoveId> {
    static ID: OnceLock<Option<MoveId>> = OnceLock::new();
    *ID.get_or_init(|| dex.moves.id("hiddenpower"))
}

fn substitute_id(dex: &Dex) -> Option<nc2000_engine::dex::CondId> {
    static ID: OnceLock<Option<nc2000_engine::dex::CondId>> = OnceLock::new();
    *ID.get_or_init(|| dex.conds_id("substitute"))
}

fn spikes_id(dex: &Dex) -> Option<nc2000_engine::dex::CondId> {
    static ID: OnceLock<Option<nc2000_engine::dex::CondId>> = OnceLock::new();
    *ID.get_or_init(|| dex.conds_id("spikes"))
}

fn confusion_id(dex: &Dex) -> Option<nc2000_engine::dex::CondId> {
    static ID: OnceLock<Option<nc2000_engine::dex::CondId>> = OnceLock::new();
    *ID.get_or_init(|| dex.conds_id("confusion"))
}

fn residualdmg_id(dex: &Dex) -> Option<nc2000_engine::dex::CondId> {
    static ID: OnceLock<Option<nc2000_engine::dex::CondId>> = OnceLock::new();
    *ID.get_or_init(|| dex.conds_id("residualdmg"))
}

/// Expected fraction of the defender's *current* HP removed by one use of
/// `move_id`: gen-2 damage core on effective stats x STAB x effectiveness x
/// mean roll x mean hits x accuracy. 0 for status moves and unknowable
/// callback damage (counter/present/ohko score 0 — same class of caveat as
/// MaxDamage's static base powers).
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
    let ms = dex.move_static(move_id);
    let a = b.poke(att);
    let d = b.poke(def);

    let (move_type, base_power) = if Some(move_id) == hiddenpower_id(dex) {
        (a.hp_type, a.hp_power)
    } else {
        (ms.move_type, ms.base_power)
    };
    // M16c-L1: callback base powers the dex lists as 0. damage_conformance
    // measured `return` at exactly 0.0000 (432 samples; it lost game 3629) —
    // the formulas mirror moveexec::modify_base_power; magnitude/present use
    // the roll-distribution mean. counter/mirrorcoat/bide stay 0: reactive
    // damage is unknowable from a static position.
    let base_power = if base_power > 0 {
        base_power
    } else {
        match dex.moves.key(move_id) {
            "return" => a.happiness as i32 * 10 / 25,
            "frustration" => (255 - a.happiness as i32) * 10 / 25,
            "flail" | "reversal" => {
                let ratio = ((a.hp as f64 * 48.0 / a.maxhp as f64).floor() as i32).max(1);
                match ratio {
                    r if r < 2 => 200,
                    r if r < 5 => 150,
                    r if r < 10 => 100,
                    r if r < 17 => 80,
                    r if r < 33 => 40,
                    _ => 20,
                }
            }
            "magnitude" => 71,
            "present" => 40,
            _ => 0,
        }
    };

    let mut eff = 1.0f64;
    for dt in d.types.iter() {
        if dex.type_immune(move_type, dt) {
            return 0.0;
        }
        match dex.eff(move_type, dt) {
            1 => eff *= 2.0,
            -1 => eff *= 0.5,
            _ => {}
        }
    }

    let acc = if couple_evasion {
        // Real gen-2 accuracy roll (attacker accuracy stage × defender evasion
        // stage), matching how the engine actually rolls hits.
        b.hit_probability(dex, att, def, move_id)
    } else {
        // Pre-fix: base move accuracy only, blind to evasion stages.
        match ms.accuracy {
            Accuracy::AlwaysHits => 1.0,
            Accuracy::Pct(p) => p as f64 / 100.0,
        }
    };

    let raw = if let Some(fd) = &ms.damage {
        // Fixed damage: immunity applies (checked above), effectiveness does not.
        match fd {
            FixedDamage::Level => a.level as f64,
            FixedDamage::Amount(n) => *n as f64,
        }
    } else {
        if ms.category == Category::Status || base_power <= 0 {
            return 0.0;
        }
        let (ai, di) = match ms.category {
            Category::Physical => (0, 1),
            _ => (2, 3),
        };
        let atk = b.get_stat(dex, att, ai, false, false, false) as f64;
        let mut defense = b.get_stat(dex, def, di, false, false, false) as f64;
        // M16c-L1: Explosion/Selfdestruct halve the physical defense
        // (moveexec:2709); the eval's copy was measured at exactly 0.50 of
        // the engine's damage (damage_conformance, 2847 samples).
        if ms.selfdestruct && di == 1 {
            defense = (defense / 2.0).floor().max(1.0);
        }
        let core = ((a.level as f64 * 2.0 / 5.0 + 2.0).floor() * base_power as f64 * atk
            / defense
            / 50.0)
            .floor()
            + 2.0;
        let stab = if a.types.has(move_type) { 1.5 } else { 1.0 };
        let hits = match &ms.multihit {
            Some(Multihit::Fixed(n)) => *n as f64,
            Some(Multihit::Range(2, 5)) => 3.0, // gen-2 hit-count distribution mean
            Some(Multihit::Range(lo, hi)) => (*lo + *hi) as f64 / 2.0,
            None => 1.0,
        };
        core * stab * eff * (236.0 / 255.0) * hits
    };

    raw * acc / d.hp.max(1) as f64
}
