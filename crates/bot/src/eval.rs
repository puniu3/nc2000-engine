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
        }
    }
}

/// Win-probability-shaped eval in (0, 1) from side 0's perspective.
pub fn eval01(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let mut diff = side_score(b, dex, w, 0) - side_score(b, dex, w, 1);
    if w.race != 0.0 {
        diff += w.race * race_margin(b, dex, w);
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
    let (p0, p1) = (b.poke(id0), b.poke(id1));
    if p0.fainted || p0.hp <= 0 || p1.fainted || p1.hp <= 0 {
        return 0.0;
    }

    let recharge = dex.conds_id("mustrecharge");
    let sleeptalk = dex.moves.id("sleeptalk");
    let heal_ids: Vec<MoveId> = ["rest", "recover", "softboiled", "milkdrink"]
        .iter()
        .filter_map(|k| dex.moves.id(k))
        .collect();
    // expected turns for `att` to KO `def` through the visible mechanics
    let turns = |att: PokeId, def: PokeId| -> f64 {
        let e = best_hit_fraction(b, dex, att, def, w.couple_evasion);
        if e <= 1e-9 {
            return f64::INFINITY;
        }
        // A defender with a usable self-heal cannot be raced down unless
        // the attacker out-damages the heal cycle (~half max HP per turn:
        // Rest refills everything but donates two free turns) — the duel
        // gate measured the heal-blind version losing 0.39, this format
        // Rests everywhere.
        let d = b.poke(def);
        if d.move_slots.iter().any(|m| m.pp > 0 && !m.disabled && heal_ids.contains(&m.id)) {
            let dmg_frac_max = e * d.hp as f64 / d.maxhp as f64;
            if dmg_frac_max < 0.5 {
                return f64::INFINITY;
            }
        }
        let mut t = (1.0 / e).ceil();
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
    let surv = |x: PokeId| -> f64 {
        let p = b.poke(x);
        if matches!(p.status, Status::Psn | Status::Tox | Status::Brn) {
            let tick = (p.maxhp as f64 / 8.0).max(1.0);
            (p.hp as f64 / tick).ceil()
        } else {
            f64::INFINITY
        }
    };
    // A side's effective "foe down" time: its own kill plan — void if its
    // residual kills it first (the b293 case: a toxed racer that dies on
    // its own clock never finishes a 2-turn plan) — or the foe rotting on
    // the foe's residual with no hit needed at all.
    let (k0, k1) = (turns(id0, id1), turns(id1, id0));
    let (v0, v1) = (surv(id0), surv(id1));
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
            let s0 = b.get_pokemon_action_speed(dex, id0);
            let s1 = b.get_pokemon_action_speed(dex, id1);
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

/// Leaf value for search cutoffs, squashed into (0.25, 0.75) so a static
/// judgment never outranks a real win/loss.
pub fn eval_leaf(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    0.25 + 0.5 * eval01(b, dex, w)
}

fn side_score(b: &Battle, dex: &Dex, w: &EvalWeights, s: usize) -> f64 {
    let side = &b.sides[s];
    let mut score = 0.0;
    let mut pp_num = 0.0;
    let mut pp_den = 0.0;
    for &slot in side.party.iter() {
        let p = &side.roster[slot as usize];
        if p.fainted || p.hp <= 0 {
            continue;
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
