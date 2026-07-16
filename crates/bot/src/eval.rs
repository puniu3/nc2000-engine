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
}

impl Default for EvalWeights {
    /// Hand-written starting point (M6 pre-tuning); replaced by the SPSA
    /// result once tuning lands.
    fn default() -> Self {
        EvalWeights {
            hp: 1.0,
            alive: 0.5,
            brn: 0.35,
            par: 0.35,
            slp: 0.6,
            frz: 0.8,
            psn: 0.25,
            tox: 0.5,
            boost: [0.15, 0.10, 0.15, 0.10, 0.15, 0.10, 0.10],
            threat: 0.5,
            pp: 0.2,
            scale: 1.5,
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
        }
    }
}

/// Win-probability-shaped eval in (0, 1) from side 0's perspective.
pub fn eval01(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    let diff = side_score(b, dex, w, 0) - side_score(b, dex, w, 1);
    1.0 / (1.0 + (-w.scale * diff).exp())
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
            Status::Slp => w.slp,
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
            if let Some(foe) = b.active_id(1 - s) {
                if !b.poke(foe).fainted && b.poke(foe).hp > 0 {
                    score += w.threat * best_hit_fraction(b, dex, id, foe).min(1.0);
                }
            }
        }
    }
    score
}

/// Best expected hit fraction over the attacker's usable move slots
/// (unclamped: >1 = expected OHKO with margin).
pub fn best_hit_fraction(b: &Battle, dex: &Dex, att: PokeId, def: PokeId) -> f64 {
    let mut best = 0.0f64;
    for ms in b.poke(att).move_slots.iter() {
        if ms.pp <= 0 || ms.disabled {
            continue;
        }
        best = best.max(expected_hit_fraction(b, dex, att, def, ms.id));
    }
    best
}

fn hiddenpower_id(dex: &Dex) -> Option<MoveId> {
    static ID: OnceLock<Option<MoveId>> = OnceLock::new();
    *ID.get_or_init(|| dex.moves.id("hiddenpower"))
}

/// Expected fraction of the defender's *current* HP removed by one use of
/// `move_id`: gen-2 damage core on effective stats x STAB x effectiveness x
/// mean roll x mean hits x accuracy. 0 for status moves and unknowable
/// callback damage (counter/present/ohko score 0 — same class of caveat as
/// MaxDamage's static base powers).
pub fn expected_hit_fraction(b: &Battle, dex: &Dex, att: PokeId, def: PokeId, move_id: MoveId) -> f64 {
    let ms = dex.move_static(move_id);
    let a = b.poke(att);
    let d = b.poke(def);

    let (move_type, base_power) = if Some(move_id) == hiddenpower_id(dex) {
        (a.hp_type, a.hp_power)
    } else {
        (ms.move_type, ms.base_power)
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

    let acc = match ms.accuracy {
        Accuracy::AlwaysHits => 1.0,
        Accuracy::Pct(p) => p as f64 / 100.0,
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
        let defense = b.get_stat(dex, def, di, false, false, false) as f64;
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
