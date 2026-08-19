//! Agent trait + the two baseline agents (uniform random, max-damage
//! heuristic). Baselines exist to calibrate search agents: any search worth
//! keeping must beat MaxDamage, and MaxDamage must beat Random.

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{Category, Dex, MoveId};
use nc2000_engine::state::{Battle, Pokemon};

use crate::rng::SplitMix64;

pub trait Agent {
    fn name(&self) -> String;

    /// Pick one of `choices` for `side`. `choices` is non-empty and was
    /// enumerated by the caller via `Battle::legal_choices` at this exact
    /// state; the return value must be a member of it.
    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice;

    /// The agent's mixed policy at this decision point: probabilities aligned
    /// with `choices`, summing to 1. This is the distribution the agent
    /// actually plays — a best-response exploiter (M7 gate) queries it as its
    /// opponent model. Default: a point mass on whatever `choose` picks,
    /// which is exactly right for every argmax/deterministic agent.
    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        let pick = self.choose(battle, dex, side, choices);
        choices.iter().map(|&c| if c == pick { 1.0 } else { 0.0 }).collect()
    }
}

// ---------------------------------------------------------------- random

pub struct RandomAgent {
    rng: SplitMix64,
}

impl RandomAgent {
    pub fn new(seed: u64) -> Self {
        RandomAgent { rng: SplitMix64::new(seed) }
    }
}

impl Agent for RandomAgent {
    fn name(&self) -> String {
        "random".into()
    }

    fn choose(
        &mut self,
        _battle: &Battle,
        _dex: &Dex,
        _side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        choices[self.rng.below(choices.len())]
    }
}

// ------------------------------------------------------------ max damage

/// Static damage estimate: base power x STAB x type effectiveness. No
/// voluntary switches, default team-preview order, healthiest bench on a
/// forced switch. The classic calibration baseline.
///
/// **Two damage models, and why the broken one is still the default.** The
/// M5 model above reads the dex's `basePower` field, and in
/// `data/gen2stadium2.json` nineteen damaging moves carry `basePower` 0
/// because their power is computed by a callback: return, frustration,
/// flail, reversal, magnitude, present, counter, mirrorcoat, bide,
/// superfang, psywave, plain (untyped) hiddenpower, the fixed-damage moves
/// (seismictoss, nightshade, dragonrage, sonicboom) and the three OHKO
/// moves. `move_score` scores all of them 0.0 — indistinguishable from a
/// status move — and `max_by` then falls through to the LAST legal move.
/// A Return-only Miltank therefore never attacks at all. Measured over the
/// 570-battle corpus (`greedy_gap`, 20,719 decisions): the acting mon
/// carries at least one such move on 8.4% of decisions, and this agent
/// picks a different move from a conformant max-damage agent on 16.1%.
///
/// `MaxDamageAgent::conformant()` swaps the scorer for
/// `eval::expected_hit_fraction` — the same estimate the shipped rollout
/// policy uses (`mcts::greedy_pick`) and the one `examples/damage_conformance.rs`
/// gates against the engine's own damage core (38 moves, every mean ratio
/// inside +/-1.5% as of this commit). It fills in the callback base powers,
/// resolves plain Hidden Power's real type and power from the attacker's
/// DVs, halves physical defense for Explosion/Selfdestruct, and multiplies
/// by the real accuracy-vs-evasion roll.
///
/// `new()` keeps the broken model **on purpose**: `maxdamage` is the anchor
/// of every published strength number in the README ladder and the fixed
/// opponent in the `skuct:300 vs maxdamage seed 1 = 14W 6L` bit-identity
/// fingerprint that four milestone entries re-assert. Changing it in place
/// would silently invalidate all of them. Controls that want an honest
/// baseline ask for it by name.
pub struct MaxDamageAgent {
    /// `false` = the frozen M5 static model. `true` = `eval::expected_hit_fraction`.
    conformant: bool,
}

impl MaxDamageAgent {
    /// The frozen M5 baseline. Do not change its behaviour.
    pub fn new() -> Self {
        MaxDamageAgent { conformant: false }
    }

    /// Same policy, damage model replaced by the conformance-gated eval
    /// estimate. Arena/play spec `greedy`.
    pub fn conformant() -> Self {
        MaxDamageAgent { conformant: true }
    }

    fn move_score(dex: &Dex, att: &Pokemon, def: &Pokemon, id: MoveId) -> f64 {
        let ms = dex.move_static(id);
        if ms.category == Category::Status {
            return 0.0;
        }
        let mut mult = 1.0f64;
        for dt in def.types.iter() {
            if dex.type_immune(ms.move_type, dt) {
                return 0.0;
            }
            match dex.eff(ms.move_type, dt) {
                1 => mult *= 2.0,
                -1 => mult *= 0.5,
                _ => {}
            }
        }
        let stab = if att.types.has(ms.move_type) { 1.5 } else { 1.0 };
        // Callback-powered moves (return/flail/magnitude/...) score their
        // static base power; good enough for a baseline.
        ms.base_power as f64 * stab * mult
    }

    fn hp_frac(battle: &Battle, side: usize, display_pos: u8) -> f64 {
        let s = &battle.sides[side];
        let slot = s.party[(display_pos - 1) as usize];
        let p = &s.roster[slot as usize];
        p.hp as f64 / p.maxhp as f64
    }
}

impl Default for MaxDamageAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl Agent for MaxDamageAgent {
    fn name(&self) -> String {
        if self.conformant { "greedy".into() } else { "maxdamage".into() }
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        // team preview: keep the builder's lead order
        if matches!(choices[0], SearchChoice::Team(_)) {
            let default = SearchChoice::Team([1, 2, 3]);
            return if choices.contains(&default) { default } else { choices[0] };
        }

        let has_moves = choices.iter().any(|c| matches!(c, SearchChoice::Move(_)));
        if !has_moves {
            // forced switch (or pass): healthiest bench mon
            return choices
                .iter()
                .copied()
                .max_by(|a, b| {
                    let f = |c: &SearchChoice| match c {
                        SearchChoice::Switch(pos) => Self::hp_frac(battle, side, *pos),
                        _ => -1.0,
                    };
                    f(a).total_cmp(&f(b))
                })
                .unwrap();
        }

        // move request: strongest static hit; never switch voluntarily
        let (Some(att), Some(def)) = (battle.active_id(side), battle.active_id(1 - side)) else {
            return choices[0];
        };
        let conformant = self.conformant;
        choices
            .iter()
            .copied()
            .filter(|c| matches!(c, SearchChoice::Move(_)))
            .max_by(|a, b| {
                let f = |c: &SearchChoice| match c {
                    SearchChoice::Move(id) => {
                        if conformant {
                            // Ranking by expected fraction of the defender's
                            // current HP is order-identical to ranking by
                            // expected damage: the divisor is the same for
                            // every move at one decision.
                            crate::eval::expected_hit_fraction(battle, dex, att, def, *id, true)
                        } else {
                            Self::move_score(dex, battle.poke(att), battle.poke(def), *id)
                        }
                    }
                    _ => -1.0,
                };
                f(a).total_cmp(&f(b))
            })
            .unwrap()
    }
}
