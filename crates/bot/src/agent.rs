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
pub struct MaxDamageAgent;

impl MaxDamageAgent {
    pub fn new() -> Self {
        MaxDamageAgent
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
        "maxdamage".into()
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
        let att = battle.active_id(side).map(|id| battle.poke(id));
        let def = battle.active_id(1 - side).map(|id| battle.poke(id));
        let (Some(att), Some(def)) = (att, def) else {
            return choices[0];
        };
        choices
            .iter()
            .copied()
            .filter(|c| matches!(c, SearchChoice::Move(_)))
            .max_by(|a, b| {
                let f = |c: &SearchChoice| match c {
                    SearchChoice::Move(id) => Self::move_score(dex, att, def, *id),
                    _ => -1.0,
                };
                f(a).total_cmp(&f(b))
            })
            .unwrap()
    }
}
