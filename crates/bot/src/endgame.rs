use std::collections::HashMap;

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::{Deserialize, Serialize};

use crate::bounds::{
    BoundCache, BoundConfig, BoundSolver, Bounds, SolveReport, Stop, StrategyCertificate,
};
use crate::rng::SplitMix64;
use crate::smmcts::{RmConfig, SkuctSearch};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EndgameConfig {
    pub solver_fraction: f64,
    pub strategy_loss: f64,
    pub leaf_width: f64,
    pub max_alive: usize,
    pub max_nodes: usize,
    pub slice_runs: usize,
    pub state_runs: usize,
    pub decision_runs: usize,
}

impl Default for EndgameConfig {
    fn default() -> Self {
        Self {
            solver_fraction: 0.1,
            strategy_loss: 0.02,
            leaf_width: 0.02,
            max_alive: 2,
            max_nodes: 12_000,
            slice_runs: 64,
            state_runs: 16_384,
            decision_runs: 1_000_000,
        }
    }
}

impl EndgameConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.solver_fraction.is_finite()
            || !(0.0..=0.2).contains(&self.solver_fraction)
            || !self.strategy_loss.is_finite()
            || !(0.0..=0.02).contains(&self.strategy_loss)
            || !self.leaf_width.is_finite()
            || !(0.0..=0.02).contains(&self.leaf_width)
            || !(1..=2).contains(&self.max_alive)
            || self.max_nodes == 0
            || !(1..=64).contains(&self.slice_runs)
            || self.state_runs == 0
            || self.decision_runs == 0
        {
            return Err("invalid endgame limits".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct EndgameStats {
    pub calls: usize,
    pub cache_hits: usize,
    pub value_certificates: usize,
    pub strategy_certificates: usize,
    pub value_replacements: usize,
    pub root_overrides: usize,
    pub root_changes: usize,
    pub unresolved: usize,
    pub node_stops: usize,
    pub runs: usize,
    pub solver_ms: f64,
    pub peak_entries: usize,
    pub max_value_error: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ConditionalValue {
    pub bounds: Bounds,
    pub estimate: f64,
    pub absolute_error: f64,
}

pub struct EndgameSession {
    cfg: EndgameConfig,
    cache: BoundCache,
    attempted: HashMap<u128, usize>,
    started_ms: f64,
    budget_ms: f64,
    pub stats: EndgameStats,
}

impl EndgameSession {
    pub fn new(cfg: EndgameConfig, started_ms: f64, budget_ms: f64) -> Result<Self, String> {
        cfg.validate()?;
        if !started_ms.is_finite()
            || !budget_ms.is_finite()
            || !(0.0..=60_000.0).contains(&budget_ms)
        {
            return Err("endgame decision budget must be between 0 and 60000 ms".into());
        }
        Ok(Self {
            cfg,
            cache: BoundCache::default(),
            attempted: HashMap::new(),
            started_ms,
            budget_ms,
            stats: EndgameStats::default(),
        })
    }

    pub fn expired(&self, now_ms: f64) -> bool {
        !now_ms.is_finite()
            || now_ms < self.started_ms
            || now_ms - self.started_ms >= self.budget_ms
    }

    pub fn eligible(&self, battle: &Battle) -> bool {
        battle.turn > 0
            && battle.outcome().is_none()
            && battle.sides.iter().all(|side| {
                let living = side
                    .party
                    .iter()
                    .filter(|&&slot| {
                        let mon = &side.roster[slot as usize];
                        !mon.fainted && mon.hp > 0
                    })
                    .count();
                (1..=self.cfg.max_alive).contains(&living)
            })
    }

    fn bound_config(&self, work_budget: usize) -> BoundConfig {
        BoundConfig {
            work_budget,
            cell_cap: 0,
            node_budget: self.cfg.max_nodes,
            eps: self.cfg.leaf_width,
            dead_damage_quotient: false,
            certified_action_pruning: false,
            ..BoundConfig::default()
        }
    }

    pub fn retain_root(&mut self, dex: &Dex, root: &Battle, clock: &mut impl FnMut() -> f64) {
        let start = clock();
        let mut solver =
            BoundSolver::from_cache(dex, self.bound_config(0), std::mem::take(&mut self.cache));
        solver.retain_root(root);
        self.cache = solver.into_cache();
        self.stats.solver_ms += (clock() - start).max(0.0);
    }

    pub fn start_decision(&mut self, started_ms: f64, budget_ms: f64) -> Result<(), String> {
        if !started_ms.is_finite()
            || !budget_ms.is_finite()
            || !(0.0..=60_000.0).contains(&budget_ms)
        {
            return Err("endgame decision budget must be between 0 and 60000 ms".into());
        }
        self.started_ms = started_ms;
        self.budget_ms = budget_ms;
        self.stats = EndgameStats::default();
        self.attempted.clear();
        Ok(())
    }

    fn probe(
        &mut self,
        dex: &Dex,
        battle: &Battle,
        side: Option<usize>,
        clock: &mut impl FnMut() -> f64,
    ) -> Option<SolveReport> {
        if self.cfg.solver_fraction == 0.0 || !self.eligible(battle) {
            return None;
        }
        let start = clock();
        if self.expired(start) {
            return None;
        }
        let key = battle.state_key128();
        let spent = self.attempted.get(&key).copied().unwrap_or(0);
        let allowance = self
            .cfg
            .slice_runs
            .min(self.cfg.state_runs.saturating_sub(spent))
            .min(self.cfg.decision_runs.saturating_sub(self.stats.runs));
        let work = if self.stats.solver_ms >= self.budget_ms * self.cfg.solver_fraction {
            0
        } else {
            allowance
        };
        let mut solver = BoundSolver::from_cache(
            dex,
            self.bound_config(work),
            std::mem::take(&mut self.cache),
        );
        let cached = solver.peek(battle);
        let cache_hit = cached.is_some_and(|b| b.width() <= self.cfg.leaf_width);
        solver.cfg.node_budget = self.cfg.max_nodes.saturating_sub(solver.closed_count());
        let has_space = solver.node_count() < solver.cfg.node_budget;
        let report = if cache_hit || (work > 0 && has_space) {
            self.stats.calls += 1;
            self.stats.cache_hits += usize::from(cache_hit);
            Some(match side {
                Some(side) => solver.solve_strategy(battle, side, self.cfg.strategy_loss),
                None => solver.solve(battle, None),
            })
        } else {
            None
        };
        if let Some(report) = &report {
            self.stats.runs += report.runs;
            *self.attempted.entry(key).or_default() += report.runs;
            if report.bounds.width() <= self.cfg.leaf_width {
                self.stats.value_certificates += 1;
            } else {
                self.stats.unresolved += 1;
            }
            if side.is_some_and(|s| report.strategy(s, self.cfg.strategy_loss).is_some()) {
                self.stats.strategy_certificates += 1;
            }
            if report.stop == Stop::NodeBudget {
                self.stats.node_stops += 1;
            }
        }
        self.cache = solver.into_cache();
        self.stats.peak_entries = self.stats.peak_entries.max(self.cache.len());
        self.stats.solver_ms += (clock() - start).max(0.0);
        report
    }

    pub fn conditional_value(
        &mut self,
        dex: &Dex,
        battle: &Battle,
        clock: &mut impl FnMut() -> f64,
    ) -> Option<ConditionalValue> {
        let report = self.probe(dex, battle, None, clock)?;
        if report.bounds.width() > self.cfg.leaf_width {
            return None;
        }
        let value = ConditionalValue {
            bounds: report.bounds,
            estimate: report.bounds.mid(),
            absolute_error: report.bounds.width() * 0.5,
        };
        Some(value)
    }

    pub fn leaf(
        &mut self,
        dex: &Dex,
        cfg: &RmConfig,
        turn_cap: u16,
        sim: &mut Battle,
        rng: &mut SplitMix64,
        rollout: bool,
        clock: &mut impl FnMut() -> f64,
    ) -> f64 {
        let start = clock();
        for _ in 0..4 {
            if clock() - start >= 2.0 {
                break;
            }
            let before = self.stats.runs;
            if let Some(value) = self.conditional_value(dex, sim, clock) {
                self.stats.value_replacements += 1;
                self.stats.max_value_error = self.stats.max_value_error.max(value.absolute_error);
                return value.estimate;
            }
            if self.stats.runs == before {
                break;
            }
        }
        crate::smmcts::default_leaf(cfg, sim, dex, turn_cap, rng, rollout)
    }
}

pub struct EndgameSearch {
    search: SkuctSearch,
    root: Battle,
    side: usize,
    cfg: RmConfig,
    turn_cap: u16,
    pub session: EndgameSession,
    policy: Option<StrategyCertificate>,
    root_done: bool,
    committed: Option<SearchChoice>,
    choice_rng: SplitMix64,
}

impl EndgameSearch {
    pub fn new(
        battle: &Battle,
        dex: &Dex,
        side: usize,
        cfg: RmConfig,
        seed: u64,
        session: EndgameSession,
    ) -> Self {
        assert!(side < 2);
        Self {
            search: SkuctSearch::new(battle, dex, cfg.clone(), seed),
            root: battle.clone(),
            side,
            turn_cap: battle.turn.saturating_add(cfg.horizon),
            cfg,
            session,
            policy: None,
            root_done: false,
            committed: None,
            choice_rng: SplitMix64::new(seed ^ 0x454e_4447_414d_4531),
        }
    }

    pub fn step(&mut self, dex: &Dex, ticks: u32, clock: &mut impl FnMut() -> f64) -> u32 {
        for _ in 0..ticks {
            if self.committed.is_some() || self.policy.is_some() || self.session.expired(clock()) {
                break;
            }
            if !self.root_done
                && self.session.stats.solver_ms
                    < self.session.budget_ms * self.session.cfg.solver_fraction * 0.5
            {
                let report = self.session.probe(dex, &self.root, Some(self.side), clock);
                if let Some(report) = &report {
                    self.policy = report
                        .strategy(self.side, self.session.cfg.strategy_loss)
                        .cloned();
                }
                self.root_done = report
                    .as_ref()
                    .is_none_or(|r| r.runs == 0 || r.stop == Stop::NodeBudget);
                continue;
            }
            self.root_done = true;
            let (session, cfg, turn_cap) = (&mut self.session, &self.cfg, self.turn_cap);
            self.search
                .step_with_leaf(dex, 1, &mut |sim, rng, rollout| {
                    session.leaf(dex, cfg, turn_cap, sim, rng, rollout, clock)
                });
        }
        self.search.iterations()
    }

    pub fn ready(&self, now_ms: f64) -> bool {
        self.committed.is_some() || self.policy.is_some() || self.session.expired(now_ms)
    }

    pub fn policy(&self) -> Option<&StrategyCertificate> {
        self.policy.as_ref()
    }

    pub fn search(&self) -> &SkuctSearch {
        &self.search
    }

    pub fn best(&self) -> Option<SearchChoice> {
        if self.policy.is_some() {
            self.committed
        } else {
            self.committed.or_else(|| self.search.best(self.side))
        }
    }

    pub fn commit(&mut self) -> Option<SearchChoice> {
        if self.committed.is_some() {
            return self.committed;
        }
        let fallback = self.search.best(self.side);
        self.committed = match &self.policy {
            Some(policy) => {
                let pick = sample_policy(policy, &mut self.choice_rng);
                self.session.stats.root_overrides += 1;
                self.session.stats.root_changes += usize::from(pick != fallback);
                pick
            }
            None => fallback,
        };
        self.committed
    }
}

fn sample_policy(policy: &StrategyCertificate, rng: &mut SplitMix64) -> Option<SearchChoice> {
    let draw = rng.next_f64();
    let mut cumulative = 0.0;
    let mut last = None;
    for &(action, probability) in &policy.actions {
        if probability <= 0.0 {
            continue;
        }
        last = action;
        cumulative += probability;
        if draw < cumulative {
            return action;
        }
    }
    last
}

#[cfg(not(target_arch = "wasm32"))]
pub struct EndgameAgent {
    config: EndgameConfig,
    search_config: RmConfig,
    milliseconds: f64,
    pub stats: EndgameStats,
    rng: SplitMix64,
    session: Option<EndgameSession>,
    committed: Option<(u128, [u8; 16], usize, SearchChoice)>,
    policy: Vec<(SearchChoice, f64)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl EndgameAgent {
    pub fn new(
        config: EndgameConfig,
        search_config: RmConfig,
        milliseconds: f64,
        seed: u64,
    ) -> Result<Self, String> {
        EndgameSession::new(config.clone(), 0.0, milliseconds)?;
        Ok(Self {
            config,
            search_config,
            milliseconds,
            stats: EndgameStats::default(),
            rng: SplitMix64::new(seed),
            session: None,
            committed: None,
            policy: Vec::new(),
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl crate::agent::Agent for EndgameAgent {
    fn name(&self) -> String {
        format!(
            "endgame:{}ms:{}",
            self.milliseconds, self.config.solver_fraction
        )
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let key = battle.state_key128();
        if let Some((old, fingerprint, old_side, choice)) = self.committed {
            if old == key
                && fingerprint == dex.replay_fingerprint
                && old_side == side
                && choices.contains(&choice)
            {
                return choice;
            }
        }
        let start = std::time::Instant::now();
        let mut clock = || start.elapsed().as_secs_f64() * 1000.0;
        let mut session = self.session.take().unwrap_or_else(|| {
            EndgameSession::new(self.config.clone(), 0.0, self.milliseconds).unwrap()
        });
        session.start_decision(0.0, self.milliseconds).unwrap();
        session.retain_root(dex, battle, &mut clock);
        let mut search = EndgameSearch::new(
            battle,
            dex,
            side,
            self.search_config.clone(),
            self.rng.next(),
            session,
        );
        while !search.ready(clock()) {
            search.step(dex, 32, &mut clock);
        }
        let choice = search.commit().unwrap_or(choices[0]);
        assert!(choices.contains(&choice));
        self.policy = search
            .policy()
            .map(|p| {
                p.actions
                    .iter()
                    .filter_map(|&(action, mass)| action.map(|a| (a, mass)))
                    .collect()
            })
            .unwrap_or_else(|| vec![(choice, 1.0)]);
        self.stats = search.session.stats.clone();
        self.session = Some(search.session);
        self.committed = Some((key, dex.replay_fingerprint, side, choice));
        choice
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        self.choose(battle, dex, side, choices);
        choices
            .iter()
            .map(|action| {
                self.policy
                    .iter()
                    .find(|(a, _)| a == action)
                    .map_or(0.0, |(_, p)| *p)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_is_sampled_without_purification() {
        let policy = StrategyCertificate {
            actions: vec![
                (Some(SearchChoice::Switch(1)), 0.2),
                (Some(SearchChoice::Switch(2)), 0.8),
            ],
            guaranteed_value: 0.5,
            max_loss: 0.0,
        };
        let mut rng = SplitMix64::new(37);
        let first = (0..10_000)
            .filter(|_| sample_policy(&policy, &mut rng) == Some(SearchChoice::Switch(1)))
            .count();
        assert!((1900..2100).contains(&first), "{first}");
    }
}
