//! Experimental finite-horizon search for measuring damage-roll abstraction.
//!
//! This module is deliberately separate from `exact`/`bounds`: only
//! `DamageRollMode::Exact` has exact chance semantics, and every mode uses a
//! static evaluator at the horizon. Results are estimates, never certified
//! bounds. Keeping the decision recursion, leaf evaluator, and matrix solver
//! identical isolates the effect of quotienting damage rolls.

use std::collections::HashMap;

use nc2000_engine::battle::enumerate::enumerate_step_with_damage_mode;
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::DamageRollMode;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::eval::{eval01, EvalWeights};
use crate::exact::solve_matrix_full;
use crate::rng::SplitMix64;
use crate::smmcts::{RmAgent, RmConfig};

#[derive(Clone, Debug)]
pub struct DamageSearchConfig {
    /// Full turns beyond the root before static evaluation.
    pub horizon: u16,
    pub damage_mode: DamageRollMode,
    pub state_budget: usize,
    pub work_budget: usize,
    pub leaf_cap: usize,
    pub weights: EvalWeights,
}

impl Default for DamageSearchConfig {
    fn default() -> Self {
        DamageSearchConfig {
            horizon: 1,
            damage_mode: DamageRollMode::Threshold2,
            state_budget: 100_000,
            work_budget: 2_000_000,
            leaf_cap: 100_000,
            weights: EvalWeights::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DamageSearchStats {
    pub states: usize,
    /// Actual scripted engine executions. A step that reaches its cap is
    /// charged the full cap even though `enumerate_step` returns no leaves.
    pub chance_runs: usize,
    pub leaves: usize,
    pub matrix_cells: usize,
    pub max_matrix: [usize; 2],
    pub worst_gap: f64,
    /// Script-run observations, including repeated enumeration probes.
    pub exact_damage_draws: usize,
    pub abstract_damage_draws: usize,
    pub damage_classes: usize,
    pub drain_recoil_draws: usize,
    pub multihit_draws: usize,
    pub substitute_draws: usize,
    pub counter_bide_draws: usize,
    pub heal_draws: usize,
}

#[derive(Clone, Debug)]
pub struct DamageSearchReport {
    pub value: f64,
    pub gap: f64,
    pub row_policy: Vec<f64>,
    pub col_policy: Vec<f64>,
    /// Exact/abstract expected continuation values, row-major.
    pub matrix: Vec<f64>,
    pub dims: [usize; 2],
    pub stats: DamageSearchStats,
}

struct NodeSolution {
    value: f64,
    gap: f64,
    row_policy: Vec<f64>,
    col_policy: Vec<f64>,
    matrix: Vec<f64>,
    dims: [usize; 2],
}

pub struct DamageSearch<'d> {
    dex: &'d Dex,
    pub cfg: DamageSearchConfig,
    memo: HashMap<u128, f64>,
    t_max: u16,
    stats: DamageSearchStats,
}

impl<'d> DamageSearch<'d> {
    pub fn new(dex: &'d Dex, cfg: DamageSearchConfig) -> Self {
        DamageSearch {
            dex,
            cfg,
            memo: HashMap::new(),
            t_max: 0,
            stats: DamageSearchStats::default(),
        }
    }

    pub fn solve(&mut self, battle: &Battle) -> Option<DamageSearchReport> {
        self.memo.clear();
        self.stats = DamageSearchStats::default();
        self.t_max = battle.turn.saturating_add(self.cfg.horizon);
        let root = self.node(battle);
        self.stats.states = self.memo.len();
        let root = root?;
        Some(DamageSearchReport {
            value: root.value,
            gap: root.gap,
            row_policy: root.row_policy,
            col_policy: root.col_policy,
            matrix: root.matrix,
            dims: root.dims,
            stats: self.stats.clone(),
        })
    }

    pub fn stats(&self) -> &DamageSearchStats {
        &self.stats
    }

    fn terminal_value(&self, battle: &Battle) -> Option<f64> {
        battle.outcome().map(|outcome| match outcome {
            Outcome::P1Win => 1.0,
            Outcome::Tie => 0.5,
            Outcome::P2Win => 0.0,
        })
    }

    fn value(&mut self, battle: &Battle) -> Option<f64> {
        if let Some(value) = self.terminal_value(battle) {
            return Some(value);
        }
        if battle.turn > self.t_max {
            return Some(eval01(battle, self.dex, &self.cfg.weights));
        }
        let key = battle.state_key128();
        if let Some(&value) = self.memo.get(&key) {
            return Some(value);
        }
        if self.memo.len() >= self.cfg.state_budget {
            return None;
        }
        let solved = self.node(battle)?;
        self.memo.insert(key, solved.value);
        Some(solved.value)
    }

    fn node(&mut self, battle: &Battle) -> Option<NodeSolution> {
        if let Some(value) = self.terminal_value(battle) {
            return Some(NodeSolution {
                value,
                gap: 0.0,
                row_policy: Vec::new(),
                col_policy: Vec::new(),
                matrix: Vec::new(),
                dims: [0, 0],
            });
        }
        if battle.turn > self.t_max {
            let value = eval01(battle, self.dex, &self.cfg.weights);
            return Some(NodeSolution {
                value,
                gap: 0.0,
                row_policy: Vec::new(),
                col_policy: Vec::new(),
                matrix: Vec::new(),
                dims: [0, 0],
            });
        }

        let needs = battle.needs_choice();
        let mut probe = battle.clone();
        let actions = |probe: &mut Battle, side: usize, needed: bool| {
            if needed {
                probe
                    .legal_choices(self.dex, side)
                    .into_iter()
                    .map(Some)
                    .collect::<Vec<_>>()
            } else {
                vec![None]
            }
        };
        let row_actions = actions(&mut probe, 0, needs[0]);
        let col_actions = actions(&mut probe, 1, needs[1]);
        let (rows, cols) = (row_actions.len(), col_actions.len());
        if rows == 0 || cols == 0 {
            return None;
        }
        self.stats.matrix_cells += rows * cols;
        if rows * cols > self.stats.max_matrix[0] * self.stats.max_matrix[1] {
            self.stats.max_matrix = [rows, cols];
        }

        let mut matrix = vec![0.0; rows * cols];
        for (i, &row_action) in row_actions.iter().enumerate() {
            for (j, &col_action) in col_actions.iter().enumerate() {
                let remaining = self.cfg.work_budget.saturating_sub(self.stats.chance_runs);
                let cap = self.cfg.leaf_cap.min(remaining);
                if cap == 0 {
                    return None;
                }
                let Some(step) = enumerate_step_with_damage_mode(
                    self.dex,
                    battle,
                    [row_action, col_action],
                    cap,
                    self.cfg.damage_mode,
                ) else {
                    self.stats.chance_runs += cap;
                    return None;
                };
                self.stats.chance_runs += step.runs;
                self.stats.leaves += step.leaves.len();
                self.stats.exact_damage_draws += step.damage.exact_draws;
                self.stats.abstract_damage_draws += step.damage.abstract_draws;
                self.stats.damage_classes += step.damage.offered_classes;
                self.stats.drain_recoil_draws += step.damage.drain_recoil_draws;
                self.stats.multihit_draws += step.damage.multihit_draws;
                self.stats.substitute_draws += step.damage.substitute_draws;
                self.stats.counter_bide_draws += step.damage.counter_bide_draws;
                self.stats.heal_draws += step.damage.heal_draws;

                // Merge equal successor states before recursive evaluation.
                let mut successors: HashMap<u128, (f64, usize)> = HashMap::new();
                for (index, leaf) in step.leaves.iter().enumerate() {
                    successors
                        .entry(leaf.battle.state_key128())
                        .and_modify(|entry| entry.0 += leaf.prob)
                        .or_insert((leaf.prob, index));
                }
                let mut expected = 0.0;
                for &(probability, index) in successors.values() {
                    expected += probability * self.value(&step.leaves[index].battle)?;
                }
                matrix[i * cols + j] = expected;
            }
        }

        let (value, gap, row_policy, col_policy) = solve_game(&matrix, rows, cols);
        self.stats.worst_gap = self.stats.worst_gap.max(gap);
        Some(NodeSolution {
            value,
            gap,
            row_policy,
            col_policy,
            matrix,
            dims: [rows, cols],
        })
    }
}

fn solve_game(matrix: &[f64], rows: usize, cols: usize) -> (f64, f64, Vec<f64>, Vec<f64>) {
    if rows > 1 && cols > 1 {
        let solution = solve_matrix_full(matrix, rows, cols);
        return (solution.value, solution.gap, solution.x, solution.y);
    }
    if rows == 1 && cols == 1 {
        return (matrix[0], 0.0, vec![1.0], vec![1.0]);
    }
    if cols == 1 {
        let best = (0..rows)
            .max_by(|&a, &b| matrix[a].partial_cmp(&matrix[b]).unwrap())
            .unwrap();
        let mut row = vec![0.0; rows];
        row[best] = 1.0;
        return (matrix[best], 0.0, row, vec![1.0]);
    }
    let best = (0..cols)
        .min_by(|&a, &b| matrix[a].partial_cmp(&matrix[b]).unwrap())
        .unwrap();
    let mut col = vec![0.0; cols];
    col[best] = 1.0;
    (matrix[best], 0.0, vec![1.0], col)
}

/// Exploitability of `(row_policy, col_policy)` in a reference matrix.
/// Returns `(row regret, column regret, sum)` from the reference value.
pub fn policy_regret(
    reference_matrix: &[f64],
    dims: [usize; 2],
    reference_value: f64,
    row_policy: &[f64],
    col_policy: &[f64],
) -> (f64, f64, f64) {
    let [rows, cols] = dims;
    if row_policy.len() != rows || col_policy.len() != cols || rows == 0 || cols == 0 {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let row_guarantee = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| row_policy[i] * reference_matrix[i * cols + j])
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    let col_guarantee = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| col_policy[j] * reference_matrix[i * cols + j])
                .sum::<f64>()
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let row_regret = (reference_value - row_guarantee).max(0.0);
    let col_regret = (col_guarantee - reference_value).max(0.0);
    (row_regret, col_regret, row_regret + col_regret)
}

/// Duel-only wrapper: use the finite-horizon damage search in small
/// endgames and the normal SM-MCTS agent elsewhere. An incomplete search
/// also falls back, so fixed-work duels measure both speed and coverage.
#[derive(Clone, Debug)]
pub struct DamageSearchAgentConfig {
    pub search: DamageSearchConfig,
    pub alive_max: usize,
    pub hp_cap: u64,
    pub fallback: RmConfig,
}

impl Default for DamageSearchAgentConfig {
    fn default() -> Self {
        DamageSearchAgentConfig {
            search: DamageSearchConfig::default(),
            alive_max: 1,
            hp_cap: 600,
            fallback: RmConfig::default(),
        }
    }
}

pub struct DamageSearchAgent {
    cfg: DamageSearchAgentConfig,
    fallback: RmAgent,
    rng: SplitMix64,
}

impl DamageSearchAgent {
    pub fn new(cfg: DamageSearchAgentConfig, seed: u64) -> Self {
        DamageSearchAgent {
            fallback: RmAgent::new(cfg.fallback.clone(), seed ^ 0x85EB_CA77_C2B2_AE63),
            cfg,
            rng: SplitMix64::new(seed ^ 0x27D4_EB2F_1656_67C5),
        }
    }

    fn eligible(&self, battle: &Battle, choices: &[SearchChoice]) -> bool {
        if battle.turn == 0 || choices.len() < 2 {
            return false;
        }
        let alive = |side: usize| {
            battle.sides[side]
                .party
                .iter()
                .filter(|&&slot| {
                    let pokemon = &battle.sides[side].roster[slot as usize];
                    !pokemon.fainted && pokemon.hp > 0
                })
                .count()
        };
        let hp: u64 = battle
            .sides
            .iter()
            .flat_map(|side| {
                side.party
                    .iter()
                    .map(|&slot| side.roster[slot as usize].hp.max(0) as u64)
            })
            .sum();
        alive(0) <= self.cfg.alive_max && alive(1) <= self.cfg.alive_max && hp <= self.cfg.hp_cap
    }

    fn sample(&mut self, probabilities: &[f64]) -> usize {
        let target = self.rng.next_f64();
        let mut cumulative = 0.0;
        for (index, &probability) in probabilities.iter().enumerate() {
            cumulative += probability.max(0.0);
            if target < cumulative {
                return index;
            }
        }
        probabilities.len().saturating_sub(1)
    }
}

impl Agent for DamageSearchAgent {
    fn name(&self) -> String {
        format!(
            "damage:{:?}:h{}:w{}",
            self.cfg.search.damage_mode, self.cfg.search.horizon, self.cfg.search.work_budget
        )
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if self.eligible(battle, choices) {
            let mut search = DamageSearch::new(dex, self.cfg.search.clone());
            if let Some(report) = search.solve(battle) {
                let policy = if side == 0 {
                    &report.row_policy
                } else {
                    &report.col_policy
                };
                if policy.len() == choices.len() {
                    return choices[self.sample(policy)];
                }
            }
        }
        self.fallback.choose(battle, dex, side, choices)
    }
}
