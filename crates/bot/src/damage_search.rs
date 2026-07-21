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

const SUPPORT_EPSILON: f64 = 1e-9;

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

impl DamageSearchStats {
    fn absorb(&mut self, other: &DamageSearchStats) {
        self.states += other.states;
        self.chance_runs += other.chance_runs;
        self.leaves += other.leaves;
        self.matrix_cells += other.matrix_cells;
        if other.max_matrix[0] * other.max_matrix[1] > self.max_matrix[0] * self.max_matrix[1] {
            self.max_matrix = other.max_matrix;
        }
        self.worst_gap = self.worst_gap.max(other.worst_gap);
        self.exact_damage_draws += other.exact_damage_draws;
        self.abstract_damage_draws += other.abstract_damage_draws;
        self.damage_classes += other.damage_classes;
        self.drain_recoil_draws += other.drain_recoil_draws;
        self.multihit_draws += other.multihit_draws;
        self.substitute_draws += other.substitute_draws;
        self.counter_bide_draws += other.counter_bide_draws;
        self.heal_draws += other.heal_draws;
    }
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

#[derive(Clone, Debug)]
pub struct SupportRefineConfig {
    pub approximate: DamageSearchConfig,
    pub exact_work_budget: usize,
    /// Include approximate best-response actions this far from the current
    /// best response, in addition to actions with positive equilibrium mass.
    pub response_margin: f64,
}

#[derive(Clone, Debug)]
pub struct ProbeRefineConfig {
    pub approximate: DamageSearchConfig,
    pub probe_work_budget: usize,
    pub exact_work_budget: usize,
    pub response_margin: f64,
    /// Exact-refine a candidate cell only when its low/high representative
    /// continuation values disagree by more than this amount.
    pub cell_threshold: f64,
}

impl Default for ProbeRefineConfig {
    fn default() -> Self {
        let support = SupportRefineConfig::default();
        ProbeRefineConfig {
            probe_work_budget: support.approximate.work_budget,
            exact_work_budget: support.exact_work_budget,
            response_margin: support.response_margin,
            approximate: support.approximate,
            cell_threshold: 0.01,
        }
    }
}

impl Default for SupportRefineConfig {
    fn default() -> Self {
        let mut approximate = DamageSearchConfig::default();
        approximate.damage_mode = DamageRollMode::ThresholdLeanMinimal;
        SupportRefineConfig {
            exact_work_budget: approximate.work_budget,
            approximate,
            response_margin: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SupportRefineAttempt {
    pub report: Option<DamageSearchReport>,
    pub stats: DamageSearchStats,
    pub approximate_stats: DamageSearchStats,
    pub refine_stats: DamageSearchStats,
    pub refined_cells: usize,
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
        self.reset(battle);
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

    fn reset(&mut self, battle: &Battle) {
        self.memo.clear();
        self.stats = DamageSearchStats::default();
        self.t_max = battle.turn.saturating_add(self.cfg.horizon);
    }

    fn root_actions(
        &self,
        battle: &Battle,
    ) -> (Vec<Option<SearchChoice>>, Vec<Option<SearchChoice>>) {
        let needs = battle.needs_choice();
        let mut probe = battle.clone();
        let mut actions = |side: usize, needed: bool| {
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
        (actions(0, needs[0]), actions(1, needs[1]))
    }

    fn solve_root_cells(
        &mut self,
        battle: &Battle,
        cells: &[(usize, usize)],
    ) -> Option<Vec<(usize, usize, f64)>> {
        self.reset(battle);
        let (row_actions, col_actions) = self.root_actions(battle);
        self.stats.matrix_cells += cells.len();
        let mut values = Vec::with_capacity(cells.len());
        for &(i, j) in cells {
            let (Some(&row), Some(&col)) = (row_actions.get(i), col_actions.get(j)) else {
                self.stats.states = self.memo.len();
                return None;
            };
            let Some(value) = self.cell_value(battle, row, col) else {
                self.stats.states = self.memo.len();
                return None;
            };
            values.push((i, j, value));
        }
        self.stats.states = self.memo.len();
        Some(values)
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

    fn cell_value(
        &mut self,
        battle: &Battle,
        row_action: Option<SearchChoice>,
        col_action: Option<SearchChoice>,
    ) -> Option<f64> {
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
        Some(expected)
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
                matrix[i * cols + j] = self.cell_value(battle, row_action, col_action)?;
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

fn support_refine_candidates(
    report: &DamageSearchReport,
    response_margin: f64,
) -> Vec<(usize, usize)> {
    let [rows, cols] = report.dims;
    if rows == 0 || cols == 0 {
        return Vec::new();
    }

    let row_payoffs: Vec<f64> = (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| report.matrix[i * cols + j] * report.col_policy[j])
                .sum()
        })
        .collect();
    let best_row = row_payoffs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let selected_rows: Vec<usize> = (0..rows)
        .filter(|&i| {
            report.row_policy[i] > SUPPORT_EPSILON
                || best_row - row_payoffs[i] <= response_margin + SUPPORT_EPSILON
        })
        .collect();

    let col_payoffs: Vec<f64> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|i| report.row_policy[i] * report.matrix[i * cols + j])
                .sum()
        })
        .collect();
    let best_col = col_payoffs.iter().copied().fold(f64::INFINITY, f64::min);
    let selected_cols: Vec<usize> = (0..cols)
        .filter(|&j| {
            report.col_policy[j] > SUPPORT_EPSILON
                || col_payoffs[j] - best_col <= response_margin + SUPPORT_EPSILON
        })
        .collect();

    let mut cells = Vec::with_capacity(selected_rows.len() * selected_cols.len());
    for &i in &selected_rows {
        for &j in &selected_cols {
            cells.push((i, j));
        }
    }
    cells
}

fn probe_refine_cells(
    low_values: &[(usize, usize, f64)],
    high_values: &[(usize, usize, f64)],
    threshold: f64,
) -> Vec<(usize, usize)> {
    low_values
        .iter()
        .zip(high_values)
        .filter_map(|(&(li, lj, low), &(hi, hj, high))| {
            debug_assert_eq!((li, lj), (hi, hj));
            ((low - high).abs() > threshold.max(0.0) + SUPPORT_EPSILON).then_some((li, lj))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(matrix: Vec<f64>) -> DamageSearchReport {
        DamageSearchReport {
            value: 0.5,
            gap: 0.0,
            row_policy: vec![0.5, 0.5, 0.0],
            col_policy: vec![1.0, 0.0, 0.0],
            matrix,
            dims: [3, 3],
            stats: DamageSearchStats::default(),
        }
    }

    #[test]
    fn support_refine_includes_support_and_pure_best_responses() {
        let report = report(vec![0.5, 0.5, 0.7, 0.5, 0.5, 0.7, 0.4, 0.45, 0.6]);
        assert_eq!(
            support_refine_candidates(&report, 0.0),
            vec![(0, 0), (0, 1), (1, 0), (1, 1)]
        );
    }

    #[test]
    fn support_refine_margin_adds_near_best_responses() {
        let report = report(vec![0.5, 0.5, 0.7, 0.5, 0.5, 0.7, 0.4, 0.45, 0.6]);
        assert_eq!(
            support_refine_candidates(&report, 0.11),
            vec![(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        );
    }

    #[test]
    fn probe_refine_selects_only_cells_above_threshold() {
        let low = vec![(0, 0, 0.2), (0, 1, 0.4), (1, 0, 0.6)];
        let high = vec![(0, 0, 0.205), (0, 1, 0.42), (1, 0, 0.59)];
        assert_eq!(probe_refine_cells(&low, &high, 0.01), vec![(0, 1)]);
    }
}

pub fn solve_support_refined(
    dex: &Dex,
    battle: &Battle,
    cfg: &SupportRefineConfig,
) -> SupportRefineAttempt {
    let mut approximate = DamageSearch::new(dex, cfg.approximate.clone());
    let Some(approximate_report) = approximate.solve(battle) else {
        let stats = approximate.stats().clone();
        return SupportRefineAttempt {
            report: None,
            stats: stats.clone(),
            approximate_stats: stats,
            refine_stats: DamageSearchStats::default(),
            refined_cells: 0,
        };
    };
    let approximate_stats = approximate_report.stats.clone();
    let cells = support_refine_candidates(&approximate_report, cfg.response_margin.max(0.0));
    if cells.is_empty() {
        return SupportRefineAttempt {
            report: Some(approximate_report),
            stats: approximate_stats.clone(),
            approximate_stats,
            refine_stats: DamageSearchStats::default(),
            refined_cells: 0,
        };
    }

    let mut exact_cfg = cfg.approximate.clone();
    exact_cfg.damage_mode = DamageRollMode::Exact;
    exact_cfg.work_budget = cfg.exact_work_budget;
    let mut exact = DamageSearch::new(dex, exact_cfg);
    let exact_values = exact.solve_root_cells(battle, &cells);
    let refine_stats = exact.stats().clone();
    let mut stats = approximate_stats.clone();
    stats.absorb(&refine_stats);
    let Some(exact_values) = exact_values else {
        return SupportRefineAttempt {
            report: None,
            stats,
            approximate_stats,
            refine_stats,
            refined_cells: cells.len(),
        };
    };

    let [rows, cols] = approximate_report.dims;
    let mut matrix = approximate_report.matrix;
    for (i, j, value) in exact_values {
        matrix[i * cols + j] = value;
    }
    let (value, gap, row_policy, col_policy) = solve_game(&matrix, rows, cols);
    stats.worst_gap = stats.worst_gap.max(gap);
    let report = DamageSearchReport {
        value,
        gap,
        row_policy,
        col_policy,
        matrix,
        dims: [rows, cols],
        stats: stats.clone(),
    };
    SupportRefineAttempt {
        report: Some(report),
        stats,
        approximate_stats,
        refine_stats,
        refined_cells: cells.len(),
    }
}

/// Cheaply screen equilibrium-relevant cells with the same damage buckets
/// using their low/high representatives. Only strategically unstable cells
/// pay for exact damage enumeration.
pub fn solve_probe_refined(
    dex: &Dex,
    battle: &Battle,
    cfg: &ProbeRefineConfig,
) -> SupportRefineAttempt {
    let mut approximate = DamageSearch::new(dex, cfg.approximate.clone());
    let Some(approximate_report) = approximate.solve(battle) else {
        let stats = approximate.stats().clone();
        return SupportRefineAttempt {
            report: None,
            stats: stats.clone(),
            approximate_stats: stats,
            refine_stats: DamageSearchStats::default(),
            refined_cells: 0,
        };
    };
    let approximate_stats = approximate_report.stats.clone();
    let candidates = support_refine_candidates(&approximate_report, cfg.response_margin.max(0.0));
    if candidates.is_empty() {
        return SupportRefineAttempt {
            report: Some(approximate_report),
            stats: approximate_stats.clone(),
            approximate_stats,
            refine_stats: DamageSearchStats::default(),
            refined_cells: 0,
        };
    }

    let mut probe_cfg = cfg.approximate.clone();
    probe_cfg.damage_mode = DamageRollMode::ThresholdLeanMinimalLow;
    probe_cfg.work_budget = cfg.probe_work_budget;
    let mut low = DamageSearch::new(dex, probe_cfg.clone());
    let low_values = low.solve_root_cells(battle, &candidates);
    let low_stats = low.stats().clone();

    probe_cfg.damage_mode = DamageRollMode::ThresholdLeanMinimalHigh;
    let mut high = DamageSearch::new(dex, probe_cfg);
    let high_values = high.solve_root_cells(battle, &candidates);
    let high_stats = high.stats().clone();

    let mut stats = approximate_stats.clone();
    stats.absorb(&low_stats);
    stats.absorb(&high_stats);
    let (Some(low_values), Some(high_values)) = (low_values, high_values) else {
        return SupportRefineAttempt {
            report: None,
            stats,
            approximate_stats,
            refine_stats: DamageSearchStats::default(),
            refined_cells: 0,
        };
    };
    let cells = probe_refine_cells(&low_values, &high_values, cfg.cell_threshold);
    if cells.is_empty() {
        let mut report = approximate_report;
        report.stats = stats.clone();
        return SupportRefineAttempt {
            report: Some(report),
            stats,
            approximate_stats,
            refine_stats: DamageSearchStats::default(),
            refined_cells: 0,
        };
    }

    let mut exact_cfg = cfg.approximate.clone();
    exact_cfg.damage_mode = DamageRollMode::Exact;
    exact_cfg.work_budget = cfg.exact_work_budget;
    let mut exact = DamageSearch::new(dex, exact_cfg);
    let exact_values = exact.solve_root_cells(battle, &cells);
    let refine_stats = exact.stats().clone();
    stats.absorb(&refine_stats);
    let Some(exact_values) = exact_values else {
        return SupportRefineAttempt {
            report: None,
            stats,
            approximate_stats,
            refine_stats,
            refined_cells: cells.len(),
        };
    };

    let [rows, cols] = approximate_report.dims;
    let mut matrix = approximate_report.matrix;
    for (i, j, value) in exact_values {
        matrix[i * cols + j] = value;
    }
    let (value, gap, row_policy, col_policy) = solve_game(&matrix, rows, cols);
    stats.worst_gap = stats.worst_gap.max(gap);
    let report = DamageSearchReport {
        value,
        gap,
        row_policy,
        col_policy,
        matrix,
        dims: [rows, cols],
        stats: stats.clone(),
    };
    SupportRefineAttempt {
        report: Some(report),
        stats,
        approximate_stats,
        refine_stats,
        refined_cells: cells.len(),
    }
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

fn endgame_eligible(
    battle: &Battle,
    choices: &[SearchChoice],
    alive_max: usize,
    hp_cap: u64,
) -> bool {
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
    alive(0) <= alive_max && alive(1) <= alive_max && hp <= hp_cap
}

fn sample_policy(rng: &mut SplitMix64, probabilities: &[f64]) -> usize {
    let target = rng.next_f64();
    let mut cumulative = 0.0;
    for (index, &probability) in probabilities.iter().enumerate() {
        cumulative += probability.max(0.0);
        if target < cumulative {
            return index;
        }
    }
    probabilities.len().saturating_sub(1)
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
        if endgame_eligible(battle, choices, self.cfg.alive_max, self.cfg.hp_cap) {
            let mut search = DamageSearch::new(dex, self.cfg.search.clone());
            if let Some(report) = search.solve(battle) {
                let policy = if side == 0 {
                    &report.row_policy
                } else {
                    &report.col_policy
                };
                if policy.len() == choices.len() {
                    return choices[sample_policy(&mut self.rng, policy)];
                }
            }
        }
        self.fallback.choose(battle, dex, side, choices)
    }
}

/// Duel-only wrapper for the low/high-screened support refinement. Kept
/// separate from `DamageSearchAgent` so the production-facing direct search
/// path remains unchanged while the experiment is gated.
#[derive(Clone, Debug)]
pub struct ProbeRefineAgentConfig {
    pub refine: ProbeRefineConfig,
    pub alive_max: usize,
    pub hp_cap: u64,
    pub fallback: RmConfig,
}

impl Default for ProbeRefineAgentConfig {
    fn default() -> Self {
        ProbeRefineAgentConfig {
            refine: ProbeRefineConfig::default(),
            alive_max: 1,
            hp_cap: 600,
            fallback: RmConfig::default(),
        }
    }
}

pub struct ProbeRefineAgent {
    cfg: ProbeRefineAgentConfig,
    fallback: RmAgent,
    rng: SplitMix64,
}

impl ProbeRefineAgent {
    pub fn new(cfg: ProbeRefineAgentConfig, seed: u64) -> Self {
        ProbeRefineAgent {
            fallback: RmAgent::new(cfg.fallback.clone(), seed ^ 0x85EB_CA77_C2B2_AE63),
            cfg,
            rng: SplitMix64::new(seed ^ 0x27D4_EB2F_1656_67C5),
        }
    }
}

impl Agent for ProbeRefineAgent {
    fn name(&self) -> String {
        format!(
            "damage:probe-refine:h{}:w{}:p{}:e{}:t{}",
            self.cfg.refine.approximate.horizon,
            self.cfg.refine.approximate.work_budget,
            self.cfg.refine.probe_work_budget,
            self.cfg.refine.exact_work_budget,
            self.cfg.refine.cell_threshold,
        )
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if endgame_eligible(battle, choices, self.cfg.alive_max, self.cfg.hp_cap) {
            let attempt = solve_probe_refined(dex, battle, &self.cfg.refine);
            if let Some(report) = attempt.report {
                let policy = if side == 0 {
                    &report.row_policy
                } else {
                    &report.col_policy
                };
                if policy.len() == choices.len() {
                    return choices[sample_policy(&mut self.rng, policy)];
                }
            }
        }
        self.fallback.choose(battle, dex, side, choices)
    }
}
