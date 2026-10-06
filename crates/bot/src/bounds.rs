//! M17e e-5 (Phase B of ENDGAME-SOLVER-ALGORITHM.md) — lazy certified
//! bound solver.
//!
//! A persistent search graph over decision states: every node carries a
//! sound value interval (side-0 win probability), unexpanded chance mass
//! counts as [0, 1], and one trial = walk root→frontier by LP-support ×
//! uncertainty, expand one pending chance script, back
//! bounds up the visited path (matrix-game values are monotone in
//! payoffs, so lo/hi matrices bracket soundly — same principle as
//! `exact`, applied incrementally). Work limits stop expansion, never
//! discard it: partial nodes persist across calls and corpus positions,
//! and bounds only ever tighten.
//!
//! Threshold certificate mode: pass `tau = (eval−m, eval+m)` and the
//! solve stops at the first PROOF — `lo > tau_hi` (proven underestimate),
//! `hi < tau_lo` (proven overestimate), or containment — the
//! chance/simultaneous analogue of a null-window search. Default for
//! corpus violation mining; width mode remains for numeric anchors.

mod correlated;
pub use correlated::{AffineCell, CorrelatedLoss, RootCorrelation};

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use nc2000_engine::battle::enumerate::{enumerate_step, enumerate_step_partial, run_scripted};
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::exact::solve_matrix_full;
use crate::stall::{
    classify_one_sided_heal, classify_two_sided_heal, MonotoneRank, OneSidedHeal, ResourceRank,
    TwoSidedHeal,
};

pub fn has_live_boom_move(dex: &Dex, battle: &Battle, side: usize) -> bool {
    battle.sides[side].party.iter().any(|&slot| {
        let pokemon=&battle.sides[side].roster[slot as usize];
        pokemon.hp>0 && !pokemon.fainted && pokemon.move_slots.iter()
            .any(|slot|matches!(dex.moves.key(slot.id),"selfdestruct"|"explosion"))
    })
}

#[derive(Clone, Debug)]
pub struct BoundConfig {
    pub time_budget: Option<Duration>,
    /// Engine executions per `solve` call.
    pub work_budget: usize,
    /// On a cell's first visit, try the EAGER enumerator (which keeps the
    /// damage-roll range merge) up to this many runs; only fans that
    /// overflow it fall back to lazy pending expansion. Measured: without
    /// this, lazy expansion re-pays merged overkill fans one run per roll
    /// class (b60: 411k runs vs the eager solver's 11.7k).
    pub cell_cap: usize,
    pub retain_eager_prefix: bool,
    pub defer_eager_frontier: bool,
    pub profile_pending_selection: bool,
    /// Max live nodes (memory guard; Battles are dropped from resolved
    /// nodes but frontier nodes hold one each).
    pub node_budget: usize,
    /// Width goal: stop when the root interval is at most this wide.
    pub eps: f64,
    /// Defensive per-trial depth cap.
    pub trial_depth: usize,
    /// Descend into a resolved child only when its mass×width term
    /// exceeds BOTH the cell's remaining pending mass and this floor —
    /// expand-shallow-first, deepen where the shallow layer is resolved.
    /// (First measurement: representative leaves carry most of a step's
    /// mass, so a naive mass×width rule needle-dives hundreds of plies
    /// with zero information gained.)
    pub descend_floor: f64,
    /// Merge states that differ only in damage-history fields when no move
    /// in either roster can observe them. The key domain is tagged, so a
    /// solver reused across safe and unsafe corpus roots cannot cross-merge.
    pub dead_damage_quotient: bool,
    pub dead_quick_claw_quotient: bool,
    pub hp_transition_capacity: usize,
    pub hp_transition_eager: bool,
    /// Fold terminal successor mass directly into its parent cell instead
    /// of retaining one graph node per terminal fingerprint.
    pub fold_terminal_nodes: bool,
    /// Periodically fold exact non-root nodes into a compact closed memo.
    /// This is graph-local and does not depend on stall classification.
    pub fold_closed_nodes: bool,
    /// After the resource closure threshold, prefer lower proven resource
    /// generations in a classified one-sided-heal subgame. Every generated
    /// edge is checked; the first violation disables only this scheduler.
    pub monotone_stall_scheduling: bool,
    /// Extend resource-ordered scheduling to certified two-sided-heal
    /// last-mon endings. HP is deliberately absent from this proof rank;
    /// total PP and distance to the turn cap make the exact graph a DAG.
    pub two_sided_resource_scheduling: bool,
    /// Permanently remove a pure action only after cell intervals prove it
    /// pointwise dominated by another live action. The reduced matrix has
    /// the same true value; unresolved actions are never assumed irrelevant.
    pub certified_action_pruning: bool,
    /// In addition to current LP-support cells, prioritize optimistic pure
    /// best responses crossed with the opponent's support. This changes only
    /// frontier order; certification still uses every unpruned action.
    pub support_br_scheduling: bool,
    pub reuse_unchanged_backups: bool,
    pub aggregate_lazy_children: bool,
    pub chance_fair_period: usize,
    pub chance_status_groups: bool,
    pub chance_material_order: bool,
    pub material_descent_period: usize,
    pub resource_closure_nodes: Option<usize>,
    pub resource_closure_period: usize,
    pub directed_policy_refinement: bool,
    pub target_aware_refinement: bool,
    pub strategy_refinement: bool,
}

impl Default for BoundConfig {
    fn default() -> Self {
        BoundConfig {
            time_budget: None,
            work_budget: 1_000_000,
            node_budget: 120_000,
            cell_cap: 4096,
            retain_eager_prefix: false,
            defer_eager_frontier: false,
            profile_pending_selection: false,
            eps: 0.02,
            trial_depth: 24,
            descend_floor: 0.1,
            dead_damage_quotient: true,
            dead_quick_claw_quotient: false,
            hp_transition_capacity: 0,
            hp_transition_eager: false,
            fold_terminal_nodes: true,
            fold_closed_nodes: true,
            monotone_stall_scheduling: true,
            two_sided_resource_scheduling: true,
            certified_action_pruning: true,
            support_br_scheduling: true,
            reuse_unchanged_backups: true,
            aggregate_lazy_children: false,
            chance_fair_period: 0,
            chance_status_groups: false,
            chance_material_order: false,
            material_descent_period: 0,
            resource_closure_nodes: None,
            resource_closure_period: 1,
            directed_policy_refinement: false,
            target_aware_refinement: false,
            strategy_refinement: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
}

#[derive(Clone, Debug)]
pub struct RootPayoffs {
    pub actions: [Vec<Option<SearchChoice>>; 2],
    pub cells: Vec<Bounds>,
    pub value: Bounds,
}

impl RootPayoffs {
    pub fn policy_value(&self, side: usize, policy: &[f64]) -> Bounds {
        assert!(side < 2 && policy.len() == self.actions[side].len());
        let policy = normalized_policy(policy).expect("nonnegative nonempty policy");
        let mut value = Bounds { lo: 1.0, hi: 1.0 };
        for reply in 0..self.actions[1 - side].len() {
            let mut bounds = Bounds { lo: 0.0, hi: 0.0 };
            for (action, &mass) in policy.iter().enumerate() {
                let cell = self.cell(side, action, reply);
                bounds.lo += mass * cell.lo;
                bounds.hi += mass * cell.hi;
            }
            value.lo = value.lo.min(bounds.lo);
            value.hi = value.hi.min(bounds.hi);
        }
        let pad = 1e-12 * (self.cells.len() + 1) as f64;
        Bounds { lo: (value.lo - pad).max(0.0), hi: (value.hi + pad).min(1.0) }
    }

    pub fn policy_loss(&self, side: usize, policy: &[f64]) -> Bounds {
        let guarantee = self.policy_value(side, policy);
        if self.actions[side].len() == 1 {
            return Bounds { lo: 0.0, hi: 0.0 };
        }
        let value = if side == 0 { self.value } else {
            Bounds { lo: 1.0 - self.value.hi, hi: 1.0 - self.value.lo }
        };
        Bounds {
            lo: (value.lo - guarantee.hi).max(0.0),
            hi: (value.hi - guarantee.lo).max(0.0),
        }
    }

    fn cell(&self, side: usize, action: usize, reply: usize) -> Bounds {
        let cols = self.actions[1].len();
        if side == 0 { self.cells[action * cols + reply] } else {
            let cell = self.cells[reply * cols + action];
            Bounds { lo: 1.0 - cell.hi, hi: 1.0 - cell.lo }
        }
    }

    fn refinement_cell(&self, side: usize, policy: &[f64], tolerance: Option<f64>) -> (Option<(usize, f64)>, usize) {
        let guarantee = self.policy_value(side, policy);
        let target = tolerance.map(|tolerance| {
            let ceiling = if side == 0 { self.value.hi } else { 1.0 - self.value.lo };
            ceiling - tolerance + 1e-12 * (self.cells.len() + 1) as f64 + 1e-10
        });
        let mut best = None;
        let mut sufficient = 0;
        for reply in 0..self.actions[1 - side].len() {
            let lower: f64 = policy.iter().enumerate()
                .map(|(action, &mass)| mass * self.cell(side, action, reply).lo).sum();
            if lower > guarantee.hi + 1e-10 { continue; }
            if target.is_some_and(|target| lower >= target) {
                sufficient += 1;
                continue;
            }
            for (action, &mass) in policy.iter().enumerate() {
                let score = mass * self.cell(side, action, reply).width();
                let index = if side == 0 { action * self.actions[1].len() + reply }
                    else { reply * self.actions[1].len() + action };
                if score > 1e-12 && best.is_none_or(|(_, old)| score > old) {
                    best = Some((index, score));
                }
            }
        }
        (best, sufficient)
    }
}

impl Bounds {
    pub fn mid(&self) -> f64 {
        (self.lo + self.hi) * 0.5
    }
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// Root width ≤ eps.
    WidthMet,
    StrategyMet,
    /// `lo > tau_hi`: the true value is proven above the threshold band.
    ProvenAbove,
    /// `hi < tau_lo`: the true value is proven below the threshold band.
    ProvenBelow,
    /// Root interval fell entirely inside the threshold band.
    Contained,
    WorkExhausted,
    TimeBudget,
    NodeBudget,
}

/// Payoffs are from this side's perspective with optimal continuation.
/// Guarantees apply only to the concrete full-information state.
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyCertificate {
    pub actions: Vec<(Option<SearchChoice>, f64)>,
    pub guaranteed_value: f64,
    pub max_loss: f64,
}

impl StrategyCertificate {
    pub fn meets(&self, tolerance: f64) -> bool {
        tolerance.is_finite() && tolerance >= 0.0 && self.max_loss <= tolerance
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolutionStatus {
    ClosedValue,
    GuaranteedStrategies,
    Unresolved,
}

#[derive(Clone, Debug)]
pub struct SolveReport {
    pub bounds: Bounds,
    pub stop: Stop,
    /// Engine runs consumed by THIS call.
    pub runs: usize,
    pub strategies: Option<[StrategyCertificate; 2]>,
}

impl SolveReport {
    pub fn status(&self, tolerance: f64) -> SolutionStatus {
        if self.bounds.width() <= 1e-12 {
            SolutionStatus::ClosedValue
        } else if self.strategies.as_ref().is_some_and(|policies| {
            policies.iter().all(|policy| policy.meets(tolerance))
        }) {
            SolutionStatus::GuaranteedStrategies
        } else {
            SolutionStatus::Unresolved
        }
    }

    pub fn strategy(&self, side: usize, tolerance: f64) -> Option<&StrategyCertificate> {
        self.strategies.as_ref()?.get(side).filter(|policy| policy.meets(tolerance))
    }
}

#[derive(Clone, Debug, Default)]
pub struct BoundStats {
    pub certificate_hits: usize,
    pub sufficient_reply_skips: usize,
    pub target_filtered_trials: usize,
    pub lower_directed_trials: usize,
    pub upper_directed_trials: usize,
    pub resource_closure_visits: usize,
    pub first_resource_closure_nodes: Option<usize>,
    pub runs: usize,
    pub hp_transition_eager_hits: usize,
    pub hp_transition_eager_eligible: usize,
    pub hp_transition_hits: usize,
    pub hp_transition_eligible: usize,
    pub hp_transition_rejected: usize,
    pub hp_transition_peak: usize,
    pub eager_runs: usize,
    pub eager_failures: usize,
    pub retained_eager_runs: usize,
    pub retained_eager_leaves: usize,
    pub deferred_eager_leaves: usize,
    pub pending_selection_ns: u128,
    pub pending_selection_calls: usize,
    pub pending_scan_items: usize,
    pub reused_nodes: usize,
    pub reused_backups: usize,
    pub expansions: usize,
    pub trials: usize,
    pub lp_solves: usize,
    pub worst_gap: f64,
    pub peak_nodes: usize,
    pub closed_folds: usize,
    pub monotone_roots: usize,
    pub monotone_invalidations: usize,
    pub two_sided_roots: usize,
    pub two_sided_invalidations: usize,
    pub one_sided_handoffs: usize,
    pub min_healing_pp: Option<i64>,
    pub min_resource_pp: Option<i64>,
    pub dominated_rows: usize,
    pub dominated_cols: usize,
    pub dominance_checks: usize,
    pub avoided_cells: usize,
    pub br_cell_picks: usize,
    pub lower_br_picks: usize,
    pub upper_br_picks: usize,
    pub legacy_support_picks: usize,
    pub fair_cell_picks: usize,
    pub trial_depth_counts: Vec<usize>,
    pub depth_limit_trials: usize,
    pub material_descent_bypasses: usize,
    pub strategy_refinement_trials: usize,
    pub strategy_ceiling_trials: usize,
    pub fixed_waking_rest_nodes: usize,
    pub lazy_edge_merges: usize,
}

#[derive(Debug)]
pub struct ChanceDiagnostics {
    pub visits: usize,
    pub runs: usize,
    pub pending_mass: f64,
    pub pending_scripts: usize,
    pub resolved_edges: usize,
    pub unique_children: usize,
    pub child_uncertainty: f64,
    pub largest_edge_uncertainty: f64,
    pub largest_child_uncertainty: f64,
    pub expanded_children: usize,
    pub max_child_runs: usize,
    pub total_child_runs: usize,
    pub largest_children: Vec<ChanceChildDiagnostics>,
}

#[derive(Debug)]
pub struct ChanceChildDiagnostics {
    pub key: String,
    pub mass: f64,
    pub bounds: Bounds,
    pub uncertainty: f64,
    pub runs: usize,
    pub visits: usize,
    pub resource_rank: Option<ResourceRank>,
    pub battle: Option<Battle>,
    pub actions: [Vec<Option<SearchChoice>>; 2],
    pub cells: Vec<Bounds>,
}

#[derive(Debug, Default)]
pub struct GraphDiagnostics {
    pub unexpanded_nodes: usize,
    pub initialized_nodes: usize,
    pub stored_battles: usize,
    pub resolved_edges: usize,
    pub unique_child_edges: usize,
    pub pending_scripts: usize,
    pub pending_script_entries: usize,
}

struct Pending {
    script: Vec<usize>,
    mass: f64,
}

/// The tag is part of identity: a raw exact fingerprint and a semantic
/// fingerprint are never compared even if their 128-bit payloads coincide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum NodeKey {
    Exact(u128),
    NoDamageBookkeeping(u128),
    NoQuickClawRoll(u128),
    NoDamageOrQuickClawRoll(u128),
}

struct Cell {
    visits: usize,
    runs: usize,
    fair_cursor: usize,
    resolved: Vec<(f64, NodeKey)>,
    /// Exact contribution from terminal/closed successors already folded
    /// out of the live graph.
    fixed_lo: f64,
    fixed_hi: f64,
    pending: Vec<Pending>,
    pending_mass: f64,
    lo: f64,
    hi: f64,
    /// One-shot eager enumeration already attempted for this cell.
    tried_eager: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundDirection {
    Lower,
    Upper,
}

struct StageWitness {
    xlo: Vec<f64>,
    ylo: Vec<f64>,
    xhi: Vec<f64>,
    yhi: Vec<f64>,
    gap_lo: f64,
    gap_hi: f64,
}

struct Node {
    chance_class: u32,
    remaining_hp: [i32; 2],
    /// Expansion source; dropped once the node is resolved or fully
    /// expanded (bounds then update purely from children).
    battle: Option<Battle>,
    acts0: Vec<Option<SearchChoice>>,
    acts1: Vec<Option<SearchChoice>>,
    cells: Vec<Cell>,
    lo: f64,
    hi: f64,
    /// LP supports cached by the last backup (lo-matrix x/y ‖ hi-matrix
    /// x/y) — the cell selector steers by these instead of re-solving.
    strat: Option<StageWitness>,
    /// Pure actions removed by a permanent interval-dominance certificate.
    active_rows: Vec<bool>,
    active_cols: Vec<bool>,
    select_count: usize,
    fair_cursor: usize,
    stall_rank: Option<(OneSidedHeal, MonotoneRank)>,
    resource_rank: Option<(TwoSidedHeal, ResourceRank, bool)>,
}

pub struct BoundSolver<'d> {
    dex: &'d Dex,
    pub cfg: BoundConfig,
    nodes: HashMap<NodeKey, Node>,
    closed: HashMap<NodeKey, Bounds>,
    closed_strategies: HashMap<NodeKey, [StrategyCertificate; 2]>,
    roots: HashSet<NodeKey>,
    active_stall: Option<OneSidedHeal>,
    active_two_sided: Option<TwoSidedHeal>,
    fixed_policy: Option<(usize, Vec<SearchChoice>)>,
    fixed_first_action: Option<(NodeKey, SearchChoice)>,
    fixed_sleep_talk: bool,
    fixed_frozen_switch: bool,
    fixed_rest_percent: Option<u8>,
    fixed_recover_percent: Option<u8>,
    fixed_recover_last_foes: Option<Vec<nc2000_engine::dex::SpeciesId>>,
    fixed_rest_after_boom: bool,
    fixed_rest_on_wake: bool,
    value_certificates: HashMap<NodeKey, Bounds>,
    hp_transitions: crate::hp_transition::HpTransitionCache,
    pub stats: BoundStats,
}

pub struct FixedPolicySolver<'d> {
    solver: BoundSolver<'d>,
    side: usize,
}

pub struct ValueCertificates {
    dex_fingerprint: [u8; 16],
    entries: HashMap<NodeKey, Bounds>,
}

impl ValueCertificates {
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

pub struct PolicyBoundReport {
    /// Fixed side's score against an optimal full-information opponent.
    pub guaranteed_score: Bounds,
    /// Side-zero value when both players may optimize.
    pub unrestricted_game_value: Bounds,
    pub stop: Stop,
    pub runs: usize,
}

impl<'d> FixedPolicySolver<'d> {
    /// At every decision, use the first legal priority action, falling back
    /// to the engine's first legal choice. Only the opponent may optimize.
    pub fn new(dex: &'d Dex, cfg: BoundConfig, side: usize, priority: Vec<SearchChoice>) -> Self {
        assert!(side < 2);
        let mut solver = BoundSolver::new(dex, cfg);
        solver.fixed_policy = Some((side, priority));
        Self { solver, side }
    }

    pub fn new_with_first_action(dex: &'d Dex, cfg: BoundConfig, side: usize,
        priority: Vec<SearchChoice>, root: &Battle, action: SearchChoice) -> Self {
        let mut probe = root.clone();
        assert!(side < 2 && probe.legal_choices(dex, side).contains(&action));
        let mut fixed = Self::new(dex, cfg, side, priority);
        fixed.solver.fixed_first_action = Some((fixed.solver.node_key(root), action));
        fixed
    }

    pub fn solve(&mut self, battle: &Battle) -> PolicyBoundReport {
        let report = self.solver.solve(battle, None);
        let value = report.bounds;
        let (guaranteed_score, unrestricted_game_value) = if self.side == 0 {
            (value, Bounds { lo: value.lo, hi: 1.0 })
        } else {
            (Bounds { lo: 1.0 - value.hi, hi: 1.0 - value.lo },
                Bounds { lo: 0.0, hi: value.hi })
        };
        PolicyBoundReport { guaranteed_score, unrestricted_game_value,
            stop: report.stop, runs: report.runs }
    }

    pub fn with_sleep_talk(mut self) -> Self {
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        self.solver.fixed_sleep_talk = true;
        self
    }

    /// Refine one first opponent reply, retaining every later opponent choice.
    /// Returned bounds are side-zero cell values, not whole-game guarantees.
    pub fn solve_reply(&mut self, battle: &Battle, reply: SearchChoice,
        threshold: Option<f64>) -> SolveReport {
        assert!(threshold.is_none_or(|t| t.is_finite() && (0.0..=1.0).contains(&t)));
        let started = Instant::now();
        let budget = self.solver.cfg.work_budget;
        self.solver.cfg.work_budget = 0;
        self.solver.solve(battle, None);
        self.solver.cfg.work_budget = budget;
        let root = self.solver.node_key(battle);
        assert!(!self.solver.closed.contains_key(&root), "reply analysis requires a retained matrix");
        self.solver.init_cells(root);
        let matrix = self.solver.root_payoffs(battle).unwrap();
        assert_eq!(matrix.actions[self.side].len(), 1);
        let cell = matrix.actions[1-self.side].iter().position(|a| *a == Some(reply))
            .expect("opponent reply must be legal");
        let start_runs = self.solver.stats.runs;
        let work_limit = start_runs.saturating_add(budget);
        let direction = if self.side == 0 { BoundDirection::Lower } else { BoundDirection::Upper };
        let mut idle = 0;
        loop {
            let current = &self.solver.nodes[&root].cells[cell];
            let bounds = Bounds { lo: current.lo, hi: current.hi };
            let stop = if bounds.width() <= self.solver.cfg.eps { Some(Stop::WidthMet) }
                else if threshold.is_some_and(|t| bounds.lo > t) { Some(Stop::ProvenAbove) }
                else if threshold.is_some_and(|t| bounds.hi < t) { Some(Stop::ProvenBelow) }
                else if self.solver.cfg.time_budget.is_some_and(|limit| started.elapsed() >= limit) { Some(Stop::TimeBudget) }
                else if self.solver.stats.runs >= work_limit || idle > 256 { Some(Stop::WorkExhausted) }
                else if self.solver.nodes.len() + self.solver.closed.len() >= self.solver.cfg.node_budget { Some(Stop::NodeBudget) }
                else { None };
            if let Some(stop) = stop {
                return SolveReport { bounds, stop, runs: self.solver.stats.runs-start_runs, strategies: None };
            }
            if self.solver.nodes[&root].battle.is_none() {
                self.solver.nodes.get_mut(&root).unwrap().battle = Some(battle.clone());
            }
            if self.solver.trial_from_cell(root,work_limit,Some(cell),Some(direction)) { idle = 0; }
                else { idle += 1; }
            if self.solver.cfg.fold_closed_nodes && self.solver.stats.trials % 4096 == 0 {
                self.solver.fold_closed_graph();
            }
        }
    }

    pub fn with_frozen_switch(mut self) -> Self {
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        self.solver.fixed_frozen_switch = true;
        self
    }

    pub fn with_rest_threshold(mut self, percent: u8) -> Self {
        assert!((1..100).contains(&percent));
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        self.solver.fixed_rest_percent = Some(percent);
        self
    }

    pub fn with_recover_threshold(mut self, percent: u8) -> Self {
        assert!((1..100).contains(&percent));
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        self.solver.fixed_recover_percent = Some(percent);
        self
    }

    pub fn with_recover_against_last(mut self, species: Vec<nc2000_engine::dex::SpeciesId>) -> Self {
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        assert!(self.solver.fixed_recover_percent.is_some() && !species.is_empty());
        self.solver.fixed_recover_last_foes=Some(species);
        self
    }

    pub fn with_rest_after_boom(mut self) -> Self {
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        assert!(self.solver.fixed_rest_percent.is_some(),"Rest threshold must be configured first");
        self.solver.fixed_rest_after_boom=true;
        self
    }

    pub fn with_rest_on_wake(mut self) -> Self {
        assert!(self.solver.nodes.is_empty() && self.solver.closed.is_empty(),
            "fixed strategy must be configured before search");
        assert!(self.solver.fixed_rest_percent.is_some(),"Rest threshold must be configured first");
        self.solver.fixed_rest_on_wake=true;
        self
    }

    pub fn node_count(&self) -> usize { self.solver.node_count() }
    pub fn retained_battles(&self) -> impl Iterator<Item=&Battle> { self.solver.retained_battles() }
    pub fn graph_diagnostics(&self) -> GraphDiagnostics { self.solver.graph_diagnostics() }
    pub fn restricted_root_payoffs(&self, battle: &Battle) -> Option<RootPayoffs> {
        self.solver.root_payoffs(battle)
    }
    pub fn root_chance_diagnostics(&self, battle: &Battle) -> Option<Vec<ChanceDiagnostics>> {
        self.solver.root_chance_diagnostics(battle)
    }
    pub fn stats(&self) -> &BoundStats { &self.solver.stats }

    pub fn into_value_certificates(self) -> ValueCertificates {
        let side = self.side;
        let bounds = self.solver.nodes.into_iter().map(|(key, node)|
            (key, Bounds { lo: node.lo, hi: node.hi }))
            .chain(self.solver.closed);
        let entries = bounds.filter_map(|(key, value)| {
            let value = if side == 0 { Bounds { lo: value.lo, hi: 1.0 } }
                else { Bounds { lo: 0.0, hi: value.hi } };
            (value.lo > 0.0 || value.hi < 1.0).then_some((key, value))
        }).collect();
        ValueCertificates { dex_fingerprint: self.solver.dex.replay_fingerprint, entries }
    }
}

#[derive(Default)]
pub struct BoundCache {
    dex_fingerprint: Option<[u8; 16]>,
    nodes: HashMap<NodeKey, Node>,
    closed: HashMap<NodeKey, Bounds>,
    closed_strategies: HashMap<NodeKey, [StrategyCertificate; 2]>,
    roots: HashSet<NodeKey>,
    stats: BoundStats,
    value_certificates: HashMap<NodeKey, Bounds>,
}

impl BoundCache {
    pub fn len(&self) -> usize {
        self.nodes.len() + self.closed.len() + self.value_certificates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

const TOTAL: f64 = (1u64 << 32) as f64;

impl<'d> BoundSolver<'d> {
    pub fn new(dex: &'d Dex, cfg: BoundConfig) -> Self {
        BoundSolver {
            dex,
            cfg,
            nodes: HashMap::new(),
            closed: HashMap::new(),
            closed_strategies: HashMap::new(),
            roots: HashSet::new(),
            active_stall: None,
            active_two_sided: None,
            fixed_policy: None,
            fixed_first_action: None,
            fixed_sleep_talk: false,
            fixed_frozen_switch: false,
            fixed_rest_percent: None,
            fixed_recover_percent: None,
            fixed_recover_last_foes: None,
            fixed_rest_after_boom: false,
            fixed_rest_on_wake: false,
            value_certificates: HashMap::new(),
            hp_transitions: crate::hp_transition::HpTransitionCache::default(),
            stats: BoundStats::default(),
        }
    }

    pub fn from_cache(dex: &'d Dex, cfg: BoundConfig, cache: BoundCache) -> Self {
        if cache.dex_fingerprint != Some(dex.replay_fingerprint) {
            return Self::new(dex, cfg);
        }
        Self {
            dex,
            cfg,
            nodes: cache.nodes,
            closed: cache.closed,
            closed_strategies: cache.closed_strategies,
            roots: cache.roots,
            active_stall: None,
            active_two_sided: None,
            fixed_policy: None,
            fixed_first_action: None,
            fixed_sleep_talk: false,
            fixed_frozen_switch: false,
            fixed_rest_percent: None,
            fixed_recover_percent: None,
            fixed_recover_last_foes: None,
            fixed_rest_after_boom: false,
            fixed_rest_on_wake: false,
            value_certificates: cache.value_certificates,
            hp_transitions: crate::hp_transition::HpTransitionCache::default(),
            stats: cache.stats,
        }
    }

    pub fn into_cache(self) -> BoundCache {
        assert!(self.fixed_policy.is_none(), "restricted values cannot enter an unrestricted cache");
        BoundCache {
            dex_fingerprint: Some(self.dex.replay_fingerprint),
            nodes: self.nodes,
            closed: self.closed,
            closed_strategies: self.closed_strategies,
            roots: self.roots,
            stats: self.stats,
            value_certificates: self.value_certificates,
        }
    }

    /// Imports bounds on unrestricted values, before constructing a search graph.
    /// Fixed-policy values must first be projected onto their valid one-sided bounds.
    pub fn import_value_certificates(&mut self, certificates: ValueCertificates) -> usize {
        assert!(self.fixed_policy.is_none());
        assert!(self.nodes.is_empty() && self.closed.is_empty());
        assert_eq!(certificates.dex_fingerprint, self.dex.replay_fingerprint);
        for (key, value) in certificates.entries {
            let entry = self.value_certificates.entry(key).or_insert(Bounds { lo: 0.0, hi: 1.0 });
            entry.lo = entry.lo.max(value.lo);
            entry.hi = entry.hi.min(value.hi);
            assert!(entry.lo <= entry.hi, "inconsistent value certificates");
        }
        self.value_certificates.len()
    }

    pub fn into_value_certificates(self) -> ValueCertificates {
        assert!(self.fixed_policy.is_none(), "restricted values require one-sided projection");
        let entries = self.nodes.into_iter().map(|(key, node)|
            (key, Bounds { lo: node.lo, hi: node.hi }))
            .chain(self.closed)
            .filter(|(_, value)| value.lo > 0.0 || value.hi < 1.0)
            .collect();
        ValueCertificates { dex_fingerprint: self.dex.replay_fingerprint, entries }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn retained_battles(&self) -> impl Iterator<Item=&Battle> {
        self.nodes.values().filter_map(|node|node.battle.as_ref())
    }

    pub fn closed_count(&self) -> usize {
        self.closed.len()
    }

    pub fn root_chance_diagnostics(&self, b: &Battle) -> Option<Vec<ChanceDiagnostics>> {
        let node = self.nodes.get(&self.node_key(b))?;
        Some(node.cells.iter().map(|cell| {
            let mut children = HashMap::<NodeKey, f64>::new();
            let mut masses = HashMap::<NodeKey, f64>::new();
            let mut largest_edge_uncertainty: f64 = 0.0;
            for &(mass, key) in &cell.resolved {
                let child = &self.nodes[&key];
                let uncertainty = mass * (child.hi - child.lo);
                largest_edge_uncertainty = largest_edge_uncertainty.max(uncertainty);
                *children.entry(key).or_default() += uncertainty;
                *masses.entry(key).or_default() += mass;
            }
            let mut ordered: Vec<_> = children.iter().map(|(&key, &width)| (key, width)).collect();
            ordered.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let largest_children = ordered.into_iter().take(5).map(|(key, uncertainty)| {
                let child = &self.nodes[&key];
                ChanceChildDiagnostics {
                    key: format!("{key:?}"), mass: masses[&key],
                    bounds: Bounds { lo: child.lo, hi: child.hi }, uncertainty,
                    runs: child.cells.iter().map(|c| c.runs).sum(),
                    visits: child.cells.iter().map(|c| c.visits).sum(),
                    resource_rank: child.resource_rank.as_ref().map(|(_, rank, _)| *rank),
                    battle: child.battle.clone(),
                    actions: [child.acts0.clone(), child.acts1.clone()],
                    cells: child.cells.iter().map(|c| Bounds { lo: c.lo, hi: c.hi }).collect(),
                }
            }).collect();
            ChanceDiagnostics {
                largest_children,
                visits: cell.visits,
                runs: cell.runs,
                pending_mass: cell.pending_mass,
                pending_scripts: cell.pending.len(),
                resolved_edges: cell.resolved.len(),
                unique_children: children.len(),
                child_uncertainty: children.values().sum(),
                largest_edge_uncertainty,
                largest_child_uncertainty: children.values().copied().fold(0.0, f64::max),
                expanded_children: children.keys().filter(|key| !self.nodes[key].cells.is_empty()).count(),
                max_child_runs: children.keys().map(|key| self.nodes[key].cells.iter().map(|c|c.runs).sum()).max().unwrap_or(0),
                total_child_runs: children.keys().map(|key| self.nodes[key].cells.iter().map(|c|c.runs).sum::<usize>()).sum(),
            }
        }).collect())
    }

    pub fn graph_diagnostics(&self) -> GraphDiagnostics {
        let mut result = GraphDiagnostics::default();
        for node in self.nodes.values() {
            if node.cells.is_empty() {
                result.unexpanded_nodes += 1;
            } else {
                result.initialized_nodes += 1;
            }
            result.stored_battles += usize::from(node.battle.is_some());
            for cell in &node.cells {
                result.pending_scripts += cell.pending.len();
                result.pending_script_entries += cell.pending.iter().map(|p|p.script.len()).sum::<usize>();
                result.resolved_edges += cell.resolved.len();
                result.unique_child_edges += cell.resolved.iter()
                    .map(|&(_, key)| key).collect::<HashSet<_>>().len();
            }
        }
        result
    }

    /// Tighten the root's certified interval until a stop condition fires.
    /// `tau` = optional threshold band (lo, hi) for certificate mode.
    /// Never discards work: repeated calls resume where the graph stands.
    pub fn solve(&mut self, b: &Battle, tau: Option<(f64, f64)>) -> SolveReport {
        self.solve_target(b, tau, None)
    }

    pub fn solve_strategy(&mut self, b: &Battle, side: usize, max_loss: f64) -> SolveReport {
        assert!(side < 2 && max_loss.is_finite() && (0.0..=1.0).contains(&max_loss));
        self.solve_target(b, None, Some((side, max_loss)))
    }

    fn solve_target(
        &mut self,
        b: &Battle,
        tau: Option<(f64, f64)>,
        strategy_target: Option<(usize, f64)>,
    ) -> SolveReport {
        let started = self.cfg.time_budget.map(|_| Instant::now());
        self.active_stall = self
            .cfg
            .monotone_stall_scheduling
            .then(|| classify_one_sided_heal(b, self.dex).ok())
            .flatten();
        if self.active_stall.is_some() {
            self.stats.monotone_roots += 1;
        }
        self.active_two_sided =
            if self.active_stall.is_none() && self.cfg.two_sided_resource_scheduling {
                classify_two_sided_heal(b, self.dex).ok()
            } else {
                None
            };
        if self.active_two_sided.is_some() {
            self.stats.two_sided_roots += 1;
        }
        let lookup = self.node_key(b);
        if let Some(&bounds) = self.closed.get(&lookup) {
            return SolveReport {
                bounds,
                stop: Stop::WidthMet,
                runs: 0,
                strategies: self.closed_strategies.get(&lookup).cloned(),
            };
        }
        let root = self.intern(b.clone());
        self.roots.insert(root);
        let start_runs = self.stats.runs;
        let work_limit = start_runs.saturating_add(self.cfg.work_budget);
        let mut idle = 0usize;
        loop {
            if self.cfg.fold_closed_nodes && self.nodes.len() >= self.cfg.node_budget {
                self.fold_closed_graph();
            }
            let n = &self.nodes[&root];
            let bounds = Bounds { lo: n.lo, hi: n.hi };
            let strategy_met = strategy_target.is_some_and(|(side, loss)| {
                node_strategies(n).is_some_and(|policies| policies[side].meets(loss))
            });
            let stop = if strategy_met {
                Some(Stop::StrategyMet)
            } else if bounds.width() <= self.cfg.eps {
                Some(Stop::WidthMet)
            } else if let Some((tlo, thi)) = tau {
                if bounds.lo > thi {
                    Some(Stop::ProvenAbove)
                } else if bounds.hi < tlo {
                    Some(Stop::ProvenBelow)
                } else if bounds.lo >= tlo && bounds.hi <= thi {
                    Some(Stop::Contained)
                } else {
                    None
                }
            } else {
                None
            };
            let stop = stop.or(if started.zip(self.cfg.time_budget).is_some_and(|(start, limit)| start.elapsed() >= limit) {
                Some(Stop::TimeBudget)
            } else if self.stats.runs >= work_limit {
                Some(Stop::WorkExhausted)
            } else if self.nodes.len() >= self.cfg.node_budget {
                Some(Stop::NodeBudget)
            } else {
                None
            });
            if let Some(stop) = stop {
                return SolveReport {
                    bounds,
                    stop,
                    runs: self.stats.runs - start_runs,
                    strategies: node_strategies(n),
                };
            }
            let (cell, direction) = if self.cfg.strategy_refinement {
                strategy_target.map_or((None,None), |(side,loss)|self.select_strategy_refinement(root,side,loss))
            } else { (None,None) };
            if self.trial_from_cell(root, work_limit, cell, direction) {
                idle = 0;
                if self.cfg.fold_closed_nodes
                    && (self.stats.trials % 4096 == 0
                        || self.nodes.len().saturating_mul(4)
                            >= self.cfg.node_budget.saturating_mul(3))
                {
                    self.fold_closed_graph();
                }
            } else {
                idle += 1;
                if idle > 256 {
                    // frontier unreachable under the current policy —
                    // report honestly rather than spinning
                    let n = &self.nodes[&root];
                    return SolveReport {
                        bounds: Bounds { lo: n.lo, hi: n.hi },
                        stop: Stop::WorkExhausted,
                        runs: self.stats.runs - start_runs,
                        strategies: node_strategies(n),
                    };
                }
            }
        }
    }

    /// Current certified interval of a state, if it is in the graph.
    pub fn peek(&self, b: &Battle) -> Option<Bounds> {
        let key = self.node_key(b);
        self.nodes
            .get(&key)
            .map(|n| Bounds { lo: n.lo, hi: n.hi })
            .or_else(|| self.closed.get(&key).copied())
    }

    pub fn retain_root(&mut self, b: &Battle) {
        let root = self.node_key(b);
        let mut keep = HashSet::new();
        let mut pending = vec![root];
        while let Some(key) = pending.pop() {
            if keep.insert(key) {
                if let Some(node) = self.nodes.get(&key) {
                    pending.extend(node.cells.iter().flat_map(|cell| cell.resolved.iter().map(|&(_, k)| k)));
                }
            }
        }
        self.nodes.retain(|key, _| keep.contains(key));
        self.closed.retain(|key, _| keep.contains(key));
        self.closed_strategies.retain(|key, _| keep.contains(key));
        self.roots.clear();
        self.roots.insert(root);
    }

    /// Fix this side's first action; the opponent best-responds and both
    /// sides play optimally thereafter. Values are from `side`'s perspective.
    pub fn action_bounds(&self, b: &Battle, side: usize) -> Vec<(Option<SearchChoice>, Bounds)> {
        assert!(side < 2);
        let Some(node) = self.nodes.get(&self.node_key(b)) else {
            return Vec::new();
        };
        let n1 = node.acts1.len();
        let actions = if side == 0 { &node.acts0 } else { &node.acts1 };
        actions
            .iter()
            .enumerate()
            .map(|(a, &action)| {
                let opponents = if side == 0 { n1 } else { node.acts0.len() };
                let mut bounds = Bounds { lo: 1.0, hi: 1.0 };
                for reply in 0..opponents {
                    let cell = &node.cells[if side == 0 { a * n1 + reply } else { reply * n1 + a }];
                    let (lo, hi) = if side == 0 { (cell.lo, cell.hi) } else { (1.0 - cell.hi, 1.0 - cell.lo) };
                    bounds.lo = bounds.lo.min(lo);
                    bounds.hi = bounds.hi.min(hi);
                }
                (action, bounds)
            })
            .collect()
    }

    pub fn root_payoffs(&self, b: &Battle) -> Option<RootPayoffs> {
        let node = self.nodes.get(&self.node_key(b))?;
        if node.cells.is_empty() { return None; }
        Some(RootPayoffs {
            actions: [node.acts0.clone(), node.acts1.clone()],
            cells: node.cells.iter().map(|c| Bounds { lo: c.lo, hi: c.hi }).collect(),
            value: Bounds { lo: node.lo, hi: node.hi },
        })
    }

    /// Targets fix only the first decision; continuation remains optimal.
    /// Requires a retained root matrix, not a value-only compacted cache entry.
    pub fn solve_policy_losses(
        &mut self, b: &Battle, targets: &[(usize, Vec<f64>)], tolerance: f64,
    ) -> SolveReport {
        assert!(!targets.is_empty() && tolerance.is_finite() && tolerance > 0.0);
        assert!(b.outcome().is_none() && !self.closed.contains_key(&self.node_key(b)));
        let started = Instant::now();
        let budget = self.cfg.work_budget;
        self.cfg.work_budget = 0;
        self.solve(b, None);
        self.cfg.work_budget = budget;
        let root = self.node_key(b);
        self.init_cells(root);
        let targets: Vec<_> = targets.iter().map(|(side, policy)| {
            assert!(*side < 2);
            let count = if *side == 0 { self.nodes[&root].acts0.len() }
                else { self.nodes[&root].acts1.len() };
            assert_eq!(policy.len(), count);
            (*side, normalized_policy(policy).expect("valid target policy"))
        }).collect();
        let start_runs = self.stats.runs;
        let work_limit = start_runs.saturating_add(budget);
        let mut cursor = 0;
        let mut value_cursor = 0;
        let mut idle = 0;
        loop {
            let payoffs = self.root_payoffs(b).unwrap();
            let pending: Vec<_> = targets.iter().filter(|(side, policy)| {
                let loss = payoffs.policy_loss(*side, policy);
                loss.hi > tolerance && loss.width() > tolerance
            }).collect();
            let stop = if pending.is_empty() { Some(Stop::StrategyMet) }
                else if self.cfg.time_budget.is_some_and(|limit| started.elapsed() >= limit) { Some(Stop::TimeBudget) }
                else if self.stats.runs >= work_limit || idle > 256 { Some(Stop::WorkExhausted) }
                else if self.nodes.len() + self.closed.len() >= self.cfg.node_budget { Some(Stop::NodeBudget) }
                else { None };
            if let Some(stop) = stop {
                return SolveReport {
                    bounds: payoffs.value, stop, runs: self.stats.runs - start_runs,
                    strategies: node_strategies(&self.nodes[&root]),
                };
            }
            let (side, policy) = pending[cursor % pending.len()];
            cursor += 1;
            let (refine, sufficient) = payoffs.refinement_cell(*side, policy,
                self.cfg.target_aware_refinement.then_some(tolerance));
            self.stats.sufficient_reply_skips += sufficient;
            self.stats.target_filtered_trials += usize::from(sufficient > 0);
            let refine_value = (payoffs.value.width() > tolerance * 0.5 && cursor % 4 == 0)
                || refine.is_none();
            let direction = self.cfg.directed_policy_refinement.then(|| {
                if refine_value {
                    let (value_side, _) = pending[value_cursor % pending.len()];
                    value_cursor += 1;
                    if *value_side == 0 { BoundDirection::Upper } else { BoundDirection::Lower }
                } else if *side == 0 { BoundDirection::Lower } else { BoundDirection::Upper }
            });
            let cell = if refine_value { self.select_cell_for_bound(root, direction) }
                else { refine.unwrap().0 };
            if self.nodes[&root].battle.is_none() {
                self.nodes.get_mut(&root).unwrap().battle = Some(b.clone());
            }
            if self.trial_from_cell(root, work_limit, Some(cell), direction) { idle = 0; }
                else { idle += 1; }
            if self.cfg.fold_closed_nodes && self.stats.trials % 4096 == 0 {
                self.fold_closed_graph();
            }
        }
    }

    fn select_strategy_refinement(&mut self, root: NodeKey, side: usize, tolerance: f64) -> (Option<usize>, Option<BoundDirection>) {
        let node = &self.nodes[&root];
        let Some(policies) = node_strategies(node) else { return (None,None); };
        let policy: Vec<_> = policies[side].actions.iter().map(|(_,mass)|*mass).collect();
        let payoffs = RootPayoffs {
            actions:[node.acts0.clone(),node.acts1.clone()],
            cells:node.cells.iter().map(|c|Bounds {lo:c.lo,hi:c.hi}).collect(),
            value:Bounds {lo:node.lo,hi:node.hi},
        };
        let (refine,sufficient) = payoffs.refinement_cell(side,&policy,Some(tolerance));
        self.stats.sufficient_reply_skips += sufficient;
        self.stats.target_filtered_trials += usize::from(sufficient>0);
        let visits = self.stats.strategy_refinement_trials + self.stats.strategy_ceiling_trials;
        let refine_ceiling = refine.is_none() || (payoffs.value.width()>tolerance*0.5 && visits%4==3);
        let direction = if (side==0)==refine_ceiling { BoundDirection::Upper } else { BoundDirection::Lower };
        let cell = if refine_ceiling {
            self.stats.strategy_ceiling_trials += 1;
            self.select_cell_for_bound(root,Some(direction))
        } else {
            self.stats.strategy_refinement_trials += 1;
            refine.unwrap().0
        };
        (Some(cell),Some(direction))
    }

    fn node_key(&self, b: &Battle) -> NodeKey {
        let omit_damage = self.cfg.dead_damage_quotient && !b.damage_bookkeeping_observable(self.dex);
        if self.cfg.dead_quick_claw_quotient && !b.quick_claw_roll_observable(self.dex) {
            let key = b.state_key128_without_quick_claw_roll(self.dex, omit_damage);
            if omit_damage { NodeKey::NoDamageOrQuickClawRoll(key) } else { NodeKey::NoQuickClawRoll(key) }
        } else if omit_damage {
            NodeKey::NoDamageBookkeeping(b.state_key128_without_damage_bookkeeping())
        } else {
            NodeKey::Exact(b.state_key128())
        }
    }

    fn intern(&mut self, b: Battle) -> NodeKey {
        let key = self.node_key(&b);
        let stall_rank = self
            .active_stall
            .as_ref()
            .and_then(|class| class.rank(&b).ok().map(|rank| (class.clone(), rank)));
        let resource_rank = self.active_two_sided.as_ref().and_then(|class| {
            class.rank(&b).ok().map(|rank| {
                let handoff = classify_one_sided_heal(&b, self.dex).is_ok();
                (class.clone(), rank, handoff)
            })
        });
        if let Some(node) = self.nodes.get_mut(&key) {
            self.stats.reused_nodes += 1;
            if stall_rank.is_some() {
                node.stall_rank = stall_rank;
            }
            if resource_rank.is_some() {
                node.resource_rank = resource_rank;
            }
            return key;
        }
        let certified = self.value_certificates.get(&key).copied();
        self.stats.certificate_hits += usize::from(certified.is_some());
        let initial = certified.unwrap_or(Bounds { lo: 0.0, hi: 1.0 });
        let mut chance_class = b.sides[0].pokemon_left as u32 | (b.sides[1].pokemon_left as u32) << 4;
        for side in 0..2 {
            chance_class |= b.active_id(side).map_or(0, |id| b.poke(id).status as u32 + 1) << (8 + 4 * side);
        }
        let needs = b.needs_choice();
        chance_class |= u32::from(needs[0]) << 16 | u32::from(needs[1]) << 17;
        let remaining_hp = [0,1].map(|side|b.sides[side].party.iter()
            .map(|&slot|b.sides[side].roster[slot as usize].hp.max(0)).sum());
        let node = match b.outcome() {
            Some(o) => {
                let v = match o {
                    Outcome::P1Win => 1.0,
                    Outcome::Tie => 0.5,
                    Outcome::P2Win => 0.0,
                };
                Node {
                    chance_class,
                    remaining_hp,
                    battle: None,
                    acts0: vec![],
                    acts1: vec![],
                    cells: vec![],
                    lo: v,
                    hi: v,
                    strat: None,
                    active_rows: Vec::new(),
                    active_cols: Vec::new(),
                    select_count: 0,
                    fair_cursor: 0,
                    stall_rank,
                    resource_rank,
                }
            }
            None => Node {
                chance_class,
                remaining_hp,
                battle: Some(b),
                acts0: vec![],
                acts1: vec![],
                cells: vec![],
                lo: initial.lo,
                hi: initial.hi,
                strat: None,
                active_rows: Vec::new(),
                active_cols: Vec::new(),
                select_count: 0,
                fair_cursor: 0,
                stall_rank,
                resource_rank,
            },
        };
        if let Some((_, rank, handoff)) = node.resource_rank.as_ref() {
            if *handoff {
                self.stats.one_sided_handoffs += 1;
            }
            self.stats.min_healing_pp = Some(
                self.stats
                    .min_healing_pp
                    .map_or(rank.healing_pp, |old| old.min(rank.healing_pp)),
            );
            self.stats.min_resource_pp = Some(
                self.stats
                    .min_resource_pp
                    .map_or(rank.pp_total, |old| old.min(rank.pp_total)),
            );
        }
        self.nodes.insert(key, node);
        self.stats.peak_nodes = self.stats.peak_nodes.max(self.nodes.len());
        key
    }

    fn init_cells(&mut self, key: NodeKey) {
        let node = self.nodes.get_mut(&key).unwrap();
        if !node.cells.is_empty() || node.battle.is_none() {
            return;
        }
        let mut probe = node.battle.as_ref().unwrap().clone();
        let needs = probe.needs_choice();
        let mut acts = |side: usize| -> Vec<Option<SearchChoice>> {
            if needs[side] {
                let legal = probe.legal_choices(self.dex, side);
                if let Some((fixed_side, priority)) = &self.fixed_policy {
                    if side == *fixed_side {
                        if let Some((root, action)) = self.fixed_first_action {
                            if root == key {
                                assert!(legal.contains(&action));
                                return vec![Some(action)];
                            }
                        }
                        if self.fixed_frozen_switch && probe.active_id(side)
                            .is_some_and(|id| probe.poke(id).status == nc2000_engine::state::Status::Frz) {
                            if let Some(&action) = legal.iter().find(|choice| match choice {
                                SearchChoice::Switch(pos) => {
                                    let team = &probe.sides[side];
                                    team.roster[team.party[(*pos - 1) as usize] as usize].status != nc2000_engine::state::Status::Frz
                                }
                                _ => false,
                            }) { return vec![Some(action)]; }
                        }
                        if self.fixed_sleep_talk && probe.active_id(side)
                            .is_some_and(|id| probe.poke(id).status == nc2000_engine::state::Status::Slp) {
                            if let Some(action) = self.dex.moves.id("sleeptalk").map(SearchChoice::Move)
                                .filter(|action| legal.contains(action)) {
                                return vec![Some(action)];
                            }
                        }
                        if let Some(percent) = self.fixed_recover_percent {
                            if self.fixed_recover_last_foes.as_ref().is_none_or(|species| {
                                probe.sides[1-side].pokemon_left==1 && probe.active_id(1-side)
                                    .is_some_and(|id|species.contains(&probe.poke(id).species))
                            }) && probe.active_id(side).is_some_and(|id| {
                                let pokemon=probe.poke(id);
                                !matches!(pokemon.status,nc2000_engine::state::Status::Slp|nc2000_engine::state::Status::Frz)
                                    && i64::from(pokemon.hp)*100 <= i64::from(pokemon.maxhp)*i64::from(percent)
                            }) {
                                if let Some(action)=self.dex.moves.id("recover").map(SearchChoice::Move)
                                    .filter(|action|legal.contains(action)) { return vec![Some(action)]; }
                            }
                        }
                        if let Some(percent) = self.fixed_rest_percent {
                            if (!self.fixed_rest_after_boom || !has_live_boom_move(self.dex,&probe,1-side))
                                && probe.active_id(side).is_some_and(|id| {
                                let pokemon=probe.poke(id);
                                pokemon.status!=nc2000_engine::state::Status::Frz
                                    && (pokemon.status!=nc2000_engine::state::Status::Slp || (self.fixed_rest_on_wake
                                        && pokemon.status_state.get_int(nc2000_engine::state::DK::Time)<=1))
                                    && i64::from(pokemon.hp)*100 <= i64::from(pokemon.maxhp)*i64::from(percent)
                            }) {
                                if let Some(action)=self.dex.moves.id("rest").map(SearchChoice::Move)
                                    .filter(|action|legal.contains(action)) {
                                    if probe.active_id(side).is_some_and(|id|probe.poke(id).status==nc2000_engine::state::Status::Slp) {
                                        self.stats.fixed_waking_rest_nodes+=1;
                                    }
                                    return vec![Some(action)];
                                }
                            }
                        }
                        let choice = priority.iter().find(|choice| legal.contains(choice))
                            .or_else(|| legal.first()).expect("requested side has a legal choice");
                        return vec![Some(*choice)];
                    }
                }
                legal
                    .into_iter()
                    .map(Some)
                    .collect()
            } else {
                vec![None]
            }
        };
        let (a0, a1) = (acts(0), acts(1));
        let node = self.nodes.get_mut(&key).unwrap();
        let n_cells = a0.len().max(1) * a1.len().max(1);
        node.cells = (0..n_cells)
            .map(|_| Cell {
                visits: 0,
                runs: 0,
                fair_cursor: 0,
                resolved: vec![],
                fixed_lo: 0.0,
                fixed_hi: 0.0,
                pending: vec![Pending {
                    script: vec![],
                    mass: 1.0,
                }],
                pending_mass: 1.0,
                lo: 0.0,
                hi: 1.0,
                tried_eager: false,
            })
            .collect();
        node.acts0 = a0;
        node.acts1 = a1;
        node.active_rows = vec![true; node.acts0.len()];
        node.active_cols = vec![true; node.acts1.len()];
    }

    fn trial_from_cell(&mut self, root: NodeKey, work_limit: usize, root_cell: Option<usize>, direction: Option<BoundDirection>) -> bool {
        match direction {
            Some(BoundDirection::Lower) => self.stats.lower_directed_trials += 1,
            Some(BoundDirection::Upper) => self.stats.upper_directed_trials += 1,
            None => {}
        }
        self.stats.trials += 1;
        let mut did_work = false;
        let mut path: Vec<(NodeKey, usize)> = Vec::new();
        let start_runs = self.stats.runs;
        let mut cur = root;
        let mut visited_depth = 0;
        let mut hit_depth_limit = false;
        for depth in 0..self.cfg.trial_depth {
            visited_depth = depth + 1;
            let n = &self.nodes[&cur];
            if n.hi - n.lo <= 1e-12 && !(depth == 0 && root_cell.is_some()) {
                break; // resolved (incl. terminals)
            }
            if n.cells.is_empty() {
                self.init_cells(cur);
            }
            let ci = if depth == 0 { root_cell.unwrap_or_else(|| self.select_cell_for_bound(cur, direction)) }
                else { self.select_cell_for_bound(cur, direction) };
            path.push((cur, ci));
            if self.try_eager(cur, ci, work_limit) {
                did_work = true;
                break;
            }
            if self.stats.runs >= work_limit {
                break;
            }
            let cell = &self.nodes[&cur].cells[ci];
            let closure_mode = (self.active_stall.is_some() || self.active_two_sided.is_some())
                && self.cfg.resource_closure_period > 0
                && (cell.visits + 1) % self.cfg.resource_closure_period == 0
                && self.cfg.resource_closure_nodes.map_or_else(
                    || self.nodes.len().saturating_mul(4) >= self.cfg.node_budget.saturating_mul(3),
                    |threshold| self.nodes.len() >= threshold,
                );
            if closure_mode {
                self.stats.resource_closure_visits += 1;
                self.stats.first_resource_closure_nodes.get_or_insert(self.nodes.len());
            }
            let best_child = if closure_mode {
                if let Some(class) = self.active_stall.as_ref() {
                    let mut best: Option<(i64, f64, NodeKey)> = None;
                    for &(mass, child) in &cell.resolved {
                        let node = &self.nodes[&child];
                        let Some((child_class, rank)) = node.stall_rank.as_ref() else {
                            continue;
                        };
                        if child_class != class {
                            continue;
                        }
                        let width_mass = mass * (node.hi - node.lo);
                        let replace = best.is_none_or(|(r, w, _)| {
                            rank.value < r || (rank.value == r && width_mass > w)
                        });
                        if replace {
                            best = Some((rank.value, width_mass, child));
                        }
                    }
                    best.map(|(_, width_mass, child)| (width_mass, child))
                } else {
                    let class = self.active_two_sided.as_ref().unwrap();
                    let mut best: Option<((u8, i64, i64, i64), f64, NodeKey)> = None;
                    for &(mass, child) in &cell.resolved {
                        let node = &self.nodes[&child];
                        let Some((child_class, rank, handoff)) = node.resource_rank.as_ref() else {
                            continue;
                        };
                        if child_class != class {
                            continue;
                        }
                        let key = (
                            u8::from(!*handoff),
                            rank.healing_pp,
                            rank.pp_total,
                            rank.turns_remaining,
                        );
                        let width_mass = mass * (node.hi - node.lo);
                        let replace = best.as_ref().is_none_or(|(old, w, _)| {
                            key < *old || (key == *old && width_mass > *w)
                        });
                        if replace {
                            best = Some((key, width_mass, child));
                        }
                    }
                    best.map(|(_, width_mass, child)| (width_mass, child))
                }
            } else {
                None
            }
            .or_else(|| self.material_chance_child(cur, ci, direction))
            .or_else(|| self.grouped_chance_child(cur, ci))
            .or_else(|| {
                cell.resolved
                    .iter()
                    .map(|&(m, k)| {
                        let c = &self.nodes[&k];
                        (m * (c.hi - c.lo), k)
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            });
            let pending_mass = cell.pending_mass;
            let has_pending = !cell.pending.is_empty();
            let directed_descent = self.cfg.chance_material_order && direction.is_some()
                && self.cfg.material_descent_period>0 && depth+1<self.cfg.trial_depth
                && (cell.visits+1)%self.cfg.material_descent_period==0;
            let best_child = self.fair_chance_child(cur, ci).or(best_child);
            match best_child {
                Some((cw, k)) if cw > pending_mass.max(self.cfg.descend_floor)
                    || (directed_descent && cw>1e-12) => {
                    self.stats.material_descent_bypasses+=usize::from(has_pending && cw<=pending_mass.max(self.cfg.descend_floor));
                    cur = k;
                    hit_depth_limit = depth + 1 == self.cfg.trial_depth;
                    continue;
                }
                _ if has_pending => {
                    self.expand(cur, ci);
                    did_work = true;
                    break;
                }
                Some((_, k)) => {
                    // cell fully expanded but children below the descend
                    // floor: descend anyway — a zero-work trial would spin
                    // the solve loop forever (runs never advance).
                    cur = k;
                    hit_depth_limit = depth + 1 == self.cfg.trial_depth;
                    continue;
                }
                None => break,
            }
        }
        if hit_depth_limit {
            self.stats.depth_limit_trials += 1;
        }
        if self.stats.trial_depth_counts.len() <= visited_depth {
            self.stats.trial_depth_counts.resize(visited_depth + 1, 0);
        }
        self.stats.trial_depth_counts[visited_depth] += 1;
        for &(key, ci) in &path {
            let cell = &mut self.nodes.get_mut(&key).unwrap().cells[ci];
            cell.visits += 1;
            cell.runs += self.stats.runs - start_runs;
        }
        for &(k, _) in path.iter().rev() {
            if !self.backup(k) {
                break; // bounds unchanged ⇒ ancestors can't change either
            }
        }
        did_work
    }

    fn material_chance_child(&self, key: NodeKey, ci: usize, direction: Option<BoundDirection>) -> Option<(f64, NodeKey)> {
        if !self.cfg.chance_material_order { return None; }
        let side = match direction { Some(BoundDirection::Lower)=>0,Some(BoundDirection::Upper)=>1,None=>return None };
        self.nodes[&key].cells[ci].resolved.iter().filter_map(|&(mass,child)| {
            let node=&self.nodes[&child];
            let uncertainty=mass*(node.hi-node.lo);
            (uncertainty>1e-12).then_some((uncertainty,child))
        }).min_by(|(wa,a),(wb,b)| {
            let a_hp=self.nodes[a].remaining_hp;let b_hp=self.nodes[b].remaining_hp;
            a_hp[1-side].cmp(&b_hp[1-side]).then_with(||b_hp[side].cmp(&a_hp[side]))
                .then_with(||wb.total_cmp(wa)).then_with(||a.cmp(b))
        })
    }

    fn grouped_chance_child(&self, key: NodeKey, ci: usize) -> Option<(f64, NodeKey)> {
        if !self.cfg.chance_status_groups { return None; }
        let mut groups = std::collections::BTreeMap::<u32, (f64, f64, NodeKey)>::new();
        for &(mass, child) in &self.nodes[&key].cells[ci].resolved {
            let node = &self.nodes[&child];
            let uncertainty = mass * (node.hi - node.lo);
            if uncertainty <= 1e-12 { continue; }
            let group = groups.entry(node.chance_class).or_insert((0.0, uncertainty, child));
            group.0 += uncertainty;
            if uncertainty > group.1 || (uncertainty == group.1 && child < group.2) {
                group.1 = uncertainty;
                group.2 = child;
            }
        }
        groups.into_values().max_by(|a,b|a.0.total_cmp(&b.0)).map(|(_,mass,child)|(mass,child))
    }

    fn fair_chance_child(&mut self, key: NodeKey, ci: usize) -> Option<(f64, NodeKey)> {
        let period = self.cfg.chance_fair_period;
        let cell = &self.nodes[&key].cells[ci];
        if period == 0 || (cell.visits + 1) % period != 0 { return None; }
        let eligible: Vec<_> = cell.resolved.iter().filter_map(|&(mass, child)| {
            let node = &self.nodes[&child];
            let uncertainty = mass * (node.hi - node.lo);
            (uncertainty > 1e-12).then_some((uncertainty, child))
        }).collect();
        if eligible.is_empty() { return None; }
        let cell = &mut self.nodes.get_mut(&key).unwrap().cells[ci];
        let selected = eligible[cell.fair_cursor % eligible.len()];
        cell.fair_cursor = cell.fair_cursor.wrapping_add(1);
        Some(selected)
    }

    /// A requested direction selects its certificate witness; otherwise the
    /// witnesses alternate. Every 32nd visit covers unresolved live cells
    /// regardless of witness support.
    fn select_cell_for_bound(&mut self, key: NodeKey, direction: Option<BoundDirection>) -> usize {
        const FAIR_PERIOD: usize = 32;
        const EPS: f64 = 1e-12;

        let support_br = self.cfg.support_br_scheduling;
        let (selected, pick_kind) = {
            let node = self.nodes.get_mut(&key).unwrap();
            let (n0, n1) = (node.acts0.len(), node.acts1.len());
            node.select_count += 1;
            let mut active_cells = Vec::new();
            for i in 0..n0 {
                if !node.active_rows[i] {
                    continue;
                }
                for j in 0..n1 {
                    if node.active_cols[j] {
                        active_cells.push(i * n1 + j);
                    }
                }
            }
            let widest = || {
                active_cells
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let aw = node.cells[a].hi - node.cells[a].lo;
                        let bw = node.cells[b].hi - node.cells[b].lo;
                        aw.partial_cmp(&bw).unwrap()
                    })
                    .expect("a live matrix keeps an action per side")
            };

            if support_br && node.select_count % FAIR_PERIOD == 0 {
                let unresolved: Vec<usize> = active_cells
                    .iter()
                    .copied()
                    .filter(|&ci| node.cells[ci].hi - node.cells[ci].lo > EPS)
                    .collect();
                if unresolved.is_empty() {
                    (widest(), 0u8)
                } else {
                    let ci = unresolved[node.fair_cursor % unresolved.len()];
                    node.fair_cursor = node.fair_cursor.wrapping_add(1);
                    (ci, 3u8)
                }
            } else if let Some(witness) = node.strat.as_ref() {
                if !support_br {
                    let mut best = (f64::NEG_INFINITY, widest());
                    for &ci in &active_cells {
                        let (i, j) = (ci / n1, ci % n1);
                        let c = &node.cells[ci];
                        let score = witness.xlo[i].max(witness.xhi[i])
                            * witness.ylo[j].max(witness.yhi[j])
                            * (c.hi - c.lo);
                        if score > best.0 {
                            best = (score, ci);
                        }
                    }
                    (if best.0 > 0.0 { best.1 } else { widest() }, 0u8)
                } else if direction.is_none() && node.select_count % 4 == 0 {
                    let mut best = (f64::NEG_INFINITY, widest());
                    for &ci in &active_cells {
                        let (i, j) = (ci / n1, ci % n1);
                        let c = &node.cells[ci];
                        let score = witness.xlo[i].max(witness.xhi[i])
                            * witness.ylo[j].max(witness.yhi[j])
                            * (c.hi - c.lo);
                        if score > best.0 {
                            best = (score, ci);
                        }
                    }
                    (if best.0 > 0.0 { best.1 } else { widest() }, 4u8)
                } else if direction == Some(BoundDirection::Lower)
                    || (direction.is_none() && node.select_count % 2 == 0) {
                    let guarantees: Vec<f64> = (0..n1)
                        .map(|j| {
                            (0..n0)
                                .map(|i| witness.xlo[i] * node.cells[i * n1 + j].lo)
                                .sum()
                        })
                        .collect();
                    let best_response = guarantees
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| node.active_cols[*j])
                        .map(|(_, &v)| v)
                        .fold(f64::INFINITY, f64::min);
                    let mut best = (f64::NEG_INFINITY, widest());
                    for &ci in &active_cells {
                        let (i, j) = (ci / n1, ci % n1);
                        let c = &node.cells[ci];
                        let support = witness.xlo[i] * witness.ylo[j];
                        let br = if guarantees[j] <= best_response + witness.gap_lo + EPS {
                            witness.xlo[i]
                        } else {
                            0.0
                        };
                        let score = support.max(br) * (c.hi - c.lo);
                        if score > best.0 {
                            best = (score, ci);
                        }
                    }
                    (if best.0 > 0.0 { best.1 } else { widest() }, 1u8)
                } else {
                    let guarantees: Vec<f64> = (0..n0)
                        .map(|i| {
                            (0..n1)
                                .map(|j| witness.yhi[j] * node.cells[i * n1 + j].hi)
                                .sum()
                        })
                        .collect();
                    let best_response = guarantees
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| node.active_rows[*i])
                        .map(|(_, &v)| v)
                        .fold(f64::NEG_INFINITY, f64::max);
                    let mut best = (f64::NEG_INFINITY, widest());
                    for &ci in &active_cells {
                        let (i, j) = (ci / n1, ci % n1);
                        let c = &node.cells[ci];
                        let support = witness.xhi[i] * witness.yhi[j];
                        let br = if guarantees[i] >= best_response - witness.gap_hi - EPS {
                            witness.yhi[j]
                        } else {
                            0.0
                        };
                        let score = support.max(br) * (c.hi - c.lo);
                        if score > best.0 {
                            best = (score, ci);
                        }
                    }
                    (if best.0 > 0.0 { best.1 } else { widest() }, 2u8)
                }
            } else {
                (widest(), 0u8)
            }
        };
        match pick_kind {
            1 => {
                self.stats.br_cell_picks += 1;
                self.stats.lower_br_picks += 1;
            }
            2 => {
                self.stats.br_cell_picks += 1;
                self.stats.upper_br_picks += 1;
            }
            3 => self.stats.fair_cell_picks += 1,
            4 => self.stats.legacy_support_picks += 1,
            _ => {}
        }
        selected
    }

    fn try_eager(&mut self, key: NodeKey, ci: usize, work_limit: usize) -> bool {
        let cap = self.cfg.cell_cap.min(work_limit.saturating_sub(self.stats.runs));
        if cap == 0 {
            return false;
        }
        let (battle, choices) = {
            let node = self.nodes.get_mut(&key).unwrap();
            let cell = &mut node.cells[ci];
            if cell.tried_eager || !cell.resolved.is_empty() {
                return false;
            }
            cell.tried_eager = true;
            let n1 = node.acts1.len();
            let (i, j) = (ci / n1, ci % n1);
            (
                node.battle.take().expect("frontier node keeps its battle"),
                [node.acts0[i], node.acts1[j]],
            )
        };
        let (leaves, mut pending, runs, scripts) = if self.cfg.retain_eager_prefix {
            assert!(!self.cfg.hp_transition_eager);
            let partial = enumerate_step_partial(self.dex, &battle, choices, cap);
            if partial.exhausted {
                self.stats.eager_failures += 1;
                self.stats.retained_eager_runs += partial.runs;
                self.stats.retained_eager_leaves += partial.leaves.len();
            }
            (partial.leaves, partial.pending.into_iter().map(|p|Pending {script:p.script,mass:p.mass}).collect::<Vec<_>>(), partial.runs, partial.leaf_scripts)
        } else {
            let step = if self.cfg.hp_transition_eager && self.cfg.hp_transition_capacity>0
                && self.cfg.dead_damage_quotient && self.cfg.dead_quick_claw_quotient {
                let cache=&mut self.hp_transitions;
                let stats=&mut self.stats;
                let dex=self.dex;
                let capacity=self.cfg.hp_transition_capacity;
                cache.enumerate(dex,&battle,choices,cap,capacity,|step,len| {
                    stats.hp_transition_hits+=usize::from(step.hit);
                    stats.hp_transition_eligible+=usize::from(step.eligible);
                    stats.hp_transition_eager_hits+=usize::from(step.hit);
                    stats.hp_transition_eager_eligible+=usize::from(step.eligible);
                    stats.hp_transition_rejected+=usize::from(step.rejected_hp);
                    stats.hp_transition_peak=stats.hp_transition_peak.max(len);
                })
            } else {
                enumerate_step(self.dex,&battle,choices,cap)
            };
            let Some(step) = step else {
                self.nodes.get_mut(&key).unwrap().battle = Some(battle);
                self.stats.runs += cap;
                self.stats.eager_runs += cap;
                self.stats.eager_failures += 1;
                return false;
            };
            (step.leaves, Vec::new(), step.runs, None)
        };
        self.stats.runs += runs;
        self.stats.eager_runs += runs;
        self.stats.expansions += 1;
        let mut agg: HashMap<NodeKey, f64> = HashMap::new();
        let mut fixed_lo = 0.0;
        let mut fixed_hi = 0.0;
        let mut scripts = scripts.map(|s|s.into_iter());
        for l in leaves {
            let script = scripts.as_mut().map(|s|s.next().expect("one script per retained leaf"));
            let prob = l.prob;
            self.check_stall_edge(&battle, &l.battle);
            if self.cfg.fold_terminal_nodes {
                if let Some(v) = outcome_value(&l.battle) {
                    fixed_lo += prob * v;
                    fixed_hi += prob * v;
                    continue;
                }
            }
            let lookup = self.node_key(&l.battle);
            if let Some(bounds) = self.closed.get(&lookup).copied() {
                fixed_lo += prob * bounds.lo;
                fixed_hi += prob * bounds.hi;
                continue;
            }
            if self.cfg.defer_eager_frontier && l.battle.outcome().is_none() && !self.nodes.contains_key(&lookup) {
                if let Some(script) = script {
                    pending.push(Pending { script, mass:prob });
                    self.stats.deferred_eager_leaves += 1;
                    continue;
                }
            }
            let k = self.intern(l.battle);
            *agg.entry(k).or_default() += prob;
        }
        self.nodes.get_mut(&key).unwrap().battle = Some(battle);
        let node = self.nodes.get_mut(&key).unwrap();
        let cell = &mut node.cells[ci];
        cell.resolved = agg.into_iter().map(|(k, m)| (m, k)).collect();
        cell.resolved.sort_unstable_by_key(|&(_, key)| key);
        cell.fixed_lo = fixed_lo;
        cell.fixed_hi = fixed_hi;
        cell.pending_mass = pending.iter().map(|p|p.mass).sum();
        cell.pending = pending;
        true
    }

    /// Expand the largest-mass pending script of one cell: execution or
    /// guarded reuse yields the representative leaf; every non-default class after the
    /// prefix becomes a pending sibling at its exact mass. The leaf plus
    /// siblings partition the expanded mass exactly.
    fn expand(&mut self, key: NodeKey, ci: usize) {
        let (battle, choices, pending) = {
            let node = self.nodes.get_mut(&key).unwrap();
            let n1 = node.acts1.len();
            let (i, j) = (ci / n1, ci % n1);
            let choices = [node.acts0[i], node.acts1[j]];
            let cell = &mut node.cells[ci];
            let started = self.cfg.profile_pending_selection.then(Instant::now);
            let pi = cell
                .pending
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.mass.partial_cmp(&b.1.mass).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            if let Some(started) = started {
                self.stats.pending_selection_ns += started.elapsed().as_nanos();
                self.stats.pending_selection_calls += 1;
                self.stats.pending_scan_items += cell.pending.len();
            }
            let pending = cell.pending.swap_remove(pi);
            (
                node.battle.take().expect("frontier node keeps its battle"),
                choices,
                pending,
            )
        };

        let (leaf, trace, runs) = if self.cfg.hp_transition_capacity>0
            && self.cfg.dead_damage_quotient && self.cfg.dead_quick_claw_quotient {
            let step=self.hp_transitions.run(self.dex,&battle,choices,&pending.script,self.cfg.hp_transition_capacity);
            self.stats.hp_transition_hits+=usize::from(step.hit);
            self.stats.hp_transition_eligible+=usize::from(step.eligible);
            self.stats.hp_transition_rejected+=usize::from(step.rejected_hp);
            self.stats.hp_transition_peak=self.stats.hp_transition_peak.max(self.hp_transitions.len());
            (step.battle,step.trace,usize::from(!step.hit))
        } else {
            let (leaf,trace)=run_scripted(self.dex,&battle,choices,&pending.script);
            (leaf,trace,1)
        };
        self.check_stall_edge(&battle, &leaf);
        self.nodes.get_mut(&key).unwrap().battle = Some(battle);
        self.stats.runs += runs;
        self.stats.expansions += 1;

        // masses from the trace: prefix[p] = Π_{q<p} prob(q)
        let mut prefix = 1.0f64;
        let mut siblings: Vec<Pending> = Vec::new();
        for (p, d) in trace.iter().enumerate() {
            if p >= pending.script.len() {
                for (c, &cnt) in d.counts.iter().enumerate() {
                    if c != d.chosen && cnt > 0 {
                        let mut script: Vec<usize> = trace[..p].iter().map(|t| t.chosen).collect();
                        script.push(c);
                        siblings.push(Pending {
                            script,
                            mass: prefix * cnt as f64 / TOTAL,
                        });
                    }
                }
            }
            prefix *= d.prob();
        }
        let leaf_mass = prefix;
        debug_assert!(
            ((leaf_mass + siblings.iter().map(|s| s.mass).sum::<f64>()) - pending.mass).abs()
                < 1e-9,
            "pending mass must be partitioned exactly"
        );

        let terminal_value = self
            .cfg
            .fold_terminal_nodes
            .then(|| outcome_value(&leaf))
            .flatten();
        let closed_bounds = if terminal_value.is_none() {
            self.closed.get(&self.node_key(&leaf)).copied()
        } else {
            None
        };
        let child =
            (terminal_value.is_none() && closed_bounds.is_none()).then(|| self.intern(leaf));
        let node = self.nodes.get_mut(&key).unwrap();
        let cell = &mut node.cells[ci];
        cell.pending_mass = (cell.pending_mass - pending.mass).max(0.0)
            + siblings.iter().map(|s| s.mass).sum::<f64>();
        cell.pending.extend(siblings);
        if let Some(v) = terminal_value {
            cell.fixed_lo += leaf_mass * v;
            cell.fixed_hi += leaf_mass * v;
        } else if let Some(bounds) = closed_bounds {
            cell.fixed_lo += leaf_mass * bounds.lo;
            cell.fixed_hi += leaf_mass * bounds.hi;
        } else {
            let child = child.unwrap();
            if self.cfg.aggregate_lazy_children {
                if let Some((mass, _)) = cell.resolved.iter_mut().find(|(_, key)| *key == child) {
                    *mass += leaf_mass;
                    self.stats.lazy_edge_merges += 1;
                } else {
                    cell.resolved.push((leaf_mass, child));
                }
            } else {
                cell.resolved.push((leaf_mass, child));
            }
        }
    }

    /// Recompute one node's cell and node bounds from current children.
    /// Bounds are clamped monotone; resolved/fully-expanded nodes drop
    /// their Battle snapshot. Returns whether the node's bounds moved.
    fn backup(&mut self, key: NodeKey) -> bool {
        let (n0, n1, cell_bounds): (usize, usize, Vec<(f64, f64)>) = {
            let node = &self.nodes[&key];
            let bounds = node
                .cells
                .iter()
                .map(|cell| {
                    let mut lo = cell.fixed_lo;
                    let mut hi = cell.fixed_hi + cell.pending_mass;
                    for &(m, k) in &cell.resolved {
                        let c = &self.nodes[&k];
                        lo += m * c.lo;
                        hi += m * c.hi;
                    }
                    (lo, hi)
                })
                .collect();
            (node.acts0.len(), node.acts1.len(), bounds)
        };

        if self.cfg.reuse_unchanged_backups {
            let node = self.nodes.get_mut(&key).unwrap();
            if node.strat.is_some()
                && node.cells.iter().zip(&cell_bounds).all(|(cell, &(lo, hi))| cell.lo == lo && cell.hi == hi)
            {
                let fully_expanded = (0..n0).filter(|&i| node.active_rows[i]).all(|i| {
                    (0..n1).filter(|&j| node.active_cols[j]).all(|j| node.cells[i * n1 + j].pending.is_empty())
                });
                if node.hi - node.lo <= 1e-12 || fully_expanded {
                    node.battle = None;
                }
                self.stats.reused_backups += 1;
                return false;
            }
        }

        let mlo: Vec<f64> = cell_bounds.iter().map(|b| b.0).collect();
        let mhi: Vec<f64> = cell_bounds.iter().map(|b| b.1).collect();
        let (removed_rows, removed_cols, dominance_checks, avoided_cells) =
            if self.cfg.certified_action_pruning {
                let node = self.nodes.get_mut(&key).unwrap();
                let before = node.active_rows.iter().filter(|&&v| v).count()
                    * node.active_cols.iter().filter(|&&v| v).count();
                let (removed_rows, removed_cols, checks) = prune_dominated_actions(
                    &mlo,
                    &mhi,
                    n0,
                    n1,
                    &mut node.active_rows,
                    &mut node.active_cols,
                );
                let after = node.active_rows.iter().filter(|&&v| v).count()
                    * node.active_cols.iter().filter(|&&v| v).count();
                (removed_rows, removed_cols, checks, before - after)
            } else {
                (0, 0, 0, 0)
            };
        self.stats.dominated_rows += removed_rows;
        self.stats.dominated_cols += removed_cols;
        self.stats.dominance_checks += dominance_checks;
        self.stats.avoided_cells += avoided_cells;

        let node = &self.nodes[&key];
        let active_rows = node.active_rows.clone();
        let active_cols = node.active_cols.clone();
        let active_n0 = active_rows.iter().filter(|&&v| v).count();
        let active_n1 = active_cols.iter().filter(|&&v| v).count();
        if active_n0 > 1 && active_n1 > 1 {
            self.stats.lp_solves += 2;
        }
        let slo = solve_restricted_game(&mlo, n0, n1, &active_rows, &active_cols);
        let shi = solve_restricted_game(&mhi, n0, n1, &active_rows, &active_cols);
        self.stats.worst_gap = self.stats.worst_gap.max(slo.gap).max(shi.gap);
        // `solve_matrix_full` reports the midpoint of its own best-response
        // bracket. Use the outward endpoints here: certificates must include
        // numeric LP error, even though measured gaps are around 1e-15.
        let lo = (slo.value - slo.gap * 0.5).max(0.0);
        let hi = (shi.value + shi.gap * 0.5).min(1.0);
        let strat = StageWitness {
            xlo: slo.x,
            ylo: slo.y,
            xhi: shi.x,
            yhi: shi.y,
            gap_lo: slo.gap,
            gap_hi: shi.gap,
        };

        let (fully_expanded, moved) = {
            let node = self.nodes.get_mut(&key).unwrap();
            for (cell, &(clo, chi)) in node.cells.iter_mut().zip(&cell_bounds) {
                cell.lo = clo;
                cell.hi = chi;
            }
            // monotone: lo never decreases, hi never increases
            let (plo, phi) = (node.lo, node.hi);
            node.lo = node.lo.max(lo.min(1.0));
            node.hi = node.hi.min(hi.max(0.0)).max(node.lo);
            node.strat = Some(strat);
            let moved = (node.lo - plo) > 1e-12 || (phi - node.hi) > 1e-12;
            let fully_expanded = (0..n0).filter(|&i| node.active_rows[i]).all(|i| {
                (0..n1)
                    .filter(|&j| node.active_cols[j])
                    .all(|j| node.cells[i * n1 + j].pending.is_empty())
            });
            (fully_expanded, moved)
        };
        let node = self.nodes.get_mut(&key).unwrap();
        if node.hi - node.lo <= 1e-12 || fully_expanded {
            node.battle = None;
        }
        moved
    }

    /// Compact every exact non-root node into its parents. This runs only
    /// between trials, after reverse-path backup. Cached parent bounds are
    /// deliberately left unchanged: replacing an edge by the same frozen
    /// interval is semantics-neutral, and the normal later backup propagates
    /// it without perturbing the current scheduling decision.
    fn fold_closed_graph(&mut self) {
        let fold: HashMap<NodeKey, Bounds> = self
            .nodes
            .iter()
            .filter_map(|(&key, node)| {
                (!self.roots.contains(&key) && node.lo == node.hi).then_some((
                    key,
                    Bounds {
                        lo: node.lo,
                        hi: node.hi,
                    },
                ))
            })
            .collect();
        if fold.is_empty() {
            return;
        }

        for node in self.nodes.values_mut() {
            for cell in &mut node.cells {
                let mut keep = Vec::with_capacity(cell.resolved.len());
                for (mass, child) in cell.resolved.drain(..) {
                    if let Some(bounds) = fold.get(&child) {
                        cell.fixed_lo += mass * bounds.lo;
                        cell.fixed_hi += mass * bounds.hi;
                    } else {
                        keep.push((mass, child));
                    }
                }
                cell.resolved = keep;
            }
        }
        for (key, bounds) in fold {
            let node = self.nodes.remove(&key).unwrap();
            if let Some(strategies) = node_strategies(&node) {
                self.closed_strategies.insert(key, strategies);
            }
            self.closed.insert(key, bounds);
            self.stats.closed_folds += 1;
        }
    }

    fn check_stall_edge(&mut self, parent: &Battle, child: &Battle) {
        if child.outcome().is_some() {
            return;
        }
        let invalid = self
            .active_stall
            .as_ref()
            .is_some_and(|class| class.check_edge(parent, child).is_err());
        if invalid {
            self.active_stall = None;
            self.stats.monotone_invalidations += 1;
        }
        let invalid_two = self
            .active_two_sided
            .as_ref()
            .is_some_and(|class| class.check_edge(parent, child).is_err());
        if invalid_two {
            self.active_two_sided = None;
            self.stats.two_sided_invalidations += 1;
        }
    }
}

fn node_strategies(node: &Node) -> Option<[StrategyCertificate; 2]> {
    let witness = node.strat.as_ref()?;
    let rows = node.acts0.len();
    let cols = node.acts1.len();
    let x = normalized_policy(&witness.xlo)?;
    let y = normalized_policy(&witness.yhi)?;
    let floor = (0..cols)
        .map(|j| (0..rows).map(|i| x[i] * node.cells[i * cols + j].lo).sum::<f64>())
        .fold(1.0, f64::min);
    let ceiling = (0..rows)
        .map(|i| (0..cols).map(|j| y[j] * node.cells[i * cols + j].hi).sum::<f64>())
        .fold(0.0, f64::max);
    // Include every opposing action, including pruned actions whose cell bounds
    // may still be loose. The value certificate alone cannot certify a policy.
    let pad = 1e-12 * (rows + cols + 1) as f64;
    let guaranteed = [(floor - pad).max(0.0), (1.0 - ceiling - pad).max(0.0)];
    let upper = [node.hi, 1.0 - node.lo];
    Some([
        StrategyCertificate {
            actions: node.acts0.iter().copied().zip(x).collect(),
            guaranteed_value: guaranteed[0],
            max_loss: (upper[0] - guaranteed[0]).max(0.0),
        },
        StrategyCertificate {
            actions: node.acts1.iter().copied().zip(y).collect(),
            guaranteed_value: guaranteed[1],
            max_loss: (upper[1] - guaranteed[1]).max(0.0),
        },
    ])
}

fn normalized_policy(policy: &[f64]) -> Option<Vec<f64>> {
    if policy.is_empty() || policy.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return None;
    }
    let total = policy.iter().sum::<f64>();
    (total.is_finite() && total > 0.0).then(|| policy.iter().map(|p| p / total).collect())
}

struct RestrictedSolution {
    value: f64,
    gap: f64,
    x: Vec<f64>,
    y: Vec<f64>,
}

fn solve_restricted_game(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    active_rows: &[bool],
    active_cols: &[bool],
) -> RestrictedSolution {
    let ri: Vec<usize> = (0..rows).filter(|&i| active_rows[i]).collect();
    let cj: Vec<usize> = (0..cols).filter(|&j| active_cols[j]).collect();
    assert!(!ri.is_empty() && !cj.is_empty());
    let reduced: Vec<f64> = ri
        .iter()
        .flat_map(|&i| cj.iter().map(move |&j| matrix[i * cols + j]))
        .collect();
    let (value, gap, rx, ry) = if ri.len() == 1 && cj.len() == 1 {
        (reduced[0], 0.0, vec![1.0], vec![1.0])
    } else if cj.len() == 1 {
        let best = (0..ri.len())
            .max_by(|&a, &b| reduced[a].partial_cmp(&reduced[b]).unwrap())
            .unwrap();
        let mut x = vec![0.0; ri.len()];
        x[best] = 1.0;
        (reduced[best], 0.0, x, vec![1.0])
    } else if ri.len() == 1 {
        let best = (0..cj.len())
            .min_by(|&a, &b| reduced[a].partial_cmp(&reduced[b]).unwrap())
            .unwrap();
        let mut y = vec![0.0; cj.len()];
        y[best] = 1.0;
        (reduced[best], 0.0, vec![1.0], y)
    } else {
        let solved = solve_matrix_full(&reduced, ri.len(), cj.len());
        (solved.value, solved.gap, solved.x, solved.y)
    };
    let mut x = vec![0.0; rows];
    let mut y = vec![0.0; cols];
    for (&i, p) in ri.iter().zip(rx) {
        x[i] = p;
    }
    for (&j, p) in cj.iter().zip(ry) {
        y[j] = p;
    }
    RestrictedSolution { value, gap, x, y }
}

/// Permanently remove only actions whose current cross-bounds prove pure
/// pointwise dominance for every still-live opposing action. Bounds tighten
/// monotonically, so a certificate never becomes invalid. One action is
/// removed per pass to avoid deleting mutually equivalent actions together.
fn prune_dominated_actions(
    lo: &[f64],
    hi: &[f64],
    rows: usize,
    cols: usize,
    active_rows: &mut [bool],
    active_cols: &mut [bool],
) -> (usize, usize, usize) {
    const EPS: f64 = 1e-12;
    let certified_le = |upper: f64, lower: f64| {
        upper + EPS <= lower || (upper == lower && (upper == 0.0 || upper == 1.0))
    };
    let mut removed_rows = 0;
    let mut removed_cols = 0;
    let mut checks = 0;
    loop {
        let mut changed = false;
        'row: for r in 0..rows {
            if !active_rows[r] || active_rows.iter().filter(|&&v| v).count() <= 1 {
                continue;
            }
            for s in 0..rows {
                if r == s || !active_rows[s] {
                    continue;
                }
                checks += 1;
                if (0..cols)
                    .filter(|&j| active_cols[j])
                    .all(|j| certified_le(hi[r * cols + j], lo[s * cols + j]))
                {
                    active_rows[r] = false;
                    removed_rows += 1;
                    changed = true;
                    break 'row;
                }
            }
        }
        if changed {
            continue;
        }
        'col: for c in 0..cols {
            if !active_cols[c] || active_cols.iter().filter(|&&v| v).count() <= 1 {
                continue;
            }
            for d in 0..cols {
                if c == d || !active_cols[d] {
                    continue;
                }
                checks += 1;
                if (0..rows)
                    .filter(|&i| active_rows[i])
                    .all(|i| certified_le(hi[i * cols + d], lo[i * cols + c]))
                {
                    active_cols[c] = false;
                    removed_cols += 1;
                    changed = true;
                    break 'col;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (removed_rows, removed_cols, checks)
}

fn outcome_value(b: &Battle) -> Option<f64> {
    b.outcome().map(|o| match o {
        Outcome::P1Win => 1.0,
        Outcome::Tie => 0.5,
        Outcome::P2Win => 0.0,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use nc2000_engine::battle::enumerate::enumerate_step;
    use nc2000_engine::battle::{Outcome, PokemonSet};
    use nc2000_engine::dex::MoveId;
    use nc2000_engine::state::{Attacker, Battle, PokeId};

    use super::{
        prune_dominated_actions, solve_restricted_game, BoundConfig, BoundSolver, Cell, Node,
        NodeKey, StageWitness,
    };
    use crate::exact::solve_matrix;

    fn mon(moves: &[&str]) -> PokemonSet {
        PokemonSet {
            name: "Pikachu".into(),
            species: "Pikachu".into(),
            item: String::new(),
            ability: String::new(),
            moves: moves.iter().map(|m| (*m).into()).collect(),
            level: 50,
            evs: None,
            ivs: None,
            happiness: None,
            gender: None,
        }
    }

    fn battle(dex: &nc2000_engine::dex::Dex, moves0: &[&str], moves1: &[&str]) -> Battle {
        Battle::from_fixture(dex, "1,2,3,4", &[mon(moves0)], &[mon(moves1)]).unwrap()
    }

    #[test]
    fn hp_cached_search_matches_exhaustive_values_and_counts_actual_runs() {
        let dex=conformance::load_dex();
        let mut base=battle(&dex,&["Seismic Toss"],&["Seismic Toss"]);
        let preview=[0,1].map(|side|base.legal_choices(&dex,side)[0]);
        base.apply_choices(&dex,preview.map(Some)).unwrap();
        base.turn=999;
        for (capacity,cell_cap) in [(0,0),(128,0),(128,1),(128,4096)] {
            let mut solver=BoundSolver::new(&dex,BoundConfig {
                hp_transition_capacity:capacity,hp_transition_eager:true,dead_quick_claw_quotient:true,
                cell_cap,eps:1e-10,work_budget:10_000,..BoundConfig::default()
            });
            for hp in [[100,100],[90,80],[60,90],[40,90]] {
                let mut root=base.clone();
                for side in 0..2 {let id=root.active_id(side).unwrap();root.poke_mut(id).hp=hp[side];}
                let exact=exhaustive_matrix(&dex,&root,2)[0];
                let report=solver.solve(&root,None);
                assert!(report.bounds.width()<1e-9,"{:?}",report);
                assert!((report.bounds.lo-exact).abs()<1e-9);
                assert!((report.bounds.hi-exact).abs()<1e-9);
            }
            if cell_cap==0 {assert_eq!(solver.stats.runs+solver.stats.hp_transition_hits,solver.stats.expansions);}
            if cell_cap==4096 {assert!(solver.stats.hp_transition_eager_hits>0);}
            assert_eq!(solver.stats.hp_transition_hits>0,capacity>0);
            assert!(solver.stats.hp_transition_peak<=capacity);
        }
    }

    #[test]
    fn quick_claw_quotient_guards_every_roster_and_keeps_key_domains_separate() {
        let dex = conformance::load_dex();
        let mut root = Battle::from_fixture(&dex,"1,2,3,4",
            &[mon(&["Thief"]),mon(&["Splash"])],&[mon(&["Splash"])]).unwrap();
        let solver = BoundSolver::new(&dex,BoundConfig { dead_quick_claw_quotient:true,..BoundConfig::default() });
        for damage in [false,true] {
            let solver = BoundSolver::new(&dex,BoundConfig { dead_quick_claw_quotient:true,
                dead_damage_quotient:damage,..BoundConfig::default() });
            let mut alternate = root.clone();
            alternate.quick_claw_roll = !root.quick_claw_roll;
            assert_eq!(solver.node_key(&root),solver.node_key(&alternate));
            assert_ne!(solver.node_key(&root),BoundSolver::new(&dex,BoundConfig {
                dead_damage_quotient:damage,..BoundConfig::default() }).node_key(&root));
        }
        for (side,slot,fainted) in [(0,0,false),(0,1,false),(0,1,true),(1,0,false)] {
            let mut holder = root.clone();
            holder.sides[side].roster[slot].item = dex.known_items.quickclaw;
            holder.sides[side].roster[slot].fainted = fainted;
            assert!(holder.quick_claw_roll_observable(&dex));
            let mut alternate = holder.clone();
            alternate.quick_claw_roll = !holder.quick_claw_roll;
            assert_ne!(solver.node_key(&holder),solver.node_key(&alternate));
        }
        root.sides[0].roster[0].move_slots = battle(&dex,&["Counter"],&["Splash"]).sides[0].roster[0].move_slots.clone();
        assert!(matches!(solver.node_key(&root),NodeKey::NoQuickClawRoll(_)));
    }

    #[test]
    fn quick_claw_quotient_preserves_complete_kernels_with_switches_and_thief() {
        let dex = conformance::load_dex();
        let mut root = Battle::from_fixture(&dex,"1,2,3,4",
            &[mon(&["Thief","Splash"]),mon(&["Quick Attack"])],
            &[mon(&["Quick Attack","Splash"])]).unwrap();
        let preview = [0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        root.apply_choices(&dex,preview).unwrap();
        let solver = BoundSolver::new(&dex,BoundConfig { dead_quick_claw_quotient:true,..BoundConfig::default() });
        let mut alternate = root.clone();
        alternate.quick_claw_roll = !root.quick_claw_roll;
        let legal = [0,1].map(|side|root.legal_choices(&dex,side));
        assert_eq!(legal,[0,1].map(|side|alternate.legal_choices(&dex,side)));
        for &a in &legal[0] { for &b in &legal[1] {
            let mut kernels = Vec::new();
            for state in [&root,&alternate] {
                let step = enumerate_step(&dex,state,[Some(a),Some(b)],100_000).unwrap();
                let mut kernel = HashMap::new();
                for leaf in step.leaves {
                    assert!(!leaf.battle.quick_claw_roll_observable(&dex));
                    let key = leaf.battle.outcome().map_or_else(||format!("{:?}",solver.node_key(&leaf.battle)),
                        |outcome|format!("terminal:{outcome:?}"));
                    *kernel.entry(key).or_insert(0.0) += leaf.prob;
                }
                assert!((kernel.values().sum::<f64>()-1.0).abs()<1e-10);
                kernels.push(kernel);
            }
            assert_eq!(kernels[0].len(),kernels[1].len());
            for (key,mass) in &kernels[0] { assert!((mass-kernels[1][key]).abs()<1e-10); }
        }}
    }

    #[test]
    fn fainted_quick_claw_is_ignored_only_at_idle_move_requests() {
        let dex = conformance::load_dex();
        let mut holder = mon(&["Splash"]);
        holder.item = "Quick Claw".into();
        let mut root = Battle::from_fixture(&dex,"1,2,3,4",
            &[mon(&["Thief"]),holder],&[mon(&["Splash"])]).unwrap();
        let preview = [0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        root.apply_choices(&dex,preview).unwrap();
        let slot = root.sides[0].party[1] as usize;
        root.sides[0].roster[slot].hp=0;
        root.sides[0].roster[slot].fainted=true;
        root.sides[0].pokemon_left=1;
        assert!(!root.quick_claw_roll_observable(&dex));
        for changed in 0..5 {
            let mut busy = root.clone();
            match changed {
                0 => busy.mid_turn=true,
                1 => busy.active_pokemon=busy.active_id(0),
                2 => busy.active_move=Some(nc2000_engine::battle::moveexec::get_active_move(&dex,dex.moves.id("thief").unwrap())),
                3 => busy.sides[0].roster[slot].hp=1,
                _ => busy.sides[0].roster[slot].fainted=false,
            }
            assert!(busy.quick_claw_roll_observable(&dex));
        }
        let solver = BoundSolver::new(&dex,BoundConfig { dead_quick_claw_quotient:true,..BoundConfig::default() });
        let choices=[0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        let mut kernels=Vec::new();
        for roll in [false,true] {
            root.quick_claw_roll=roll;
            let step=enumerate_step(&dex,&root,choices,100_000).unwrap();
            let mut kernel=HashMap::new();
            for leaf in step.leaves {
                assert!(leaf.battle.sides.iter().flat_map(|s|s.roster.iter())
                    .filter(|p|!p.fainted).all(|p|p.item!=dex.known_items.quickclaw));
                *kernel.entry(solver.node_key(&leaf.battle)).or_insert(0.0)+=leaf.prob;
            }
            kernels.push(kernel);
        }
        assert_eq!(kernels[0].len(),kernels[1].len());
        for (key,mass) in &kernels[0] { assert!((mass-kernels[1][key]).abs()<1e-10); }
    }

    #[test]
    fn unselected_quick_claw_cannot_enter_play_or_change_successor_kernels() {
        use nc2000_engine::battle::SearchChoice;
        let dex=conformance::load_dex();
        let mut holder=mon(&["Splash"]);holder.item="Quick Claw".into();
        let mut root=Battle::from_fixture(&dex,"1,2,3,4",
            &[mon(&["Thief","Splash"]),mon(&["Splash"]),mon(&["Splash"]),holder],
            &[mon(&["Seismic Toss"])]).unwrap();
        assert!(root.quick_claw_roll_observable(&dex));
        root.apply_choices(&dex,[Some(SearchChoice::Team([1,2,3])),Some(SearchChoice::Team([1,0,0]))]).unwrap();
        assert_eq!(root.sides[0].party.as_slice(),&[0,1,2]);
        assert!(!root.quick_claw_roll_observable(&dex));
        assert!(!root.sides[0].roster[3].fainted && root.sides[0].roster[3].hp>0);
        let mut unfinished=root.clone();unfinished.started=false;
        assert!(unfinished.quick_claw_roll_observable(&dex));
        unfinished=root.clone();unfinished.turn=0;
        assert!(unfinished.quick_claw_roll_observable(&dex));
        unfinished=root.clone();unfinished.request_state=nc2000_engine::state::RequestState::TeamPreview;
        assert!(unfinished.quick_claw_roll_observable(&dex));
        let solver=BoundSolver::new(&dex,BoundConfig { dead_quick_claw_quotient:true,..BoundConfig::default() });
        let active=root.active_id(0).unwrap();root.poke_mut(active).hp=1;
        let legal=[0,1].map(|side|root.legal_choices(&dex,side));
        assert!(!legal[0].contains(&SearchChoice::Switch(4)));
        for &a in &legal[0] { for &b in &legal[1] {
            let mut kernels=Vec::new();
            for roll in [false,true] {
                root.quick_claw_roll=roll;
                let step=enumerate_step(&dex,&root,[Some(a),Some(b)],100_000).unwrap();
                let mut kernel=HashMap::new();
                for leaf in step.leaves {
                    let mut next=leaf.battle;
                    assert!(!next.sides[0].party.contains(&3));
                    assert!(next.sides[0].party.iter().all(|&slot|next.sides[0].roster[slot as usize].item!=dex.known_items.quickclaw));
                    assert!(!next.legal_choices(&dex,0).contains(&SearchChoice::Switch(4)));
                    let key=next.outcome().map_or_else(||format!("{:?}",solver.node_key(&next)),|outcome|format!("terminal:{outcome:?}"));
                    *kernel.entry(key).or_insert(0.0)+=leaf.prob;
                }
                kernels.push(kernel);
            }
            assert_eq!(kernels[0].len(),kernels[1].len());
            for (key,mass) in &kernels[0] { assert!((mass-kernels[1][key]).abs()<1e-10); }
        }}
    }

    #[test]
    fn held_quick_claw_changes_outcomes_and_cannot_be_quotiented() {
        let dex = conformance::load_dex();
        let mut holder = mon(&["Seismic Toss"]);
        holder.item = "Quick Claw".into();
        let mut root = Battle::from_fixture(&dex,"1,2,3,4",&[holder],&[mon(&["Seismic Toss"])]).unwrap();
        let preview = [0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        root.apply_choices(&dex,preview).unwrap();
        for side in 0..2 { let id=root.active_id(side).unwrap();root.poke_mut(id).hp=1; }
        root.turn=1000;
        let choices=[0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        let mut values=Vec::new();
        for roll in [false,true] {
            root.quick_claw_roll=roll;
            let step=enumerate_step(&dex,&root,choices,100_000).unwrap();
            values.push(step.leaves.iter().map(|leaf|leaf.prob*super::outcome_value(&leaf.battle).unwrap()).sum::<f64>());
        }
        assert!((values[0]-0.5).abs()<1e-10);
        assert!((values[1]-1.0).abs()<1e-10);
    }

    fn perturb_damage_history(b: &mut Battle, move_id: MoveId) {
        b.last_damage = 991;
        let p = &mut b.sides[0].roster[0];
        p.hurt_this_turn = Some(17);
        p.last_damage = 83;
        p.times_attacked = 4;
        p.attacked_by.push(Attacker {
            source: PokeId { side: 1, slot: 0 },
            damage: 37,
            move_id,
            this_turn: true,
            damage_value: Some(37),
        });
    }

    #[test]
    fn semantic_key_omits_only_damage_history_payload() {
        let dex = conformance::load_dex();
        let a = battle(&dex, &["Tackle"], &["Splash"]);
        let mut b = a.clone();
        perturb_damage_history(&mut b, dex.moves.id("tackle").unwrap());

        assert_ne!(a.state_key128(), b.state_key128());
        assert_eq!(
            a.state_key128_without_damage_bookkeeping(),
            b.state_key128_without_damage_bookkeeping()
        );
        b.sides[0].roster[0].hp -= 1;
        assert_ne!(
            a.state_key128_without_damage_bookkeeping(),
            b.state_key128_without_damage_bookkeeping(),
            "decision-relevant HP must remain exact"
        );
    }

    #[test]
    fn observer_guard_covers_direct_and_generated_counter_moves() {
        let dex = conformance::load_dex();
        for move_name in ["Counter", "Mirror Coat"] {
            let b = battle(&dex, &[move_name], &["Splash"]);
            assert!(b.damage_bookkeeping_observable(&dex), "{move_name}");
        }
        for move_name in [
            "Splash",
            "Mimic",
            "Transform",
            "Mirror Move",
            "Metronome",
            "Bide",
        ] {
            let b = battle(&dex, &[move_name], &["Splash"]);
            assert!(!b.damage_bookkeeping_observable(&dex), "{move_name}");
        }

        // Acquired moves and original slots are both relevant: Mimic and
        // Transform can rewrite current slots while the old base set remains.
        let safe = battle(&dex, &["Splash"], &["Splash"]);
        let counter = battle(&dex, &["Counter"], &["Splash"]);
        let mut acquired = safe.clone();
        acquired.sides[0].roster[0].move_slots = counter.sides[0].roster[0].move_slots.clone();
        assert!(acquired.damage_bookkeeping_observable(&dex));
        let mut replaced = counter;
        replaced.sides[0].roster[0].move_slots = safe.sides[0].roster[0].move_slots.clone();
        assert!(replaced.damage_bookkeeping_observable(&dex));
    }

    #[test]
    fn tagged_solver_key_merges_safe_states_but_not_observer_states() {
        let dex = conformance::load_dex();
        let mut safe_a = battle(&dex, &["Tackle"], &["Splash"]);
        let mut safe_b = safe_a.clone();
        perturb_damage_history(&mut safe_b, dex.moves.id("tackle").unwrap());
        let mut solver = BoundSolver::new(&dex, BoundConfig::default());
        assert_eq!(solver.intern(safe_a.clone()), solver.intern(safe_b.clone()));
        assert_eq!(solver.node_count(), 1);

        // The exact and semantic domains are tagged, even if a future hash
        // implementation happened to produce the same payload.
        safe_a.sides[0].roster[0].move_slots = battle(&dex, &["Counter"], &["Splash"]).sides[0]
            .roster[0]
            .move_slots
            .clone();
        safe_b.sides[0].roster[0].move_slots = safe_a.sides[0].roster[0].move_slots.clone();
        assert_ne!(solver.intern(safe_a), solver.intern(safe_b));
        assert_eq!(solver.node_count(), 3);
    }

    #[test]
    fn safe_states_have_identical_semantic_successor_distributions() {
        let dex = conformance::load_dex();
        let mut a = battle(
            &dex,
            &["Quick Attack", "Splash"],
            &["Quick Attack", "Splash"],
        );
        let mut b = a.clone();

        // Advance both through preview, then compare a real stochastic move
        // step under the semantic quotient.
        for state in [&mut a, &mut b] {
            let c0 = state.legal_choices(&dex, 0)[0];
            let c1 = state.legal_choices(&dex, 1)[0];
            state.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        }
        for state in [&mut a, &mut b] {
            state.turn = 1000;
            for side in 0..2 {
                let id = state.active_id(side).unwrap();
                state.poke_mut(id).hp = 1;
            }
        }
        perturb_damage_history(&mut b, dex.moves.id("quickattack").unwrap());
        assert_ne!(a.state_key128(), b.state_key128());
        assert_eq!(
            a.state_key128_without_damage_bookkeeping(),
            b.state_key128_without_damage_bookkeeping()
        );
        let aa = [a.legal_choices(&dex, 0), a.legal_choices(&dex, 1)];
        let ab = [b.legal_choices(&dex, 0), b.legal_choices(&dex, 1)];
        assert_eq!(aa, ab);
        let aggregate = |step: &nc2000_engine::battle::enumerate::StepEnum| {
            let mut out: HashMap<u128, f64> = HashMap::new();
            for leaf in &step.leaves {
                *out.entry(leaf.battle.state_key128_without_damage_bookkeeping())
                    .or_default() += leaf.prob;
            }
            out
        };
        let payoff = |step: &nc2000_engine::battle::enumerate::StepEnum| {
            step.leaves
                .iter()
                .map(|leaf| {
                    let v = match leaf
                        .battle
                        .outcome()
                        .expect("turn-1000 successor is terminal")
                    {
                        Outcome::P1Win => 1.0,
                        Outcome::Tie => 0.5,
                        Outcome::P2Win => 0.0,
                    };
                    leaf.prob * v
                })
                .sum::<f64>()
        };
        let mut ma = Vec::new();
        let mut mb = Vec::new();
        for &c0 in &aa[0] {
            for &c1 in &aa[1] {
                let sa = enumerate_step(&dex, &a, [Some(c0), Some(c1)], 100_000).unwrap();
                let sb = enumerate_step(&dex, &b, [Some(c0), Some(c1)], 100_000).unwrap();
                assert_eq!(aggregate(&sa), aggregate(&sb));
                ma.push(payoff(&sa));
                mb.push(payoff(&sb));
            }
        }
        assert_eq!(ma, mb);
        let (va, ga) = solve_matrix(&ma, aa[0].len(), aa[1].len());
        let (vb, gb) = solve_matrix(&mb, ab[0].len(), ab[1].len());
        assert!((va - vb).abs() < 1e-12);
        assert!(ga < 1e-12 && gb < 1e-12);
    }

    #[test]
    fn reroot_retains_solved_continuations_and_discards_ancestors() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Splash"], &["Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 999;
        let actions = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        let (child, _) = nc2000_engine::battle::enumerate::run_scripted(&dex, &root, actions, &[]);
        let cfg = BoundConfig { eps: 0.0, ..BoundConfig::default() };
        let mut solver = BoundSolver::new(&dex, cfg.clone());
        assert_eq!(solver.solve(&root, None).bounds.lo, 0.5);
        let before = solver.node_count();
        assert!(solver.peek(&child).is_some());
        solver.retain_root(&child);
        assert!(solver.node_count() < before);
        assert!(solver.peek(&root).is_none());
        let warm = solver.solve(&child, None);
        let cold = BoundSolver::new(&dex, cfg).solve(&child, None);
        assert_eq!((warm.bounds.lo, warm.bounds.hi), (cold.bounds.lo, cold.bounds.hi));
        assert_eq!(warm.runs, 0);
        assert!(cold.runs > 0);
    }

    #[test]
    fn distance_to_turn_limit_changes_certified_value() {
        let dex = conformance::load_dex();
        let mut early = battle(&dex, &["Seismic Toss"], &["Splash"]);
        let preview = [0, 1].map(|s| early.legal_choices(&dex, s).first().copied());
        early.apply_choices(&dex, preview).unwrap();
        let foe = early.active_id(1).unwrap();
        early.poke_mut(foe).hp = 75;
        early.turn = 999;
        let mut late = early.clone();
        late.turn = 1000;
        assert_ne!(early.state_key128_without_damage_bookkeeping(), late.state_key128_without_damage_bookkeeping());
        let cfg = BoundConfig { eps: 0.0, ..BoundConfig::default() };
        let early_value = BoundSolver::new(&dex, cfg.clone()).solve(&early, None).bounds;
        let late_value = BoundSolver::new(&dex, cfg).solve(&late, None).bounds;
        assert!(early_value.lo > late_value.hi + 0.4);
    }

    #[test]
    fn elapsed_budget_stops_without_discarding_the_interval() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Splash"], &["Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        let mut solver = BoundSolver::new(&dex, BoundConfig {
            time_budget: Some(std::time::Duration::ZERO),
            ..BoundConfig::default()
        });
        let partial = solver.solve(&root, None);
        assert_eq!(partial.stop, super::Stop::TimeBudget);
        assert_eq!(partial.runs, 0);
        assert_eq!((partial.bounds.lo, partial.bounds.hi), (0.0, 1.0));
        solver.cfg.time_budget = None;
        let done = solver.solve(&root, None);
        assert_eq!(done.bounds.lo, 0.5);
        assert_eq!(done.bounds.hi, 0.5);
    }

    #[test]
    fn mixed_policy_loss_uses_one_reply_to_the_distribution() {
        let matrix = super::RootPayoffs {
            actions: [vec![None; 2], vec![None; 2]],
            cells: [1.0, 0.0, 0.0, 1.0].map(|v| super::Bounds { lo: v, hi: v }).to_vec(),
            value: super::Bounds { lo: 0.5, hi: 0.5 },
        };
        for side in 0..2 {
            let mixed = matrix.policy_loss(side, &[0.5, 0.5]);
            assert!(mixed.hi < 1e-10);
            let pure = matrix.policy_loss(side, &[1.0, 0.0]);
            assert!((pure.lo - 0.5).abs() < 1e-10);
            assert!((pure.hi - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn policy_target_skips_sufficient_replies_without_changing_certificates() {
        let cells = [(0.5, 0.7), (0.65, 1.0), (0.0, 0.4), (0.0, 0.4)];
        for side in 0..2 {
            let matrix = super::RootPayoffs {
                actions: [vec![None; 2], vec![None; 2]],
                cells: (0..4).map(|i| {
                    let (lo, hi) = if side == 0 { cells[i] } else { cells[(i % 2) * 2 + i / 2] };
                    if side == 0 { super::Bounds { lo, hi } }
                    else { super::Bounds { lo: 1.0 - hi, hi: 1.0 - lo } }
                }).collect(),
                value: if side == 0 { super::Bounds { lo: 0.5, hi: 0.7 } }
                    else { super::Bounds { lo: 0.3, hi: 0.5 } },
            };
            let before = matrix.policy_loss(side, &[1.0, 0.0]);
            let (original, skipped) = matrix.refinement_cell(side, &[1.0, 0.0], None);
            assert_eq!(original.unwrap().0, if side == 0 { 1 } else { 2 });
            assert_eq!(skipped, 0);
            let (targeted, skipped) = matrix.refinement_cell(side, &[1.0, 0.0], Some(0.1));
            assert_eq!(targeted.unwrap().0, 0);
            assert_eq!(skipped, 1);
            let (tightened, skipped) = matrix.refinement_cell(side, &[1.0, 0.0], Some(0.01));
            assert_eq!(tightened.unwrap().0, original.unwrap().0);
            assert_eq!(skipped, 0);
            let after = matrix.policy_loss(side, &[1.0, 0.0]);
            assert_eq!(before.lo, after.lo);
            assert_eq!(before.hi, after.hi);
            assert!(after.hi > 0.1);
        }
    }

    #[test]
    fn forced_action_has_zero_loss_even_when_the_value_is_unknown() {
        let matrix = super::RootPayoffs {
            actions: [vec![None], vec![None; 2]],
            cells: vec![super::Bounds { lo: 0.0, hi: 1.0 }; 2],
            value: super::Bounds { lo: 0.0, hi: 1.0 },
        };
        assert_eq!(matrix.policy_loss(0, &[1.0]).hi, 0.0);
        assert_eq!(matrix.policy_loss(1, &[0.5, 0.5]).hi, 1.0);
    }

    #[test]
    fn policy_refinement_continues_after_the_value_closes() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Splash"], &["Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        let foe = root.active_id(1).unwrap();
        root.poke_mut(foe).hp = 1;
        let mut solver = BoundSolver::new(&dex, BoundConfig { eps: 0.0, ..BoundConfig::default() });
        let value = solver.solve(&root, None);
        assert!(value.bounds.lo > 0.99);
        let targets = vec![(0, vec![1.0, 0.0]), (0, vec![0.0, 1.0]), (0, vec![0.5, 0.5])];
        let report = solver.solve_policy_losses(&root, &targets, 0.0001);
        assert_eq!(report.stop, super::Stop::StrategyMet);
        let matrix = solver.root_payoffs(&root).unwrap();
        assert!(matrix.policy_loss(0, &[1.0, 0.0]).hi < 1e-10);
        assert!((matrix.policy_loss(0, &[0.0, 1.0]).lo - 0.5).abs() < 1e-10);
        assert!((matrix.policy_loss(0, &[0.5, 0.5]).lo - 0.25).abs() < 1e-10);
    }

    #[test]
    fn policy_refinement_partial_bounds_enclose_exhaustive_microgame() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Quick Attack", "Splash"], &["Quick Attack", "Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let acts = [0, 1].map(|s| root.legal_choices(&dex, s));
        let mut values = Vec::new();
        for &a in &acts[0] { for &b in &acts[1] {
            let step = enumerate_step(&dex, &root, [Some(a), Some(b)], 100_000).unwrap();
            values.push(step.leaves.iter().map(|l| l.prob * super::outcome_value(&l.battle).unwrap()).sum::<f64>());
        }}
        let v = solve_matrix(&values, 2, 2).0;
        let exact = super::RootPayoffs {
            actions: acts.map(|aa| aa.into_iter().map(Some).collect()),
            cells: values.into_iter().map(|v| super::Bounds { lo: v, hi: v }).collect(),
            value: super::Bounds { lo: v, hi: v },
        };
        let targets: Vec<_> = (0..2).flat_map(|side| {
            [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.3, 0.7]].map(|p| (side, p))
        }).collect();
        for (retain_eager_prefix,defer_eager_frontier) in [(false,false),(true,false),(true,true)] {
            let mut solver = BoundSolver::new(&dex, BoundConfig {
                work_budget: 1, cell_cap: usize::from(retain_eager_prefix), retain_eager_prefix, defer_eager_frontier,
                ..BoundConfig::default()
            });
            let mut closed = false;
            for _ in 0..1000 {
                let report = solver.solve_policy_losses(&root, &targets, 1e-7);
                let matrix = solver.root_payoffs(&root).unwrap();
                for (side, policy) in &targets {
                    let found = matrix.policy_loss(*side, policy);
                    let expected = exact.policy_loss(*side, policy);
                    assert!(found.lo <= expected.hi + 1e-10);
                    assert!(found.hi >= expected.lo - 1e-10);
                }
                if report.stop == super::Stop::StrategyMet { closed = true; break; }
            }
            assert!(closed,"policy certificates did not converge");
            if retain_eager_prefix { assert!(solver.stats.retained_eager_runs>0); }
        }
    }

    #[test]
    fn fixed_policy_bounds_keep_the_opponent_free_and_orient_both_sides() {
        use nc2000_engine::battle::SearchChoice;
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Splash"], &["Seismic Toss", "Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let cfg = BoundConfig { eps: 1e-8, ..BoundConfig::default() };
        let mut unrestricted = BoundSolver::new(&dex, cfg.clone());
        let value = unrestricted.solve(&root, None).bounds;
        assert!((value.lo - 0.5).abs() < 1e-8 && (value.hi - 0.5).abs() < 1e-8);
        for side in 0..2 {
            for (name, expected) in [("splash", 0.0), ("seismictoss", 0.5)] {
                let priority = vec![SearchChoice::Move(dex.moves.id(name).unwrap())];
                let mut fixed = super::FixedPolicySolver::new(&dex, cfg.clone(), side, priority);
                let report = fixed.solve(&root);
                assert!((report.guaranteed_score.lo - expected).abs() < 1e-8);
                assert!((report.guaranteed_score.hi - expected).abs() < 1e-8);
                assert!(report.unrestricted_game_value.lo <= value.lo + 1e-8);
                assert!(report.unrestricted_game_value.hi >= value.hi - 1e-8);
                if side == 0 { assert_eq!(report.unrestricted_game_value.hi, 1.0); }
                else { assert_eq!(report.unrestricted_game_value.lo, 0.0); }
            }
        }
    }

    #[test]
    fn imported_policy_certificates_bound_values_without_certifying_actions() {
        use nc2000_engine::battle::SearchChoice;
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Splash"], &["Seismic Toss", "Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let cfg = BoundConfig { eps: 1e-8, ..BoundConfig::default() };
        let mut target = BoundSolver::new(&dex, cfg.clone());
        for side in 0..2 {
            for name in ["splash", "seismictoss"] {
                let mut witness = super::FixedPolicySolver::new(&dex, cfg.clone(), side,
                    vec![SearchChoice::Move(dex.moves.id(name).unwrap())]);
                witness.solve(&root);
                let certificates = witness.into_value_certificates();
                if name == "splash" { assert!(certificates.is_empty()); }
                else { assert_eq!(certificates.len(), 1); }
                target.import_value_certificates(certificates);
            }
        }
        let cache = target.into_cache();
        assert!(!cache.is_empty());
        let mut target = BoundSolver::from_cache(&dex, cfg, cache);
        let solved = target.solve(&root, None);
        assert_eq!(solved.runs, 0);
        assert_eq!(target.stats.certificate_hits, 1);
        assert!((solved.bounds.lo - 0.5).abs() < 1e-8);
        assert!((solved.bounds.hi - 0.5).abs() < 1e-8);
        assert!(solved.strategies.is_none());
        let report = target.solve_policy_losses(&root, &[(0, vec![0.0, 1.0])], 1e-8);
        assert!(report.runs > 0);
        let loss = target.root_payoffs(&root).unwrap().policy_loss(0, &[0.0, 1.0]);
        assert!((loss.lo - 0.5).abs() < 1e-8 && (loss.hi - 0.5).abs() < 1e-8);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn value_certificate_import_rejects_a_different_dex() {
        let dex = conformance::load_dex();
        let witness = super::FixedPolicySolver::new(&dex, BoundConfig::default(), 0, vec![]);
        let mut certificates = witness.into_value_certificates();
        certificates.dex_fingerprint[0] ^= 1;
        BoundSolver::new(&dex, BoundConfig::default()).import_value_certificates(certificates);
    }

    #[test]
    fn imported_descendant_bounds_preserve_a_two_turn_game() {
        use nc2000_engine::battle::SearchChoice;
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Protect"], &["Seismic Toss", "Protect"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 999;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = if side == 0 { 40 } else { 60 };
        }
        let cfg = BoundConfig { eps: 1e-8, ..BoundConfig::default() };
        let exact = BoundSolver::new(&dex, cfg.clone()).solve(&root, None).bounds;
        let mut witness = super::FixedPolicySolver::new(&dex, cfg.clone(), 1,
            vec![SearchChoice::Move(dex.moves.id("seismictoss").unwrap())]);
        witness.solve(&root);
        let certificates = witness.into_value_certificates();
        assert!(certificates.len() > 1);
        let mut target = BoundSolver::new(&dex, cfg);
        target.import_value_certificates(certificates);
        let result = target.solve(&root, None).bounds;
        assert!(target.stats.certificate_hits > 1);
        assert!((result.lo - exact.lo).abs() < 1e-8);
        assert!((result.hi - exact.hi).abs() < 1e-8);
    }

    #[test]
    fn fixed_policy_applies_after_the_root_and_falls_back_when_unavailable() {
        use nc2000_engine::battle::SearchChoice;
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Splash"], &["Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 999;
        let foe = root.active_id(1).unwrap();
        root.poke_mut(foe).hp = 25;
        let cfg = BoundConfig { eps: 1e-8, ..BoundConfig::default() };
        for (priority, expected) in [
            (vec![SearchChoice::Move(dex.moves.id("splash").unwrap())], 0.5),
            (vec![SearchChoice::Move(dex.moves.id("seismictoss").unwrap())], 1.0),
            (vec![SearchChoice::Move(dex.moves.id("thunder").unwrap())], 1.0),
        ] {
            let mut solver = super::FixedPolicySolver::new(&dex, cfg.clone(), 0, priority);
            let result = solver.solve(&root);
            assert!((result.guaranteed_score.lo - expected).abs() < 1e-8);
            assert!((result.guaranteed_score.hi - expected).abs() < 1e-8);
        }
    }

    #[test]
    fn unrestricted_descendant_certificates_preserve_exhaustive_values_and_actions() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Protect"], &["Seismic Toss", "Protect"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 999;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = if side == 0 { 40 } else { 60 };
        }
        let exact = exhaustive_matrix(&dex, &root, 2);
        let value = solve_matrix(&exact, 2, 2).0;
        let actions = [0, 1].map(|side| root.legal_choices(&dex, side));
        let children = enumerate_step(&dex, &root, [Some(actions[0][1]), Some(actions[1][1])], 100_000).unwrap();
        let cfg = BoundConfig { eps: 1e-10, ..BoundConfig::default() };
        for work_budget in [1, 1_000_000] {
            let mut donor = BoundSolver::new(&dex, BoundConfig { work_budget, ..cfg.clone() });
            for leaf in &children.leaves { donor.solve(&leaf.battle, None); }
            let certificates = donor.into_value_certificates();
            if work_budget > 1 { assert!(!certificates.is_empty()); }
            let mut target = BoundSolver::new(&dex, cfg.clone());
            target.import_value_certificates(certificates);
            let report = target.solve_policy_losses(&root,
                &[(0, vec![1.0, 0.0]), (1, vec![0.0, 1.0])], 1e-10);
            assert!(report.bounds.lo <= value + 1e-10 && report.bounds.hi >= value - 1e-10);
            let matrix = target.root_payoffs(&root).unwrap();
            for (bound, exact) in matrix.cells.iter().zip(&exact) {
                assert!(bound.lo <= exact + 1e-10 && bound.hi >= exact - 1e-10);
            }
            if work_budget > 1 { assert!(target.stats.certificate_hits > 0); }
            let solved = target.solve(&root, None);
            let mut imported = BoundSolver::new(&dex, cfg.clone());
            imported.import_value_certificates(target.into_value_certificates());
            let result = imported.solve(&root, None);
            assert_eq!(result.runs, 0);
            assert_eq!((result.bounds.lo, result.bounds.hi), (solved.bounds.lo, solved.bounds.hi));
            assert!(result.strategies.is_none());
        }
    }

    #[test]
    fn fixed_first_action_does_not_replace_continuation_priority() {
        use nc2000_engine::battle::SearchChoice;
        let dex = conformance::load_dex();
        let splash = SearchChoice::Move(dex.moves.id("splash").unwrap());
        let attack = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        for side in 0..2 {
            for (turn, expected) in [(999, 1.0), (1000, 0.5)] {
                let mut root = if side == 0 {
                    battle(&dex, &["Splash", "Seismic Toss"], &["Splash"])
                } else { battle(&dex, &["Splash"], &["Splash", "Seismic Toss"]) };
                let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
                root.apply_choices(&dex, preview).unwrap();
                root.turn = turn;
                let foe = root.active_id(1 - side).unwrap();
                root.poke_mut(foe).hp = 25;
                let mut solver = super::FixedPolicySolver::new_with_first_action(&dex,
                    BoundConfig { eps: 1e-10, ..BoundConfig::default() }, side,
                    vec![attack], &root, splash);
                let report = solver.solve(&root);
                assert!((report.guaranteed_score.lo - expected).abs() < 1e-10);
                assert!((report.guaranteed_score.hi - expected).abs() < 1e-10);
                assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[side], vec![Some(splash)]);
            }
        }
    }

    #[test]
    fn frozen_strategy_switches_only_to_legal_nonfrozen_benches() {
        use nc2000_engine::battle::{EffectHandle, SearchChoice};
        let dex = conformance::load_dex();
        let attack = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let splash = SearchChoice::Move(dex.moves.id("splash").unwrap());
        let switch = SearchChoice::Switch(2);
        for side in 0..2 {
            for (frozen, bench_frozen, bench_fainted, first, selected) in [
                (false, false, false, None, attack), (true, false, false, None, switch),
                (true, true, false, None, attack), (true, false, true, None, attack),
                (true, false, false, Some(splash), splash),
            ] {
                let team = [mon(&["Seismic Toss", "Splash"]), mon(&["Splash"])];
                let foe = [mon(&["Splash"])];
                let (a,b) = if side == 0 { (&team[..], &foe[..]) } else { (&foe[..], &team[..]) };
                let mut root = Battle::from_fixture(&dex, "1,2,3,4", a, b).unwrap();
                let preview = [0,1].map(|s| root.legal_choices(&dex,s).first().copied());
                root.apply_choices(&dex, preview).unwrap();
                root.turn = 1000;
                let me = root.active_id(side).unwrap();
                let target = root.active_id(1-side).unwrap();
                root.poke_mut(target).hp = 1;
                if frozen { root.set_status(&dex, me, "frz", Some(me), EffectHandle::None, true); }
                let bench = root.sides[side].party[1] as usize;
                let bench_id = PokeId { side: side as u8, slot: bench as u8 };
                if bench_frozen { root.set_status(&dex, bench_id, "frz", Some(bench_id), EffectHandle::None, true); }
                if bench_fainted {
                    root.sides[side].roster[bench].hp = 0;
                    root.sides[side].roster[bench].fainted = true;
                    root.sides[side].pokemon_left = 1;
                }
                let mut choices = [Some(splash);2];
                choices[side] = Some(selected);
                let step = enumerate_step(&dex, &root, choices, 100_000).unwrap();
                let expected: f64 = step.leaves.iter().map(|leaf| {
                    let v = super::outcome_value(&leaf.battle).unwrap();
                    leaf.prob * if side == 0 { v } else { 1.0-v }
                }).sum();
                let cfg = BoundConfig { eps: 1e-10, ..BoundConfig::default() };
                let mut fixed = match first {
                    Some(action) => super::FixedPolicySolver::new_with_first_action(&dex,
                        cfg, side, vec![attack], &root, action),
                    None => super::FixedPolicySolver::new(&dex, cfg, side, vec![attack]),
                }.with_frozen_switch();
                let report = fixed.solve(&root);
                assert!((report.guaranteed_score.lo-expected).abs() < 1e-10);
                assert!((report.guaranteed_score.hi-expected).abs() < 1e-10);
                assert_eq!(fixed.restricted_root_payoffs(&root).unwrap().actions[side], vec![Some(selected)]);
            }
        }
    }

    #[test]
    fn frozen_switch_continuation_uses_moves_when_switches_are_not_prioritized() {
        use nc2000_engine::battle::{EffectHandle, SearchChoice};
        let dex = conformance::load_dex();
        let attack = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let surf = SearchChoice::Move(dex.moves.id("surf").unwrap());
        let splash = SearchChoice::Move(dex.moves.id("splash").unwrap());
        for keep_switch in [false,true] {
            let mut root = Battle::from_fixture(&dex, "1,2,3,4",
                &[mon(&["Seismic Toss"]),mon(&["Surf"])], &[mon(&["Splash"])]).unwrap();
            let preview = [0,1].map(|s|root.legal_choices(&dex,s).first().copied());
            root.apply_choices(&dex,preview).unwrap();
            let active = root.active_id(0).unwrap();
            root.set_status(&dex,active,"frz",Some(active),EffectHandle::None,true);
            let mut priority = vec![attack];
            if keep_switch { priority.push(SearchChoice::Switch(2)); }
            let mut fixed = super::FixedPolicySolver::new(&dex,
                BoundConfig { work_budget: 0, ..BoundConfig::default() },0,priority).with_frozen_switch();
            fixed.solve(&root);
            let key = fixed.solver.node_key(&root);
            fixed.solver.init_cells(key);
            assert_eq!(fixed.restricted_root_payoffs(&root).unwrap().actions[0],vec![Some(SearchChoice::Switch(2))]);
            root.apply_choices(&dex,[Some(SearchChoice::Switch(2)),Some(splash)]).unwrap();
            fixed.solve(&root);
            let key = fixed.solver.node_key(&root);
            fixed.solver.init_cells(key);
            let expected = if keep_switch { SearchChoice::Switch(2) } else { surf };
            assert_eq!(fixed.restricted_root_payoffs(&root).unwrap().actions[0],vec![Some(expected)]);
        }
    }

    #[test]
    fn a_frozen_bench_blocks_ice_beam_freeze_on_the_switch_in() {
        use nc2000_engine::battle::{EffectHandle, SearchChoice};
        use nc2000_engine::state::Status;
        let dex = conformance::load_dex();
        for frozen in [false,true] {
            let mut root = Battle::from_fixture(&dex, "1,2,3,4",
                &[mon(&["Splash"]),mon(&["Splash"])], &[mon(&["Ice Beam"])]).unwrap();
            let preview = [0,1].map(|s| root.legal_choices(&dex,s).first().copied());
            root.apply_choices(&dex,preview).unwrap();
            let active = root.active_id(0).unwrap();
            if frozen { root.set_status(&dex, active, "frz", Some(active), EffectHandle::None, true); }
            let step = enumerate_step(&dex, &root, [Some(SearchChoice::Switch(2)),
                Some(SearchChoice::Move(dex.moves.id("icebeam").unwrap()))],100_000).unwrap();
            let mass: f64 = step.leaves.iter().filter(|leaf|
                leaf.battle.poke(leaf.battle.active_id(0).unwrap()).status == Status::Frz)
                .map(|leaf|leaf.prob).sum();
            if frozen { assert_eq!(mass,0.0); } else { assert!(mass > 0.05); }
        }
    }

    #[test]
    fn sleeping_fixed_strategy_matches_exhaustive_turn_and_respects_root_override() {
        use nc2000_engine::battle::SearchChoice;
        use nc2000_engine::battle::EffectHandle;
        let dex = conformance::load_dex();
        let attack = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let talk = SearchChoice::Move(dex.moves.id("sleeptalk").unwrap());
        let splash = SearchChoice::Move(dex.moves.id("splash").unwrap());
        for side in 0..2 {
            for (sleeping, pp, first, selected) in [
                (false, 10, None, attack), (true, 10, None, talk),
                (true, 0, None, attack), (true, 10, Some(splash), splash),
            ] {
                let moves = &["Seismic Toss", "Sleep Talk", "Splash"];
                let mut root = if side == 0 { battle(&dex, moves, &["Splash"]) }
                    else { battle(&dex, &["Splash"], moves) };
                let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
                root.apply_choices(&dex, preview).unwrap();
                root.turn = 1000;
                let me = root.active_id(side).unwrap();
                let foe = root.active_id(1 - side).unwrap();
                root.poke_mut(foe).hp = 25;
                if sleeping { root.set_status(&dex, me, "slp", Some(me), EffectHandle::None, true); }
                root.poke_mut(me).move_slots[1].pp = pp;
                root.poke_mut(me).base_move_slots[1].pp = pp;
                let mut choices = [Some(splash); 2];
                choices[side] = Some(selected);
                let step = enumerate_step(&dex, &root, choices, 100_000).unwrap();
                let expected: f64 = step.leaves.iter().map(|leaf| {
                    let value = super::outcome_value(&leaf.battle).unwrap();
                    leaf.prob * if side == 0 { value } else { 1.0 - value }
                }).sum();
                let cfg = BoundConfig { eps: 1e-10, ..BoundConfig::default() };
                let mut solver = match first {
                    Some(action) => super::FixedPolicySolver::new_with_first_action(&dex,
                        cfg, side, vec![attack], &root, action),
                    None => super::FixedPolicySolver::new(&dex, cfg, side, vec![attack]),
                }.with_sleep_talk();
                let report = solver.solve(&root);
                assert!((report.guaranteed_score.lo - expected).abs() < 1e-10);
                assert!((report.guaranteed_score.hi - expected).abs() < 1e-10);
                assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[side], vec![Some(selected)]);
            }
        }
    }

    #[test]
    #[should_panic(expected = "fixed strategy must be configured before search")]
    fn fixed_strategy_cannot_change_after_search() {
        let dex = conformance::load_dex();
        let root = battle(&dex, &["Splash"], &["Splash"]);
        let mut solver = super::FixedPolicySolver::new(&dex,
            BoundConfig { work_budget: 0, ..BoundConfig::default() }, 0, vec![]);
        solver.solve(&root);
        solver.with_sleep_talk();
    }

    #[test]
    fn fixed_reply_bounds_keep_future_replies_free_and_resume() {
        use nc2000_engine::battle::SearchChoice;
        fn exact(dex: &nc2000_engine::dex::Dex, b: &Battle, side: usize, first: Option<SearchChoice>) -> f64 {
            if let Some(value) = super::outcome_value(b) { return value; }
            assert!(b.turn >= 999);
            let attack = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
            let mut probe = b.clone();
            let replies = first.map_or_else(||probe.legal_choices(dex,1-side),|reply|vec![reply]);
            let scores = replies.into_iter().map(|reply| {
                let actions = if side==0 { [attack,reply] } else { [reply,attack] };
                enumerate_step(dex,b,actions.map(Some),100_000).unwrap().leaves.iter()
                    .map(|leaf|leaf.prob*exact(dex,&leaf.battle,side,None)).sum::<f64>()
            });
            if side==0 { scores.fold(1.0,f64::min) } else { scores.fold(0.0,f64::max) }
        }
        let dex = conformance::load_dex();
        let mut root = battle(&dex,&["Seismic Toss","Protect"],&["Seismic Toss","Protect"]);
        let preview = [0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        root.apply_choices(&dex,preview).unwrap();
        root.turn=999;
        for side in 0..2 { let id=root.active_id(side).unwrap();root.poke_mut(id).hp=if side==0 {40} else {60}; }
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let mut bypasses=0;
        for (side,material,period) in [(0,false,0),(1,false,0),(0,true,0),(1,true,0),(0,true,2),(1,true,2),(0,true,4),(1,true,4)] { for reply in root.legal_choices(&dex,1-side) {
            let expected=exact(&dex,&root,side,Some(reply));
            let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig {
                work_budget:1,cell_cap:0,eps:1e-8,chance_material_order:material,material_descent_period:period,
                chance_fair_period:2,..BoundConfig::default()
            },side,vec![attack]);
            let mut closed=false;
            let mut previous=super::Bounds {lo:0.0,hi:1.0};
            for _ in 0..1000 {
                let report=solver.solve_reply(&root,reply,None);
                assert!(report.runs<=1 && report.strategies.is_none());
                assert!(report.bounds.lo>=previous.lo && report.bounds.hi<=previous.hi);
                previous=report.bounds;
                assert!(report.bounds.lo<=expected+1e-10 && report.bounds.hi>=expected-1e-10);
                assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[1-side].len(),2);
                if report.stop==super::Stop::WidthMet { closed=true;break; }
            }
            assert!(closed);
            if period==0 {assert_eq!(solver.stats().material_descent_bypasses,0);}
            bypasses+=solver.stats().material_descent_bypasses;
            let chance=solver.root_chance_diagnostics(&root).unwrap();
            assert_eq!(chance.iter().filter(|cell|cell.runs>0).count(),1);
        }}
        assert!(bypasses>0);
    }

    #[test]
    fn winning_fixed_reply_is_not_exported_as_a_whole_game_guarantee() {
        use nc2000_engine::battle::SearchChoice;
        let dex=conformance::load_dex();
        let mut root=battle(&dex,&["Seismic Toss","Splash"],&["Seismic Toss","Splash"]);
        let preview=[0,1].map(|side|root.legal_choices(&dex,side).first().copied());
        root.apply_choices(&dex,preview).unwrap();root.turn=1000;
        for side in 0..2 { let id=root.active_id(side).unwrap();root.poke_mut(id).hp=1; }
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let splash=SearchChoice::Move(dex.moves.id("splash").unwrap());
        for side in 0..2 {
            let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig::default(),side,vec![attack]);
            let report=solver.solve_reply(&root,splash,Some(0.75));
            let expected=if side==0 {1.0} else {0.0};
            assert!((report.bounds.lo-expected).abs()<1e-10 && (report.bounds.hi-expected).abs()<1e-10);
            let root_bounds=solver.restricted_root_payoffs(&root).unwrap().value;
            assert!(root_bounds.lo<=0.5 && root_bounds.hi>=0.5);
            let key=solver.solver.node_key(&root);
            let certificates=solver.into_value_certificates();
            assert!(certificates.entries.get(&key).is_none_or(|b|b.lo<=0.5 && b.hi>=0.5));
        }
    }

    #[test]
    fn healing_strategy_matches_exhaustive_turn_and_respects_status_pp_and_first_action() {
        use nc2000_engine::battle::{EffectHandle,SearchChoice};
        let dex=conformance::load_dex();
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        for healing in ["rest","recover"] {
            let rest=SearchChoice::Move(dex.moves.id(healing).unwrap());
            for side in 0..2 {
                for (hp,status,pp,first,selected) in [
                    (49,"",10,None,rest),(50,"",10,None,rest),(51,"",10,None,attack),
                    (49,"slp",10,None,attack),(49,"frz",10,None,attack),
                    (49,"par",10,None,rest),(49,"",0,None,attack),
                    (49,"",10,Some(attack),attack),
                ] {
                    let mut root=battle(&dex,&["Seismic Toss",healing],&["Seismic Toss",healing]);
                    let preview=[0,1].map(|s|root.legal_choices(&dex,s).first().copied());
                    root.apply_choices(&dex,preview).unwrap();root.turn=1000;
                    let me=root.active_id(side).unwrap();let foe=root.active_id(1-side).unwrap();
                    root.poke_mut(me).maxhp=100;root.poke_mut(me).hp=hp;root.poke_mut(foe).hp=1;
                    root.poke_mut(me).move_slots[1].pp=pp;root.poke_mut(me).base_move_slots[1].pp=pp;
                    if !status.is_empty() { root.set_status(&dex,me,status,Some(me),EffectHandle::None,true); }
                    let expected=root.legal_choices(&dex,1-side).iter().map(|&reply| {
                        let pair=if side==0 {[selected,reply]} else {[reply,selected]};
                        enumerate_step(&dex,&root,pair.map(Some),100_000).unwrap().leaves.iter().map(|leaf| {
                            let value=super::outcome_value(&leaf.battle).unwrap();
                            leaf.prob*if side==0 {value} else {1.0-value}
                        }).sum::<f64>()
                    }).fold(1.0,f64::min);
                    let cfg=BoundConfig {eps:1e-10,..BoundConfig::default()};
                    let mut solver=match first {
                        Some(action)=>super::FixedPolicySolver::new_with_first_action(&dex,cfg,side,vec![attack],&root,action),
                        None=>super::FixedPolicySolver::new(&dex,cfg,side,vec![attack]),
                    };
                    solver=if healing=="rest" {solver.with_rest_threshold(50)} else {solver.with_recover_threshold(50)};
                    let report=solver.solve(&root);
                    assert!((report.guaranteed_score.lo-expected).abs()<1e-10);
                    assert!((report.guaranteed_score.hi-expected).abs()<1e-10);
                    assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[side],vec![Some(selected)]);
                }
            }
        }
    }

    #[test]
    fn recover_guard_uses_public_last_foe_and_visible_species() {
        use nc2000_engine::battle::SearchChoice;
        let dex=conformance::load_dex();
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let recover=SearchChoice::Move(dex.moves.id("recover").unwrap());
        for side in 0..2 {for count in [1,2] {for matches in [false,true] {
            let own=vec![mon(&["Seismic Toss","Recover"])];
            let enemies=vec![mon(&["Seismic Toss"]);count];
            let teams=if side==0 {[&own,&enemies]} else {[&enemies,&own]};
            let mut root=Battle::from_fixture(&dex,"1,2,3,4",teams[0],teams[1]).unwrap();
            let preview=[0,1].map(|s|root.legal_choices(&dex,s).first().copied());
            root.apply_choices(&dex,preview).unwrap();root.turn=1000;
            let own_id=root.active_id(side).unwrap();let enemy_id=root.active_id(1-side).unwrap();
            root.poke_mut(own_id).maxhp=100;root.poke_mut(own_id).hp=49;
            let species=if matches {root.poke(enemy_id).species} else {dex.species.id("magikarp").unwrap()};
            let selected=if count==1 && matches {recover} else {attack};
            let expected=root.legal_choices(&dex,1-side).iter().map(|&reply| {
                let pair=if side==0 {[selected,reply]} else {[reply,selected]};
                enumerate_step(&dex,&root,pair.map(Some),100_000).unwrap().leaves.iter().map(|leaf| {
                    let value=super::outcome_value(&leaf.battle).unwrap();
                    leaf.prob*if side==0 {value} else {1.0-value}
                }).sum::<f64>()
            }).fold(1.0,f64::min);
            let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig {eps:1e-10,..BoundConfig::default()},side,vec![attack])
                .with_recover_threshold(50).with_recover_against_last(vec![species]);
            let report=solver.solve(&root);
            assert!((report.guaranteed_score.lo-expected).abs()<1e-10);
            assert!((report.guaranteed_score.hi-expected).abs()<1e-10);
            assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[side],vec![Some(selected)]);
        }}}
    }

    #[test]
    fn boom_aware_rest_matches_exhaustive_response_on_both_sides() {
        use nc2000_engine::battle::SearchChoice;
        let dex=conformance::load_dex();
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let rest=SearchChoice::Move(dex.moves.id("rest").unwrap());
        for side in 0..2 { for threat in ["Self-Destruct","Explosion","Splash"] {
            let teams=if side==0 {[&["Seismic Toss","Rest"][..],&["Seismic Toss",threat][..]]}
                else {[&["Seismic Toss",threat][..],&["Seismic Toss","Rest"][..]]};
            let mut root=battle(&dex,teams[0],teams[1]);
            let preview=[0,1].map(|s|root.legal_choices(&dex,s).first().copied());
            root.apply_choices(&dex,preview).unwrap();root.turn=1000;
            let own=root.active_id(side).unwrap();let foe=root.active_id(1-side).unwrap();
            root.poke_mut(own).maxhp=100;root.poke_mut(own).hp=50;root.poke_mut(foe).hp=1;
            let selected=if threat=="Splash" {rest} else {attack};
            let expected=root.legal_choices(&dex,1-side).iter().map(|&reply| {
                let pair=if side==0 {[selected,reply]} else {[reply,selected]};
                enumerate_step(&dex,&root,pair.map(Some),100_000).unwrap().leaves.iter().map(|leaf| {
                    let value=super::outcome_value(&leaf.battle).unwrap();leaf.prob*if side==0 {value} else {1.0-value}
                }).sum::<f64>()
            }).fold(1.0,f64::min);
            let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig {eps:1e-10,..BoundConfig::default()},side,vec![attack])
                .with_rest_threshold(75).with_rest_after_boom();
            let report=solver.solve(&root);
            assert!((report.guaranteed_score.lo-expected).abs()<1e-10);
            assert!((report.guaranteed_score.hi-expected).abs()<1e-10);
            assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[side],vec![Some(selected)]);
        }}
    }

    #[test]
    fn waking_rest_matches_exhaustive_responses_and_respects_strategy_guards() {
        use nc2000_engine::battle::{EffectHandle,SearchChoice};
        use nc2000_engine::state::DK;
        let dex=conformance::load_dex();
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let rest=SearchChoice::Move(dex.moves.id("rest").unwrap());
        for side in 0..2 { for waking in [false,true] {
            for (status,time,pp,first,threat) in [
                ("slp",1,10,false,false),("slp",2,10,false,false),
                ("",0,10,false,false),("frz",0,10,false,false),
                ("slp",1,0,false,false),("slp",1,10,true,false),("slp",1,10,false,true),
            ] {
                let foe_moves=if threat {&["Seismic Toss","Self-Destruct"][..]} else {&["Seismic Toss","Rest"][..]};
                let mut root=if side==0 {battle(&dex,&["Seismic Toss","Rest"],foe_moves)}
                    else {battle(&dex,foe_moves,&["Seismic Toss","Rest"])};
                let preview=[0,1].map(|s|root.legal_choices(&dex,s).first().copied());
                root.apply_choices(&dex,preview).unwrap();root.turn=1000;
                let me=root.active_id(side).unwrap();let foe=root.active_id(1-side).unwrap();
                root.poke_mut(me).maxhp=100;root.poke_mut(me).hp=49;root.poke_mut(foe).hp=1;
                root.poke_mut(me).move_slots[1].pp=pp;root.poke_mut(me).base_move_slots[1].pp=pp;
                if !status.is_empty() {root.set_status(&dex,me,status,Some(me),EffectHandle::None,true);}
                if status=="slp" {root.poke_mut(me).status_state.set_int(DK::Time,time);}
                let selected=if pp>0 && !first && !threat && (status.is_empty() || (waking && status=="slp" && time==1)) {rest} else {attack};
                let replies=root.legal_choices(&dex,1-side);
                let expected=replies.iter().map(|&reply| {
                    let pair=if side==0 {[selected,reply]} else {[reply,selected]};
                    enumerate_step(&dex,&root,pair.map(Some),100_000).unwrap().leaves.iter().map(|leaf| {
                        let value=super::outcome_value(&leaf.battle).unwrap();leaf.prob*if side==0 {value} else {1.0-value}
                    }).sum::<f64>()
                }).fold(1.0,f64::min);
                let cfg=BoundConfig {eps:1e-10,..BoundConfig::default()};
                let mut solver=if first {super::FixedPolicySolver::new_with_first_action(&dex,cfg,side,vec![attack],&root,attack)}
                    else {super::FixedPolicySolver::new(&dex,cfg,side,vec![attack])};
                solver=solver.with_rest_threshold(75).with_rest_after_boom();
                if waking {solver=solver.with_rest_on_wake();}
                let report=solver.solve(&root);
                assert!((report.guaranteed_score.lo-expected).abs()<1e-10);
                assert!((report.guaranteed_score.hi-expected).abs()<1e-10);
                let matrix=solver.restricted_root_payoffs(&root).unwrap();
                assert_eq!(matrix.actions[side],vec![Some(selected)]);
                assert_eq!(matrix.actions[1-side],replies.into_iter().map(Some).collect::<Vec<_>>());
                assert_eq!(solver.stats().fixed_waking_rest_nodes,usize::from(selected==rest && status=="slp"));
            }
        }}
    }

    #[test]
    #[should_panic(expected="fixed strategy must be configured before search")]
    fn waking_rest_cannot_change_after_search() {
        let dex=conformance::load_dex();let root=battle(&dex,&["Rest","Splash"],&["Splash"]);
        let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig {work_budget:0,..BoundConfig::default()},0,vec![])
            .with_rest_threshold(75);
        solver.solve(&root);solver.with_rest_on_wake();
    }

    #[test]
    fn rest_strategy_reacts_to_damage_and_returns_to_attack_after_healing() {
        use nc2000_engine::battle::SearchChoice;
        let dex=conformance::load_dex();
        let attack=SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
        let rest=SearchChoice::Move(dex.moves.id("rest").unwrap());
        let splash=SearchChoice::Move(dex.moves.id("splash").unwrap());
        let mut root=battle(&dex,&["Seismic Toss","Rest"],&["Seismic Toss","Splash"]);
        let preview=[0,1].map(|s|root.legal_choices(&dex,s).first().copied());
        root.apply_choices(&dex,preview).unwrap();
        let me=root.active_id(0).unwrap();let foe=root.active_id(1).unwrap();
        root.poke_mut(me).maxhp=100;root.poke_mut(me).hp=75;
        root.poke_mut(foe).maxhp=100;root.poke_mut(foe).hp=100;
        let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig {work_budget:0,..BoundConfig::default()},0,vec![attack])
            .with_rest_threshold(50);
        for (expected,reply,hp) in [(attack,attack,75),(rest,splash,25),(attack,splash,100)] {
            assert_eq!(root.poke(me).hp,hp);
            solver.solve(&root);
            let key=solver.solver.node_key(&root);solver.solver.init_cells(key);
            assert_eq!(solver.restricted_root_payoffs(&root).unwrap().actions[0],vec![Some(expected)]);
            root.apply_choices(&dex,[Some(expected),Some(reply)]).unwrap();
        }
    }

    #[test]
    #[should_panic(expected="fixed strategy must be configured before search")]
    fn rest_strategy_cannot_change_after_search() {
        let dex=conformance::load_dex();
        let root=battle(&dex,&["Rest","Splash"],&["Splash"]);
        let mut solver=super::FixedPolicySolver::new(&dex,BoundConfig {work_budget:0,..BoundConfig::default()},0,vec![]);
        solver.solve(&root);solver.with_rest_threshold(50);
    }

    #[test]
    fn fixed_policy_partial_bounds_enclose_exhaustive_two_turn_best_response() {
        use nc2000_engine::battle::SearchChoice;
        fn exhaustive(dex: &nc2000_engine::dex::Dex, b: &Battle, side: usize) -> f64 {
            if let Some(value) = super::outcome_value(b) { return value; }
            assert!(b.turn >= 999);
            let mut probe = b.clone();
            let choices = [0, 1].map(|s| probe.legal_choices(dex, s));
            let fixed = SearchChoice::Move(dex.moves.id("seismictoss").unwrap());
            assert!(choices[side].contains(&fixed));
            let scores = choices[1 - side].iter().map(|&reply| {
                let pair = if side == 0 { [fixed, reply] } else { [reply, fixed] };
                let step = enumerate_step(dex, b, pair.map(Some), 100_000).unwrap();
                step.leaves.iter().map(|leaf| leaf.prob * exhaustive(dex, &leaf.battle, side)).sum::<f64>()
            });
            if side == 0 { scores.fold(1.0, f64::min) }
            else { scores.fold(0.0, f64::max) }
        }
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Seismic Toss", "Protect"], &["Seismic Toss", "Protect"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 999;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = if side == 0 { 40 } else { 60 };
        }
        for (side, aggregate) in [(0, false), (1, false), (0, true), (1, true)] {
            let value = exhaustive(&dex, &root, side);
            let score = if side == 0 { value } else { 1.0 - value };
            let mut solver = super::FixedPolicySolver::new(&dex, BoundConfig {
                work_budget: 1, cell_cap: 0, eps: 1e-8,
                aggregate_lazy_children: aggregate, chance_fair_period: if aggregate { 2 } else { 0 },
                ..BoundConfig::default()
            }, side, vec![SearchChoice::Move(dex.moves.id("seismictoss").unwrap())]);
            let mut closed = false;
            for _ in 0..1000 {
                let report = solver.solve(&root);
                assert!(report.guaranteed_score.lo <= score + 1e-10);
                assert!(report.guaranteed_score.hi >= score - 1e-10);
                if report.stop == super::Stop::WidthMet { closed = true; break; }
            }
            assert!(closed);
        }
    }

    #[test]
    fn retained_eager_prefix_resumes_exhaustive_successor_distribution() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Body Slam"], &["Splash"]);
        let preview = [0,1].map(|s|root.legal_choices(&dex,s).first().copied());
        root.apply_choices(&dex,preview).unwrap();
        let choices = [0,1].map(|s|Some(root.legal_choices(&dex,s)[0]));
        let exact = enumerate_step(&dex,&root,choices,100_000).unwrap();
        for (cap,defer_eager_frontier) in [(1,false),(7,false),(1,true),(7,true)] {
            let mut solver = BoundSolver::new(&dex,BoundConfig {
                retain_eager_prefix:true,defer_eager_frontier,fold_terminal_nodes:false,..Default::default()
            });
            let key = solver.intern(root.clone());
            solver.init_cells(key);
            assert!(solver.try_eager(key,0,cap));
            assert_eq!(solver.stats.runs,cap);
            assert_eq!(solver.stats.retained_eager_runs,cap);
            assert!(solver.stats.retained_eager_leaves>0);
            if defer_eager_frontier { assert!(solver.stats.deferred_eager_leaves>0); }
            assert!(!solver.nodes[&key].cells[0].pending.is_empty());
            while !solver.nodes[&key].cells[0].pending.is_empty() {
                solver.expand(key,0);
                assert!(solver.stats.runs<100_000);
            }
            let mut expected = std::collections::HashMap::<_,f64>::new();
            for leaf in &exact.leaves { *expected.entry(solver.node_key(&leaf.battle)).or_default() += leaf.prob; }
            let mut found = std::collections::HashMap::<_,f64>::new();
            for &(mass,child) in &solver.nodes[&key].cells[0].resolved { *found.entry(child).or_default() += mass; }
            assert_eq!(found.len(),expected.len());
            for (child,mass) in expected { assert!((found[&child]-mass).abs()<1e-10); }
            assert!((found.values().sum::<f64>()-1.).abs()<1e-10);
        }
    }

    #[test]
    fn aggregated_lazy_edges_match_exhaustive_successor_probabilities() {
        use std::collections::HashMap;
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Splash"], &["Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        let choices = [0, 1].map(|s| Some(root.legal_choices(&dex, s)[0]));
        let exact = enumerate_step(&dex, &root, choices, 100_000).unwrap();
        for aggregate in [false, true] {
            let mut solver = BoundSolver::new(&dex, BoundConfig {
                cell_cap: 0, aggregate_lazy_children: aggregate, ..BoundConfig::default()
            });
            let key = solver.intern(root.clone());
            solver.init_cells(key);
            while !solver.nodes[&key].cells[0].pending.is_empty() {
                solver.expand(key, 0);
                assert!(solver.stats.runs < 100_000);
            }
            let mut expected = HashMap::<_, f64>::new();
            for leaf in &exact.leaves {
                assert!(leaf.battle.outcome().is_none());
                *expected.entry(solver.node_key(&leaf.battle)).or_default() += leaf.prob;
            }
            let cell = &solver.nodes[&key].cells[0];
            let mut found = HashMap::<_, f64>::new();
            for &(mass, child) in &cell.resolved { *found.entry(child).or_default() += mass; }
            assert_eq!(found.len(), expected.len());
            for (child, mass) in expected { assert!((found[&child] - mass).abs() < 1e-10); }
            assert!((found.values().sum::<f64>() - 1.0).abs() < 1e-10);
            if aggregate {
                assert!(solver.stats.lazy_edge_merges > 0);
                assert_eq!(cell.resolved.len(), found.len());
            } else {
                assert_eq!(solver.stats.lazy_edge_merges, 0);
                assert!(cell.resolved.len() > found.len());
            }
        }
    }

    #[test]
    fn partial_action_bounds_enclose_exhaustive_first_action_values() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Quick Attack", "Splash"], &["Quick Attack", "Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let acts = [0, 1].map(|s| root.legal_choices(&dex, s));
        let mut matrix = Vec::new();
        for &a in &acts[0] {
            for &b in &acts[1] {
                let step = enumerate_step(&dex, &root, [Some(a), Some(b)], 100_000).unwrap();
                matrix.push(step.leaves.iter().map(|leaf| {
                    leaf.prob * super::outcome_value(&leaf.battle).unwrap()
                }).sum::<f64>());
            }
        }
        let mut solver = BoundSolver::new(&dex, BoundConfig {
            work_budget: 1,
            cell_cap: 0,
            eps: 0.0,
            ..BoundConfig::default()
        });
        for _ in 0..1000 {
            let report = solver.solve(&root, None);
            for side in 0..2 {
                for (action, bounds) in solver.action_bounds(&root, side) {
                    let index = acts[side].iter().position(|&a| Some(a) == action).unwrap();
                    let expected = (0..acts[1 - side].len()).map(|reply| {
                        if side == 0 { matrix[index * acts[1].len() + reply] }
                        else { 1.0 - matrix[reply * acts[1].len() + index] }
                    }).fold(1.0, f64::min);
                    assert!(bounds.lo <= expected + 1e-10);
                    assert!(bounds.hi >= expected - 1e-10);
                }
            }
            if report.bounds.width() <= 1e-10 {
                assert!((report.bounds.mid() - solve_matrix(&matrix, acts[0].len(), acts[1].len()).0).abs() < 1e-10);
                return;
            }
        }
        panic!("microgame did not close");
    }

    #[test]
    fn terminal_folding_preserves_bounds_and_releases_leaf_nodes() {
        let dex = conformance::load_dex();
        let mut root = battle(
            &dex,
            &["Quick Attack", "Splash"],
            &["Quick Attack", "Splash"],
        );
        let c0 = root.legal_choices(&dex, 0)[0];
        let c1 = root.legal_choices(&dex, 1)[0];
        root.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }

        let run = |fold_terminal_nodes| {
            let mut solver = BoundSolver::new(
                &dex,
                BoundConfig {
                    work_budget: 100_000,
                    cell_cap: 100_000,
                    eps: 0.0,
                    fold_terminal_nodes,
                    fold_closed_nodes: false,
                    ..BoundConfig::default()
                },
            );
            let report = solver.solve(&root, None);
            (report, solver.node_count())
        };
        let (folded, folded_nodes) = run(true);
        let (raw, raw_nodes) = run(false);
        assert!((folded.bounds.lo - raw.bounds.lo).abs() < 1e-12);
        assert!((folded.bounds.hi - raw.bounds.hi).abs() < 1e-12);
        assert!(folded_nodes < raw_nodes, "{folded_nodes} !< {raw_nodes}");
    }

    #[test]
    fn pruning_and_support_br_preserve_a_closed_game_value() {
        let dex = conformance::load_dex();
        let mut root = battle(
            &dex,
            &["Quick Attack", "Splash"],
            &["Quick Attack", "Splash"],
        );
        let c0 = root.legal_choices(&dex, 0)[0];
        let c1 = root.legal_choices(&dex, 1)[0];
        root.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let run = |certified_action_pruning, support_br_scheduling| {
            let mut solver = BoundSolver::new(
                &dex,
                BoundConfig {
                    work_budget: 100_000,
                    cell_cap: 100_000,
                    eps: 0.0,
                    certified_action_pruning,
                    support_br_scheduling,
                    ..BoundConfig::default()
                },
            );
            solver.solve(&root, None).bounds
        };
        let baseline = run(false, false);
        for candidate in [run(true, false), run(false, true), run(true, true)] {
            assert!((candidate.lo - baseline.lo).abs() < 1e-12);
            assert!((candidate.hi - baseline.hi).abs() < 1e-12);
        }
    }

    #[test]
    fn closed_sweep_preserves_root_bounds_and_compacts_exact_children() {
        let dex = conformance::load_dex();
        let mut root = battle(
            &dex,
            &["Quick Attack", "Splash"],
            &["Quick Attack", "Splash"],
        );
        let c0 = root.legal_choices(&dex, 0)[0];
        let c1 = root.legal_choices(&dex, 1)[0];
        root.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let mut solver = BoundSolver::new(
            &dex,
            BoundConfig {
                work_budget: 100_000,
                cell_cap: 100_000,
                eps: 0.0,
                fold_terminal_nodes: false,
                fold_closed_nodes: false,
                ..BoundConfig::default()
            },
        );
        let before = solver.solve(&root, None).bounds;
        let before_nodes = solver.node_count();
        solver.fold_closed_graph();
        let after = solver.peek(&root).unwrap();
        assert_eq!(before.lo, after.lo);
        assert_eq!(before.hi, after.hi);
        assert!(solver.node_count() < before_nodes);
        assert!(solver.closed_count() > 0);
        assert!(solver.stats.closed_folds > 0);
    }

    #[test]
    fn cross_bounds_certify_iterated_row_and_column_dominance() {
        // Row 1 can never beat row 0. After removing it, column 0 can never
        // improve on column 1 for the minimizer.
        let lo = vec![0.80, 0.70, 0.20, 0.30, 0.90, 0.50];
        let hi = vec![0.82, 0.72, 0.40, 0.50, 0.92, 0.52];
        let mut rows = vec![true; 3];
        let mut cols = vec![true; 2];
        let (pr, pc, checks) = prune_dominated_actions(&lo, &hi, 3, 2, &mut rows, &mut cols);
        assert_eq!(rows, vec![true, false, false]);
        assert_eq!(cols, vec![false, true]);
        assert_eq!((pr, pc), (2, 1));
        assert!(checks > 0);

        // Overlapping intervals are not a proof.
        let mut rows = vec![true; 2];
        let mut cols = vec![true; 2];
        let (pr, pc, _) = prune_dominated_actions(
            &[0.4, 0.4, 0.3, 0.3],
            &[0.7, 0.7, 0.6, 0.6],
            2,
            2,
            &mut rows,
            &mut cols,
        );
        assert_eq!((pr, pc), (0, 0));
        assert!(rows.into_iter().all(|v| v) && cols.into_iter().all(|v| v));

        // Endpoint-equivalent actions may collapse, but never all of them.
        let mut rows = vec![true; 2];
        let mut cols = vec![true; 2];
        prune_dominated_actions(&[1.0; 4], &[1.0; 4], 2, 2, &mut rows, &mut cols);
        assert_eq!(rows.iter().filter(|&&v| v).count(), 1);
        assert_eq!(cols.iter().filter(|&&v| v).count(), 1);

        // Every exact corner inside a certified interval has the same value
        // before and after removing the dominated row.
        let lo = [0.8, 0.7, 0.2, 0.3];
        let hi = [0.9, 0.8, 0.4, 0.5];
        let mut active_rows = vec![true; 2];
        let mut active_cols = vec![true; 2];
        prune_dominated_actions(&lo, &hi, 2, 2, &mut active_rows, &mut active_cols);
        for mask in 0..16 {
            let matrix: Vec<f64> = (0..4)
                .map(|i| if mask & (1 << i) == 0 { lo[i] } else { hi[i] })
                .collect();
            let (full, full_gap) = solve_matrix(&matrix, 2, 2);
            let reduced = solve_restricted_game(&matrix, 2, 2, &active_rows, &active_cols);
            assert!((full - reduced.value).abs() <= full_gap + reduced.gap + 1e-12);
        }
    }

    #[test]
    fn restricted_game_embeds_policies_in_original_action_space() {
        let matrix = vec![0.8, 0.7, 0.2, 0.3, 0.9, 0.5];
        let solved = solve_restricted_game(&matrix, 3, 2, &[true, false, true], &[false, true]);
        assert!((solved.value - 0.7).abs() < 1e-12);
        assert_eq!(solved.x, vec![1.0, 0.0, 0.0]);
        assert_eq!(solved.y, vec![0.0, 1.0]);
        assert_eq!(solved.gap, 0.0);
    }

    fn synthetic_cell(lo: f64, hi: f64) -> Cell {
        Cell {
            visits: 0,
            runs: 0,
            fair_cursor: 0,
            resolved: Vec::new(),
            fixed_lo: 0.0,
            fixed_hi: 0.0,
            pending: Vec::new(),
            pending_mass: 0.0,
            lo,
            hi,
            tried_eager: true,
        }
    }

    fn synthetic_node(select_count: usize, fair_cursor: usize) -> Node {
        Node {
            chance_class: 0,
            remaining_hp: [1,1],
            battle: None,
            acts0: vec![None, None],
            acts1: vec![None, None],
            cells: vec![
                synthetic_cell(0.2, 0.3),
                synthetic_cell(0.2, 0.9),
                synthetic_cell(0.0, 1.0),
                synthetic_cell(0.0, 1.0),
            ],
            lo: 0.0,
            hi: 1.0,
            strat: Some(StageWitness {
                xlo: vec![1.0, 0.0],
                ylo: vec![1.0, 0.0],
                xhi: vec![1.0, 0.0],
                yhi: vec![1.0, 0.0],
                gap_lo: 0.0,
                gap_hi: 0.0,
            }),
            active_rows: vec![true, true],
            active_cols: vec![true, true],
            select_count,
            fair_cursor,
            stall_rank: None,
            resource_rank: None,
        }
    }

    #[test]
    fn chance_groups_prioritize_total_uncertainty_without_merging_states() {
        let dex = conformance::load_dex();
        let mut solver = BoundSolver::new(&dex, BoundConfig {
            chance_status_groups: true, ..BoundConfig::default()
        });
        let keys = [100,101,102,103].map(super::NodeKey::Exact);
        for key in keys { solver.nodes.insert(key, synthetic_node(0,0)); }
        solver.nodes.get_mut(&keys[2]).unwrap().chance_class = 1;
        solver.nodes.get_mut(&keys[3]).unwrap().chance_class = 1;
        solver.nodes.get_mut(&keys[0]).unwrap().cells[0].resolved = vec![(0.4,keys[1]),(0.3,keys[2]),(0.3,keys[3])];
        assert_eq!(solver.grouped_chance_child(keys[0],0),Some((0.3,keys[2])));
        assert_eq!(solver.nodes.len(),4);
        solver.nodes.get_mut(&keys[2]).unwrap().hi = 0.0;
        assert_eq!(solver.grouped_chance_child(keys[0],0),Some((0.4,keys[1])));
        solver.cfg.chance_status_groups = false;
        assert!(solver.grouped_chance_child(keys[0],0).is_none());
    }

    #[test]
    fn chance_fairness_visits_small_unresolved_branches_and_skips_closed_ones() {
        let dex = conformance::load_dex();
        let mut solver = BoundSolver::new(&dex, BoundConfig {
            chance_fair_period: 32, ..BoundConfig::default()
        });
        let keys = [100, 101, 102, 103].map(super::NodeKey::Exact);
        for key in keys { solver.nodes.insert(key, synthetic_node(0, 0)); }
        solver.nodes.get_mut(&keys[3]).unwrap().hi = 0.0;
        let root = solver.nodes.get_mut(&keys[0]).unwrap();
        root.cells[0].resolved = vec![(0.8, keys[1]), (0.19, keys[2]), (0.01, keys[3])];
        assert!(solver.fair_chance_child(keys[0], 0).is_none());
        solver.nodes.get_mut(&keys[0]).unwrap().cells[0].visits = 31;
        for expected in [keys[1], keys[2], keys[1], keys[2]] {
            assert_eq!(solver.fair_chance_child(keys[0], 0).unwrap().1, expected);
        }
        solver.cfg.chance_fair_period = 0;
        assert!(solver.fair_chance_child(keys[0], 0).is_none());
    }

    #[test]
    fn mixed_policy_certificates_and_side_swap_match_best_responses() {
        for matrix in [[0.9, 0.1, 0.2, 0.8], [0.1, 0.8, 0.9, 0.2]] {
            let mut node = synthetic_node(0, 0);
            node.cells = matrix.iter().map(|&v| synthetic_cell(v, v)).collect();
            let solved = solve_restricted_game(&matrix, 2, 2, &[true; 2], &[true; 2]);
            node.lo = solved.value - solved.gap * 0.5;
            node.hi = solved.value + solved.gap * 0.5;
            node.strat = Some(StageWitness {
                xlo: solved.x.clone(),
                ylo: solved.y.clone(),
                xhi: solved.x,
                yhi: solved.y,
                gap_lo: solved.gap,
                gap_hi: solved.gap,
            });
            let policies = super::node_strategies(&node).unwrap();
            for (side, policy) in policies.iter().enumerate() {
                assert!(policy.actions.iter().all(|(_, p)| *p > 0.0));
                assert!(policy.meets(0.02));
                let floor = (0..2).map(|reply| {
                    policy.actions.iter().enumerate().map(|(a, (_, p))| {
                        p * if side == 0 { matrix[a * 2 + reply] }
                            else { 1.0 - matrix[reply * 2 + a] }
                    }).sum::<f64>()
                }).fold(1.0, f64::min);
                assert!(policy.guaranteed_value <= floor);
                assert!((policy.guaranteed_value - floor).abs() < 1e-10);
                assert!(policy.max_loss < 1e-10);
            }
        }
    }

    #[test]
    fn closed_value_does_not_certify_a_policy_against_unchecked_replies() {
        let mut node = synthetic_node(0, 0);
        node.lo = 0.5;
        node.hi = 0.5;
        node.active_rows[1] = false;
        node.active_cols[1] = false;
        node.cells[0] = synthetic_cell(0.5, 0.5);
        node.cells[1] = synthetic_cell(0.0, 1.0);
        let report = super::SolveReport {
            bounds: super::Bounds { lo: node.lo, hi: node.hi },
            stop: super::Stop::WidthMet,
            runs: 0,
            strategies: super::node_strategies(&node),
        };
        assert_eq!(report.status(0.02), super::SolutionStatus::ClosedValue);
        for side in 0..2 {
            assert!(report.strategy(side, 0.02).is_none());
            assert!(report.strategy(side, f64::NAN).is_none());
            assert!(report.strategy(side, -1.0).is_none());
        }
        node.strat.as_mut().unwrap().xlo = vec![f64::NAN, 0.0];
        assert!(super::node_strategies(&node).is_none());
    }

    #[test]
    fn compacted_mixed_policy_survives_while_value_only_cache_has_no_policy() {
        let dex = conformance::load_dex();
        let root = battle(&dex, &["Splash"], &["Splash"]);
        let mut solver = BoundSolver::new(&dex, BoundConfig::default());
        let key = solver.node_key(&root);
        let mut node = synthetic_node(0, 0);
        node.lo = 0.5;
        node.hi = 0.5;
        node.cells = [1.0, 0.0, 0.0, 1.0].iter().map(|&v| synthetic_cell(v, v)).collect();
        let witness = node.strat.as_mut().unwrap();
        witness.xlo = vec![0.5, 0.5];
        witness.yhi = vec![0.5, 0.5];
        let expected = super::node_strategies(&node).unwrap();
        solver.nodes.insert(key, node);
        solver.fold_closed_graph();
        assert!(!solver.nodes.contains_key(&key));
        assert!(solver.closed.contains_key(&key));
        let report = solver.solve_strategy(&root, 0, 0.02);
        assert_eq!(report.runs, 0);
        assert_eq!(report.strategies.unwrap(), expected);
        solver.closed_strategies.clear();
        assert!(solver.solve(&root, None).strategies.is_none());
    }

    #[test]
    fn strategy_target_stops_before_value_closure_and_small_budgets_resume() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Quick Attack", "Splash"], &["Quick Attack", "Splash"]);
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 1000;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let mut solver = BoundSolver::new(&dex, BoundConfig {
            work_budget: 1,
            cell_cap: 4096,
            eps: 0.0,
            certified_action_pruning: false,
            ..BoundConfig::default()
        });
        for _ in 0..1000 {
            let report = solver.solve_strategy(&root, 0, 0.51);
            assert!(report.runs <= 1);
            if report.stop == super::Stop::StrategyMet {
                assert!(report.strategy(0, 0.51).is_some());
                assert!(report.bounds.width() > 1e-10);
                return;
            }
        }
        panic!("strategy target never reached");
    }

    #[test]
    fn strategy_refinement_encloses_exhaustive_policies_and_resumes() {
        let dex = conformance::load_dex();
        for (moves,turn,hp) in [(["Quick Attack","Splash"],1000,[1,1]),
                              (["Recover","Seismic Toss"],999,[60,40])] {
            let mut root = battle(&dex,&moves,&moves);
            let preview = [0,1].map(|s|root.legal_choices(&dex,s).first().copied());
            root.apply_choices(&dex,preview).unwrap();
            root.turn=turn;
            for side in 0..2 {
                let id=root.active_id(side).unwrap();
                root.poke_mut(id).hp=hp[side];
            }
            let values=exhaustive_matrix(&dex,&root,2);
            let value=solve_matrix(&values,2,2).0;
            let exact=super::RootPayoffs {
                actions:[0,1].map(|s|root.legal_choices(&dex,s).into_iter().map(Some).collect()),
                cells:values.into_iter().map(|v|super::Bounds {lo:v,hi:v}).collect(),
                value:super::Bounds {lo:value,hi:value},
            };
            for side in 0..2 {
                let mut solver=BoundSolver::new(&dex,BoundConfig {
                    work_budget:7,cell_cap:0,eps:1e-12,strategy_refinement:true,
                    resource_closure_nodes:Some(0),..BoundConfig::default()
                });
                let mut certified=false;
                let mut previous=super::Bounds {lo:0.,hi:1.};
                for _ in 0..2000 {
                    let report=solver.solve_strategy(&root,side,1e-8);
                    assert!(report.runs<=7);
                    assert!(report.bounds.lo<=value+1e-9 && report.bounds.hi>=value-1e-9);
                    assert!(report.bounds.lo>=previous.lo && report.bounds.hi<=previous.hi);
                    previous=report.bounds;
                    if let Some(policies)=report.strategies.as_ref() {
                        for s in 0..2 {
                            let policy:Vec<_>=policies[s].actions.iter().map(|(_,p)|*p).collect();
                            assert!(exact.policy_loss(s,&policy).hi<=policies[s].max_loss+1e-8);
                        }
                    }
                    if let Some(model)=solver.root_correlation(&root) {
                        let correlated=model.loss_bounds();
                        for s in 0..2 {
                            assert!(exact.policy_loss(s,&model.policies[s]).hi<=correlated.max_loss[s]+1e-8);
                        }
                    }
                    if report.strategy(side,1e-8).is_some() { certified=true;break; }
                    assert!(report.runs>0);
                }
                assert!(certified);
                assert!(solver.stats.strategy_refinement_trials>0);
            }
        }
    }

    fn exhaustive_matrix(dex: &nc2000_engine::dex::Dex, root: &Battle, depth: usize) -> Vec<f64> {
        assert!(depth > 0);
        let mut probe = root.clone();
        let needs = root.needs_choice();
        let actions = [0, 1].map(|side| {
            if needs[side] { probe.legal_choices(dex, side).into_iter().map(Some).collect() }
            else { vec![None] }
        });
        let mut values = Vec::new();
        for &a in &actions[0] {
            for &b in &actions[1] {
                let step = enumerate_step(dex, root, [a, b], 100_000).unwrap();
                values.push(step.leaves.into_iter().map(|leaf| {
                    let value = super::outcome_value(&leaf.battle).unwrap_or_else(|| {
                        let matrix = exhaustive_matrix(dex, &leaf.battle, depth - 1);
                        let mut child = leaf.battle.clone();
                        let needs = child.needs_choice();
                        let sizes = [0, 1].map(|s| if needs[s] { child.legal_choices(dex, s).len() } else { 1 });
                        solve_matrix(&matrix, sizes[0], sizes[1]).0
                    });
                    leaf.prob * value
                }).sum());
            }
        }
        values
    }

    fn assert_policy_against_exhaustive(dex: &nc2000_engine::dex::Dex, root: &Battle, depth: usize) {
        let matrix = exhaustive_matrix(dex, root, depth);
        let mut probe = root.clone();
        let needs = root.needs_choice();
        let actions = [0, 1].map(|side| {
            if needs[side] { probe.legal_choices(dex, side).into_iter().map(Some).collect::<Vec<_>>() }
            else { vec![None] }
        });
        let value = solve_matrix(&matrix, actions[0].len(), actions[1].len()).0;
        let mut solver = BoundSolver::new(dex, BoundConfig {
            work_budget: 1,
            cell_cap: 0,
            eps: 1e-10,
            certified_action_pruning: false,
            ..BoundConfig::default()
        });
        let mut previous = super::Bounds { lo: 0.0, hi: 1.0 };
        for _ in 0..20_000 {
            let report = solver.solve(root, None);
            assert!(report.bounds.lo <= value + 1e-10);
            assert!(report.bounds.hi >= value - 1e-10);
            assert!(report.bounds.lo >= previous.lo);
            assert!(report.bounds.hi <= previous.hi);
            previous = report.bounds;
            if let Some(policies) = &report.strategies {
                for side in 0..2 {
                    let policy = &policies[side];
                    let floor = (0..actions[1 - side].len()).map(|reply| {
                        policy.actions.iter().map(|(action, p)| {
                            let a = actions[side].iter().position(|x| x == action).unwrap();
                            p * if side == 0 { matrix[a * actions[1].len() + reply] }
                                else { 1.0 - matrix[reply * actions[1].len() + a] }
                        }).sum::<f64>()
                    }).fold(1.0, f64::min);
                    let optimum = if side == 0 { value } else { 1.0 - value };
                    assert!(policy.guaranteed_value <= floor + 1e-10);
                    assert!(policy.max_loss >= optimum - floor - 1e-10);
                }
            }
            if report.bounds.width() <= 1e-10 {
                assert!(report.strategy(0, 0.02).is_some());
                assert!(report.strategy(1, 0.02).is_some());
                let policies = report.strategies.unwrap();
                solver.roots.clear();
                solver.fold_closed_graph();
                let cached = solver.solve(root, None);
                assert_eq!(cached.runs, 0);
                assert_eq!(cached.strategies.unwrap(), policies);
                return;
            }
        }
        panic!("microgame did not close");
    }

    #[test]
    fn partial_strategies_cover_random_damage_recovery_and_turn_cap() {
        let dex = conformance::load_dex();
        for (moves0, moves1) in [
            (vec!["Quick Attack", "Splash"], vec!["Quick Attack", "Splash"]),
            (vec!["Rest", "Seismic Toss"], vec!["Seismic Toss", "Splash"]),
            (vec!["Seismic Toss", "Splash"], vec!["Rest", "Seismic Toss"]),
        ] {
            let mut root = battle(&dex, &moves0, &moves1);
            let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
            root.apply_choices(&dex, preview).unwrap();
            root.turn = 1000;
            for side in 0..2 {
                let id = root.active_id(side).unwrap();
                root.poke_mut(id).hp = 1;
            }
            assert_policy_against_exhaustive(&dex, &root, 2);
        }
    }

    #[test]
    fn forced_replacement_policy_and_no_choice_opponent_are_certified() {
        let dex = conformance::load_dex();
        let mut root = Battle::from_fixture(&dex, "1,2,3,4",
            &[mon(&["Splash"]), mon(&["Seismic Toss"]), mon(&["Splash"])],
            &[mon(&["Seismic Toss"])]).unwrap();
        let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, preview).unwrap();
        root.turn = 999;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = 1;
        }
        let choices = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
        root.apply_choices(&dex, choices).unwrap();
        assert_eq!(root.needs_choice(), [true, false]);
        assert!(root.legal_choices(&dex, 0).len() > 1);
        assert_policy_against_exhaustive(&dex, &root, 3);
    }

    #[test]
    fn support_br_scheduler_visits_tied_pure_response_and_fair_cursor() {
        let dex = conformance::load_dex();
        let key = NodeKey::Exact(7);
        let mut solver = BoundSolver::new(&dex, BoundConfig::default());
        // Next visit is lower-directed. Column 1 is a tied pure response to
        // xlo but has zero ylo support; the scheduler must still select it.
        solver.nodes.insert(key, synthetic_node(1, 0));
        assert_eq!(solver.select_cell_for_bound(key, None), 1);
        assert_eq!(solver.stats.lower_br_picks, 1);

        // Every 32nd visit ignores policy scores and advances a deterministic
        // cursor over all unresolved active cells.
        solver.nodes.insert(key, synthetic_node(31, 2));
        assert_eq!(solver.select_cell_for_bound(key, None), 2);
        assert_eq!(solver.stats.fair_cell_picks, 1);
    }

    #[test]
    fn resource_and_directional_search_enclose_exhaustive_two_turn_policies() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Recover", "Seismic Toss"], &["Recover", "Seismic Toss"]);
        let choices = [0, 1].map(|side| root.legal_choices(&dex, side)[0]);
        root.apply_choices(&dex, choices.map(Some)).unwrap();
        root.turn = 999;
        for side in 0..2 {
            let id = root.active_id(side).unwrap();
            root.poke_mut(id).hp = if side == 0 { 60 } else { 40 };
        }
        assert!(crate::stall::classify_two_sided_heal(&root, &dex).is_ok());
        let matrix = exhaustive_matrix(&dex, &root, 2);
        let value = solve_matrix(&matrix, 2, 2).0;
        let exact = super::RootPayoffs {
            actions: [0, 1].map(|side| root.legal_choices(&dex, side).into_iter().map(Some).collect()),
            cells: matrix.iter().map(|&v| super::Bounds { lo: v, hi: v }).collect(),
            value: super::Bounds { lo: value, hi: value },
        };
        let targets: Vec<_> = (0..2).flat_map(|side| {
            [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.3, 0.7]].map(|policy| (side, policy))
        }).collect();
        for (period, directed, targeted, grouped, quotient) in [(0, false, false, false, false), (1, false, false, false, false),
            (2, false, false, false, false), (1, true, false, false, false), (1, false, true, false, false), (1, true, true, false, false),
            (0, false, false, true, false), (1, true, true, true, false), (0, false, false, false, true)] {
            let mut solver = BoundSolver::new(&dex, BoundConfig {
                work_budget: 17, cell_cap: 0, eps: 1e-10,
                resource_closure_nodes: Some(0), resource_closure_period: period,
                directed_policy_refinement: directed, target_aware_refinement: targeted,
                chance_status_groups: grouped,
                dead_quick_claw_quotient: quotient,
                ..BoundConfig::default()
            });
            let mut previous = super::Bounds { lo: 0.0, hi: 1.0 };
            for _ in 0..2_000 {
                let report = solver.solve_policy_losses(&root, &targets, 1e-8);
                let payoffs = solver.root_payoffs(&root).unwrap();
                for (side, policy) in &targets {
                    let expected = exact.policy_loss(*side, policy);
                    let found = payoffs.policy_loss(*side, policy);
                    assert!(found.lo <= expected.lo + 1e-9);
                    assert!(found.hi >= expected.hi - 1e-9);
                }
                assert!(report.bounds.lo <= value + 1e-10);
                assert!(report.bounds.hi >= value - 1e-10);
                assert!(report.bounds.lo >= previous.lo);
                assert!(report.bounds.hi <= previous.hi);
                previous = report.bounds;
                if report.stop == super::Stop::StrategyMet { break; }
                assert!(report.runs > 0, "period={period}, bounds={:?}, stop={:?}", report.bounds, report.stop);
            }
            for (side, policy) in &targets {
                assert!(solver.root_payoffs(&root).unwrap().policy_loss(*side, policy).width() < 1e-8);
            }
            assert_eq!(solver.stats.lower_directed_trials > 0, directed);
            assert_eq!(solver.stats.upper_directed_trials > 0, directed);
            assert_eq!(solver.stats.resource_closure_visits > 0, period > 0);
            assert_eq!(solver.stats.two_sided_invalidations, 0);
        }
    }

    #[test]
    fn explicit_resource_threshold_preserves_prefix_across_node_budgets() {
        let dex = conformance::load_dex();
        let mut root = battle(&dex, &["Rest", "Tackle", "Protect"], &["Rest", "Tackle", "Protect"]);
        let choices = [0, 1].map(|side| root.legal_choices(&dex, side)[0]);
        root.apply_choices(&dex, choices.map(Some)).unwrap();
        let targets: Vec<_> = (0..2).map(|side| {
            let mut policy = vec![0.0; root.legal_choices(&dex, side).len()];
            policy[0] = 1.0;
            (side, policy)
        }).collect();
        let mut results = Vec::new();
        for node_budget in [10_000, 20_000] {
            let mut solver = BoundSolver::new(&dex, BoundConfig {
                work_budget: 2_000,
                node_budget,
                cell_cap: 0,
                resource_closure_nodes: Some(0),
                resource_closure_period: 2,
                ..BoundConfig::default()
            });
            let report = solver.solve_policy_losses(&root, &targets, 0.02);
            assert!(solver.node_count() < 10_000);
            assert_eq!(solver.stats.two_sided_invalidations, 0);
            assert!(solver.stats.resource_closure_visits > 0);
            results.push((report.bounds.lo, report.bounds.hi, solver.stats.runs,
                solver.node_count(), solver.stats.resource_closure_visits,
                solver.stats.first_resource_closure_nodes));
        }
        assert_eq!(results[0], results[1]);
    }

    #[test]
    fn solver_tracks_a_two_sided_resource_root_without_invariant_failure() {
        let dex = conformance::load_dex();
        let mut root = battle(
            &dex,
            &["Rest", "Tackle", "Protect"],
            &["Rest", "Tackle", "Protect"],
        );
        let c0 = root.legal_choices(&dex, 0)[0];
        let c1 = root.legal_choices(&dex, 1)[0];
        root.apply_choices(&dex, [Some(c0), Some(c1)]).unwrap();
        assert!(crate::stall::classify_two_sided_heal(&root, &dex).is_ok());

        let mut solver = BoundSolver::new(
            &dex,
            BoundConfig {
                work_budget: 1,
                cell_cap: 100,
                ..BoundConfig::default()
            },
        );
        solver.solve(&root, None);
        assert_eq!(solver.stats.two_sided_roots, 1);
        assert_eq!(solver.stats.two_sided_invalidations, 0);
        assert!(solver.stats.min_resource_pp.is_some());
    }
}
