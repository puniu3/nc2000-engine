//! M17e e-5 (Phase B of ENDGAME-SOLVER-ALGORITHM.md) — lazy certified
//! bound solver.
//!
//! A persistent search graph over decision states: every node carries a
//! sound value interval (side-0 win probability), unexpanded chance mass
//! counts as [0, 1], and one trial = walk root→frontier by LP-support ×
//! uncertainty, expand ONE pending chance script (one engine run), back
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

use std::collections::{HashMap, HashSet};

use nc2000_engine::battle::enumerate::{enumerate_step, run_scripted};
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::exact::solve_matrix_full;
use crate::stall::{classify_one_sided_heal, MonotoneRank, OneSidedHeal};

#[derive(Clone, Debug)]
pub struct BoundConfig {
    /// Engine executions per `solve` call.
    pub work_budget: usize,
    /// On a cell's first visit, try the EAGER enumerator (which keeps the
    /// damage-roll range merge) up to this many runs; only fans that
    /// overflow it fall back to lazy pending expansion. Measured: without
    /// this, lazy expansion re-pays merged overkill fans one run per roll
    /// class (b60: 411k runs vs the eager solver's 11.7k).
    pub cell_cap: usize,
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
    /// Fold terminal successor mass directly into its parent cell instead
    /// of retaining one graph node per terminal fingerprint.
    pub fold_terminal_nodes: bool,
    /// Periodically fold exact non-root nodes into a compact closed memo.
    /// This is graph-local and does not depend on stall classification.
    pub fold_closed_nodes: bool,
    /// Near the node budget, prefer lower proven resource generations in a
    /// conservatively classified one-sided-heal subgame. Every generated
    /// edge is checked; the first violation disables only this scheduler.
    pub monotone_stall_scheduling: bool,
    /// Permanently remove a pure action only after cell intervals prove it
    /// pointwise dominated by another live action. The reduced matrix has
    /// the same true value; unresolved actions are never assumed irrelevant.
    pub certified_action_pruning: bool,
    /// In addition to current LP-support cells, prioritize optimistic pure
    /// best responses crossed with the opponent's support. This changes only
    /// frontier order; certification still uses every unpruned action.
    pub support_br_scheduling: bool,
}

impl Default for BoundConfig {
    fn default() -> Self {
        BoundConfig {
            work_budget: 1_000_000,
            node_budget: 120_000,
            cell_cap: 4096,
            eps: 0.02,
            trial_depth: 24,
            descend_floor: 0.1,
            dead_damage_quotient: true,
            fold_terminal_nodes: true,
            fold_closed_nodes: true,
            monotone_stall_scheduling: true,
            certified_action_pruning: true,
            support_br_scheduling: true,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub lo: f64,
    pub hi: f64,
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
    /// `lo > tau_hi`: the true value is proven above the threshold band.
    ProvenAbove,
    /// `hi < tau_lo`: the true value is proven below the threshold band.
    ProvenBelow,
    /// Root interval fell entirely inside the threshold band.
    Contained,
    WorkExhausted,
    NodeBudget,
}

#[derive(Clone, Copy, Debug)]
pub struct SolveReport {
    pub bounds: Bounds,
    pub stop: Stop,
    /// Engine runs consumed by THIS call.
    pub runs: usize,
}

#[derive(Clone, Debug, Default)]
pub struct BoundStats {
    pub runs: usize,
    pub expansions: usize,
    pub trials: usize,
    pub lp_solves: usize,
    pub worst_gap: f64,
    pub peak_nodes: usize,
    pub closed_folds: usize,
    pub monotone_roots: usize,
    pub monotone_invalidations: usize,
    pub dominated_rows: usize,
    pub dominated_cols: usize,
    pub dominance_checks: usize,
    pub avoided_cells: usize,
    pub br_cell_picks: usize,
    pub lower_br_picks: usize,
    pub upper_br_picks: usize,
    pub legacy_support_picks: usize,
    pub fair_cell_picks: usize,
}

struct Pending {
    script: Vec<usize>,
    mass: f64,
}

/// The tag is part of identity: a raw exact fingerprint and a semantic
/// fingerprint are never compared even if their 128-bit payloads coincide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum NodeKey {
    Exact(u128),
    NoDamageBookkeeping(u128),
}

struct Cell {
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

struct StageWitness {
    xlo: Vec<f64>,
    ylo: Vec<f64>,
    xhi: Vec<f64>,
    yhi: Vec<f64>,
    gap_lo: f64,
    gap_hi: f64,
}

struct Node {
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
}

pub struct BoundSolver<'d> {
    dex: &'d Dex,
    pub cfg: BoundConfig,
    nodes: HashMap<NodeKey, Node>,
    closed: HashMap<NodeKey, Bounds>,
    roots: HashSet<NodeKey>,
    active_stall: Option<OneSidedHeal>,
    pub stats: BoundStats,
}

const TOTAL: f64 = (1u64 << 32) as f64;

impl<'d> BoundSolver<'d> {
    pub fn new(dex: &'d Dex, cfg: BoundConfig) -> Self {
        BoundSolver {
            dex,
            cfg,
            nodes: HashMap::new(),
            closed: HashMap::new(),
            roots: HashSet::new(),
            active_stall: None,
            stats: BoundStats::default(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn closed_count(&self) -> usize {
        self.closed.len()
    }

    /// Tighten the root's certified interval until a stop condition fires.
    /// `tau` = optional threshold band (lo, hi) for certificate mode.
    /// Never discards work: repeated calls resume where the graph stands.
    pub fn solve(&mut self, b: &Battle, tau: Option<(f64, f64)>) -> SolveReport {
        self.active_stall = self
            .cfg
            .monotone_stall_scheduling
            .then(|| classify_one_sided_heal(b, self.dex).ok())
            .flatten();
        if self.active_stall.is_some() {
            self.stats.monotone_roots += 1;
        }
        let lookup = self.node_key(b);
        if let Some(&bounds) = self.closed.get(&lookup) {
            return SolveReport {
                bounds,
                stop: Stop::WidthMet,
                runs: 0,
            };
        }
        let root = self.intern(b.clone());
        self.roots.insert(root);
        let start_runs = self.stats.runs;
        let work_limit = start_runs + self.cfg.work_budget;
        let mut idle = 0usize;
        loop {
            if self.cfg.fold_closed_nodes && self.nodes.len() >= self.cfg.node_budget {
                self.fold_closed_graph();
            }
            let n = &self.nodes[&root];
            let bounds = Bounds { lo: n.lo, hi: n.hi };
            let stop = if bounds.width() <= self.cfg.eps {
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
            let stop = stop.or(if self.stats.runs >= work_limit {
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
                };
            }
            if self.trial(root) {
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

    fn node_key(&self, b: &Battle) -> NodeKey {
        if self.cfg.dead_damage_quotient && !b.damage_bookkeeping_observable(self.dex) {
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
        if let Some(node) = self.nodes.get_mut(&key) {
            if stall_rank.is_some() {
                node.stall_rank = stall_rank;
            }
            return key;
        }
        let node = match b.outcome() {
            Some(o) => {
                let v = match o {
                    Outcome::P1Win => 1.0,
                    Outcome::Tie => 0.5,
                    Outcome::P2Win => 0.0,
                };
                Node {
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
                }
            }
            None => Node {
                battle: Some(b),
                acts0: vec![],
                acts1: vec![],
                cells: vec![],
                lo: 0.0,
                hi: 1.0,
                strat: None,
                active_rows: Vec::new(),
                active_cols: Vec::new(),
                select_count: 0,
                fair_cursor: 0,
                stall_rank,
            },
        };
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
                probe
                    .legal_choices(self.dex, side)
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

    /// One trial: descend by LP-support × uncertainty, expand one pending
    /// chance script at the frontier, back up along the path.
    fn trial(&mut self, root: NodeKey) -> bool {
        self.stats.trials += 1;
        let mut did_work = false;
        let mut path: Vec<NodeKey> = Vec::new();
        let mut cur = root;
        for _ in 0..self.cfg.trial_depth {
            let n = &self.nodes[&cur];
            if n.hi - n.lo <= 1e-12 {
                break; // resolved (incl. terminals)
            }
            if n.cells.is_empty() {
                self.init_cells(cur);
            }
            path.push(cur);
            let ci = self.select_cell(cur);
            if self.try_eager(cur, ci) {
                did_work = true;
                break;
            }
            let cell = &self.nodes[&cur].cells[ci];
            // Expand pending mass while it dominates the cell's remaining
            // uncertainty; otherwise descend into the widest child term.
            let closure_mode = self.active_stall.is_some()
                && self.nodes.len().saturating_mul(4) >= self.cfg.node_budget.saturating_mul(3);
            let best_child = if closure_mode {
                let class = self.active_stall.as_ref().unwrap();
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
                None
            }
            .or_else(|| {
                cell.resolved
                    .iter()
                    .map(|&(m, k)| {
                        let c = &self.nodes[&k];
                        (m * (c.hi - c.lo), k)
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            });
            match best_child {
                Some((cw, k)) if cw > cell.pending_mass.max(self.cfg.descend_floor) => {
                    cur = k;
                    continue;
                }
                _ if !cell.pending.is_empty() => {
                    self.expand(cur, ci);
                    did_work = true;
                    break;
                }
                Some((_, k)) => {
                    // cell fully expanded but children below the descend
                    // floor: descend anyway — a zero-work trial would spin
                    // the solve loop forever (runs never advance).
                    cur = k;
                    continue;
                }
                None => break,
            }
        }
        for &k in path.iter().rev() {
            if !self.backup(k) {
                break; // bounds unchanged ⇒ ancestors can't change either
            }
        }
        did_work
    }

    /// Cell choice alternates the two certificate witnesses. For the lower
    /// bound, cross the lower row strategy with every tied pure best-response
    /// column. For the upper bound, cross every tied pure best-response row
    /// with the upper column strategy. Every 32nd visit is a round-robin fair
    /// pick, so zero-support cells cannot remain starved forever.
    fn select_cell(&mut self, key: NodeKey) -> usize {
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
                } else if node.select_count % 4 == 0 {
                    // Keep a sparse copy of the old union-support priority.
                    // It is redundant as a proof witness but empirically
                    // valuable on tiny lethal fans that close immediately.
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
                } else if node.select_count % 2 == 0 {
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

    /// First visit of a virgin cell: attempt the eager enumerator (range
    /// merge intact) within `cell_cap` runs; on success the whole cell
    /// resolves at once. Returns whether it did the work.
    fn try_eager(&mut self, key: NodeKey, ci: usize) -> bool {
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
        let step = enumerate_step(self.dex, &battle, choices, self.cfg.cell_cap);
        let Some(step) = step else {
            self.nodes.get_mut(&key).unwrap().battle = Some(battle);
            self.stats.runs += self.cfg.cell_cap; // the aborted probe's cost
            return false;
        };
        self.stats.runs += step.runs;
        self.stats.expansions += 1;
        let mut agg: HashMap<NodeKey, f64> = HashMap::new();
        let mut fixed_lo = 0.0;
        let mut fixed_hi = 0.0;
        for l in step.leaves {
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
            let k = self.intern(l.battle);
            *agg.entry(k).or_default() += prob;
        }
        self.nodes.get_mut(&key).unwrap().battle = Some(battle);
        let node = self.nodes.get_mut(&key).unwrap();
        let cell = &mut node.cells[ci];
        cell.resolved = agg.into_iter().map(|(k, m)| (m, k)).collect();
        cell.fixed_lo = fixed_lo;
        cell.fixed_hi = fixed_hi;
        cell.pending.clear();
        cell.pending_mass = 0.0;
        true
    }

    /// Expand the largest-mass pending script of one cell: one engine run
    /// yields the representative leaf; every non-default class after the
    /// prefix becomes a pending sibling at its exact mass. The leaf plus
    /// siblings partition the expanded mass exactly.
    fn expand(&mut self, key: NodeKey, ci: usize) {
        let (battle, choices, pending) = {
            let node = self.nodes.get_mut(&key).unwrap();
            let n1 = node.acts1.len();
            let (i, j) = (ci / n1, ci % n1);
            let choices = [node.acts0[i], node.acts1[j]];
            let cell = &mut node.cells[ci];
            let pi = cell
                .pending
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.mass.partial_cmp(&b.1.mass).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let pending = cell.pending.swap_remove(pi);
            (
                node.battle.take().expect("frontier node keeps its battle"),
                choices,
                pending,
            )
        };

        let (leaf, trace) = run_scripted(self.dex, &battle, choices, &pending.script);
        self.check_stall_edge(&battle, &leaf);
        self.nodes.get_mut(&key).unwrap().battle = Some(battle);
        self.stats.runs += 1;
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
            cell.resolved.push((leaf_mass, child.unwrap()));
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
            self.nodes.remove(&key);
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
    }
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
        }
    }

    #[test]
    fn support_br_scheduler_visits_tied_pure_response_and_fair_cursor() {
        let dex = conformance::load_dex();
        let key = NodeKey::Exact(7);
        let mut solver = BoundSolver::new(&dex, BoundConfig::default());
        // Next visit is lower-directed. Column 1 is a tied pure response to
        // xlo but has zero ylo support; the scheduler must still select it.
        solver.nodes.insert(key, synthetic_node(1, 0));
        assert_eq!(solver.select_cell(key), 1);
        assert_eq!(solver.stats.lower_br_picks, 1);

        // Every 32nd visit ignores policy scores and advances a deterministic
        // cursor over all unresolved active cells.
        solver.nodes.insert(key, synthetic_node(31, 2));
        assert_eq!(solver.select_cell(key), 2);
        assert_eq!(solver.stats.fair_cell_picks, 1);
    }
}
