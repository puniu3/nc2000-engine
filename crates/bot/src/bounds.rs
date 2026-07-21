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

use std::collections::HashMap;

use nc2000_engine::battle::enumerate::{enumerate_step, run_scripted};
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::exact::solve_matrix_full;

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
}

struct Pending {
    script: Vec<usize>,
    mass: f64,
}

struct Cell {
    resolved: Vec<(f64, u128)>,
    pending: Vec<Pending>,
    pending_mass: f64,
    lo: f64,
    hi: f64,
    /// One-shot eager enumeration already attempted for this cell.
    tried_eager: bool,
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
    strat: Option<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)>,
}

pub struct BoundSolver<'d> {
    dex: &'d Dex,
    pub cfg: BoundConfig,
    nodes: HashMap<u128, Node>,
    pub stats: BoundStats,
}

const TOTAL: f64 = (1u64 << 32) as f64;

impl<'d> BoundSolver<'d> {
    pub fn new(dex: &'d Dex, cfg: BoundConfig) -> Self {
        BoundSolver { dex, cfg, nodes: HashMap::new(), stats: BoundStats::default() }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Tighten the root's certified interval until a stop condition fires.
    /// `tau` = optional threshold band (lo, hi) for certificate mode.
    /// Never discards work: repeated calls resume where the graph stands.
    pub fn solve(&mut self, b: &Battle, tau: Option<(f64, f64)>) -> SolveReport {
        let root = self.intern(b.clone());
        let start_runs = self.stats.runs;
        let work_limit = start_runs + self.cfg.work_budget;
        let mut idle = 0usize;
        loop {
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
                return SolveReport { bounds, stop, runs: self.stats.runs - start_runs };
            }
            if self.trial(root) {
                idle = 0;
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
    pub fn peek(&self, key: u128) -> Option<Bounds> {
        self.nodes.get(&key).map(|n| Bounds { lo: n.lo, hi: n.hi })
    }

    fn intern(&mut self, b: Battle) -> u128 {
        let key = b.state_key128();
        if !self.nodes.contains_key(&key) {
            let node = match b.outcome() {
                Some(o) => {
                    let v = match o {
                        Outcome::P1Win => 1.0,
                        Outcome::Tie => 0.5,
                        Outcome::P2Win => 0.0,
                    };
                    Node { battle: None, acts0: vec![], acts1: vec![], cells: vec![], lo: v, hi: v, strat: None }
                }
                None => Node {
                    battle: Some(b),
                    acts0: vec![],
                    acts1: vec![],
                    cells: vec![],
                    lo: 0.0,
                    hi: 1.0,
                    strat: None,
                },
            };
            self.nodes.insert(key, node);
        }
        key
    }

    fn init_cells(&mut self, key: u128) {
        let node = self.nodes.get_mut(&key).unwrap();
        if !node.cells.is_empty() || node.battle.is_none() {
            return;
        }
        let mut probe = node.battle.as_ref().unwrap().clone();
        let needs = probe.needs_choice();
        let mut acts = |side: usize| -> Vec<Option<SearchChoice>> {
            if needs[side] {
                probe.legal_choices(self.dex, side).into_iter().map(Some).collect()
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
                pending: vec![Pending { script: vec![], mass: 1.0 }],
                pending_mass: 1.0,
                lo: 0.0,
                hi: 1.0,
                tried_eager: false,
            })
            .collect();
        node.acts0 = a0;
        node.acts1 = a1;
    }

    /// One trial: descend by LP-support × uncertainty, expand one pending
    /// chance script at the frontier, back up along the path.
    fn trial(&mut self, root: u128) -> bool {
        self.stats.trials += 1;
        let mut did_work = false;
        let mut path: Vec<u128> = Vec::new();
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
            let best_child = cell
                .resolved
                .iter()
                .map(|&(m, k)| {
                    let c = &self.nodes[&k];
                    (m * (c.hi - c.lo), k)
                })
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

    /// Cell choice: product of the lo/hi LP supports times cell width —
    /// spend work where the equilibria actually look.
    fn select_cell(&mut self, key: u128) -> usize {
        let node = &self.nodes[&key];
        let (n0, n1) = (node.acts0.len(), node.acts1.len());
        if n0 * n1 == 1 {
            return 0;
        }
        let widest = || -> usize {
            node.cells
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    (a.1.hi - a.1.lo).partial_cmp(&(b.1.hi - b.1.lo)).unwrap()
                })
                .map(|(i, _)| i)
                .unwrap()
        };
        if n0 == 1 || n1 == 1 {
            return widest();
        }
        let Some((xlo, ylo, xhi, yhi)) = node.strat.as_ref() else {
            return widest(); // no backup yet: probe the widest cell
        };
        let mut best = (f64::NEG_INFINITY, 0usize);
        for i in 0..n0 {
            for j in 0..n1 {
                let c = &node.cells[i * n1 + j];
                let w = xlo[i].max(xhi[i]) * ylo[j].max(yhi[j]) * (c.hi - c.lo);
                if w > best.0 {
                    best = (w, i * n1 + j);
                }
            }
        }
        if best.0 <= 0.0 {
            return widest();
        }
        best.1
    }

    /// First visit of a virgin cell: attempt the eager enumerator (range
    /// merge intact) within `cell_cap` runs; on success the whole cell
    /// resolves at once. Returns whether it did the work.
    fn try_eager(&mut self, key: u128, ci: usize) -> bool {
        let (battle, choices) = {
            let node = self.nodes.get_mut(&key).unwrap();
            let cell = &mut node.cells[ci];
            if cell.tried_eager || !cell.resolved.is_empty() {
                return false;
            }
            cell.tried_eager = true;
            let n1 = node.acts1.len();
            let (i, j) = (ci / n1, ci % n1);
            (node.battle.take().expect("frontier node keeps its battle"), [node.acts0[i], node.acts1[j]])
        };
        let step = enumerate_step(self.dex, &battle, choices, self.cfg.cell_cap);
        self.nodes.get_mut(&key).unwrap().battle = Some(battle);
        let Some(step) = step else {
            self.stats.runs += self.cfg.cell_cap; // the aborted probe's cost
            return false;
        };
        self.stats.runs += step.runs;
        self.stats.expansions += 1;
        let mut agg: HashMap<u128, f64> = HashMap::new();
        for l in step.leaves {
            let prob = l.prob;
            let k = self.intern(l.battle);
            *agg.entry(k).or_default() += prob;
        }
        let node = self.nodes.get_mut(&key).unwrap();
        let cell = &mut node.cells[ci];
        cell.resolved = agg.into_iter().map(|(k, m)| (m, k)).collect();
        cell.pending.clear();
        cell.pending_mass = 0.0;
        true
    }

    /// Expand the largest-mass pending script of one cell: one engine run
    /// yields the representative leaf; every non-default class after the
    /// prefix becomes a pending sibling at its exact mass. The leaf plus
    /// siblings partition the expanded mass exactly.
    fn expand(&mut self, key: u128, ci: usize) {
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
            (node.battle.take().expect("frontier node keeps its battle"), choices, pending)
        };

        let (leaf, trace) = run_scripted(self.dex, &battle, choices, &pending.script);
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
                        let mut script: Vec<usize> =
                            trace[..p].iter().map(|t| t.chosen).collect();
                        script.push(c);
                        siblings.push(Pending { script, mass: prefix * cnt as f64 / TOTAL });
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

        let child = self.intern(leaf);
        let node = self.nodes.get_mut(&key).unwrap();
        let cell = &mut node.cells[ci];
        cell.pending_mass = (cell.pending_mass - pending.mass).max(0.0)
            + siblings.iter().map(|s| s.mass).sum::<f64>();
        cell.pending.extend(siblings);
        cell.resolved.push((leaf_mass, child));
    }

    /// Recompute one node's cell and node bounds from current children.
    /// Bounds are clamped monotone; resolved/fully-expanded nodes drop
    /// their Battle snapshot. Returns whether the node's bounds moved.
    fn backup(&mut self, key: u128) -> bool {
        let (n0, n1, cell_bounds): (usize, usize, Vec<(f64, f64)>) = {
            let node = &self.nodes[&key];
            let bounds = node
                .cells
                .iter()
                .map(|cell| {
                    let mut lo = 0.0;
                    let mut hi = cell.pending_mass;
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
        let mut strat = None;
        let (lo, hi) = if n0 == 1 && n1 == 1 {
            (mlo[0], mhi[0])
        } else if n1 == 1 {
            (
                mlo.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                mhi.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            )
        } else if n0 == 1 {
            (
                mlo.iter().cloned().fold(f64::INFINITY, f64::min),
                mhi.iter().cloned().fold(f64::INFINITY, f64::min),
            )
        } else {
            self.stats.lp_solves += 2;
            let slo = solve_matrix_full(&mlo, n0, n1);
            let shi = solve_matrix_full(&mhi, n0, n1);
            self.stats.worst_gap = self.stats.worst_gap.max(slo.gap).max(shi.gap);
            let (vlo, vhi) = (slo.value, shi.value);
            strat = Some((slo.x, slo.y, shi.x, shi.y));
            (vlo, vhi)
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
            if strat.is_some() {
                node.strat = strat;
            }
            let moved = (node.lo - plo) > 1e-12 || (phi - node.hi) > 1e-12;
            (node.cells.iter().all(|c| c.pending.is_empty()), moved)
        };
        let node = self.nodes.get_mut(&key).unwrap();
        if node.hi - node.lo <= 1e-12 || fully_expanded {
            node.battle = None;
        }
        moved
    }
}
