//! M7 mixed strategies: state-keyed tree + regret-matching equilibrium at
//! the root simultaneous decision, root play sampling the average strategy.
//!
//! Three pieces, each earning its place by measurement (all numbers: 60
//! games vs maxdamage, seed 11):
//!
//! 1. **State-keyed tree.** Nodes are keyed by `Battle::state_key_bucketed`
//!    in a per-decision transposition table, so chance outcomes that differ
//!    in anything discrete (KOs, status procs, request kinds, volatile
//!    durations) get their own nodes instead of aliasing into one
//!    joint-action edge like the M6 open-loop tree. A node therefore has a
//!    stable request kind and legal-action set (enumerated once, cached).
//!    HP is bucketed (default 16 per maxhp) because *exact* keys split every
//!    damage roll into its own node and starve the tree of depth: skuct:300
//!    scores 0.78 with buckets vs 0.73 exact. The turn counter is in the
//!    key, so the DAG is cycle-free.
//!
//! 2. **Decoupled UCB1 selection** (per side, over the node's cached
//!    actions) everywhere except the root's mixed-strategy computation —
//!    identical in spirit to M6, reproduced on the state-keyed tree at
//!    parity (skuct:300 0.78 ≈ mcts:300 0.82). Two RM-based selection rules
//!    were built first and rejected by measurement: online outcome-sampling
//!    RM at every node (0.30–0.43) and online RM at the root only (0.50–0.58
//!    even with argmax play) — the importance-weighted estimator `u/p`
//!    (spikes up to |A|/γ) plus the flat γ exploration tax never converge
//!    the root stage game at product budgets. RM+ had additionally to be
//!    dropped for plain RM online: regret clamping erases the negative
//!    memory that absorbs IS spikes (0.30 → ping-pong strategies).
//!
//! 3. **Root stage game estimated by dedicated probes, solved full-width,
//!    offline.** The budget splits into a *tree phase* (pure UCB — builds
//!    the tree and ranks root actions) and a *probe phase* (default 25%):
//!    round-robin over the joint cells of the **top-m root actions per
//!    side** (by visits, default m=3), each probe forcing that root joint
//!    and continuing with normal selection below. Probing after the tree
//!    matures matters: cell means taken over the whole search history are
//!    polluted by early exploration below the root, and the solver then
//!    ranks actions worse than plain UCB does (measured 0.44 vs 0.50
//!    against mcts:300 before this split; γ-uniform root exploration
//!    instead of probes was equally flat, and an EMA over sparse probes
//!    starved the estimate to 0.30). Cells are seeded with the *late half*
//!    of the tree phase's on-policy root joints — mature-tree samples that
//!    concentrate exactly on each side's best replies. The resulting m×m
//!    matrix is solved by **full-width RM+ with linear
//!    averaging** — a few thousand matrix sweeps, microseconds, zero
//!    sampling noise — and play **samples the average strategy**
//!    (thresholded to shed dust). When the purified solution is a point
//!    mass, play defers to argmax-visits: the matrix's job is deciding
//!    where to mix and with what weights, while a point prediction is
//!    better estimated by the visit statistics (hundreds of samples vs
//!    ~tens per cell) — this final rule took rm:1000 vs mcts:1000 from
//!    0.46 to 0.51. This is regret matching exactly where M7 wants it: a
//!    per-decision-point equilibrium approximation over UCB-quality
//!    continuation values; whole-game CFR stays out of scope.
//!
//! **Team preview stays UCB1 + argmax.** 120 ordered picks is outside any
//! sampled-equilibrium regime at these budgets (RM previews measured
//! 0.30–0.40 where UCB previews sit at 0.78+), and M8 bakes preview policy
//! offline anyway. In-battle simultaneous nodes (|A| ≤ ~13) are where the
//! per-turn mixed equilibria (switch-vs-attack, counter-vs-setup) live.
//!
//! Playouts and leaf eval reuse the M6 heavy machinery unchanged (ε-greedy
//! max-damage rollouts, truncation, weighted static eval).

use std::collections::HashMap;

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::mcts::{outcome_reward, playout_value, Playout};
use crate::rng::SplitMix64;

/// Selection rule for the root decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelRule {
    /// Solve the estimated root stage game with RM+, play the (thresholded)
    /// average strategy — the M7 mixed agent.
    Rm,
    /// Argmax visits at the root — the ablation that isolates tree keying
    /// from the equilibrium layer (and the frozen-argmax exploitability
    /// reference on the identical tree).
    Ucb,
}

#[derive(Clone, Debug)]
pub struct RmConfig {
    /// Simulations per decision.
    pub iterations: u32,
    /// Root behavior: RM-solved mixed play vs argmax.
    pub rule: SelRule,
    /// Fraction of the budget spent on root-matrix probes (the rest builds
    /// the tree with pure UCB first).
    pub probe: f64,
    /// Matrix support: top-m root actions per side by tree-phase visits.
    pub mix_actions: usize,
    /// UCB1 exploration constant.
    pub c: f64,
    /// Tree horizon in turns (same meaning as `MctsConfig::horizon`).
    pub horizon: u16,
    /// Rollout policy + leaf eval (shared with the M6 agent).
    pub playout: Playout,
    /// Play-time purification: root actions whose average-strategy
    /// probability is below `threshold × max_prob` are dropped and the rest
    /// renormalized before sampling. Sheds solver dust without flattening
    /// genuine mixing.
    pub threshold: f64,
    /// HP buckets for the node key (`Battle::state_key_bucketed`); 0 = exact
    /// keys (measured weaker — see the module doc).
    pub hp_buckets: i64,
    /// Full-width RM+ sweeps over the estimated root matrix.
    pub solve_sweeps: u32,
}

impl Default for RmConfig {
    fn default() -> Self {
        RmConfig {
            iterations: 1000,
            rule: SelRule::Rm,
            probe: 0.25,
            mix_actions: 3,
            c: 1.0,
            horizon: 100,
            playout: Playout::heavy(),
            threshold: 0.5,
            hp_buckets: 16,
            solve_sweeps: 2000,
        }
    }
}

pub(crate) struct Node {
    /// Legal actions per side at this state (empty = side owes nothing).
    pub(crate) acts: [Vec<SearchChoice>; 2],
    /// Per-action sample counts (UCB1).
    pub(crate) n: [Vec<u32>; 2],
    /// Per-action reward sums (UCB1).
    pub(crate) w: [Vec<f64>; 2],
    /// Team-preview node (always UCB1 + argmax).
    pub(crate) preview: bool,
}

impl Node {
    pub(crate) fn at(sim: &mut Battle, dex: &Dex) -> Node {
        let acts = [sim.legal_choices(dex, 0), sim.legal_choices(dex, 1)];
        let preview = acts
            .iter()
            .any(|a| matches!(a.first(), Some(SearchChoice::Team(_))));
        Node {
            n: [vec![0; acts[0].len()], vec![0; acts[1].len()]],
            w: [vec![0.0; acts[0].len()], vec![0.0; acts[1].len()]],
            preview,
            acts,
        }
    }
}

/// Probe statistics over the top-m×top-m root joint cells: the estimated
/// stage-game payoff matrix (side-0 perspective). Probes run against the
/// already-mature tree, so plain per-cell means are unbiased and every
/// sample counts (an EMA was tried and starved the estimate).
struct ProbeStats {
    /// Per-side probed action indices (into the root's action lists).
    support: [Vec<usize>; 2],
    n: Vec<u32>,
    v: Vec<f64>,
}

impl ProbeStats {
    fn new(support: [Vec<usize>; 2]) -> ProbeStats {
        let cells = support[0].len().max(1) * support[1].len().max(1);
        ProbeStats { support, n: vec![0; cells], v: vec![0.5; cells] }
    }

    fn dims(&self) -> [usize; 2] {
        [self.support[0].len().max(1), self.support[1].len().max(1)]
    }

    fn record(&mut self, cell: usize, reward0: f64) {
        self.n[cell] += 1;
        self.v[cell] += (reward0 - self.v[cell]) / self.n[cell] as f64;
    }
}

// ------------------------------------------------- search core (free fns)
//
// The iteration machinery lives in free functions so `RmAgent` (one-shot
// search per decision) and `SkuctSearch` (persistent, steppable — the M9
// wasm/ponder form) share it verbatim. Extracted mechanically from the M7
// `RmAgent` methods; bodies unchanged so agent behavior stays bit-identical
// (verified by the arena sanity run in M9a).

pub(crate) fn key_of(cfg: &RmConfig, b: &Battle) -> u64 {
    if cfg.hp_buckets > 0 {
        b.state_key_bucketed(cfg.hp_buckets)
    } else {
        b.state_key()
    }
}

/// UCB1 (untried-first, then mean + c·sqrt(ln N / n)).
pub(crate) fn select_ucb(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    node: &mut Node,
    side: usize,
) -> usize {
    let k = node.acts[side].len();
    let untried: Vec<usize> = (0..k).filter(|&a| node.n[side][a] == 0).collect();
    let pick = if !untried.is_empty() {
        untried[rng.below(untried.len())]
    } else {
        let total: u32 = node.n[side].iter().sum();
        let ln_total = (total as f64).ln();
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for a in 0..k {
            let (n, w) = (node.n[side][a] as f64, node.w[side][a]);
            let v = w / n + cfg.c * (ln_total / n).sqrt();
            if v > best_v {
                best_v = v;
                best = a;
            }
        }
        best
    };
    node.n[side][pick] += 1;
    pick
}

/// One iteration starting at node `start` (0 for the classic per-decision
/// tree; a per-determinization root for the M10b blind search). Per-side
/// `force_root` fixes that side's root action index instead of UCB
/// selection (the probe phase forces both sides; the blind search forces
/// only its own, globally-selected action); forced picks still feed the
/// root's per-action means (an unconditional sample of an action is an
/// unbiased sample of it), and everything below the root selects normally.
/// Returns the iteration's side-0 reward and writes the root joint's action
/// indices into `root_joint`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_iteration(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    nodes: &mut Vec<Node>,
    table: &mut HashMap<u64, usize>,
    sim: &mut Battle,
    dex: &Dex,
    turn_cap: u16,
    start: usize,
    force_root: [Option<usize>; 2],
    root_joint: &mut [usize; 2],
) -> f64 {
    let mut path: Vec<(usize, usize, usize)> = Vec::new(); // (node, side, act)
    let mut node_idx = start;

    // ---- selection until a leaf: terminal, horizon, or unexpanded state
    let reward0 = loop {
        let at_root = node_idx == start;
        let mut joint = [None, None];
        for s in 0..2 {
            let k = nodes[node_idx].acts[s].len();
            if k == 0 {
                continue;
            }
            let ai = if k == 1 {
                // forced: skip the stats machinery
                0
            } else {
                let ai = match force_root[s] {
                    Some(f) if at_root => {
                        nodes[node_idx].n[s][f] += 1;
                        f
                    }
                    _ => select_ucb(cfg, rng, &mut nodes[node_idx], s),
                };
                path.push((node_idx, s, ai));
                ai
            };
            joint[s] = Some(nodes[node_idx].acts[s][ai]);
            if at_root {
                root_joint[s] = ai;
            }
        }
        if joint == [None, None] {
            // defensive: a rest point where neither side owes a choice
            // (never reached in practice — battles end instead)
            break leaf_eval(cfg, sim, dex);
        }
        sim.apply_choices(dex, joint)
            .expect("cached legal choice rejected (state_key collision?)");
        if let Some(o) = sim.outcome() {
            break outcome_reward(o);
        }
        if sim.turn > turn_cap {
            break leaf_eval(cfg, sim, dex);
        }
        let key = key_of(cfg, sim);
        match table.get(&key) {
            Some(&child) => node_idx = child,
            None => {
                // expand exactly one node per iteration, then roll out
                let child = nodes.len();
                nodes.push(Node::at(sim, dex));
                table.insert(key, child);
                break playout_value(sim, dex, &cfg.playout, turn_cap, rng);
            }
        }
    };

    // ---- backprop: UCB stats along the path
    for (ni, s, ai) in path {
        nodes[ni].w[s][ai] += if s == 0 { reward0 } else { 1.0 - reward0 };
    }
    reward0
}

fn leaf_eval(cfg: &RmConfig, sim: &Battle, dex: &Dex) -> f64 {
    match &cfg.playout {
        Playout::Uniform => crate::mcts::hp_eval(sim),
        Playout::Heavy { weights, .. } => crate::eval::eval_leaf(sim, dex, weights),
    }
}

// ---------------------------------------------------- stepped search (M9)

/// Persistent, incrementally steppable state-keyed UCB search over ONE
/// decision point — the `skuct` flagship in the form the wasm bridge's
/// ponder loop needs: create it at the current battle state, pump
/// `step(n)` in small slices (returning to the JS event loop between
/// slices), read `best()` / visit stats whenever the move is actually
/// wanted. `cfg.iterations` is ignored — the caller owns the budget.
///
/// `RmAgent` drives this same struct internally (tree phase = `step_one`,
/// probe phase = `step_forced`), so the stepped form can never drift from
/// the gate-measured agents.
pub struct SkuctSearch {
    cfg: RmConfig,
    rng: SplitMix64,
    root: Battle,
    turn_cap: u16,
    nodes: Vec<Node>,
    table: HashMap<u64, usize>,
    done: u32,
}

impl SkuctSearch {
    pub fn new(battle: &Battle, dex: &Dex, cfg: RmConfig, seed: u64) -> SkuctSearch {
        Self::with_rng(battle, dex, cfg, SplitMix64::new(seed))
    }

    fn with_rng(battle: &Battle, dex: &Dex, cfg: RmConfig, rng: SplitMix64) -> SkuctSearch {
        let mut root = battle.clone();
        root.set_log_enabled(false);
        let turn_cap = root.turn.saturating_add(cfg.horizon);
        let nodes = vec![Node::at(&mut root, dex)];
        let mut table = HashMap::new();
        table.insert(key_of(&cfg, &root), 0usize);
        SkuctSearch { cfg, rng, root, turn_cap, nodes, table, done: 0 }
    }

    /// One UCB iteration (clone root, fresh chance seed, select/expand/
    /// rollout/backprop). Returns the side-0 reward and the root joint's
    /// action indices — `RmAgent`'s late-tree stage-game seeding needs both.
    pub fn step_one(&mut self, dex: &Dex) -> (f64, [usize; 2]) {
        let mut sim = self.root.clone();
        sim.reseed(self.rng.next());
        let mut joint = [0usize; 2];
        let r = run_iteration(
            &self.cfg,
            &mut self.rng,
            &mut self.nodes,
            &mut self.table,
            &mut sim,
            dex,
            self.turn_cap,
            0,
            [None, None],
            &mut joint,
        );
        self.done += 1;
        (r, joint)
    }

    /// One probe iteration with the root joint forced (`RmAgent`'s matrix
    /// estimation phase).
    fn step_forced(&mut self, dex: &Dex, force: [usize; 2]) -> f64 {
        let mut sim = self.root.clone();
        sim.reseed(self.rng.next());
        let mut joint = [0usize; 2];
        let r = run_iteration(
            &self.cfg,
            &mut self.rng,
            &mut self.nodes,
            &mut self.table,
            &mut sim,
            dex,
            self.turn_cap,
            0,
            [Some(force[0]), Some(force[1])],
            &mut joint,
        );
        self.done += 1;
        r
    }

    /// Pump `n` iterations, return the total run so far.
    pub fn step(&mut self, dex: &Dex, n: u32) -> u32 {
        for _ in 0..n {
            self.step_one(dex);
        }
        self.done
    }

    pub fn iterations(&self) -> u32 {
        self.done
    }

    /// The root's legal actions for `side` (empty = side owes nothing).
    pub fn actions(&self, side: usize) -> &[SearchChoice] {
        &self.nodes[0].acts[side]
    }

    /// Per-action visit counts at the root, aligned with `actions`.
    pub fn visits(&self, side: usize) -> &[u32] {
        &self.nodes[0].n[side]
    }

    /// Per-action mean rewards (side's own perspective), 0.5 when unvisited.
    pub fn means(&self, side: usize) -> Vec<f64> {
        let node = &self.nodes[0];
        (0..node.acts[side].len())
            .map(|a| {
                if node.n[side][a] == 0 {
                    0.5
                } else {
                    node.w[side][a] / node.n[side][a] as f64
                }
            })
            .collect()
    }

    /// Current best choice: argmax visits (the `skuct` play rule). `None`
    /// when the side owes nothing at this decision point.
    pub fn best(&self, side: usize) -> Option<SearchChoice> {
        let node = &self.nodes[0];
        (0..node.acts[side].len())
            .max_by_key(|&a| node.n[side][a])
            .map(|a| node.acts[side][a])
    }

    /// Whether the root decision is a team preview.
    pub fn is_preview(&self) -> bool {
        self.nodes[0].preview
    }
}

// -------------------------------------------------------------- the agent

pub struct RmAgent {
    pub cfg: RmConfig,
    rng: SplitMix64,
}

impl RmAgent {
    pub fn new(cfg: RmConfig, seed: u64) -> Self {
        RmAgent { cfg, rng: SplitMix64::new(seed) }
    }

    /// Run the search and return the root play distribution for `side`
    /// (probabilities aligned with the root's legal actions, which equal the
    /// caller's `choices`).
    fn search(&mut self, battle: &Battle, dex: &Dex, side: usize) -> (Vec<SearchChoice>, Vec<f64>) {
        let mut ts = SkuctSearch::with_rng(battle, dex, self.cfg.clone(), self.rng.clone());

        let mixed_root = !ts.nodes[0].preview && self.cfg.rule == SelRule::Rm;
        let probes = if mixed_root {
            (self.cfg.iterations as f64 * self.cfg.probe).round() as u32
        } else {
            0
        };

        // ---- tree phase: pure UCB. The late half's root joints are kept:
        // they are on-policy samples from a tree mature enough to trust, and
        // they concentrate exactly on the cells the equilibrium cares about
        // most (each side's best replies).
        let tree_iters = self.cfg.iterations - probes;
        let k1_full = ts.nodes[0].acts[1].len().max(1);
        let cells_full = ts.nodes[0].acts[0].len().max(1) * k1_full;
        let mut late_n = vec![0u32; cells_full];
        let mut late_w = vec![0.0f64; cells_full];
        for i in 0..tree_iters {
            let (r, joint) = ts.step_one(dex);
            if mixed_root && i >= tree_iters / 2 {
                let cell = joint[0] * k1_full + joint[1];
                late_n[cell] += 1;
                late_w[cell] += r;
            }
        }

        let acts = ts.nodes[0].acts[side].clone();

        // preview root (or argmax ablation): most-visited action, point mass
        if !mixed_root {
            let best = (0..acts.len()).max_by_key(|&a| ts.nodes[0].n[side][a]).unwrap();
            let mut probs = vec![0.0; acts.len()];
            probs[best] = 1.0;
            self.rng = ts.rng;
            return (acts, probs);
        }

        // ---- probe phase: round-robin the top-m×top-m root joint cells,
        // seeded with the late-tree on-policy samples
        let support = [0, 1].map(|s| top_actions(&ts.nodes[0], s, self.cfg.mix_actions));
        let mut stats = ProbeStats::new(support);
        let [m0, m1] = stats.dims();
        for cell in 0..m0 * m1 {
            let a0 = stats.support[0].get(cell / m1).copied().unwrap_or(0);
            let a1 = stats.support[1].get(cell % m1).copied().unwrap_or(0);
            let full = a0 * k1_full + a1;
            if late_n[full] > 0 {
                stats.n[cell] = late_n[full];
                stats.v[cell] = late_w[full] / late_n[full] as f64;
            }
        }
        for i in 0..probes {
            let cell = (i as usize) % (m0 * m1);
            let force = [
                stats.support[0].get(cell / m1).copied().unwrap_or(0),
                stats.support[1].get(cell % m1).copied().unwrap_or(0),
            ];
            let r = ts.step_forced(dex, force);
            stats.record(cell, r);
        }
        self.rng = ts.rng.clone();

        // ---- solve the probed stage game, embed into the full action list
        let (s0, s1) = solve_rm_plus(&stats.v, stats.dims(), self.cfg.solve_sweeps);
        let mixed = if side == 0 { s0 } else { s1 };
        let mut probs = vec![0.0; acts.len()];
        for (j, &a) in stats.support[side].iter().enumerate() {
            probs[a] = mixed[j];
        }

        // purification: drop solver dust relative to the modal action
        // (the modal action itself always survives)
        let imax = (0..probs.len()).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap();
        let pmax = probs[imax];
        for (i, p) in probs.iter_mut().enumerate() {
            if i != imax && *p < self.cfg.threshold * pmax {
                *p = 0.0;
            }
        }
        let z: f64 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= z;
        }

        // solver-pure spot: the matrix's job is deciding where to mix and
        // with what weights; a point prediction is better estimated by the
        // visit statistics (hundreds of samples vs ~tens per matrix cell),
        // so a purified point mass defers to argmax-visits.
        if probs.iter().filter(|&&p| p > 0.0).count() == 1 {
            let best = (0..acts.len()).max_by_key(|&a| ts.nodes[0].n[side][a]).unwrap();
            probs.iter_mut().for_each(|p| *p = 0.0);
            probs[best] = 1.0;
        }
        (acts, probs)
    }

    fn sample(&mut self, acts: &[SearchChoice], probs: &[f64]) -> SearchChoice {
        let u = self.rng.next_f64();
        let mut acc = 0.0;
        for (a, p) in acts.iter().zip(probs) {
            acc += p;
            if u < acc {
                return *a;
            }
        }
        acts[acts.len() - 1]
    }
}

/// The side's top-`m` root actions by visit count (all of them when the
/// side owes ≤ m actions; empty when it owes none).
fn top_actions(root: &Node, side: usize, m: usize) -> Vec<usize> {
    let k = root.acts[side].len();
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| root.n[side][b].cmp(&root.n[side][a]));
    idx.truncate(m.max(1));
    idx
}

/// Full-width RM+ with linear averaging on the zero-sum stage game
/// `matrix` (side-0 payoff, side-1 payoff = 1 − u). Returns both sides'
/// average strategies. Public since M8: the preview baker solves the
/// offline meta matchup matrices with the same solver.
pub fn solve_rm_plus(matrix: &[f64], k: [usize; 2], sweeps: u32) -> (Vec<f64>, Vec<f64>) {
    let (k0, k1) = (k[0], k[1]);
    let mut r0 = vec![0.0f64; k0];
    let mut r1 = vec![0.0f64; k1];
    let mut s0 = vec![0.0f64; k0];
    let mut s1 = vec![0.0f64; k1];
    let strategy = |r: &[f64]| -> Vec<f64> {
        let total: f64 = r.iter().map(|v| v.max(0.0)).sum();
        if total > 1e-12 {
            r.iter().map(|v| v.max(0.0) / total).collect()
        } else {
            vec![1.0 / r.len() as f64; r.len()]
        }
    };
    for t in 1..=sweeps {
        let sig0 = strategy(&r0);
        let sig1 = strategy(&r1);
        let tw = t as f64;
        for a in 0..k0 {
            s0[a] += tw * sig0[a];
        }
        for b in 0..k1 {
            s1[b] += tw * sig1[b];
        }
        // side 0: expected payoff of each row vs σ1, and of σ0 itself
        let mut u0 = vec![0.0f64; k0];
        for a in 0..k0 {
            for b in 0..k1 {
                u0[a] += matrix[a * k1 + b] * sig1[b];
            }
        }
        let v0: f64 = (0..k0).map(|a| u0[a] * sig0[a]).sum();
        for a in 0..k0 {
            r0[a] = (r0[a] + u0[a] - v0).max(0.0); // RM+
        }
        // side 1: payoff 1 − u ⇒ minimizing u; regrets on (v0 − column value)
        let mut u1 = vec![0.0f64; k1];
        for b in 0..k1 {
            for a in 0..k0 {
                u1[b] += (1.0 - matrix[a * k1 + b]) * sig0[a];
            }
        }
        let v1: f64 = (0..k1).map(|b| u1[b] * sig1[b]).sum();
        for b in 0..k1 {
            r1[b] = (r1[b] + u1[b] - v1).max(0.0);
        }
    }
    let norm = |s: Vec<f64>| -> Vec<f64> {
        let z: f64 = s.iter().sum();
        s.into_iter().map(|v| v / z).collect()
    };
    (norm(s0), norm(s1))
}

impl Agent for RmAgent {
    fn name(&self) -> String {
        match self.cfg.rule {
            SelRule::Rm => format!(
                "rm:{}:{}:{}:{}",
                self.cfg.iterations, self.cfg.probe, self.cfg.threshold, self.cfg.hp_buckets
            ),
            SelRule::Ucb => {
                format!("skuct:{}:{}:{}", self.cfg.iterations, self.cfg.c, self.cfg.hp_buckets)
            }
        }
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        let (acts, probs) = self.search(battle, dex, side);
        debug_assert_eq!(acts.as_slice(), choices, "root action set drifted from caller's choices");
        self.sample(&acts, &probs)
    }

    /// The true play distribution (RM+-solved average strategy, thresholded)
    /// — the mixed policy the exploitability gate probes.
    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        if choices.len() == 1 {
            return vec![1.0];
        }
        let (acts, probs) = self.search(battle, dex, side);
        // align defensively even though acts == choices in practice
        choices
            .iter()
            .map(|c| acts.iter().position(|a| a == c).map_or(0.0, |i| probs[i]))
            .collect()
    }
}
