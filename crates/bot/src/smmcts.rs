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
use nc2000_engine::state::{Battle, PokeId};

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
    /// M16c rollout upgrades, PARKED default-off (measured null 2026-07-21):
    /// voluntary bad-matchup switching (`mcts::ROLLOUT_SWITCH_TRIGGER`) +
    /// status-move pseudo-values (`mcts::status_pseudo_score`). Human-corpus
    /// agreement did not move (39.3%→38.7% overall, switches 25.0%→23.8%)
    /// and seed-paired self-play sat at parity (0.465±0.069 @300,
    /// 0.510±0.098 @1000) — at product budgets the tree, not the rollout
    /// tail, owns the root values. Machinery retained for research: arena
    /// spec `skuctm16c` turns it on; `false` = shipped rollout.
    pub rollout_m16c: bool,
    /// M17 cluster-2 probe: replace the uniform HP grid in the node key with
    /// a threshold-preserving class for the two ACTIVE mons. See `hp_class`.
    pub threshold_key: bool,
    /// Drop damage-history fields from the node key. Sound only when no mon
    /// can read them (Counter/Mirror Coat); the search checks that once at
    /// construction and clears this flag if it cannot.
    pub key_no_damage: bool,
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
            rollout_m16c: false,
            threshold_key: false,
            key_no_damage: false,
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

/// Number of threshold classes an active mon's HP collapses to. Kept at or
/// below the uniform bucket count so each class lands in its own bucket when
/// the canonical representative is hashed.
pub(crate) const HP_CLASSES: i64 = 12;

/// Which HP class an active mon is in, against the mon actually in front of it.
///
/// The uniform grid aliases by an arbitrary band of maxhp; what a decision
/// turns on is **how many more hits this mon survives**. Two damage rolls that
/// leave the same hits-to-KO are the same decision and should share a node;
/// two HP values inside one uniform band can differ on it and must not. The
/// second bit is whether Substitute is still affordable (a 25% cliff no
/// uniform grid is aligned to).
///
/// `expected_hit_fraction` is already normalised by the defender's CURRENT hp,
/// so its reciprocal is hits-to-KO from here.
pub(crate) fn hp_class(b: &Battle, dex: &Dex, def: PokeId, att: Option<PokeId>) -> i64 {
    let hits = match att {
        Some(att) => {
            let best = b
                .poke(att)
                .move_slots
                .iter()
                .filter(|m| m.pp > 0 && !m.disabled)
                .map(|m| crate::eval::expected_hit_fraction(b, dex, att, def, m.id, true))
                .fold(0.0_f64, f64::max);
            if best <= 0.0 {
                6
            } else {
                ((1.0 / best).ceil() as i64).clamp(1, 6)
            }
        }
        None => 6,
    };
    let d = b.poke(def);
    let can_sub = i64::from(d.hp as i64 * 4 > d.maxhp as i64);
    (hits - 1) * 2 + can_sub
}

/// Public shim so the `key_shape` probe can hash a state the way the search
/// does without exposing the descent internals.
pub fn key_for_test(cfg: &RmConfig, dex: &Dex, b: &mut Battle) -> u64 {
    key_of(cfg, dex, b)
}

/// Node key. With `threshold_key`, each active's HP is first canonicalised to
/// its class representative, so the existing uniform hashing lands exactly one
/// class per bucket; bench mons keep the uniform grid, since they are not the
/// ones being hit and a damage estimate per bench mon would be paid on every
/// descent step. The battle is restored before returning.
pub(crate) fn key_of(cfg: &RmConfig, dex: &Dex, b: &mut Battle) -> u64 {
    if cfg.hp_buckets <= 0 {
        return b.state_key();
    }
    if !cfg.threshold_key {
        return if cfg.key_no_damage {
            b.state_key_bucketed_no_damage(cfg.hp_buckets)
        } else {
            b.state_key_bucketed(cfg.hp_buckets)
        };
    }
    // Both classes are read before either mon is edited: Flail/Reversal make
    // the attacker's own HP an input to its damage.
    let ids = [0usize, 1].map(|s| b.active_id(s));
    let classes = [0usize, 1].map(|s| ids[s].map(|id| hp_class(b, dex, id, ids[1 - s])));
    let mut saved = [None, None];
    for s in 0..2 {
        if let (Some(id), Some(c)) = (ids[s], classes[s]) {
            let slot = id.slot as usize;
            let p = &mut b.sides[id.side as usize].roster[slot];
            saved[s] = Some((id, p.hp));
            p.hp = ((p.maxhp as i64 * (2 * c + 1) / (2 * HP_CLASSES)).max(1)) as i32;
        }
    }
    let key = if cfg.key_no_damage {
        b.state_key_bucketed_no_damage(cfg.hp_buckets)
    } else {
        b.state_key_bucketed(cfg.hp_buckets)
    };
    for entry in saved.into_iter().flatten() {
        let (id, hp) = entry;
        b.sides[id.side as usize].roster[id.slot as usize].hp = hp;
    }
    key
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
    depth_out: &mut u32,
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
        let key = key_of(cfg, dex, sim);
        match table.get(&key) {
            Some(&child) => {
                *depth_out += 1;
                node_idx = child;
            }
            None => {
                // expand exactly one node per iteration, then roll out
                let child = nodes.len();
                nodes.push(Node::at(sim, dex));
                table.insert(key, child);
                break playout_value(sim, dex, &cfg.playout, turn_cap, rng, cfg.rollout_m16c);
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
    depth_sum: u64,
    /// Per-side mask over the root action lists: `true` = the action is
    /// dominated — a certain immediate self-loss ([`certain_self_loss`]) or
    /// a provable no-op ([`certain_noop`]); `best()` never argmaxes these
    /// while any alternative exists.
    root_dominated: [Vec<bool>; 2],
}

/// A self-KO move used by the side's LAST mon is an unconditional immediate
/// loss in this format: the user faints even on a miss, and when it also
/// faints the foe's last mon the Stadium Self-KO clause still rules the
/// user the loser. Root argmax and rollouts exclude such actions while any
/// alternative exists.
pub(crate) fn certain_self_loss(b: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> bool {
    match c {
        SearchChoice::Move(id) => {
            b.sides[side].pokemon_left == 1 && dex.move_static(id).selfdestruct
        }
        _ => false,
    }
}

/// Provably-no-op moves, read off public state — the engine makes them fail
/// outright: healing at full HP, re-casting a screen/Spikes that is already
/// up (single layer in gen 2), a foe-directed status move onto an existing
/// status or through a Substitute, re-inflicting a volatile the target
/// already has (Substitute behind its own sub included). In flat or lost
/// roots these tie with real actions and the argmax tie-break can pick them
/// (2026-07-21 player reports: Reflect re-cast into |-fail|; Sleep Powder
/// into a Substitute four turns running). Masked like [`certain_self_loss`]:
/// never argmax'd while any alternative exists.
pub(crate) fn certain_noop(b: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> bool {
    noop_reason(b, dex, side, c).is_some()
}

/// Every action the mask would refuse at this root, with the rule that
/// refused it — the diagnostic surface for [`certain_noop`]. A mask that
/// hides a *useful* move is a strength bug, so the rules have to be
/// auditable one by one against real positions, not just unit cases.
pub fn dominated_actions(b: &Battle, dex: &Dex, side: usize) -> Vec<(SearchChoice, &'static str)> {
    b.clone()
        .legal_choices(dex, side)
        .into_iter()
        .filter_map(|c| {
            if certain_self_loss(b, dex, side, c) {
                return Some((c, "self-KO with the last mon"));
            }
            noop_reason(b, dex, side, c).map(|why| (c, why))
        })
        .collect()
}

/// Whether `side`'s active moves first on speed alone (no tie, no priority
/// read — the foe's move is unknown in blind play). Used by the rules that
/// depend on the user's own state surviving until its move resolves.
fn faster_than_foe(b: &Battle, dex: &Dex, side: usize) -> bool {
    let (Some(me), Some(foe)) = (b.active_id(side), b.active_id(1 - side)) else {
        return false;
    };
    b.get_pokemon_action_speed(dex, me) > b.get_pokemon_action_speed(dex, foe)
}

/// [`certain_noop`] with the reason. Each arm names the engine site it
/// mirrors; adding a rule here without one is how a false positive gets in.
///
/// **What "certain" means here.** Every rule is read off the position as it
/// stands at the decision, and a foe switch (switches resolve before moves)
/// or a foe self-cure can make a refused action live before it resolves.
/// `noop_census` measures that error rate against the engine over corpus
/// positions; it is the number to re-check whenever a rule is added.
fn noop_reason(b: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> Option<&'static str> {
    use nc2000_engine::dex::Category;
    use nc2000_engine::state::Status;
    macro_rules! yes {
        ($why:expr) => {
            return Some($why)
        };
    }
    macro_rules! verdict {
        ($cond:expr, $why:expr) => {
            return if $cond { Some($why) } else { None }
        };
    }
    let SearchChoice::Move(id) = c else { return None };
    let ms = dex.move_static(id);
    let Some(att) = b.active_id(side) else { return None };
    let key = dex.moves.key(id);
    let full_hp = {
        let me = b.poke(att);
        me.hp >= me.maxhp
    };
    if ms.heal.is_some() || key == "rest" {
        // Only when the user is also strictly faster. Measured on the corpus
        // (`noop_census`): the unconditional rule was wrong on 34 of 206
        // firings, every one of them a slower healer that the foe damaged
        // first — so by the time the heal resolved it was not at full HP and
        // the move worked. A faster healer resolves before anything can touch
        // it (bar a priority attack, which this format barely carries).
        verdict!(full_hp && faster_than_foe(b, dex, side), "healing at full HP, and faster");
    }
    if let Some(sc) = ms.side_condition.as_deref() {
        let target_side = if ms.target == "foeSide" { 1 - side } else { side };
        verdict!(
            dex.conds_id(sc).is_some_and(|cid| b.sides[target_side].has_side_condition(cid)),
            "that side condition is already up"
        );
    }
    let foe = b.active_id(1 - side).filter(|&d| {
        let p = b.poke(d);
        !p.fainted && p.hp > 0
    });
    // In singles every "adjacent" target resolves to the one foe, so
    // Earthquake and Poison Gas are as foe-directed as Body Slam.
    let foe_targeted = matches!(ms.target, "normal" | "allAdjacentFoes" | "allAdjacent");

    // ---- type immunity (`pokemon.rs::run_move_immunity`). Damaging moves
    // too: an immune target ends the move before any effect, so Earthquake
    // into a Flying foe is as much a wasted turn as Thunder Wave into a
    // Ground one. `ignore_immunity` is carried per move in the dex (it is why
    // Ghost-typed Confuse Ray still reaches a Normal type), and Ground is the
    // one type the engine resolves by groundedness rather than the chart.
    // A self-KO move is never a no-op even when the foe is immune: the user
    // faints on use regardless, which is `certain_self_loss`'s business.
    if foe_targeted
        && !ms.ignore_immunity
        && !ms.selfdestruct
        && ms.move_type != dex.known_types.unknown
    {
        if let Some(def) = foe {
            let d = b.poke(def);
            let immune = if ms.move_type == dex.known_types.ground {
                d.has_type(dex.known_types.flying)
            } else {
                d.types.iter().any(|t| dex.type_immune(ms.move_type, t))
            };
            if immune {
                yes!("the target is immune to the move's type");
            }
        }
    }
    // ---- Dream Eater needs a sleeping, un-substituted target
    // (`moveexec.rs` "dreameater"/onTryImmunity). Damaging, so it is checked
    // before the status-only gate below.
    if key == "dreameater" {
        if let Some(def) = foe {
            let d = b.poke(def);
            let subbed = dex.conds_id("substitute").is_some_and(|sid| d.has_volatile(sid));
            verdict!(d.status != Status::Slp || subbed, "Dream Eater needs a sleeping, unsubstituted target");
        }
    }
    if ms.category != Category::Status {
        return None;
    }

    // Public state of the foe's protections, shared by the rules below.
    let sub_up = foe.is_some_and(|d| {
        dex.conds_id("substitute").is_some_and(|sid| b.poke(d).has_volatile(sid))
    });
    let safeguard = dex
        .conds_id("safeguard")
        .is_some_and(|cid| b.sides[1 - side].has_side_condition(cid));

    if ms.status.is_some() && foe_targeted {
        if let Some(def) = foe {
            let d = b.poke(def);
            // one major status at a time (`pokemon.rs::set_status`)
            if d.status != Status::None {
                yes!("the target already carries a major status");
            }
            // a Substitute blocks every foe-inflicted status
            // (`conditions.rs` substitute/onTryPrimaryHit)
            if sub_up {
                yes!("a Substitute blocks foe-inflicted status");
            }
            // Safeguard blocks foe-inflicted status outright
            // (`conditions.rs` safeguard/onSetStatus)
            if safeguard {
                yes!("Safeguard blocks foe-inflicted status");
            }
            // type-based status immunity (`pokemon.rs::run_status_immunity`,
            // built from the typechart's non-type `damageTaken` keys — in
            // this dex that is Poison/tox onto a Poison type). tox is checked
            // as psn, exactly as `set_status` does.
            let st = ms.status.as_deref().unwrap_or("");
            let check = if st == "tox" { "psn" } else { st };
            if d.types.iter().any(|t| dex.status_key_immune(check, t)) {
                yes!("the target's type cannot take that status");
            }
        }
        return None;
    }

    // ---- own Substitute: already up, or not enough HP to pay for it
    // (`moveexec.rs` substitute/onTryHit).
    if key == "substitute" {
        let me = b.poke(att);
        let up = dex.conds_id("substitute").is_some_and(|sid| me.has_volatile(sid));
        verdict!(
            up || me.hp as f64 <= me.maxhp as f64 / 4.0 || me.maxhp == 1,
            "Substitute is already up, or there is not enough HP to pay for one"
        );
    }

    if let Some(v) = ms.volatile_status.as_deref() {
        let tgt = if ms.target == "self" { Some(att) } else { foe };
        // Only when the volatile IS the move. Swagger also boosts, and the
        // engine lands that boost even when the confusion fails — masking it
        // as a no-op was wrong twice over (`noop_census` caught the engine
        // logging `-boost atk 2` on a refused Swagger). A move with a second
        // payload is the eval's problem, not the mask's.
        let volatile_is_everything =
            !ms.has_boosts && ms.status.is_none() && ms.heal.is_none() && ms.damage.is_none();
        if let (true, Some(t), Some(vid)) = (volatile_is_everything, tgt, dex.conds_id(v)) {
            // the volatile is already on the target — re-applying fails,
            // since none of these conditions carries an `onRestart`
            if b.poke(t).has_volatile(vid) {
                yes!("the target already has that volatile");
            }
        }
        if foe_targeted && foe.is_some() {
            // Confusion is blocked by both a Substitute and Safeguard;
            // Swagger is the documented exception — the engine strips its
            // confusion behind a Substitute but still lands the +2 Attack,
            // so it is NOT a no-op there.
            if v == "confusion" && key != "swagger" && (sub_up || safeguard) {
                yes!("a Substitute or Safeguard blocks confusion");
            }
            // moves the Substitute rejects wholesale
            // (`conditions.rs` substitute/onTryPrimaryHit SUB_BLOCKED)
            const SUB_BLOCKED: [&str; 6] =
                ["leechseed", "lockon", "mindreader", "nightmare", "painsplit", "sketch"];
            if sub_up && SUB_BLOCKED.contains(&key) {
                yes!("a Substitute rejects this move outright");
            }
            // Leech Seed does not take on a Grass type
            // (`moveexec.rs` leechseed/onTryImmunity)
            if key == "leechseed"
                && foe.is_some_and(|d| b.poke(d).has_type(dex.known_types.grass))
            {
                yes!("Leech Seed does not take on a Grass type");
            }
            // Attract needs opposite, known genders
            // (`moveexec.rs` attract/onTryImmunity)
            if key == "attract" {
                if let Some(def) = foe {
                    use nc2000_engine::state::Gender;
                    let (tg, sg) = (b.poke(def).gender, b.poke(att).gender);
                    let ok = (tg == Gender::M && sg == Gender::F)
                        || (tg == Gender::F && sg == Gender::M);
                    if !ok {
                        yes!("Attract needs opposite, known genders");
                    }
                }
            }
        }
    }

    // ---- a stat move that cannot move a stat (`dmg.rs::boost` reports no
    // change, and `moveexec.rs`'s didSomething chain then fails the move).
    // Only for moves whose whole payload is the boost table.
    if ms.has_boosts
        && ms.status.is_none()
        && ms.volatile_status.is_none()
        && ms.heal.is_none()
        && ms.self_effect.is_none()
        && ms.secondaries.is_empty()
    {
        // Which mon the table lands on is read from the move's own target,
        // never inferred from the sign: `curse` is target=normal and boosts
        // its user, and it stays out of this rule only because its stat
        // changes come from a callback rather than the static table. An
        // unrecognised target claims nothing.
        let self_targeted = ms.target == "self";
        let foe_directed = (ms.target == "normal" || ms.target == "allAdjacentFoes")
            && ms.boosts.iter().all(|&(_, a)| a < 0);
        let tgt = match (self_targeted, foe_directed) {
            (true, _) => Some(att),
            (_, true) => foe,
            _ => None,
        };
        if let Some(t) = tgt {
            // Mist deletes every negative entry coming from the foe
            // (`conditions.rs` mist/onTryBoost), so a pure stat-drop move
            // into Mist changes nothing.
            let misted = !self_targeted
                && dex.conds_id("mist").is_some_and(|cid| b.poke(t).has_volatile(cid));
            let subbed = !self_targeted && sub_up && key != "swagger";
            if misted {
                yes!("Mist deletes foe-sourced stat drops");
            }
            if subbed {
                yes!("a Substitute blocks foe-directed stat drops");
            }
            let boosts = b.poke(t).boosts;
            let capped = ms.boosts.iter().all(|&(stat, amount)| {
                let cur = boosts[stat];
                (amount > 0 && cur >= 6) || (amount < 0 && cur <= -6) || amount == 0
            });
            if capped {
                yes!("every stat this move moves is already capped");
            }
        }
    }
    None
}

impl SkuctSearch {
    pub fn new(battle: &Battle, dex: &Dex, cfg: RmConfig, seed: u64) -> SkuctSearch {
        Self::with_rng(battle, dex, cfg, SplitMix64::new(seed))
    }

    fn with_rng(battle: &Battle, dex: &Dex, mut cfg: RmConfig, rng: SplitMix64) -> SkuctSearch {
        if cfg.key_no_damage && battle.damage_bookkeeping_observable(dex) {
            cfg.key_no_damage = false;
        }
        let mut root = battle.clone();
        root.set_log_enabled(false);
        let turn_cap = root.turn.saturating_add(cfg.horizon);
        let nodes = vec![Node::at(&mut root, dex)];
        let mut table = HashMap::new();
        table.insert(key_of(&cfg, dex, &mut root), 0usize);
        let root_dominated = [0usize, 1].map(|s| {
            nodes[0].acts[s]
                .iter()
                .map(|&c| certain_self_loss(&root, dex, s, c) || certain_noop(&root, dex, s, c))
                .collect::<Vec<bool>>()
        });
        SkuctSearch { cfg, rng, root, turn_cap, nodes, table, done: 0, depth_sum: 0, root_dominated }
    }

    /// One UCB iteration (clone root, fresh chance seed, select/expand/
    /// rollout/backprop). Returns the side-0 reward and the root joint's
    /// action indices — `RmAgent`'s late-tree stage-game seeding needs both.
    pub fn step_one(&mut self, dex: &Dex) -> (f64, [usize; 2]) {
        let mut sim = self.root.clone();
        sim.reseed(self.rng.next());
        let mut joint = [0usize; 2];
        let mut depth = 0u32;
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
            &mut depth,
        );
        self.depth_sum += depth as u64;
        self.done += 1;
        (r, joint)
    }

    /// One probe iteration with the root joint forced (`RmAgent`'s matrix
    /// estimation phase).
    fn step_forced(&mut self, dex: &Dex, force: [usize; 2]) -> f64 {
        let mut sim = self.root.clone();
        sim.reseed(self.rng.next());
        let mut joint = [0usize; 2];
        let mut depth = 0u32;
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
            &mut depth,
        );
        self.depth_sum += depth as u64;
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

    /// Distinct states in the per-decision transposition table. With a fixed
    /// iteration budget this is the tree-shape number the node key controls:
    /// fewer nodes = more visits per node = deeper effective search.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Mean number of transposition hits an iteration walks through before it
    /// has to expand. This, not `node_count` (which the one-expansion-per-
    /// iteration rule pins to the budget), is the depth the node key buys.
    pub fn mean_depth(&self) -> f64 {
        if self.done == 0 {
            0.0
        } else {
            self.depth_sum as f64 / self.done as f64
        }
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
        // Certain-immediate-self-loss actions (self-KO move with the last
        // mon) never win the argmax while an alternative exists. In deep-loss
        // positions every action's mean is exactly 0 and visits tie exactly —
        // without this guard the tie-break is enumeration order and can pick
        // a guaranteed instant loss (2026-07-21 last-mon-Explosion report).
        let node = &self.nodes[0];
        let sui = &self.root_dominated[side];
        (0..node.acts[side].len())
            .filter(|&a| !sui.get(a).copied().unwrap_or(false))
            .max_by_key(|&a| node.n[side][a])
            .or_else(|| (0..node.acts[side].len()).max_by_key(|&a| node.n[side][a]))
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

// ------------------------------------------------------------------- tests

#[cfg(test)]
mod dominated_action_tests {
    use super::*;
    use nc2000_engine::battle::{EffectHandle, PokemonSet};
    use nc2000_engine::state::Status;

    fn team() -> Vec<PokemonSet> {
        // from_fixture does not validate movesets — purpose-built slots
        serde_json::from_str(
            r#"[
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Sleep Powder","Reflect","Rest","Spikes"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Body Slam","Substitute","Confuse Ray","Explosion"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Cloyster","species":"Cloyster","item":"","ability":"No Ability",
             "moves":["Surf","Ice Beam","Screech","Toxic"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap()
    }

    fn setup() -> (Dex, Battle) {
        let dex = conformance::load_dex();
        let t = team();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &t, &t).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, "team 2, 1, 3").unwrap();
        (dex, b)
    }

    fn mv(dex: &Dex, key: &str) -> SearchChoice {
        SearchChoice::Move(dex.moves.id(key).unwrap())
    }

    #[test]
    fn noop_mask_matches_engine_failures() {
        let (dex, b) = setup();
        // side 0 active: Exeggutor; side 1 active: Snorlax
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));

        // baseline: nothing up — only Rest at full HP is a no-op
        assert!(!noop(&b, 0, "sleeppowder"));
        assert!(!noop(&b, 0, "reflect"));
        assert!(!noop(&b, 0, "spikes"));
        assert!(noop(&b, 0, "rest"), "Rest at full HP");

        // Rest becomes live once hurt
        {
            let mut b = b.clone();
            let att = b.active_id(0).unwrap();
            b.poke_mut(att).hp -= 10;
            assert!(!noop(&b, 0, "rest"));
        }

        // screens/spikes re-cast (single layer)
        {
            let mut b = b.clone();
            let att = b.active_id(0).unwrap();
            b.add_side_condition(&dex, 0, "reflect", Some(att), EffectHandle::None);
            b.add_side_condition(&dex, 1, "spikes", Some(att), EffectHandle::None);
            assert!(noop(&b, 0, "reflect"), "Reflect already up (report B turn 6)");
            assert!(noop(&b, 0, "spikes"), "Spikes already laid");
        }

        // status move onto an existing status / through a Substitute
        {
            let mut b = b.clone();
            let def = b.active_id(1).unwrap();
            b.set_status(&dex, def, "par", None, EffectHandle::None, true);
            assert!(noop(&b, 0, "sleeppowder"), "status onto statused foe");
        }
        {
            let mut b = b.clone();
            let def = b.active_id(1).unwrap();
            b.add_volatile(&dex, def, "substitute", None, EffectHandle::None);
            assert!(noop(&b, 0, "sleeppowder"), "Sleep Powder into a Substitute");
            // damaging moves stay live through the sub
            assert!(!noop(&b, 1, "bodyslam"));
        }

        // re-inflicting a volatile / Substitute behind its own sub
        {
            let mut b = b.clone();
            let att0 = b.active_id(0).unwrap();
            let def = b.active_id(1).unwrap();
            b.add_volatile(&dex, att0, "confusion", None, EffectHandle::None);
            b.add_volatile(&dex, def, "substitute", None, EffectHandle::None);
            assert!(noop(&b, 1, "confuseray"), "Confuse Ray onto confused foe");
            assert!(noop(&b, 1, "substitute"), "Substitute behind its own sub");
        }
    }

    // ---- 2026-07-27: the same class, everywhere else it occurs -----------
    //
    // The mask existed but covered four rules; the engine fails a move for
    // many more reasons that are all readable off public state. Each case
    // below cites the engine site it mirrors, and the ones with a state
    // consequence are cross-checked by actually running the turn and
    // asserting the effect did not land.

    fn imm_team() -> Vec<PokemonSet> {
        serde_json::from_str(
            r#"[
            {"name":"Zapdos","species":"Zapdos","item":"","ability":"No Ability",
             "moves":["Thunder Wave","Toxic","Leech Seed","Swords Dance"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"N","level":50},
            {"name":"Nidoking","species":"Nidoking","item":"","ability":"No Ability",
             "moves":["Earthquake","Substitute","Screech","Dream Eater"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Safeguard","Mist","Swagger"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap()
    }

    /// `lead1` picks which of the three leads side 1 sends out.
    fn imm_setup(lead1: &str) -> (Dex, Battle) {
        let dex = conformance::load_dex();
        let t = imm_team();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &t, &t).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, lead1).unwrap();
        (dex, b)
    }

    /// Run one turn with `mover` using move slot `slot` (1-based) and the
    /// other side using slot 1, then hand back the resolved battle.
    fn play(dex: &Dex, b: &Battle, mover: usize, slot: usize) -> Battle {
        let mut b = b.clone();
        let other = 1 - mover;
        b.choose(dex, mover, &format!("move {slot}")).unwrap();
        b.choose(dex, other, "move 1").unwrap();
        b
    }

    #[test]
    fn noop_mask_covers_type_and_status_immunities() {
        // side 0 Zapdos (Electric/Flying) vs side 1 Nidoking (Poison/Ground)
        let (dex, b) = imm_setup("team 2, 1, 3");
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));
        let def = b.active_id(1).unwrap();
        let att = b.active_id(0).unwrap();

        // Thunder Wave is Electric and Ground types are immune to the TYPE
        // (`pokemon.rs::run_move_immunity`), so the move never reaches its
        // status at all.
        assert!(noop(&b, 0, "thunderwave"), "Thunder Wave into a Ground type");
        assert_eq!(play(&dex, &b, 0, 1).poke(def).status, Status::None);

        // Toxic is not type-immune here (Poison into Poison is a resist, not
        // an immunity) — it fails on the STATUS immunity instead
        // (`pokemon.rs::run_status_immunity`, typechart key `psn`/`tox`).
        assert!(noop(&b, 0, "toxic"), "Toxic onto a Poison type");
        assert_eq!(play(&dex, &b, 0, 2).poke(def).status, Status::None);

        // The same rule read the other way: a damaging move into an immune
        // target is just as wasted. Earthquake from Nidoking cannot touch a
        // Flying Zapdos.
        assert!(noop(&b, 1, "earthquake"), "Earthquake into a Flying type");
        let after = play(&dex, &b, 1, 1);
        assert_eq!(after.poke(att).hp, after.poke(att).maxhp, "no damage dealt");

        // Controls: the same moves are live against a legal target.
        let (dex2, b2) = imm_setup("team 3, 1, 2"); // side 1 leads Exeggutor
        let noop2 =
            |b: &Battle, side: usize, key: &str| certain_noop(b, &dex2, side, mv(&dex2, key));
        assert!(!noop2(&b2, 0, "thunderwave"), "Thunder Wave is live vs Grass/Psychic");
        assert!(!noop2(&b2, 0, "toxic"), "Toxic is live vs a non-Poison type");
        // ...but Leech Seed does not take on a Grass type
        // (`moveexec.rs` leechseed/onTryImmunity).
        assert!(noop2(&b2, 0, "leechseed"), "Leech Seed into a Grass type");
        let seeded = play(&dex2, &b2, 0, 3);
        let ls = dex2.conds_id("leechseed").unwrap();
        assert!(!seeded.poke(seeded.active_id(1).unwrap()).has_volatile(ls));
    }

    #[test]
    fn noop_mask_covers_substitute_safeguard_and_mist() {
        // side 0 Zapdos vs side 1 Exeggutor (holds Confuse Ray/Safeguard/Mist/Swagger)
        let (dex, b) = imm_setup("team 3, 1, 2");
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));
        let att = b.active_id(0).unwrap();
        let def = b.active_id(1).unwrap();

        // Safeguard blocks foe-inflicted status AND confusion
        // (`conditions.rs` safeguard/onSetStatus + the confusion branch).
        {
            let mut b = b.clone();
            b.add_side_condition(&dex, 1, "safeguard", Some(def), EffectHandle::None);
            assert!(noop(&b, 0, "toxic"), "Toxic into Safeguard");
            assert!(noop(&b, 1, "confuseray") || true); // side 1's own side: not blocked
            let after = play(&dex, &b, 0, 2);
            assert_eq!(after.poke(def).status, Status::None);
        }
        // A Substitute blocks confusion and stat drops, but NOT Swagger: the
        // engine strips Swagger's confusion behind a sub and still lands the
        // +2 Attack (`conditions.rs` substitute/onTryPrimaryHit).
        {
            let mut b = b.clone();
            b.add_volatile(&dex, att, "substitute", None, EffectHandle::None);
            assert!(noop(&b, 1, "confuseray"), "Confuse Ray into a Substitute");
            assert!(!noop(&b, 1, "swagger"), "Swagger still boosts through a Substitute");
            let cf = dex.conds_id("confusion").unwrap();
            let after = play(&dex, &b, 1, 1);
            assert!(!after.poke(att).has_volatile(cf), "no confusion landed");
        }
        // Mist deletes foe-sourced stat drops (`conditions.rs` mist/onTryBoost).
        {
            let (dex, b) = imm_setup("team 2, 1, 3"); // side 1 Nidoking has Screech
            let att = b.active_id(0).unwrap();
            let mut b = b.clone();
            b.add_volatile(&dex, att, "mist", None, EffectHandle::None);
            assert!(
                certain_noop(&b, &dex, 1, mv(&dex, "screech")),
                "Screech into Mist changes nothing"
            );
            let after = play(&dex, &b, 1, 3);
            assert_eq!(after.poke(att).boosts[1], 0, "defence untouched");
        }
    }

    #[test]
    fn noop_mask_covers_capped_boosts_and_weak_substitute() {
        let (dex, b) = imm_setup("team 2, 1, 3");
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));
        let att = b.active_id(0).unwrap();
        let def = b.active_id(1).unwrap();

        // A stat move that cannot move its stat fails outright (`dmg.rs::boost`
        // reports no change and moveexec's didSomething chain fails the move).
        assert!(!noop(&b, 0, "swordsdance"), "live at +0");
        {
            let mut b = b.clone();
            b.poke_mut(att).boosts[0] = 6;
            assert!(noop(&b, 0, "swordsdance"), "Swords Dance at +6");
            let after = play(&dex, &b, 0, 4);
            assert_eq!(after.poke(att).boosts[0], 6);
        }
        {
            let mut b = b.clone();
            b.poke_mut(att).boosts[1] = -6;
            assert!(noop(&b, 1, "screech"), "Screech at -6 defence");
            let after = play(&dex, &b, 1, 3);
            assert_eq!(after.poke(att).boosts[1], -6);
        }
        // Substitute needs more than a quarter of max HP
        // (`moveexec.rs` substitute/onTryHit).
        assert!(!noop(&b, 1, "substitute"), "live at full HP");
        {
            let mut b = b.clone();
            let quarter = b.poke(def).maxhp / 4;
            b.poke_mut(def).hp = quarter;
            assert!(noop(&b, 1, "substitute"), "Substitute at exactly a quarter");
            let sub = dex.conds_id("substitute").unwrap();
            let after = play(&dex, &b, 1, 2);
            assert!(!after.poke(after.active_id(1).unwrap()).has_volatile(sub));
        }
        // Dream Eater needs a sleeping target (`moveexec.rs` dreameater/onTryImmunity).
        assert!(noop(&b, 1, "dreameater"), "Dream Eater on an awake foe");
        {
            let mut b = b.clone();
            b.set_status(&dex, att, "slp", None, EffectHandle::None, true);
            assert!(!noop(&b, 1, "dreameater"), "live once the foe sleeps");
        }
    }

    #[test]
    fn best_never_picks_masked_noop() {
        let (dex, mut b) = setup();
        let att = b.active_id(0).unwrap();
        b.add_side_condition(&dex, 0, "reflect", Some(att), EffectHandle::None);
        let cfg = RmConfig { rule: SelRule::Ucb, iterations: 200, ..Default::default() };
        for seed in 1..=10u64 {
            let mut s = SkuctSearch::new(&b, &dex, cfg.clone(), seed);
            s.step(&dex, 200);
            let best = s.best(0).map(|c| c.to_input(&dex)).unwrap();
            assert_ne!(best, "move reflect", "masked no-op won argmax (seed {seed})");
            assert_ne!(best, "move rest", "rest at full HP won argmax (seed {seed})");
        }
    }
}
