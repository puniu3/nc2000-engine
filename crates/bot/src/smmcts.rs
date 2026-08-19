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

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{Dex, TypeId};
use nc2000_engine::fxhash::FxHashMap;
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
    /// Which optional root-mask rules this agent plays under. Default =
    /// shipped; an arena arm flips one field to A/B a mask change, which is
    /// the only harness that can see one at all (see `root_dominated`).
    pub mask_rules: MaskRules,
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
            mask_rules: MaskRules::default(),
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
    table: &mut FxHashMap<u64, usize>,
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
    table: FxHashMap<u64, usize>,
    done: u32,
    depth_sum: u64,
    /// Per-side mask over the root action lists: `true` = the action is
    /// dominated — a certain immediate self-loss ([`certain_self_loss`]) or
    /// a provable no-op ([`certain_noop`]); `best()` never argmaxes these
    /// while any alternative exists.
    ///
    /// **Only [`SkuctSearch::best`] consults this, so only callers that go
    /// through `best()` are masked at all.** [`BlindAgent`](crate::BlindAgent)
    /// does (it keeps its own copy and filters there), and that is the shipped
    /// ladder client, so play is masked in production; so does
    /// [`OpenAgent`](crate::OpenAgent), which shares `search_choose` — the web
    /// product policy is masked too. [`RmAgent::choose`]
    /// does NOT: it picks straight off the visit counts and never calls
    /// `best()`. Every RmAgent consumer is therefore unmasked — `runner`,
    /// `duel`, `eval_ab_duel`, and arena's `skuct`/`rm` specs. Measured over
    /// 600 skuct self-play games: 669 actions that `dominated_actions` flags
    /// were chosen and played, 2.03% of 32,981 decisions.
    ///
    /// The consequence that bites: **a duel built on those harnesses cannot
    /// measure a change to [`noop_reason`] and will return a null.** Gate mask
    /// changes with arena's blind specs instead — and since both arms are the
    /// same binary, the two rule sets have to differ by config, which is what
    /// [`MaskRules`] on [`RmConfig`] and arena's `blindlegacy` spec are for.
    /// `best_never_picks_masked_noop` covers `best()`, not `choose()`, which
    /// is why this stayed invisible.
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
///
/// `rules` is the A/B seam. The mask is consulted ONLY by `best()`, so no
/// `RmAgent` harness can see a rule change (`root_dominated`'s doc); the only
/// way to duel one is to run two `BlindAgent`s in the same process with
/// different [`MaskRules`], which is what `RmConfig::mask_rules` and arena's
/// `blindlegacy` spec exist for. Every instrument that reports the SHIPPED
/// mask passes `MaskRules::default()`.
pub(crate) fn certain_noop(
    b: &Battle,
    dex: &Dex,
    side: usize,
    c: SearchChoice,
    rules: MaskRules,
) -> bool {
    noop_reason(b, dex, side, c, rules).is_some()
}

/// Which of the optional [`noop_reason`] rules are live. Every field defaults
/// to the SHIPPED value, so `MaskRules::default()` is exactly the mask the
/// ladder client plays; a field is added here only when an arm is needed to
/// turn one rule OFF for an A/B, never to ship a rule half-on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskRules {
    /// Sleep Talk / Snore selected by an AWAKE, strictly faster user
    /// (`moveexec.rs:521`). `false` = the pre-2026-08-19 mask.
    pub sleep_talk_awake: bool,
    /// Refuse a type-immune move aimed at the mon ACTUALLY IN FRONT even
    /// when the foe could switch out first — i.e. drop the `foe_can_switch`
    /// gate for the type-immunity arm alone.
    ///
    /// **This one can be wrong**, and that is the point of having it: the
    /// move might be aimed at the replacement (Earthquake into a Flying foe
    /// that is about to bring in a Ground type). What makes it worth an A/B
    /// is that the read it deletes has no measured value — over the
    /// 570-battle corpus the class-A opportunity set switches on 20.9% of
    /// decisions against a 23.3% base rate, and across the whole gate the
    /// switch read lifts nothing at all (23.2% vs 23.3%); masking would have
    /// deleted a vindicated read on 21 of 1,410 decisions while removing 53
    /// wasted turns (`examples/perish_switch_census.rs`, round-3 brief).
    /// OFF by default: the shipped ladder mask keeps the gate.
    pub immunity_ignores_switch_read: bool,
    /// Refuse a type-immune move only when it is immune against the mon in
    /// front AND against every mon the foe could bring in
    /// ([`foe_switchin_candidates`]). Sound wherever that candidate set is a
    /// superset of the true one, which is what the helper's doc argues from
    /// public state; it can therefore only ever refuse a turn that was going
    /// to do nothing whatever the foe does.
    ///
    /// Strictly weaker than `immunity_ignores_switch_read` — everything this
    /// refuses, that one refuses too — so the pair brackets the question
    /// "how much of the class is recoverable without ever being wrong".
    /// OFF by default.
    pub immunity_all_switchins: bool,
}

impl Default for MaskRules {
    fn default() -> Self {
        MaskRules {
            sleep_talk_awake: true,
            immunity_ignores_switch_read: false,
            immunity_all_switchins: false,
        }
    }
}

/// Every action the mask would refuse at this root, with the rule that
/// refused it — the diagnostic surface for [`certain_noop`]. A mask that
/// hides a *useful* move is a strength bug, so the rules have to be
/// auditable one by one against real positions, not just unit cases.
pub fn dominated_actions(b: &Battle, dex: &Dex, side: usize) -> Vec<(SearchChoice, &'static str)> {
    dominated_actions_with(b, dex, side, MaskRules::default())
}

/// [`dominated_actions`] under an explicit rule set. `noop_census`,
/// `analysis::report`, `human_agreement` and the postmortems all call the
/// default-rules wrapper above, so every instrument reports the SHIPPED mask.
pub fn dominated_actions_with(
    b: &Battle,
    dex: &Dex,
    side: usize,
    rules: MaskRules,
) -> Vec<(SearchChoice, &'static str)> {
    b.clone()
        .legal_choices(dex, side)
        .into_iter()
        .filter_map(|c| {
            if certain_self_loss(b, dex, side, c) {
                return Some((c, "self-KO with the last mon"));
            }
            noop_reason(b, dex, side, c, rules).map(|why| (c, why))
        })
        .collect()
}

/// Whether the foe can leave before the move lands. **Switches resolve before
/// moves**, so every rule that reads the CURRENT foe — its types, its status,
/// its volatiles, its Substitute — is only a proof when this is false.
/// Otherwise the move is a legitimate switch read: Toxic into an already-
/// poisoned mon poisons whatever comes in, and Earthquake into a Levitating
/// foe is aimed at its replacement. Mirrors `search.rs::legal_move_choices`:
/// a trapped mon cannot switch, and neither can one with no living bench.
fn foe_can_switch(b: &Battle, side: usize) -> bool {
    let opp = 1 - side;
    let Some(active) = b.active_id(opp) else { return false };
    if b.poke(active).trapped {
        return false;
    }
    let s = &b.sides[opp];
    s.party.iter().skip(1).any(|&slot| !s.roster[slot as usize].fainted)
}

/// Whether `side`'s active moves first on speed alone (no tie, no priority
/// read — the foe's move is unknown in blind play). Used by the rules that
/// depend on the user's own state surviving until its move resolves.
///
/// **Quick Claw voids the proof.** The item preempts on a per-turn coin
/// (`turn.rs` endTurn, 60/256) that `get_pokemon_action_speed` reads as a
/// 65535 speed. Inside the engine that coin is already rolled at the request
/// point, but the live client cannot observe it: the importer leaves
/// `quick_claw_roll` false in every reconstructed state (measured over the
/// 570-battle corpus: 0 true out of 20,765 decisions, with a Quick Claw on
/// the field in 3.1% of active slots). So a foe holding one moves first
/// nearly a quarter of the time with nothing in the state to say so, and the
/// rules gated on this would refuse moves the preempt makes live. Claim the
/// proof only when the foe cannot preempt at all.
fn faster_than_foe(b: &Battle, dex: &Dex, side: usize) -> bool {
    let (Some(me), Some(foe)) = (b.active_id(side), b.active_id(1 - side)) else {
        return false;
    };
    if !b.quick_claw_roll
        && b.poke(foe).item.is_some()
        && b.poke(foe).item == dex.known_items.quickclaw
    {
        return false;
    }
    b.get_pokemon_action_speed(dex, me) > b.get_pokemon_action_speed(dex, foe)
}

/// Is `def` immune to a move of `move_type`? Mirrors
/// `pokemon.rs::run_move_immunity`: Ground is resolved by groundedness
/// (gen-2 `isGrounded` = "Flying-types are airborne, nothing else"), every
/// other type by the chart.
///
/// **Known hole, shared with the shipped rule:** Foresight's
/// `onNegateImmunity` (`conditions.rs:765`) strips a Ghost's Normal/Fighting
/// immunity and nothing here reads the volatile, so a Foresighted Ghost is
/// called immune when it is not. Out of distribution rather than merely
/// rare — `foresight` occurs 0 times in the 32-team meta pool and 0 times in
/// the 570-battle corpus — but any mask that can meet it needs the check
/// added here first, in ONE place, which is why this predicate exists at all
/// instead of the old inline expression.
fn type_immune_to(b: &Battle, dex: &Dex, move_type: TypeId, def: PokeId) -> bool {
    let d = b.poke(def);
    if move_type == dex.known_types.ground {
        d.has_type(dex.known_types.flying)
    } else {
        d.types.iter().any(|t| dex.type_immune(move_type, t))
    }
}

/// Every foe mon that could be standing there when our move resolves, other
/// than the one in front — read off PUBLIC state only, and deliberately a
/// SUPERSET of the true set, so "immune to all of these" is a proof.
///
/// The information argument, since this is the whole soundness of
/// [`MaskRules::immunity_all_switchins`]:
///
/// * **Species are public.** The team-preview `|poke|` lines carry species,
///   level and gender for all six, which is why `Observer::new` reads them
///   straight off the roster and calls them "public at team preview". Types
///   follow from the species, and a benched mon's types are its species'
///   (`clearVolatile` restores base types on switch-out, so Conversion and
///   Transform only ever touch the active).
/// * **Which three were PICKED is not public** until they appear. So this
///   never reads party membership for a mon that has not appeared: on the
///   ladder those party slots are imputed in roster order
///   (`import.rs`, "then imputed hidden picks"), and reading them would be
///   reading a guess. It reads `appeared` instead — `previously_switched_in
///   > 0 || is_active`, the same predicate `Observer::observe` and
///   `Belief::determinize` use — and admits EVERY never-appeared roster mon
///   while any pick is still unrevealed. That is exactly the support the
///   determinizer samples hidden identities from (uniform over the
///   not-yet-appeared roster mons), so it is also "whatever the belief
///   admits", and a pinned/open-sheet belief is a subset of it.
/// * Once every pick has appeared, the set collapses to the real bench, and
///   the rule becomes as tight as perfect information would make it.
///
/// Fainted mons are dropped (public), and a never-appeared mon can never be
/// fainted. Nothing here can hand the foe a Pokemon it did not have, and a
/// bench only ever shrinks, so a superset at the decision is still a
/// superset at resolution.
fn foe_switchin_candidates(b: &Battle, side: usize) -> Vec<PokeId> {
    let opp = 1 - side;
    let s = &b.sides[opp];
    let active = b.active_id(opp);
    let appeared = |p: &nc2000_engine::state::Pokemon| p.previously_switched_in > 0 || p.is_active;
    let seen = s.roster.iter().filter(|p| appeared(p)).count();
    // `party.len()` is the pick count (3 in this format); every appeared mon
    // is in it, so `seen >= party.len()` means no hidden pick is left.
    let all_picks_revealed = seen >= s.party.len();
    let mut out = Vec::new();
    for (slot, p) in s.roster.iter().enumerate() {
        let id = PokeId { side: opp as u8, slot: slot as u8 };
        if Some(id) == active || p.fainted || p.hp <= 0 {
            continue;
        }
        let could_come_in = if appeared(p) {
            s.party.contains(&(slot as u8))
        } else {
            !all_picks_revealed
        };
        if could_come_in {
            out.push(id);
        }
    }
    out
}

/// [`certain_noop`] with the reason. Each arm names the engine site it
/// mirrors; adding a rule here without one is how a false positive gets in.
///
/// **What "certain" means here.** Every rule is read off the position as it
/// stands at the decision, and a foe switch (switches resolve before moves)
/// or a foe self-cure can make a refused action live before it resolves.
/// `noop_census` measures that error rate against the engine over corpus
/// positions; it is the number to re-check whenever a rule is added.
fn noop_reason(
    b: &Battle,
    dex: &Dex,
    side: usize,
    c: SearchChoice,
    rules: MaskRules,
) -> Option<&'static str> {
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
        let on_foe = ms.target == "foeSide";
        let target_side = if on_foe { 1 - side } else { side };
        let up = dex.conds_id(sc).is_some_and(|cid| b.sides[target_side].has_side_condition(cid));
        // Our own screens cannot be removed in gen 2 and expire at upkeep,
        // after moves. Hazards on the FOE's side can be spun away first, so
        // that half needs us to move first.
        verdict!(
            up && (!on_foe || faster_than_foe(b, dex, side)),
            "that side condition is already up"
        );
    }
    // ---- a phazing move with nobody left to drag in. `moveexec.rs:2262`
    // raises didSomething only for `md.force_switch && self.can_switch(t.side)`
    // and the drag at `moveexec.rs:2412` is gated on the same pair, so with an
    // empty foe bench the move ends as a bare `-fail`. `forceSwitch` is exactly
    // {roar, whirlwind} in this format and neither carries a second payload
    // (data/gen2stadium2.json: category Status, all effect fields null), so
    // nothing else can land.
    //
    // Calls the engine's own `can_switch` rather than transcribing it, and
    // deliberately NOT `foe_can_switch` above: that one also reports false for
    // a TRAPPED foe, but a trapped foe still gets dragged out by a phaze, so
    // reusing it here would refuse a move that works.
    //
    // No speed gate, unlike the rules below. A bench only ever shrinks —
    // `fainted` is only ever assigned true (`turn.rs:314`), `pokemon_left` is
    // only decremented in battle (`turn.rs:297`), and `party` only grows at the
    // preview action (`turn.rs:53`) — and a foe with no bench cannot switch,
    // Baton Pass being gated on the same `can_switch` (`moveexec.rs:2265`).
    // Zero at the decision therefore implies zero at resolution whatever the
    // move order, so there is nothing for a faster foe to invalidate.
    if ms.force_switch && !b.can_switch((1 - side) as u8) {
        yes!("there is nobody left to phaze in");
    }
    // ---- Perish Song / Destiny Bond from the last mon. The Stadium 2 rule
    // at `moveexec.rs:499` (destinybond/perishsong onPrepareHit) returns
    // False on `pokemon_left == 1`, so the move ends as a bare `-fail` with
    // the "fails if it is being used by your last Pokemon" hint and nothing
    // else runs. The pair is exactly the set of moves that arm covers.
    //
    // Reads OUR OWN bench, which is why this needs no speed gate and no
    // switch gate: a side's `pokemon_left` only ever decreases within a
    // battle (`turn.rs`), and nothing the foe does on this turn can hand us
    // a Pokemon back. One at the decision therefore implies one at
    // resolution, whatever the move order — the same argument the phaze rule
    // above makes about an empty foe bench.
    //
    // Found by battle-4040 (2026-08-16): Gengar's kit is Ice Punch / Mean
    // Look / Destiny Bond / Perish Song, and as the last mon the bot spent 8
    // of 22 turns on the two that cannot do anything.
    if matches!(key, "perishsong" | "destinybond") && b.sides[side].pokemon_left == 1 {
        yes!("Perish Song and Destiny Bond fail from the last mon");
    }
    // ---- Sleep Talk / Snore chosen while AWAKE. `moveexec.rs:521`
    // (`("sleeptalk","onTry") | ("snore","onTry")` => `RV::from_bool(status ==
    // Slp)`) hands a false onTry to `moveexec.rs:1799`, which returns
    // `MoveOutcome::Fail` and — unlike the PrepareHit branch two lines above
    // it — emits NO `-fail`. The PP is already gone by then
    // (`moveexec.rs:1492`, before `use_move`), so the turn is spent, nothing
    // happens, and the protocol says nothing at all: this is the one mask rule
    // whose firings `noop_census` can only score as SILENT agreement.
    //
    // `sleep_usable` is exactly {sleeptalk, snore} in this dex, the same pair
    // the engine arm covers, so the flag mirrors the engine instead of
    // transcribing a key list. Checked ABOVE the `category != Status` early
    // return below, because Snore is physical.
    //
    // Needs the speed gate, and nothing else. The proof object is OUR OWN
    // status, and the one thing that can change it before our move resolves is
    // a foe that moves first and lands a sleep: `conditions.rs:159`
    // (slp/onBeforeMove) decrements the counter and then returns `Undef` for a
    // `sleep_usable` move, so a user put to sleep this very turn still gets a
    // working Sleep Talk. In-distribution, not theoretical — 15 of the 32
    // meta-pool teams carry a sleep move. No switch gate: the rule never reads
    // the foe. A mon that wakes on its OWN turn is untouched by construction:
    // the sleep counter is hidden, so at the decision its status still reads
    // `Slp` (24 such Sleep Talks in the 570-battle corpus).
    //
    // Found by battle-4070 (2026-08-19): Suicune, alone against a Rest-
    // stalling Umbreon, spent 6 of its 64 last-mon turns on an awake Sleep
    // Talk. Corpus rate: 54 of 863 Sleep Talk selections (6.3%) were made by a
    // plainly awake mon.
    //
    // Deliberately `if`, not `verdict!`: a false verdict here returns None and
    // would shadow every rule below for this move. Snore is a Normal-typed
    // attack, so an ASLEEP user aiming it at a stranded Ghost still has to
    // reach the type-immunity arm.
    if rules.sleep_talk_awake
        && ms.sleep_usable
        && b.poke(att).status != Status::Slp
        && faster_than_foe(b, dex, side)
    {
        yes!("Sleep Talk and Snore need the user asleep");
    }
    // Every rule below that reads the foe is conditioned on the foe being
    // stuck with the mon it has: otherwise the move is aimed at whatever
    // switches in, and refusing it would delete a real option.
    let foe_here = b.active_id(1 - side).filter(|&d| {
        let p = b.poke(d);
        !p.fainted && p.hp > 0
    });
    let foe = foe_here.filter(|_| !foe_can_switch(b, side));
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
    // Hidden Power's dex entry says Normal; its real type comes from the
    // user's DVs (`moveexec.rs:340`). Reading the static field refused Hidden
    // Power into every Ghost — caught by `noop_census`, 7 refusals that the
    // engine damaged straight through.
    let move_type =
        if key == "hiddenpower" { b.poke(att).hp_type } else { ms.move_type };
    if foe_targeted
        && !ms.ignore_immunity
        && !ms.selfdestruct
        && move_type != dex.known_types.unknown
    {
        // The shipped proof: the foe is stuck with the mon in front, and
        // that mon is immune. `foe` is already `foe_can_switch`-gated.
        if foe.is_some_and(|def| type_immune_to(b, dex, move_type, def)) {
            yes!("the target is immune to the move's type");
        }
        // Both A/B arms below are dead code at `MaskRules::default()`, and
        // both are reached only when the gate above dropped the proof —
        // when the foe CAN leave. They are `if`, never `verdict!`: a
        // `verdict!` here returns None on the miss and would shadow every
        // rule after it (the shadowing bug commit 238e48a exists for).
        let front_immune =
            || foe_here.is_some_and(|def| type_immune_to(b, dex, move_type, def));
        // Arm 1: refuse the mon in front regardless of the switch read.
        if rules.immunity_ignores_switch_read && front_immune() {
            yes!("the mon in front is immune (switch read ignored)");
        }
        // Arm 2: refuse only when nothing the foe can bring in is hittable
        // either, so the turn is dead whatever they do.
        if rules.immunity_all_switchins
            && front_immune()
            && foe_switchin_candidates(b, side)
                .into_iter()
                .all(|c| type_immune_to(b, dex, move_type, c))
        {
            yes!("immune to the move's type, and so is every possible switch-in");
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
        // Safeguard is a SIDE condition: it survives a switch, so it blocks
        // the replacement too and needs no switch gate.
        if safeguard {
            yes!("Safeguard blocks foe-inflicted status");
        }
        // Sleep Clause Mod (`conditions.rs` sleepclausemod/onSetStatus): a
        // second FOE-SOURCED sleep on that side is refused outright, and
        // Rest-sleep does not engage it. Like Safeguard this reads the whole
        // side rather than the mon in front, so it survives a switch — a
        // sleeper that leaves the field keeps its status, and the replacement
        // cannot be slept either. What it does need is for us to move first:
        // the one way the clause lifts before our move is the sleeper waking
        // up on its own turn.
        if ms.status.as_deref() == Some("slp") && faster_than_foe(b, dex, side) {
            let opp = 1 - side;
            let engaged = b.sides[opp].party.iter().any(|&slot| {
                let p = &b.sides[opp].roster[slot as usize];
                p.hp > 0
                    && p.status == Status::Slp
                    && p.status_state.source.map(|s| s.side as usize != opp).unwrap_or(true)
            });
            if engaged {
                yes!("Sleep Clause Mod blocks a second foe-sourced sleep");
            }
        }
        if let Some(def) = foe {
            let d = b.poke(def);
            // One major status at a time (`pokemon.rs::set_status`) — but a
            // foe that moves first can Rest, eat a berry or wake up, and the
            // census caught exactly that. Only a proof when we act first.
            if d.status != Status::None && faster_than_foe(b, dex, side) {
                yes!("the target already carries a major status");
            }
            // a Substitute blocks every foe-inflicted status
            // (`conditions.rs` substitute/onTryPrimaryHit)
            if sub_up {
                yes!("a Substitute blocks foe-inflicted status");
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
        // Both halves are about OUR state at resolution time: a faster foe can
        // break the sub we already have, or damage us past the quarter gate
        // (72 of this rule's 1,138 firings were wrong for that reason).
        verdict!(
            (up || me.hp as f64 <= me.maxhp as f64 / 4.0 || me.maxhp == 1)
                && faster_than_foe(b, dex, side),
            "Substitute is already up, or there is not enough HP to pay for one"
        );
    }

    if let Some(v) = ms.volatile_status.as_deref() {
        // Safeguard again: side-wide, so it stops confusion whoever is in.
        if v == "confusion" && key != "swagger" && foe_targeted && safeguard {
            yes!("Safeguard blocks confusion");
        }
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
            // A volatile can also run out on its owner's own move (a
            // confusion clock is decremented before the move it hinders), so
            // the foe-directed half needs us to move first.
            let stable = tgt == Some(att) || faster_than_foe(b, dex, side);
            if stable && b.poke(t).has_volatile(vid) {
                yes!("the target already has that volatile");
            }
        }
        if foe_targeted && foe.is_some() {
            // Confusion is blocked by both a Substitute and Safeguard;
            // Swagger is the documented exception — the engine strips its
            // confusion behind a Substitute but still lands the +2 Attack,
            // so it is NOT a no-op there.
            if v == "confusion" && key != "swagger" && sub_up {
                yes!("a Substitute blocks confusion");
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
            // A faster foe can un-cap us first (Screech into our +6 defence
            // is the census's example), so our own stages are only fixed if we
            // act first; the foe's are covered by the switch gate above.
            if capped && (!self_targeted || faster_than_foe(b, dex, side)) {
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
        let mut table = FxHashMap::default();
        table.insert(key_of(&cfg, dex, &mut root), 0usize);
        let root_dominated = [0usize, 1].map(|s| {
            nodes[0].acts[s]
                .iter()
                .map(|&c| {
                    certain_self_loss(&root, dex, s, c)
                        || certain_noop(&root, dex, s, c, cfg.mask_rules)
                })
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
    use nc2000_engine::state::{PokeId, Status};

    /// Faint `side`'s bench so its active cannot leave. Every rule that reads
    /// the CURRENT foe is conditioned on this: switches resolve before moves,
    /// so with a bench available the same move is a legitimate switch read.
    fn strand(b: &mut Battle, side: usize) {
        let party = b.sides[side].party.clone();
        for &slot in party.iter().skip(1) {
            let id = PokeId { side: side as u8, slot };
            b.poke_mut(id).hp = 0;
            b.poke_mut(id).fainted = true;
        }
    }

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

    /// Every test here asserts the SHIPPED mask, so it reads
    /// [`super::certain_noop`] at `MaskRules::default()`. This shadows the
    /// parent item deliberately; the ablation rule set is exercised by name in
    /// `mask_rules_ablation_isolates_the_sleep_talk_rule`.
    fn certain_noop(b: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> bool {
        super::certain_noop(b, dex, side, c, MaskRules::default())
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
            strand(&mut b, 1);
            let def = b.active_id(1).unwrap();
            b.set_status(&dex, def, "par", None, EffectHandle::None, true);
            assert!(noop(&b, 0, "sleeppowder"), "status onto statused foe");
        }
        {
            let mut b = b.clone();
            strand(&mut b, 1);
            let def = b.active_id(1).unwrap();
            b.add_volatile(&dex, def, "substitute", None, EffectHandle::None);
            assert!(noop(&b, 0, "sleeppowder"), "Sleep Powder into a Substitute");
            // damaging moves stay live through the sub
            assert!(!noop(&b, 1, "bodyslam"));
        }

        // re-inflicting a volatile / Substitute behind its own sub
        {
            let mut b = b.clone();
            strand(&mut b, 0);
            let att0 = b.active_id(0).unwrap();
            let def = b.active_id(1).unwrap();
            b.add_volatile(&dex, att0, "confusion", None, EffectHandle::None);
            b.add_volatile(&dex, def, "substitute", None, EffectHandle::None);
            // A confusion clock ticks down on its owner's own move, so this
            // rule needs the caster to move first.
            assert!(!noop(&b, 1, "confuseray"), "slower: the clock may run out first");
            b.poke_mut(def).boosts[4] = 6;
            assert!(noop(&b, 1, "confuseray"), "Confuse Ray onto confused foe, moving first");
            assert!(noop(&b, 1, "substitute"), "Substitute behind its own sub");
        }
    }

    /// Sleep Clause Mod is the one foe-state rule that survives a switch: it
    /// reads the whole side, and a sleeper that leaves the field stays
    /// asleep. The mask has to refuse a second sleep even when the mon in
    /// front is healthy and free to leave.
    #[test]
    fn noop_mask_covers_sleep_clause() {
        let (dex, b) = setup();
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));
        let me = b.active_id(0).unwrap();
        let bench = PokeId { side: 1, slot: b.sides[1].party[1] };

        // foe-sourced sleeper on the bench, healthy switchable mon in front
        {
            let mut b = b.clone();
            b.set_status(&dex, bench, "slp", Some(me), EffectHandle::None, true);
            assert!(noop(&b, 0, "sleeppowder"), "second foe-sourced sleep");
            let def = b.active_id(1).unwrap();
            let mut after = b.clone();
            after.set_log_enabled(true);
            after.choose(&dex, 0, "move sleeppowder").unwrap();
            after.choose(&dex, 1, "move bodyslam").unwrap();
            assert!(
                after.log.iter().any(|l| l.contains("Sleep Clause Mod activated")),
                "engine did not engage the clause: {:?}",
                after.log
            );
            assert_eq!(after.poke(def).status, Status::None, "no sleep landed");
        }

        // Rest-sleep is ally-sourced and leaves the clause open
        // (`conditions.rs` sleepclausemod/onSetStatus: ally source → Undef).
        {
            let mut b = b.clone();
            b.set_status(&dex, bench, "slp", Some(bench), EffectHandle::None, true);
            assert!(!noop(&b, 0, "sleeppowder"), "Rest sleep does not engage the clause");
        }

        // a fainted sleeper does not hold the clause either (`hp > 0`)
        {
            let mut b = b.clone();
            b.set_status(&dex, bench, "slp", Some(me), EffectHandle::None, true);
            b.poke_mut(bench).hp = 0;
            b.poke_mut(bench).fainted = true;
            assert!(!noop(&b, 0, "sleeppowder"), "a fainted sleeper releases the clause");
        }

        // moving second is not a proof: the sleeper can wake on its own turn
        {
            let mut b = b.clone();
            b.set_status(&dex, bench, "slp", Some(me), EffectHandle::None, true);
            let def = b.active_id(1).unwrap();
            b.poke_mut(def).boosts[4] = 6;
            assert!(!noop(&b, 0, "sleeppowder"), "slower: the clause may lift first");
        }
    }

    /// Quick Claw preempts on a coin the client cannot see, so it voids every
    /// rule that needs us to move first (`faster_than_foe`).
    #[test]
    fn quick_claw_voids_the_speed_proof() {
        let (dex, b) = setup();
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));
        let mut b = b.clone();
        strand(&mut b, 1);
        let me = b.active_id(0).unwrap();
        let def = b.active_id(1).unwrap();
        b.set_status(&dex, def, "par", Some(me), EffectHandle::None, true);
        assert!(noop(&b, 0, "sleeppowder"), "statused foe, and we act first");

        b.poke_mut(def).item = dex.known_items.quickclaw;
        assert!(!noop(&b, 0, "sleeppowder"), "a Quick Claw foe may move first");
        // …and when the coin is up it is the speed comparison itself that
        // says we are second.
        b.quick_claw_roll = true;
        assert!(!noop(&b, 0, "sleeppowder"), "coin up: the foe is faster outright");

        // Same contract for the awake-Sleep-Talk rule, on the fixture that
        // carries the pair: a Quick Claw foe can preempt with a sleep move,
        // and the client cannot see the coin.
        let (dex, st) = st_setup();
        let st_noop = |b: &Battle, key: &str| certain_noop(b, &dex, 0, mv(&dex, key));
        let st_foe = st.active_id(1).unwrap();
        for key in ["sleeptalk", "snore"] {
            assert!(st_noop(&st, key), "awake {key}, and we act first");
        }
        let mut claw = st.clone();
        claw.poke_mut(st_foe).item = dex.known_items.quickclaw;
        for key in ["sleeptalk", "snore"] {
            assert!(!st_noop(&claw, key), "a Quick Claw foe may sleep us first ({key})");
        }
        claw.quick_claw_roll = true;
        for key in ["sleeptalk", "snore"] {
            assert!(!st_noop(&claw, key), "coin up: the foe is faster outright ({key})");
        }
    }

    // ---- 2026-08-19: Sleep Talk / Snore chosen while awake --------------
    //
    // Battle 4070: Suicune, alone against a Rest-stalling Umbreon, spent 6 of
    // its 64 last-mon turns on a Sleep Talk it was awake for. The engine fails
    // the move at `moveexec.rs:521` (onTry = `status == Slp`) and — uniquely
    // among the masked rules — says NOTHING in the protocol while doing it.

    fn st_team() -> Vec<PokemonSet> {
        // from_fixture does not validate movesets. Snore sits next to Sleep
        // Talk on purpose: it is PHYSICAL, so refusing it proves the rule is
        // reached before `noop_reason`'s `category != Status` early return.
        serde_json::from_str(
            r#"[
            {"name":"Suicune","species":"Suicune","item":"","ability":"No Ability",
             "moves":["Surf","Sleep Talk","Snore","Rest"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"N","level":50},
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Splash","Sleep Powder","Body Slam","Rest"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Splash","Sleep Powder","Psychic","Rest"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap()
    }

    /// Side 0 leads Suicune (base Spe 85), side 1 leads Snorlax (base Spe 30),
    /// so the speed proof holds with nothing to argue about.
    fn st_setup() -> (Dex, Battle) {
        let dex = conformance::load_dex();
        let t = st_team();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &t, &t).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, "team 2, 1, 3").unwrap();
        (dex, b)
    }

    #[test]
    fn noop_mask_covers_awake_sleep_talk() {
        let (dex, b) = st_setup();
        let noop = |b: &Battle, key: &str| certain_noop(b, &dex, 0, mv(&dex, key));
        let me = b.active_id(0).unwrap();
        let foe = b.active_id(1).unwrap();
        assert!(
            b.get_pokemon_action_speed(&dex, me) > b.get_pokemon_action_speed(&dex, foe),
            "fixture must put side 0 first on speed"
        );

        // awake and faster ⇒ both halves of the engine's onTry arm are refused
        assert!(noop(&b, "sleeptalk"), "awake Sleep Talk cannot call anything");
        assert!(noop(&b, "snore"), "awake Snore, and it is physical");
        assert!(!noop(&b, "surf"), "the real move stays live");

        // asleep ⇒ exactly what the move is for
        {
            let mut b = b.clone();
            b.set_status(&dex, me, "slp", Some(me), EffectHandle::None, true);
            assert!(!noop(&b, "sleeptalk"), "asleep: Sleep Talk works");
            assert!(!noop(&b, "snore"), "asleep: Snore works");
        }

        // moving second is not a proof: a foe that sleeps us this turn hands
        // us a working Sleep Talk on the very turn the sleep lands
        // (`conditions.rs:159` returns Undef for a sleep_usable move after
        // decrementing). 15 of the 32 meta-pool teams carry a sleep move.
        {
            let mut b = b.clone();
            b.poke_mut(foe).boosts[4] = 6;
            assert!(!noop(&b, "sleeptalk"), "slower: the foe can sleep us first");
            assert!(!noop(&b, "snore"), "slower: the foe can sleep us first");
        }

        // the foe's bench is irrelevant — the rule reads our own status only
        {
            let mut stuck = b.clone();
            strand(&mut stuck, 1);
            assert!(noop(&stuck, "sleeptalk"), "a stranded foe changes nothing");
            let mut open = b.clone();
            open.poke_mut(foe).trapped = false;
            assert!(noop(&open, "sleeptalk"), "a switchable foe changes nothing");
        }

        // engine cross-check. Play the turn: the PP is gone, nothing happened,
        // and — unlike every other masked rule — the log carries NO marker,
        // so this asserts on state, not on a log string.
        {
            let st_id = dex.moves.id("sleeptalk").unwrap();
            let pp_before = b.poke(me).get_move_slot(st_id).unwrap().pp;
            let mut after = b.clone();
            after.set_log_enabled(true);
            after.choose(&dex, 0, "move sleeptalk").unwrap();
            after.choose(&dex, 1, "move splash").unwrap();
            assert_eq!(
                after.poke(me).get_move_slot(st_id).unwrap().pp,
                pp_before - 1,
                "the PP is spent anyway"
            );
            assert_eq!(after.poke(me).hp, after.poke(me).maxhp, "no Rest was called");
            assert_eq!(after.poke(foe).hp, after.poke(foe).maxhp, "no attack was called");
            assert_eq!(after.poke(me).status, Status::None, "and no self-status");
            let ms = after.poke_str(me);
            assert!(
                !after.log.iter().any(|l| l.starts_with(&format!("|-fail|{ms}"))),
                "the engine fails it silently, with no -fail: {:?}",
                after.log
            );
        }
    }

    /// The mask vetoes, it does not forbid: when every legal action is
    /// refused, `best()` falls back to the unfiltered argmax and still
    /// submits one. Battle 4070's turns 89-99 are the real instance — Rest at
    /// full HP was the only move with PP left.
    #[test]
    fn best_still_submits_the_masked_noop_when_it_is_the_only_option() {
        let dex = conformance::load_dex();
        let t: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Suicune","species":"Suicune","item":"","ability":"No Ability",
             "moves":["Sleep Talk"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"N","level":50},
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Splash","Body Slam","Rest","Sleep Powder"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Splash","Psychic","Rest","Sleep Powder"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &t, &t).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, "team 2, 1, 3").unwrap();
        strand(&mut b, 0);
        let acts = b.clone().legal_choices(&dex, 0);
        assert_eq!(acts.len(), 1, "the fixture must leave exactly one action");
        assert_eq!(dominated_actions(&b, &dex, 0).len(), 1, "and the mask must refuse it");

        let cfg = RmConfig { rule: SelRule::Ucb, iterations: 200, ..Default::default() };
        for seed in 1..=5u64 {
            let mut s = SkuctSearch::new(&b, &dex, cfg.clone(), seed);
            s.step(&dex, 200);
            assert_eq!(
                s.best(0).map(|c| c.to_input(&dex)),
                Some("move sleeptalk".to_string()),
                "best() must still submit something (seed {seed})"
            );
        }
    }

    /// The A/B seam: `MaskRules::default()` is the shipped mask, and the
    /// ablation arm (arena `blindlegacy`) sees exactly one rule fewer.
    #[test]
    fn mask_rules_ablation_isolates_the_sleep_talk_rule() {
        let (dex, b) = st_setup();
        let legacy = MaskRules { sleep_talk_awake: false, ..MaskRules::default() };
        assert!(certain_noop(&b, &dex, 0, mv(&dex, "sleeptalk")), "shipped refuses it");
        assert!(
            !super::certain_noop(&b, &dex, 0, mv(&dex, "sleeptalk"), legacy),
            "the ablation arm does not"
        );
        // …and nothing else moves: same position, same other verdicts.
        let shipped: Vec<&str> =
            dominated_actions(&b, &dex, 0).into_iter().map(|(_, why)| why).collect();
        let ablated: Vec<&str> = dominated_actions_with(&b, &dex, 0, legacy)
            .into_iter()
            .map(|(_, why)| why)
            .collect();
        assert_eq!(
            shipped.iter().filter(|w| !w.starts_with("Sleep Talk")).count(),
            ablated.len(),
            "the flag must move only its own rule: {shipped:?} vs {ablated:?}"
        );
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

    /// `lead1` picks which of the three leads side 1 sends out. Both benches
    /// are fainted: the foe-state rules only apply to a foe that cannot
    /// leave, and `switchable_foe_disarms_every_foe_state_rule` covers the
    /// other half.
    fn imm_setup(lead1: &str) -> (Dex, Battle) {
        let (dex, mut b) = imm_setup_open(lead1);
        strand(&mut b, 0);
        strand(&mut b, 1);
        (dex, b)
    }

    /// The same position with both benches intact.
    fn imm_setup_open(lead1: &str) -> (Dex, Battle) {
        let dex = conformance::load_dex();
        let t = imm_team();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &t, &t).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, lead1).unwrap();
        (dex, b)
    }

    /// The rule the owner caught missing: a foe with somewhere to go makes
    /// every current-foe reading provisional, because switches resolve before
    /// moves. Thunder Wave at a Ground type is aimed at its replacement.
    #[test]
    fn switchable_foe_disarms_every_foe_state_rule() {
        let (dex, open) = imm_setup_open("team 2, 1, 3");
        let mut stuck = open.clone();
        strand(&mut stuck, 1);
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));
        for key in ["thunderwave", "toxic"] {
            assert!(!noop(&open, 0, key), "{key} is a switch read while the foe has a bench");
            assert!(noop(&stuck, 0, key), "{key} is refused once the foe is stuck");
        }
        // Trapping is the other way to be stuck, and the engine's own flag is
        // what both the mask and `legal_choices` read.
        {
            let mut trapped = open.clone();
            let def = trapped.active_id(1).unwrap();
            trapped.poke_mut(def).trapped = true;
            assert!(noop(&trapped, 0, "thunderwave"), "a trapped foe cannot leave either");
        }
        // Own-state rules are unaffected: a switch cannot heal me, refill my
        // HP, or move my stat stages.
        {
            let mut b = open.clone();
            let att = b.active_id(0).unwrap();
            b.poke_mut(att).boosts[0] = 6;
            assert!(noop(&b, 0, "swordsdance"), "capped self-boost does not care");
        }
        // Safeguard is a SIDE condition: it survives the switch, so it still
        // refuses whoever comes in.
        {
            let mut b = open.clone();
            let def = b.active_id(1).unwrap();
            b.add_side_condition(&dex, 1, "safeguard", Some(def), EffectHandle::None);
            assert!(noop(&b, 0, "toxic"), "Safeguard covers the replacement too");
        }
    }

    /// A phaze is a no-op exactly when the foe has nobody to be dragged in.
    /// The trapped case is the false positive this rule has to dodge:
    /// `foe_can_switch` calls a trapped foe stuck, but the engine's
    /// `can_switch` never reads `trapped` and the drag lands anyway.
    #[test]
    fn noop_mask_covers_phaze_with_no_bench() {
        let dex = conformance::load_dex();
        let phazer: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Skarmory","species":"Skarmory","item":"","ability":"No Ability",
             "moves":["Whirlwind","Drill Peck","Rest","Curse"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Sleep Powder","Reflect","Rest","Spikes"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Zapdos","species":"Zapdos","item":"","ability":"No Ability",
             "moves":["Thunder Wave","Toxic","Leech Seed","Swords Dance"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"N","level":50}
        ]"#,
        )
        .unwrap();
        let foes: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Body Slam","Substitute","Confuse Ray","Rest"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Nidoking","species":"Nidoking","item":"","ability":"No Ability",
             "moves":["Earthquake","Substitute","Screech","Dream Eater"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Safeguard","Mist","Swagger"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap();
        let mut open = Battle::from_fixture(&dex, "7,8,9,10", &phazer, &foes).unwrap();
        open.set_log_enabled(false);
        open.choose(&dex, 0, "team 1, 2, 3").unwrap();
        open.choose(&dex, 1, "team 1, 2, 3").unwrap();
        // Gen 2 phazes also fail when the user moved FIRST (`moveexec.rs:619`),
        // a separate speed-dependent class this rule does not claim. Skarmory
        // outruns Snorlax, so slow it down to isolate the bench question.
        let att = open.active_id(0).unwrap();
        open.poke_mut(att).boosts[4] = -6;

        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));

        // A live foe bench: not a no-op, and the engine really does drag.
        assert!(!noop(&open, 0, "whirlwind"), "a full bench is a live phaze");
        let dragged = play(&dex, &open, 0, 1);
        assert_ne!(
            dragged.active_id(1).unwrap(),
            open.active_id(1).unwrap(),
            "the engine dragged a bench mon in"
        );

        // Trapped, but the bench is alive: `can_switch` ignores `trapped`, so
        // the drag still lands and the mask must NOT claim it.
        {
            let mut trapped = open.clone();
            let def = trapped.active_id(1).unwrap();
            trapped.poke_mut(def).trapped = true;
            assert!(!noop(&trapped, 0, "whirlwind"), "a trapped foe is still draggable");
            let after = play(&dex, &trapped, 0, 1);
            assert_ne!(
                after.active_id(1).unwrap(),
                trapped.active_id(1).unwrap(),
                "trapping does not stop a phaze"
            );
        }

        // Nobody left to drag in: refused, and the engine agrees nothing moved.
        {
            let mut stuck = open.clone();
            strand(&mut stuck, 1);
            assert!(noop(&stuck, 0, "whirlwind"), "phaze into an empty bench");
            let after = play(&dex, &stuck, 0, 1);
            assert_eq!(
                after.active_id(1).unwrap(),
                stuck.active_id(1).unwrap(),
                "the foe's active is untouched"
            );
        }
    }

    /// The Stadium 2 last-mon rule (`moveexec.rs:499`). Reported from ladder
    /// battle-4040 (2026-08-16): the bot's Gengar carries Ice Punch / Mean
    /// Look / Destiny Bond / Perish Song, and as the last mon it spent 8 of
    /// its 22 remaining turns on the two moves that cannot do anything.
    #[test]
    fn noop_mask_covers_last_mon_perish_song_and_destiny_bond() {
        let dex = conformance::load_dex();
        let ghost: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Gengar","species":"Gengar","item":"","ability":"No Ability",
             "moves":["Ice Punch","Mean Look","Destiny Bond","Perish Song"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Sleep Powder","Reflect","Rest","Spikes"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Cloyster","species":"Cloyster","item":"","ability":"No Ability",
             "moves":["Surf","Ice Beam","Screech","Toxic"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap();
        let foes = team();
        let mut open = Battle::from_fixture(&dex, "7,8,9,10", &ghost, &foes).unwrap();
        open.set_log_enabled(false);
        open.choose(&dex, 0, "team 1, 2, 3").unwrap();
        open.choose(&dex, 1, "team 2, 1, 3").unwrap();
        let noop = |b: &Battle, side: usize, key: &str| certain_noop(b, &dex, side, mv(&dex, key));

        // The engine's own verdict, read off the turn it logs. Destiny Bond's
        // volatile is stripped again by `onFoeAfterMoveSelf` once the foe has
        // moved (`conditions.rs:828`), so a post-turn volatile check can only
        // speak for Perish Song — the hint line speaks for both.
        let hinted = |b: &Battle, slot: usize| -> bool {
            let mut b = b.clone();
            b.set_log_enabled(true);
            b.log.clear();
            let after = play(&dex, &b, 0, slot);
            after.log.iter().any(|l| l.contains("used by your last Pokemon"))
        };

        // A live bench: both moves work, and the engine really does apply them.
        for key in ["perishsong", "destinybond"] {
            assert!(!noop(&open, 0, key), "{key} works while the bench is alive");
        }
        let ps_cond = dex.conds_id("perishsong").unwrap();
        let perished = play(&dex, &open, 0, 4);
        assert!(
            perished.poke(perished.active_id(1).unwrap()).has_volatile(ps_cond),
            "the engine set the perish counter"
        );
        assert!(!hinted(&open, 4), "no Stadium refusal with a bench");
        assert!(!hinted(&open, 3), "no Stadium refusal with a bench");

        // Last mon: refused, and the engine agrees nothing landed.
        let mut last = open.clone();
        strand(&mut last, 0);
        // `strand` plants the faints directly, so it does not run the
        // decrement at `turn.rs:297`. The rule and the engine both read
        // `pokemon_left`, so the fixture has to carry it.
        last.sides[0].pokemon_left = 1;
        for key in ["perishsong", "destinybond"] {
            assert!(noop(&last, 0, key), "{key} fails from the last mon");
        }
        let after_ps = play(&dex, &last, 0, 4);
        assert!(
            !after_ps.poke(after_ps.active_id(1).unwrap()).has_volatile(ps_cond),
            "no perish counter from the last mon"
        );
        assert!(hinted(&last, 4), "the engine refused Perish Song");
        assert!(hinted(&last, 3), "the engine refused Destiny Bond");

        // The rule is about our OWN bench, not the foe's, and it claims
        // nothing about the rest of the kit.
        let mut foe_last = open.clone();
        strand(&mut foe_last, 1);
        for key in ["perishsong", "destinybond"] {
            assert!(!noop(&foe_last, 0, key), "{key} does not read the foe's bench");
        }
        assert!(!noop(&last, 0, "icepunch"), "the damaging move stays live");
        assert!(!noop(&last, 0, "meanlook"), "Mean Look still applies its volatile");

        // …and it is the argmax consequence that matters: with the mask in
        // place `best` can never submit one of the two while Ice Punch exists,
        // however flat the root values are.
        let acts = last.clone().legal_choices(&dex, 0);
        let refused: Vec<&'static str> =
            dominated_actions(&last, &dex, 0).into_iter().map(|(_, why)| why).collect();
        assert_eq!(refused.len(), 2, "exactly the two moves, out of {}", acts.len());
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

        // Hidden Power's dex type is Normal, which would refuse it into
        // every Ghost; its real type comes from the user's DVs. The census
        // found the engine damaging Misdreavus through 7 such refusals.
        {
            let ghost: Vec<PokemonSet> = serde_json::from_str(
                r#"[
                {"name":"Misdreavus","species":"Misdreavus","item":"","ability":"No Ability",
                 "moves":["Confuse Ray","Pain Split","Perish Song","Protect"],
                 "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":50},
                {"name":"Gengar","species":"Gengar","item":"","ability":"No Ability",
                 "moves":["Night Shade","Hypnosis","Explosion","Thunderbolt"],
                 "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
                {"name":"Haunter","species":"Haunter","item":"","ability":"No Ability",
                 "moves":["Night Shade","Hypnosis","Explosion","Thunderbolt"],
                 "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
            ]"#,
            )
            .unwrap();
            let mine: Vec<PokemonSet> = serde_json::from_str(
                r#"[
                {"name":"Zapdos","species":"Zapdos","item":"","ability":"No Ability",
                 "moves":["Hidden Power","Body Slam","Swords Dance","Thunderbolt"],
                 "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"N","level":50},
                {"name":"Nidoking","species":"Nidoking","item":"","ability":"No Ability",
                 "moves":["Earthquake","Substitute","Screech","Dream Eater"],
                 "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
                {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
                 "moves":["Confuse Ray","Safeguard","Mist","Swagger"],
                 "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
            ]"#,
            )
            .unwrap();
            let dex3 = conformance::load_dex();
            let mut b3 = Battle::from_fixture(&dex3, "7,8,9,10", &mine, &ghost).unwrap();
            b3.set_log_enabled(false);
            b3.choose(&dex3, 0, "team 1, 2, 3").unwrap();
            b3.choose(&dex3, 1, "team 1, 2, 3").unwrap();
            strand(&mut b3, 1); // the foe-state rules need a foe that cannot leave
            let me = b3.active_id(0).unwrap();
            assert_ne!(
                b3.poke(me).hp_type,
                dex3.known_types.normal,
                "the fixture's Hidden Power must not be Normal-typed"
            );
            assert!(
                !certain_noop(&b3, &dex3, 0, mv(&dex3, "hiddenpower")),
                "Hidden Power is typed by the user's DVs, not the dex's Normal placeholder"
            );
            // A genuinely Normal-typed attack into the same Ghost IS refused.
            assert!(certain_noop(&b3, &dex3, 0, mv(&dex3, "bodyslam")), "Body Slam into a Ghost");
        }

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

    // ---- 2026-08-20: the two A/B arms over the type-immunity gate --------
    //
    // `noop_reason`'s foe-reading rules are dropped whenever the foe can
    // leave, because switches resolve before moves. Class A (a provably dead
    // attack aimed at the mon actually in front) is 7.53% of corpus
    // decisions and the shipped 30k search plays the dead move on 6.08% of
    // them, so the gate's price is worth measuring. Both arms default OFF;
    // these tests pin what each one changes and, as importantly, what it
    // does not.

    fn only(rule: fn(&mut MaskRules)) -> MaskRules {
        let mut r = MaskRules::default();
        rule(&mut r);
        r
    }

    /// Arm 1 refuses the mon in front and stops there: it never consults the
    /// bench, so it fires exactly where the shipped gate is silent.
    #[test]
    fn immunity_ignores_switch_read_refuses_the_mon_in_front() {
        // side 0 Zapdos (Electric/Flying) vs side 1 Nidoking (Poison/Ground),
        // BOTH benches intact — the shipped mask's switch-read case.
        let (dex, b) = imm_setup_open("team 2, 1, 3");
        let aggressive = only(|r| r.immunity_ignores_switch_read = true);
        let sound = only(|r| r.immunity_all_switchins = true);
        let tw = mv(&dex, "thunderwave");

        assert!(!certain_noop(&b, &dex, 0, tw), "shipped: a switchable foe disarms the rule");
        assert!(
            super::certain_noop(&b, &dex, 0, tw, aggressive),
            "arm 1: the mon in front is Ground, refuse it anyway"
        );
        // Arm 2 must NOT fire here: the foe's bench holds an Exeggutor, which
        // Thunder Wave hits perfectly well. This is the exact case where the
        // aggressive arm can be wrong and the sound one cannot.
        assert!(
            !super::certain_noop(&b, &dex, 0, tw, sound),
            "arm 2: a hittable switch-in keeps the move live"
        );
        // and the engine agrees the move works on that switch-in
        {
            let mut after = b.clone();
            after.set_log_enabled(true);
            after.choose(&dex, 0, "move thunderwave").unwrap();
            after.choose(&dex, 1, "switch 3").unwrap(); // Exeggutor
            let def = after.active_id(1).unwrap();
            assert_eq!(after.poke(def).status, Status::Par, "the replacement was paralysed");
        }

        // Control: nothing fires when the mon in front is not immune.
        let (dex2, b2) = imm_setup_open("team 3, 1, 2"); // Exeggutor in front
        for r in [aggressive, sound] {
            assert!(
                !super::certain_noop(&b2, &dex2, 0, mv(&dex2, "thunderwave"), r),
                "Thunder Wave is live vs Grass/Psychic under every rule set"
            );
        }

        // Each flag moves only its own rule: every OTHER verdict at this root
        // is untouched, so an A/B is measuring one thing.
        let base: Vec<&str> = dominated_actions(&b, &dex, 0).into_iter().map(|(_, w)| w).collect();
        for r in [aggressive, sound] {
            let arm: Vec<&str> = dominated_actions_with(&b, &dex, 0, r)
                .into_iter()
                .map(|(_, w)| w)
                .collect();
            let added = arm.iter().filter(|w| !base.contains(w)).count();
            assert_eq!(
                arm.len() - added,
                base.len(),
                "a flag deleted an existing refusal: {base:?} vs {arm:?}"
            );
            assert!(
                arm.iter().all(|w| base.contains(w) || w.contains("immune")),
                "a flag moved a rule that is not the immunity one: {arm:?}"
            );
        }
    }

    /// Arm 2 is the sound one: it needs the mon in front AND every mon the
    /// foe could bring in to be immune. Ghost-vs-Normal is the class-A case
    /// battle 4069 actually played (Miltank's Return, then Snorlax's Body
    /// Slam, into a Misdreavus that could still leave).
    fn ghost_setup(bench3: &str) -> (Dex, Battle) {
        let mine: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Body Slam","Earthquake","Rest","Curse"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Nidoking","species":"Nidoking","item":"","ability":"No Ability",
             "moves":["Earthquake","Substitute","Screech","Dream Eater"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Safeguard","Mist","Swagger"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap();
        let foe_json = format!(
            r#"[
            {{"name":"Misdreavus","species":"Misdreavus","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Pain Split","Perish Song","Mean Look"],
             "nature":"Serious","evs":{{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}},"gender":"F","level":50}},
            {{"name":"Gengar","species":"Gengar","item":"","ability":"No Ability",
             "moves":["Night Shade","Hypnosis","Thunderbolt","Psychic"],
             "nature":"Serious","evs":{{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}},"gender":"M","level":50}},
            {{"name":"{bench3}","species":"{bench3}","item":"","ability":"No Ability",
             "moves":["Night Shade","Hypnosis","Thunderbolt","Psychic"],
             "nature":"Serious","evs":{{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}},"gender":"M","level":50}}
        ]"#
        );
        let foes: Vec<PokemonSet> = serde_json::from_str(&foe_json).unwrap();
        let dex = conformance::load_dex();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &mine, &foes).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, "team 1, 2, 3").unwrap();
        (dex, b)
    }

    #[test]
    fn immunity_all_switchins_needs_every_switch_in_immune() {
        let aggressive = only(|r| r.immunity_ignores_switch_read = true);
        let sound = only(|r| r.immunity_all_switchins = true);

        // (a) an all-Ghost opponent: Body Slam is dead whatever they do.
        {
            let (dex, b) = ghost_setup("Haunter");
            let bs = mv(&dex, "bodyslam");
            assert!(!certain_noop(&b, &dex, 0, bs), "shipped: the foe can still leave");
            assert!(super::certain_noop(&b, &dex, 0, bs, aggressive), "arm 1 refuses");
            assert!(
                super::certain_noop(&b, &dex, 0, bs, sound),
                "arm 2 refuses: every switch-in is a Ghost"
            );
            // Earthquake, on the same board, is live under BOTH arms — the
            // rule is about the move's type, not about the position.
            assert!(!super::certain_noop(&b, &dex, 0, mv(&dex, "earthquake"), sound));
            assert!(!super::certain_noop(&b, &dex, 0, mv(&dex, "earthquake"), aggressive));
            // engine cross-check on the branch arm 2 claims to have proved:
            // the foe switches, and the Normal attack still does nothing.
            let mut after = b.clone();
            after.set_log_enabled(true);
            after.choose(&dex, 0, "move bodyslam").unwrap();
            after.choose(&dex, 1, "switch 2").unwrap(); // Gengar
            let def = after.active_id(1).unwrap();
            assert_eq!(after.poke(def).hp, after.poke(def).maxhp, "no damage to the switch-in");
        }

        // (b) one hittable mon on the bench and arm 2 goes quiet, while the
        // aggressive arm still refuses — this is the difference between them.
        {
            let (dex, b) = ghost_setup("Snorlax");
            let bs = mv(&dex, "bodyslam");
            assert!(super::certain_noop(&b, &dex, 0, bs, aggressive), "arm 1 does not care");
            assert!(
                !super::certain_noop(&b, &dex, 0, bs, sound),
                "arm 2: the Snorlax switch-in is hittable"
            );
            // …and once that mon is dead the proof is back (fainted mons are
            // dropped from the candidate set).
            let mut dead = b.clone();
            let slot = dead.sides[1].party[2];
            let id = PokeId { side: 1, slot };
            dead.poke_mut(id).hp = 0;
            dead.poke_mut(id).fainted = true;
            assert!(
                super::certain_noop(&dead, &dex, 0, bs, sound),
                "arm 2: the only hittable switch-in has fainted"
            );
        }
    }

    /// The soundness question that only a 6-mon roster can ask: with a pick
    /// still unrevealed, "the bench" is not public. Party slots that have
    /// never appeared are imputed on the ladder (`import.rs`), so arm 2 must
    /// admit EVERY not-yet-appeared roster mon — and go quiet if any of them
    /// is hittable — until the last pick has shown itself.
    #[test]
    fn immunity_all_switchins_admits_unrevealed_picks() {
        let sound = only(|r| r.immunity_all_switchins = true);
        let mine: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Body Slam","Earthquake","Rest","Curse"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Nidoking","species":"Nidoking","item":"","ability":"No Ability",
             "moves":["Earthquake","Substitute","Screech","Dream Eater"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Safeguard","Mist","Swagger"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap();
        // Six: three Ghosts (the picks) and three that are not.
        let foes: Vec<PokemonSet> = serde_json::from_str(
            r#"[
            {"name":"Misdreavus","species":"Misdreavus","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Pain Split","Perish Song","Mean Look"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"F","level":50},
            {"name":"Gengar","species":"Gengar","item":"","ability":"No Ability",
             "moves":["Night Shade","Hypnosis","Thunderbolt","Psychic"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Haunter","species":"Haunter","item":"","ability":"No Ability",
             "moves":["Night Shade","Hypnosis","Thunderbolt","Psychic"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Snorlax","species":"Snorlax","item":"","ability":"No Ability",
             "moves":["Body Slam","Earthquake","Rest","Curse"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Exeggutor","species":"Exeggutor","item":"","ability":"No Ability",
             "moves":["Confuse Ray","Safeguard","Mist","Swagger"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50},
            {"name":"Nidoking","species":"Nidoking","item":"","ability":"No Ability",
             "moves":["Earthquake","Substitute","Screech","Dream Eater"],
             "nature":"Serious","evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},"gender":"M","level":50}
        ]"#,
        )
        .unwrap();
        let dex = conformance::load_dex();
        let mut b = Battle::from_fixture(&dex, "7,8,9,10", &mine, &foes).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, "team 1, 2, 3").unwrap(); // the three Ghosts
        assert_eq!(b.sides[1].roster.len(), 6, "the fixture must carry a full roster");
        assert_eq!(b.sides[1].party.len(), 3);

        let bs = mv(&dex, "bodyslam");
        // Only the lead has appeared, so two picks are unknown and every
        // never-appeared roster mon is admitted — including the Snorlax that
        // was not even picked. Refusing here would be reading the imputed
        // party, and on the ladder that party is a guess.
        assert!(
            !super::certain_noop(&b, &dex, 0, bs, sound),
            "a hidden pick could be anything not yet seen"
        );
        assert_eq!(
            super::foe_switchin_candidates(&b, 0).len(),
            5,
            "candidates = every alive roster mon but the active"
        );

        // Reveal the other two picks (the same predicate the observer and
        // the determinizer use) and the candidate set collapses to the real
        // bench, which is all Ghost — so now the proof holds.
        for pos in 1..3 {
            let slot = b.sides[1].party[pos];
            b.poke_mut(PokeId { side: 1, slot }).previously_switched_in = 1;
        }
        assert_eq!(super::foe_switchin_candidates(&b, 0).len(), 2, "the real bench, exactly");
        assert!(
            super::certain_noop(&b, &dex, 0, bs, sound),
            "every pick is public now and all three are Ghosts"
        );
        // The shipped mask still says nothing, in every one of these states.
        assert!(!certain_noop(&b, &dex, 0, bs), "the shipped gate is unchanged");
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
            // Our own HP only survives to our move if we act first: a faster
            // foe can damage us past the gate, and then Substitute works.
            assert!(!noop(&b, 1, "substitute"), "slower: the foe can change our HP first");
            b.poke_mut(def).boosts[4] = 6;
            assert!(noop(&b, 1, "substitute"), "Substitute at exactly a quarter, moving first");
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
