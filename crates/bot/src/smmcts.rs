//! M7 mixed strategies: state-keyed tree + regret-matching equilibrium at
//! the root simultaneous decision, root play sampling the average strategy.
//!
//! Three pieces, each earning its place by measurement (all numbers: 60
//! games vs maxdamage, seed 11):
//!
//! 1. **State-keyed tree.** Nodes are keyed by `Battle::search_key`
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
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::mcts::{outcome_reward, Playout, RolloutRules};
use crate::rng::SplitMix64;

pub use crate::endgame_root::{
    reply_risk_and_mistake_ev, tactical_desperation_bonus, EndgameActionMetrics,
    EndgameDecisionInfo, EndgameRegime, EndgameRootStats,
};
pub(crate) use crate::endgame_root::cmp_visits_then_mean;
pub use crate::mask::dominated_actions;
pub(crate) use crate::mask::{dominated_reason, public_noop_reason};

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
    /// HP buckets for the node key (`Battle::search_key`); 0 = exact
    /// keys (measured weaker — see the module doc).
    pub hp_buckets: i64,
    /// Full-width RM+ sweeps over the estimated root matrix.
    pub solve_sweeps: u32,
    /// Leaf tactical settlement & evaluation mode.
    pub settlement: crate::accounting::LeafSettlementMode,
}

impl RmConfig {
    pub fn rollout_rules(&self) -> RolloutRules {
        RolloutRules {
            perish_combo: true,
            baton_pass: true,
            settlement: self.settlement,
        }
    }
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
            settlement: crate::accounting::LeafSettlementMode::BlindSpotLeaf,
        }
    }
}

pub(crate) struct Node {
    /// Legal actions per side at this state (empty = side owes nothing; <= 6 for all move/switch states).
    pub(crate) acts: [nc2000_engine::smallvec::SmallVec<[SearchChoice; 6]>; 2],
    /// Per-action sample counts (UCB1).
    pub(crate) n: [nc2000_engine::smallvec::SmallVec<[u32; 6]>; 2],
    /// Per-action reward sums (UCB1).
    pub(crate) w: [nc2000_engine::smallvec::SmallVec<[f64; 6]>; 2],
    /// Team-preview node (always UCB1 + argmax).
    pub(crate) preview: bool,
}

pub enum SearchTrace<'a> {
    /// With multiple legal actions, visits include this selection; rewards exclude its backpropagation.
    Choice {
        battle: &'a Battle,
        node: usize,
        actions: &'a [Vec<SearchChoice>; 2],
        visits: &'a [Vec<u32>; 2],
        rewards: &'a [Vec<f64>; 2],
        chosen: [Option<SearchChoice>; 2],
    },
    Leaf {
        battle: &'a Battle,
        rng: &'a SplitMix64,
        rollout: bool,
    },
    Result {
        battle: &'a Battle,
        reward0: f64,
    },
}

impl Node {
    pub(crate) fn at(sim: &mut Battle, dex: &Dex) -> Node {
        use nc2000_engine::smallvec::{smallvec, SmallVec};
        let mut acts = [SmallVec::new(), SmallVec::new()];
        sim.legal_choices_into(dex, 0, &mut acts[0]);
        sim.legal_choices_into(dex, 1, &mut acts[1]);
        let preview = acts
            .iter()
            .any(|a| matches!(a.first(), Some(SearchChoice::Team(_))));
        Node {
            n: [smallvec![0; acts[0].len()], smallvec![0; acts[1].len()]],
            w: [smallvec![0.0; acts[0].len()], smallvec![0.0; acts[1].len()]],
            preview,
            acts,
        }
    }

    /// Blend a raw child trajectory return `raw_val0` toward the empirical-Bayes minimax value
    /// estimate `\hat{V}^*(v) = \bar{R}(v) + (\mu_0^*(v) - \bar{R}(v)) - (\mu_1^*(v) - \bar{R}_1(v))`
    /// of this internal node. This removes both Side 0's UCB exploration tax and Side 1's UCB
    /// exploration blunders when backing up to parent nodes, while remaining a strict contraction
    /// in `[0, 1]`.
    pub(crate) fn backed_up_value(&self, raw_val0: f64) -> f64 {
        let n0: u32 = self.n[0].iter().sum();
        let n1: u32 = self.n[1].iter().sum();
        let n = n0.max(n1);
        if n < 2 {
            return raw_val0;
        }
        let r_bar0 = if n0 > 0 {
            self.w[0].iter().sum::<f64>() / f64::from(n0)
        } else if n1 > 0 {
            1.0 - (self.w[1].iter().sum::<f64>() / f64::from(n1))
        } else {
            return raw_val0;
        };
        const TAU: f64 = 1.5;
        let gain0 = if self.acts[0].len() > 1 && n0 >= 2 {
            let mu0 = self.n[0]
                .iter()
                .zip(self.w[0].iter())
                .filter(|&(&na, _)| na > 0)
                .map(|(&na, &wa)| (wa + TAU * r_bar0) / (f64::from(na) + TAU))
                .fold(r_bar0, f64::max);
            (mu0 - r_bar0).max(0.0)
        } else {
            0.0
        };
        let blunder1 = if self.acts[1].len() > 1 && n1 >= 2 {
            let r_bar1 = self.w[1].iter().sum::<f64>() / f64::from(n1);
            let mu1 = self.n[1]
                .iter()
                .zip(self.w[1].iter())
                .filter(|&(&nb, _)| nb > 0)
                .map(|(&nb, &wb)| (wb + TAU * r_bar1) / (f64::from(nb) + TAU))
                .fold(r_bar1, f64::max);
            (mu1 - r_bar1).max(0.0)
        } else {
            0.0
        };
        let v_star = (r_bar0 + gain0 - blunder1).clamp(0.000_002, 0.999_998);
        let lambda = 0.85 * (f64::from(n) / (f64::from(n) + 6.0));
        ((1.0 - lambda) * raw_val0 + lambda * v_star).clamp(0.000_002, 0.999_998)
    }
}

/// Zero-allocation tree node for the 256-byte `CompactBattle` search core.
/// Action codes use `CompactBattle` encoding (`0..4` = move slot, `4..7` = party slot switch);
/// waiting sides (`request == 0`) have `len[s] == 0`.
#[derive(Clone, Copy)]
pub(crate) struct CompactNode {
    pub(crate) acts: [[u8; 6]; 2],
    pub(crate) len: [u8; 2],
    pub(crate) n: [[u32; 6]; 2],
    pub(crate) w: [[f64; 6]; 2],
}

impl CompactNode {
    #[inline]
    pub(crate) fn at(
        cb: &nc2000_engine::compact::CompactBattle,
        table: &nc2000_engine::compact::MatchTable,
    ) -> Self {
        let mut acts = [[0u8; 6]; 2];
        let mut len = [0u8; 2];
        if cb.outcome == 0 {
            for s in 0..2usize {
                if cb.sides[s].request == 0 {
                    continue;
                }
                let mut buf = [0u8; 7];
                let k = cb.legal_actions(table, s, &mut buf).min(6);
                acts[s][..k].copy_from_slice(&buf[..k]);
                len[s] = k as u8;
            }
        }
        Self {
            acts,
            len,
            n: [[0; 6]; 2],
            w: [[0.0; 6]; 2],
        }
    }

    /// Verify 1:1 root action order and choice identity against `Battle::legal_choices_into`.
    pub(crate) fn matches_battle_acts(
        &self,
        cb: &nc2000_engine::compact::CompactBattle,
        table: &nc2000_engine::compact::MatchTable,
        dex: &Dex,
        battle_acts: &[nc2000_engine::smallvec::SmallVec<[SearchChoice; 6]>; 2],
        battle_party: [&[u8]; 2],
    ) -> bool {
        for s in 0..2usize {
            let k = self.len[s] as usize;
            if k != battle_acts[s].len() {
                return false;
            }
            for i in 0..k {
                if cb.decode_action(table, dex, s, self.acts[s][i], battle_party[s])
                    != Some(battle_acts[s][i])
                {
                    return false;
                }
            }
        }
        true
    }

    /// Identical empirical-Bayes minimax backup contraction to [`Node::backed_up_value`].
    #[inline]
    pub(crate) fn backed_up_value(&self, raw_val0: f64) -> f64 {
        let k0 = self.len[0] as usize;
        let k1 = self.len[1] as usize;
        let n0: u32 = self.n[0][..k0].iter().sum();
        let n1: u32 = self.n[1][..k1].iter().sum();
        let n = n0.max(n1);
        if n < 2 {
            return raw_val0;
        }
        let r_bar0 = if n0 > 0 {
            self.w[0][..k0].iter().sum::<f64>() / f64::from(n0)
        } else if n1 > 0 {
            1.0 - (self.w[1][..k1].iter().sum::<f64>() / f64::from(n1))
        } else {
            return raw_val0;
        };
        const TAU: f64 = 1.5;
        let gain0 = if k0 > 1 && n0 >= 2 {
            let mut mu0 = r_bar0;
            for a in 0..k0 {
                let na = self.n[0][a];
                if na > 0 {
                    let q = (self.w[0][a] + TAU * r_bar0) / (f64::from(na) + TAU);
                    if q > mu0 {
                        mu0 = q;
                    }
                }
            }
            (mu0 - r_bar0).max(0.0)
        } else {
            0.0
        };
        let blunder1 = if k1 > 1 && n1 >= 2 {
            let r_bar1 = self.w[1][..k1].iter().sum::<f64>() / f64::from(n1);
            let mut mu1 = r_bar1;
            for b in 0..k1 {
                let nb = self.n[1][b];
                if nb > 0 {
                    let q = (self.w[1][b] + TAU * r_bar1) / (f64::from(nb) + TAU);
                    if q > mu1 {
                        mu1 = q;
                    }
                }
            }
            (mu1 - r_bar1).max(0.0)
        } else {
            0.0
        };
        let v_star = (r_bar0 + gain0 - blunder1).clamp(0.000_002, 0.999_998);
        let lambda = 0.85 * (f64::from(n) / (f64::from(n) + 6.0));
        ((1.0 - lambda) * raw_val0 + lambda * v_star).clamp(0.000_002, 0.999_998)
    }
}

pub fn collapse_sleep_choices(b: &Battle, dex: &Dex, side: usize, choices: &mut Vec<SearchChoice>) {
    let Some(id) = b.active_id(side) else { return };
    let p = b.poke(id);
    if p.status != nc2000_engine::state::Status::Slp
        || p.status_state.get_int(nc2000_engine::state::DK::Time) <= 1
        || p.item.is_some_and(|item| {
            let si = Some(item);
            si == dex.known_items.mintberry || si == dex.known_items.miracleberry
        })
    {
        return;
    }
    let mut priorities = [false; 256];
    choices.retain(|action| {
        let SearchChoice::Move(id) = action else {
            return true;
        };
        let ms = dex.move_static(*id);
        if ms.sleep_usable || ms.callbacks.iter().any(|key| key == "beforeTurnCallback") {
            return true;
        }
        let priority = (i16::from(ms.priority) + 128) as usize;
        let first = !priorities[priority];
        priorities[priority] = true;
        first
    });
}

struct ProbeStats {
    /// Per-side probed action indices (into the root's action lists).
    support: [Vec<usize>; 2],
    n: Vec<u32>,
    v: Vec<f64>,
}

impl ProbeStats {
    fn new(support: [Vec<usize>; 2]) -> ProbeStats {
        let cells = support[0].len().max(1) * support[1].len().max(1);
        ProbeStats {
            support,
            n: vec![0; cells],
            v: vec![0.5; cells],
        }
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

#[inline]
pub(crate) fn build_search_cache(
    cfg: &RmConfig,
    dex: &Dex,
    b: &Battle,
) -> nc2000_engine::battle::BattleSearchCache {
    let buckets = (cfg.hp_buckets > 0).then_some(cfg.hp_buckets);
    nc2000_engine::battle::BattleSearchCache::new(b, dex, buckets, false)
}

#[inline]
pub(crate) fn build_search_cache_with_template(
    cfg: &RmConfig,
    dex: &Dex,
    b: &Battle,
    own_side: usize,
    template: &nc2000_engine::battle::BattleSearchCache,
) -> nc2000_engine::battle::BattleSearchCache {
    let buckets = (cfg.hp_buckets > 0).then_some(cfg.hp_buckets);
    nc2000_engine::battle::BattleSearchCache::new_with_own_side_template(
        b, dex, buckets, false, own_side, template,
    )
}

/// Public shim so the `key_shape` probe can hash a state the way the search
/// does without exposing the descent internals.
#[inline]
pub fn key_for_test(cfg: &RmConfig, dex: &Dex, b: &mut Battle) -> u64 {
    key_of(cfg, dex, b)
}

/// Node search key over canonicalized HP buckets.
#[inline]
pub(crate) fn key_of(cfg: &RmConfig, dex: &Dex, b: &mut Battle) -> u64 {
    key_of_cached(cfg, dex, b, None)
}

#[inline]
fn quick_claw_observable_fast(
    b: &Battle,
    dex: &Dex,
    cache: &nc2000_engine::battle::BattleSearchCache,
) -> bool {
    if !cache.any_quick_claw {
        return false;
    }
    let Some(item) = dex.known_items.quickclaw else {
        return false;
    };
    if cache.selected
        && b.started
        && b.turn > 0
        && b.request_state != nc2000_engine::state::RequestState::TeamPreview
        && b.queue.is_empty()
    {
        let mut has_fainted_holder = false;
        for side in &b.sides {
            for &slot in &side.party {
                if let Some(p) = side.roster.get(slot as usize) {
                    if p.item == Some(item) {
                        if !p.fainted || p.hp > 0 {
                            return true;
                        }
                        has_fainted_holder = true;
                    }
                }
            }
        }
        if !has_fainted_holder {
            return false;
        }
        return b.mid_turn
            || b.request_state != nc2000_engine::state::RequestState::Move
            || !b.is_idle_move_request();
    }
    b.quick_claw_roll_observable(dex)
}

#[inline]
pub(crate) fn key_of_with_claw_omit(
    cfg: &RmConfig,
    dex: &Dex,
    b: &mut Battle,
    force_omit_claw: bool,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> u64 {
    let buckets = (cfg.hp_buckets > 0).then_some(cfg.hp_buckets);
    if let Some(c) = cache {
        let omit_qc = force_omit_claw || !quick_claw_observable_fast(b, dex, c);
        let saved_qc = b.quick_claw_rolls;
        let any_qc = saved_qc[0] || saved_qc[1];
        if omit_qc && any_qc {
            b.quick_claw_rolls = [false, false];
        }
        let key = b.search_key_with_cache(dex, buckets, false, false, c);
        if omit_qc && any_qc {
            b.quick_claw_rolls = saved_qc;
        }
        debug_assert_eq!(
            key,
            if force_omit_claw {
                let q = b.quick_claw_rolls;
                b.quick_claw_rolls = [false, false];
                let k = b.search_key(buckets, false);
                b.quick_claw_rolls = q;
                k
            } else {
                b.search_key_canonical_claw(dex, buckets, false)
            }
        );
        return key;
    }
    if force_omit_claw {
        let saved_qc = b.quick_claw_rolls;
        b.quick_claw_rolls = [false, false];
        let key = b.search_key(buckets, false);
        b.quick_claw_rolls = saved_qc;
        return key;
    }
    b.search_key_canonical_claw(dex, buckets, false)
}

#[inline]
pub(crate) fn key_of_cached(
    cfg: &RmConfig,
    dex: &Dex,
    b: &mut Battle,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> u64 {
    key_of_with_claw_omit(cfg, dex, b, false, cache)
}

/// UCB1 (untried-first, then mean + c·sqrt(ln N / n)).
pub(crate) fn select_ucb(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    node: &mut Node,
    side: usize,
) -> usize {
    select_ucb_counts(cfg.c, rng, &mut node.n[side], &node.w[side])
}

pub(crate) fn select_ucb_counts(
    c: f64,
    rng: &mut SplitMix64,
    counts: &mut [u32],
    rewards: &[f64],
) -> usize {
    let k = counts.len();
    let mut n_untried = 0usize;
    for &cnt in counts.iter() {
        if cnt == 0 {
            n_untried += 1;
        }
    }
    let pick = if n_untried > 0 {
        let mut target = rng.below(n_untried);
        let mut chosen = 0;
        for (a, &cnt) in counts.iter().enumerate() {
            if cnt == 0 {
                if target == 0 {
                    chosen = a;
                    break;
                }
                target -= 1;
            }
        }
        chosen
    } else {
        let total: u32 = counts.iter().sum();
        let ln_total = (total as f64).ln();
        let mut best = 0;
        let mut best_v = f64::NEG_INFINITY;
        for a in 0..k {
            let (n, w) = (counts[a] as f64, rewards[a]);
            let v = w / n + c * (ln_total / n).sqrt();
            if v > best_v {
                best_v = v;
                best = a;
            }
        }
        best
    };
    counts[pick] += 1;
    pick
}

/// One iteration starting at node `start` (0 for the classic per-decision
/// tree; a per-determinization root for the M10b blind search). Per-side
/// `force_root` fixes that side's root action index instead of UCB
/// selection (the probe phase forces both sides; the blind search forces
/// only its own, globally-selected action); forced picks still feed the
/// root's per-action means, and everything below the root selects normally.
/// Returns the iteration's side-0 reward and writes the root joint's action
/// indices into `root_joint`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_iteration_cached(
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
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> f64 {
    run_iteration_with_root_child_value_cached(
        cfg,
        rng,
        nodes,
        table,
        sim,
        dex,
        turn_cap,
        start,
        force_root,
        root_joint,
        depth_out,
        &mut |sim, rng, rollout| default_leaf_cached(cfg, sim, dex, turn_cap, rng, rollout, cache),
        None,
        None,
        cache,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_iteration_with_leaf_cached(
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
    leaf: &mut (impl FnMut(&mut Battle, &mut SplitMix64, bool) -> f64 + ?Sized),
    trace: Option<&mut dyn FnMut(SearchTrace<'_>)>,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> f64 {
    run_iteration_with_root_child_value_cached(
        cfg, rng, nodes, table, sim, dex, turn_cap, start, force_root, root_joint, depth_out, leaf,
        trace, None, cache,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_iteration_with_root_child_value_cached(
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
    leaf: &mut (impl FnMut(&mut Battle, &mut SplitMix64, bool) -> f64 + ?Sized),
    mut trace: Option<&mut dyn FnMut(SearchTrace<'_>)>,
    mut root_child_value: Option<&mut dyn FnMut(&Battle) -> Option<f64>>,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> f64 {
    let mut path = nc2000_engine::smallvec::SmallVec::<[(usize, [Option<usize>; 2]); 16]>::new();
    let mut node_idx = start;

    // ---- selection until a leaf: terminal, horizon, or unexpanded state
    let reward0 = loop {
        let at_root = node_idx == start;
        let mut joint = [None, None];
        let mut selected = [None; 2];
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
                selected[s] = Some(ai);
                ai
            };
            joint[s] = Some(nodes[node_idx].acts[s][ai]);
            if at_root {
                root_joint[s] = ai;
            }
        }
        if selected.iter().any(Option::is_some) {
            path.push((node_idx, selected));
        }
        if let Some(trace) = trace.as_deref_mut() {
            let node = &nodes[node_idx];
            let t_acts = [node.acts[0].to_vec(), node.acts[1].to_vec()];
            let t_n = [node.n[0].to_vec(), node.n[1].to_vec()];
            let t_w = [node.w[0].to_vec(), node.w[1].to_vec()];
            trace(SearchTrace::Choice {
                battle: sim,
                node: node_idx,
                actions: &t_acts,
                visits: &t_n,
                rewards: &t_w,
                chosen: joint,
            });
        }
        if joint == [None, None] {
            // defensive: a rest point where neither side owes a choice
            // (never reached in practice — battles end instead)
            if let Some(trace) = trace.as_deref_mut() {
                trace(SearchTrace::Leaf {
                    battle: sim,
                    rng,
                    rollout: false,
                });
            }
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_battle(sim, sim.outcome()));
            break leaf(sim, rng, false);
        }
        sim.apply_choices(dex, joint)
            .expect("cached legal choice rejected (state_key collision?)");
        if let Some(o) = sim.outcome() {
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_battle(sim, Some(o)));
            break outcome_reward(o);
        }
        if let Some(v) = crate::eval::turn_limit_outcome_value(sim) {
            let inferred = if (v - 1.0).abs() < 1e-9 {
                Some(nc2000_engine::battle::Outcome::P1Win)
            } else if v.abs() < 1e-9 {
                Some(nc2000_engine::battle::Outcome::P2Win)
            } else {
                Some(nc2000_engine::battle::Outcome::Tie)
            };
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_battle(sim, inferred));
            break v;
        }
        if at_root {
            if let Some(value) = root_child_value
                .as_deref_mut()
                .and_then(|evaluate| evaluate(sim))
            {
                assert!((0. ..=1.).contains(&value));
                crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_battle(
                    sim,
                    sim.outcome(),
                ));
                break value;
            }
        }
        if sim.turn > turn_cap {
            if let Some(trace) = trace.as_deref_mut() {
                trace(SearchTrace::Leaf {
                    battle: sim,
                    rng,
                    rollout: false,
                });
            }
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_battle(sim, sim.outcome()));
            break leaf(sim, rng, false);
        }
        if cache.map_or(true, |c| c.any_quick_claw) {
            let s0 = sim.sides[0].active.is_some_and(|r| {
                sim.sides[0].roster[r as usize].speed
                    == nc2000_engine::state::QUICK_CLAW_ACTION_SPEED
            });
            let s1 = sim.sides[1].active.is_some_and(|r| {
                sim.sides[1].roster[r as usize].speed
                    == nc2000_engine::state::QUICK_CLAW_ACTION_SPEED
            });
            if (s0 || s1) && sim.is_idle_move_request() {
                let qc = sim.quick_claw_rolls;
                sim.quick_claw_rolls = [false, false];
                if s0 {
                    if let Some(id) = sim.active_id(0) {
                        sim.update_speed(dex, id);
                    }
                }
                if s1 {
                    if let Some(id) = sim.active_id(1) {
                        sim.update_speed(dex, id);
                    }
                }
                sim.quick_claw_rolls = qc;
            }
        }
        let key = key_of_cached(cfg, dex, sim, cache);
        match table.get(&key) {
            Some(&child) => {
                *depth_out += 1;
                node_idx = child;
            }
            None => {
                // expand exactly one node per iteration, then settle leaf
                let child = nodes.len();
                nodes.push(Node::at(sim, dex));
                table.insert(key, child);
                if let Some(trace) = trace.as_deref_mut() {
                    trace(SearchTrace::Leaf {
                        battle: sim,
                        rng,
                        rollout: true,
                    });
                }
                crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_battle(
                    sim,
                    sim.outcome(),
                ));
                break leaf(sim, rng, true);
            }
        }
    };

    let root_reward0 = back_up_path(nodes, &path, start, reward0);
    if let Some(trace) = trace {
        trace(SearchTrace::Result {
            battle: sim,
            reward0: root_reward0,
        });
    }
    root_reward0
}

fn back_up_path(
    nodes: &mut [Node],
    path: &[(usize, [Option<usize>; 2])],
    root: usize,
    reward0: f64,
) -> f64 {
    let mut value = reward0;
    let mut root_value = reward0;
    for &(index, actions) in path.iter().rev() {
        let node = &mut nodes[index];
        for (side, pick) in actions.into_iter().enumerate() {
            if let Some(action) = pick {
                node.w[side][action] += if side == 0 { value } else { 1.0 - value };
            }
        }
        if index == root {
            root_value = value;
        } else if value > 0.000_001 && value < 0.999_999 {
            value = node.backed_up_value(value);
        }
    }
    root_value
}

#[inline]
pub(crate) fn leaf_eval_cached(
    cfg: &RmConfig,
    sim: &Battle,
    dex: &Dex,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> f64 {
    let rc = cache.map(|c| &c.roster_compact);
    match &cfg.playout {
        Playout::Uniform => crate::mcts::hp_eval(sim),
        Playout::Heavy { weights, .. } => {
            if cfg.settlement.uses_relay_bellman() {
                crate::eval::eval_leaf_blindspot_cached(sim, dex, weights, rc)
            } else if cfg.settlement.uses_blindspot_leaf() {
                crate::eval::eval_leaf_blindspot_norelay_cached(sim, dex, weights, rc)
            } else {
                crate::eval::eval_leaf_cached(sim, dex, weights, rc)
            }
        }
    }
}

pub(crate) fn default_leaf(
    cfg: &RmConfig,
    sim: &mut Battle,
    dex: &Dex,
    turn_cap: u16,
    rng: &mut SplitMix64,
    rollout: bool,
) -> f64 {
    default_leaf_cached(cfg, sim, dex, turn_cap, rng, rollout, None)
}

#[inline]
pub(crate) fn compact_leaf_eval(
    cfg: &RmConfig,
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
) -> f64 {
    match &cfg.playout {
        Playout::Uniform => crate::mcts::compact_hp_eval(cb, table),
        Playout::Heavy { weights, .. } => {
            if cfg.settlement.uses_relay_bellman() {
                crate::eval::compact_eval_leaf_blindspot(cb, table, weights)
            } else if cfg.settlement.uses_blindspot_leaf() {
                crate::eval::compact_eval_leaf_blindspot_norelay(cb, table, weights)
            } else {
                crate::eval::compact_eval_leaf(cb, table, weights)
            }
        }
    }
}

#[inline]
pub(crate) fn compact_default_leaf(
    cfg: &RmConfig,
    cb: &mut nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    turn_cap: u16,
    rng: &mut SplitMix64,
    rollout: bool,
) -> f64 {
    if !rollout {
        crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
        return compact_leaf_eval(cfg, cb, table);
    }
    if let Some(v) = crate::eval::compact_turn_limit_outcome(cb, table) {
        let mut info = crate::mcts::SimEndInfo::from_compact(cb);
        if info.outcome == 0 {
            info.outcome = crate::mcts::value_to_outcome_code(v);
        }
        crate::mcts::set_last_sim_end(info);
        return v;
    }
    if cb.turn > turn_cap || matches!(cfg.playout, Playout::Heavy { turns: 0, .. }) {
        crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
        return compact_leaf_eval(cfg, cb, table);
    }
    let effective_cap = table
        .ruleset
        .turn_limit()
        .map_or(turn_cap, |lim| turn_cap.min(lim));
    let cutoff = match cfg.playout {
        Playout::Uniform => effective_cap,
        Playout::Heavy { turns, .. } => {
            effective_cap.min(cb.turn.saturating_add(turns.saturating_sub(1)))
        }
    };
    let rules = cfg.rollout_rules();
    if !rules.baton_pass {
        cb.sides[0].side_flags |= nc2000_engine::compact::SIDE0_DISABLE_BP_FIX;
    }
    crate::mcts::compact_playout_value(cb, table, &cfg.playout, cutoff, rng, rules)
}

#[inline]
fn back_up_compact_path(
    nodes: &mut [CompactNode],
    path: &[(usize, [Option<usize>; 2])],
    root: usize,
    reward0: f64,
) -> f64 {
    let mut value = reward0;
    let mut root_value = reward0;
    for &(index, actions) in path.iter().rev() {
        let node = &mut nodes[index];
        if let Some(action) = actions[0] {
            node.w[0][action] += value;
        }
        if let Some(action) = actions[1] {
            node.w[1][action] += 1.0 - value;
        }
        if index == root {
            root_value = value;
        } else if value > 0.000_001 && value < 0.999_999 {
            value = node.backed_up_value(value);
        }
    }
    root_value
}

/// Hot-path iteration natively over 256-byte `CompactBattle` with zero `Battle` allocation,
/// zero `from_battle` conversion, and identical UCB1 / HP-bucketed TT / empirical-Bayes backup.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compact_iteration(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    nodes: &mut Vec<CompactNode>,
    table: &mut FxHashMap<u64, usize>,
    cb: &mut nc2000_engine::compact::CompactBattle,
    match_table: &nc2000_engine::compact::MatchTable,
    turn_cap: u16,
    start: usize,
    force_root: [Option<usize>; 2],
    root_joint: &mut [usize; 2],
    depth_out: &mut u32,
) -> f64 {
    let mut path = nc2000_engine::smallvec::SmallVec::<[(usize, [Option<usize>; 2]); 16]>::new();
    let mut node_idx = start;

    let reward0 = loop {
        let at_root = node_idx == start;
        let mut joint = [7u8, 7u8];
        let mut selected = [None; 2];
        for s in 0..2usize {
            let k = nodes[node_idx].len[s] as usize;
            if k == 0 {
                continue;
            }
            let ai = if k == 1 {
                0
            } else {
                let ai = match force_root[s] {
                    Some(f) if at_root => {
                        nodes[node_idx].n[s][f] += 1;
                        f
                    }
                    _ => {
                        let node = &mut nodes[node_idx];
                        select_ucb_counts(cfg.c, rng, &mut node.n[s][..k], &node.w[s][..k])
                    }
                };
                selected[s] = Some(ai);
                ai
            };
            joint[s] = nodes[node_idx].acts[s][ai];
            if at_root {
                root_joint[s] = ai;
            }
        }
        if selected[0].is_some() || selected[1].is_some() {
            path.push((node_idx, selected));
        }
        if joint == [7, 7] {
            break compact_default_leaf(cfg, cb, match_table, turn_cap, rng, false);
        }
        cb.apply_actions(match_table, joint);
        if cb.outcome != 0 {
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
            break match cb.outcome {
                1 => 1.0,
                2 => 0.0,
                _ => 0.5,
            };
        }
        if let Some(v) = crate::eval::compact_turn_limit_outcome(cb, match_table) {
            let mut info = crate::mcts::SimEndInfo::from_compact(cb);
            if info.outcome == 0 {
                info.outcome = crate::mcts::value_to_outcome_code(v);
            }
            crate::mcts::set_last_sim_end(info);
            break v;
        }
        if cb.turn > turn_cap {
            break compact_default_leaf(cfg, cb, match_table, turn_cap, rng, false);
        }
        let omit_qc = !cb.quick_claw_observable(match_table);
        let key = cb.search_key(match_table, cfg.hp_buckets, omit_qc);
        match table.get(&key) {
            Some(&child) => {
                *depth_out += 1;
                node_idx = child;
            }
            None => {
                let child = nodes.len();
                nodes.push(CompactNode::at(cb, match_table));
                table.insert(key, child);
                break compact_default_leaf(cfg, cb, match_table, turn_cap, rng, true);
            }
        }
    };

    back_up_compact_path(nodes, &path, start, reward0)
}

/// Fast-path `CompactBattle` iteration with lazy `EndgameSession` leaf interception:
/// runs with ZERO `Battle` reset / `sync_to_battle` / `from_battle` overhead whenever the
/// reached leaf is ineligible for `EndgameSession` (e.g. 3v3 / 3v2 turns in WASM production),
/// materializing `Battle` via `sync_to_battle` ONLY on eligible small-endgame leaves (`<= 2v2`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_compact_iteration_with_endgame(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    nodes: &mut Vec<CompactNode>,
    table: &mut FxHashMap<u64, usize>,
    cb: &mut nc2000_engine::compact::CompactBattle,
    match_table: &nc2000_engine::compact::MatchTable,
    turn_cap: u16,
    start: usize,
    force_root: [Option<usize>; 2],
    root_joint: &mut [usize; 2],
    depth_out: &mut u32,
    session: &mut crate::endgame::EndgameSession,
    clock: &mut impl FnMut() -> f64,
    materialize: &mut impl FnMut(
        &nc2000_engine::compact::CompactBattle,
        &nc2000_engine::compact::MatchTable,
        &mut SplitMix64,
        bool,
        &mut crate::endgame::EndgameSession,
        &mut dyn FnMut() -> f64,
    ) -> f64,
) -> f64 {
    let mut path = nc2000_engine::smallvec::SmallVec::<[(usize, [Option<usize>; 2]); 16]>::new();
    let mut node_idx = start;

    let reward0 = loop {
        let at_root = node_idx == start;
        let mut joint = [7u8, 7u8];
        let mut selected = [None; 2];
        for s in 0..2usize {
            let k = nodes[node_idx].len[s] as usize;
            if k == 0 {
                continue;
            }
            let ai = if k == 1 {
                0
            } else {
                let ai = match force_root[s] {
                    Some(f) if at_root => {
                        nodes[node_idx].n[s][f] += 1;
                        f
                    }
                    _ => {
                        let node = &mut nodes[node_idx];
                        select_ucb_counts(cfg.c, rng, &mut node.n[s][..k], &node.w[s][..k])
                    }
                };
                selected[s] = Some(ai);
                ai
            };
            joint[s] = nodes[node_idx].acts[s][ai];
            if at_root {
                root_joint[s] = ai;
            }
        }
        if selected[0].is_some() || selected[1].is_some() {
            path.push((node_idx, selected));
        }
        if joint == [7, 7] {
            break if session.can_probe() && session.eligible_compact(cb, match_table) {
                materialize(cb, match_table, rng, false, session, clock)
            } else {
                compact_default_leaf(cfg, cb, match_table, turn_cap, rng, false)
            };
        }
        cb.apply_actions(match_table, joint);
        if cb.outcome != 0 {
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
            break match cb.outcome {
                1 => 1.0,
                2 => 0.0,
                _ => 0.5,
            };
        }
        if let Some(v) = crate::eval::compact_turn_limit_outcome(cb, match_table) {
            let mut info = crate::mcts::SimEndInfo::from_compact(cb);
            if info.outcome == 0 {
                info.outcome = crate::mcts::value_to_outcome_code(v);
            }
            crate::mcts::set_last_sim_end(info);
            break v;
        }
        if cb.turn > turn_cap {
            break if session.can_probe() && session.eligible_compact(cb, match_table) {
                materialize(cb, match_table, rng, false, session, clock)
            } else {
                compact_default_leaf(cfg, cb, match_table, turn_cap, rng, false)
            };
        }
        let omit_qc = !cb.quick_claw_observable(match_table);
        let key = cb.search_key(match_table, cfg.hp_buckets, omit_qc);
        match table.get(&key) {
            Some(&child) => {
                *depth_out += 1;
                node_idx = child;
            }
            None => {
                let child = nodes.len();
                nodes.push(CompactNode::at(cb, match_table));
                table.insert(key, child);
                break if session.can_probe() && session.eligible_compact(cb, match_table) {
                    materialize(cb, match_table, rng, true, session, clock)
                } else {
                    compact_default_leaf(cfg, cb, match_table, turn_cap, rng, true)
                };
            }
        }
    };

    back_up_compact_path(nodes, &path, start, reward0)
}

#[cold]
#[inline(never)]
fn emit_compact_choice_trace(
    cb: &nc2000_engine::compact::CompactBattle,
    match_table: &nc2000_engine::compact::MatchTable,
    dex: &Dex,
    battle_buf: &mut Battle,
    node_idx: usize,
    node: &CompactNode,
    joint: [u8; 2],
    trace: &mut dyn FnMut(SearchTrace<'_>),
) {
    cb.sync_to_battle(match_table, dex, battle_buf);
    let mut t_acts = [Vec::new(), Vec::new()];
    let mut t_n = [Vec::new(), Vec::new()];
    let mut t_w = [Vec::new(), Vec::new()];
    let mut t_chosen = [None, None];
    for s in 0..2usize {
        let k = node.len[s] as usize;
        t_acts[s].reserve(k);
        t_n[s].extend_from_slice(&node.n[s][..k]);
        t_w[s].extend_from_slice(&node.w[s][..k]);
        for i in 0..k {
            if let Some(sc) =
                cb.decode_action(match_table, dex, s, node.acts[s][i], &battle_buf.sides[s].party)
            {
                t_acts[s].push(sc);
            }
        }
        if joint[s] < 7 {
            t_chosen[s] =
                cb.decode_action(match_table, dex, s, joint[s], &battle_buf.sides[s].party);
        }
    }
    trace(SearchTrace::Choice {
        battle: battle_buf,
        node: node_idx,
        actions: &t_acts,
        visits: &t_n,
        rewards: &t_w,
        chosen: t_chosen,
    });
}

/// Callback-enabled `CompactBattle` iteration (`step_observed`, `step_with_leaf`,
/// `step_with_endgame`, `step_forced_joint_with_root_child_value`). Executes the exact same
/// `CompactBattle` tree transitions, keys, and RNG stream as [`run_compact_iteration`],
/// materializing `battle_buf` via `sync_to_battle` only at callback points.
#[allow(clippy::too_many_arguments)]
#[cold]
pub(crate) fn run_compact_iteration_with_hooks(
    cfg: &RmConfig,
    rng: &mut SplitMix64,
    nodes: &mut Vec<CompactNode>,
    table: &mut FxHashMap<u64, usize>,
    cb: &mut nc2000_engine::compact::CompactBattle,
    match_table: &nc2000_engine::compact::MatchTable,
    dex: &Dex,
    battle_buf: &mut Battle,
    turn_cap: u16,
    start: usize,
    force_root: [Option<usize>; 2],
    root_joint: &mut [usize; 2],
    depth_out: &mut u32,
    mut leaf: Option<&mut dyn FnMut(&mut Battle, &mut SplitMix64, bool) -> f64>,
    mut trace: Option<&mut dyn FnMut(SearchTrace<'_>)>,
    mut root_child_value: Option<&mut dyn FnMut(&Battle) -> Option<f64>>,
) -> f64 {
    let mut path = nc2000_engine::smallvec::SmallVec::<[(usize, [Option<usize>; 2]); 16]>::new();
    let mut node_idx = start;

    let reward0 = loop {
        let at_root = node_idx == start;
        let mut joint = [7u8, 7u8];
        let mut selected = [None; 2];
        for s in 0..2usize {
            let k = nodes[node_idx].len[s] as usize;
            if k == 0 {
                continue;
            }
            let ai = if k == 1 {
                0
            } else {
                let ai = match force_root[s] {
                    Some(f) if at_root => {
                        nodes[node_idx].n[s][f] += 1;
                        f
                    }
                    _ => {
                        let node = &mut nodes[node_idx];
                        select_ucb_counts(cfg.c, rng, &mut node.n[s][..k], &node.w[s][..k])
                    }
                };
                selected[s] = Some(ai);
                ai
            };
            joint[s] = nodes[node_idx].acts[s][ai];
            if at_root {
                root_joint[s] = ai;
            }
        }
        if selected[0].is_some() || selected[1].is_some() {
            path.push((node_idx, selected));
        }
        if let Some(tr) = trace.as_deref_mut() {
            emit_compact_choice_trace(
                cb,
                match_table,
                dex,
                battle_buf,
                node_idx,
                &nodes[node_idx],
                joint,
                tr,
            );
        }
        if joint == [7, 7] {
            if let Some(tr) = trace.as_deref_mut() {
                cb.sync_to_battle(match_table, dex, battle_buf);
                tr(SearchTrace::Leaf {
                    battle: battle_buf,
                    rng,
                    rollout: false,
                });
            }
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
            break if let Some(lf) = leaf.as_deref_mut() {
                cb.sync_to_battle(match_table, dex, battle_buf);
                lf(battle_buf, rng, false)
            } else {
                compact_default_leaf(cfg, cb, match_table, turn_cap, rng, false)
            };
        }
        cb.apply_actions(match_table, joint);
        if cb.outcome != 0 {
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
            break match cb.outcome {
                1 => 1.0,
                2 => 0.0,
                _ => 0.5,
            };
        }
        if let Some(v) = crate::eval::compact_turn_limit_outcome(cb, match_table) {
            let mut info = crate::mcts::SimEndInfo::from_compact(cb);
            if info.outcome == 0 {
                info.outcome = crate::mcts::value_to_outcome_code(v);
            }
            crate::mcts::set_last_sim_end(info);
            break v;
        }
        if at_root {
            if let Some(eval_child) = root_child_value.as_deref_mut() {
                cb.sync_to_battle(match_table, dex, battle_buf);
                if let Some(value) = eval_child(battle_buf) {
                    assert!((0. ..=1.).contains(&value));
                    crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
                    break value;
                }
            }
        }
        if cb.turn > turn_cap {
            if let Some(tr) = trace.as_deref_mut() {
                cb.sync_to_battle(match_table, dex, battle_buf);
                tr(SearchTrace::Leaf {
                    battle: battle_buf,
                    rng,
                    rollout: false,
                });
            }
            crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
            break if let Some(lf) = leaf.as_deref_mut() {
                cb.sync_to_battle(match_table, dex, battle_buf);
                lf(battle_buf, rng, false)
            } else {
                compact_default_leaf(cfg, cb, match_table, turn_cap, rng, false)
            };
        }
        let omit_qc = !cb.quick_claw_observable(match_table);
        let key = cb.search_key(match_table, cfg.hp_buckets, omit_qc);
        match table.get(&key) {
            Some(&child) => {
                *depth_out += 1;
                node_idx = child;
            }
            None => {
                let child = nodes.len();
                nodes.push(CompactNode::at(cb, match_table));
                table.insert(key, child);
                if let Some(tr) = trace.as_deref_mut() {
                    cb.sync_to_battle(match_table, dex, battle_buf);
                    tr(SearchTrace::Leaf {
                        battle: battle_buf,
                        rng,
                        rollout: true,
                    });
                }
                crate::mcts::set_last_sim_end(crate::mcts::SimEndInfo::from_compact(cb));
                break if let Some(lf) = leaf.as_deref_mut() {
                    cb.sync_to_battle(match_table, dex, battle_buf);
                    lf(battle_buf, rng, true)
                } else {
                    compact_default_leaf(cfg, cb, match_table, turn_cap, rng, true)
                };
            }
        }
    };

    let root_reward0 = back_up_compact_path(nodes, &path, start, reward0);
    if let Some(tr) = trace {
        cb.sync_to_battle(match_table, dex, battle_buf);
        tr(SearchTrace::Result {
            battle: battle_buf,
            reward0: root_reward0,
        });
    }
    root_reward0
}

#[inline]
pub(crate) fn default_leaf_cached(
    cfg: &RmConfig,
    sim: &mut Battle,
    dex: &Dex,
    turn_cap: u16,
    rng: &mut SplitMix64,
    rollout: bool,
    cache: Option<&nc2000_engine::battle::BattleSearchCache>,
) -> f64 {
    if rollout {
        crate::mcts::playout_value_cached(
            sim,
            dex,
            &cfg.playout,
            turn_cap,
            rng,
            cfg.rollout_rules(),
            cache.map(|c| &c.roster_compact),
        )
    } else {
        leaf_eval_cached(cfg, sim, dex, cache)
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
struct CompactSkuct {
    match_table: nc2000_engine::compact::MatchTable,
    root_cb: nc2000_engine::compact::CompactBattle,
    nodes: Vec<CompactNode>,
    table: FxHashMap<u64, usize>,
}

pub struct SkuctSearch {
    cfg: RmConfig,
    rng: SplitMix64,
    root: Battle,
    cache: nc2000_engine::battle::BattleSearchCache,
    scratch: Option<Box<Battle>>,
    turn_cap: u16,
    nodes: Vec<Node>,
    table: FxHashMap<u64, usize>,
    compact: Option<Box<CompactSkuct>>,
    done: u32,
    depth_sum: u64,
    /// Final-choice exclusions from `dominated_reason`. Only `best()` uses
    /// this mask; RmAgent's sampled policy reads unfiltered root statistics.
    root_dominated: [Vec<bool>; 2],
    endgame: [EndgameRootStats; 2],
    joint_n: Vec<u32>,
    joint_w0: Vec<f64>,
}


impl SkuctSearch {
    pub fn new(battle: &Battle, dex: &Dex, cfg: RmConfig, seed: u64) -> SkuctSearch {
        Self::with_rng(battle, dex, cfg, SplitMix64::new(seed))
    }

    fn with_rng(battle: &Battle, dex: &Dex, mut cfg: RmConfig, rng: SplitMix64) -> SkuctSearch {
        if matches!(
            battle.request_state,
            nc2000_engine::state::RequestState::TeamPreview
        ) && matches!(
            cfg.settlement,
            crate::accounting::LeafSettlementMode::BlindSpotLeaf
                | crate::accounting::LeafSettlementMode::BlindSpotLeafNoRelay
        ) {
            cfg.settlement = crate::accounting::LeafSettlementMode::LegacyRollout;
        }
        let mut root = battle.clone();
        root.set_log_enabled(false);
        let cache = build_search_cache(&cfg, dex, &root);
        let raw_cap = root.turn.saturating_add(cfg.horizon);
        let turn_cap = root
            .ruleset
            .turn_limit()
            .map_or(raw_cap, |lim| raw_cap.min(lim));
        let nodes = vec![Node::at(&mut root, dex)];
        let mut table = FxHashMap::default();
        table.insert(key_of_cached(&cfg, dex, &mut root, Some(&cache)), 0usize);
        let compact = if nc2000_engine::compact::is_compact_tree_supported(&root, dex) {
            let (match_table, root_cb) =
                nc2000_engine::compact::CompactBattle::from_battle_with_roster_cache(
                    &root,
                    dex,
                    Some(&cache.roster_compact),
                );
            let root_node = CompactNode::at(&root_cb, &match_table);
            if root_node.matches_battle_acts(
                &root_cb,
                &match_table,
                dex,
                &nodes[0].acts,
                [&root.sides[0].party, &root.sides[1].party],
            ) {
                let mut c_table = FxHashMap::default();
                let omit_qc = !root_cb.quick_claw_observable(&match_table);
                c_table.insert(
                    root_cb.search_key(&match_table, cfg.hp_buckets, omit_qc),
                    0usize,
                );
                Some(Box::new(CompactSkuct {
                    match_table,
                    root_cb,
                    nodes: vec![root_node],
                    table: c_table,
                }))
            } else {
                None
            }
        } else {
            None
        };
        let root_dominated = [0usize, 1].map(|s| {
            nodes[0].acts[s]
                .iter()
                .map(|&c| dominated_reason(&root, dex, s, c).is_some())
                .collect::<Vec<bool>>()
        });
        let endgame = [0usize, 1].map(|s| EndgameRootStats::new(&root, dex, s, &nodes[0].acts[s]));
        let cells = nodes[0].acts[0].len().max(1) * nodes[0].acts[1].len().max(1);
        SkuctSearch {
            cfg,
            rng,
            root,
            cache,
            scratch: None,
            turn_cap,
            nodes,
            table,
            compact,
            done: 0,
            depth_sum: 0,
            root_dominated,
            endgame,
            joint_n: vec![0u32; cells],
            joint_w0: vec![0.0f64; cells],
        }
    }

    #[inline]
    fn sync_compact_root_stats(&mut self) {
        if let Some(c) = &self.compact {
            let cn = &c.nodes[0];
            for s in 0..2usize {
                let k = cn.len[s] as usize;
                self.nodes[0].n[s][..k].copy_from_slice(&cn.n[s][..k]);
                self.nodes[0].w[s][..k].copy_from_slice(&cn.w[s][..k]);
            }
        }
    }

    fn record_root_endgame(&mut self, r0: f64, joint: [usize; 2]) {
        let end = crate::mcts::last_sim_end();
        let k0 = self.nodes[0].acts[0].len();
        let k1 = self.nodes[0].acts[1].len();
        if k0 > 0 && joint[0] < k0 {
            self.endgame[0].record(self.root.turn, 0, joint[0], r0, end);
        }
        if k1 > 0 && joint[1] < k1 {
            self.endgame[1].record(self.root.turn, 1, joint[1], 1.0 - r0, end);
        }
        let k1_stride = k1.max(1);
        let cell = joint[0] * k1_stride + joint[1];
        if cell < self.joint_n.len() {
            self.joint_n[cell] += 1;
            self.joint_w0[cell] += r0;
        }
    }

    fn reply_metrics(&self, side: usize, own: usize, n_i: u32, mean_i: f64) -> (f64, f64) {
        let foe = 1 - side;
        let k_foe = self.nodes[0].acts[foe].len();
        let k1_stride = self.nodes[0].acts[1].len().max(1);
        if k_foe <= 1 {
            return (0.0, mean_i);
        }
        reply_risk_and_mistake_ev(
            n_i,
            mean_i,
            (0..k_foe).map(|j| {
                let cell = if side == 0 {
                    own * k1_stride + j
                } else {
                    j * k1_stride + own
                };
                let n = self.joint_n.get(cell).copied().unwrap_or(0);
                let w0 = self.joint_w0.get(cell).copied().unwrap_or(0.0);
                let w = if side == 0 { w0 } else { n as f64 - w0 };
                (n, w)
            }),
        )
    }

    pub fn endgame_analysis(&self, side: usize) -> Option<EndgameDecisionInfo> {
        let node = &self.nodes[0];
        let sui = &self.root_dominated[side];
        let safe = (0..node.acts[side].len()).any(|a| !sui.get(a).copied().unwrap_or(false));
        let eligible = |a: usize| !safe || !sui.get(a).copied().unwrap_or(false);
        self.endgame[side].analyze(
            &node.acts[side],
            &node.n[side],
            &node.n[side],
            &node.w[side],
            eligible,
            |own, n_i, mean_i| self.reply_metrics(side, own, n_i, mean_i),
        )
    }

    /// One UCB iteration (reset root, fresh chance seed, select/expand/
    /// rollout/backprop). Returns the side-0 reward and the root joint's
    /// action indices — `RmAgent`'s late-tree stage-game seeding needs both.
    pub fn step_one(&mut self, dex: &Dex) -> (f64, [usize; 2]) {
        if let Some(c) = self.compact.as_mut() {
            let mut cb = c.root_cb;
            cb.prng_seed = self.rng.next();
            let mut joint = [0usize; 2];
            let mut depth = 0u32;
            let r = run_compact_iteration(
                &self.cfg,
                &mut self.rng,
                &mut c.nodes,
                &mut c.table,
                &mut cb,
                &c.match_table,
                self.turn_cap,
                0,
                [None, None],
                &mut joint,
                &mut depth,
            );
            self.sync_compact_root_stats();
            self.depth_sum += depth as u64;
            self.done += 1;
            self.record_root_endgame(r, joint);
            return (r, joint);
        }
        let mut sim = match self.scratch.take() {
            Some(mut sim) => {
                sim.reset_search_sim(&self.root, [true, true]);
                sim
            }
            None => Box::new(self.root.clone()),
        };
        std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
        sim.reseed(self.rng.next());
        let mut joint = [0usize; 2];
        let mut depth = 0u32;
        let r = run_iteration_cached(
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
            Some(&self.cache),
        );
        std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
        self.scratch = Some(sim);
        self.depth_sum += depth as u64;
        self.done += 1;
        self.record_root_endgame(r, joint);
        (r, joint)
    }

    /// One probe iteration with the root joint forced (`RmAgent`'s matrix
    /// estimation phase).
    fn step_forced(&mut self, dex: &Dex, force: [usize; 2]) -> f64 {
        if let Some(c) = self.compact.as_mut() {
            let mut cb = c.root_cb;
            cb.prng_seed = self.rng.next();
            let mut joint = [0usize; 2];
            let mut depth = 0u32;
            let r = run_compact_iteration(
                &self.cfg,
                &mut self.rng,
                &mut c.nodes,
                &mut c.table,
                &mut cb,
                &c.match_table,
                self.turn_cap,
                0,
                [Some(force[0]), Some(force[1])],
                &mut joint,
                &mut depth,
            );
            self.sync_compact_root_stats();
            self.depth_sum += depth as u64;
            self.done += 1;
            self.record_root_endgame(r, joint);
            return r;
        }
        let mut sim = match self.scratch.take() {
            Some(mut sim) => {
                sim.reset_search_sim(&self.root, [true, true]);
                sim
            }
            None => Box::new(self.root.clone()),
        };
        std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
        sim.reseed(self.rng.next());
        let mut joint = [0usize; 2];
        let mut depth = 0u32;
        let r = run_iteration_cached(
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
            Some(&self.cache),
        );
        std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
        self.scratch = Some(sim);
        self.depth_sum += depth as u64;
        self.done += 1;
        self.record_root_endgame(r, joint);
        r
    }

    /// `step` with a caller-supplied value at every expansion and horizon
    /// cutoff in place of the rollout/eval (`value::NetSearchAgent`).
    pub fn step_with_leaf(
        &mut self,
        dex: &Dex,
        n: u32,
        leaf: &mut dyn FnMut(&mut Battle, &mut SplitMix64, bool) -> f64,
    ) -> u32 {
        for _ in 0..n {
            if self.compact.is_some() {
                let mut sim = match self.scratch.take() {
                    Some(mut sim) => {
                        sim.reset_search_sim(&self.root, [false, false]);
                        sim
                    }
                    None => Box::new(self.root.clone()),
                };
                std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
                let (r, joint, depth) = {
                    let c = self.compact.as_mut().unwrap();
                    let mut cb = c.root_cb;
                    cb.prng_seed = self.rng.next();
                    let mut joint = [0usize; 2];
                    let mut depth = 0u32;
                    let r = run_compact_iteration_with_hooks(
                        &self.cfg,
                        &mut self.rng,
                        &mut c.nodes,
                        &mut c.table,
                        &mut cb,
                        &c.match_table,
                        dex,
                        &mut sim,
                        self.turn_cap,
                        0,
                        [None, None],
                        &mut joint,
                        &mut depth,
                        Some(leaf),
                        None,
                        None,
                    );
                    (r, joint, depth)
                };
                std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
                self.scratch = Some(sim);
                self.sync_compact_root_stats();
                self.depth_sum += depth as u64;
                self.done += 1;
                self.record_root_endgame(r, joint);
                continue;
            }
            let mut sim = match self.scratch.take() {
                Some(mut sim) => {
                    sim.reset_search_sim(&self.root, [true, true]);
                    sim
                }
                None => Box::new(self.root.clone()),
            };
            std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
            sim.reseed(self.rng.next());
            let mut joint = [0usize; 2];
            let mut depth = 0u32;
            let r = run_iteration_with_leaf_cached(
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
                leaf,
                None,
                Some(&self.cache),
            );
            std::mem::swap(&mut sim.listener_pool, &mut self.root.listener_pool);
            self.scratch = Some(sim);
            self.depth_sum += depth as u64;
            self.done += 1;
            self.record_root_endgame(r, joint);
        }
        self.done
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
        if let Some(c) = &self.compact {
            c.nodes.len()
        } else {
            self.nodes.len()
        }
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
        if !self.is_preview() {
            let safe = (0..node.acts[side].len()).any(|a| !sui.get(a).copied().unwrap_or(false));
            let eligible = |a: usize| !safe || !sui.get(a).copied().unwrap_or(false);
            return self.endgame[side]
                .select_best(
                    &node.acts[side],
                    &node.n[side],
                    &node.n[side],
                    &node.w[side],
                    eligible,
                    |own, n_i, mean_i| self.reply_metrics(side, own, n_i, mean_i),
                )
                .map(|a| node.acts[side][a]);
        }
        let n = &node.n[side];
        let w = &node.w[side];
        (0..node.acts[side].len())
            .filter(|&a| !sui.get(a).copied().unwrap_or(false))
            .max_by(|&a, &b| cmp_visits_then_mean(n, n, w, a, b))
            .or_else(|| (0..node.acts[side].len()).max_by(|&a, &b| cmp_visits_then_mean(n, n, w, a, b)))
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
        RmAgent {
            cfg,
            rng: SplitMix64::new(seed),
        }
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

        let acts = ts.nodes[0].acts[side].to_vec();

        // preview root (or argmax ablation): most-visited action, point mass
        if !mixed_root {
            let n = &ts.nodes[0].n[side];
            let w = &ts.nodes[0].w[side];
            let best = if !ts.is_preview() {
                ts.best(side)
                    .and_then(|c| acts.iter().position(|&a| a == c))
                    .unwrap_or_else(|| {
                        (0..acts.len())
                            .max_by(|&a, &b| cmp_visits_then_mean(n, n, w, a, b))
                            .unwrap()
                    })
            } else {
                (0..acts.len())
                    .max_by(|&a, &b| cmp_visits_then_mean(n, n, w, a, b))
                    .unwrap()
            };
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
        let imax = (0..probs.len())
            .max_by(|&a, &b| probs[a].total_cmp(&probs[b]).then_with(|| b.cmp(&a)))
            .unwrap();
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
            let n = &ts.nodes[0].n[side];
            let w = &ts.nodes[0].w[side];
            let best = (0..acts.len())
                .max_by(|&a, &b| cmp_visits_then_mean(n, n, w, a, b))
                .unwrap();
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
    idx.sort_by(|&a, &b| root.n[side][b].cmp(&root.n[side][a]).then_with(|| a.cmp(&b)));
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
                format!(
                    "skuct:{}:{}:{}",
                    self.cfg.iterations, self.cfg.c, self.cfg.hp_buckets
                )
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
        debug_assert_eq!(
            acts.as_slice(),
            choices,
            "root action set drifted from caller's choices"
        );
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
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;

    #[test]
    fn deep_child_value_reaches_root_across_inline_capacity() {
        use nc2000_engine::smallvec::smallvec;
        for depth in [2, 16, 17, 33] {
            let mut nodes: Vec<_> = (0..depth).map(|_| Node {
                acts: std::array::from_fn(|_| smallvec![SearchChoice::Switch(2), SearchChoice::Switch(3)]),
                n: [smallvec![1, 0], smallvec![1, 0]],
                w: [smallvec![0.0, 0.0], smallvec![0.0, 0.0]],
                preview: false,
            }).collect();
            let child = depth - 1;
            nodes[child].n = [smallvec![10, 0], smallvec![10, 0]];
            nodes[child].w = [smallvec![9.0, 0.0], smallvec![0.0, 0.0]];
            let path: Vec<_> = (0..depth).map(|i| (i, [Some(0), Some(0)])).collect();
            let leaf = 0.25;
            let root = back_up_path(&mut nodes, &path, 0, leaf);
            let expected = nodes[child].backed_up_value(leaf);
            assert!(expected > leaf);
            assert_eq!(root, expected, "depth {depth}");
            for node in &nodes[..child] {
                assert_eq!((node.w[0][0], node.w[1][0]), (expected, 1.0 - expected));
            }
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

    #[test]
    fn canonical_claw_key_preserves_observable_rolls_and_restores_state() {
        let (dex, root) = setup();
        let mut states = vec![(root.clone(), false)];
        for slot in [0, 1] {
            let mut holder = root.clone();
            holder.sides[0].roster[slot].item = dex.known_items.quickclaw;
            states.push((holder, true));
        }
        let mut unselected = root.clone();
        let mut holder = unselected.sides[0].roster[0].clone();
        holder.item = dex.known_items.quickclaw;
        unselected.sides[0].roster.push(holder);
        states.push((unselected.clone(), false));
        unselected.request_state = nc2000_engine::state::RequestState::TeamPreview;
        states.push((unselected, true));
        let mut fainted = states[2].0.clone();
        fainted.sides[0].roster[1].hp = 0;
        fainted.sides[0].roster[1].fainted = true;
        fainted.sides[0].pokemon_left -= 1;
        states.push((fainted.clone(), false));
        fainted.mid_turn = true;
        states.push((fainted, true));
        for (state, observable) in states {
            assert_eq!(state.quick_claw_roll_observable(&dex), observable);
            for hp_buckets in [0, 16] {
                let candidate = RmConfig {
                    hp_buckets,
                    ..Default::default()
                };
                let cache = build_search_cache(&candidate, &dex, &state);
                let mut a = state.clone();
                a.set_quick_claw_roll(false);
                let mut b = state.clone();
                b.set_quick_claw_roll(true);
                let before = [a.state_key128(), b.state_key128()];
                assert_eq!(
                    key_of(&candidate, &dex, &mut a) != key_of(&candidate, &dex, &mut b),
                    observable
                );
                assert_eq!(
                    key_of_cached(&candidate, &dex, &mut a, Some(&cache)),
                    key_of(&candidate, &dex, &mut a)
                );
                assert_eq!(
                    key_of_cached(&candidate, &dex, &mut b, Some(&cache)),
                    key_of(&candidate, &dex, &mut b)
                );
                assert_eq!(before, [a.state_key128(), b.state_key128()]);
            }
        }
    }

    #[test]
    fn canonical_claw_strips_stale_previous_turn_65535_speed_at_idle_continuation() {
        let (dex, mut b0) = setup();
        b0.sides[0].roster[0].item = dex.known_items.quickclaw;
        b0.sides[1].roster[0].hp = b0.sides[1].roster[0].maxhp;
        let mut b1 = b0.clone();
        // Simulate two trajectories reaching the exact same Turn 2 idle board
        // (`quick_claw_roll == false`), differing only in whether Turn 1 had
        // `quick_claw_roll == true` during Turn 1 Residual (`speed == 65535`).
        b0.set_quick_claw_roll(false);
        b0.choose(&dex, 0, "move reflect").unwrap();
        b0.choose(&dex, 1, "move confuseray").unwrap();
        b1.set_quick_claw_roll(true);
        b1.choose(&dex, 0, "move reflect").unwrap();
        b1.choose(&dex, 1, "move confuseray").unwrap();
        b0.set_quick_claw_roll(false);
        b1.set_quick_claw_roll(false);
        assert!(b0.is_idle_move_request() && b1.is_idle_move_request());
        assert_ne!(
            b0.sides[0].roster[0].speed,
            nc2000_engine::state::QUICK_CLAW_ACTION_SPEED
        );
        assert_eq!(
            b1.sides[0].roster[0].speed,
            nc2000_engine::state::QUICK_CLAW_ACTION_SPEED
        );
        let cfg = RmConfig::default();
        let cache = build_search_cache(&cfg, &dex, &b0);
        // Before stripping stale speed, raw hashes differ; after `run_iteration`'s
        // idle canonicalization, they merge and still produce identical transitions.
        let id = b1.active_id(0).unwrap();
        b1.update_speed(&dex, id);
        assert_eq!(
            key_of_cached(&cfg, &dex, &mut b0, Some(&cache)),
            key_of_cached(&cfg, &dex, &mut b1, Some(&cache))
        );
    }

    #[test]
    fn compact_skuct_activates_on_supported_ps_and_castle_and_falls_back_on_unsupported() {
        let (dex, b_ps) = setup();
        let cfg = RmConfig {
            iterations: 64,
            ..Default::default()
        };
        let mut s_ps_a = SkuctSearch::new(&b_ps, &dex, cfg.clone(), 99);
        let mut s_ps_b = SkuctSearch::new(&b_ps, &dex, cfg.clone(), 99);
        assert!(s_ps_a.compact.is_some(), "supported PS position must use CompactSkuct");
        s_ps_a.step(&dex, 64);
        s_ps_b.step(&dex, 64);
        assert_eq!(s_ps_a.visits(0), s_ps_b.visits(0));
        assert_eq!(s_ps_a.means(0), s_ps_b.means(0));
        assert!(s_ps_a.node_count() > 1);

        // Castle ruleset supported position also activates CompactSkuct
        let mut b_castle = b_ps.clone();
        b_castle.ruleset = nc2000_engine::rules::Ruleset::castle_tournament();
        let mut s_castle = SkuctSearch::new(&b_castle, &dex, cfg.clone(), 99);
        assert!(s_castle.compact.is_some(), "supported Castle position must use CompactSkuct");
        s_castle.step(&dex, 64);
        assert!(s_castle.node_count() > 1);

        // Unsupported move (Solar Beam) transparently falls back to Battle tree
        let mut b_unsup = b_ps.clone();
        let sb_id = dex.moves.id("solarbeam").unwrap();
        b_unsup.sides[0].roster[0].move_slots[0].id = sb_id;
        b_unsup.sides[0].roster[0].base_move_slots[0].id = sb_id;
        let mut s_unsup = SkuctSearch::new(&b_unsup, &dex, cfg.clone(), 99);
        assert!(s_unsup.compact.is_none(), "Solar Beam must fall back to Battle tree");
        s_unsup.step(&dex, 32);
        assert_eq!(s_unsup.visits(0).iter().sum::<u32>(), 32);
    }

}
