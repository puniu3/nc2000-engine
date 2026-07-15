//! DUCT MCTS (decoupled UCT for simultaneous moves), open-loop over the
//! stochastic engine.
//!
//! Why open-loop: engine transitions are chance events (damage rolls, crits,
//! secondary effects), so a history node does not correspond to one state —
//! even the *request kind* at a node can differ between iterations (a
//! stochastic KO turns a move request into a forced switch). Nodes therefore
//! key statistics by `SearchChoice` maps, selection considers only the
//! actions legal in the *current* simulation, and every iteration re-simulates
//! from a fresh root clone with a fresh PRNG seed (`reseed`) so chance is
//! resampled — the determinized-playout pattern the M3 API was built for.
//!
//! Decoupled UCT: at each node every side owing a choice runs an independent
//! UCB1 selection over its own action stats; the joint action indexes the
//! child edge. Rewards are from side 0's perspective (win 1 / tie 0.5 /
//! loss 0); side 1 backs up `1 - r`.

use std::collections::HashMap;

use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::rng::SplitMix64;

#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Simulations per decision.
    pub iterations: u32,
    /// UCB1 exploration constant.
    pub c: f64,
    /// Rollout horizon: turns beyond the current one before cutting off and
    /// scoring by HP fraction.
    pub horizon: u16,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig { iterations: 1000, c: 1.0, horizon: 100 }
    }
}

#[derive(Default)]
struct ActStats {
    n: u32,
    w: f64,
}

type Joint = (Option<SearchChoice>, Option<SearchChoice>);

struct Node {
    stats: [HashMap<SearchChoice, ActStats>; 2],
    children: HashMap<Joint, usize>,
}

impl Node {
    fn new() -> Self {
        Node { stats: [HashMap::new(), HashMap::new()], children: HashMap::new() }
    }
}

pub struct MctsAgent {
    pub cfg: MctsConfig,
    rng: SplitMix64,
}

impl MctsAgent {
    pub fn new(cfg: MctsConfig, seed: u64) -> Self {
        MctsAgent { cfg, rng: SplitMix64::new(seed) }
    }

    fn run_iteration(
        &mut self,
        nodes: &mut Vec<Node>,
        sim: &mut Battle,
        dex: &Dex,
        turn_cap: u16,
    ) {
        let mut path: Vec<(usize, Joint)> = Vec::new();
        let mut node_idx = 0usize;

        // ---- selection / expansion
        let reward0 = loop {
            if let Some(o) = sim.outcome() {
                break outcome_reward(o);
            }
            if sim.turn > turn_cap {
                break hp_eval(sim);
            }
            let mut joint: Joint = (None, None);
            for s in 0..2 {
                let legal = sim.legal_choices(dex, s);
                if legal.is_empty() {
                    continue;
                }
                let a = select_ucb(&nodes[node_idx], s, &legal, self.cfg.c, &mut self.rng);
                if s == 0 {
                    joint.0 = Some(a);
                } else {
                    joint.1 = Some(a);
                }
            }
            sim.apply_choices(dex, [joint.0, joint.1])
                .expect("legal_choices produced an illegal choice");
            path.push((node_idx, joint));
            match nodes[node_idx].children.get(&joint) {
                Some(&child) => node_idx = child,
                None => {
                    let child = nodes.len();
                    nodes.push(Node::new());
                    nodes[node_idx].children.insert(joint, child);
                    break self.rollout(sim, dex, turn_cap);
                }
            }
        };

        // ---- backprop (decoupled: each side's chosen action, own reward)
        for (ni, joint) in path {
            let node = &mut nodes[ni];
            if let Some(a) = joint.0 {
                let e = node.stats[0].entry(a).or_default();
                e.n += 1;
                e.w += reward0;
            }
            if let Some(a) = joint.1 {
                let e = node.stats[1].entry(a).or_default();
                e.n += 1;
                e.w += 1.0 - reward0;
            }
        }
    }

    fn rollout(&mut self, sim: &mut Battle, dex: &Dex, turn_cap: u16) -> f64 {
        loop {
            if let Some(o) = sim.outcome() {
                return outcome_reward(o);
            }
            if sim.turn > turn_cap {
                return hp_eval(sim);
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = sim.legal_choices(dex, s);
                if !cs.is_empty() {
                    picks[s] = Some(cs[self.rng.below(cs.len())]);
                }
            }
            sim.apply_choices(dex, picks)
                .expect("legal_choices produced an illegal choice");
        }
    }
}

impl Agent for MctsAgent {
    fn name(&self) -> String {
        format!("mcts:{}:{}", self.cfg.iterations, self.cfg.c)
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
        let mut root = battle.clone();
        root.set_log_enabled(false);
        let turn_cap = root.turn.saturating_add(self.cfg.horizon);

        let mut nodes = vec![Node::new()];
        for _ in 0..self.cfg.iterations {
            let mut sim = root.clone();
            sim.reseed(self.rng.next());
            self.run_iteration(&mut nodes, &mut sim, dex, turn_cap);
        }

        // robust child: most-visited root action for our side
        choices
            .iter()
            .copied()
            .max_by_key(|c| nodes[0].stats[side].get(c).map(|s| s.n).unwrap_or(0))
            .unwrap()
    }
}

fn outcome_reward(o: Outcome) -> f64 {
    match o {
        Outcome::P1Win => 1.0,
        Outcome::P2Win => 0.0,
        Outcome::Tie => 0.5,
    }
}

/// Horizon cutoff evaluation: mean party HP fraction differential, squashed
/// into [0.25, 0.75] so it never outranks a real win/loss.
fn hp_eval(b: &Battle) -> f64 {
    let f = |s: usize| {
        let side = &b.sides[s];
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for &slot in side.party.iter() {
            let p = &side.roster[slot as usize];
            sum += (p.hp.max(0)) as f64 / p.maxhp as f64;
            cnt += 1.0;
        }
        if cnt == 0.0 {
            0.0
        } else {
            sum / cnt
        }
    };
    0.5 + (f(0) - f(1)) * 0.25
}

/// UCB1 over the actions legal *now*; unvisited legal actions first
/// (uniformly at random among them).
fn select_ucb(
    node: &Node,
    side: usize,
    legal: &[SearchChoice],
    c: f64,
    rng: &mut SplitMix64,
) -> SearchChoice {
    let stats = &node.stats[side];
    let untried: Vec<SearchChoice> = legal
        .iter()
        .copied()
        .filter(|a| stats.get(a).map(|s| s.n).unwrap_or(0) == 0)
        .collect();
    if !untried.is_empty() {
        return untried[rng.below(untried.len())];
    }
    let total: u32 = legal.iter().map(|a| stats[a].n).sum();
    let ln_total = (total as f64).ln();
    let mut best = legal[0];
    let mut best_v = f64::NEG_INFINITY;
    for &a in legal {
        let s = &stats[&a];
        let v = s.w / s.n as f64 + c * (ln_total / s.n as f64).sqrt();
        if v > best_v {
            best_v = v;
            best = a;
        }
    }
    best
}
