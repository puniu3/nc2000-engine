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
//!
//! M6 (`Playout::Heavy`): rollouts are ε-greedy max-damage instead of
//! uniform, truncated a few turns past the rollout start, and scored by the
//! weighted static eval (`eval.rs`) instead of raw HP fractions. The M5
//! configuration survives untouched as `Playout::Uniform` (same PRNG draw
//! order) so gate measurements compare against the real baseline.

use std::collections::HashMap;

use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::eval::{self, EvalWeights};
use crate::rng::SplitMix64;

/// Rollout policy + leaf evaluation (M6).
#[derive(Clone, Debug)]
pub enum Playout {
    /// M5 baseline: uniform-random playout to the horizon, HP-fraction leaf
    /// eval. Kept bit-identical (same PRNG draw order) as the reference the
    /// M6 gate measures against.
    Uniform,
    /// M6 heavy playout: ε-greedy max-damage policy, truncated `turns` past
    /// the rollout start, weighted static eval at the cutoff.
    Heavy { eps: f64, turns: u16, weights: EvalWeights },
}

impl Playout {
    pub fn heavy() -> Playout {
        Playout::Heavy { eps: 0.2, turns: 8, weights: EvalWeights::default() }
    }
}

#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Simulations per decision.
    pub iterations: u32,
    /// UCB1 exploration constant.
    pub c: f64,
    /// Tree horizon: turns beyond the current one before a simulation is cut
    /// off and scored statically even inside the selection phase.
    pub horizon: u16,
    /// Rollout policy + leaf eval.
    pub playout: Playout,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig { iterations: 1000, c: 1.0, horizon: 100, playout: Playout::heavy() }
    }
}

impl MctsConfig {
    /// The M5 agent (uniform full rollouts, HP-fraction eval).
    pub fn uniform(iterations: u32, c: f64) -> Self {
        MctsConfig { iterations, c, horizon: 100, playout: Playout::Uniform }
    }
}

#[derive(Default)]
pub(crate) struct ActStats {
    pub(crate) n: u32,
    pub(crate) w: f64,
}

pub(crate) type Joint = (Option<SearchChoice>, Option<SearchChoice>);

pub(crate) struct Node {
    pub(crate) stats: [HashMap<SearchChoice, ActStats>; 2],
    pub(crate) children: HashMap<Joint, usize>,
}

impl Node {
    pub(crate) fn new() -> Self {
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
                break self.leaf_eval(sim, dex);
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
        let cutoff = match &self.cfg.playout {
            Playout::Uniform => turn_cap,
            Playout::Heavy { turns, .. } => turn_cap.min(sim.turn.saturating_add(*turns)),
        };
        loop {
            if let Some(o) = sim.outcome() {
                return outcome_reward(o);
            }
            if sim.turn > cutoff {
                return self.leaf_eval(sim, dex);
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = sim.legal_choices(dex, s);
                if !cs.is_empty() {
                    picks[s] = Some(self.rollout_pick(sim, dex, s, &cs));
                }
            }
            sim.apply_choices(dex, picks)
                .expect("legal_choices produced an illegal choice");
        }
    }

    fn rollout_pick(
        &mut self,
        sim: &Battle,
        dex: &Dex,
        side: usize,
        cs: &[SearchChoice],
    ) -> SearchChoice {
        let eps = match &self.cfg.playout {
            Playout::Uniform => return cs[self.rng.below(cs.len())],
            Playout::Heavy { eps, .. } => *eps,
        };
        if cs.len() == 1 {
            return cs[0];
        }
        if self.rng.next_f64() < eps {
            return cs[self.rng.below(cs.len())];
        }
        // M5/M6 historical agents keep the no-switch rollout bit-identical.
        greedy_pick(sim, dex, side, cs, &mut self.rng, false)
    }

    fn leaf_eval(&self, sim: &Battle, dex: &Dex) -> f64 {
        match &self.cfg.playout {
            Playout::Uniform => hp_eval(sim),
            Playout::Heavy { weights, .. } => eval::eval_leaf(sim, dex, weights),
        }
    }
}

/// One playout from `sim` to termination/cutoff under `playout`, returning
/// the side-0 reward. Free-function twin of `MctsAgent::rollout` (same
/// policy, same PRNG draw order) shared by the M7 agents; `MctsAgent` keeps
/// its own methods so the M5/M6 gate references stay bit-identical.
/// Rollout switch policy (M16c): when the active's best expected hit
/// removes less than TRIGGER of the foe per turn, the rollout may switch to
/// the bench mon with the strongest offense against the foe active, if that
/// beats staying by MARGIN. Humans switch at 22% of decisions (570-battle
/// corpus) and the no-switch rollout misvalued every such line — the tree's
/// switch arms were evaluated by playouts in which nobody ever switches.
const ROLLOUT_SWITCH_TRIGGER: f64 = 0.15;
const ROLLOUT_SWITCH_MARGIN: f64 = 0.10;

/// M16c-2 rollout pseudo-values for the format's core status moves, so the
/// greedy rollout can prefer them over a weak attack instead of scoring
/// every non-damaging move 0 (the corpus' top disagreement clusters: sleep,
/// Spikes, Curse/Rest — M16b). Rough per-turn HP-fraction equivalents,
/// accuracy-scaled, and 0 whenever the move would be a no-op (sleep clause,
/// existing status, spikes already laid, healing near full, boosting past
/// +2) — which also removes Rest-at-full-HP from greedy rollouts.
const PSEUDO_SLEEP: f64 = 0.30;
const PSEUDO_PAR: f64 = 0.10;
const PSEUDO_SPIKES: f64 = 0.15;
const PSEUDO_BOOST: f64 = 0.15;
const PSEUDO_HEAL: f64 = 0.25;

fn status_pseudo_score(
    sim: &Battle,
    dex: &Dex,
    side: usize,
    att: nc2000_engine::state::PokeId,
    def: nc2000_engine::state::PokeId,
    id: nc2000_engine::dex::MoveId,
) -> f64 {
    use nc2000_engine::dex::Accuracy;
    use nc2000_engine::state::Status;
    let ms = dex.move_static(id);
    let acc = match ms.accuracy {
        Accuracy::AlwaysHits => 1.0,
        Accuracy::Pct(p) => p as f64 / 100.0,
    };
    match ms.status.as_deref() {
        Some("slp") => {
            let clause_free = sim.sides[1 - side]
                .roster
                .iter()
                .all(|p| p.hp == 0 || p.status != Status::Slp);
            if clause_free && sim.poke(def).status == Status::None {
                return PSEUDO_SLEEP * acc;
            }
            return 0.0;
        }
        Some("par") => {
            if sim.poke(def).status == Status::None {
                return PSEUDO_PAR * acc;
            }
            return 0.0;
        }
        _ => {}
    }
    if ms.side_condition.as_deref() == Some("spikes") {
        let foe = &sim.sides[1 - side];
        let laid = dex.conds_id("spikes").is_some_and(|sid| foe.has_side_condition(sid));
        if foe.pokemon_left > 1 && !laid {
            return PSEUDO_SPIKES;
        }
        return 0.0;
    }
    if ms.heal.is_some() {
        let me = sim.poke(att);
        if (me.hp as f64) < 0.5 * me.maxhp as f64 {
            return PSEUDO_HEAL;
        }
        return 0.0;
    }
    match dex.moves.key(id) {
        "curse" => {
            let me = sim.poke(att);
            let ghost = dex.type_id("Ghost").is_some_and(|g| me.types.has(g));
            if !ghost && me.boosts[0] < 2 {
                PSEUDO_BOOST
            } else {
                0.0
            }
        }
        "swordsdance" => {
            if sim.poke(att).boosts[0] < 2 {
                PSEUDO_BOOST
            } else {
                0.0
            }
        }
        "amnesia" => {
            if sim.poke(att).boosts[2] < 2 {
                PSEUDO_BOOST
            } else {
                0.0
            }
        }
        "bellydrum" => {
            let me = sim.poke(att);
            if me.boosts[0] < 6 && (me.hp as f64) > 0.6 * me.maxhp as f64 {
                PSEUDO_BOOST + 0.10
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

pub(crate) fn playout_value(
    sim: &mut Battle,
    dex: &Dex,
    playout: &Playout,
    turn_cap: u16,
    rng: &mut SplitMix64,
    m16c: bool,
) -> f64 {
    let cutoff = match playout {
        Playout::Uniform => turn_cap,
        Playout::Heavy { turns, .. } => turn_cap.min(sim.turn.saturating_add(*turns)),
    };
    loop {
        if let Some(o) = sim.outcome() {
            return outcome_reward(o);
        }
        if sim.turn > cutoff {
            return match playout {
                Playout::Uniform => hp_eval(sim),
                Playout::Heavy { weights, .. } => eval::eval_leaf(sim, dex, weights),
            };
        }
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = sim.legal_choices(dex, s);
            if !cs.is_empty() {
                picks[s] = Some(playout_pick(sim, dex, playout, s, &cs, rng, m16c));
            }
        }
        sim.apply_choices(dex, picks)
            .expect("legal_choices produced an illegal choice");
    }
}

pub(crate) fn playout_pick(
    sim: &Battle,
    dex: &Dex,
    playout: &Playout,
    side: usize,
    cs: &[SearchChoice],
    rng: &mut SplitMix64,
    m16c: bool,
) -> SearchChoice {
    let eps = match playout {
        Playout::Uniform => return cs[rng.below(cs.len())],
        Playout::Heavy { eps, .. } => *eps,
    };
    if cs.len() == 1 {
        return cs[0];
    }
    if rng.next_f64() < eps {
        return cs[rng.below(cs.len())];
    }
    greedy_pick(sim, dex, side, cs, rng, m16c)
}

/// Greedy rollout move: strongest expected hit (never a voluntary switch);
/// forced switch → healthiest bench; team preview / all-zero scores →
/// uniform random.
fn greedy_pick(
    sim: &Battle,
    dex: &Dex,
    side: usize,
    cs: &[SearchChoice],
    rng: &mut SplitMix64,
    m16c: bool,
) -> SearchChoice {
    let att = sim.active_id(side);
    let def = sim.active_id(1 - side);
    // A self-KO move from the side's LAST mon is an unconditional immediate
    // loss (the user faints even on a miss; the Stadium Self-KO clause rules
    // the user the loser even when it takes the foe's last mon down), yet its
    // bp 250 tops every expected-damage ranking — pre-guard, rollouts from a
    // last-mon position suicided in EVERY playout and flattened the root to
    // all-zeros (2026-07-21 last-mon-Explosion report; selfko_probe).
    let last_mon = sim.sides[side].pokemon_left == 1;
    let suicidal = |c: &SearchChoice| match c {
        SearchChoice::Move(id) => last_mon && dex.move_static(*id).selfdestruct,
        _ => false,
    };
    let rand_non_suicidal = |rng: &mut SplitMix64| {
        if last_mon {
            let safe: Vec<SearchChoice> = cs.iter().copied().filter(|c| !suicidal(c)).collect();
            if !safe.is_empty() {
                return safe[rng.below(safe.len())];
            }
        }
        cs[rng.below(cs.len())]
    };
    if let (Some(att), Some(def)) = (att, def) {
        let mut best: Option<(SearchChoice, f64)> = None;
        for &c in cs {
            if let SearchChoice::Move(id) = c {
                if suicidal(&c) {
                    continue;
                }
                // Rollout policy always couples evasion (the accurate estimate).
                let mut score = eval::expected_hit_fraction(sim, dex, att, def, id, true);
                if score <= 0.0 && m16c {
                    score = status_pseudo_score(sim, dex, side, att, def, id);
                }
                if best.map_or(true, |(_, b)| score > b) {
                    best = Some((c, score));
                }
            }
        }
        // M16c voluntary switch: only from a bad matchup (weak best hit), to
        // the bench mon with the strongest offense vs the foe active, and
        // only when that clearly beats staying. Draws no rng, so rollouts
        // where the trigger never fires keep their exact streams.
        if m16c {
            if let Some((_, stay)) = best {
                if stay < ROLLOUT_SWITCH_TRIGGER {
                    let mut sw: Option<(SearchChoice, f64)> = None;
                    for &c in cs {
                        if let SearchChoice::Switch(pos) = c {
                            let s0 = &sim.sides[side];
                            let cand = nc2000_engine::state::PokeId {
                                side: side as u8,
                                slot: s0.party[(pos - 1) as usize],
                            };
                            let off = eval::best_hit_fraction(sim, dex, cand, def, true);
                            if sw.as_ref().map_or(true, |&(_, b)| off > b) {
                                sw = Some((c, off));
                            }
                        }
                    }
                    if let Some((c, off)) = sw {
                        if off > stay + ROLLOUT_SWITCH_MARGIN {
                            return c;
                        }
                    }
                }
            }
        }
        if let Some((c, score)) = best {
            if score > 0.0 {
                return c;
            }
            // only status/unknowable moves: fall through to random
            return rand_non_suicidal(rng);
        }
    }
    // forced switch: healthiest bench target
    let hp_frac = |pos: u8| {
        let s = &sim.sides[side];
        let p = &s.roster[s.party[(pos - 1) as usize] as usize];
        p.hp as f64 / p.maxhp as f64
    };
    if cs.iter().any(|c| matches!(c, SearchChoice::Switch(_))) {
        return cs
            .iter()
            .copied()
            .max_by(|a, b| {
                let f = |c: &SearchChoice| match c {
                    SearchChoice::Switch(pos) => hp_frac(*pos),
                    _ => -1.0,
                };
                f(a).total_cmp(&f(b))
            })
            .unwrap();
    }
    rand_non_suicidal(rng)
}

impl Agent for MctsAgent {
    fn name(&self) -> String {
        match &self.cfg.playout {
            Playout::Uniform => format!("mcts5:{}:{}", self.cfg.iterations, self.cfg.c),
            Playout::Heavy { eps, turns, .. } => {
                format!("mcts:{}:{}:{}:{}", self.cfg.iterations, self.cfg.c, eps, turns)
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

pub(crate) fn outcome_reward(o: Outcome) -> f64 {
    match o {
        Outcome::P1Win => 1.0,
        Outcome::P2Win => 0.0,
        Outcome::Tie => 0.5,
    }
}

/// Horizon cutoff evaluation: mean party HP fraction differential, squashed
/// into [0.25, 0.75] so it never outranks a real win/loss.
pub(crate) fn hp_eval(b: &Battle) -> f64 {
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
pub(crate) fn select_ucb(
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
