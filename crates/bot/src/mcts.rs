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

use std::cell::Cell;

use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::state::Battle;

use crate::agent::Agent;
use crate::eval::{self, EvalWeights};
use crate::rng::SplitMix64;

/// Summary of the state at the end of a single MCTS simulation (tree terminal or rollout cutoff).
#[derive(Clone, Copy, Debug, Default)]
pub struct SimEndInfo {
    pub turn: u16,
    /// 0 = non-terminal (cutoff/leaf), 1 = P1Win, 2 = P2Win, 3 = Tie
    pub outcome: u8,
    /// Living Pokemon counts `[side 0, side 1]` at simulation end.
    pub alive: [u8; 2],
}

impl SimEndInfo {
    #[inline]
    pub fn from_battle(b: &Battle, o: Option<Outcome>) -> Self {
        let outcome = match o {
            Some(Outcome::P1Win) => 1,
            Some(Outcome::P2Win) => 2,
            Some(Outcome::Tie) => 3,
            None => 0,
        };
        let alive = [
            b.sides[0].pokemon_left.max(0) as u8,
            b.sides[1].pokemon_left.max(0) as u8,
        ];
        SimEndInfo { turn: b.turn, outcome, alive }
    }

    #[inline]
    pub fn from_compact(cb: &nc2000_engine::compact::CompactBattle) -> Self {
        SimEndInfo {
            turn: cb.turn,
            outcome: cb.outcome,
            alive: [
                cb.sides[0].alive_mask.count_ones() as u8,
                cb.sides[1].alive_mask.count_ones() as u8,
            ],
        }
    }
}

thread_local! {
    static LAST_SIM_END: Cell<SimEndInfo> = const {
        Cell::new(SimEndInfo {
            turn: 0,
            outcome: 0,
            alive: [0, 0],
        })
    };
}

#[inline]
pub fn set_last_sim_end(info: SimEndInfo) {
    LAST_SIM_END.with(|c| c.set(info));
}

#[inline]
pub fn last_sim_end() -> SimEndInfo {
    LAST_SIM_END.with(|c| c.get())
}

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
    // FxHash maps (behavior-identical to SipHash ones: these are only ever
    // get/entry'd, never iterated, so the hasher cannot affect decisions).
    pub(crate) stats: [FxHashMap<SearchChoice, ActStats>; 2],
    pub(crate) children: FxHashMap<Joint, usize>,
}

impl Node {
    pub(crate) fn new() -> Self {
        Node { stats: Default::default(), children: Default::default() }
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
            if let Some(v) = eval::turn_limit_outcome_value(sim) {
                break v;
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
        playout_value(
            sim,
            dex,
            &self.cfg.playout,
            turn_cap,
            &mut self.rng,
            RolloutRules::default(),
        )
    }

    fn leaf_eval(&self, sim: &Battle, dex: &Dex) -> f64 {
        match &self.cfg.playout {
            Playout::Uniform => hp_eval(sim),
            Playout::Heavy { weights, .. } => eval::eval_leaf(sim, dex, weights),
        }
    }
}

/// Optional rules layered over the heavy rollout's ε-greedy pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutRules {
    /// Rollout sides play the Perish trap and its counter (`compact_perish_combo`),
    /// as well as escaping at Perish count 1 (`compact_perish_escape`).
    pub perish_combo: bool,
    pub baton_pass: bool,
    pub settlement: crate::accounting::LeafSettlementMode,
}

impl Default for RolloutRules {
    fn default() -> Self {
        RolloutRules {
            perish_combo: false,
            baton_pass: true,
            settlement: crate::accounting::LeafSettlementMode::LegacyRollout,
        }
    }
}

/// Unfinished rollouts return a leaf evaluation rather than a terminal outcome.
#[inline]
pub fn playout_value(
    sim: &mut Battle,
    dex: &Dex,
    playout: &Playout,
    turn_cap: u16,
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> f64 {
    playout_value_cached(sim, dex, playout, turn_cap, rng, rules, None)
}

#[inline]
fn value_to_outcome(v: f64) -> Outcome {
    if v >= 0.999_999 {
        Outcome::P1Win
    } else if v <= 0.000_001 {
        Outcome::P2Win
    } else {
        Outcome::Tie
    }
}

#[inline]
pub(crate) fn value_to_outcome_code(v: f64) -> u8 {
    if v >= 0.999_999 {
        1
    } else if v <= 0.000_001 {
        2
    } else {
        3
    }
}

pub fn playout_value_cached(
    sim: &mut Battle,
    dex: &Dex,
    playout: &Playout,
    turn_cap: u16,
    rng: &mut SplitMix64,
    rules: RolloutRules,
    roster_cache: Option<&nc2000_engine::compact::RosterCompactCache>,
) -> f64 {
    if let Some(v) = eval::turn_limit_outcome_value(sim) {
        let o = sim.outcome().unwrap_or_else(|| value_to_outcome(v));
        set_last_sim_end(SimEndInfo::from_battle(sim, Some(o)));
        return v;
    }
    // If either side is still in TeamPreview, advance that single step on `Battle`
    // before converting the 3v3 party state into the 256-byte `CompactBattle` kernel.
    while sim.sides[0].request_state() == Some(nc2000_engine::state::RequestKind::TeamPreview)
        || sim.sides[1].request_state() == Some(nc2000_engine::state::RequestKind::TeamPreview)
    {
        if let Some(v) = eval::turn_limit_outcome_value(sim) {
            let o = sim.outcome().unwrap_or_else(|| value_to_outcome(v));
            set_last_sim_end(SimEndInfo::from_battle(sim, Some(o)));
            return v;
        }
        if sim.turn > turn_cap {
            set_last_sim_end(SimEndInfo::from_battle(sim, None));
            return match playout {
                Playout::Uniform => hp_eval(sim),
                Playout::Heavy { weights, .. } => {
                    if rules.settlement.uses_relay_bellman() {
                        eval::eval_leaf_blindspot_cached(sim, dex, weights, roster_cache)
                    } else if rules.settlement.uses_blindspot_leaf() {
                        eval::eval_leaf_blindspot_norelay_cached(sim, dex, weights, roster_cache)
                    } else {
                        eval::eval_leaf_cached(sim, dex, weights, roster_cache)
                    }
                }
            };
        }
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = sim.legal_choices(dex, s);
            if !cs.is_empty() {
                picks[s] = Some(playout_pick(sim, dex, playout, s, &cs, rng, rules));
            }
        }
        sim.apply_choices(dex, picks)
            .expect("legal_choices produced an illegal choice");
    }

    if let Some(v) = eval::turn_limit_outcome_value(sim) {
        let o = sim.outcome().unwrap_or_else(|| value_to_outcome(v));
        set_last_sim_end(SimEndInfo::from_battle(sim, Some(o)));
        return v;
    }
    if sim.turn > turn_cap || matches!(playout, Playout::Heavy { turns: 0, .. }) {
        set_last_sim_end(SimEndInfo::from_battle(sim, None));
        return match playout {
            Playout::Uniform => hp_eval(sim),
            Playout::Heavy { weights, .. } => {
                if rules.settlement.uses_relay_bellman() {
                    eval::eval_leaf_blindspot_cached(sim, dex, weights, roster_cache)
                } else if rules.settlement.uses_blindspot_leaf() {
                    eval::eval_leaf_blindspot_norelay_cached(sim, dex, weights, roster_cache)
                } else {
                    eval::eval_leaf_cached(sim, dex, weights, roster_cache)
                }
            }
        };
    }

    let effective_cap = sim
        .ruleset
        .turn_limit()
        .map_or(turn_cap, |lim| turn_cap.min(lim));
    let cutoff = match playout {
        Playout::Uniform => effective_cap,
        Playout::Heavy { turns, .. } => {
            effective_cap.min(sim.turn.saturating_add(turns.saturating_sub(1)))
        }
    };

    let (table, mut cb) =
        nc2000_engine::compact::CompactBattle::from_battle_with_roster_cache(sim, dex, roster_cache);
    if !rules.baton_pass {
        cb.sides[0].side_flags |= nc2000_engine::compact::SIDE0_DISABLE_BP_FIX;
    }
    compact_playout_value(&mut cb, &table, playout, cutoff, rng, rules)
}

/// Execute the rollout and cutoff leaf evaluation directly on the 256-byte `CompactBattle` kernel.
#[inline]
pub fn compact_playout_value(
    cb: &mut nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    cutoff: u16,
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> f64 {
    let start_turn = cb.turn;
    let mut pivot_token = [1u8; 2];
    let effective_cutoff = table
        .ruleset
        .turn_limit()
        .map_or(cutoff, |lim| cutoff.min(lim));
    loop {
        if let Some(v) = eval::compact_turn_limit_outcome(cb, table) {
            let mut info = SimEndInfo::from_compact(cb);
            if info.outcome == 0 {
                info.outcome = value_to_outcome_code(v);
            }
            set_last_sim_end(info);
            return v;
        }
        if cb.turn > effective_cutoff {
            set_last_sim_end(SimEndInfo::from_compact(cb));
            return match playout {
                Playout::Uniform => compact_hp_eval(cb, table),
                Playout::Heavy { weights, .. } => {
                    if rules.settlement.uses_relay_bellman() {
                        eval::compact_eval_leaf_blindspot(cb, table, weights)
                    } else if rules.settlement.uses_blindspot_leaf() {
                        eval::compact_eval_leaf_blindspot_norelay(cb, table, weights)
                    } else {
                        eval::compact_eval_leaf(cb, table, weights)
                    }
                }
            };
        }
        let turn_offset = cb.turn.saturating_sub(start_turn).min(255) as u8;
        let mut picks = [7u8; 2];
        for s in 0..2usize {
            if cb.sides[s].request == 0 {
                continue;
            }
            let mut cs = [0u8; 7];
            let n = cb.legal_actions(table, s, &mut cs);
            if n > 0 {
                picks[s] = compact_playout_pick_ctx(
                    cb,
                    table,
                    playout,
                    s,
                    &cs[..n],
                    turn_offset,
                    &mut pivot_token,
                    rng,
                    rules,
                );
            }
        }
        apply_compact_actions_with_bp_target(cb, table, picks);
    }
}

/// Execute the exact shipped `CompactBattle` heavy rollout (`compact_playout_pick_ctx` with
/// Perish combo/escape, Baton Pass, Setup-Relay, and `linear.rs` endgame picker) up to `cutoff`,
/// then evaluate non-terminal cutoff leaves via caller-supplied `eval_leaf`.
#[inline]
pub(crate) fn compact_playout_with_leaf(
    cb: &mut nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    cutoff: u16,
    rng: &mut SplitMix64,
    rules: RolloutRules,
    mut eval_leaf: impl FnMut(
        &nc2000_engine::compact::CompactBattle,
        &nc2000_engine::compact::MatchTable,
    ) -> f64,
) -> f64 {
    let start_turn = cb.turn;
    let mut pivot_token = [1u8; 2];
    let effective_cutoff = table
        .ruleset
        .turn_limit()
        .map_or(cutoff, |lim| cutoff.min(lim));
    loop {
        if let Some(v) = eval::compact_turn_limit_outcome(cb, table) {
            let mut info = SimEndInfo::from_compact(cb);
            if info.outcome == 0 {
                info.outcome = value_to_outcome_code(v);
            }
            set_last_sim_end(info);
            return v;
        }
        if cb.turn > effective_cutoff {
            set_last_sim_end(SimEndInfo::from_compact(cb));
            return eval_leaf(cb, table);
        }
        let turn_offset = cb.turn.saturating_sub(start_turn).min(255) as u8;
        let mut picks = [7u8; 2];
        for s in 0..2usize {
            if cb.sides[s].request == 0 {
                continue;
            }
            let mut cs = [0u8; 7];
            let n = cb.legal_actions(table, s, &mut cs);
            if n > 0 {
                picks[s] = compact_playout_pick_ctx(
                    cb,
                    table,
                    playout,
                    s,
                    &cs[..n],
                    turn_offset,
                    &mut pivot_token,
                    rng,
                    rules,
                );
            }
        }
        apply_compact_actions_with_bp_target(cb, table, picks);
    }
}

/// Apply compact rollout actions while honoring directed Baton Pass receiver codes
/// (`((recv_slot + 1) << 4) | move_slot`) without touching `crates/engine`.
#[inline]
pub fn apply_compact_actions_with_bp_target(
    cb: &mut nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    picks: [u8; 2],
) {
    if picks[0] < 16 && picks[1] < 16 {
        cb.apply_actions(table, picks);
        return;
    }
    let raw_picks = [
        if picks[0] >= 16 { picks[0] & 3 } else { picks[0] },
        if picks[1] >= 16 { picks[1] & 3 } else { picks[1] },
    ];
    if table.ruleset.turn_limit().is_some_and(|lim| cb.turn >= lim) {
        cb.apply_actions(table, raw_picks);
        return;
    }
    let backup = *cb;
    let mut masked_bits = [0u8; 2];
    for s in 0..2usize {
        if picks[s] >= 16 && cb.sides[s].request == 1 {
            let recv_slot = ((picks[s] >> 4).wrapping_sub(1)).min(2);
            let act_slot = cb.sides[s].active_idx.min(2);
            let foe = 1 - s;
            let foe_act = cb.sides[foe].active_idx as usize;
            let foe_phazing = raw_picks[foe] < 4
                && foe_act < 3
                && table.move_specs[foe * 3 + foe_act][raw_picks[foe] as usize].special_kind
                    == nc2000_engine::compact::SpecialMoveKind::Phaze;
            let keep = (1u8 << act_slot) | (1u8 << recv_slot);
            if !foe_phazing && (cb.sides[s].alive_mask & (1u8 << recv_slot)) != 0 {
                masked_bits[s] = cb.sides[s].alive_mask & !keep;
                cb.sides[s].alive_mask &= keep;
            }
        }
    }
    cb.apply_actions(table, raw_picks);
    // Restore hidden alive bench bits; if the battle terminated (e.g. turn-limit judge) or
    // a directed receiver/passer fainted mid-turn while a 3rd mon was masked (avoiding
    // single-bench auto-switch), replay cleanly with unmasked alive_mask.
    let mut need_fallback = false;
    for s in 0..2usize {
        if masked_bits[s] != 0 {
            let recv_slot = ((picks[s] >> 4).wrapping_sub(1)).min(2) as usize;
            if cb.outcome != 0
                || cb.mons[s * 3 + recv_slot].hp() <= 0
                || cb.mons[s * 3 + backup.sides[s].active_idx as usize].hp() <= 0
            {
                need_fallback = true;
            } else {
                cb.sides[s].alive_mask |= masked_bits[s];
            }
        }
    }
    if need_fallback {
        *cb = backup;
        cb.apply_actions(table, raw_picks);
    }
}

#[inline]
pub(crate) fn compact_hp_eval(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
) -> f64 {
    let f = |s: usize| {
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for &slot_u8 in &cb.party_order(s) {
            let slot = slot_u8 as usize;
            let idx = s * 3 + slot;
            let maxhp = table.mons[idx].maxhp;
            if maxhp > 0 {
                sum += (cb.mons[idx].hp().max(0) as f64) / (maxhp as f64);
                cnt += 1.0;
            }
        }
        if cnt == 0.0 { 0.0 } else { sum / cnt }
    };
    0.5 + (f(0) - f(1)) * 0.25
}

#[inline]
fn compact_playout_pick(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    side: usize,
    cs: &[u8],
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> u8 {
    let mut pivot_token = [1u8; 2];
    compact_playout_pick_ctx(cb, table, playout, side, cs, 0, &mut pivot_token, rng, rules)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn compact_playout_pick_ctx(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    side: usize,
    cs: &[u8],
    turn_offset: u8,
    pivot_token: &mut [u8; 2],
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> u8 {
    let pick = compact_ordinary_pick(cb, table, playout, side, cs, turn_offset, pivot_token, rng, rules);
    if rules.perish_combo {
        if let Some(escape) = compact_perish_escape(cb, table, side, cs) {
            return escape;
        }
        if let Some(combo) = compact_perish_combo(cb, table, side, cs) {
            return combo;
        }
    }
    pick
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn compact_ordinary_pick(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    side: usize,
    cs: &[u8],
    turn_offset: u8,
    pivot_token: &mut [u8; 2],
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> u8 {
    let (eps, weights_opt) = match playout {
        Playout::Uniform => return cs[rng.below(cs.len())],
        Playout::Heavy { eps, weights, .. } => (*eps, Some(weights)),
    };
    if cs.len() == 1 {
        return cs[0];
    }
    if rng.next_f64() < eps {
        return cs[rng.below(cs.len())];
    }
    let mut eff_rules = rules;
    if turn_offset < 5
        && rules.baton_pass
        && rules.settlement.uses_relay_bellman()
        && (cb.sides[0].side_flags & nc2000_engine::compact::SIDE0_DISABLE_BP_FIX) == 0
    {
        if let Some(weights) = weights_opt {
            match crate::relay::rollout_relay_pick(
                cb,
                table,
                weights,
                side,
                cs,
                turn_offset,
                pivot_token,
                eps <= 0.0,
                rng,
            ) {
                crate::relay::RelayRolloutDecision::Chosen(act) => return act,
                crate::relay::RelayRolloutDecision::GateClosedSuppressLegacyBp => {
                    eff_rules.baton_pass = false;
                }
                crate::relay::RelayRolloutDecision::NotApplicable => {}
            }
        }
    }
    compact_greedy_pick(cb, table, playout, side, cs, rng, eff_rules)
}

#[inline]
fn compact_perish_escape(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    side: usize,
    cs: &[u8],
) -> Option<u8> {
    use nc2000_engine::compact::vol_bit;
    if !cs.iter().any(|&c| c < 4) {
        return None;
    }
    let att_slot = cb.sides[side].active_idx as usize;
    let att_idx = side * 3 + att_slot;
    if (cb.mons[att_idx].vol_flags & vol_bit::PERISH_SONG) == 0 || cb.mons[att_idx].perish_turns() != 1 {
        return None;
    }
    let foe_side = 1 - side;
    let foe_slot = cb.sides[foe_side].active_idx as usize;
    let foe_alive = (cb.sides[foe_side].alive_mask & (1 << foe_slot)) != 0
        && cb.mons[foe_side * 3 + foe_slot].hp() > 0;

    let mut best_act = None;
    let mut best_kept = f64::NEG_INFINITY;
    for &c in cs {
        if (4..7).contains(&c) {
            let slot = (c - 4) as usize;
            let idx = side * 3 + slot;
            let incoming = if foe_alive {
                cb.best_hit_fraction(table, foe_side, foe_slot, side, slot)
            } else {
                0.0
            };
            let kept = (cb.mons[idx].hp() as f64) / (table.mons[idx].maxhp.max(1) as f64)
                * (1.0 - incoming).max(0.0);
            if best_act.is_none() || kept >= best_kept {
                best_kept = kept;
                best_act = Some(c);
            }
        }
    }
    best_act
}

#[inline]
fn compact_perish_combo(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    side: usize,
    cs: &[u8],
) -> Option<u8> {
    use nc2000_engine::compact::{role_bit, vol_bit};
    let me_slot = cb.sides[side].active_idx as usize;
    let me_idx = side * 3 + me_slot;
    if (table.mons[me_idx].role_flags & role_bit::PERISH_PHAZE_TRAP) == 0 {
        return None;
    }
    let mut song = None;
    let mut phaze = None;
    let mut trap = None;
    for &c in cs {
        if c < 4 {
            let sk = table.move_specs[me_idx][c as usize].special_kind;
            if sk == nc2000_engine::compact::SpecialMoveKind::PerishSong && song.is_none() {
                song = Some(c);
            } else if sk == nc2000_engine::compact::SpecialMoveKind::Phaze && phaze.is_none() {
                phaze = Some(c);
            } else if sk == nc2000_engine::compact::SpecialMoveKind::MeanLook && trap.is_none() {
                trap = Some(c);
            }
        }
    }
    if song.is_none() && phaze.is_none() {
        return None;
    }
    let foe_side = 1 - side;
    let foe_slot = cb.sides[foe_side].active_idx as usize;
    let foe_idx = foe_side * 3 + foe_slot;
    let me = &cb.mons[me_idx];
    let foe = &cb.mons[foe_idx];
    if me.hp() <= 0 || foe.hp() <= 0 {
        return None;
    }
    let foe_bench = cb.sides[foe_side].alive_mask.count_ones() > 1;
    let me_trapped = me.is_trapped();
    if let Some(phaze_act) = phaze {
        let counting = (me.vol_flags & vol_bit::PERISH_SONG) != 0 && me.perish_turns() >= 2;
        if me_trapped && counting && foe_bench {
            return Some(phaze_act);
        }
    }
    let song_act = song?;
    if me_trapped
        || cb.sides[side].alive_mask.count_ones() < 2
        || (foe.vol_flags & vol_bit::PERISH_SONG) != 0
    {
        return None;
    }
    if cb.best_hit_fraction(table, side, me_slot, foe_side, foe_slot) >= 1.0 {
        return None;
    }
    if foe.is_trapped() || !foe_bench {
        return Some(song_act);
    }
    trap
}

#[inline]
fn compact_greedy_pick(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    side: usize,
    cs: &[u8],
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> u8 {
    let att_slot = cb.sides[side].active_idx as usize;
    let att_idx = side * 3 + att_slot;
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    let last_mon = cb.sides[side].alive_mask.count_ones() == 1;
    let suicidal = |act: u8| -> bool {
        act < 4 && last_mon && table.move_specs[att_idx][act as usize].is_selfdestruct
    };
    let rand_non_suicidal = |rng: &mut SplitMix64| -> u8 {
        if last_mon {
            let mut safe = [0u8; 7];
            let mut n_safe = 0usize;
            for &c in cs {
                if !suicidal(c) {
                    safe[n_safe] = c;
                    n_safe += 1;
                }
            }
            if n_safe > 0 {
                return safe[rng.below(n_safe)];
            }
        }
        cs[rng.below(cs.len())]
    };

    if cb.sides[side].request == 1
        && (cb.sides[side].alive_mask & (1 << att_slot)) != 0
        && (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
    {
        if let Some(la_pick) = crate::linear::compact_rollout_linear_pick(cb, table, side, cs) {
            return la_pick;
        }
        let mut best: Option<(u8, f64, f64)> = None;
        for &c in cs {
            if c < 4 {
                if suicidal(c) {
                    continue;
                }
                let m_slot = c as usize;
                let (raw_score, _accuracy) = cb.expected_hit_fraction_with_acc(
                    table, side, att_slot, def_side, def_slot, m_slot,
                );
                let score = raw_score;
                if best.map_or(true, |(_, old_score, _)| score > old_score) {
                    best = Some((c, score, raw_score));
                }
            }
        }
        // Landed Baton Pass receiver behind an intact Substitute (`!has_bp && team_has_bp`):
        // spend the passed Substitute to click Swords Dance / Growth once when active foe cannot
        // bypass Substitute (`Phaze`/`Haze`/`Encore`) and we cannot already OHKO.
        if rules.settlement.uses_relay_bellman()
            && (cb.sides[0].side_flags & nc2000_engine::compact::SIDE0_DISABLE_BP_FIX) == 0
            && (table.mons[att_idx].role_flags & nc2000_engine::compact::role_bit::BATON_PASS) == 0
            && ((table.mons[side * 3].role_flags
                | table.mons[side * 3 + 1].role_flags
                | table.mons[side * 3 + 2].role_flags)
                & nc2000_engine::compact::role_bit::BATON_PASS)
                != 0
        {
            use nc2000_engine::compact::{role_bit, vol_bit, SpecialMoveKind};
            let stay_score = best.map_or(0.0, |(_, s, _)| s);
            let me = &cb.mons[att_idx];
            let foe_idx = def_side * 3 + def_slot;
            if stay_score < 0.92
                && (me.vol_flags & vol_bit::SUBSTITUTE) != 0
                && (me.vol_flags & (vol_bit::ENCORE | vol_bit::PERISH_SONG)) == 0
                && me.boosts[0] < 2
                && me.boosts[2] < 2
                && (table.mons[foe_idx].role_flags & role_bit::PHAZE_OR_HAZE) == 0
            {
                let foe_has_encore = (0..table.mons[foe_idx].num_moves as usize).any(|mi| {
                    cb.mons[foe_idx].pp[mi] > 0
                        && table.move_specs[foe_idx][mi].special_kind == SpecialMoveKind::Encore
                });
                if !foe_has_encore {
                    for &c in cs {
                        if c < 4 {
                            let ms = &table.move_specs[att_idx][c as usize];
                            if ms.is_status
                                && ms.boosts_self
                                && ((ms.boosts[0].0 == 0 && ms.boosts[0].1 >= 2)
                                    || (ms.boosts[0].0 == 2 && ms.boosts[0].1 >= 1))
                            {
                                return c;
                            }
                        }
                    }
                }
            }
        }
        if rules.baton_pass
            && (cb.sides[0].side_flags & nc2000_engine::compact::SIDE0_DISABLE_BP_FIX) == 0
            && (table.mons[att_idx].role_flags
                & (nc2000_engine::compact::role_bit::BATON_PASS
                    | nc2000_engine::compact::role_bit::SWAGGER))
                != 0
        {
            use nc2000_engine::compact::{role_bit, timer_idx, vol_bit, SpecialMoveKind};
            use nc2000_engine::state::Status;
            let stay_score = best.map_or(0.0, |(_, s, _)| s);
            let me = &cb.mons[att_idx];
            let foe_idx = def_side * 3 + def_slot;
            let foe = &cb.mons[foe_idx];
            if stay_score < 0.85 {
                let has_sub = (me.vol_flags & vol_bit::SUBSTITUTE) != 0;
                let has_pos_boost =
                    me.boosts[0] > 0 || me.boosts[2] > 0 || me.boosts[4] > 0;
                let mut me_faster_opt: Option<bool> = None;
                // 1. If we already have a Substitute or boosts and a legal Baton Pass move, pass to a stronger bench threat.
                if (table.mons[att_idx].role_flags & role_bit::BATON_PASS) != 0
                    && (has_sub || has_pos_boost)
                {
                    let foe_asleep =
                        foe.status_raw() == Status::Slp as u8 && foe.status_counter() >= 2;
                    let can_pass = has_sub
                        || foe_asleep
                        || *me_faster_opt.get_or_insert_with(|| {
                            eval::compare_action_speed(cb, table, side, att_slot, def_side, def_slot)
                                == std::cmp::Ordering::Greater
                        });
                    if can_pass {
                        if let Some(&bp_c) = cs.iter().find(|&&c| {
                            c < 4
                                && table.move_specs[att_idx][c as usize].special_kind
                                    == SpecialMoveKind::BatonPass
                        }) {
                            let mut best_recv_hit = 0.0f64;
                            for slot in 0..3usize {
                                if slot == att_slot || (cb.sides[side].alive_mask & (1 << slot)) == 0 {
                                    continue;
                                }
                                let r_idx = side * 3 + slot;
                                let r_mon = &cb.mons[r_idx];
                                let r_st = r_mon.status_raw();
                                if r_mon.hp() * 2 <= table.mons[r_idx].maxhp.max(1)
                                    || r_st == Status::Slp as u8
                                    || r_st == Status::Frz as u8
                                {
                                    continue;
                                }
                                let mut tmp = *cb;
                                tmp.mons[r_idx].boosts = me.boosts;
                                let rh = tmp.best_hit_fraction(table, side, slot, def_side, def_slot);
                                if rh > best_recv_hit {
                                    best_recv_hit = rh;
                                }
                            }
                            if best_recv_hit > stay_score + 0.15 && best_recv_hit >= 0.42 {
                                return bp_c;
                            }
                        }
                    }
                }
                // 2. SwagSub: if faster against a confused foe (`timers[CONFUSION] >= 2`) and we carry both
                //    Substitute and Swagger, put up Substitute while the foe is confused.
                if (table.mons[att_idx].role_flags & role_bit::SWAGGER) != 0
                    && !has_sub
                    && (foe.vol_flags & vol_bit::CONFUSION) != 0
                    && foe.timers[timer_idx::CONFUSION] >= 2
                    && me.hp() * 2 > table.mons[att_idx].maxhp.max(1)
                    && *me_faster_opt.get_or_insert_with(|| {
                        eval::compare_action_speed(cb, table, side, att_slot, def_side, def_slot)
                            == std::cmp::Ordering::Greater
                    })
                {
                    if let Some(&sub_c) = cs
                        .iter()
                        .find(|&&c| c < 4 && table.move_specs[att_idx][c as usize].is_substitute)
                    {
                        return sub_c;
                    }
                }
            }
        }
        if let Some((c, score, _)) = best {
            if score > 0.0 {
                return c;
            }
            return rand_non_suicidal(rng);
        }
    }

    // Forced or self-switch target selection:
    // When the outgoing active Pokemon is alive (`hp > 0`, i.e. `self_switch` / Baton Pass),
    // score receivers by HP fraction + offensive effectiveness with passed boosts/Substitute;
    // when fainted (`hp <= 0`), keep exact healthiest-bench selection.
    let mut best_sw = None;
    let mut best_score = -1.0f64;
    let has_sub_or_spe = cb.mons[att_idx].boosts[4] > 0
        || (cb.mons[att_idx].vol_flags & nc2000_engine::compact::vol_bit::SUBSTITUTE) != 0;
    let passing_alive = rules.baton_pass
        && cb.sides[side].request == 2
        && att_slot < 3
        && cb.mons[att_idx].hp() > 0
        && (cb.mons[att_idx].boosts[0] > 0
            || cb.mons[att_idx].boosts[1] > 0
            || cb.mons[att_idx].boosts[2] > 0
            || cb.mons[att_idx].boosts[3] > 0
            || cb.mons[att_idx].boosts[6] > 0
            || has_sub_or_spe)
        && (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
        && cb.mons[def_side * 3 + def_slot].hp() > 0;
    if passing_alive && rules.settlement.uses_relay_bellman() {
        if let Playout::Heavy { weights, .. } = playout {
            if let Some(sw) = crate::relay::best_midturn_bp_switch(cb, table, weights, side, cs) {
                return sw;
            }
        }
    }
    for &c in cs {
        if (4..7).contains(&c) {
            let slot = (c - 4) as usize;
            let idx = side * 3 + slot;
            let mut score = (cb.mons[idx].hp() as f64) / (table.mons[idx].maxhp.max(1) as f64);
            if passing_alive {
                let base_h = cb.best_hit_fraction(table, side, slot, def_side, def_slot);
                let mut tmp = *cb;
                tmp.mons[idx].boosts = cb.mons[att_idx].boosts;
                tmp.mons[idx].vol_flags |=
                    cb.mons[att_idx].vol_flags & nc2000_engine::compact::vol_bit::SUBSTITUTE;
                let boosted_h = tmp.best_hit_fraction(table, side, slot, def_side, def_slot);
                score += (boosted_h - base_h).clamp(0.0, 1.0);
                if has_sub_or_spe {
                    score += 0.65 * boosted_h.clamp(0.0, 1.2);
                }
            }
            if best_sw.is_none() || score >= best_score {
                best_score = score;
                best_sw = Some(c);
            }
        }
    }
    if let Some(sw) = best_sw {
        return sw;
    }
    rand_non_suicidal(rng)
}

#[inline]
fn encode_choices_to_compact(
    table: &nc2000_engine::compact::MatchTable,
    cb: &nc2000_engine::compact::CompactBattle,
    side: usize,
    cs: &[SearchChoice],
) -> ([u8; 7], usize) {
    let mut out = [0u8; 7];
    let mut n = 0usize;
    let act_slot = cb.sides[side].active_idx as usize;
    let num_moves = if act_slot < 3 {
        table.mons[side * 3 + act_slot].num_moves as usize
    } else {
        0
    };
    for &c in cs {
        if n >= 7 {
            break;
        }
        match c {
            SearchChoice::Move(id) => {
                if let Some(m_i) = (0..num_moves)
                    .find(|&m_i| table.mons[side * 3 + act_slot].moves[m_i] == id)
                {
                    out[n] = m_i as u8;
                    n += 1;
                }
            }
            SearchChoice::Switch(pos) if (1..=3).contains(&pos) => {
                out[n] = pos + 3;
                n += 1;
            }
            _ => {}
        }
    }
    (out, n)
}

#[inline]
fn decode_compact_choice(
    table: &nc2000_engine::compact::MatchTable,
    cb: &nc2000_engine::compact::CompactBattle,
    side: usize,
    act: u8,
) -> SearchChoice {
    if act < 4 || act >= 16 {
        let act_slot = cb.sides[side].active_idx as usize;
        SearchChoice::Move(table.mons[side * 3 + act_slot].moves[(act & 3) as usize])
    } else if (4..7).contains(&act) {
        SearchChoice::Switch(act - 3)
    } else {
        SearchChoice::Pass
    }
}

pub fn playout_pick(
    sim: &Battle,
    dex: &Dex,
    playout: &Playout,
    side: usize,
    cs: &[SearchChoice],
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> SearchChoice {
    if cs.iter().any(|c| matches!(c, SearchChoice::Team(_) | SearchChoice::Pass)) {
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
        return cs[rng.below(cs.len())];
    }
    let (table, cb) = nc2000_engine::compact::CompactBattle::from_battle(sim, dex);
    let (compact_cs, n) = encode_choices_to_compact(&table, &cb, side, cs);
    if n == 0 {
        return cs[rng.below(cs.len())];
    }
    let act = compact_playout_pick(&cb, &table, playout, side, &compact_cs[..n], rng, rules);
    decode_compact_choice(&table, &cb, side, act)
}

/// The switch a side makes when its active faints to Perish Song at the end
/// of this turn (counter 1) and a switch is legal: the bench mon expected to
/// keep the most HP through the foe active's best hit. `None` otherwise.
pub fn perish_escape(
    sim: &Battle,
    dex: &Dex,
    side: usize,
    cs: &[SearchChoice],
) -> Option<SearchChoice> {
    let (table, cb) = nc2000_engine::compact::CompactBattle::from_battle(sim, dex);
    let (compact_cs, n) = encode_choices_to_compact(&table, &cb, side, cs);
    let act = compact_perish_escape(&cb, &table, side, &compact_cs[..n])?;
    Some(decode_compact_choice(&table, &cb, side, act))
}

/// The Perish trap as a rollout side plays it, and the standard answer to it.
pub fn perish_combo(
    sim: &Battle,
    dex: &Dex,
    side: usize,
    cs: &[SearchChoice],
) -> Option<SearchChoice> {
    let (table, cb) = nc2000_engine::compact::CompactBattle::from_battle(sim, dex);
    let (compact_cs, n) = encode_choices_to_compact(&table, &cb, side, cs);
    let act = compact_perish_combo(&cb, &table, side, &compact_cs[..n])?;
    Some(decode_compact_choice(&table, &cb, side, act))
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
        let raw_cap = root.turn.saturating_add(self.cfg.horizon);
        let turn_cap = root
            .ruleset
            .turn_limit()
            .map_or(raw_cap, |lim| raw_cap.min(lim));

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

#[cfg(test)]
mod perish_rollout_tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;

    fn set(species: &str, moves: &[&str]) -> serde_json::Value {
        serde_json::json!({"name":species,"species":species,"item":"","ability":"No Ability",
            "moves":moves,"nature":"Serious","gender":"M","level":50,
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}})
    }

    fn teams() -> [Vec<PokemonSet>; 2] {
        [
            serde_json::from_value(serde_json::json!([
                set("Misdreavus", &["Mean Look", "Perish Song", "Protect", "Thunderbolt"]),
                set("Skarmory", &["Drill Peck", "Whirlwind", "Rest", "Toxic"]),
                set("Blissey", &["Seismic Toss", "Soft-Boiled", "Toxic", "Heal Bell"]),
            ]))
            .unwrap(),
            serde_json::from_value(serde_json::json!([
                set("Snorlax", &["Body Slam", "Earthquake", "Curse", "Rest"]),
                set("Zapdos", &["Whirlwind", "Thunderbolt", "Rest", "Thunder Wave"]),
                set("Cloyster", &["Surf", "Ice Beam", "Spikes", "Toxic"]),
            ]))
            .unwrap(),
        ]
    }

    /// Both sides past team preview, side 1 leading with its `lead`-th mon.
    fn start(lead: usize) -> (Dex, Battle) {
        let dex = conformance::load_dex();
        let [t0, t1] = teams();
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t0, &t1).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        let order = if lead == 1 { "team 1, 2, 3" } else { "team 2, 1, 3" };
        b.choose(&dex, 1, order).unwrap();
        (dex, b)
    }

    fn mv(dex: &Dex, key: &str) -> SearchChoice {
        SearchChoice::Move(dex.moves.id(key).unwrap())
    }

    fn turn(dex: &Dex, b: &mut Battle, moves: [&str; 2]) {
        b.apply_choices(dex, [Some(mv(dex, moves[0])), Some(mv(dex, moves[1]))]).unwrap();
    }

    fn combo(dex: &Dex, b: &mut Battle, side: usize) -> Option<SearchChoice> {
        let cs = b.legal_choices(dex, side);
        perish_combo(b, dex, side, &cs)
    }

    fn escape(dex: &Dex, b: &mut Battle, side: usize) -> Option<SearchChoice> {
        let cs = b.legal_choices(dex, side);
        perish_escape(b, dex, side, &cs)
    }

    #[test]
    fn rollout_trapper_traps_sings_and_escapes() {
        let (dex, mut b) = start(1);
        assert_eq!(combo(&dex, &mut b, 0), Some(mv(&dex, "meanlook")));
        assert_eq!(combo(&dex, &mut b, 1), None, "Snorlax has no part in the combo");
        turn(&dex, &mut b, ["meanlook", "curse"]);
        assert!(b.poke(b.active_id(1).unwrap()).trapped);
        assert_eq!(combo(&dex, &mut b, 0), Some(mv(&dex, "perishsong")));
        turn(&dex, &mut b, ["perishsong", "curse"]);
        assert_eq!(combo(&dex, &mut b, 0), None, "the foe is already counting");
        for _ in 0..2 {
            assert_eq!(escape(&dex, &mut b, 0), None);
            turn(&dex, &mut b, ["thunderbolt", "curse"]);
        }
        assert!(matches!(escape(&dex, &mut b, 0), Some(SearchChoice::Switch(_))));
        assert_eq!(escape(&dex, &mut b, 1), None, "the trapped foe has no switch");
    }

    #[test]
    fn trapped_victim_phazes_while_it_still_has_time() {
        let (dex, mut b) = start(2);
        turn(&dex, &mut b, ["meanlook", "thunderwave"]);
        turn(&dex, &mut b, ["perishsong", "thunderwave"]);
        assert_eq!(combo(&dex, &mut b, 1), Some(mv(&dex, "whirlwind")));
    }

    #[test]
    fn no_combo_from_the_last_mon_or_with_a_kill_on_the_board() {
        let (dex, mut b) = start(1);
        let foe = b.active_id(1).unwrap();
        b.poke_mut(foe).hp = 1;
        assert_eq!(combo(&dex, &mut b, 0), None, "an expected KO is taken instead");
        let (dex, mut b) = start(1);
        b.sides[0].pokemon_left = 1;
        assert_eq!(combo(&dex, &mut b, 0), None, "Perish Song fails from the last mon");
    }

    #[test]
    fn perish_rules_draw_the_same_rng_as_the_ordinary_pick() {
        let (dex, mut b) = start(1);
        let cs = b.legal_choices(&dex, 0);
        let playout = Playout::heavy();
        for seed in 0..64 {
            let (mut r0, mut r1) = (SplitMix64::new(seed), SplitMix64::new(seed));
            playout_pick(&b, &dex, &playout, 0, &cs, &mut r0, RolloutRules::default());
            let rules = RolloutRules { perish_combo: true, ..Default::default() };
            let pick = playout_pick(&b, &dex, &playout, 0, &cs, &mut r1, rules);
            assert_eq!(pick, mv(&dex, "meanlook"));
            assert_eq!(r0.next(), r1.next());
        }
    }

    #[test]
    fn playout_value_executes_exact_requested_rollout_turns() {
        let (dex, b) = start(1);
        assert_eq!(b.turn, 1);
        for req_turns in [0u16, 1, 2, 8] {
            let mut sim = b.clone();
            let playout = Playout::Heavy {
                eps: 0.0,
                turns: req_turns,
                weights: eval::EvalWeights::default(),
            };
            let mut rng = SplitMix64::new(42);
            let _ = playout_value(
                &mut sim,
                &dex,
                &playout,
                100,
                &mut rng,
                RolloutRules::default(),
            );
            let end = last_sim_end();
            if end.outcome == 0 {
                assert_eq!(
                    end.turn,
                    1 + req_turns,
                    "turns={req_turns} started at turn 1 must stop at turn {}, got {}",
                    1 + req_turns,
                    end.turn
                );
            }
        }
    }

    #[test]
    fn compact_p2_baton_pass_respects_sides0_disable_bit() {
        let dex = conformance::load_dex();
        let t0: Vec<PokemonSet> = serde_json::from_value(serde_json::json!([
            set("Snorlax", &["Splash"]),
            set("Zapdos", &["Thunderbolt"]),
        ]))
        .unwrap();
        let t1: Vec<PokemonSet> = serde_json::from_value(serde_json::json!([
            set("Jolteon", &["Quick Attack", "Baton Pass"]),
            set("Snorlax", &["Return", "Earthquake"]),
        ]))
        .unwrap();
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t0, &t1).unwrap();
        b.choose(&dex, 0, "team 1, 2").unwrap();
        b.choose(&dex, 1, "team 1, 2").unwrap();
        let (table, mut cb) = nc2000_engine::compact::CompactBattle::from_battle(&b, &dex);
        cb.mons[3].boosts[0] = 4; // Side 1 (P2) Jolteon has +4 Atk boost to Baton Pass to Snorlax.
        let mut legal_buf = [0u8; 7];
        let n = cb.legal_actions(&table, 1, &mut legal_buf);
        let cs = &legal_buf[..n];
        let rules_on = RolloutRules {
            baton_pass: true,
            ..RolloutRules::default()
        };
        let pick_enabled = compact_playout_pick(
            &cb,
            &table,
            &Playout::heavy(),
            1,
            cs,
            &mut SplitMix64::new(7),
            rules_on,
        );
        assert_eq!(pick_enabled, 1, "P2 must select Baton Pass (slot 1) when baton_pass fix enabled");

        // Disabling via sides[0].side_flags & SIDE0_DISABLE_BP_FIX must also disable P2 Baton Pass heuristic.
        cb.sides[0].side_flags |= nc2000_engine::compact::SIDE0_DISABLE_BP_FIX;
        let pick_disabled = compact_playout_pick(
            &cb,
            &table,
            &Playout::heavy(),
            1,
            cs,
            &mut SplitMix64::new(7),
            rules_on,
        );
        assert_eq!(pick_disabled, 0, "Disabling via sides[0] bit 0x80 must fall back to Quick Attack (slot 0)");
    }
}
