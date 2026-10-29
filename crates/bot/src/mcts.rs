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

use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
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

const ROLLOUT_SWITCH_TRIGGER: f64 = 0.15;
const ROLLOUT_SWITCH_MARGIN: f64 = 0.10;

const PSEUDO_SLEEP: f64 = 0.30;
const PSEUDO_PAR: f64 = 0.10;
const PSEUDO_SPIKES: f64 = 0.15;
const PSEUDO_BOOST: f64 = 0.15;
const PSEUDO_HEAL: f64 = 0.25;

/// Optional rules layered over the heavy rollout's ε-greedy pick. The
/// default is none of them; the shipped set is
/// `RmConfig::default().rollout_rules()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RolloutRules {
    /// Parked M16c upgrade: status-move pseudo-values
    /// (`RmConfig::rollout_status`).
    pub status: bool,
    /// Parked M16c upgrade: bad-matchup voluntary switching
    /// (`RmConfig::rollout_switch`).
    pub switch: bool,
    /// A side whose active faints to Perish Song at the end of this turn
    /// switches out whenever a switch is legal ([`perish_escape`]).
    pub perish_escape: bool,
    /// Rollout sides play the Perish trap and its counter ([`perish_combo`]).
    /// Implies `perish_escape`, without which the combo kills its own singer.
    pub perish_combo: bool,
    pub capped_damage: bool,
    pub rollout_residual: bool,
    pub baton_pass: bool,
}

impl Default for RolloutRules {
    fn default() -> Self {
        RolloutRules {
            status: false,
            switch: false,
            perish_escape: false,
            perish_combo: false,
            capped_damage: false,
            rollout_residual: false,
            baton_pass: true,
        }
    }
}

/// Unfinished rollouts return a leaf evaluation rather than a terminal outcome.
pub fn playout_value(
    sim: &mut Battle,
    dex: &Dex,
    playout: &Playout,
    turn_cap: u16,
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> f64 {
    let cutoff = match playout {
        Playout::Uniform => turn_cap,
        Playout::Heavy { turns, .. } => turn_cap.min(sim.turn.saturating_add(*turns)),
    };
    if let Some(o) = sim.outcome() {
        return outcome_reward(o);
    }
    // If either side is still in TeamPreview, advance that single step on `Battle`
    // before converting the 3v3 party state into the 256-byte `CompactBattle` kernel.
    while sim.sides[0].request_state() == Some(nc2000_engine::state::RequestKind::TeamPreview)
        || sim.sides[1].request_state() == Some(nc2000_engine::state::RequestKind::TeamPreview)
    {
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
                picks[s] = Some(playout_pick(sim, dex, playout, s, &cs, rng, rules));
            }
        }
        sim.apply_choices(dex, picks)
            .expect("legal_choices produced an illegal choice");
    }

    let (table, mut cb) = nc2000_engine::compact::CompactBattle::from_battle(sim, dex);
    if !rules.baton_pass {
        cb.sides[0].side_flags |= 0x80;
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
    loop {
        match cb.outcome {
            1 => return 1.0,
            2 => return 0.0,
            3 => return 0.5,
            _ => {}
        }
        if cb.turn > cutoff {
            return match playout {
                Playout::Uniform => compact_hp_eval(cb, table),
                Playout::Heavy { weights, .. } => eval::compact_eval_leaf(cb, table, weights),
            };
        }
        let mut picks = [7u8; 2];
        for s in 0..2usize {
            if cb.sides[s].request == 0 {
                continue;
            }
            let mut cs = [0u8; 7];
            let n = cb.legal_actions(table, s, &mut cs);
            if n > 0 {
                picks[s] = compact_playout_pick(cb, table, playout, s, &cs[..n], rng, rules);
            }
        }
        cb.apply_actions(table, picks);
    }
}

#[inline]
fn compact_hp_eval(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
) -> f64 {
    let f = |s: usize| {
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for slot in 0..3usize {
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
    let pick = compact_ordinary_pick(cb, table, playout, side, cs, rng, rules);
    if rules.perish_escape || rules.perish_combo {
        if let Some(escape) = compact_perish_escape(cb, table, side, cs) {
            return escape;
        }
    }
    if rules.perish_combo {
        if let Some(combo) = compact_perish_combo(cb, table, side, cs) {
            return combo;
        }
    }
    pick
}

#[inline]
fn compact_ordinary_pick(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    playout: &Playout,
    side: usize,
    cs: &[u8],
    rng: &mut SplitMix64,
    rules: RolloutRules,
) -> u8 {
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
    compact_greedy_pick(cb, table, side, cs, rng, rules)
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
    if (cb.mons[att_idx].vol_flags & vol_bit::PERISH_SONG) == 0 || cb.mons[att_idx].timers[3] != 1 {
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
                cb.best_hit_fraction(table, foe_side, foe_slot, side, slot, true)
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
    use nc2000_engine::compact::vol_bit;
    let me_slot = cb.sides[side].active_idx as usize;
    let me_idx = side * 3 + me_slot;
    let mut song = None;
    let mut phaze = None;
    let mut trap = None;
    for &c in cs {
        if c < 4 {
            let sk = table.move_specs[me_idx][c as usize].special_kind;
            if sk == 19 && song.is_none() {
                song = Some(c);
            } else if sk == 7 && phaze.is_none() {
                phaze = Some(c);
            } else if sk == 26 && trap.is_none() {
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
    let me_trapped = (me.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0;
    if let Some(phaze_act) = phaze {
        let counting = (me.vol_flags & vol_bit::PERISH_SONG) != 0 && me.timers[3] >= 2;
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
    if cb.best_hit_fraction(table, side, me_slot, foe_side, foe_slot, true) >= 1.0 {
        return None;
    }
    if (foe.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0 || !foe_bench {
        return Some(song_act);
    }
    trap
}

#[inline]
fn compact_status_pseudo_score(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
    side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    m_slot: usize,
) -> f64 {
    let att_idx = side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let mspec = &table.move_specs[att_idx][m_slot];
    let acc = if mspec.always_hit {
        1.0
    } else {
        mspec.base_acc_pct as f64 / 100.0
    };
    match mspec.pseudo_kind {
        1 => {
            if cb.mons[def_idx].status_raw() != 0 {
                return 0.0;
            }
            for p_i in 0..3usize {
                if (cb.sides[def_side].alive_mask & (1 << p_i)) != 0 {
                    let m = &cb.mons[def_side * 3 + p_i];
                    if m.hp() > 0 && m.status_raw() == nc2000_engine::state::Status::Slp as u8 {
                        return 0.0;
                    }
                }
            }
            PSEUDO_SLEEP * acc
        }
        2 => {
            if cb.mons[def_idx].status_raw() == 0 {
                PSEUDO_PAR * acc
            } else {
                0.0
            }
        }
        3 => {
            let foe = &cb.sides[def_side];
            if foe.alive_mask.count_ones() > 1 && (foe.side_flags & 0x03) == 0 {
                PSEUDO_SPIKES
            } else {
                0.0
            }
        }
        4 => {
            let me = &cb.mons[att_idx];
            let maxhp = table.mons[att_idx].maxhp;
            if (me.hp() as f64) < 0.5 * (maxhp as f64) {
                PSEUDO_HEAL
            } else {
                0.0
            }
        }
        5 => {
            let me = &cb.mons[att_idx];
            let is_ghost =
                (table.mons[att_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
            if !is_ghost && me.boosts[0] < 2 {
                PSEUDO_BOOST
            } else {
                0.0
            }
        }
        6 => {
            if cb.mons[att_idx].boosts[0] < 2 {
                PSEUDO_BOOST
            } else {
                0.0
            }
        }
        7 => {
            if cb.mons[att_idx].boosts[2] < 2 {
                PSEUDO_BOOST
            } else {
                0.0
            }
        }
        8 => {
            let me = &cb.mons[att_idx];
            let maxhp = table.mons[att_idx].maxhp;
            if me.boosts[0] < 6 && (me.hp() as f64) > 0.6 * (maxhp as f64) {
                PSEUDO_BOOST + 0.10
            } else {
                0.0
            }
        }
        _ => 0.0,
    }
}

#[inline]
fn compact_greedy_pick(
    cb: &nc2000_engine::compact::CompactBattle,
    table: &nc2000_engine::compact::MatchTable,
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
                let raw_score = cb.expected_hit_fraction(
                    table, side, att_slot, def_side, def_slot, m_slot, true,
                );
                let mut score = if rules.capped_damage {
                    let accuracy =
                        cb.hit_probability(table, side, att_slot, def_side, def_slot, m_slot);
                    if accuracy > 0.0 {
                        (raw_score / accuracy).min(1.0) * accuracy
                    } else {
                        0.0
                    }
                } else {
                    raw_score
                };
                if score <= 0.0 && rules.status {
                    score = compact_status_pseudo_score(
                        cb, table, side, att_slot, def_side, def_slot, m_slot,
                    );
                }
                if best.map_or(true, |(_, old_score, old_raw)| {
                    score > old_score
                        || (rules.capped_damage && score == old_score && raw_score > old_raw)
                }) {
                    best = Some((c, score, raw_score));
                }
            }
        }
        if rules.switch {
            if let Some((_, stay, _)) = best {
                if stay < ROLLOUT_SWITCH_TRIGGER {
                    let mut sw: Option<(u8, f64)> = None;
                    for &c in cs {
                        if (4..7).contains(&c) {
                            let cand_slot = (c - 4) as usize;
                            let off = cb.best_hit_fraction(
                                table, side, cand_slot, def_side, def_slot, true,
                            );
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
        if rules.rollout_residual {
            let stay_dmg = best.map_or(0.0, |(_, s, _)| s) as f32;
            if let Some(deltas) =
                crate::eval::compute_rollout_action_residuals(cb, table, side, cs, stay_dmg)
            {
                for (i, &c) in cs.iter().take(7).enumerate() {
                    if suicidal(c) {
                        continue;
                    }
                    let dq = deltas[i] as f64;
                    if dq <= 0.0 {
                        continue;
                    }
                    let base = if c < 4 {
                        cb.expected_hit_fraction(
                            table,
                            side,
                            att_slot,
                            def_side,
                            def_slot,
                            c as usize,
                            true,
                        )
                    } else {
                        0.0
                    };
                    let total = base + dq;
                    if best.map_or(true, |(_, old_score, _)| total > old_score) {
                        best = Some((c, total, base));
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
    // score receivers by HP fraction + offensive effectiveness with passed boosts;
    // when fainted (`hp <= 0`), keep exact healthiest-bench selection.
    let mut best_sw = None;
    let mut best_score = -1.0f64;
    let passing_alive = rules.baton_pass
        && cb.sides[side].request == 2
        && att_slot < 3
        && cb.mons[att_idx].hp() > 0
        && (cb.mons[att_idx].boosts[0] > 0 || cb.mons[att_idx].boosts[2] > 0)
        && (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
        && cb.mons[def_side * 3 + def_slot].hp() > 0;
    for &c in cs {
        if (4..7).contains(&c) {
            let slot = (c - 4) as usize;
            let idx = side * 3 + slot;
            let mut score = (cb.mons[idx].hp() as f64) / (table.mons[idx].maxhp.max(1) as f64);
            if passing_alive {
                let base_h = cb.best_hit_fraction(table, side, slot, def_side, def_slot, true);
                let mut tmp = *cb;
                tmp.mons[idx].boosts = cb.mons[att_idx].boosts;
                let boosted_h = tmp.best_hit_fraction(table, side, slot, def_side, def_slot, true);
                score += (boosted_h - base_h).clamp(0.0, 1.0);
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
    if act < 4 {
        let act_slot = cb.sides[side].active_idx as usize;
        SearchChoice::Move(table.mons[side * 3 + act_slot].moves[act as usize])
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
}
