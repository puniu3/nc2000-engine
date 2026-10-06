//! Counterfactual evaluation under immutable policies extracted from a search tree.
//! Unknown continuations use the configured rollout; returns are not optimal-play bounds.

use nc2000_engine::{battle::SearchChoice, dex::Dex, fxhash::FxHashMap, state::Battle};

use crate::{
    mcts::{outcome_reward, playout_value, Playout},
    rng::SplitMix64,
    smmcts::{key_of, Node, RmConfig},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrozenPolicy {
    MostVisited,
    SampleVisits,
}

impl FrozenPolicy {
    fn select(self, counts: &[u32], rng: &mut SplitMix64) -> Option<usize> {
        let total: u64 = counts.iter().map(|&n| n as u64).sum();
        if total == 0 {
            return None;
        }
        Some(match self {
            Self::MostVisited => counts.iter().enumerate().max_by_key(|&(_, n)| n).unwrap().0,
            Self::SampleVisits => {
                let mut draw = rng.next_f64() * total as f64;
                counts
                    .iter()
                    .position(|&n| {
                        draw -= n as f64;
                        draw < 0.0
                    })
                    .unwrap()
            }
        })
    }
}

pub struct FrozenChoice<'a> {
    pub battle: &'a Battle,
    pub node: Option<usize>,
    pub chosen: [Option<SearchChoice>; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct FrozenResult {
    pub reward0: f64,
    pub terminal: bool,
    pub prefix_choices: u32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn evaluate_sim(
    dex: &Dex,
    cfg: &RmConfig,
    sim: &mut Battle,
    nodes: &[Node],
    table: &FxHashMap<u64, usize>,
    forced: [Option<SearchChoice>; 2],
    policy: [FrozenPolicy; 2],
    rng: &mut SplitMix64,
    trace: &mut impl FnMut(FrozenChoice<'_>),
) -> FrozenResult {
    let turn_cap = sim.turn.saturating_add(cfg.horizon);
    let mut tree_choices = 0;
    let reward0 = loop {
        if let Some(outcome) = sim.outcome() {
            break outcome_reward(outcome);
        }
        if sim.turn > turn_cap {
            break match &cfg.playout {
                Playout::Uniform => crate::mcts::hp_eval(sim),
                Playout::Heavy { weights, .. } => crate::eval::eval_leaf(sim, dex, weights),
            };
        }
        let key = if tree_choices == 0 {
            crate::blind::root_search_key(cfg, dex, sim)
        } else { key_of(cfg, dex, sim) };
        let node_id = table.get(&key).copied();
        let node = node_id.map(|i| &nodes[i]);
        let acts = [sim.legal_choices(dex, 0), sim.legal_choices(dex, 1)];
        if tree_choices == 0 {
            for side in 0..2 {
                assert!(forced[side].is_none_or(|action| acts[side].contains(&action)));
            }
        }
        let mut chosen = [None, None];
        let mut missing = false;
        for side in 0..2 {
            if acts[side].is_empty() {
                continue;
            }
            if tree_choices == 0 && forced[side].is_some() {
                assert!(acts[side].contains(&forced[side].unwrap()));
                chosen[side] = forced[side];
            } else if acts[side].len() == 1 {
                chosen[side] = Some(acts[side][0]);
            } else if let Some(node) = node.filter(|n| n.acts[side] == acts[side]) {
                let extraction = if tree_choices == 0 {
                    FrozenPolicy::MostVisited
                } else {
                    policy[side]
                };
                chosen[side] = extraction.select(&node.n[side], rng).map(|a| acts[side][a]);
                missing |= chosen[side].is_none();
            } else {
                missing = true;
            }
        }
        if missing {
            if tree_choices > 0 {
                break playout_value(sim, dex, &cfg.playout, turn_cap, rng, cfg.rollout_rules());
            }
            for side in 0..2 {
                if chosen[side].is_none() && !acts[side].is_empty() {
                    chosen[side] = Some(crate::mcts::playout_pick(
                        sim,
                        dex,
                        &cfg.playout,
                        side,
                        &acts[side],
                        rng,
                        cfg.rollout_rules(),
                    ));
                }
            }
        }
        assert_ne!(chosen, [None, None]);
        trace(FrozenChoice {
            battle: sim,
            node: node_id,
            chosen,
        });
        sim.apply_choices(dex, chosen)
            .expect("frozen policy chose an illegal action");
        tree_choices += 1;
        if missing {
            break playout_value(sim, dex, &cfg.playout, turn_cap, rng, cfg.rollout_rules());
        }
    };
    FrozenResult {
        reward0,
        terminal: sim.outcome().is_some(),
        prefix_choices: tree_choices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_does_not_credit_exploratory_opponent_losses() {
        let mut rng = SplitMix64::new(71);
        let mut mean = [0.0_f64; 2];
        for (i, policy) in [FrozenPolicy::MostVisited, FrozenPolicy::SampleVisits]
            .into_iter()
            .enumerate()
        {
            for _ in 0..10_000 {
                let reply = policy.select(&[17, 14, 14, 7, 14], &mut rng).unwrap();
                mean[i] += if reply == 0 { 0.25 } else { 1.0 };
            }
            mean[i] /= 10_000.0;
        }
        assert_eq!(mean[0], 0.25);
        assert!((mean[1] - (0.25 * 17.0 + 49.0) / 66.0).abs() < 0.02);
        assert!(mean[0] < 0.5 && mean[1] > 0.5);
    }

    #[test]
    fn extraction_is_invariant_to_visit_scale_and_has_no_unvisited_value() {
        let mut a = SplitMix64::new(5);
        let mut b = a.clone();
        for policy in [FrozenPolicy::MostVisited, FrozenPolicy::SampleVisits] {
            assert_eq!(policy.select(&[0, 0], &mut a), None);
            for _ in 0..100 {
                assert_eq!(
                    policy.select(&[0, 2, 5], &mut a),
                    policy.select(&[0, 20, 50], &mut b)
                );
            }
        }
    }
}
