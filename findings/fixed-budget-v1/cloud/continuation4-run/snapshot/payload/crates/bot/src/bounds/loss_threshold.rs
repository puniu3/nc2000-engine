use std::time::Instant;

use nc2000_engine::state::Battle;

use super::{normalized_policy, BoundDirection, BoundSolver, Bounds, RootPayoffs};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyLossThresholdStop {
    AboveThreshold,
    AtMostThreshold,
    WorkExhausted,
    TimeBudget,
    NodeBudget,
}

#[derive(Clone, Debug)]
pub struct PolicyLossThresholdReport {
    pub value_p0: Bounds,
    pub policy_loss: Bounds,
    pub stop: PolicyLossThresholdStop,
    pub runs: usize,
}

fn witness_cell(payoffs: &RootPayoffs, side: usize, policy: &[f64]) -> Option<usize> {
    let ceiling = payoffs.policy_value(side, policy).hi;
    let mut best: Option<(f64, f64, f64, usize)> = None;
    for reply in 0..payoffs.actions[1 - side].len() {
        let mut lower = 0.0;
        let mut upper = 0.0;
        let mut widest: Option<(f64, usize)> = None;
        for (action, &mass) in policy.iter().enumerate() {
            let bounds = payoffs.cell(side, action, reply);
            lower += mass * bounds.lo;
            upper += mass * bounds.hi;
            let score = mass * bounds.width();
            let index = if side == 0 {
                action * payoffs.actions[1].len() + reply
            } else {
                reply * payoffs.actions[1].len() + action
            };
            if score > 1e-12 && widest.is_none_or(|(old, _)| score > old) {
                widest = Some((score, index));
            }
        }
        if lower >= ceiling {
            continue;
        }
        let Some((width, index)) = widest else {
            continue;
        };
        let candidate = (upper, lower, -width, index);
        if best.is_none_or(|old| candidate < old) {
            best = Some(candidate);
        }
    }
    best.map(|(_, _, _, index)| index)
}

impl BoundSolver<'_> {
    pub fn solve_policy_loss_threshold(
        &mut self,
        battle: &Battle,
        side: usize,
        policy: &[f64],
        threshold: f64,
    ) -> PolicyLossThresholdReport {
        assert!(side < 2 && threshold.is_finite() && (0.0..1.0).contains(&threshold));
        assert!(battle.outcome().is_none() && !self.closed.contains_key(&self.node_key(battle)));
        let started = Instant::now();
        let budget = self.cfg.work_budget;
        self.cfg.work_budget = 0;
        self.solve(battle, None);
        self.cfg.work_budget = budget;
        let root = self.node_key(battle);
        self.init_cells(root);
        assert_eq!(
            policy.len(),
            self.root_payoffs(battle).unwrap().actions[side].len()
        );
        let policy = normalized_policy(policy).expect("valid target policy");
        let start_runs = self.stats.runs;
        let work_limit = start_runs.saturating_add(budget);
        let mut cursor = self.stats.trials;
        let mut idle = 0usize;
        loop {
            let payoffs = self.root_payoffs(battle).unwrap();
            let loss = payoffs.policy_loss(side, &policy);
            let stop = if loss.lo > threshold {
                Some(PolicyLossThresholdStop::AboveThreshold)
            } else if loss.hi <= threshold {
                Some(PolicyLossThresholdStop::AtMostThreshold)
            } else if self
                .cfg
                .time_budget
                .is_some_and(|limit| started.elapsed() >= limit)
            {
                Some(PolicyLossThresholdStop::TimeBudget)
            } else if self.stats.runs >= work_limit || idle > 256 {
                Some(PolicyLossThresholdStop::WorkExhausted)
            } else if self.nodes.len() + self.closed.len() >= self.cfg.node_budget {
                Some(PolicyLossThresholdStop::NodeBudget)
            } else {
                None
            };
            if let Some(stop) = stop {
                return PolicyLossThresholdReport {
                    value_p0: payoffs.value,
                    policy_loss: loss,
                    stop,
                    runs: self.stats.runs - start_runs,
                };
            }
            let witness = witness_cell(&payoffs, side, &policy);
            let refine_value =
                witness.is_none() || (payoffs.value.width() > 1e-12 && cursor % 2 == 0);
            cursor += 1;
            let direction = if (side == 0) == refine_value {
                BoundDirection::Lower
            } else {
                BoundDirection::Upper
            };
            let cell = if refine_value {
                self.select_cell_for_bound(root, Some(direction))
            } else {
                witness.unwrap()
            };
            if self.nodes[&root].battle.is_none() {
                self.nodes.get_mut(&root).unwrap().battle = Some(battle.clone());
            }
            if self.trial_from_cell(root, work_limit, Some(cell), Some(direction)) {
                idle = 0;
            } else {
                idle += 1;
            }
            if self.cfg.fold_closed_nodes && self.stats.trials % 4096 == 0 {
                self.fold_closed_graph();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bounds::BoundConfig;
    use crate::exact::solve_matrix;
    use nc2000_engine::battle::{enumerate::enumerate_step, PokemonSet};
    use serde_json::json;

    #[test]
    fn witness_targets_a_low_upper_reply_in_either_orientation() {
        let payoffs = RootPayoffs {
            actions: [vec![None, None], vec![None, None]],
            cells: vec![
                Bounds { lo: 0.0, hi: 1.0 },
                Bounds { lo: 0.0, hi: 0.3 },
                Bounds { lo: 0.8, hi: 1.0 },
                Bounds { lo: 0.1, hi: 0.9 },
            ],
            value: Bounds { lo: 0.1, hi: 1.0 },
        };
        assert_eq!(witness_cell(&payoffs, 0, &[1.0, 0.0]), Some(1));
        assert_eq!(witness_cell(&payoffs, 1, &[1.0, 0.0]), Some(2));
    }

    #[test]
    fn threshold_search_encloses_exhaustive_losses_and_resumes() {
        let dex = conformance::load_dex();
        for moves in [
            [
                vec!["Seismic Toss", "Splash"],
                vec!["Seismic Toss", "Splash"],
            ],
            [vec!["Thunder", "Splash"], vec!["Recover", "Seismic Toss"]],
        ] {
            let teams: [Vec<PokemonSet>; 2] = moves.map(|moves| {
                vec![serde_json::from_value(json!({
                    "name":"Alakazam", "species":"Alakazam", "level":50, "moves":moves,
                }))
                .unwrap()]
            });
            let mut root = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
            let preview = [0, 1].map(|s| root.legal_choices(&dex, s).first().copied());
            root.apply_choices(&dex, preview).unwrap();
            root.turn = 1000;
            for side in 0..2 {
                let id = root.active_id(side).unwrap();
                root.poke_mut(id).hp = 40;
            }
            let actions = [0, 1].map(|s| root.legal_choices(&dex, s));
            let mut cells = Vec::new();
            for &a in &actions[0] {
                for &b in &actions[1] {
                    let step = enumerate_step(&dex, &root, [Some(a), Some(b)], 100_000).unwrap();
                    let value: f64 = step
                        .leaves
                        .iter()
                        .map(|leaf| {
                            leaf.prob
                                * super::super::outcome_value(&leaf.battle)
                                    .expect("one turn ends fixture")
                        })
                        .sum();
                    cells.push(Bounds {
                        lo: value,
                        hi: value,
                    });
                }
            }
            let value = solve_matrix(&cells.iter().map(|b| b.lo).collect::<Vec<_>>(), 2, 2).0;
            let exact = RootPayoffs {
                actions: actions.map(|a| a.into_iter().map(Some).collect()),
                cells,
                value: Bounds {
                    lo: value,
                    hi: value,
                },
            };
            for side in 0..2 {
                for policy in [vec![1.0, 0.0], vec![0.0, 1.0], vec![0.3, 0.7]] {
                    let expected = exact.policy_loss(side, &policy);
                    let threshold = if expected.lo > 0.1 {
                        expected.lo - 0.05
                    } else {
                        0.1
                    };
                    for work_budget in [1, 17] {
                        let mut solver = BoundSolver::new(
                            &dex,
                            BoundConfig {
                                work_budget,
                                cell_cap: 0,
                                eps: 1e-10,
                                dead_quick_claw_quotient: true,
                                ..Default::default()
                            },
                        );
                        let mut previous = Bounds { lo: 0.0, hi: 1.0 };
                        let mut decided = false;
                        for _ in 0..10000 {
                            let report =
                                solver.solve_policy_loss_threshold(&root, side, &policy, threshold);
                            assert!(report.runs <= work_budget);
                            assert!(report.policy_loss.lo <= expected.lo + 1e-9);
                            assert!(report.policy_loss.hi >= expected.hi - 1e-9);
                            assert!(
                                report.policy_loss.lo + 1e-9 >= previous.lo
                                    && report.policy_loss.hi <= previous.hi + 1e-9
                            );
                            previous = report.policy_loss;
                            match report.stop {
                                PolicyLossThresholdStop::AboveThreshold => {
                                    assert!(expected.lo > threshold);
                                    decided = true;
                                }
                                PolicyLossThresholdStop::AtMostThreshold => {
                                    assert!(expected.hi <= threshold);
                                    decided = true;
                                }
                                PolicyLossThresholdStop::WorkExhausted => assert!(report.runs > 0),
                                _ => panic!("unexpected stop: {:?}", report.stop),
                            }
                            if decided {
                                break;
                            }
                        }
                        assert!(decided, "side={side}, policy={policy:?}, expected={expected:?}, budget={work_budget}");
                    }
                }
            }
        }
    }
}
