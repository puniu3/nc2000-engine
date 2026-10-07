#![allow(dead_code)]

include!("policy_witness.rs");

struct Numeric<'a> {
    rules: Witness<'a>,
    bounds: HashMap<CacheKey, [f64; 2]>,
    root_branches: Vec<Value>,
    alias_count: usize,
    threshold: f64,
    pruned_leaves: usize,
}

impl Numeric<'_> {
    fn visit(&mut self, mut battle: Battle, depth: usize) -> [f64; 2] {
        let terminal = battle.outcome().map(|o| match o {
            Outcome::Tie => 0.5,
            Outcome::P1Win => f64::from(self.rules.cfg.target_side == 0),
            Outcome::P2Win => f64::from(self.rules.cfg.target_side == 1),
        });
        if let Some(value) = terminal {
            return [value, value];
        }
        if depth >= self.rules.cfg.horizon {
            return [0., 1.];
        }
        let key = self.rules.key(&battle, depth);
        if let Some(&bounds) = self.bounds.get(&key) {
            return bounds;
        }
        let result = if self.rules.runs >= self.rules.cfg.work_budget {
            [0., 1.]
        } else {
            let dex = self.rules.dex;
            let legal = [0, 1].map(|s| {
                if battle.needs_choice()[s] {
                    battle
                        .legal_choices(dex, s)
                        .into_iter()
                        .map(Some)
                        .collect::<Vec<_>>()
                } else {
                    vec![None]
                }
            });
            let selected = [0, 1].map(|s| self.rules.selected(&battle, s, &legal[s], depth));
            let mut result = [1f64, 1f64];
            let mut representatives: Vec<([Option<SearchChoice>; 2], [f64; 2])> = Vec::new();
            for &a in &selected[0] {
                for &b in &selected[1] {
                    let choices = [a, b];
                    let side = self.rules.cfg.target_side;
                    let alias = representatives.iter().position(|(prior, _)| {
                        if prior[side] != choices[side] {
                            return false;
                        }
                        let (Some(a), Some(b)) = (prior[1 - side], choices[1 - side]) else {
                            return false;
                        };
                        let mut pair = vec![a, b];
                        nc2000_bot::smmcts::collapse_sleep_choices(
                            &battle,
                            dex,
                            1 - side,
                            &mut pair,
                        );
                        pair == [a]
                    });
                    let branch = if let Some(index) = alias {
                        self.alias_count += 1;
                        representatives[index].1
                    } else {
                        let remaining = self.rules.cfg.work_budget.saturating_sub(self.rules.runs);
                        let mut branch = [0., 1.];
                        if remaining > 0 {
                            let mut step = enumerate_step_partial(
                                dex,
                                &battle,
                                choices,
                                self.rules.cfg.cell_cap.min(remaining),
                            );
                            self.rules.runs += step.runs;
                            let pending: f64 = step.pending.iter().map(|p| p.mass).sum();
                            branch = [0., pending];
                            let pad = 1e-12 * (step.leaves.len() + 2) as f64;
                            if depth == 0 && self.threshold < 1. {
                                let weights = nc2000_bot::eval::EvalWeights::default();
                                step.leaves.sort_by_cached_key(|leaf| {
                                    let value =
                                        nc2000_bot::eval::eval01(&leaf.battle, dex, &weights);
                                    let own = if self.rules.cfg.target_side == 0 {
                                        value
                                    } else {
                                        1. - value
                                    };
                                    std::cmp::Reverse((leaf.prob * own).to_bits())
                                });
                            }
                            for leaf in step.leaves {
                                if depth == 0 && branch[0] - pad > self.threshold {
                                    self.pruned_leaves += 1;
                                    branch[1] += leaf.prob;
                                } else {
                                    let bounds = self.visit(leaf.battle, depth + 1);
                                    for k in 0..2 {
                                        branch[k] += leaf.prob * bounds[k];
                                    }
                                }
                            }
                            branch = [(branch[0] - pad).max(0.), (branch[1] + pad).min(1.)];
                        }
                        representatives.push((choices, branch));
                        branch
                    };
                    for k in 0..2 {
                        result[k] = result[k].min(branch[k]);
                    }
                    if depth == 0 {
                        self.root_branches.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(dex))),"bounds":branch}));
                    }
                }
            }
            result
        };
        self.bounds.insert(key, result);
        result
    }
}

pub fn solve(dex: &Dex, battle: Battle, config: Value, threshold: f64) -> Value {
    assert!(threshold.is_finite() && (0. ..=1.).contains(&threshold));
    let cfg: Config = serde_json::from_value(config).unwrap();
    assert!(cfg.target_side < 2 && cfg.policy_side == cfg.target_side && cfg.guarded_keys);
    assert!(cfg.horizon > 0 && cfg.cell_cap > 0 && cfg.work_budget > 0);
    assert!(
        !cfg.material_order
            && !cfg.share_root_budget
            && !cfg.stop_if_asleep
            && !cfg.reverse_replies
    );
    assert!(
        cfg.minimum_target_living.is_none()
            && cfg.share_budget_depth.is_none()
            && !cfg.share_unique_children
    );
    assert!(
        cfg.finishing_hp.is_none()
            && cfg.finishing_priority.is_empty()
            && cfg.opponent_priority.is_empty()
    );
    let key = battle.state_key128().to_string();
    let mut solver = Numeric {
        rules: Witness {
            dex,
            cfg,
            runs: 0,
            memo: HashMap::new(),
            nodes: Vec::new(),
            prior: HashMap::new(),
            prioritize: false,
        },
        bounds: HashMap::new(),
        root_branches: Vec::new(),
        alias_count: 0,
        threshold,
        pruned_leaves: 0,
    };
    let bounds = solver.visit(battle, 0);
    json!({"config":solver.rules.cfg,"runs":solver.rules.runs,"bounds":bounds,"root":0,
        "numeric_only":true,"threshold":threshold,"pruned_leaves":solver.pruned_leaves,"alias_count":solver.alias_count,"nodes":[{"state_key":key,"bounds":bounds,"branches":solver.root_branches}]})
}
