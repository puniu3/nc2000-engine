#![allow(dead_code)]

include!("policy_witness.rs");

#[path = "stream_step/mod.rs"]
mod stream_step;

struct Streaming<'a> {
    rules: Witness<'a>,
    memo: HashMap<CacheKey, usize>,
    values: Vec<[f64; 2]>,
    records: Vec<Value>,
    proof: bool,
    audit: bool,
    ordered: bool,
    audit_runs: usize,
    audit_cells: usize,
    root_branches: Vec<Value>,
}

fn projected_key(b: &Battle) -> u128 {
    match b.outcome() {
        None => 0,
        Some(Outcome::P1Win) => 1,
        Some(Outcome::P2Win) => 2,
        Some(Outcome::Tie) => 3,
    }
}

impl Streaming<'_> {
    fn visit(&mut self, mut battle: Battle, depth: usize, target: f64) -> usize {
        let terminal = battle.outcome().map(|o| match o {
            Outcome::Tie => 0.5,
            Outcome::P1Win => f64::from(self.rules.cfg.target_side == 0),
            Outcome::P2Win => f64::from(self.rules.cfg.target_side == 1),
        });
        let cache = if self.proof || terminal.is_none() && depth < self.rules.cfg.horizon {
            Some(self.rules.key(&battle, depth))
        } else {
            None
        };
        if let Some(id) = cache.and_then(|k| self.memo.get(&k).copied()) {
            return id;
        }
        let mut record = if self.proof {
            let (damage, claw, key, _) = cache.unwrap();
            json!({"state_key":battle.state_key128().to_string(),"depth":depth,"cache_key":[damage,claw,key.to_string()]})
        } else {
            Value::Null
        };
        let bounds = if let Some(value) = terminal {
            if self.proof {
                record["kind"] = json!("terminal");
            }
            [value, value]
        } else if depth >= self.rules.cfg.horizon || self.rules.runs >= self.rules.cfg.work_budget {
            if self.proof {
                record["kind"] = json!(if depth >= self.rules.cfg.horizon {
                    "horizon"
                } else {
                    "budget"
                });
            }
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
            let mut bounds = [1f64, 1f64];
            let mut branches = Vec::new();
            let mut representatives: Vec<([Option<SearchChoice>; 2], [f64; 2])> = Vec::new();
            for &a in &selected[0] {
                for &b in &selected[1] {
                    let choices = [a, b];
                    let side = self.rules.cfg.target_side;
                    let alias = if self.proof {
                        None
                    } else {
                        representatives.iter().position(|(prior, _)| {
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
                        })
                    };
                    let mut children = Vec::new();
                    let mut kernel = BTreeMap::<u128, f64>::new();
                    let mut outcome_audit = Value::Null;
                    let mut pending = 1.;
                    let mut branch_runs = 0;
                    let branch = if let Some(i) = alias {
                        representatives[i].1
                    } else {
                        let mut sum = [0f64, 0f64];
                        let cap = self
                            .rules
                            .cfg
                            .cell_cap
                            .min(self.rules.cfg.work_budget.saturating_sub(self.rules.runs));
                        let mut stream = stream_step::Stream::new(
                            dex,
                            &battle,
                            choices,
                            cap,
                            self.ordered.then_some(self.rules.cfg.target_side),
                        );
                        let final_step = depth + 1 == self.rules.cfg.horizon;
                        let mut projected = BTreeMap::<u128, f64>::new();
                        let mut eager = if final_step {
                            let step =
                                nc2000_engine::battle::enumerate::enumerate_step_with_damage_mode(
                                    dex,
                                    &battle,
                                    choices,
                                    cap,
                                    nc2000_engine::prng::DamageRollMode::ThresholdLean,
                                );
                            let leaves = if let Some(step) = step {
                                branch_runs = step.runs;
                                step.leaves
                            } else {
                                branch_runs = cap;
                                Vec::new()
                            };
                            self.rules.runs += branch_runs;
                            if self.audit {
                                for leaf in &leaves {
                                    *projected.entry(projected_key(&leaf.battle)).or_default() +=
                                        leaf.prob;
                                }
                            }
                            Some(leaves.into_iter())
                        } else if depth == 0 {
                            let mut step = enumerate_step_partial(dex, &battle, choices, cap);
                            branch_runs = step.runs;
                            self.rules.runs += step.runs;
                            if target < 1. {
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
                            Some(step.leaves.into_iter())
                        } else {
                            None
                        };
                        let mut emitted = 0.;
                        let mut count = 0;
                        loop {
                            let pad = 1e-12 * (count + 2) as f64;
                            if sum[0] - pad > target {
                                break;
                            }
                            stream.restrict(
                                self.rules.cfg.work_budget.saturating_sub(self.rules.runs),
                            );
                            let leaf = if let Some(leaves) = eager.as_mut() {
                                leaves.next()
                            } else {
                                let before = stream.runs;
                                let leaf = stream.next();
                                self.rules.runs += stream.runs - before;
                                leaf
                            };
                            let Some(leaf) = leaf else {
                                break;
                            };
                            emitted += leaf.prob;
                            let child_target = ((target + 1e-9 - sum[0]) / leaf.prob).clamp(0., 1.);
                            if self.audit {
                                let key = if final_step {
                                    projected_key(&leaf.battle)
                                } else {
                                    leaf.battle.state_key128()
                                };
                                *kernel.entry(key).or_default() += leaf.prob;
                            }
                            let child = self.visit(leaf.battle, depth + 1, child_target);
                            for k in 0..2 {
                                sum[k] += leaf.prob * self.values[child][k];
                            }
                            count += 1;
                            if self.proof {
                                children.push(json!({"mass":leaf.prob,"node":child}));
                            }
                        }
                        pending = (1. - emitted).max(0.);
                        branch_runs += stream.runs;
                        assert!(pending <= 1. && self.rules.runs <= self.rules.cfg.work_budget);
                        if self.audit && branch_runs > 0 {
                            let full = nc2000_engine::battle::enumerate::enumerate_step(
                                dex, &battle, choices, 200000,
                            )
                            .expect("audit cap");
                            self.audit_runs += full.runs;
                            self.audit_cells += 1;
                            let mut expected = BTreeMap::<u128, f64>::new();
                            for leaf in full.leaves {
                                let key = if final_step {
                                    projected_key(&leaf.battle)
                                } else {
                                    leaf.battle.state_key128()
                                };
                                *expected.entry(key).or_default() += leaf.prob;
                            }
                            if final_step && !projected.is_empty() {
                                for key in 0..4 {
                                    let actual = projected.get(&key).copied().unwrap_or(0.);
                                    let native = expected.get(&key).copied().unwrap_or(0.);
                                    assert!((actual-native).abs()<1e-10,
                                        "terminal projection mismatch depth={depth} state={} choices={:?} projected={projected:?} native={expected:?}",
                                        battle.state_key128(), choices.map(|a|a.map(|a|a.to_input(dex))));
                                }
                            }
                            if final_step {
                                outcome_audit = json!({
                                    "native":(0..4).map(|k|expected.get(&k).copied().unwrap_or(0.)).collect::<Vec<_>>(),
                                    "abstract":(0..4).map(|k|projected.get(&k).copied().unwrap_or(0.)).collect::<Vec<_>>()
                                });
                            }
                            for (key, mass) in &kernel {
                                assert!(*mass <= expected.get(key).copied().unwrap_or(0.) + 1e-10);
                            }
                            assert!((kernel.values().sum::<f64>() + pending - 1.).abs() < 1e-9);
                        }
                        let pad = 1e-12 * (count + 2) as f64;
                        let branch = [(sum[0] - pad).max(0.), (sum[1] + pending + pad).min(1.)];
                        representatives.push((choices, branch));
                        branch
                    };
                    for k in 0..2 {
                        bounds[k] = bounds[k].min(branch[k]);
                    }
                    if self.proof {
                        branches.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(dex))),
                    "runs":branch_runs,"pending":pending,"children":children,"bounds":branch,"outcome_audit":outcome_audit}));
                    }
                    if depth == 0 {
                        self.root_branches.push(json!({"choices":choices.map(|a|a.map(|a|a.to_input(dex))),"bounds":branch}));
                    }
                }
            }
            if self.proof {
                let names = |v: [Vec<Option<SearchChoice>>; 2]| {
                    v.map(|v| {
                        v.into_iter()
                            .map(|a| a.map(|a| a.to_input(dex)))
                            .collect::<Vec<_>>()
                    })
                };
                record["kind"] = json!("decision");
                record["maximizing"] = json!(false);
                record["legal"] = json!(names(legal));
                record["selected"] = json!(names(selected));
                record["branches"] = json!(branches);
            }
            bounds
        };
        let id = self.values.len();
        self.values.push(bounds);
        if self.proof {
            record["bounds"] = json!(bounds);
            self.records.push(record);
        }
        if let Some(key) = cache {
            self.memo.insert(key, id);
        }
        id
    }
}

pub fn solve(
    dex: &Dex,
    battle: Battle,
    config: Value,
    threshold: f64,
    proof: bool,
    audit: bool,
    ordered: bool,
) -> Value {
    let cfg: Config = serde_json::from_value(config).unwrap();
    assert!(cfg.target_side < 2 && cfg.target_side == cfg.policy_side && cfg.guarded_keys);
    assert!(
        cfg.horizon > 0
            && cfg.cell_cap > 0
            && cfg.work_budget > 0
            && (0. ..=1.).contains(&threshold)
    );
    assert!(
        !cfg.share_root_budget
            && !cfg.material_order
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
    let mut s = Streaming {
        rules: Witness {
            dex,
            cfg,
            runs: 0,
            memo: HashMap::new(),
            nodes: Vec::new(),
            prior: HashMap::new(),
            prioritize: false,
        },
        memo: HashMap::new(),
        values: Vec::new(),
        records: Vec::new(),
        proof,
        audit,
        ordered,
        audit_runs: 0,
        audit_cells: 0,
        root_branches: Vec::new(),
    };
    let id = s.visit(battle, 0, threshold);
    let bounds = s.values[id];
    let nodes = if proof {
        s.records
    } else {
        vec![json!({"state_key":key,"bounds":bounds,"branches":s.root_branches})]
    };
    json!({"config":s.rules.cfg,"runs":s.rules.runs,"root":if proof {id} else {0},"bounds":bounds,"nodes":nodes,
        "streamed":true,"eager_root":true,"terminal_projection":true,"ordered":ordered,"threshold":threshold,"audit_runs":s.audit_runs,"audit_cells":s.audit_cells})
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::enumerate::{enumerate_step, enumerate_step_with_damage_mode};
    use nc2000_engine::prng::DamageRollMode;

    #[test]
    fn sandstorm_counterexample_rejects_unconditional_terminal_projection() {
        let dex = conformance::load_dex();
        let mon = |species: &str, moves: Vec<&str>| {
            serde_json::from_value::<PokemonSet>(
                json!({"species":species,"name":species,"level":50,"moves":moves}),
            )
            .unwrap()
        };
        let own = vec![
            mon("Marowak", vec!["Rock Slide"]),
            mon("Snorlax", vec!["Tackle"]),
            mon("Tauros", vec!["Tackle"]),
        ];
        let foe = vec![
            mon("Zapdos", vec!["Sandstorm"]),
            mon("Miltank", vec!["Tackle"]),
            mon("Starmie", vec!["Tackle"]),
        ];
        let mut base = Battle::from_fixture(&dex, "1,2,3,4", &own, &foe).unwrap();
        let preview = [0, 1].map(|s| Some(base.legal_choices(&dex, s)[0]));
        base.apply_choices(&dex, preview).unwrap();
        for side in &mut base.sides {
            let active = side.active.unwrap() as usize;
            for (i, mon) in side.roster.iter_mut().enumerate() {
                if i != active {
                    mon.hp = 0;
                    mon.fainted = true;
                }
            }
            side.pokemon_left = 1;
        }
        let slot = base.sides[1].active.unwrap() as usize;
        let choices = [0, 1].map(|s| {
            Some(
                base.legal_choices(&dex, s)
                    .into_iter()
                    .find(|a| matches!(a, SearchChoice::Move(_)))
                    .unwrap(),
            )
        });
        let project = |step: nc2000_engine::battle::enumerate::StepEnum| {
            let mut values = [0.; 4];
            for leaf in step.leaves {
                values[projected_key(&leaf.battle) as usize] += leaf.prob;
            }
            values
        };
        for hp in 1..=base.sides[1].roster[slot].maxhp {
            let mut battle = base.clone();
            battle.sides[1].roster[slot].hp = hp;
            let native = project(enumerate_step(&dex, &battle, choices, 200000).unwrap());
            let abstracted = project(
                enumerate_step_with_damage_mode(
                    &dex,
                    &battle,
                    choices,
                    200000,
                    DamageRollMode::ThresholdLean,
                )
                .unwrap(),
            );
            if native
                .iter()
                .zip(abstracted)
                .any(|(a, b)| (a - b).abs() > 1e-6)
            {
                println!(
                    "counterexample {}",
                    json!({"hp":hp,"native":native,"abstract":abstracted})
                );
                return;
            }
        }
        panic!("expected a residual-damage counterexample");
    }
}
