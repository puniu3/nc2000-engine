#![allow(dead_code)]

include!("policy_witness.rs");

#[path = "terminal_projection_guard.rs"]
mod projection_guard;
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
                    let last_step = depth + 1 == self.rules.cfg.horizon;
                    let rejection = last_step
                        .then(|| projection_guard::rejection(dex, &battle, choices))
                        .flatten();
                    let project_outcome = last_step && rejection.is_none();
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
                        let final_step = project_outcome;
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
                    "runs":branch_runs,"pending":pending,"children":children,"bounds":branch,"outcome_audit":outcome_audit,"outcome_projected":project_outcome,"projection_rejection":rejection}));
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
        "streamed":true,"eager_root":true,"terminal_projection":true,"projection_guarded":true,"ordered":ordered,"threshold":threshold,"audit_runs":s.audit_runs,"audit_cells":s.audit_cells})
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::enumerate::{
        enumerate_step, enumerate_step_with_damage_mode, StepEnum,
    };
    use nc2000_engine::prng::DamageRollMode;

    fn fixture(dex: &Dex, own_item: &str, foe_item: &str, foe_move: &str) -> Battle {
        let mon = |species: &str, item: &str, moves: Vec<&str>| {
            serde_json::from_value::<PokemonSet>(
                json!({"species":species,"name":species,"level":50,"item":item,"moves":moves}),
            )
            .unwrap()
        };
        let own = vec![
            mon("Marowak", own_item, vec!["Rock Slide"]),
            mon("Snorlax", "", vec!["Tackle"]),
            mon("Tauros", "", vec!["Tackle"]),
        ];
        let foe = vec![
            mon(
                "Zapdos",
                foe_item,
                if foe_move == "Sleep Talk" {
                    vec!["Sleep Talk", "Double-Edge"]
                } else {
                    vec![foe_move]
                },
            ),
            mon("Miltank", "", vec!["Tackle"]),
            mon("Starmie", "", vec!["Tackle"]),
        ];
        let mut base = Battle::from_fixture(dex, "1,2,3,4", &own, &foe).unwrap();
        let preview = [0, 1].map(|s| Some(base.legal_choices(dex, s)[0]));
        base.apply_choices(dex, preview).unwrap();
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
        base
    }

    fn move_choices(dex: &Dex, battle: &mut Battle) -> [Option<SearchChoice>; 2] {
        [0, 1].map(|s| {
            Some(
                battle
                    .legal_choices(dex, s)
                    .into_iter()
                    .find(|a| matches!(a, SearchChoice::Move(_)))
                    .unwrap(),
            )
        })
    }

    fn project(step: StepEnum) -> [f64; 4] {
        let mut values = [0.; 4];
        for leaf in step.leaves {
            values[projected_key(&leaf.battle) as usize] += leaf.prob;
        }
        values
    }

    #[test]
    fn admitted_hp_item_move_grid_preserves_terminal_mass() {
        let dex = conformance::load_dex();
        let mut cases = 0;
        for own_item in ["", "Quick Claw", "Thick Club"] {
            for foe_item in ["", "Focus Band", "Bright Powder", "Gold Berry"] {
                for foe_move in ["Thunderbolt", "Hidden Power Ice", "Rest", "Steel Wing"] {
                    let base = fixture(&dex, own_item, foe_item, foe_move);
                    for own_hp in [1, 33, 100] {
                        for foe_hp in [
                            1,
                            50,
                            77,
                            130,
                            base.sides[1].roster[base.sides[1].active.unwrap() as usize].maxhp,
                        ] {
                            let mut battle = base.clone();
                            for (s, hp) in [own_hp, foe_hp].into_iter().enumerate() {
                                let slot = battle.sides[s].active.unwrap() as usize;
                                battle.sides[s].roster[slot].hp = hp;
                            }
                            let choices = move_choices(&dex, &mut battle);
                            assert_eq!(projection_guard::rejection(&dex, &battle, choices), None);
                            let native =
                                project(enumerate_step(&dex, &battle, choices, 200000).unwrap());
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
                            assert!(native
                                .iter()
                                .zip(abstracted)
                                .all(|(a, b)| (a - b).abs() < 1e-12));
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(cases, 720);
    }

    #[test]
    fn hp_observing_effects_are_rejected() {
        let dex = conformance::load_dex();
        for name in [
            "Sandstorm",
            "Double-Edge",
            "Bide",
            "Swagger",
            "Substitute",
            "Pin Missile",
            "Toxic",
            "Metronome",
        ] {
            let mut battle = fixture(&dex, "", "", name);
            let choices = move_choices(&dex, &mut battle);
            assert_eq!(
                projection_guard::rejection(&dex, &battle, choices),
                Some("selected_move")
            );
        }
        let mut battle = fixture(&dex, "", "", "Sleep Talk");
        let selected = move_choices(&dex, &mut battle);
        assert_eq!(
            projection_guard::rejection(&dex, &battle, selected),
            Some("called_move")
        );
        let mut battle = fixture(&dex, "", "", "Thunderbolt");
        let selected = move_choices(&dex, &mut battle);
        battle.field.weather = dex.conds_id("sandstorm");
        assert_eq!(
            projection_guard::rejection(&dex, &battle, selected),
            Some("weather")
        );
        let mut battle = fixture(&dex, "", "Berserk Gene", "Thunderbolt");
        let choices = move_choices(&dex, &mut battle);
        assert!(projection_guard::rejection(&dex, &battle, choices).is_some());
        let mut battle = fixture(&dex, "", "", "Thunderbolt");
        let choices = move_choices(&dex, &mut battle);
        let slot = battle.sides[1].active.unwrap() as usize;
        for status in [Status::Brn, Status::Psn, Status::Tox] {
            battle.sides[1].roster[slot].status = status;
            assert_eq!(
                projection_guard::rejection(&dex, &battle, choices),
                Some("residual_status")
            );
        }
    }

    #[test]
    fn sandstorm_counterexample_uses_the_exact_fallback() {
        let dex = conformance::load_dex();
        let mut battle = fixture(&dex, "", "", "Sandstorm");
        let slot = battle.sides[1].active.unwrap() as usize;
        battle.sides[1].roster[slot].hp = 77;
        let choices = move_choices(&dex, &mut battle);
        assert_eq!(
            projection_guard::rejection(&dex, &battle, choices),
            Some("selected_move")
        );
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
        assert!((native[1] - abstracted[1]).abs() > 0.08);
        let cfg = json!({"game":0,"decision":0,"target_side":0,"policy_side":0,"first_choice":"move rockslide",
            "priority":["move rockslide"],"rest_at_percent":null,"horizon":1,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
        let result = solve(&dex, battle, cfg, 1., true, true, true);
        let lower = result["bounds"][0].as_f64().unwrap();
        assert!(lower <= native[1] && native[1] - lower < 1e-9);
        let root = &result["nodes"][result["root"].as_u64().unwrap() as usize];
        assert!(root["branches"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b["outcome_projected"] == false));
        println!(
            "fallback {}",
            json!({"native":native,"abstract":abstracted,"guarded_bounds":result["bounds"]})
        );
    }
}
