#[allow(dead_code)]
mod witness {
    include!("policy_witness.rs");

    fn score(b: &Battle, side: usize) -> (i32, i32) {
        let material =
            i32::from(b.sides[side].pokemon_left) - i32::from(b.sides[1 - side].pokemon_left);
        let hp = |s: usize| b.sides[s].roster.iter().map(|p| p.hp.max(0)).sum::<i32>();
        (material, hp(side) - hp(1 - side))
    }

    fn guided_step(
        dex: &Dex,
        base: &Battle,
        choices: [Option<SearchChoice>; 2],
        cap: usize,
        side: usize,
    ) -> (
        Vec<nc2000_engine::battle::enumerate::ChanceLeaf>,
        f64,
        usize,
    ) {
        use nc2000_engine::battle::enumerate::{run_scripted, ChanceLeaf};
        use nc2000_engine::prng::Draw;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut queue = BinaryHeap::from([((0, 0), Reverse(0usize), Vec::<usize>::new())]);
        let mut scheduled = HashSet::from([Vec::<usize>::new()]);
        let mut observed = BTreeMap::new();
        let mut runs = 0;
        while runs < cap {
            let Some((_, _, prefix)) = queue.pop() else {
                break;
            };
            let (battle, trace) = run_scripted(dex, base, choices, &prefix);
            runs += 1;
            let script: Vec<_> = trace.iter().map(|d| d.chosen).collect();
            if observed.contains_key(&script) {
                continue;
            }
            let priority = score(&battle, side);
            let mass = trace
                .iter()
                .map(|draw| {
                    assert_eq!(draw.counts.iter().sum::<u64>(), Draw::TOTAL);
                    assert!(draw.counts[draw.chosen] > 0);
                    draw.prob()
                })
                .product();
            for (i, draw) in trace.iter().enumerate() {
                let choices: Vec<_> = draw
                    .counts
                    .iter()
                    .enumerate()
                    .filter_map(|(c, &n)| (n > 0).then_some(c))
                    .collect();
                for (j, &c) in choices.iter().enumerate() {
                    if c == draw.chosen || choices.len() > 4 && j != 0 && j + 1 != choices.len() {
                        continue;
                    }
                    let mut next = script[..i].to_vec();
                    next.push(c);
                    if scheduled.insert(next.clone()) {
                        queue.push((priority, Reverse(scheduled.len()), next));
                    }
                }
            }
            observed.insert(
                script,
                ChanceLeaf {
                    battle,
                    prob: mass,
                    draws: trace.len(),
                },
            );
        }
        let scripts: Vec<_> = observed.keys().collect();
        assert!(scripts.windows(2).all(|pair| !pair[1].starts_with(pair[0])));
        let leaves: Vec<_> = observed.into_values().collect();
        let mass = leaves.iter().map(|l| l.prob).sum::<f64>();
        assert!(mass <= 1. + 1e-10);
        (leaves, (1. - mass).max(0.), runs)
    }

    impl Witness<'_> {
        fn support_visit(&mut self, mut battle: Battle, depth: usize, limit: usize) -> usize {
            let cache_key = self.key(&battle, depth);
            if let Some(&id) = self.memo.get(&cache_key) {
                return id;
            }
            let (damage, claw, semantic, _) = cache_key;
            let terminal = battle.outcome().map(|o| match o {
                Outcome::Tie => 0.5,
                Outcome::P1Win => f64::from(self.cfg.target_side == 0),
                Outcome::P2Win => f64::from(self.cfg.target_side == 1),
            });
            let mut record = json!({"state_key": battle.state_key128().to_string(), "depth": depth,
                "cache_key": [damage, claw, semantic.to_string()]});
            if let Some(value) = terminal {
                record["kind"] = json!("terminal");
                record["bounds"] = json!([value, value]);
            } else if depth >= self.cfg.horizon || self.runs >= limit {
                record["kind"] = json!(if depth >= self.cfg.horizon {
                    "horizon"
                } else {
                    "budget"
                });
                record["bounds"] = json!([0., 1.]);
            } else {
                let legal = [0, 1].map(|s| {
                    if battle.needs_choice()[s] {
                        battle
                            .legal_choices(self.dex, s)
                            .into_iter()
                            .map(Some)
                            .collect::<Vec<_>>()
                    } else {
                        vec![None]
                    }
                });
                let selected = [0, 1].map(|s| self.selected(&battle, s, &legal[s], depth));
                let mut bounds = [1f64, 1f64];
                let mut branches = Vec::new();
                let mut branches_left = selected[0].len() * selected[1].len();
                for &a in &selected[0] {
                    for &b in &selected[1] {
                        let branch_limit =
                            self.runs + limit.saturating_sub(self.runs) / branches_left;
                        branches_left -= 1;
                        let remaining = branch_limit.saturating_sub(self.runs);
                        let mut children = Vec::new();
                        let mut branch = [0., 1.];
                        let mut pending = 1.;
                        let mut runs = 0;
                        let mut deferred = 0.;
                        let mut engine_pending = 1.;
                        if remaining > 0 {
                            let (mut leaves, missing, executions) = if self.prioritize {
                                guided_step(
                                    self.dex,
                                    &battle,
                                    [a, b],
                                    self.cfg.cell_cap.min(remaining),
                                    self.cfg.target_side,
                                )
                            } else {
                                let step = enumerate_step_partial(
                                    self.dex,
                                    &battle,
                                    [a, b],
                                    self.cfg.cell_cap.min(remaining),
                                );
                                (
                                    step.leaves,
                                    step.pending.iter().map(|p| p.mass).sum::<f64>(),
                                    step.runs,
                                )
                            };
                            runs = executions;
                            self.runs += runs;
                            engine_pending = missing;
                            pending = engine_pending;
                            branch = [0., engine_pending];
                            leaves.sort_by_cached_key(|leaf| {
                                std::cmp::Reverse(score(&leaf.battle, self.cfg.target_side))
                            });
                            let mut covered = engine_pending;
                            for leaf in leaves {
                                covered += leaf.prob;
                                if (branch[0] > 1e-8 || self.runs >= branch_limit)
                                    && leaf.battle.outcome().is_none()
                                {
                                    pending += leaf.prob;
                                    deferred += leaf.prob;
                                    branch[1] += leaf.prob;
                                    continue;
                                }
                                let child =
                                    self.support_visit(leaf.battle, depth + 1, branch_limit);
                                for k in 0..2 {
                                    branch[k] += leaf.prob
                                        * self.nodes[child]["bounds"][k].as_f64().unwrap();
                                }
                                children.push(json!({"mass": leaf.prob, "node": child}));
                            }
                            assert!((covered - 1.).abs() < 1e-10);
                            pending = pending.min(1.);
                            let pad = 1e-12 * (children.len() + 2) as f64;
                            branch = [(branch[0] - pad).max(0.), (branch[1] + pad).min(1.)];
                        }
                        for k in 0..2 {
                            bounds[k] = bounds[k].min(branch[k]);
                        }
                        branches.push(
                            json!({"choices": ([a, b].map(|a| a.map(|a| a.to_input(self.dex)))),
                        "runs": runs, "pending": pending, "engine_pending": engine_pending,
                        "deferred_mass": deferred, "children": children, "bounds": branch}),
                        );
                    }
                }
                record["kind"] = json!("decision");
                record["maximizing"] = json!(false);
                record["legal"] = json!(legal.map(|aa| aa
                    .into_iter()
                    .map(|a| a.map(|a| a.to_input(self.dex)))
                    .collect::<Vec<_>>()));
                record["selected"] = json!(selected.map(|aa| aa
                    .into_iter()
                    .map(|a| a.map(|a| a.to_input(self.dex)))
                    .collect::<Vec<_>>()));
                record["branches"] = json!(branches);
                record["bounds"] = json!(bounds);
            }
            let id = self.nodes.len();
            self.nodes.push(record);
            self.memo.insert(cache_key, id);
            id
        }
    }

    pub fn run() {
        let args: Vec<_> = std::env::args().collect();
        assert!(
            (4..=5).contains(&args.len()),
            "corpus config output [guided]"
        );
        let guided = args.get(4).is_some_and(|s| {
            assert_eq!(s, "guided");
            true
        });
        let cfg: Config = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
        assert!(cfg.target_side < 2 && cfg.target_side == cfg.policy_side);
        assert!(cfg.horizon > 0 && cfg.work_budget > 0 && cfg.cell_cap > 0);
        assert!(
            !cfg.share_root_budget
                && cfg.share_budget_depth.is_none()
                && !cfg.share_unique_children
        );
        assert!(
            cfg.minimum_target_living.is_none() && !cfg.stop_if_asleep && !cfg.support_certificate
        );
        let dex = conformance::load_dex();
        let game = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap())
            .lines()
            .map(|l| serde_json::from_str::<Game>(&l.unwrap()).unwrap())
            .find(|g| g.game == cfg.game)
            .unwrap();
        let mut battle =
            Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        for d in game.decisions.iter().take(cfg.decision) {
            assert_eq!(battle.state_key128().to_string(), d.key);
            let legal = [0, 1].map(|s| battle.legal_choices(&dex, s));
            battle
                .apply_choices(&dex, [0, 1].map(|s| d.choices[s].map(|i| legal[s][i])))
                .unwrap();
        }
        assert_eq!(
            battle.state_key128().to_string(),
            game.decisions[cfg.decision].key
        );
        let mut w = Witness {
            dex: &dex,
            cfg,
            runs: 0,
            memo: HashMap::new(),
            nodes: Vec::new(),
            prior: HashMap::new(),
            prioritize: guided,
        };
        let root = w.support_visit(battle, 0, w.cfg.work_budget);
        let output = json!({"config": w.cfg, "runs": w.runs, "root": root,
            "bounds": w.nodes[root]["bounds"], "nodes": w.nodes,
            "scheduler": if guided { "guided-support-v1" } else { "positive-support-v1" }});
        std::fs::write(&args[3], serde_json::to_vec(&output).unwrap()).unwrap();
        eprintln!(
            "runs={} nodes={} bounds={}",
            w.runs,
            w.nodes.len(),
            output["bounds"]
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn guided_support_is_a_submeasure_of_the_exhaustive_kernel() {
            let dex = conformance::load_dex();
            for moves in [["Seismic Toss", "Splash"], ["Body Slam", "Thunderbolt"]] {
                let mon = PokemonSet {
                    name: "Pikachu".into(),
                    species: "Pikachu".into(),
                    item: String::new(),
                    ability: String::new(),
                    moves: moves.iter().map(|m| (*m).into()).collect(),
                    level: 50,
                    evs: None,
                    ivs: None,
                    happiness: None,
                    gender: None,
                };
                let mut b = Battle::from_fixture(&dex, "1,2,3,4", &[mon.clone()], &[mon]).unwrap();
                let preview = [0, 1].map(|s| b.legal_choices(&dex, s).first().copied());
                b.apply_choices(&dex, preview).unwrap();
                for s in [0, 1] {
                    let id = b.active_id(s).unwrap();
                    b.poke_mut(id).hp = 1;
                }
                let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
                for &a in &legal[0] {
                    for &c in &legal[1] {
                        let exact = enumerate_step_partial(&dex, &b, [Some(a), Some(c)], 10000);
                        assert!(!exact.exhausted);
                        let mut reference = BTreeMap::<u128, f64>::new();
                        for leaf in exact.leaves {
                            *reference.entry(leaf.battle.state_key128()).or_default() += leaf.prob;
                        }
                        for cap in [1, 8, 128] {
                            for side in [0, 1] {
                                let (leaves, pending, runs) =
                                    guided_step(&dex, &b, [Some(a), Some(c)], cap, side);
                                assert!(runs > 0 && runs <= cap);
                                let mut found = BTreeMap::<u128, f64>::new();
                                for leaf in leaves {
                                    *found.entry(leaf.battle.state_key128()).or_default() +=
                                        leaf.prob;
                                }
                                assert!((pending + found.values().sum::<f64>() - 1.).abs() < 1e-10);
                                for (key, mass) in found {
                                    assert!(mass <= reference[&key] + 1e-10);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    witness::run();
}
