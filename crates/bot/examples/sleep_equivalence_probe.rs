use nc2000_bot::{
    preview::load_meta_pool,
    smmcts::{collapse_sleep_choices, SearchTrace, SelRule},
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::BufRead, time::Instant};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[3] == "--verify-corpus"),
        "corpus output [--verify-corpus]"
    );
    let verify_corpus = args.len() == 4;
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut cases = Vec::new();
    let mut census = [0usize; 3];
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        let mut observers = [0, 1].map(|side| Observer::new(&b, side));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let legal = [0, 1].map(|side| b.legal_choices(&dex, side));
            for side in 0..2 {
                if legal[side].len() < 2 || b.active_id(side).is_none() {
                    continue;
                }
                census[0] += 1;
                let mut reduced = legal[side].clone();
                collapse_sleep_choices(&b, &dex, side, &mut reduced);
                census[1] += usize::from(reduced.len() < legal[side].len());
                census[2] += legal[side].len() - reduced.len();
                let order = b
                    .state_key128()
                    .wrapping_mul(0x9e3779b97f4a7c15)
                    .wrapping_add(side as u128);
                cases.push((
                    order,
                    game["game"].clone(),
                    decision,
                    side,
                    b.clone(),
                    observers[side].clone(),
                ));
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|side| d["choices"][side].as_u64().map(|i| legal[side][i as usize])),
            )
            .unwrap();
            for observer in &mut observers {
                observer.observe(&b, &dex);
            }
        }
    }
    cases.sort_by_key(|c| c.0);
    if verify_corpus {
        let mut rows = Vec::new();
        for (_, game, decision, side, mut battle, _) in cases {
            let legal = battle.legal_choices(&dex, side);
            let mut reduced = legal.clone();
            collapse_sleep_choices(&battle, &dex, side, &mut reduced);
            if legal == reduced {
                continue;
            }
            let mut comparisons = 0;
            let mut unresolved = 0;
            let mut runs = 0;
            for reply in battle.legal_choices(&dex, 1 - side) {
                let mut distributions = BTreeMap::new();
                for (index, &action) in legal.iter().enumerate() {
                    if reduced.contains(&action) {
                        continue;
                    }
                    let SearchChoice::Move(id) = action else {
                        unreachable!()
                    };
                    let representative = legal
                        .iter()
                        .position(|&candidate| {
                            let SearchChoice::Move(other) = candidate else {
                                return false;
                            };
                            let ms = dex.move_static(other);
                            reduced.contains(&candidate)
                                && ms.priority == dex.move_static(id).priority
                                && !ms.sleep_usable
                                && !ms.callbacks.iter().any(|key| key == "beforeTurnCallback")
                        })
                        .unwrap();
                    for index in [index, representative] {
                        distributions.entry(index).or_insert_with(|| {
                            let joint = if side == 0 {
                                [legal[index], reply]
                            } else {
                                [reply, legal[index]]
                            };
                            enumerate_step(&dex, &battle, joint.map(Some), 50_000).map(|step| {
                                runs += step.runs;
                                let mut out = BTreeMap::<u128, f64>::new();
                                for leaf in step.leaves {
                                    *out.entry(leaf.battle.state_key128()).or_default() +=
                                        leaf.prob;
                                }
                                assert!((out.values().sum::<f64>() - 1.0).abs() < 1e-10);
                                out
                            })
                        });
                    }
                    match (&distributions[&index], &distributions[&representative]) {
                        (Some(a), Some(b)) => {
                            assert_eq!(a,b,"game={game} decision={decision} side={side} action={action:?} reply={reply:?}");
                            comparisons += 1;
                        }
                        _ => unresolved += 1,
                    }
                }
            }
            rows.push(json!({"game":game,"decision":decision,"side":side,"key":battle.state_key128().to_string(),"comparisons":comparisons,"unresolved":unresolved,"completed_enumeration_runs":runs}));
        }
        std::fs::write(
            &args[2],
            serde_json::to_vec(&json!({"census":census,"rows":rows})).unwrap(),
        )
        .unwrap();
        return;
    }
    cases.truncate(24);
    let mut rows = Vec::new();
    for (index, (order, game, decision, side, battle, observer)) in cases.into_iter().enumerate() {
        let seed = 0x20261008 ^ order as u64;
        let belief = Belief::new(&dex, &pool, &observer);
        let cfg = RmConfig {
            c: 0.4,
            rule: SelRule::Ucb,
            ..Default::default()
        };
        let mut profile = [0usize; 4];
        let mut instrumented = BlindSearch::new(&battle, &dex, cfg.clone(), side, seed);
        for _ in 0..1000 {
            let mut first = true;
            instrumented.step_observed(&dex, &belief, &observer, 1, &mut |trace| {
                if let SearchTrace::Choice {
                    battle, actions, ..
                } = trace
                {
                    if first {
                        first = false;
                        return;
                    }
                    for side in 0..2 {
                        let mut reduced = actions[side].clone();
                        collapse_sleep_choices(battle, &dex, side, &mut reduced);
                        profile[0] += 1;
                        profile[1] += usize::from(reduced.len() != actions[side].len());
                        profile[2] += actions[side].len();
                        profile[3] += actions[side].len() - reduced.len();
                    }
                }
            });
        }
        let mut arms = Vec::new();
        for enabled in if index % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let cfg = RmConfig {
                collapse_sleep_actions: enabled,
                ..cfg.clone()
            };
            let mut search = BlindSearch::new(&battle, &dex, cfg, side, seed);
            let start = Instant::now();
            let mut checkpoints = Vec::new();
            for budget in [3000, 27000] {
                search.step(&dex, &belief, &observer, budget - search.iterations());
                checkpoints.push(json!({"iterations":budget,"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means(),"nodes":search.node_count(),"elapsed_ms":start.elapsed().as_secs_f64()*1000.}));
            }
            arms.push(json!({"enabled":enabled,"actions":search.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"checkpoints":checkpoints}));
        }
        arms.sort_by_key(|a| a["enabled"].as_bool().unwrap());
        rows.push(json!({"game":game,"decision":decision,"side":side,"seed":seed,"key":battle.state_key128().to_string(),"profile":profile,"arms":arms}));
        eprintln!("sleep root {}/24 complete", index + 1);
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec(&json!({"census":census,"rows":rows})).unwrap(),
    )
    .unwrap();
}
