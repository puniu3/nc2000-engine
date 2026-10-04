use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_bot::player::action_input;
use nc2000_bot::{RmConfig, SkuctSearch};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Decision {
    key: String,
    choices: [Option<usize>; 2],
}

#[derive(Deserialize)]
struct Game {
    schema: String,
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    decisions: Vec<Decision>,
    samples: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    game: usize,
    decision: usize,
    side: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    source: String,
    selection: Vec<Selection>,
    selection_reason: String,
    iterations: Vec<u32>,
    search_seeds: usize,
    seed: u64,
    threads: usize,
    scope: String,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let coverage = args.get(2).is_some_and(|a| a == "--coverage");
    assert!(
        args.len() == 2 || args.len() == 4 || (args.len() == 3 && coverage),
        "endgame_case_audit CORPUS.jsonl [--coverage | PROTOCOL.json OUTPUT_DIRECTORY]"
    );
    let dex = conformance::load_dex();
    let mut states = Vec::new();
    for line in BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Game = serde_json::from_str(&line.unwrap()).unwrap();
        assert_eq!(game.schema, "nc2000-selfplay-endgames-v1");
        let mut battle =
            Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        let mut chosen_moves = Vec::new();
        for (decision, record) in game.decisions.iter().enumerate() {
            assert_eq!(battle.state_key128().to_string(), record.key);
            let legal = [0, 1].map(|side| battle.legal_choices(&dex, side));
            if game.samples.contains(&decision) && !coverage {
                let sides = [0, 1].map(|side| {
                    let s = &battle.sides[side];
                    s.party.iter().map(|&slot| {
                        let p = &s.roster[slot as usize];
                        json!({"slot":slot,"position":p.position,"active":s.active==Some(slot),
                            "species":dex.species.key(p.species),"level":p.level,
                            "hp":p.hp,"maxhp":p.maxhp,"fainted":p.fainted,
                            "status":p.status.as_str(),"status_state":format!("{:?}",p.status_state),
                            "boosts":p.boosts,"item":p.item.map(|i|dex.items.key(i)),
                            "trapped":p.trapped,"previously_switched_in":p.previously_switched_in,
                            "volatiles":p.volatiles.iter().map(|(id,e)|json!({"id":dex.conds_key(*id),"state":format!("{e:?}")})).collect::<Vec<_>>(),
                            "moves":p.move_slots.iter().map(|m|json!({"id":dex.moves.key(m.id),"pp":m.pp,"maxpp":m.maxpp,"disabled":m.disabled,"used":m.used})).collect::<Vec<_>>()})
                    }).collect::<Vec<_>>()
                });
                let record = json!({"game":game.game,"decision":decision,"turn":battle.turn,
                    "state_key":record.key,"sides":sides,"essence":battle.essence(&dex),
                    "actions":legal.clone().map(|aa|aa.into_iter().map(|a|action_input(&dex,a)).collect::<Vec<_>>())});
                if args.len() == 2 {
                    println!("{record}");
                } else {
                    states.push((battle.clone(), record));
                }
            }
            if coverage {
                for side in 0..2 {
                    if let Some(index) = record.choices[side] {
                        let action = action_input(&dex, legal[side][index]);
                        if action.starts_with("move ") {
                            let slot = battle.sides[side].active.unwrap();
                            let mon = &battle.sides[side].roster[slot as usize];
                            chosen_moves
                                .push(json!({"decision":decision,"turn":battle.turn,"side":side,
                                "species":dex.species.key(mon.species),"action":action}));
                        }
                    }
                }
            }
            battle
                .apply_choices(
                    &dex,
                    [0, 1].map(|side| record.choices[side].map(|a| legal[side][a])),
                )
                .unwrap();
        }
        if coverage {
            let picked = [0,1].map(|side|battle.sides[side].party.iter().map(|&slot| {
                let mon=&battle.sides[side].roster[slot as usize];
                json!({"species":dex.species.key(mon.base_species),"moves":mon.base_move_slots.iter().map(|m|dex.moves.key(m.id)).collect::<Vec<_>>()})
            }).collect::<Vec<_>>());
            println!(
                "{}",
                json!({"game":game.game,"picked":picked,"chosen_moves":chosen_moves,"sample_count":game.samples.len()})
            );
        }
    }
    if args.len() == 2 || coverage {
        return;
    }
    let bytes = std::fs::read(&args[2]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-headroom-cases-v1");
    assert!(
        !protocol.source.is_empty()
            && !protocol.selection_reason.is_empty()
            && !protocol.scope.is_empty()
    );
    assert!(protocol.threads > 0 && protocol.search_seeds > 0 && !protocol.iterations.is_empty());
    assert!(protocol.iterations[0] > 0 && protocol.iterations.windows(2).all(|w| w[0] < w[1]));
    for selected in &protocol.selection {
        assert!(selected.side < 2);
        assert_eq!(
            states
                .iter()
                .filter(|(_, r)| r["game"] == selected.game && r["decision"] == selected.decision)
                .count(),
            1
        );
    }
    states.retain(|(_, r)| {
        protocol
            .selection
            .iter()
            .any(|s| r["game"] == s.game && r["decision"] == s.decision)
    });
    let directory = std::path::Path::new(&args[3]);
    std::fs::create_dir(directory).unwrap();
    let hash = |p: &str| format!("{:x}", Sha256::digest(std::fs::read(p).unwrap()));
    let sources = [
        "crates/bot/examples/endgame_case_audit.rs",
        "crates/bot/src/smmcts.rs",
        "data/gen2stadium2.json",
    ];
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),
        "corpus_sha256":hash(&args[1]),"protocol_sha256":hash(&args[2]),
        "binary_sha256":hash(std::env::current_exe().unwrap().to_str().unwrap()),
        "source_hashes":sources.map(|p|(p,hash(p))),"expected_states":states.len()});
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some((battle, snapshot)) = states.get(index) else { break };
                let mut records = Vec::new();
                for seed_index in 0..protocol.search_seeds {
                    let seed = protocol.seed
                        ^ snapshot["game"].as_u64().unwrap().wrapping_mul(0x9e3779b97f4a7c15)
                        ^ snapshot["decision"].as_u64().unwrap().wrapping_mul(0xbf58476d1ce4e5b9)
                        ^ (seed_index as u64).wrapping_mul(0x94d049bb133111eb);
                    let mut search = SkuctSearch::new(battle, &dex, RmConfig { c: 0.4, ..Default::default() }, seed);
                    let started = Instant::now();
                    for &iterations in &protocol.iterations {
                        search.step(&dex, iterations - search.iterations());
                        records.push(json!({"seed_index":seed_index,"seed":seed,"iterations":iterations,
                            "elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"nodes":search.node_count(),
                            "mean_depth":search.mean_depth(),
                            "sides":([0,1].map(|side| json!({
                                "choice":search.best(side).map(|a|action_input(&dex,a)),
                                "actions":search.actions(side).iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>(),
                                "visits":search.visits(side),"means":search.means(side)})))}));
                    }
                }
                let record = json!({"snapshot":snapshot,"search_records":records});
                std::fs::write(directory.join(format!("game-{:03}-d{:04}.json",snapshot["game"].as_u64().unwrap(),snapshot["decision"].as_u64().unwrap())),serde_json::to_vec_pretty(&record).unwrap()).unwrap();
                eprintln!("{}/{} game={} decision={}", index+1, states.len(), snapshot["game"], snapshot["decision"]);
            });
        }
    });
}
