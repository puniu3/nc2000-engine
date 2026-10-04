use std::collections::BTreeSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::{Agent, BlindAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, Status};
use nc2000_engine::validate::{validate_team, Learnsets};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Family {
    id: String,
    team: String,
    lead: u8,
    required: Vec<u8>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    catalog: String,
    catalog_sha256: String,
    prior: String,
    prior_sha256: String,
    families: Vec<Family>,
    opponents: Vec<String>,
    seed: u64,
    repetitions: usize,
    iterations: u32,
    c: f64,
    max_turns: u16,
    threads: usize,
    arms: Vec<String>,
    #[serde(default)]
    prefix: Vec<[Option<String>; 2]>,
    interpretation: String,
}

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn specialist(
    b: &Battle,
    dex: &Dex,
    side: usize,
    actions: &[SearchChoice],
    family: &str,
    substitute_attempts: &mut usize,
) -> Option<SearchChoice> {
    let me = b.poke(b.active_id(side)?);
    let foe = b.poke(b.active_id(1 - side)?);
    if me.fainted || foe.fainted {
        return None;
    }
    let available = |name| {
        dex.moves
            .id(name)
            .map(SearchChoice::Move)
            .filter(|a| actions.contains(a))
    };
    let volatile = |p: &nc2000_engine::state::Pokemon, name| {
        dex.conds_id(name).is_some_and(|c| p.volatile(c).is_some())
    };
    match family {
        "encore" => {
            let previous = foe.last_move.map(|m| dex.moves.key(m));
            let exploitable = previous.is_some_and(|m| {
                matches!(
                    m,
                    "curse"
                        | "swordsdance"
                        | "bellydrum"
                        | "growth"
                        | "amnesia"
                        | "agility"
                        | "doubleteam"
                        | "substitute"
                        | "rest"
                        | "recover"
                        | "milkdrink"
                        | "softboiled"
                        | "reflect"
                        | "lightscreen"
                        | "spikes"
                )
            });
            if exploitable && !volatile(foe, "encore") {
                return available("encore");
            }
        }
        "toxic" => {
            let immune = foe
                .types
                .iter()
                .any(|t| matches!(dex.type_name(t), "Poison" | "Steel"));
            if foe.status == Status::None && !immune && !volatile(foe, "substitute") {
                return available("toxic");
            }
        }
        "batonpass" => {
            if volatile(me, "substitute") {
                return available("batonpass");
            }
            if available("batonpass").is_some() && me.hp * 2 > me.maxhp && *substitute_attempts < 2
            {
                if let Some(action) = available("substitute") {
                    *substitute_attempts += 1;
                    return Some(action);
                }
            }
        }
        other => panic!("unsupported family {other}"),
    }
    None
}

fn game(
    dex: &Dex,
    catalog: &MetaPool,
    prior: &Arc<MetaPool>,
    protocol: &Protocol,
    family: &Family,
    opponent: &str,
    repetition: usize,
    block: usize,
    arm: &str,
) -> Value {
    let source_side = block % 2;
    let get = |id: &str| &catalog.teams.iter().find(|t| t.id == id).unwrap().sets;
    let teams = if source_side == 0 {
        [get(&family.team), get(opponent)]
    } else {
        [get(opponent), get(&family.team)]
    };
    let mut seeds =
        SplitMix64::new(protocol.seed ^ (block as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let battle_seed = seeds.battle_seed();
    let agent_seeds = [seeds.next(), seeds.next()];
    let mut battle = Battle::from_fixture(dex, &battle_seed, teams[0], teams[1]).unwrap();
    let config = RmConfig {
        iterations: protocol.iterations,
        c: protocol.c,
        ..Default::default()
    };
    let mut agents =
        [0, 1].map(|s| BlindAgent::new(config.clone(), prior.clone(), None, agent_seeds[s]));
    let mut records = Vec::new();
    let mut substitute_attempts = 0;
    let started = Instant::now();
    while battle.outcome().is_none() && battle.turn <= protocol.max_turns {
        let actions = [0, 1].map(|s| battle.legal_choices(dex, s));
        let search_seeds = agents.each_ref().map(|a| a.search_rng_state());
        let ordinary: [Option<SearchChoice>; 2] = std::array::from_fn(|s| {
            (!actions[s].is_empty()).then(|| agents[s].choose(&battle, dex, s, &actions[s]))
        });
        let mut choices = ordinary;
        let preview = matches!(ordinary[source_side], Some(SearchChoice::Team(_)));
        let mut override_reason = None;
        if let Some(SearchChoice::Team(normal)) = ordinary[source_side] {
            let forced = actions[source_side]
                .iter()
                .filter_map(|a| match a {
                    SearchChoice::Team(t)
                        if t[0] == family.lead && family.required.iter().all(|s| t.contains(s)) =>
                    {
                        Some(*t)
                    }
                    _ => None,
                })
                .max_by_key(|t| {
                    (
                        t.iter().filter(|s| normal.contains(s)).count(),
                        std::cmp::Reverse(*t),
                    )
                })
                .unwrap();
            choices[source_side] = Some(SearchChoice::Team(forced));
            override_reason = Some("forced_tactical_lead");
        } else if arm == "specialist" {
            if let Some(action) = specialist(
                &battle,
                dex,
                source_side,
                &actions[source_side],
                &family.id,
                &mut substitute_attempts,
            ) {
                choices[source_side] = Some(action);
                override_reason = Some("public_state_script");
            }
        }
        if let Some(inputs) = protocol.prefix.get(records.len()) {
            choices = [0, 1].map(|side| {
                inputs[side].as_ref().map(|input| {
                    *actions[side]
                        .iter()
                        .find(|a| a.to_input(dex) == *input)
                        .unwrap_or_else(|| panic!("illegal prefix action {input}"))
                })
            });
            override_reason = Some("legal_exposure_prefix");
        }
        for side in 0..2 {
            assert_eq!(choices[side].is_none(), actions[side].is_empty());
            assert!(choices[side].is_none_or(|c| actions[side].contains(&c)));
        }
        let context: [Value; 2] = std::array::from_fn(|side| {
            let Some(belief) = agents[side].belief() else {
                return Value::Null;
            };
            let observer = agents[side].observer().unwrap();
            let mut rng = SplitMix64::new(91771);
            let models: Vec<_> = if preview {
                Vec::new()
            } else {
                (0..8).filter_map(|_| {
                let model = belief.determinize(dex, &battle, observer, &mut rng);
                model.active_id(1-side).map(|id| {
                    let p = model.poke(id);
                    json!({"species":dex.species.key(p.species),"moves":p.move_slots.iter().map(|m|dex.moves.key(m.id)).collect::<Vec<_>>()})
                })
            }).collect()
            };
            json!({"fallback":belief.is_fallback(),"candidates":belief.candidate_count(),"models":models,
                "reveals":observer.mons().iter().map(|m|json!({"species":dex.species.key(m.species),"appeared":m.appeared,"moves":m.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>()})).collect::<Vec<_>>()})
        });
        let before = battle.essence(dex);
        let state_key = battle.state_key128().to_string();
        let log_start = battle.log.len();
        battle.apply_choices(dex, choices).unwrap();
        records.push(
            json!({"decision":records.len(),"state_key":state_key,"state":before,
            "search_seeds":search_seeds,"ordinary":ordinary.map(|a|a.map(|a|a.to_input(dex))),
            "choices":choices.map(|a|a.map(|a|a.to_input(dex))),"override_reason":override_reason,
            "actions":actions.map(|aa|aa.into_iter().map(|a|a.to_input(dex)).collect::<Vec<_>>()),
            "context":context,"log":battle.log[log_start..]}),
        );
    }
    let score = battle.outcome().map(|o| match o {
        Outcome::P1Win => {
            if source_side == 0 {
                1.0
            } else {
                0.0
            }
        }
        Outcome::P2Win => {
            if source_side == 1 {
                1.0
            } else {
                0.0
            }
        }
        Outcome::Tie => 0.5,
    });
    json!({"schema":"nc2000-tactical-exposure-game-v1","block":block,"family":family.id,"opponent":opponent,
        "source_team":family.team,"source_side":source_side,"repetition":repetition,"arm":arm,
        "battle_seed":battle_seed,"agent_seeds":agent_seeds,"teams":teams,"source_score":score,
        "terminal":battle.outcome().is_some(),"outcome":format!("{:?}",battle.outcome()),"turns":battle.turn,
        "seconds":started.elapsed().as_secs_f64(),"records":records})
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "tactical_exposure PROTOCOL.json OUTPUT_DIRECTORY"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-tactical-exposure-v1");
    assert!(protocol.iterations > 0 && protocol.threads > 0 && protocol.repetitions > 0);
    assert!(protocol
        .arms
        .iter()
        .all(|a| a == "ordinary" || a == "specialist"));
    assert!(!protocol.interpretation.is_empty());
    assert_eq!(hash(Path::new(&protocol.catalog)), protocol.catalog_sha256);
    assert_eq!(hash(Path::new(&protocol.prior)), protocol.prior_sha256);
    let dex = conformance::load_dex();
    let catalog = load_meta_pool(Path::new(&protocol.catalog));
    let prior = Arc::new(load_meta_pool(Path::new(&protocol.prior)));
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let ids: BTreeSet<_> = protocol
        .families
        .iter()
        .map(|f| &f.team)
        .chain(protocol.opponents.iter())
        .collect();
    for id in ids {
        let team = catalog.teams.iter().find(|t| &t.id == id).unwrap();
        let result = validate_team(
            &dex,
            &learnsets,
            &serde_json::to_string(&team.sets).unwrap(),
        );
        assert_eq!(result["ok"], true, "{id}: {result}");
    }
    let mut jobs = Vec::new();
    let mut block = 0;
    for family in &protocol.families {
        for opponent in &protocol.opponents {
            for repetition in 0..protocol.repetitions {
                for arm in &protocol.arms {
                    jobs.push((family, opponent, repetition, block, arm));
                }
                block += 1;
            }
        }
    }
    let directory = Path::new(&args[2]);
    std::fs::create_dir(directory).unwrap();
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),"protocol_sha256":hash(Path::new(&args[1])),
        "source_sha256":hash(Path::new("crates/bot/examples/tactical_exposure.rs")),
        "binary_sha256":hash(&std::env::current_exe().unwrap()),"expected_games":jobs.len(),
        "revision":String::from_utf8(std::process::Command::new("git").args(["rev-parse","HEAD"]).output().unwrap().stdout).unwrap().trim()});
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
                let Some((family, opponent, repetition, block, arm)) = jobs.get(index) else {
                    break;
                };
                let record = game(
                    &dex,
                    &catalog,
                    &prior,
                    &protocol,
                    family,
                    opponent,
                    *repetition,
                    *block,
                    arm,
                );
                std::fs::write(
                    directory.join(format!("game-{block:03}-{arm}.json")),
                    serde_json::to_vec(&record).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "{}/{} {} {} {} score={} turns={} seconds={:.1}",
                    index + 1,
                    jobs.len(),
                    family.id,
                    opponent,
                    arm,
                    record["source_score"],
                    record["turns"],
                    record["seconds"].as_f64().unwrap()
                );
            });
        }
    });
}
