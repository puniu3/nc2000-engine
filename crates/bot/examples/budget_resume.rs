//! Checkpointed execution of budget_allocation, preserving its exact policy.
//! Replay saved legal singletons through BlindAgent to restore observations
//! without consuming search RNG; initialize each agent at its saved RNG state.
use nc2000_bot::{preview::load_meta_pool, Agent, BlindAgent, RmConfig, SplitMix64};
use nc2000_engine::{
    battle::{Outcome, SearchChoice},
    dex::Dex,
    state::Battle,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

#[derive(Clone, Deserialize)]
struct Profile {
    id: String,
    preview: u32,
    battle: u32,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    purpose: String,
    seed: u64,
    pairs: usize,
    threads: usize,
    catalog: String,
    prior: String,
    profiles: Vec<Profile>,
    opponent: Profile,
    draw: String,
    ponder: bool,
    max_turns: u16,
}
struct Scheduled {
    teams: [usize; 2],
    battle_seed: String,
    seeds: [u64; 2],
}
fn hash(path: impl AsRef<Path>) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}
fn write_json(path: &Path, value: &Value) {
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap().to_str().unwrap()
    ));
    std::fs::write(&tmp, serde_json::to_vec(value).unwrap()).unwrap();
    std::fs::rename(tmp, path).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn run(
    dex: &Dex,
    sets: [&[nc2000_engine::battle::PokemonSet]; 2],
    seed: &str,
    agents: &mut [BlindAgent; 2],
    profiles: [&Profile; 2],
    limit: u16,
    checkpoint: Option<&Path>,
    context: &Value,
    resume: Option<&Value>,
    stop_after_requests: Option<usize>,
) -> Option<Value> {
    if let Some(old) = resume {
        assert_eq!(&old["context"], context);
    }
    let replay = resume
        .map(|r| r["decisions"].as_array().unwrap().clone())
        .unwrap_or_default();
    let mut b = Battle::from_fixture(dex, seed, sets[0], sets[1]).unwrap();
    b.set_log_enabled(true);
    let mut records = vec![];
    let mut requests = 0;
    while b.outcome().is_none() && b.turn <= limit {
        let legal = [0, 1].map(|s| b.legal_choices(dex, s));
        let mut picks = [None, None];
        let key = b.state_key128().to_string();
        for s in 0..2 {
            if legal[s].is_empty() {
                continue;
            }
            let phase = if matches!(legal[s][0], SearchChoice::Team(_)) {
                "preview"
            } else if legal[s].len() == 1 {
                "forced"
            } else {
                "battle"
            };
            let budget = if phase == "preview" {
                profiles[s].preview
            } else {
                profiles[s].battle
            };
            agents[s].set_iterations(budget);
            let before = agents[s].search_rng_state();
            let choice;
            if let Some(saved) = replay.get(records.len()) {
                assert_eq!(saved["side"], s);
                assert_eq!(saved["turn"], b.turn);
                assert_eq!(saved["key"], key);
                assert_eq!(saved["phase"], phase);
                assert_eq!(saved["choices"], legal[s].len());
                assert_eq!(
                    saved["iterations"],
                    if phase == "forced" { 0 } else { budget }
                );
                choice = *legal[s]
                    .iter()
                    .find(|c| c.to_input(dex) == saved["choice"].as_str().unwrap())
                    .expect("saved choice must remain legal");
                // choose() always observes/syncs before its singleton shortcut.
                // Tables, adaptive mixtures and pruning are absent in this study.
                assert_eq!(agents[s].choose(&b, dex, s, &[choice]), choice);
                assert_eq!(before, agents[s].search_rng_state());
                records.push(saved.clone());
            } else {
                let start = Instant::now();
                choice = agents[s].choose(&b, dex, s, &legal[s]);
                let ns = start.elapsed().as_nanos() as u64;
                assert!(legal[s].contains(&choice));
                if phase == "forced" {
                    assert_eq!(before, agents[s].search_rng_state());
                }
                records.push(json!({"side":s,"turn":b.turn,"key":key,"phase":phase,
                    "choices":legal[s].len(),"iterations":if phase=="forced"{0}else{budget},
                    "ns":ns,"choice":choice.to_input(dex)}));
            }
            picks[s] = Some(choice);
        }
        assert!(picks.iter().any(Option::is_some));
        b.apply_choices(dex, picks).unwrap();
        requests += 1;
        if records.len() == replay.len() && !replay.is_empty() {
            assert_eq!(resume.unwrap()["next_key"], b.state_key128().to_string());
        }
        // During replay the agents already carry the RNG at the END of the
        // saved prefix. Never write an incomplete replay with those RNGs.
        if records.len() >= replay.len() {
            if let Some(path) = checkpoint {
                write_json(
                    path,
                    &json!({"context":context,"decisions":records,
                    "rng_states":[agents[0].search_rng_state(),agents[1].search_rng_state()],
                    "next_key":b.state_key128().to_string(),"turn":b.turn}),
                );
            }
            if stop_after_requests == Some(requests) {
                return None;
            }
        }
    }
    assert!(
        records.len() >= replay.len(),
        "checkpoint extends past terminal state"
    );
    let outcome = b.outcome();
    Some(
        json!({"p1_score":outcome.map(|o|match o{Outcome::P1Win=>1.,Outcome::P2Win=>0.,Outcome::Tie=>0.5}),
        "outcome":format!("{outcome:?}"),"cap":outcome.is_none(),"turns":b.turn,"decisions":records}),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=5).contains(&args.len()),
        "budget_resume PROTOCOL NEW_OUTPUT_DIRECTORY [JOB_LIST] [RESUME_DIRECTORY]"
    );
    let resume_dir = args.get(4).map(std::path::PathBuf::from);
    let selected: Option<Vec<String>> = args
        .get(3)
        .map(|path| serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap());
    let p: Protocol = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let threads: usize = std::env::var("NC2000_BUDGET_THREADS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(p.threads);
    assert!(threads > 0);
    assert_eq!(p.schema, "nc2000-budget-allocation-v1");
    assert_eq!(p.draw, "drawWeight");
    assert!(!p.ponder);
    assert!(p.pairs > 0 && p.threads > 0 && !p.purpose.is_empty());
    let mut ids = std::collections::HashSet::new();
    for q in &p.profiles {
        assert!(ids.insert(&q.id));
    }
    for q in p.profiles.iter().chain([&p.opponent]) {
        assert!(q.preview > 0 && q.battle > 0);
        assert!(!q.id.is_empty() && q.id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'));
    }
    let out = Path::new(&args[2]);
    std::fs::create_dir(out).unwrap();
    let dex = conformance::load_dex();
    let prior = Arc::new(load_meta_pool(Path::new(&p.prior)));
    let catalog = load_meta_pool(Path::new(&p.catalog));
    let raw: Value = serde_json::from_slice(&std::fs::read(&p.catalog).unwrap()).unwrap();
    let weights: Vec<f64> = raw["teams"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&catalog.teams)
        .map(|(r, t)| {
            assert_eq!(r["id"], t.id);
            r["drawWeight"].as_f64().unwrap()
        })
        .collect();
    assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.));
    let total: f64 = weights.iter().sum();
    assert!(total > 0.);
    let mut rng = SplitMix64::new(p.seed);
    let mut schedules = vec![];
    let mut seen = std::collections::HashSet::new();
    for _ in 0..p.pairs {
        let teams = [0, 1].map(|_| {
            let mut draw = rng.next_f64() * total;
            weights
                .iter()
                .position(|w| {
                    draw -= w;
                    draw < 0.
                })
                .unwrap()
        });
        let battle_seed = rng.battle_seed();
        let seeds = [rng.next(), rng.next()];
        for s in seeds {
            assert!(seen.insert(s));
        }
        schedules.push(Scheduled {
            teams,
            battle_seed,
            seeds,
        });
    }
    let frozen:Vec<_>=schedules.iter().enumerate().map(|(pair,g)|json!({"pair":pair,"teams":g.teams.map(|i|catalog.teams[i].id.clone()),"battle_seed":g.battle_seed,"seat_seeds":g.seeds})).collect();
    let metadata = json!({"protocol":serde_json::from_slice::<Value>(&std::fs::read(&args[1]).unwrap()).unwrap(),
        "protocol_sha256":hash(&args[1]),"binary_sha256":hash(std::env::current_exe().unwrap()),
        "source_sha256":hash("crates/bot/examples/budget_resume.rs"),"reference_source_sha256":hash("crates/bot/examples/budget_allocation.rs"),"blind_sha256":hash("crates/bot/src/blind.rs"),
        "dex_sha256":hash("data/gen2stadium2.json"),"catalog_sha256":hash(&p.catalog),"prior_sha256":hash(&p.prior),
        "schedules":frozen,"execution_threads":threads,"selected_jobs":selected,
        "seed_design":"Distinct physical-seat seeds within each pair; same seeds and fixed physical-seat teams across side swaps and profile comparisons. No first-pair collision."});
    let checkpoint_identity = json!({"abi":"budget-singleton-replay-v1",
        "build":nc2000_bot::compiled_preview::build_fingerprint(),
        "protocol":metadata["protocol_sha256"],"blind":metadata["blind_sha256"],
        "dex":metadata["dex_sha256"],"catalog":metadata["catalog_sha256"],"prior":metadata["prior_sha256"],
        "reference_source":metadata["reference_source_sha256"]});
    std::fs::write(
        out.join("manifest.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    // Interleave profiles and side assignments so machine load does not align with a profile.
    let mut jobs = vec![];
    for pair in 0..p.pairs {
        for profile in 0..p.profiles.len() {
            for a_side in 0..2 {
                jobs.push((pair, profile, a_side));
            }
        }
    }
    let mut order = SplitMix64::new(p.seed ^ 0x531c37f8a9d2e461);
    for i in (1..jobs.len()).rev() {
        let j = order.below(i + 1);
        jobs.swap(i, j);
    }
    if let Some(selected) = selected {
        let allowed: std::collections::HashSet<_> = selected.iter().collect();
        assert_eq!(allowed.len(), selected.len());
        jobs.retain(|&(pair, profile, side)| {
            allowed.contains(&format!(
                "{}-p{pair:03}-s{side}.json",
                p.profiles[profile].id
            ))
        });
        assert_eq!(jobs.len(), selected.len(), "unknown shard job");
    }
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(||loop{
        let index=cursor.fetch_add(1,Ordering::Relaxed);let Some(&(pair,profile,a_side))=jobs.get(index)else{break;};let g=&schedules[pair];let q=&p.profiles[profile];
        let configs=[0,1].map(|s|if s==a_side{q}else{&p.opponent});
        let name=format!("{}-p{pair:03}-s{a_side}",q.id);
        let context=json!({"inputs":checkpoint_identity,"job":name});
        let checkpoint=out.join(format!("{name}.partial"));
        let resume:Option<Value>=resume_dir.as_ref().map(|dir|dir.join(format!("{name}.partial"))).filter(|p|p.exists()).map(|p|serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap());
        if let Some(old)=&resume {assert_eq!(old["context"],context);}
        let seeds=[0,1].map(|s|resume.as_ref().map(|r|r["rng_states"][s].as_u64().unwrap()).unwrap_or(g.seeds[s]));
        let mut agents=[0,1].map(|s|BlindAgent::new(RmConfig{iterations:configs[s].battle,c:0.4,..Default::default()},prior.clone(),None,seeds[s]));
        let result=run(&dex,g.teams.map(|i|catalog.teams[i].sets.as_slice()),&g.battle_seed,&mut agents,configs,p.max_turns,Some(&checkpoint),&context,resume.as_ref(),None).unwrap();
        assert!(!result["cap"].as_bool().unwrap(),"harness cap");let score=result["p1_score"].as_f64().unwrap();
        let record=json!({"pair":pair,"profile":q.id,"a_side":a_side,"a_score":if a_side==0{score}else{1.-score},"result":result});
        let path=out.join(format!("{name}.json"));assert!(!path.exists());write_json(&path,&record);std::fs::remove_file(checkpoint).unwrap();
        let n=done.fetch_add(1,Ordering::Relaxed)+1;if n%8==0||n==jobs.len(){eprintln!("{n}/{} games, {:.1}s",jobs.len(),start.elapsed().as_secs_f64());}
    });
        }
    });
}

#[test]
fn unchanged_budget_preserves_fixed_agent_trajectory() {
    let dex = conformance::load_dex();
    let prior = Arc::new(load_meta_pool(Path::new(
        "../../data/belief-pool-v3/belief-pool.json",
    )));
    let teams = [&prior.teams[0].sets, &prior.teams[1].sets];
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", teams[0], teams[1]).unwrap();
    b.set_log_enabled(true);
    let new_agent = |s: u64| {
        BlindAgent::new(
            RmConfig {
                iterations: 128,
                c: 0.4,
                ..Default::default()
            },
            prior.clone(),
            None,
            731 + s,
        )
    };
    let mut fixed = [new_agent(0), new_agent(1)];
    let mut allocated = [new_agent(0), new_agent(1)];
    let mut calls = 0;
    while b.outcome().is_none() && b.turn < 16 {
        let mut picks = [None, None];
        for s in 0..2 {
            let legal = b.legal_choices(&dex, s);
            if legal.is_empty() {
                continue;
            }
            allocated[s].set_iterations(128);
            let a = fixed[s].choose(&b, &dex, s, &legal);
            let z = allocated[s].choose(&b, &dex, s, &legal);
            assert_eq!(a, z);
            assert_eq!(fixed[s].search_rng_state(), allocated[s].search_rng_state());
            picks[s] = Some(a);
            calls += 1;
        }
        b.apply_choices(&dex, picks).unwrap();
    }
    assert!(calls >= 10);
}

#[test]
fn checkpoint_replay_preserves_observations_rng_and_full_trajectory() {
    let dex = conformance::load_dex();
    let prior = Arc::new(load_meta_pool(Path::new(
        "../../data/belief-pool-v3/belief-pool.json",
    )));
    let teams = [
        prior.teams[0].sets.as_slice(),
        prior.teams[1].sets.as_slice(),
    ];
    let high = Profile {
        id: "high".into(),
        preview: 384,
        battle: 384,
    };
    let low = Profile {
        id: "low".into(),
        preview: 384,
        battle: 128,
    };
    let profiles = [&high, &low];
    for seed in [731, 1097, 4921] {
        let make_agents = |states: [u64; 2]| {
            [0, 1].map(|s| {
                BlindAgent::new(
                    RmConfig {
                        iterations: profiles[s].battle,
                        c: 0.4,
                        ..Default::default()
                    },
                    prior.clone(),
                    None,
                    states[s],
                )
            })
        };
        let initial = [seed, seed + 1];
        let context = json!({"test_seed":seed});
        let mut uninterrupted = make_agents(initial);
        let mut expected = run(
            &dex,
            teams,
            "1,2,3,4",
            &mut uninterrupted,
            profiles,
            1100,
            None,
            &context,
            None,
            None,
        )
        .unwrap();
        assert!(!expected["cap"].as_bool().unwrap());
        let final_rng = [
            uninterrupted[0].search_rng_state(),
            uninterrupted[1].search_rng_state(),
        ];
        let dir = std::env::temp_dir().join(format!("nc2000-resume-{}-{seed}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let path = dir.join("checkpoint.partial");
        let boundaries: std::collections::BTreeSet<_> =
            [1, 3, expected["decisions"].as_array().unwrap().len() / 4]
                .into_iter()
                .collect();
        let mut saved: Option<Value> = None;
        for boundary in boundaries {
            let states = [0, 1].map(|s| {
                saved
                    .as_ref()
                    .map(|v| v["rng_states"][s].as_u64().unwrap())
                    .unwrap_or(initial[s])
            });
            let mut agents = make_agents(states);
            assert!(run(
                &dex,
                teams,
                "1,2,3,4",
                &mut agents,
                profiles,
                1100,
                Some(&path),
                &context,
                saved.as_ref(),
                Some(boundary)
            )
            .is_none());
            saved = Some(serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap());
        }
        let states = [0, 1].map(|s| saved.as_ref().unwrap()["rng_states"][s].as_u64().unwrap());
        let mut agents = make_agents(states);
        let mut resumed = run(
            &dex,
            teams,
            "1,2,3,4",
            &mut agents,
            profiles,
            1100,
            Some(&path),
            &context,
            saved.as_ref(),
            None,
        )
        .unwrap();
        assert_eq!(
            [agents[0].search_rng_state(), agents[1].search_rng_state()],
            final_rng
        );
        for result in [&mut expected, &mut resumed] {
            for decision in result["decisions"].as_array_mut().unwrap() {
                decision.as_object_mut().unwrap().remove("ns");
            }
        }
        assert_eq!(resumed, expected);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
