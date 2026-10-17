//! Isolate preview/in-battle iteration budgets. Same BlindAgent and information.
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
fn run(
    dex: &Dex,
    sets: [&[nc2000_engine::battle::PokemonSet]; 2],
    seed: &str,
    agents: &mut [BlindAgent; 2],
    profiles: [&Profile; 2],
    limit: u16,
) -> Value {
    let mut b = Battle::from_fixture(dex, seed, sets[0], sets[1]).unwrap();
    b.set_log_enabled(true);
    let mut records = vec![];
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
            let start = Instant::now();
            let choice = agents[s].choose(&b, dex, s, &legal[s]);
            let ns = start.elapsed().as_nanos() as u64;
            assert!(legal[s].contains(&choice));
            if phase == "forced" {
                assert_eq!(before, agents[s].search_rng_state());
            }
            records.push(json!({"side":s,"turn":b.turn,"key":key,"phase":phase,
                "choices":legal[s].len(),"iterations":if phase=="forced"{0}else{budget},
                "ns":ns,"choice":choice.to_input(dex)}));
            picks[s] = Some(choice);
        }
        assert!(picks.iter().any(Option::is_some));
        b.apply_choices(dex, picks).unwrap();
    }
    let outcome = b.outcome();
    json!({"p1_score":outcome.map(|o|match o{Outcome::P1Win=>1.,Outcome::P2Win=>0.,Outcome::Tie=>0.5}),
        "outcome":format!("{outcome:?}"),"cap":outcome.is_none(),"turns":b.turn,"decisions":records})
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        (3..=4).contains(&args.len()),
        "budget_allocation PROTOCOL NEW_OUTPUT_DIRECTORY [JOB_LIST]"
    );
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
        "source_sha256":hash("crates/bot/examples/budget_allocation.rs"),"blind_sha256":hash("crates/bot/src/blind.rs"),
        "dex_sha256":hash("data/gen2stadium2.json"),"catalog_sha256":hash(&p.catalog),"prior_sha256":hash(&p.prior),
        "schedules":frozen,"execution_threads":threads,"selected_jobs":selected,
        "seed_design":"Distinct physical-seat seeds within each pair; same seeds and fixed physical-seat teams across side swaps and profile comparisons. No first-pair collision."});
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
        let mut agents=[0,1].map(|s|BlindAgent::new(RmConfig{iterations:configs[s].battle,c:0.4,..Default::default()},prior.clone(),None,g.seeds[s]));
        let result=run(&dex,g.teams.map(|i|catalog.teams[i].sets.as_slice()),&g.battle_seed,&mut agents,configs,p.max_turns);
        assert!(!result["cap"].as_bool().unwrap(),"harness cap");let score=result["p1_score"].as_f64().unwrap();
        let record=json!({"pair":pair,"profile":q.id,"a_side":a_side,"a_score":if a_side==0{score}else{1.-score},"result":result});
        let path=out.join(format!("{}-p{pair:03}-s{a_side}.json",q.id));let file=std::fs::File::create_new(path).unwrap();serde_json::to_writer(file,&record).unwrap();
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
