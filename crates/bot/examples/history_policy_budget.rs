use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::{json, Value};
use std::{io::BufRead, path::Path, time::Instant};

fn snapshot(s: &BlindSearch, dex: &Dex) -> Value {
    json!({"best":s.best().unwrap().to_input(dex),"visits":s.visits(),"means":s.means(),"rng":s.search_rng_state(),"nodes":s.node_count()})
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 5, "corpus baseline output repeats");
    let repeats: usize = args[4].parse().unwrap();
    assert!(repeats > 0);
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let baseline: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let mut cases = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let prior = baseline.iter().find(|r| r["game"] == game["game"]).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            if decision as u64 == prior["decision"].as_u64().unwrap() {
                let side = prior["side"].as_u64().unwrap() as usize;
                assert_eq!(b.state_key128().to_string(), prior["key"].as_str().unwrap());
                cases.push((prior.clone(), b.clone(), observers[side].clone()));
                break;
            }
            let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| acts[s][i as usize])),
            )
            .unwrap();
            for obs in &mut observers {
                obs.observe(&b, &dex);
            }
        }
    }
    assert_eq!(cases.len(), 24);
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut rows = Vec::new();
    for (prior, b, obs) in cases {
        let side = prior["side"].as_u64().unwrap() as usize;
        let seed = prior["seed"].as_u64().unwrap();
        let belief = Belief::new(&dex, &pool, &obs);
        let mut measurements = Vec::new();
        let mut snapshots: [Option<Value>; 4] = [None, None, None, None];
        for repeat in 0..repeats {
            for offset in 0..4 {
                let mode = (offset + repeat) % 4;
                let begin = Instant::now();
                let mut search = BlindSearch::new(&b, &dex, cfg.clone(), side, seed);
                match mode {
                    1 => search.enable_history_continuation(false),
                    2 => search.enable_first_continuation(true),
                    3 => search.enable_history_continuation(true),
                    _ => {}
                }
                search.step(&dex, &belief, &obs, 27000);
                let elapsed_ms = begin.elapsed().as_secs_f64() * 1000.;
                let snap = snapshot(&search, &dex);
                if mode < 2 {
                    assert_eq!(
                        snap, prior["snapshot"],
                        "baseline/control differ at game {}",
                        prior["game"]
                    );
                }
                if mode == 2 {
                    assert_eq!(snap, prior["shared_snapshot"]);
                }
                if let Some(expected) = &snapshots[mode] {
                    assert_eq!(&snap, expected);
                } else {
                    snapshots[mode] = Some(snap.clone());
                }
                let mut depth_counts = std::collections::BTreeMap::<usize, u32>::new();
                let (groups, selections, rewards, histories, log_bytes) = if let Some(p) =
                    search.history_continuation()
                {
                    for g in p.groups() {
                        *depth_counts
                            .entry(p.histories()[g.history - 1].depth)
                            .or_default() += g.visits.iter().sum::<u32>();
                    }
                    (
                        p.groups().len(),
                        p.groups().iter().flat_map(|g| g.visits.iter()).sum::<u32>(),
                        p.groups()
                            .iter()
                            .flat_map(|g| g.rewards.iter())
                            .sum::<f64>(),
                        p.histories().len(),
                        p.histories()
                            .iter()
                            .flat_map(|h| h.log.iter())
                            .map(|line| line.len())
                            .sum::<usize>(),
                    )
                } else if let Some(p) = search.first_continuation() {
                    let selections = p.groups().iter().flat_map(|g| g.visits.iter()).sum::<u32>();
                    depth_counts.insert(1, selections);
                    (
                        p.groups().len(),
                        selections,
                        p.groups()
                            .iter()
                            .flat_map(|g| g.rewards.iter())
                            .sum::<f64>(),
                        0,
                        0,
                    )
                } else {
                    (0, 0, 0., 0, 0)
                };
                assert!(rewards >= 0. && rewards <= f64::from(selections));
                measurements.push(json!({"mode":(["baseline","record_only","first_shared","history_shared"][mode]),"repeat":repeat,
                    "elapsed_ms":elapsed_ms,"snapshot":snap,"groups":groups,"selections":selections,"rewards":rewards,"depth_counts":depth_counts,"histories":histories,"log_bytes":log_bytes}));
                eprintln!("game={} repeat={repeat} mode={mode} ms={elapsed_ms:.1} groups={groups} selections={selections}",prior["game"]);
            }
        }
        rows.push(json!({"game":prior["game"],"decision":prior["decision"],"key":prior["key"],"side":side,"seed":seed,"measurements":measurements}));
        std::fs::write(
            &args[3],
            serde_json::to_vec(&json!({"iterations":27000,"repeats":repeats,"rows":rows})).unwrap(),
        )
        .unwrap();
    }
}
