use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::{json, Value};
use std::{io::BufRead, time::Instant};

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 4, "corpus prior-roots output");
    let prior: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut cases = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
            for old in prior["rows"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|r| r["game"] == game["game"] && r["decision"] == decision)
            {
                let side = old["side"].as_u64().unwrap() as usize;
                assert_eq!(old["key"], b.state_key128().to_string());
                cases.push((old.clone(), b.clone(), observers[side].clone()));
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize])),
            )
            .unwrap();
            for observer in &mut observers {
                observer.observe(&b, &dex);
            }
        }
    }
    assert_eq!(cases.len(), 24);
    cases.sort_by_key(|(old, _, _)| {
        (
            old["game"].as_u64().unwrap(),
            old["decision"].as_u64().unwrap(),
            old["side"].as_u64().unwrap(),
        )
    });
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let (_, warm, obs) = &cases[0];
    let belief = Belief::new(&dex, &pool, obs);
    BlindSearch::new(
        warm,
        &dex,
        cfg.clone(),
        cases[0].0["side"].as_u64().unwrap() as usize,
        0,
    )
    .step(&dex, &belief, obs, 3000);
    let mut rows = Vec::new();
    for (index, (old, battle, observer)) in cases.into_iter().enumerate() {
        let side = old["side"].as_u64().unwrap() as usize;
        let seed = old["seed"].as_u64().unwrap();
        let belief = Belief::new(&dex, &pool, &observer);
        let mut runs = Vec::new();
        for repeat in 0..3 {
            for enabled in if (index + repeat) % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let mut search = BlindSearch::new(
                    &battle,
                    &dex,
                    RmConfig {
                        canonical_claw_key: enabled,
                        ..cfg.clone()
                    },
                    side,
                    seed,
                );
                let start = Instant::now();
                let mut checkpoints = Vec::new();
                for budget in [3000, 27000] {
                    search.step(&dex, &belief, &observer, budget - search.iterations());
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                    let snapshot = json!({"iterations":budget,"best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means(),"nodes":search.node_count()});
                    if !enabled {
                        let previous = old["arms"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|a| a["enabled"] == false)
                            .unwrap()["checkpoints"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .find(|c| c["iterations"] == budget)
                            .unwrap();
                        for field in ["best", "visits", "means", "nodes"] {
                            assert_eq!(
                                snapshot[field], previous[field],
                                "baseline changed: {field}"
                            );
                        }
                    }
                    checkpoints.push(json!({"snapshot":snapshot,"elapsed_ms":elapsed_ms}));
                }
                runs.push(json!({"repeat":repeat,"enabled":enabled,"checkpoints":checkpoints}));
            }
        }
        rows.push(json!({"game":old["game"],"decision":old["decision"],"side":side,"key":old["key"],"seed":seed,"runs":runs}));
        eprintln!("claw key root {}/24 complete", index + 1);
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec(&json!({"iterations":27000,"repeats":3,"rows":rows})).unwrap(),
    )
    .unwrap();
}
