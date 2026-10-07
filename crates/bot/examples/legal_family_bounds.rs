use nc2000_bot::{preview::load_meta_pool, rng};
use nc2000_engine::validate::Learnsets;
use serde_json::{json, Value};
use std::time::Instant;

mod legal_move_models;

#[allow(dead_code)]
mod replay {
    include!("hidden_family_search.rs");
    pub fn model(dex: &Dex, game: &Value, row: &Value) -> (Battle, Observer) {
        replay(dex, game, row, false)
    }
}

#[allow(dead_code)]
mod certificate {
    include!("hidden_terminal.rs");
    pub fn solve(dex: &Dex, b: Battle, config: Value) -> Value {
        finite::solve(dex, b, config)
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus case output");
    let case: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let game: Value = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|g| g["game"] == case["game"])
        .unwrap();
    let dex = conformance::load_dex();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let sources: Value =
        serde_json::from_slice(&std::fs::read("data/legal-move-sources-gen2.json").unwrap())
            .unwrap();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let (battle, observer) = replay::model(&dex, &game, &case);
    let start = Instant::now();
    let models =
        legal_move_models::generate(&dex, &learnsets, &sources, &pool, &battle, &observer, 0)
            .unwrap();
    assert!(models.excluded.is_empty());
    let generation_seconds = start.elapsed().as_secs_f64();
    let mut rows = Vec::new();
    for (name, battle) in models.names.iter().zip(&models.worlds) {
        let config = json!({"game":case["game"],"decision":case["decision"],"target_side":case["side"],"policy_side":case["side"],
            "first_choice":"move rockslide","priority":["move rockslide"],"rest_at_percent":null,
            "horizon":2,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
        let begin = Instant::now();
        let ledger = certificate::solve(&dex, battle.clone(), config);
        let seconds = begin.elapsed().as_secs_f64();
        eprintln!(
            "{name} seconds={seconds} runs={} bounds={}",
            ledger["runs"], ledger["bounds"]
        );
        rows.push(json!({"move":name,"seconds":seconds,"ledger":ledger}));
    }
    let seconds = start.elapsed().as_secs_f64();
    std::fs::write(&args[3],serde_json::to_vec(&json!({"case":case,"generation_seconds":generation_seconds,"seconds":seconds,"rows":rows})).unwrap()).unwrap();
}
