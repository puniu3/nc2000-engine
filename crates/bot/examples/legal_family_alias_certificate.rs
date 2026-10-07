use nc2000_bot::{preview::load_meta_pool, rng};
use nc2000_engine::prng::BattleRng;
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

mod aliased_witness;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "corpus case output mode");
    let (merge, fast) = match args[4].as_str() {
        "control" => (false, false),
        "merged" => (true, false),
        "fast" => (true, true),
        "rivals" => (true, false),
        "cap-one" => (false, true),
        _ => panic!("invalid mode"),
    };
    let rivals_only = args[4] == "rivals";
    let capped = args[4] == "cap-one";
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
    let side = case["side"].as_u64().unwrap() as usize;
    let index = models
        .names
        .iter()
        .position(|name| name == "doubleedge")
        .unwrap();
    let upper_start = Instant::now();
    let upper_config = json!({"game":case["game"],"decision":case["decision"],"target_side":1-side,"policy_side":1-side,
        "first_choice":"move sleeptalk","priority":["move hiddenpowerice"],"rest_at_percent":null,
        "opening":["move sleeptalk","move sleeptalk","move hiddenpowerice"],
        "horizon":3,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
    let upper = if fast || rivals_only {
        let (ledger, audits) = aliased_witness::solve(
            &dex,
            models.worlds[index].clone(),
            upper_config,
            if rivals_only { 0.9 } else { 1. },
            false,
            fast,
            if rivals_only {
                Some("move rockslide")
            } else {
                None
            },
        );
        assert!(audits.is_empty());
        ledger
    } else {
        aliased_witness::upper(&dex, models.worlds[index].clone(), upper_config)
    };
    let upper_seconds = upper_start.elapsed().as_secs_f64();
    let upper_by_action: serde_json::Map<String, Value> = upper["nodes"]
        [upper["root"].as_u64().unwrap() as usize]["branches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| {
            (
                b["choices"][side].as_str().unwrap().to_string(),
                json!(1. - b["bounds"][0].as_f64().unwrap()),
            )
        })
        .collect();
    let threshold = upper_by_action
        .iter()
        .filter(|(a, _)| a.as_str() != "move rockslide")
        .map(|(_, v)| v.as_f64().unwrap())
        .fold(0., f64::max)
        + 1e-6;
    assert!((0. ..1.).contains(&threshold));
    let mut rows = Vec::new();
    let mut tasks = Vec::new();
    for (name, battle) in models.names.iter().zip(&models.worlds) {
        let config = json!({"game":case["game"],"decision":case["decision"],"target_side":case["side"],"policy_side":case["side"],
            "first_choice":"move rockslide","priority":["move rockslide"],"rest_at_percent":null,
            "horizon":2,"work_budget":if capped {1} else {200000},"cell_cap":if capped {1} else {10000},"guarded_keys":true,"support_certificate":true});
        let begin = Instant::now();
        let (ledger, aliases) =
            aliased_witness::solve(&dex, battle.clone(), config, threshold, merge, fast, None);
        tasks.extend(aliases.into_iter().map(|task| (name.clone(), task)));
        let seconds = begin.elapsed().as_secs_f64();
        eprintln!(
            "{name} seconds={seconds} runs={} bounds={}",
            ledger["runs"], ledger["bounds"]
        );
        rows.push(json!({"move":name,"seconds":seconds,"ledger":ledger}));
    }
    let seconds = start.elapsed().as_secs_f64();
    let audit_start = Instant::now();
    let audits: Vec<_> = tasks
        .into_iter()
        .map(|(name, task)| audit(&dex, name, task))
        .collect();
    let audit_seconds = audit_start.elapsed().as_secs_f64();
    std::fs::write(&args[3],serde_json::to_vec(&json!({"case":case,"threshold":threshold,"merged":merge,"fast":fast,"rivals_only":rivals_only,"capped":capped,
        "generation_seconds":generation_seconds,"upper_world":"doubleedge","upper_by_action":upper_by_action,
        "upper_seconds":upper_seconds,"upper":upper,"seconds":seconds,"rows":rows,"audit_seconds":audit_seconds,"audits":audits})).unwrap()).unwrap();
}

fn audit(dex: &nc2000_engine::dex::Dex, world: String, task: aliased_witness::AliasTask) -> Value {
    let mut pending = vec![Vec::<usize>::new()];
    let mut runs = 0;
    let mut leaves = 0;
    let mut mass = 0.;
    while let Some(script) = pending.pop() {
        let results: Vec<_> = [task.choices, task.representative]
            .into_iter()
            .map(|choices| {
                let mut battle = task.battle.clone();
                battle.set_log_enabled(false);
                battle.set_log_enabled(true);
                battle.prng = BattleRng::enumerating(script.clone());
                battle.apply_choices(dex, choices).unwrap();
                let trace = battle.prng.oracle.take().unwrap().trace;
                (battle, trace)
            })
            .collect();
        assert_eq!(results[0].0.state_key128(), results[1].0.state_key128());
        assert_eq!(results[0].0.log, results[1].0.log);
        assert_eq!(format!("{:?}", results[0].1), format!("{:?}", results[1].1));
        runs += 2;
        assert!(runs <= 100000);
        let trace = &results[0].1;
        if script.len() < trace.len() {
            for (i, &count) in trace[script.len()].counts.iter().enumerate().rev() {
                if count > 0 {
                    let mut next = script.clone();
                    next.push(i);
                    pending.push(next);
                }
            }
        } else {
            assert_eq!(script.len(), trace.len());
            leaves += 1;
            mass += trace.iter().map(|d| d.prob()).product::<f64>();
        }
    }
    assert!((mass - 1.).abs() < 1e-10);
    json!({"world":world,"key":task.key,"depth":task.depth,
        "choices":task.choices.map(|a|a.map(|a|a.to_input(dex))),
        "representative":task.representative.map(|a|a.map(|a|a.to_input(dex))),
        "runs":runs,"leaves":leaves,"mass":mass})
}
