//! Native evaluation adapter vs the actual WASM bridge lifecycle.
#[path = "../../bot/examples/support/execution_agent.rs"]
mod execution_agent;
use execution_agent::ExecutionAgent;
use nc2000_bot::{Agent, decision::PolicyMode, preview::MetaPool};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use nc2000_wasm::{WasmDex, WasmBattle, WasmBlindSearcher};
use serde_json::{json, Value};
use std::{fs, sync::Arc};
fn main() {
    let dex = conformance::load_dex(); let wdex = WasmDex::new().unwrap();
    let cat: Value = serde_json::from_slice(&fs::read("data/team-pool-v2/team-pool.json").unwrap()).unwrap();
    let pool_json = fs::read_to_string("data/belief-pool-v3/belief-pool.json").unwrap();
    let pool = Arc::new(serde_json::from_str::<MetaPool>(&pool_json).unwrap());
    let mut decisions = 0;
    let mut records = vec![];
    for mode in ["legacy", "visit-sample", "estimated-mix"] {
        for seed in [7, 813001] {
            let teams: [Vec<PokemonSet>;2] = [0, 7].map(|i|serde_json::from_value(cat["teams"][i]["sets"].clone()).unwrap());
            let strings = teams.each_ref().map(|t|serde_json::to_string(t).unwrap());
            let mut native = Battle::from_fixture(&dex,"1,2,3,4",&teams[0],&teams[1]).unwrap();
            native.set_log_enabled(true);
            let mut bridge = WasmBattle::new(&wdex,&strings[0],&strings[1],"1,2,3,4").unwrap();
            let mut a = ExecutionAgent::new(pool.clone(),PolicyMode::parse(mode).unwrap(),512,seed);
            let mut b = WasmBlindSearcher::new(&bridge,1,&pool_json,seed,Some(0.4),None).unwrap();
            for _ in 0..4 {
                let choices = native.legal_choices(&dex,1);
                if choices.is_empty() { break; }
                let action = a.choose(&native,&dex,1,&choices);
                b.observe(&bridge).unwrap(); b.step(512).unwrap();
                let decision: Value = serde_json::from_str(&b.commit_decision(mode).unwrap()).unwrap();
                assert_eq!(decision["input"],action.to_input(&dex));
                let expected = a.last_policy.as_ref().unwrap().map(|a|a.to_input(&dex));
                let expected = serde_json::to_value(expected).unwrap();
                assert_eq!(decision["policy"]["source"],expected["source"]);
                let actual_rows = decision["policy"]["actions"].as_array().unwrap();
                let expected_rows = expected["actions"].as_array().unwrap();
                assert_eq!(actual_rows.len(),expected_rows.len());
                for (x,y) in actual_rows.iter().zip(expected_rows) {
                    assert_eq!(x["input"],y["input"]);
                    assert!((x["probability"].as_f64().unwrap()-y["probability"].as_f64().unwrap()).abs()<1e-12);
                }
                let diagnostics: Value = serde_json::from_str(&b.root_policy()).unwrap();
                for (act,n,mean) in &a.last_visits {
                    let row = diagnostics["actions"].as_array().unwrap().iter().find(|r|r["input"]==act.to_input(&dex)).unwrap();
                    assert_eq!(row["visits"],*n);
                    assert!((row["mean"].as_f64().unwrap()-mean).abs()<1e-12);
                }
                let own = native.legal_choices(&dex,0).first().copied();
                records.push(json!({"mode":mode,"seed":seed,"input":action.to_input(&dex),"policy":expected,
                    "own":own.map(|c|c.to_input(&dex)),"diagnostics":a.last_visits.iter().map(|(a,n,m)|
                        json!({"input":a.to_input(&dex),"visits":n,"mean":m})).collect::<Vec<_>>()}));
                native.apply_choices(&dex,[own,Some(action)]).unwrap();
                if let Some(c) = own { bridge.apply_choice(0,&c.to_input(&dex)).unwrap(); }
                bridge.apply_choice(1,&action.to_input(&dex)).unwrap();
                decisions += 1;
            }
        }
    }
    println!("{}",json!({"modes":3,"seeds_per_mode":2,"decisions":decisions,
        "checks":["same chosen action","same execution distribution","same per-action visits and means"],
        "path":"native evaluation adapter vs actual WasmBlindSearcher Rust bridge, 512 iterations","records":records}));
}
