use nc2000_bot::root_policy::{mixed_root, RootCell};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct Cell {
    own: usize,
    reply: String,
    samples: u32,
    mean: f64,
}
#[derive(Deserialize)]
struct Reply {
    reply: String,
    samples: u32,
}
#[derive(Deserialize)]
struct Row {
    actions: Vec<String>,
    choice: String,
    dominated: Vec<bool>,
    iterations: u32,
    joint: Vec<Cell>,
    reply_availability: Vec<Reply>,
}
fn policy(row: &Row) -> Value {
    assert!(row.iterations > 0 && row.actions.len() == row.dominated.len());
    let best = row.actions.iter().position(|a| *a == row.choice).unwrap();
    let fallback: Vec<f64> = (0..row.actions.len())
        .map(|i| f64::from(i == best))
        .collect();
    let eligible: Vec<bool> = row.dominated.iter().map(|&v| !v).collect();
    let availability: Vec<u32> = row.reply_availability.iter().map(|r| r.samples).collect();
    let cells: Vec<RootCell> = row
        .joint
        .iter()
        .map(|c| RootCell {
            own: c.own,
            reply: row
                .reply_availability
                .iter()
                .position(|r| r.reply == c.reply)
                .unwrap(),
            samples: c.samples,
            mean: c.mean,
        })
        .collect();
    match mixed_root(&eligible, row.iterations, &availability, &cells) {
        Some(mix) => {
            json!({"policy":mix.policy,"mixed":true,"estimated_advantage":mix.estimated_advantage})
        }
        None => json!({"policy":fallback,"mixed":false}),
    }
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 3, "input output");
    assert!(!std::path::Path::new(&args[2]).exists());
    let raw = std::fs::read(&args[1]).unwrap();
    let cases: Vec<Value> = serde_json::from_slice(&raw).unwrap();
    let output: Vec<Value> = cases
        .into_iter()
        .map(|case| {
            let row: Row = serde_json::from_value(case["row"].clone()).unwrap();
            json!({"id":case["id"],"result":policy(&row)})
        })
        .collect();
    let sha = |raw: &[u8]| format!("{:x}", Sha256::digest(raw));
    let paths = [
        "crates/bot/examples/root_mix_policy.rs",
        "crates/bot/src/exact.rs",
        "crates/bot/src/root_policy.rs",
    ];
    let output = json!({"schema":"nc2000-root-mix-policy-v1","rows":output,"input_sha256":sha(&raw),
        "source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "interpretation":"Experimental extraction from nonstationary search estimates. The sample-count margin is a gate heuristic, not a confidence certificate. Requires invariant sampled reply availability and a padded mixed value above every padded pure value by 0.02. No game-specific move or species rules."});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
