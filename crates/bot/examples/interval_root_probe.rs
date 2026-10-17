use nc2000_bot::root_policy::interval::{analyze, InformationModel, PayoffBounds};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

#[derive(Deserialize)]
struct Case {
    id: String,
    eligible: Vec<bool>,
    baseline: Vec<f64>,
    types: Vec<PayoffBounds>,
    model: InformationModel,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "input.jsonl new-output.jsonl");
    let input = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap());
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .unwrap();
    for line in input.lines() {
        let c: Case = serde_json::from_str(&line.unwrap()).unwrap();
        let result: Value = match analyze(&c.eligible, &c.baseline, &c.types, &c.model, 0.) {
            Ok(a) => serde_json::to_value(a).unwrap(),
            Err(e) => json!({"error":e}),
        };
        writeln!(out, "{}", json!({"id":c.id,"result":result})).unwrap();
    }
}
