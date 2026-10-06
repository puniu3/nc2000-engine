#[allow(dead_code)]
mod finite {
    include!("policy_witness.rs");
    pub fn solve(dex: &Dex, battle: Battle, config: Value) -> Value {
        let cfg: Config = serde_json::from_value(config).unwrap();
        assert!(cfg.target_side < 2 && cfg.target_side == cfg.policy_side && cfg.guarded_keys);
        assert!(cfg.horizon > 0 && cfg.work_budget > 0 && cfg.cell_cap > 0);
        let mut w = Witness {
            dex,
            cfg,
            runs: 0,
            memo: HashMap::new(),
            nodes: Vec::new(),
            prior: HashMap::new(),
            prioritize: false,
        };
        let root = w.visit(battle, 0, w.cfg.work_budget);
        json!({"config":w.cfg,"runs":w.runs,"root":root,"bounds":w.nodes[root]["bounds"],"nodes":w.nodes})
    }
}
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, PokemonSet},
    dex::Dex,
    state::Battle,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::BufRead;

fn key(dex: &Dex, b: &Battle, guarded: bool) -> Value {
    let damage = guarded && !b.damage_bookkeeping_observable(dex);
    let claw = guarded && !b.quick_claw_roll_observable(dex);
    let semantic = if claw {
        b.state_key128_without_quick_claw_roll(dex, damage)
    } else if damage {
        b.state_key128_without_damage_bookkeeping()
    } else {
        b.state_key128()
    };
    json!([damage, claw, semantic.to_string()])
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus manifest output");
    let raw = std::fs::read(&args[2]).unwrap();
    let m: Value = serde_json::from_slice(&raw).unwrap();
    let cfg = &m["config"];
    let corpus = std::fs::read(&args[1]).unwrap();
    let dex = conformance::load_dex();
    let game = corpus
        .lines()
        .map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap())
        .find(|g| g["game"] == cfg["game"])
        .unwrap();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut root =
        Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    let decision = cfg["decision"].as_u64().unwrap() as usize;
    for d in game["decisions"].as_array().unwrap().iter().take(decision) {
        assert_eq!(root.state_key128().to_string(), d["key"].as_str().unwrap());
        let aa = [0, 1].map(|s| root.legal_choices(&dex, s));
        root.apply_choices(
            &dex,
            [0, 1].map(|s| d["choices"][s].as_u64().map(|i| aa[s][i as usize])),
        )
        .unwrap();
    }
    assert_eq!(
        root.state_key128().to_string(),
        game["decisions"][decision]["key"].as_str().unwrap()
    );
    let guarded = cfg["guarded_keys"].as_bool().unwrap();
    let cap = m["cap"].as_u64().unwrap() as usize;
    let targets = m["targets"].as_array().unwrap();
    assert_eq!(targets.len(), 1);
    let target = &targets[0];
    let mut b = root.clone();
    let mut trace = Vec::new();
    for step in target["path"].as_array().unwrap() {
        let choices = [0, 1].map(|s| {
            let aa = if b.needs_choice()[s] {
                b.legal_choices(&dex, s).into_iter().map(Some).collect()
            } else {
                vec![None]
            };
            aa.into_iter()
                .find(|a| json!(a.map(|a| a.to_input(&dex))) == step["choices"][s])
                .unwrap()
        });
        let found = enumerate_step_partial(&dex, &b, choices, cap);
        b = found
            .leaves
            .into_iter()
            .find(|l| key(&dex, &l.battle, guarded) == step["child"])
            .expect("recorded child must be reachable under its guarded identity")
            .battle;
        trace.push(json!({"runs":found.runs,"cache_key":key(&dex,&b,guarded),"state_key":b.state_key128().to_string()}));
    }
    assert_eq!(key(&dex, &b, guarded), target["cache_key"]);
    assert!(b.outcome().is_none());
    let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
    assert_eq!(key(&dex, &b, guarded), target["cache_key"]);
    assert_eq!(m["witness"]["game"], cfg["game"]);
    assert_eq!(m["witness"]["decision"], cfg["decision"]);
    let ledger = finite::solve(&dex, b.clone(), m["witness"].clone());
    let output = json!({"manifest_sha256":format!("{:x}",Sha256::digest(raw)),
        "corpus_sha256":format!("{:x}",Sha256::digest(corpus)),"root_state_key":root.state_key128().to_string(),
        "id":target["id"],"cache_key":key(&dex,&b,guarded),"trace":trace,"state":b.essence(&dex),
        "legal":legal.map(|aa|aa.into_iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),"ledger":ledger});
    eprintln!(
        "runs={} bounds={}",
        output["ledger"]["runs"], output["ledger"]["bounds"]
    );
    std::fs::write(&args[3], serde_json::to_vec(&output).unwrap()).unwrap();
}
