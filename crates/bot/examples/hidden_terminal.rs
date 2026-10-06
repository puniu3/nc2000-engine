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
use nc2000_bot::Observer;
use nc2000_engine::{
    battle::{enumerate::enumerate_step_with_runner, PokemonSet, SearchChoice},
    dex::Dex,
    prng::{BattleRng, Draw},
    state::Battle,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn key(dex: &Dex, b: &Battle) -> Value {
    let damage = !b.damage_bookkeeping_observable(dex);
    let claw = !b.quick_claw_roll_observable(dex);
    let semantic = if claw {
        b.state_key128_without_quick_claw_roll(dex, damage)
    } else if damage {
        b.state_key128_without_damage_bookkeeping()
    } else {
        b.state_key128()
    };
    json!([damage, claw, semantic.to_string()])
}

fn observed(dex: &Dex, obs: &Observer) -> Value {
    json!(obs
        .mons()
        .iter()
        .map(|m| json!({"species":dex.species.key(m.species),
        "level":m.level,"gender":format!("{:?}",m.gender),"name":m.name,
        "preview_has_item":m.preview_has_item,"appeared":m.appeared,"hp_type_mask":m.hp_type_mask,
        "moves":m.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),
        "item":format!("{:?}",m.item)}))
        .collect::<Vec<_>>())
}

fn apply(
    dex: &Dex,
    base: &Battle,
    choices: [Option<SearchChoice>; 2],
    script: &[usize],
) -> (Battle, Vec<Draw>) {
    let mut b = base.clone();
    b.prng = BattleRng::enumerating(script.to_vec());
    b.apply_choices(dex, choices).unwrap();
    let trace = b.prng.oracle.take().unwrap().trace;
    (b, trace)
}

fn visible_log(log: &[String], side: usize) -> Vec<String> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < log.len() {
        if log[i].starts_with("|split|") {
            let own = log[i] == format!("|split|p{}", side + 1);
            result.push(log[i + if own { 1 } else { 2 }].clone());
            i += 3;
        } else {
            result.push(log[i].clone());
            i += 1;
        }
    }
    result
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus manifest output");
    let raw = std::fs::read(&args[2]).unwrap();
    let m: Value = serde_json::from_slice(&raw).unwrap();
    let corpus = std::fs::read(&args[1]).unwrap();
    let dex = conformance::load_dex();
    let game = std::str::from_utf8(&corpus)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|g| g["game"] == m["config"]["game"])
        .unwrap();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut alt = teams.clone();
    let change = &m["change"];
    let side = change["side"].as_u64().unwrap() as usize;
    let mon = change["mon"].as_u64().unwrap() as usize;
    let slot = change["slot"].as_u64().unwrap() as usize;
    assert_eq!(alt[side][mon].moves[slot], change["from"].as_str().unwrap());
    alt[side][mon].moves[slot] = change["to"].as_str().unwrap().into();
    let learnsets = nc2000_engine::validate::Learnsets::from_json(
        &std::fs::read_to_string("data/learnsets-gen2.json").unwrap(),
    )
    .unwrap();
    let validation = [&teams, &alt].map(|t| {
        t.each_ref().map(|team| {
            nc2000_engine::validate::validate_team(
                &dex,
                &learnsets,
                &serde_json::to_string(team).unwrap(),
            )
        })
    });
    assert!(validation.iter().flatten().all(|v| v["ok"] == true));
    let mut battles = [&teams, &alt].map(|t| {
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &t[0], &t[1]).unwrap();
        b.set_log_enabled(true);
        b
    });
    let observer_side = 1 - side;
    let mut obs = battles.each_ref().map(|b| Observer::new(b, observer_side));
    let mut history = obs.each_ref().map(|o| vec![observed(&dex, o)]);
    let mut trace = Vec::new();
    assert_eq!(observed(&dex, &obs[0]), observed(&dex, &obs[1]));
    assert_eq!(battles[0].log, battles[1].log);
    let decision = m["config"]["decision"].as_u64().unwrap() as usize;
    for d in game["decisions"].as_array().unwrap().iter().take(decision) {
        assert_eq!(
            battles[0].state_key128().to_string(),
            d["key"].as_str().unwrap()
        );
        let aa = battles
            .each_mut()
            .map(|b| [0, 1].map(|s| b.legal_choices(&dex, s)));
        for s in [0, 1] {
            assert_eq!(
                aa[0][s]
                    .iter()
                    .map(|a| a.to_input(&dex))
                    .collect::<Vec<_>>(),
                aa[1][s]
                    .iter()
                    .map(|a| a.to_input(&dex))
                    .collect::<Vec<_>>()
            );
        }
        for i in [0, 1] {
            battles[i]
                .apply_choices(
                    &dex,
                    [0, 1].map(|s| d["choices"][s].as_u64().map(|a| aa[i][s][a as usize])),
                )
                .unwrap();
            obs[i].observe(&battles[i], &dex);
        }
        assert_eq!(battles[0].log, battles[1].log);
        assert_eq!(observed(&dex, &obs[0]), observed(&dex, &obs[1]));
        for i in [0, 1] {
            history[i].push(observed(&dex, &obs[i]));
        }
    }
    assert_eq!(
        battles[0].state_key128().to_string(),
        game["decisions"][decision]["key"].as_str().unwrap()
    );
    let target = &m["targets"][0];
    assert_eq!(m["targets"].as_array().unwrap().len(), 1);
    for step in target["path"].as_array().unwrap() {
        let aa = battles.each_mut().map(|b| {
            [0, 1].map(|s| {
                let choices = if b.needs_choice()[s] {
                    b.legal_choices(&dex, s).into_iter().map(Some).collect()
                } else {
                    vec![None]
                };
                choices
                    .into_iter()
                    .find(|a| json!(a.map(|a| a.to_input(&dex))) == step["choices"][s])
                    .unwrap()
            })
        });
        let mut found = None;
        let mut runner = |script: &[usize]| {
            let (b, t) = apply(&dex, &battles[0], aa[0], script);
            if found.is_none() && key(&dex, &b) == step["child"] {
                found = Some(t.iter().map(|d| d.chosen).collect::<Vec<_>>());
            }
            (b, t, true)
        };
        let enumeration = enumerate_step_with_runner(
            &dex,
            &battles[0],
            aa[0],
            m["cap"].as_u64().unwrap() as usize,
            &mut runner,
        )
        .expect("path kernel must be complete");
        let script = found.expect("recorded child must be reachable");
        let mut draws = Vec::new();
        for i in [0, 1] {
            let (b, t) = apply(&dex, &battles[i], aa[i], &script);
            battles[i] = b;
            obs[i].observe(&battles[i], &dex);
            draws.push(json!(t
                .iter()
                .map(|d| json!({"label":d.label,"counts":d.counts,"chosen":d.chosen}))
                .collect::<Vec<_>>()));
        }
        assert_eq!(draws[0], draws[1]);
        assert_eq!(key(&dex, &battles[0]), step["child"]);
        assert_eq!(battles[0].log, battles[1].log);
        assert_eq!(observed(&dex, &obs[0]), observed(&dex, &obs[1]));
        for i in [0, 1] {
            history[i].push(observed(&dex, &obs[i]));
        }
        trace.push(json!({"script":script,"draws":draws[0],"runs":enumeration.runs,"keys":battles.each_ref().map(|b|key(&dex,b))}));
    }
    assert_eq!(key(&dex, &battles[0]), target["cache_key"]);
    let mut ledgers = Vec::new();
    for w in m["witnesses"].as_array().unwrap() {
        let i = w["variant"].as_u64().unwrap() as usize;
        assert!(i < 2);
        assert_eq!(w["config"]["game"], m["config"]["game"]);
        assert_eq!(w["config"]["decision"], m["config"]["decision"]);
        let ledger = finite::solve(&dex, battles[i].clone(), w["config"].clone());
        eprintln!(
            "{} runs={} bounds={}",
            w["id"], ledger["runs"], ledger["bounds"]
        );
        ledgers.push(json!({"id":w["id"],"variant":i,"ledger":ledger}));
    }
    let output = json!({"manifest_sha256":format!("{:x}",Sha256::digest(raw)),"corpus_sha256":format!("{:x}",Sha256::digest(corpus)),
        "teams":alt,"validation":validation,"observer_side":observer_side,"histories":history,
        "engine_logs":battles.each_ref().map(|b|&b.log),
        "public_logs":battles.each_ref().map(|b|visible_log(&b.log,observer_side)),"trace":trace,
        "states":battles.each_ref().map(|b|b.essence(&dex)),"keys":battles.each_ref().map(|b|key(&dex,b)),"ledgers":ledgers});
    std::fs::write(&args[3], serde_json::to_vec(&output).unwrap()).unwrap();
}
