use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_bot::blind::BlindSearch;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Belief, Observer, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::prng::{BattleRng, Prng};
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    game: String,
    game_sha256: String,
    decision: usize,
    side: usize,
    arms: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    prior: String,
    prior_sha256: String,
    cases: Vec<Case>,
    iterations: u32,
    c: f64,
    trials: usize,
    seed: u64,
    max_turns: u16,
    threads: usize,
    root_budgets: Vec<u32>,
    root_seeds: usize,
    interpretation: String,
}

fn hash(path: &Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "tactical_continuation PROTOCOL.json OUTPUT_DIRECTORY"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-tactical-continuation-v1");
    assert!(protocol.trials > 0 && protocol.threads > 0 && protocol.iterations > 0);
    assert!(!protocol.interpretation.is_empty());
    assert!(protocol.root_budgets.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(hash(Path::new(&protocol.prior)), protocol.prior_sha256);
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new(&protocol.prior));
    let cfg = RmConfig {
        iterations: protocol.iterations,
        c: protocol.c,
        ..Default::default()
    };
    let dir = Path::new(&args[2]);
    std::fs::create_dir(dir).unwrap();
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),"protocol_sha256":hash(Path::new(&args[1])),
        "source_sha256":hash(Path::new("crates/bot/examples/tactical_continuation.rs")),"binary_sha256":hash(&std::env::current_exe().unwrap())});
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let mut states = Vec::new();
    for case in &protocol.cases {
        assert_eq!(hash(Path::new(&case.game)), case.game_sha256);
        let game: Value = serde_json::from_slice(&std::fs::read(&case.game).unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b = Battle::from_fixture(
            &dex,
            game["battle_seed"].as_str().unwrap(),
            &teams[0],
            &teams[1],
        )
        .unwrap();
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        let mut beliefs = [0, 1].map(|s| Belief::new(&dex, &pool, &observers[s]));
        for r in &game["records"].as_array().unwrap()[..=case.decision] {
            assert_eq!(
                b.state_key128().to_string(),
                r["state_key"].as_str().unwrap()
            );
            let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
            for side in 0..2 {
                if !legal[side].is_empty() {
                    observers[side].observe(&b, &dex);
                    beliefs[side].sync(&dex, &observers[side]);
                }
            }
            if r["decision"] == case.decision {
                break;
            }
            let chosen = [0, 1].map(|s| {
                r["choices"][s]
                    .as_str()
                    .map(|input| *legal[s].iter().find(|a| a.to_input(&dex) == input).unwrap())
            });
            b.apply_choices(&dex, chosen).unwrap();
        }
        let row = &game["records"][case.decision];
        let legal = b.legal_choices(&dex, case.side);
        for action in &case.arms {
            assert!(legal.iter().any(|a| a.to_input(&dex) == *action));
        }
        let seed = row["search_seeds"][case.side].as_u64().unwrap();
        let mut check = BlindSearch::new(&b, &dex, cfg.clone(), case.side, seed);
        check.step(
            &dex,
            &beliefs[case.side],
            &observers[case.side],
            protocol.iterations,
        );
        assert_eq!(
            check.best().unwrap().to_input(&dex),
            row["ordinary"][case.side].as_str().unwrap()
        );
        let mut roots = vec![
            json!({"kind":"archived_reproduction","seed":seed,"choice":check.best().unwrap().to_input(&dex),
            "actions":check.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"visits":check.visits(),"means":check.means(),"mask":check.dominated()}),
        ];
        for k in 0..protocol.root_seeds {
            let seed = protocol.seed ^ (k as u64).wrapping_mul(0x94d049bb133111eb);
            let mut search = BlindSearch::new(&b, &dex, cfg.clone(), case.side, seed);
            let started = Instant::now();
            for &budget in &protocol.root_budgets {
                search.step(
                    &dex,
                    &beliefs[case.side],
                    &observers[case.side],
                    budget - search.iterations(),
                );
                roots.push(json!({"kind":"fresh_root","seed":seed,"iterations":budget,"choice":search.best().unwrap().to_input(&dex),
                    "actions":search.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"visits":search.visits(),"means":search.means(),"mask":search.dominated(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
            }
        }
        std::fs::write(dir.join(format!("{}-roots.json",case.id)),serde_json::to_vec_pretty(&json!({"case":case.id,"state_key":b.state_key128().to_string(),"state":b.essence(&dex),"roots":roots})).unwrap()).unwrap();
        states.push((b, observers));
    }
    let jobs: Vec<_> = protocol
        .cases
        .iter()
        .enumerate()
        .flat_map(|(case_index, case)| {
            (0..protocol.trials).flat_map(move |trial| {
                (0..case.arms.len()).map(move |arm| (case_index, trial, arm))
            })
        })
        .collect();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let job = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(case_index,trial,arm)) = jobs.get(job) else {break};
                let case=&protocol.cases[case_index];
                let (root,root_observers)=&states[case_index];
                let mut b=root.clone();
                let mut observers=root_observers.clone();
                let mut beliefs=[0,1].map(|s|Belief::new(&dex,&pool,&observers[s]));
                let mut trial_rng=SplitMix64::new(protocol.seed ^ (trial as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (case_index as u64).wrapping_mul(0xbf58476d1ce4e5b9));
                let battle_seed=trial_rng.next();
                b.prng=BattleRng::seeded(Prng::new(battle_seed));
                let mut rngs=[SplitMix64::new(trial_rng.next()),SplitMix64::new(trial_rng.next())];
                let mut decisions=Vec::new();
                let started=Instant::now();
                while b.outcome().is_none() && b.turn <= protocol.max_turns {
                    let legal=[0,1].map(|s|b.legal_choices(&dex,s));
                    let picks=[0,1].map(|side| {
                        if legal[side].is_empty() {return None}
                        observers[side].observe(&b,&dex);
                        beliefs[side].sync(&dex,&observers[side]);
                        let seed=rngs[side].next();
                        if decisions.is_empty() && side==case.side {
                            return Some(*legal[side].iter().find(|a|a.to_input(&dex)==case.arms[arm]).unwrap());
                        }
                        if legal[side].len()==1 {return Some(legal[side][0])}
                        let mut search=BlindSearch::new(&b,&dex,cfg.clone(),side,seed);
                        search.step(&dex,&beliefs[side],&observers[side],protocol.iterations);
                        search.best()
                    });
                    assert!(picks.iter().any(Option::is_some));
                    decisions.push(json!({"key":b.state_key128().to_string(),"turn":b.turn,"choices":picks.map(|a|a.map(|a|a.to_input(&dex)))}));
                    b.apply_choices(&dex,picks).unwrap();
                }
                let score=b.outcome().map(|o|match o {
                    Outcome::Tie=>0.5,
                    Outcome::P1Win=>if case.side==0 {1.0}else{0.0},
                    Outcome::P2Win=>if case.side==1 {1.0}else{0.0},
                });
                let row=json!({"case":case.id,"trial":trial,"arm":case.arms[arm],"side":case.side,"battle_seed":battle_seed,"score":score,"terminal":b.outcome().is_some(),"outcome":format!("{:?}",b.outcome()),"turns":b.turn,"seconds":started.elapsed().as_secs_f64(),"decisions":decisions});
                std::fs::write(dir.join(format!("{}-t{trial:03}-a{arm}.json",case.id)),serde_json::to_vec(&row).unwrap()).unwrap();
                eprintln!("{}/{} {} trial={} arm={} score={} turns={}",job+1,jobs.len(),case.id,trial,case.arms[arm],row["score"],row["turns"]);
            });
        }
    });
}
