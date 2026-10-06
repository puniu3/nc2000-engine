use nc2000_bot::smmcts::{RmConfig, SelRule, SkuctSearch};
use nc2000_bot::{belief::Belief, blind::BlindSearch, observe::Observer, rng::SplitMix64};
use nc2000_engine::{battle::{enumerate::enumerate_step_partial, PokemonSet}, dex::Dex, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::BufRead;

fn key(dex: &Dex, b: &Battle, guarded: bool) -> Value {
    let damage=guarded && !b.damage_bookkeeping_observable(dex);
    let claw=guarded && !b.quick_claw_roll_observable(dex);
    let semantic=if claw {b.state_key128_without_quick_claw_roll(dex,damage)}
        else if damage {b.state_key128_without_damage_bookkeeping()} else {b.state_key128()};
    json!([damage,claw,semantic.to_string()])
}

fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"corpus manifest output");
    let raw=std::fs::read(&args[2]).unwrap();let m:Value=serde_json::from_slice(&raw).unwrap();let cfg=&m["config"];
    let corpus=std::fs::read(&args[1]).unwrap();let dex=conformance::load_dex();
    let game=corpus.lines().map(|l|serde_json::from_str::<Value>(&l.unwrap()).unwrap())
        .find(|g|g["game"]==cfg["game"]).unwrap();
    let teams:[Vec<PokemonSet>;2]=serde_json::from_value(game["teams"].clone()).unwrap();
    let mut root=Battle::from_fixture(&dex,game["seed"].as_str().unwrap(),&teams[0],&teams[1]).unwrap();
    let decision=cfg["decision"].as_u64().unwrap() as usize;
    for d in game["decisions"].as_array().unwrap().iter().take(decision) {
        assert_eq!(root.state_key128().to_string(),d["key"].as_str().unwrap());
        let aa=[0,1].map(|s|root.legal_choices(&dex,s));
        root.apply_choices(&dex,[0,1].map(|s|d["choices"][s].as_u64().map(|i|aa[s][i as usize]))).unwrap();
    }
    assert_eq!(root.state_key128().to_string(),game["decisions"][decision]["key"].as_str().unwrap());
    let guarded=cfg["guarded_keys"].as_bool().unwrap();let cap=m["cap"].as_u64().unwrap() as usize;
    let targets=m["targets"].as_array().unwrap();assert_eq!(targets.len(),1);let target=&targets[0];
    let mut b=root.clone();let mut trace=Vec::new();
    for step in target["path"].as_array().unwrap() {
        let choices=[0,1].map(|s| {
            let aa=if b.needs_choice()[s] {b.legal_choices(&dex,s).into_iter().map(Some).collect()} else {vec![None]};
            aa.into_iter().find(|a|json!(a.map(|a|a.to_input(&dex)))==step["choices"][s]).unwrap()
        });
        let found=enumerate_step_partial(&dex,&b,choices,cap);
        b=found.leaves.into_iter().find(|l|key(&dex,&l.battle,guarded)==step["child"])
            .expect("recorded child must be reachable under its guarded identity").battle;
        trace.push(json!({"runs":found.runs,"cache_key":key(&dex,&b,guarded),"state_key":b.state_key128().to_string()}));
    }
    assert_eq!(key(&dex,&b,guarded),target["cache_key"]);assert!(b.outcome().is_none());
    let legal=[0,1].map(|s|b.legal_choices(&dex,s));assert_eq!(key(&dex,&b,guarded),target["cache_key"]);
    let budgets:Vec<u32>=serde_json::from_value(m["search_budgets"].clone()).unwrap();
    assert!(!budgets.is_empty() && budgets[0]>0 && budgets.windows(2).all(|a|a[0]<a[1]));
    let mut searches=Vec::new();
    for c in m["search_c"].as_array().unwrap() {for seed in m["search_seeds"].as_array().unwrap() {
        let c=c.as_f64().unwrap();let seed=seed.as_u64().unwrap();assert!(c.is_finite() && c>=0.0);
        let cfg=RmConfig {iterations:*budgets.last().unwrap(),rule:SelRule::Ucb,c,..RmConfig::default()};
        let mut search=SkuctSearch::new(&b,&dex,cfg,seed);let mut checkpoints=Vec::new();
        for &budget in &budgets {
            search.step(&dex,budget-search.iterations());assert_eq!(search.iterations(),budget);
            let sides=[0,1].map(|s| {
                assert_eq!(search.actions(s),legal[s]);
                json!({"actions":search.actions(s).iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                    "visits":search.visits(s),"means":search.means(s),"chosen":search.best(s).unwrap().to_input(&dex)})
            });
            checkpoints.push(json!({"iterations":budget,"nodes":search.node_count(),"mean_depth":search.mean_depth(),"sides":sides}));
        }
        searches.push(json!({"kind":"skuct","c":c,"seed":seed,"checkpoints":checkpoints}));
    }}
    if m["search_pinned"].as_bool().unwrap_or(false) {
        let observers=[0,1].map(|s|Observer::new(&b,s));
        let beliefs=[0,1].map(|s|Belief::pinned_from_battle(&b,&observers[s]));
        for s in [0,1] {for seed in 0..32 {
            let sampled=beliefs[s].determinize(&dex,&b,&observers[s],&mut SplitMix64::new(seed));
            assert_eq!(key(&dex,&sampled,guarded),key(&dex,&b,guarded),"pinned determinization changed the certified state");
        }}
        for c in m["search_c"].as_array().unwrap() {for seed in m["search_seeds"].as_array().unwrap() {
            let c=c.as_f64().unwrap();let seed=seed.as_u64().unwrap();
            let cfg=RmConfig {iterations:*budgets.last().unwrap(),rule:SelRule::Ucb,c,..RmConfig::default()};
            let mut search=[0,1].map(|s|BlindSearch::new(&b,&dex,cfg.clone(),s,seed));let mut checkpoints=Vec::new();
            for &budget in &budgets {
                let sides=[0,1].map(|s| {
                    let remaining=budget-search[s].iterations();
                    search[s].step(&dex,&beliefs[s],&observers[s],remaining);assert_eq!(search[s].iterations(),budget);
                    assert_eq!(search[s].actions(),legal[s]);
                    json!({"actions":search[s].actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                        "visits":search[s].visits(),"means":search[s].means(),"chosen":search[s].best().unwrap().to_input(&dex)})
                });
                checkpoints.push(json!({"iterations":budget,"nodes":search.iter().map(BlindSearch::node_count).sum::<usize>(),
                    "mean_depth":Value::Null,"sides":sides}));
            }
            searches.push(json!({"kind":"pinned","c":c,"seed":seed,"checkpoints":checkpoints}));
        }}
    }
    let output=json!({"manifest_sha256":format!("{:x}",Sha256::digest(raw)),
        "corpus_sha256":format!("{:x}",Sha256::digest(corpus)),"root_state_key":root.state_key128().to_string(),
        "id":target["id"],"cache_key":key(&dex,&b,guarded),"trace":trace,"state":b.essence(&dex),"searches":searches});
    std::fs::write(&args[3],serde_json::to_vec(&output).unwrap()).unwrap();
    eprintln!("searches={}",output["searches"].as_array().unwrap().len());
}
