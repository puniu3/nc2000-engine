use super::*;
use nc2000_bot::{Belief,BlindSearch,Observer,RmConfig,SkuctSearch};
use nc2000_bot::bounds::{Bounds,RootPayoffs};
use nc2000_bot::smmcts::SelRule;
use std::path::Path;

const BUDGETS:[u32;4]=[100,1000,10000,27000];
const SEEDS:usize=8;
const SEED_BASE:u64=2026102401;
const POOL:&str="data/belief-pool-v3/belief-pool.json";

fn named(dex:&Dex,b:&mut Battle,side:usize,name:&Value)->Option<SearchChoice> {
    if name.is_null() {assert!(!b.needs_choice()[side]);return None;}
    Some(b.legal_choices(dex,side).into_iter().find(|&a|nc2000_bot::player::action_input(dex,a)==name.as_str().unwrap()).unwrap())
}

fn bounds(value:&Value)->Bounds {Bounds {lo:value["lo"].as_f64().unwrap(),hi:value["hi"].as_f64().unwrap()}}

fn assessment(dex:&Dex,matrix:&RootPayoffs,choice:SearchChoice)->Value {
    let policy=matrix.actions[0].iter().map(|&a|if a==Some(choice) {1.0} else {0.0}).collect::<Vec<_>>();
    assert_eq!(policy.iter().sum::<f64>(),1.0);
    let loss=matrix.policy_loss(0,&policy);let score=matrix.policy_value(0,&policy);
    json!({"choice":nc2000_bot::player::action_input(dex,choice),
        "true_state_first_action_loss":{"lo":loss.lo,"hi":loss.hi},
        "true_state_optimal_continuation_score":{"lo":score.lo,"hi":score.hi}})
}

pub(super) fn execute(args:&[String]) {
    assert_eq!(args.len(),5,"corpus continuation-search certified_record output");
    let raw=std::fs::read(&args[1]).unwrap();let source=std::fs::read(&args[3]).unwrap();
    let record:Value=serde_json::from_slice(&source).unwrap();let dex=conformance::load_dex();
    assert_eq!(record["schema"],"nc2000-waking-continuations-v1");
    assert_eq!(record["manifest"]["corpus_sha256"],sha(&raw));
    for pair in record["manifest"]["source_hashes"].as_array().unwrap() {
        let path=pair[0].as_str().unwrap();
        if path.starts_with("crates/engine/") || path.starts_with("crates/bot/src/") {
            assert_eq!(sha(&std::fs::read(path).unwrap()),pair[1].as_str().unwrap(),"certificate source changed: {path}");
        }
    }
    assert_eq!(record["manifest"]["dex_fingerprint"],format!("{:?}",dex.replay_fingerprint));
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==17).unwrap();
    assert_eq!(record["positions"].as_array().unwrap().len(),1);
    let position=&record["positions"][0];let state=&position["state"];
    let mut b=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();b.set_log_enabled(true);
    let mut obs=Observer::new(&b,0);
    let pool=nc2000_bot::preview::load_meta_pool(Path::new(POOL));
    let mut blind=Belief::new(&dex,&pool,&obs);let mut pinned=Belief::pinned_from_battle(&b,&obs);
    for step in &game.decisions[..21] {
        assert_eq!(b.state_key128().to_string(),step.key);
        let legal=[0,1].map(|s|b.legal_choices(&dex,s));
        b.apply_choices(&dex,[0,1].map(|s|step.choices[s].map(|i|legal[s][i]))).unwrap();
        obs.observe(&b,&dex);blind.sync(&dex,&obs);pinned.sync(&dex,&obs);
    }
    assert_eq!(b.state_key128().to_string(),record["root_key"].as_str().unwrap());
    b.reseed(seed(record["config"]["seed"].as_u64().unwrap(),state["trial"].as_u64().unwrap() as usize));
    for step in state["prefix"].as_array().unwrap() {
        let pair=[0,1].map(|s|named(&dex,&mut b,s,&step[s]));
        b.apply_choices(&dex,pair).unwrap();obs.observe(&b,&dex);blind.sync(&dex,&obs);pinned.sync(&dex,&obs);
    }
    assert_eq!(b.state_key128().to_string(),state["key"].as_str().unwrap());
    assert_eq!(b.essence(&dex),state["essence"]);
    let proof=&position["unrestricted"];assert!(proof.is_object());
    let actions=[0,1].map(|s|proof["root_actions"][s].as_array().unwrap().iter().map(|name|named(&dex,&mut b,s,name)).collect::<Vec<_>>());
    let matrix=RootPayoffs {actions,cells:proof["root_cells"].as_array().unwrap().iter().map(bounds).collect(),value:bounds(&proof["value"])};
    let mut rows=Vec::new();
    for mode in ["full_state","pinned","blind"] {
        for index in 0..SEEDS {
            let search_seed=seed(SEED_BASE,index);let cfg=RmConfig {c:0.4,rule:SelRule::Ucb,..Default::default()};
            let started=Instant::now();
            if mode=="full_state" {
                let mut search=SkuctSearch::new(&b,&dex,cfg,search_seed);
                for iterations in BUDGETS {
                    search.step(&dex,iterations-search.iterations());
                    rows.push(json!({"mode":mode,"seed_index":index,"seed":search_seed,"iterations":iterations,
                        "assessment":assessment(&dex,&matrix,search.best(0).unwrap()),
                        "actions":search.actions(0).iter().map(|&a|nc2000_bot::player::action_input(&dex,a)).collect::<Vec<_>>(),
                        "visits":search.visits(0),"means":search.means(0),"nodes":search.node_count(),
                        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
                }
            } else {
                let belief=if mode=="blind" {&blind} else {&pinned};
                let mut search=BlindSearch::new(&b,&dex,cfg,0,search_seed);
                for iterations in BUDGETS {
                    search.step(&dex,belief,&obs,iterations-search.iterations());
                    rows.push(json!({"mode":mode,"seed_index":index,"seed":search_seed,"iterations":iterations,
                        "assessment":assessment(&dex,&matrix,search.best().unwrap()),
                        "actions":search.actions().iter().map(|&a|nc2000_bot::player::action_input(&dex,a)).collect::<Vec<_>>(),
                        "visits":search.visits(),"means":search.means(),"dominated":search.dominated(),"nodes":search.node_count(),
                        "elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
                }
            }
            eprintln!("mode={mode} seed_index={index} complete");
        }
    }
    let paths=["crates/bot/examples/fixed_reply_counterpolicy/continuation_search.rs","crates/bot/src/blind.rs",
        "crates/bot/src/smmcts.rs","crates/bot/src/observe.rs","crates/bot/src/belief.rs","crates/bot/src/mcts.rs",
        "crates/bot/src/eval.rs","crates/bot/src/prior.rs","crates/bot/src/agent.rs","crates/bot/src/rng.rs"];
    let output=json!({"schema":"nc2000-certified-continuation-search-v1","state_key":b.state_key128().to_string(),
        "budgets":BUDGETS,"seeds":SEEDS,"seed_base":SEED_BASE,"rows":rows,"manifest":manifest(&dex,&raw),
        "certificate_sha256":sha(&source),"search_source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "belief_pool_sha256":sha(&std::fs::read(POOL).unwrap()),
        "belief":{"alive":blind.alive().len(),"fallback":blind.is_fallback(),"species_prior":blind.has_prior()},
        "interpretation":"Exact recorded trajectory and PRNG are reconstructed with public logs from preview onward. Full-state Skuct, pinned-team BlindSearch, and shipped-pool BlindSearch use c=0.4 and default play/mask rules. Belief is synchronized after every real transition. The certified score/loss is conditional on the actual full state with optimal continuation, not a blind information-set optimality bound. Search estimates are logged separately and never used as certificates. One reachable anchor and eight seeds do not establish general bot strength or convergence."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
