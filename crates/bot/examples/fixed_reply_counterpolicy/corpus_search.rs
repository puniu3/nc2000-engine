use super::*;
use nc2000_bot::{Belief,BlindSearch,Observer,RmConfig,SkuctSearch};
use std::path::Path;

#[derive(Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Config {
    game:usize,decision:usize,iterations:Vec<u32>,search_seeds:usize,seed:u64,
    #[serde(default,skip_serializing_if="Option::is_none")]
    observed_side:Option<usize>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    full_state:Option<bool>,
}

fn search_seed(cfg:&Config,index:usize)->u64 {
    cfg.seed ^ (cfg.game as u64).wrapping_mul(0x9e3779b97f4a7c15)
        ^ (cfg.decision as u64).wrapping_mul(0xbf58476d1ce4e5b9)
        ^ (index as u64).wrapping_mul(0x94d049bb133111eb)
}

fn observed_search(dex:&Dex,game:&Game,cfg:&Config,root:&Battle)->Value {
    let Some(side)=cfg.observed_side else {return Value::Null;};assert!(side<2);
    let path="data/belief-pool-v3/belief-pool.json";
    let mut b=Battle::from_fixture(dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();b.set_log_enabled(true);
    let mut obs=Observer::new(&b,side);
    let mut blind=Belief::new(dex,&nc2000_bot::preview::load_meta_pool(Path::new(path)),&obs);
    let mut pinned=Belief::pinned_from_battle(&b,&obs);
    for step in &game.decisions[..cfg.decision] {
        assert_eq!(b.state_key128().to_string(),step.key);
        let legal=[0,1].map(|s|b.legal_choices(dex,s));
        b.apply_choices(dex,[0,1].map(|s|step.choices[s].map(|i|legal[s][i]))).unwrap();
        obs.observe(&b,dex);blind.sync(dex,&obs);pinned.sync(dex,&obs);
    }
    assert_eq!(b.state_key128(),root.state_key128());assert_eq!(b.essence(dex),root.essence(dex));
    let observations=obs.mons().iter().map(|mon|json!({"species":dex.species.key(mon.species),
        "appeared":mon.appeared,"moves":mon.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),
        "current_item":mon.item.current.map(|item|item.map(|i|dex.items.key(i)))})).collect::<Vec<_>>();
    let mut rows=Vec::new();
    for (mode,belief) in [("pinned",&pinned),("blind",&blind)] {
        for index in 0..cfg.search_seeds {
            let seed=search_seed(cfg,index);let started=Instant::now();
            let mut search=BlindSearch::new(&b,dex,RmConfig {c:0.4,..Default::default()},side,seed);
            for &iterations in &cfg.iterations {
                search.step(dex,belief,&obs,iterations-search.iterations());
                rows.push(json!({"mode":mode,"seed_index":index,"seed":seed,"iterations":iterations,
                    "choice":search.best().map(|a|nc2000_bot::player::action_input(dex,a)),
                    "actions":search.actions().iter().map(|&a|nc2000_bot::player::action_input(dex,a)).collect::<Vec<_>>(),
                    "visits":search.visits(),"means":search.means(),"dominated":search.dominated(),"nodes":search.node_count(),
                    "joint":search.root_matrix().into_iter().map(|(own,reply,samples,mean)|json!({"own":own,"reply":nc2000_bot::player::action_input(dex,reply),"samples":samples,"mean":mean})).collect::<Vec<_>>(),
                    "reply_availability":search.root_replies().iter().map(|&(reply,n)|json!({"reply":nc2000_bot::player::action_input(dex,reply),"samples":n})).collect::<Vec<_>>(),
                    "elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
            }
            eprintln!("mode={mode} seed_index={index} complete");
        }
    }
    json!({"side":side,"rows":rows,"observations":observations,"pool_sha256":sha(&std::fs::read(path).unwrap()),
        "belief":{"alive":blind.alive().len(),"fallback":blind.is_fallback(),"species_prior":blind.has_prior(),
            "candidate_ids":blind.alive().iter().map(|&i|blind.candidate_id(i)).collect::<Vec<_>>()}})
}

pub(super) fn execute(args:&[String]) {
    assert_eq!(args.len(),5,"corpus corpus-search config output");
    let raw=std::fs::read(&args[1]).unwrap();let cfg_raw=std::fs::read(&args[3]).unwrap();
    let cfg:Config=serde_json::from_slice(&cfg_raw).unwrap();
    assert!(cfg.search_seeds>0 && !cfg.iterations.is_empty() && cfg.iterations[0]>0);
    assert!(cfg.iterations.windows(2).all(|w|w[0]<w[1]));
    let dex=conformance::load_dex();
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==cfg.game).unwrap();
    let root=reconstruct(&dex,&game,cfg.decision);let mut rows=Vec::new();
    for index in 0..if cfg.full_state.unwrap_or(true) {cfg.search_seeds} else {0} {
        let seed=search_seed(&cfg,index);
        let mut search=SkuctSearch::new(&root,&dex,RmConfig {c:0.4,..Default::default()},seed);
        let dimensions=[0,1].map(|side|search.actions(side).len().max(1));
        let mut joint_n=vec![0u32;dimensions[0]*dimensions[1]];let mut joint_w=vec![0.0; joint_n.len()];
        let started=Instant::now();
        for &iterations in &cfg.iterations {
            for _ in search.iterations()..iterations {
                let (reward,pair)=search.step_one(&dex);let cell=pair[0]*dimensions[1]+pair[1];
                joint_n[cell]+=1;joint_w[cell]+=reward;
            }
            rows.push(json!({"seed_index":index,"seed":seed,"iterations":iterations,
                "choices":([0,1].map(|side|search.best(side).map(|a|nc2000_bot::player::action_input(&dex,a)))),
                "actions":([0,1].map(|side|search.actions(side).iter().map(|&a|nc2000_bot::player::action_input(&dex,a)).collect::<Vec<_>>())),
                "visits":([0,1].map(|side|search.visits(side))),"means":([0,1].map(|side|search.means(side))),
                "joint_samples":joint_n,"joint_side_zero_rewards":joint_w,
                "nodes":search.node_count(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
        }
    }
    let observed=observed_search(&dex,&game,&cfg,&root);
    let paths=["crates/bot/examples/fixed_reply_counterpolicy/corpus_search.rs","crates/bot/src/smmcts.rs",
        "crates/bot/src/mcts.rs","crates/bot/src/eval.rs","crates/bot/src/agent.rs","crates/bot/src/rng.rs",
        "crates/bot/src/blind.rs","crates/bot/src/belief.rs","crates/bot/src/observe.rs"];
    let output=json!({"schema":"nc2000-corpus-search-v1","config":cfg,"config_sha256":sha(&cfg_raw),
        "state_key":root.state_key128().to_string(),"state":root.essence(&dex),"rows":rows,"observed":observed,
        "manifest":manifest(&dex,&raw),"search_source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "interpretation":"Current full-information SkuctSearch with c=0.4 and the original policy-gap seed schedule. Optional pinned-team and shipped-pool BlindSearch rebuild public observations from preview through the exact trajectory. Budgets increase on the same search tree. Root joint means are search estimates under sampled continuations, not optimal-continuation certificates. The actual-state full-information certificate cannot alone establish regret under an uncertain blind information set."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
