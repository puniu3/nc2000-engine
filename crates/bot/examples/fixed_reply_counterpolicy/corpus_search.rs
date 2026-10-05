use super::*;
use nc2000_bot::{RmConfig,SkuctSearch};

#[derive(Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Config {game:usize,decision:usize,iterations:Vec<u32>,search_seeds:usize,seed:u64}

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
    for index in 0..cfg.search_seeds {
        let seed=cfg.seed ^ (cfg.game as u64).wrapping_mul(0x9e3779b97f4a7c15)
            ^ (cfg.decision as u64).wrapping_mul(0xbf58476d1ce4e5b9)
            ^ (index as u64).wrapping_mul(0x94d049bb133111eb);
        let mut search=SkuctSearch::new(&root,&dex,RmConfig {c:0.4,..Default::default()},seed);
        let started=Instant::now();
        for &iterations in &cfg.iterations {
            search.step(&dex,iterations-search.iterations());
            rows.push(json!({"seed_index":index,"seed":seed,"iterations":iterations,
                "choices":([0,1].map(|side|search.best(side).map(|a|nc2000_bot::player::action_input(&dex,a)))),
                "actions":([0,1].map(|side|search.actions(side).iter().map(|&a|nc2000_bot::player::action_input(&dex,a)).collect::<Vec<_>>())),
                "visits":([0,1].map(|side|search.visits(side))),"means":([0,1].map(|side|search.means(side))),
                "nodes":search.node_count(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
        }
    }
    let paths=["crates/bot/examples/fixed_reply_counterpolicy/corpus_search.rs","crates/bot/src/smmcts.rs",
        "crates/bot/src/mcts.rs","crates/bot/src/eval.rs","crates/bot/src/agent.rs","crates/bot/src/rng.rs"];
    let output=json!({"schema":"nc2000-corpus-search-v1","config":cfg,"config_sha256":sha(&cfg_raw),
        "state_key":root.state_key128().to_string(),"state":root.essence(&dex),"rows":rows,
        "manifest":manifest(&dex,&raw),"search_source_hashes":paths.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "interpretation":"Current full-information SkuctSearch with c=0.4, using the original policy-gap seed schedule and increasing iteration budgets on the same search tree. Root means and chosen actions are search measurements, not game-value certificates or a blind-policy assessment."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
