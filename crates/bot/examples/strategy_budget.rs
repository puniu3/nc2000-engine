use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration,Instant};

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>;2] }

#[derive(Deserialize)]
struct Game {
    seed: String,
    teams: [Vec<PokemonSet>;2],
    decisions: Vec<Decision>,
    samples: Vec<usize>,
    prefix_only: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(),3,"corpus output");
    let game: Game = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    assert!(game.prefix_only && game.samples.len()==1);
    let dex=conformance::load_dex();
    let mut battle=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();
    battle.set_log_enabled(true);
    let mut obs=[0,1].map(|s|Observer::new(&battle,s));
    let mut beliefs=[0,1].map(|s|Belief::pinned_from_battle(&battle,&obs[s]));
    for decision in &game.decisions[..game.samples[0]] {
        assert_eq!(battle.state_key128().to_string(),decision.key);
        let legal=[0,1].map(|s|battle.legal_choices(&dex,s));
        battle.apply_choices(&dex,[0,1].map(|s|decision.choices[s].map(|i|legal[s][i]))).unwrap();
        for s in 0..2 { obs[s].observe(&battle,&dex); beliefs[s].sync(&dex,&obs[s]); }
    }
    assert_eq!(battle.state_key128().to_string(),game.decisions[game.samples[0]].key);
    assert!(battle.outcome().is_none());
    let actions=[0,1].map(|s|battle.legal_choices(&dex,s));
    let mut search_rows=Vec::new();
    for side in 0..2 {
        for seed in 2026111101..2026111104 {
            let start=Instant::now();
            let mut search=BlindSearch::new(&battle,&dex,RmConfig {c:0.4,..Default::default()},side,seed);
            search.step(&dex,&beliefs[side],&obs[side],27000);
            search_rows.push(json!({"side":side,"seed":seed,"iterations":search.iterations(),
                "milliseconds":start.elapsed().as_secs_f64()*1000.,
                "choice":search.best().map(|a|a.to_input(&dex))}));
        }
    }
    let mut rows=Vec::new();
    let config=BoundConfig {time_budget:Some(Duration::from_millis(250)),work_budget:10000,
        eps:1e-10,..Default::default()};
    for side in 0..2 {
        let mut solver=BoundSolver::new(&dex,config.clone());
        let start=Instant::now();
        for target in [0.02,0.005,0.001] {
            while start.elapsed()<Duration::from_secs(60) && solver.stats.runs<2_000_000 {
                let before=solver.stats.runs;
                let report=solver.solve_strategy(&battle,side,target);
                let certificate=report.strategies.as_ref().map(|s| {
                    let cert=&s[side];
                    let policy:Vec<_>=actions[side].iter().map(|a|
                        cert.actions.iter().find(|(c,_)|*c==Some(*a)).map_or(0.,|(_,p)|*p)).collect();
                    json!({"policy":policy,"guaranteed_value":cert.guaranteed_value,"max_loss":cert.max_loss})
                });
                rows.push(json!({"side":side,"target":target,"stop":format!("{:?}",report.stop),
                    "milliseconds":start.elapsed().as_secs_f64()*1000.,"runs":solver.stats.runs,
                    "call_runs":report.runs,"value":[report.bounds.lo,report.bounds.hi],
                    "certificate":certificate,"met":report.strategy(side,target).is_some()}));
                if report.strategy(side,target).is_some() || solver.stats.runs==before {break;}
            }
        }
        eprintln!("side {side}: runs={} ms={:.1}",solver.stats.runs,start.elapsed().as_secs_f64()*1000.);
    }
    let output=json!({"schema":"nc2000-strategy-budget-v1","state_key":battle.state_key128().to_string(),
        "actions":actions.map(|aa|aa.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),
        "config":format!("{config:?}"),"search":search_rows,"rows":rows,
        "scope":"Known full-information state reconstructed through every legal prefix decision. Solver calls retain the graph; time and run counts are cumulative per side. Certificates concern only the first action under optimal continuation."});
    std::fs::write(&args[2],serde_json::to_vec_pretty(&output).unwrap()).unwrap();
}
