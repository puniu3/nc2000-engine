use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, Write};
use std::time::Instant;

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }

#[derive(Deserialize)]
struct Game {
    game: usize, seed: String, teams: [Vec<PokemonSet>; 2], decisions: Vec<Decision>,
}

#[derive(Deserialize)]
struct Request { game: usize, decision: usize, side: usize }

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!((11..=12).contains(&args.len()), "corpus requests output runs nodes depth call-runs cell-cap retain-prefix defer-frontier [profile-selection]");
    let requests: Vec<Request> = serde_json::from_str(&std::fs::read_to_string(&args[2]).unwrap()).unwrap();
    let work_budget: usize = args[4].parse().unwrap();
    let node_budget: usize = args[5].parse().unwrap();
    let trial_depth: usize = args[6].parse().unwrap();
    let call_budget: usize = args[7].parse().unwrap();
    let cell_cap: usize = args[8].parse().unwrap();
    let retain_eager_prefix: bool = args[9].parse().unwrap();
    let defer_eager_frontier: bool = args[10].parse().unwrap();
    let profile_pending_selection = args.get(11).is_some_and(|s|s.parse::<bool>().unwrap());
    assert!(!defer_eager_frontier || retain_eager_prefix);
    assert!(work_budget>0 && call_budget>0);
    let dex = conformance::load_dex();
    let input = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap());
    let mut output = std::fs::File::create_new(&args[3]).unwrap();
    for line in input.lines() {
        let game: Game = serde_json::from_str(&line.unwrap()).unwrap();
        let Some(last) = requests.iter().filter(|r|r.game==game.game).map(|r|r.decision).max() else { continue; };
        let mut battle = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        for (index, decision) in game.decisions.iter().enumerate().take(last+1) {
            assert_eq!(battle.state_key128().to_string(), decision.key);
            let legal = [0,1].map(|s|battle.legal_choices(&dex,s));
            if let Some(request) = requests.iter().find(|r|r.game==game.game && r.decision==index) {
                let config = BoundConfig { work_budget:call_budget, node_budget, trial_depth, cell_cap, retain_eager_prefix, defer_eager_frontier, profile_pending_selection, eps:1e-10, ..Default::default() };
                let started = Instant::now();
                let mut solver = BoundSolver::new(&dex,config);
                let report = loop {
                    solver.cfg.work_budget=call_budget.min(work_budget-solver.stats.runs);
                    let report=solver.solve_strategy(&battle,request.side,0.005);
                    if report.strategy(request.side,0.005).is_some() || report.runs==0 || solver.stats.runs>=work_budget { break report; }
                };
                let milliseconds = started.elapsed().as_secs_f64()*1000.;
                let graph = solver.graph_diagnostics();
                let chance: Vec<_> = solver.root_chance_diagnostics(&battle).unwrap().into_iter().map(|c|json!({
                    "visits":c.visits,"runs":c.runs,"pending_mass":c.pending_mass,
                    "pending_scripts":c.pending_scripts,"resolved_edges":c.resolved_edges,
                    "unique_children":c.unique_children,"expanded_children":c.expanded_children,
                    "child_uncertainty":c.child_uncertainty,"largest_child_uncertainty":c.largest_child_uncertainty,
                    "max_child_runs":c.max_child_runs,"total_child_runs":c.total_child_runs,
                    "largest_children":c.largest_children.into_iter().map(|child|json!({
                        "mass":child.mass,"uncertainty":child.uncertainty,"runs":child.runs,
                        "value":[child.bounds.lo,child.bounds.hi],"visits":child.visits,
                        "actions":child.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|a.to_input(&dex))).collect::<Vec<_>>()),
                        "turn":child.battle.as_ref().map(|b|b.turn),"cells":child.cells.iter().map(|b|[b.lo,b.hi]).collect::<Vec<_>>()
                    })).collect::<Vec<_>>()
                })).collect();
                let matrix = solver.root_payoffs(&battle).unwrap();
                let stats = &solver.stats;
                let mut record = json!({"game":game.game,"decision":index,"side":request.side,"state_key":decision.key,
                    "work_budget":work_budget,"node_budget":node_budget,"trial_depth":trial_depth,
                    "call_budget":call_budget,"cell_cap":cell_cap,
                    "retain_eager_prefix":retain_eager_prefix,
                    "defer_eager_frontier":defer_eager_frontier,
                    "milliseconds":milliseconds,"runs":stats.runs,"stop":format!("{:?}",report.stop),
                    "value":[report.bounds.lo,report.bounds.hi],"max_loss":report.strategies.map(|p|p.map(|p|p.max_loss)),
                    "actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|a.to_input(&dex))).collect::<Vec<_>>()),
                    "cells":matrix.cells.iter().map(|b|[b.lo,b.hi]).collect::<Vec<_>>(),"chance":chance,
                    "graph":{"nodes":solver.node_count(),"closed":solver.closed_count(),
                        "unexpanded":graph.unexpanded_nodes,"initialized":graph.initialized_nodes,"stored_battles":graph.stored_battles,
                        "edges":graph.resolved_edges,"unique_edges":graph.unique_child_edges,
                        "pending_scripts":graph.pending_scripts,"pending_script_entries":graph.pending_script_entries},
                    "stats":{"trials":stats.trials,"depth_limit_trials":stats.depth_limit_trials,
                        "depth_counts":stats.trial_depth_counts,"eager_runs":stats.eager_runs,
                        "eager_failures":stats.eager_failures,"expansions":stats.expansions,
                        "retained_eager_runs":stats.retained_eager_runs,"retained_eager_leaves":stats.retained_eager_leaves,
                        "deferred_eager_leaves":stats.deferred_eager_leaves,
                        "reused_nodes":stats.reused_nodes,"lp_solves":stats.lp_solves,
                        "dominated_rows":stats.dominated_rows,"dominated_cols":stats.dominated_cols}});
                if profile_pending_selection {
                    record["pending_profile"]=json!({"nanoseconds":stats.pending_selection_ns,
                        "calls":stats.pending_selection_calls,"scanned_items":stats.pending_scan_items});
                }
                writeln!(output,"{record}").unwrap(); output.flush().unwrap();
                eprintln!("game={} decision={index} side={} runs={} stop={:?}",game.game,request.side,stats.runs,report.stop);
            }
            if index<last { battle.apply_choices(&dex,[0,1].map(|s|decision.choices[s].map(|i|legal[s][i]))).unwrap(); }
        }
    }
}
