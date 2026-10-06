use nc2000_bot::bounds::{BoundConfig, BoundSolver, Stop};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }
#[derive(Deserialize)]
struct Game { game: usize, seed: String, teams: [Vec<PokemonSet>; 2], decisions: Vec<Decision> }
#[derive(Deserialize, Serialize)]
struct Target { side: usize, choice: String }
#[derive(Deserialize, Serialize)]
struct Request { game: usize, decision: usize, targets: Vec<Target> }

fn canonical(action: String) -> String {
    if action.starts_with("move hiddenpower") { "move hiddenpower".to_owned() } else { action }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "corpus requests output milliseconds");
    let budget = Duration::from_millis(args[4].parse().unwrap());
    assert!(!budget.is_zero());
    let requests: Vec<Request> = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let dex = conformance::load_dex();
    let input = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap());
    let mut output = std::fs::File::create_new(&args[3]).unwrap();
    for line in input.lines() {
        let game: Game = serde_json::from_str(&line.unwrap()).unwrap();
        let Some(last) = requests.iter().filter(|r|r.game == game.game).map(|r|r.decision).max() else { continue; };
        let mut battle = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        for (index, decision) in game.decisions.iter().enumerate().take(last + 1) {
            assert_eq!(battle.state_key128().to_string(), decision.key);
            let legal = [0, 1].map(|s|battle.legal_choices(&dex, s));
            if let Some(request) = requests.iter().find(|r|r.game == game.game && r.decision == index) {
                let targets: Vec<_> = request.targets.iter().map(|t| {
                    let policy: Vec<_> = legal[t.side].iter().map(|a|f64::from(canonical(a.to_input(&dex)) == t.choice)).collect();
                    assert_eq!(policy.iter().sum::<f64>(), 1.);
                    (t.side, policy)
                }).collect();
                let started = Instant::now();
                let mut solver = BoundSolver::new(&dex, BoundConfig {
                    work_budget:1000, node_budget:120000, trial_depth:24, cell_cap:4096,
                    resource_closure_nodes:Some(9000), eps:1e-10, ..Default::default()
                });
                let mut calls = Vec::new();
                let reason = loop {
                    solver.cfg.time_budget = Some(budget.saturating_sub(started.elapsed()));
                    let report = solver.solve_policy_losses(&battle, &targets, 0.005);
                    let matrix = solver.root_payoffs(&battle).unwrap();
                    let guarantees: Vec<_> = targets.iter().map(|(s,p)|matrix.policy_value(*s,p)).collect();
                    calls.push(json!({"milliseconds":started.elapsed().as_secs_f64()*1000.,
                        "runs":solver.stats.runs,"call_runs":report.runs,"stop":format!("{:?}",report.stop),
                        "value":[matrix.value.lo,matrix.value.hi],
                        "guarantees":guarantees.iter().map(|b|[b.lo,b.hi]).collect::<Vec<_>>()}));
                    if guarantees.iter().all(|b|b.width()<=0.005) { break "GuaranteesTight"; }
                    if started.elapsed()>=budget { break "TimeBudget"; }
                    if report.runs==0 { break "NoProgress"; }
                    if report.stop==Stop::NodeBudget { break "NodeBudget"; }
                };
                let matrix = solver.root_payoffs(&battle).unwrap();
                let policies = matrix.actions.each_ref().map(|aa|vec![1./aa.len() as f64; aa.len()]);
                let model = solver.root_correlation_for_policies(&battle, policies).unwrap();
                let graph = solver.graph_diagnostics();
                let chance: Vec<_> = solver.root_chance_diagnostics(&battle).unwrap().iter().map(|c|json!({
                    "pending_mass":c.pending_mass,"child_uncertainty":c.child_uncertainty,
                    "runs":c.runs,"unique_children":c.unique_children,"expanded_children":c.expanded_children
                })).collect();
                let record = json!({"request":request,"game":game.game,"decision":index,"state_key":decision.key,
                    "budget_ms":budget.as_millis(),"reason":reason,"calls":calls,
                    "actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|canonical(a.to_input(&dex)))).collect::<Vec<_>>()),
                    "value":[matrix.value.lo,matrix.value.hi],"model":model,"chance":chance,
                    "nodes":solver.node_count(),"closed":solver.closed_count(),"unexpanded_nodes":graph.unexpanded_nodes,
                    "trials":solver.stats.trials,"depth_limit_trials":solver.stats.depth_limit_trials});
                writeln!(output, "{record}").unwrap(); output.flush().unwrap();
                eprintln!("game={} decision={index} reason={reason} runs={}",game.game,solver.stats.runs);
            }
            if index<last { battle.apply_choices(&dex,[0,1].map(|s|decision.choices[s].map(|i|legal[s][i]))).unwrap(); }
        }
    }
}
