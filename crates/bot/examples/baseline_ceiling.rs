use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>;2] }
#[derive(Deserialize)]
struct Game { game: usize, seed: String, teams: [Vec<PokemonSet>;2], decisions: Vec<Decision> }
#[derive(Deserialize)]
struct Request { game: usize, decision: usize, side: usize, baseline_choice: String }

fn main() {
    let args:Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),5,"corpus requests output milliseconds");
    let budget=Duration::from_millis(args[4].parse().unwrap());
    assert!(!budget.is_zero());
    let requests:Vec<Request>=serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let dex=conformance::load_dex();
    let input=std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap());
    let mut output=std::fs::File::create_new(&args[3]).unwrap();
    for line in input.lines() {
        let game:Game=serde_json::from_str(&line.unwrap()).unwrap();
        let Some(last)=requests.iter().filter(|r|r.game==game.game).map(|r|r.decision).max() else {continue;};
        let mut battle=Battle::from_fixture(&dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();
        for (index,decision) in game.decisions.iter().enumerate().take(last+1) {
            assert_eq!(battle.state_key128().to_string(),decision.key);
            let legal=[0,1].map(|s|battle.legal_choices(&dex,s));
            if let Some(request)=requests.iter().find(|r|r.game==game.game && r.decision==index) {
                let side=request.side;
                let baseline:Vec<_>=legal[side].iter().map(|a|f64::from(a.to_input(&dex)==request.baseline_choice)).collect();
                assert_eq!(baseline.iter().sum::<f64>(),1.);
                let started=Instant::now();
                let mut solver=BoundSolver::new(&dex,BoundConfig {
                    work_budget:1000,node_budget:120000,trial_depth:24,cell_cap:4096,
                    resource_closure_nodes:Some(9000),eps:1e-10,..Default::default()
                });
                let mut calls=Vec::new();
                let mut witness_fallbacks=0;
                let (reason,model,candidate_bounds,baseline_bounds,baseline_loss,matrix)=loop {
                    solver.cfg.time_budget=Some(budget.saturating_sub(started.elapsed()));
                    let report=solver.solve_policy_losses(&battle,&[(side,baseline.clone())],0.005);
                    let matrix=solver.root_payoffs(&battle).unwrap();
                    let baseline_loss=matrix.policy_loss(side,&baseline);
                    let model=solver.root_correlation(&battle).unwrap_or_else(|| {
                        witness_fallbacks+=1;
                        let mut policies=[0,1].map(|s|vec![1./matrix.actions[s].len() as f64;matrix.actions[s].len()]);
                        policies[side]=baseline.clone();
                        solver.root_correlation_for_policies(&battle,policies).unwrap()
                    });
                    let candidate_bounds=model.loss_bounds();
                    let mut baseline_model=model.clone();
                    baseline_model.policies[side]=baseline.clone();
                    baseline_model.independent_loss[side]=baseline_loss.hi;
                    let baseline_bounds=baseline_model.loss_bounds();
                    let elapsed=started.elapsed();
                    calls.push(json!({"milliseconds":elapsed.as_secs_f64()*1000.,"runs":solver.stats.runs,
                        "call_runs":report.runs,"stop":format!("{:?}",report.stop),"value":[report.bounds.lo,report.bounds.hi],
                        "baseline_loss":[baseline_loss.lo,baseline_bounds.max_loss[side]],
                        "candidate_max_loss":candidate_bounds.max_loss[side]}));
                    let reason=if candidate_bounds.max_loss[side]+1e-8<baseline_loss.lo {Some("ImprovementProven")}
                        else if baseline_bounds.max_loss[side]<=0.005 {Some("BaselineCertified")}
                        else if baseline_loss.lo>0.005+1e-8 {Some("BaselineError")}
                        else if elapsed>=budget {Some("TimeBudget")}
                        else if report.runs==0 {Some("NoProgress")}
                        else if report.stop==nc2000_bot::bounds::Stop::NodeBudget {Some("NodeBudget")}
                        else if baseline_loss.width()<=0.005 {Some("PolicyIntervalTight")}
                        else {None};
                    if let Some(reason)=reason {break (reason,model,candidate_bounds,baseline_bounds,baseline_loss,matrix);}
                };
                let graph=solver.graph_diagnostics();
                let record=json!({"game":game.game,"decision":index,"side":side,"state_key":decision.key,
                    "budget_ms":budget.as_millis(),"node_budget":120000,"call_budget":1000,"resource_closure_nodes":9000,
                    "witness_fallbacks":witness_fallbacks,"baseline_choice":request.baseline_choice,"baseline_policy":baseline,"reason":reason,"calls":calls,
                    "actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|a.to_input(&dex))).collect::<Vec<_>>()),
                    "value":[matrix.value.lo,matrix.value.hi],"baseline_independent_loss":[baseline_loss.lo,baseline_loss.hi],
                    "model":model,"candidate_bounds":candidate_bounds,"baseline_bounds":baseline_bounds,
                    "nodes":solver.node_count(),"closed":solver.closed_count(),"unexpanded_nodes":graph.unexpanded_nodes});
                writeln!(output,"{record}").unwrap();output.flush().unwrap();
                eprintln!("game={} decision={index} side={side} reason={reason} runs={} milliseconds={:.1}",game.game,solver.stats.runs,started.elapsed().as_secs_f64()*1000.);
            }
            if index<last {battle.apply_choices(&dex,[0,1].map(|s|decision.choices[s].map(|i|legal[s][i]))).unwrap();}
        }
    }
}
