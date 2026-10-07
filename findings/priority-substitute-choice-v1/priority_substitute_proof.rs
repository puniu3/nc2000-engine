use conformance::load_dex;
use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use serde_json::{json, Value};

mod fixture {
    include!("priority_substitute_choice.rs");
    pub fn root(dex: &Dex, side: usize) -> Battle { position(dex, side).1 }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(),2);
    let p: Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let dex=load_dex();
    for side in 0..2 {
        let mut b=fixture::root(&dex,side);
        let legal=[0,1].map(|s|b.legal_choices(&dex,s));
        let policies:Vec<_>=(0..legal[side].len()).map(|i|(side,
            (0..legal[side].len()).map(|j|f64::from(i==j)).collect::<Vec<_>>())).collect();
        let mut solver=BoundSolver::new(&dex,BoundConfig {
            work_budget:p["budgets"]["proof_engine_runs"].as_u64().unwrap() as usize,
            node_budget:p["budgets"]["proof_nodes"].as_u64().unwrap() as usize,
            cell_cap:p["budgets"]["first_step_cell_cap"].as_u64().unwrap() as usize,
            eps:p["budgets"]["proof_loss_target"].as_f64().unwrap(),
            ..Default::default()
        });
        let report=solver.solve_policy_losses(&b,&policies,p["budgets"]["proof_loss_target"].as_f64().unwrap());
        let m=solver.root_payoffs(&b).unwrap();
        let values:Vec<_>=policies.iter().map(|(s,p)|{let v=m.policy_value(*s,p);[v.lo,v.hi]}).collect();
        let losses:Vec<_>=policies.iter().map(|(s,p)|{let v=m.policy_loss(*s,p);[v.lo,v.hi]}).collect();
        println!("{}",json!({"side":side,"key":b.state_key128().to_string(),"stop":format!("{:?}",report.stop),
            "runs":report.runs,"nodes":solver.node_count(),"closed":solver.closed_count(),
            "value_p0":[report.bounds.lo,report.bounds.hi],
            "actions":legal.map(|cs|cs.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),
            "cells_p0":m.cells.iter().map(|v|[v.lo,v.hi]).collect::<Vec<_>>(),
            "pure_values":values,"pure_losses":losses}));
    }
}
