use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::{json, Value};
use std::{
    io::Write,
    time::{Duration, Instant},
};
#[allow(dead_code)]
mod view {
    include!("theoretical_headroom.rs");
    pub fn state(b: &Battle, d: &Dex) -> Value {
        state_view_json(b, d)
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3);
    let input: Vec<Value> =
        serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .unwrap();
    let dex = conformance::load_dex();
    let mut solver = BoundSolver::new(
        &dex,
        BoundConfig {
            time_budget: Some(Duration::from_secs(30)),
            work_budget: 1_000_000,
            eps: 1e-10,
            certified_action_pruning: false,
            fold_closed_nodes: false,
            ..Default::default()
        },
    );
    for case in input {
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(case["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, case["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        let decisions = case["decisions"].as_array().unwrap();
        for d in &decisions[..decisions.len() - 1] {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let a = [0, 1].map(|s| {
                d["choices"][s]
                    .as_u64()
                    .map(|i| b.legal_choices(&dex, s)[i as usize])
            });
            b.apply_choices(&dex, a).unwrap();
        }
        for _ in 0..case["extra_waits"].as_u64().unwrap() {
            b.apply_choices(
                &dex,
                ["curse", "destinybond"]
                    .map(|name| Some(SearchChoice::Move(dex.moves.id(name).unwrap()))),
            )
            .unwrap();
        }
        let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
        let mut targets = Vec::new();
        for side in 0..2 {
            for i in 0..actions[side].len() {
                let mut p = vec![0.; actions[side].len()];
                p[i] = 1.;
                targets.push((side, p));
            }
        }
        let started = Instant::now();
        let before = solver.stats.runs;
        solver.solve_policy_losses(&b, &targets, 1e-8);
        let p = solver.root_payoffs(&b).unwrap();
        writeln!(out,"{}",json!({"id":case["id"],"key":b.state_key128().to_string(),"state":view::state(&b,&dex),"seed":b.prng.seed_str(),
            "actions":actions.map(|a|a.iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>()),"value":[p.value.lo,p.value.hi],
            "cells":p.cells.iter().map(|v|[v.lo,v.hi]).collect::<Vec<_>>(),"runs":solver.stats.runs-before,"seconds":started.elapsed().as_secs_f64()})).unwrap();
        eprintln!("{} complete", case["id"]);
    }
}
