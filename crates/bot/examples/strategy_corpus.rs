use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_bot::{RmConfig, SkuctSearch};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde::Deserialize;
use serde_json::json;
use std::io::{BufRead, Write};
use std::time::{Duration, Instant};

#[derive(Deserialize)]
struct Decision { key: String, choices: [Option<usize>; 2] }

#[derive(Deserialize)]
struct Game {
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    decisions: Vec<Decision>,
    samples: Vec<usize>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus output milliseconds");
    let budget = Duration::from_millis(args[3].parse().unwrap());
    assert!(!budget.is_zero());
    let dex = conformance::load_dex();
    let input = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap());
    let mut output = std::fs::File::create_new(&args[2]).unwrap();
    for line in input.lines() {
        let game: Game = serde_json::from_str(&line.unwrap()).unwrap();
        let Some(&last) = game.samples.iter().max() else { continue; };
        let mut battle = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        for (index, decision) in game.decisions.iter().enumerate().take(last + 1) {
            assert_eq!(battle.state_key128().to_string(), decision.key);
            let legal = [0, 1].map(|s| battle.legal_choices(&dex, s));
            if game.samples.contains(&index) && legal.iter().any(|a| a.len() > 1) {
                let seed = 2026111201 ^ (game.game as u64).wrapping_mul(0x94d049bb133111eb) ^ index as u64;
                let started = Instant::now();
                let mut search = SkuctSearch::new(&battle, &dex, RmConfig { c: 0.4, ..Default::default() }, seed);
                while started.elapsed() < budget { search.step(&dex, 32); }
                let search_ms = started.elapsed().as_secs_f64() * 1000.;
                for side in 0..2 {
                    if legal[side].len() <= 1 { continue; }
                    let chosen = search.best(side).unwrap();
                    let started = Instant::now();
                    let mut solver = BoundSolver::new(&dex, BoundConfig {
                        eps: 1e-10, work_budget: 1000, node_budget: 12000,
                        ..Default::default()
                    });
                    let mut calls = Vec::new();
                    loop {
                        solver.cfg.time_budget = Some(budget.saturating_sub(started.elapsed()));
                        let report = solver.solve_strategy(&battle, side, 0.005);
                        let certificate = report.strategies.as_ref().map(|p| {
                            let cert = &p[side];
                            let policy: Vec<_> = legal[side].iter().map(|a| cert.actions.iter()
                                .find(|(action, _)| *action == Some(*a)).map_or(0., |(_, mass)| *mass)).collect();
                            json!({"policy":policy,"max_loss":cert.max_loss,"guaranteed_value":cert.guaranteed_value})
                        });
                        let elapsed = started.elapsed();
                        calls.push(json!({"milliseconds":elapsed.as_secs_f64()*1000.,"runs":solver.stats.runs,
                            "call_runs":report.runs,"stop":format!("{:?}",report.stop),
                            "value":[report.bounds.lo,report.bounds.hi],"certificate":certificate}));
                        if report.strategy(side, 0.005).is_some() || report.runs == 0 || elapsed >= budget { break; }
                    }
                    let matrix = solver.root_payoffs(&battle).map(|m| json!({
                        "actions":m.actions.map(|aa| aa.into_iter().map(|a| a.map(|a|a.to_input(&dex))).collect::<Vec<_>>()),
                        "value":{"lo":m.value.lo,"hi":m.value.hi},
                        "cells":m.cells.iter().map(|b|json!({"lo":b.lo,"hi":b.hi})).collect::<Vec<_>>()
                    }));
                    let record = json!({"game":game.game,"decision":index,"side":side,"state_key":decision.key,
                        "budget_ms":budget.as_millis(),"actions":legal[side].iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),
                        "baseline":{"seed":seed,"milliseconds":search_ms,"iterations":search.iterations(),"choice":chosen.to_input(&dex)},
                        "calls":calls,"matrix":matrix,"nodes":solver.node_count(),"closed":solver.closed_count()});
                    writeln!(output, "{record}").unwrap();
                    output.flush().unwrap();
                    eprintln!("game={} decision={index} side={side} runs={}",game.game,solver.stats.runs);
                }
            }
            if index < last {
                battle.apply_choices(&dex, [0,1].map(|s|decision.choices[s].map(|i|legal[s][i]))).unwrap();
            }
        }
    }
}
