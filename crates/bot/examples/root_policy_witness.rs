use nc2000_bot::bounds::{BoundConfig, FixedPolicySolver};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    state::Battle,
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::io::BufRead;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    game: usize,
    decision: usize,
    side: usize,
    first: String,
    priority: Vec<String>,
    replies: Vec<String>,
    rounds: usize,
    work_budget: usize,
    node_budget: usize,
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus manifest output");
    let raw = std::fs::read(&args[2]).unwrap();
    let m: Manifest = serde_json::from_slice(&raw).unwrap();
    assert!(m.side < 2 && m.rounds > 0 && m.work_budget > 0 && m.node_budget > 0);
    assert!(!m.priority.is_empty() && !m.replies.is_empty());
    let corpus = std::fs::read(&args[1]).unwrap();
    let dex = conformance::load_dex();
    let game = corpus
        .lines()
        .map(|l| serde_json::from_str::<serde_json::Value>(&l.unwrap()).unwrap())
        .find(|g| g["game"] == m.game)
        .unwrap();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut b =
        Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    for d in game["decisions"]
        .as_array()
        .unwrap()
        .iter()
        .take(m.decision)
    {
        assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
        let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
        b.apply_choices(
            &dex,
            [0, 1].map(|s| d["choices"][s].as_u64().map(|i| legal[s][i as usize])),
        )
        .unwrap();
    }
    assert_eq!(
        b.state_key128().to_string(),
        game["decisions"][m.decision]["key"].as_str().unwrap()
    );
    assert!(b.outcome().is_none() && b.needs_choice() == [true, true]);
    let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
    let first = legal[m.side]
        .iter()
        .copied()
        .find(|a| a.to_input(&dex) == m.first)
        .unwrap();
    let replies: Vec<_> = m
        .replies
        .iter()
        .map(|name| {
            legal[1 - m.side]
                .iter()
                .copied()
                .find(|a| a.to_input(&dex) == *name)
                .unwrap()
        })
        .collect();
    let priority = m
        .priority
        .iter()
        .map(|name| SearchChoice::Move(dex.moves.id(name.strip_prefix("move ").unwrap()).unwrap()))
        .collect();
    let cfg = BoundConfig {
        work_budget: m.work_budget,
        node_budget: m.node_budget,
        eps: 1e-6,
        dead_quick_claw_quotient: true,
        retain_eager_prefix: true,
        aggregate_lazy_children: true,
        chance_fair_period: 8,
        ..BoundConfig::default()
    };
    let mut solver =
        FixedPolicySolver::new_with_first_action(&dex, cfg, m.side, priority, &b, first);
    let mut rounds = Vec::new();
    for i in 0..m.rounds {
        let reply = replies[i % replies.len()];
        let report = solver.solve_reply(&b, reply, None);
        let matrix = solver.restricted_root_payoffs(&b).unwrap();
        assert_eq!(matrix.actions[m.side], vec![Some(first)]);
        assert_eq!(
            matrix.actions[1 - m.side],
            legal[1 - m.side]
                .iter()
                .copied()
                .map(Some)
                .collect::<Vec<_>>()
        );
        let bounds = if m.side == 0 {
            [matrix.value.lo, matrix.value.hi]
        } else {
            [1.0 - matrix.value.hi, 1.0 - matrix.value.lo]
        };
        let cells: Vec<_> = matrix.actions[1 - m.side].iter().zip(&matrix.cells)
            .map(|(a, v)| json!({"reply": a.unwrap().to_input(&dex), "side_zero_bounds": [v.lo, v.hi]})).collect();
        rounds.push(json!({"reply": reply.to_input(&dex), "runs": report.runs,
            "total_runs": solver.stats().runs, "fixed_side_bounds": bounds,
            "reply_side_zero_bounds": [report.bounds.lo, report.bounds.hi],
            "stop": format!("{:?}", report.stop), "nodes": solver.node_count(), "root_cells": cells}));
        eprintln!(
            "round={} reply={} runs={} bounds={bounds:?}",
            i + 1,
            reply.to_input(&dex),
            report.runs
        );
    }
    let output = json!({"manifest_sha256": format!("{:x}", Sha256::digest(raw)),
        "corpus_sha256": format!("{:x}", Sha256::digest(corpus)),
        "state_key": b.state_key128().to_string(), "state": b.essence(&dex),
        "legal": legal.map(|aa| aa.into_iter().map(|a| a.to_input(&dex)).collect::<Vec<_>>()),
        "rounds": rounds});
    std::fs::write(&args[3], serde_json::to_vec(&output).unwrap()).unwrap();
}
