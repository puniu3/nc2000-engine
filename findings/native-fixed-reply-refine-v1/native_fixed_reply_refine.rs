use std::{io::Write, time::Instant};
use nc2000_bot::{bounds::{BoundConfig, FixedPolicySolver}, player::action_input};
use nc2000_engine::{battle::SearchChoice, dex::Dex, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5);
    let protocol: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let roots_bytes = std::fs::read(&args[2]).unwrap();
    let dex_bytes = std::fs::read(&args[3]).unwrap();
    assert_eq!(protocol["roots_sha256"], sha(&roots_bytes));
    assert_eq!(protocol["dex_sha256"], sha(&dex_bytes));
    let roots: Vec<Value> = std::str::from_utf8(&roots_bytes).unwrap().lines()
        .map(|line| serde_json::from_str(line).unwrap()).collect();
    assert_eq!(roots.len(), protocol["source_roots"].as_u64().unwrap() as usize);
    let dex = Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).unwrap();
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(&args[4]).unwrap());
    for job in protocol["jobs"].as_array().unwrap() {
        let input = roots.iter().find(|r| r["record"]["game"] == job["game"]
            && r["record"]["frame"] == job["frame"]).unwrap();
        let record = &input["record"];
        let state_bytes = serde_json::to_vec(&input["battle"]).unwrap();
        assert_eq!(record["state_sha256"], sha(&state_bytes));
        let mut battle: Battle = serde_json::from_slice(&state_bytes).unwrap();
        assert_eq!(record["state_key"], battle.state_key());
        assert!(battle.is_idle_move_request());
        assert!(battle.sides.iter().all(|s| s.pokemon_left == 1));
        let side = record["side"].as_u64().unwrap() as usize;
        let legal = [0, 1].map(|s| battle.legal_choices(&dex, s));
        let actions = legal.each_ref().map(|aa| aa.iter().map(|&a| action_input(&dex, a)).collect::<Vec<_>>());
        assert_eq!(record["legal_actions"], json!(actions[side]));
        assert_eq!(record["opponent_legal_actions"], json!(actions[1-side]));
        let target = actions[side].iter().position(|s| s == job["target"].as_str().unwrap()).unwrap();
        let priority: Vec<SearchChoice> = job["opponent_priority"].as_array().unwrap().iter()
            .map(|v| SearchChoice::Move(dex.moves.id(v.as_str().unwrap()).unwrap())).collect();
        assert!(legal[1-side].contains(&priority[0]));
        let cfg = BoundConfig {
            work_budget: protocol["work_budget"].as_u64().unwrap() as usize,
            node_budget: protocol["node_budget"].as_u64().unwrap() as usize,
            cell_cap: protocol["cell_cap"].as_u64().unwrap() as usize,
            eps: protocol["eps"].as_f64().unwrap(),
            dead_quick_claw_quotient: true,
            ..Default::default()
        };
        let config = format!("{cfg:?}");
        let first = SearchChoice::Move(dex.moves.id(job["opponent_first"].as_str().unwrap()).unwrap());
        let mut solver = FixedPolicySolver::new_with_first_action(&dex, cfg, 1-side, priority, &battle, first);
        let started = Instant::now();
        let result = solver.solve_reply(&battle, legal[side][target], None);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.;
        let matrix = solver.restricted_root_payoffs(&battle).unwrap();
        assert_eq!(matrix.actions[side], legal[side].iter().copied().map(Some).collect::<Vec<_>>());
        assert_eq!(matrix.actions[1-side].len(), 1);
        let cell = matrix.cells[target];
        assert_eq!((cell.lo, cell.hi), (result.bounds.lo, result.bounds.hi));
        let own_bounds = if side == 0 { [cell.lo, cell.hi] } else { [1.-cell.hi, 1.-cell.lo] };
        let r = json!({
            "id":job["id"], "game":record["game"], "frame":record["frame"],
            "state_sha256":record["state_sha256"], "side":side,
            "actual_action":record["current_response"]["action"], "target":job["target"],
            "opponent_priority":job["opponent_priority"], "opponent_first":job["opponent_first"], "actions":actions,
            "restricted_actions":matrix.actions.each_ref().map(|aa|aa.iter().map(|a|a.map(|a|action_input(&dex,a))).collect::<Vec<_>>()),
            "restricted_cells_p0":matrix.cells.iter().map(|b|[b.lo,b.hi]).collect::<Vec<_>>(),
            "target_cell_p0":[cell.lo,cell.hi], "target_score_against_fixed_opponent":own_bounds,
            "unrestricted_target_upper":own_bounds[1],
            "stop":format!("{:?}",result.stop), "runs":result.runs,
            "total_runs":solver.stats().runs,"nodes":solver.node_count(),
            "trials":solver.stats().trials, "elapsed_ms":elapsed_ms,"config":config,
        });
        writeln!(out,"{r}").unwrap();out.flush().unwrap();
        eprintln!("{} upper={:.12} runs={} nodes={} ms={:.0}",job["id"],own_bounds[1],result.runs,solver.node_count(),elapsed_ms);
    }
}
