use std::io::{BufRead, BufReader, Write};
use std::sync::{atomic::{AtomicUsize, Ordering}, Mutex};
use std::time::Instant;

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Bounds};
use nc2000_bot::player::action_input;
use nc2000_engine::{dex::Dex, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn interval(b: Bounds) -> Value { json!([b.lo,b.hi]) }

fn solve(dex: &Dex, input: &Value, protocol: &Value) -> Value {
    let record = &input["record"];
    let original = serde_json::to_vec(&input["battle"]).unwrap();
    assert_eq!(record["state_sha256"],format!("{:x}",Sha256::digest(&original)));
    let mut battle: Battle = serde_json::from_slice(&original).unwrap();
    assert_eq!(record["state_key"],battle.state_key());
    assert!(battle.is_idle_move_request());
    assert!(battle.sides.iter().all(|s|s.pokemon_left==1));
    let side = record["side"].as_u64().unwrap() as usize;
    let legal = [0,1].map(|s|battle.legal_choices(dex,s));
    let actions = legal.each_ref().map(|aa|aa.iter().map(|&a|action_input(dex,a)).collect::<Vec<_>>());
    assert_eq!(record["legal_actions"],json!(actions[side]));
    assert_eq!(record["opponent_legal_actions"],json!(actions[1-side]));
    let selected = actions[side].iter().position(|a|*a==record["current_response"]["action"]).unwrap();
    let tolerance=protocol["tolerance"].as_f64().unwrap();
    let cfg=BoundConfig {
        work_budget:protocol["work_budget"].as_u64().unwrap() as usize,
        node_budget:protocol["node_budget"].as_u64().unwrap() as usize,
        cell_cap:protocol["cell_cap"].as_u64().unwrap() as usize,
        eps:tolerance,
        ..Default::default()
    };
    let config_debug=format!("{:?}",cfg);
    let mut solver=BoundSolver::new(dex,cfg);
    let targets:Vec<_>=(0..legal[side].len()).map(|i|
        (side,(0..legal[side].len()).map(|j|f64::from(i==j)).collect::<Vec<_>>())).collect();
    let started=Instant::now();
    let report=solver.solve_policy_losses(&battle,&targets,tolerance);
    let matrix=solver.root_payoffs(&battle).unwrap();
    for s in 0..2 {
        assert_eq!(matrix.actions[s],legal[s].iter().copied().map(Some).collect::<Vec<_>>());
    }
    let values:Vec<_>=targets.iter().map(|(s,p)|interval(matrix.policy_value(*s,p))).collect();
    let losses:Vec<_>=targets.iter().map(|(s,p)|interval(matrix.policy_loss(*s,p))).collect();
    let cell=|own:usize,reply:usize| {
        if side==0 {matrix.cells[own*legal[1].len()+reply]}
        else {let b=matrix.cells[reply*legal[1].len()+own];Bounds{lo:1.-b.hi,hi:1.-b.lo}}
    };
    let domination:Vec<_>=(0..legal[side].len()).filter(|&a|a!=selected).map(|a| {
        let gaps:Vec<_>=(0..legal[1-side].len()).map(|reply|{
            let alt=cell(a,reply);let chosen=cell(selected,reply);
            (alt.lo-chosen.hi-1e-10).max(-1.)
        }).collect();
        let lower=gaps.iter().copied().fold(f64::INFINITY,f64::min);
        json!({"action":actions[side][a],"reply_gap_lower":gaps,"minimum_gap_lower":lower})
    }).collect();
    let graph = solver.graph_diagnostics();
    let chance:Vec<_>=solver.root_chance_diagnostics(&battle).unwrap().into_iter().map(|c|json!({
        "visits":c.visits,"runs":c.runs,"pending_mass":c.pending_mass,"pending_scripts":c.pending_scripts,
        "resolved_edges":c.resolved_edges,"unique_children":c.unique_children,"child_uncertainty":c.child_uncertainty,
        "largest_edge_uncertainty":c.largest_edge_uncertainty,"largest_child_uncertainty":c.largest_child_uncertainty,
        "expanded_children":c.expanded_children,"max_child_runs":c.max_child_runs,"total_child_runs":c.total_child_runs,
        "largest_children":c.largest_children.into_iter().map(|child|json!({
            "key":child.key,"mass":child.mass,"bounds":interval(child.bounds),"uncertainty":child.uncertainty,
            "runs":child.runs,"visits":child.visits,"resource_rank":format!("{:?}",child.resource_rank),
            "active":child.battle.as_ref().map(|b|[0,1].map(|s|b.active_id(s).map(|id|{
                let p=b.poke(id);json!({"species":dex.species.key(p.species),"hp":p.hp,"status":p.status.as_str(),
                    "moves":p.move_slots.iter().map(|m|json!({"id":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})}))),
            "actions":child.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
            "cells":child.cells.into_iter().map(interval).collect::<Vec<_>>() })).collect::<Vec<_>>() })).collect();
    let mut retained=0usize;
    let mut omit_claw=0usize;
    let mut projected=std::collections::BTreeMap::<(u8,u128),usize>::new();
    for b in solver.retained_battles() {
        retained+=1;
        let damage=!b.damage_bookkeeping_observable(dex);
        let claw=!b.quick_claw_roll_observable(dex);
        omit_claw+=usize::from(claw);
        let key=if claw {b.state_key128_without_quick_claw_roll(dex,damage)}
            else if damage {b.state_key128_without_damage_bookkeeping()} else {b.state_key128()};
        *projected.entry((u8::from(damage)|u8::from(claw)<<1,key)).or_default()+=1;
    }
    let alias_histogram=projected.values().fold(std::collections::BTreeMap::<usize,usize>::new(),|mut m,&n|{
        *m.entry(n).or_default()+=1;m});
    json!({"game":record["game"],"frame":record["frame"],"turn":record["turn"],"side":side,
        "state_sha256":record["state_sha256"],"selected":selected,"actions":actions,
        "value_p0":interval(report.bounds),"cells_p0":matrix.cells.iter().map(|&b|interval(b)).collect::<Vec<_>>(),
        "pure_action_values":values,"pure_action_losses":losses,"domination":domination,
        "stop":format!("{:?}",report.stop),"runs":report.runs,"total_runs":solver.stats.runs,
        "nodes":solver.node_count(),"closed":solver.closed_count(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.,
        "config":config_debug,"diagnostics":{
            "graph":{"unexpanded_nodes":graph.unexpanded_nodes,"initialized_nodes":graph.initialized_nodes,
                "stored_battles":graph.stored_battles,"resolved_edges":graph.resolved_edges,
                "unique_child_edges":graph.unique_child_edges,"pending_scripts":graph.pending_scripts,
                "pending_script_entries":graph.pending_script_entries},
            "retained_claw_projection":{"retained":retained,"claw_unobservable":omit_claw,
                "projected_keys":projected.len(),"group_size_histogram":alias_histogram},
            "stats":{"trials":solver.stats.trials,"expansions":solver.stats.expansions,
                "eager_runs":solver.stats.eager_runs,"eager_failures":solver.stats.eager_failures,
                "reused_nodes":solver.stats.reused_nodes,"reused_backups":solver.stats.reused_backups,
                "lp_solves":solver.stats.lp_solves,"depth_limit_trials":solver.stats.depth_limit_trials,
                "trial_depth_counts":solver.stats.trial_depth_counts,"closed_folds":solver.stats.closed_folds,
                "dominated_rows":solver.stats.dominated_rows,"dominated_cols":solver.stats.dominated_cols,
                "avoided_cells":solver.stats.avoided_cells,"lazy_edge_merges":solver.stats.lazy_edge_merges},
            "root_chance":chance}})
}

fn main() {
    let args:Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),5,"native_endgame_bounds PROTOCOL ROOTS DEX OUTPUT");
    let protocol:Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let bytes=std::fs::read(&args[2]).unwrap();
    assert_eq!(protocol["roots_sha256"],format!("{:x}",Sha256::digest(&bytes)));
    let dex_bytes=std::fs::read(&args[3]).unwrap();
    assert_eq!(protocol["dex_sha256"],format!("{:x}",Sha256::digest(&dex_bytes)));
    let dex=Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).unwrap();
    let mut roots:Vec<Value>=BufReader::new(bytes.as_slice()).lines().map(|l|serde_json::from_str(&l.unwrap()).unwrap()).collect();
    roots.sort_by_key(|r|(r["record"]["game"].as_u64().unwrap(),r["record"]["frame"].as_u64().unwrap()));
    assert_eq!(roots.len(),protocol["source_roots"].as_u64().unwrap() as usize);
    let indices:Vec<_>=protocol["indices"].as_array().unwrap().iter().map(|i|i.as_u64().unwrap() as usize).collect();
    let output=Mutex::new(std::io::BufWriter::new(std::fs::File::create_new(&args[4]).unwrap()));
    let cursor=AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol["threads"].as_u64().unwrap() {
            scope.spawn(|| loop {
                let i=cursor.fetch_add(1,Ordering::Relaxed);
                if i>=indices.len(){break;}
                let result=solve(&dex,&roots[indices[i]],&protocol);
                let mut out=output.lock().unwrap();
                writeln!(out,"{result}").unwrap();out.flush().unwrap();
                eprintln!("root={}/{} game={} frame={} runs={} ms={:.0}",i+1,indices.len(),
                    result["game"],result["frame"],result["runs"],result["elapsed_ms"].as_f64().unwrap());
            });
        }
    });
}
