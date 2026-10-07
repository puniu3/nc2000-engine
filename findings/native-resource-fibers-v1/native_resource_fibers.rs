use std::io::{BufRead, BufReader, Write};
use std::sync::{atomic::{AtomicUsize, Ordering}, Mutex};
use std::time::Instant;

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Bounds};
use nc2000_bot::player::action_input;
use nc2000_engine::{dex::Dex, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn interval(b: Bounds) -> Value { json!([b.lo,b.hi]) }


fn projection_key(dex: &Dex, b: &Battle, mask: u8) -> (u8,u128) {
    let damage = !b.damage_bookkeeping_observable(dex);
    let claw = !b.quick_claw_roll_observable(dex);
    let mut projected = b.clone();
    for side in 0..2 {
        if let Some(id) = projected.active_id(side) {
            let pokemon = projected.poke_mut(id);
            if mask & 1 != 0 && pokemon.hp > 0 { pokemon.hp = 1; }
            if mask & 2 != 0 {
                for slot in &mut pokemon.move_slots { slot.pp = 0; }
            }
        }
    }
    if mask & 4 != 0 { projected.turn = 1; }
    let hash = if claw { projected.state_key128_without_quick_claw_roll(dex,damage) }
        else if damage { projected.state_key128_without_damage_bookkeeping() }
        else { projected.state_key128() };
    (u8::from(damage) | u8::from(claw)<<1,hash)
}

fn diagnostics(dex: &Dex, solver: &BoundSolver<'_>, battle: &Battle) -> Value {
    let graph=solver.graph_diagnostics();
    let retained:Vec<_>=solver.retained_battles().collect();
    let projections:Vec<_>=(0..8).map(|mask| {
        let mut groups=std::collections::BTreeMap::<(u8,u128),usize>::new();
        for b in &retained { *groups.entry(projection_key(dex,b,mask)).or_default()+=1; }
        let mut histogram=std::collections::BTreeMap::<usize,usize>::new();
        for &n in groups.values() {*histogram.entry(n).or_default()+=1;}
        json!({"mask":mask,"groups":groups.len(),"histogram":histogram})
    }).collect();
    let chance:Vec<_>=solver.root_chance_diagnostics(battle).unwrap().into_iter().map(|c|json!({
        "pending_mass":c.pending_mass,"child_uncertainty":c.child_uncertainty,
        "resolved_edges":c.resolved_edges,"unique_children":c.unique_children,
        "expanded_children":c.expanded_children,"largest_child_uncertainty":c.largest_child_uncertainty,
        "runs":c.runs,"visits":c.visits})).collect();
    json!({"retained":retained.len(),"projections":projections,"root_chance":chance,
        "graph":{"unexpanded_nodes":graph.unexpanded_nodes,"initialized_nodes":graph.initialized_nodes,
            "stored_battles":graph.stored_battles,"resolved_edges":graph.resolved_edges,
            "unique_child_edges":graph.unique_child_edges,"pending_scripts":graph.pending_scripts},
        "trial_depth_counts":solver.stats.trial_depth_counts,"depth_limit_trials":solver.stats.depth_limit_trials,
        "resource_closure_visits":solver.stats.resource_closure_visits,
        "monotone_roots":solver.stats.monotone_roots,"two_sided_roots":solver.stats.two_sided_roots,
        "monotone_invalidations":solver.stats.monotone_invalidations,
        "two_sided_invalidations":solver.stats.two_sided_invalidations})
}

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
    let target_action=protocol["action_overrides"][format!("{}/{}",record["game"],record["frame"])].as_str()
        .unwrap_or_else(||record["current_response"]["action"].as_str().unwrap());
    let target=actions[side].iter().position(|a|a==target_action).unwrap();
    let tolerance=protocol["tolerance"].as_f64().unwrap();
    let cfg=BoundConfig {
        work_budget:protocol["work_budget"].as_u64().unwrap() as usize,
        node_budget:protocol["node_budget"].as_u64().unwrap() as usize,
        cell_cap:protocol["cell_cap"].as_u64().unwrap() as usize,
        eps:tolerance,
        dead_quick_claw_quotient:protocol["dead_quick_claw_quotient"].as_bool().unwrap(),
        retain_eager_prefix:protocol["retain_eager_prefix"].as_bool().unwrap(),
        ..Default::default()
    };
    let config_debug=format!("{:?}",cfg);
    let mut solver=BoundSolver::new(dex,cfg);
    let policies:Vec<_>=(0..legal[side].len()).map(|i|
        (side,(0..legal[side].len()).map(|j|f64::from(i==j)).collect::<Vec<_>>())).collect();
    let targets=if protocol["selected_only"].as_bool().unwrap() {vec![policies[selected].clone()]} else {policies.clone()};
    let started=Instant::now();
    let (value,stop,runs,threshold_loss)=if protocol["mode"]=="loss_lower" {
        let report=solver.solve_policy_loss_threshold(&battle,side,&policies[target].1,tolerance);
        (report.value_p0,format!("{:?}",report.stop),report.runs,Some(interval(report.policy_loss)))
    } else {
        assert_eq!(protocol["mode"],"legacy");
        let report=solver.solve_policy_losses(&battle,&targets,tolerance);
        (report.bounds,format!("{:?}",report.stop),report.runs,None)
    };
    let matrix=solver.root_payoffs(&battle).unwrap();
    for s in 0..2 {
        assert_eq!(matrix.actions[s],legal[s].iter().copied().map(Some).collect::<Vec<_>>());
    }
    let values:Vec<_>=policies.iter().map(|(s,p)|interval(matrix.policy_value(*s,p))).collect();
    let losses:Vec<_>=policies.iter().map(|(s,p)|interval(matrix.policy_loss(*s,p))).collect();
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
    json!({"game":record["game"],"frame":record["frame"],"turn":record["turn"],"side":side,
        "state_sha256":record["state_sha256"],"selected":selected,"actions":actions,
        "value_p0":interval(value),"cells_p0":matrix.cells.iter().map(|&b|interval(b)).collect::<Vec<_>>(),
        "pure_action_values":values,"pure_action_losses":losses,"domination":domination,
        "stop":stop,"runs":runs,"mode":protocol["mode"],"target":target,"threshold_loss":threshold_loss,"total_runs":solver.stats.runs,
        "nodes":solver.node_count(),"closed":solver.closed_count(),"elapsed_ms":started.elapsed().as_secs_f64()*1000.,
        "config":config_debug,"diagnostics":diagnostics(dex,&solver,&battle),"stats":{"eager_runs":solver.stats.eager_runs,"eager_failures":solver.stats.eager_failures,
            "retained_eager_runs":solver.stats.retained_eager_runs,"retained_eager_leaves":solver.stats.retained_eager_leaves,
            "expansions":solver.stats.expansions,"trials":solver.stats.trials,"reused_nodes":solver.stats.reused_nodes}})
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
