use std::{io::Write, time::Instant};
use nc2000_bot::{bounds::{BoundConfig, FixedPolicySolver}, player::action_input};
use nc2000_engine::{battle::SearchChoice, dex::Dex, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn main() {
    let args: Vec<_> = std::env::args().collect(); assert_eq!(args.len(), 5);
    let protocol: Value = serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let roots_bytes = std::fs::read(&args[2]).unwrap(); let dex_bytes = std::fs::read(&args[3]).unwrap();
    assert_eq!(protocol["roots_sha256"], sha(&roots_bytes)); assert_eq!(protocol["dex_sha256"], sha(&dex_bytes));
    let roots: Vec<Value> = std::str::from_utf8(&roots_bytes).unwrap().lines().map(|s|serde_json::from_str(s).unwrap()).collect();
    assert_eq!(roots.len(), 360);
    let input=roots.iter().find(|r|r["record"]["game"]==87 && r["record"]["frame"]==49).unwrap();
    let record=&input["record"];let bytes=serde_json::to_vec(&input["battle"]).unwrap();
    assert_eq!(record["state_sha256"],sha(&bytes));
    let mut battle:Battle=serde_json::from_slice(&bytes).unwrap();
    assert_eq!(record["state_key"],battle.state_key());assert!(battle.is_idle_move_request());
    let dex=Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).unwrap();
    let side=record["side"].as_u64().unwrap() as usize;
    let legal=[0,1].map(|s|battle.legal_choices(&dex,s));
    let actions=legal.each_ref().map(|aa|aa.iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>());
    assert_eq!(record["legal_actions"],json!(actions[side]));assert_eq!(record["opponent_legal_actions"],json!(actions[1-side]));
    let mut out=std::io::BufWriter::new(std::fs::File::create_new(&args[4]).unwrap());
    for job in protocol["jobs"].as_array().unwrap() {
        let lower=job["direction"]=="lower";assert!(lower || job["direction"]=="upper");
        let fixed_side=if lower {side} else {1-side};
        let move_choice=|s:&str|SearchChoice::Move(dex.moves.id(s).unwrap());
        let first=move_choice(job["first"].as_str().unwrap());
        let priority=job["priority"].as_array().unwrap().iter().map(|s|move_choice(s.as_str().unwrap())).collect();
        let cfg=BoundConfig {work_budget:protocol["work_budget"].as_u64().unwrap() as usize,
            node_budget:protocol["node_budget"].as_u64().unwrap() as usize,cell_cap:4096,eps:1e-6,
            dead_quick_claw_quotient:true,retain_eager_prefix:job["defer"].as_bool().unwrap(),
            defer_eager_frontier:job["defer"].as_bool().unwrap(),..Default::default()};let config=format!("{cfg:?}");
        let mut solver=FixedPolicySolver::new_with_first_action(&dex,cfg,fixed_side,priority,&battle,first);
        if job["sleep_talk"].as_bool().unwrap() { solver=solver.with_sleep_talk(); }
        let start=Instant::now();
        let (score,stop,runs)=if lower {
            let r=solver.solve(&battle);([r.guaranteed_score.lo,r.guaranteed_score.hi],format!("{:?}",r.stop),r.runs)
        } else {
            let r=solver.solve_reply(&battle,move_choice(job["target"].as_str().unwrap()),None);
            let b=if side==0 {[r.bounds.lo,r.bounds.hi]} else {[1.-r.bounds.hi,1.-r.bounds.lo]};(b,format!("{:?}",r.stop),r.runs)
        };
        let matrix=solver.restricted_root_payoffs(&battle).unwrap();
        assert_eq!(matrix.actions[fixed_side],vec![Some(first)]);
        assert_eq!(matrix.actions[1-fixed_side],legal[1-fixed_side].iter().copied().map(Some).collect::<Vec<_>>());
        let row=json!({"id":job["id"],"game":87,"frame":49,"state_sha256":record["state_sha256"],"side":side,
            "actual_action":record["current_response"]["action"],"job":job,"config":config,"fixed_side":fixed_side,
            "actions":actions,"restricted_actions":matrix.actions.each_ref().map(|aa|aa.iter().map(|a|a.map(|a|action_input(&dex,a))).collect::<Vec<_>>()),
            "restricted_cells_p0":matrix.cells.iter().map(|b|[b.lo,b.hi]).collect::<Vec<_>>(),
            "restricted_value_p0":[matrix.value.lo,matrix.value.hi],"target_score_interval":score,
            "transfer_direction":if lower {"lower"} else {"upper"},"transfer_bound":score[usize::from(!lower)],
            "runs":runs,"total_runs":solver.stats().runs,"nodes":solver.node_count(),"trials":solver.stats().trials,
            "deferred_eager_leaves":solver.stats().deferred_eager_leaves,"retained_eager_runs":solver.stats().retained_eager_runs,
            "stop":stop,"elapsed_ms":start.elapsed().as_secs_f64()*1000.});
        writeln!(out,"{row}").unwrap();out.flush().unwrap();
        eprintln!("{} {}={:.12} runs={} nodes={} stop={}",job["id"],row["transfer_direction"],row["transfer_bound"].as_f64().unwrap(),runs,solver.node_count(),stop);
    }
}
