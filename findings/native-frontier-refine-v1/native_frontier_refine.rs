use nc2000_bot::{bounds::{BoundConfig,FixedPolicySolver},player::action_input};
use nc2000_engine::{battle::{SearchChoice,Outcome,enumerate::enumerate_step_partial},dex::Dex,state::Battle};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn sha(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
fn main(){
    let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),4);
    let p:Value=serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
    let d=std::fs::read(&a[2]).unwrap();assert_eq!(p["dex_sha256"],sha(&d));
    let dex=Dex::from_json(std::str::from_utf8(&d).unwrap()).unwrap();
    let r:Value=serde_json::from_slice(&std::fs::read(&a[3]).unwrap()).unwrap();
    let side=r["side"].as_u64().unwrap() as usize;let job=&p["jobs"][0];
    assert_eq!(p["input_records_sha256"],sha(&std::fs::read(&a[3]).unwrap()));
    let mut refined=r.clone();let mut runs=0;let mut policy_runs=0;let mut samples=Vec::new();
    for sample in r["frontier_probe"]["samples"].as_array().unwrap(){
        let raw=sample["battle_json"].as_str().unwrap();assert_eq!(sample["state_sha256"],sha(raw.as_bytes()));
        let b:Battle=serde_json::from_str(raw).unwrap();assert!(b.is_idle_move_request());
        assert_eq!(sample["state_key128"],format!("{:032x}",b.state_key128()));
        let priority=job["priority"].as_array().unwrap().iter().map(|m|SearchChoice::Move(dex.moves.id(m.as_str().unwrap()).unwrap())).collect();
        let mut policy=FixedPolicySolver::new(&dex,BoundConfig {work_budget:1,cell_cap:0,..Default::default()},side,priority).with_sleep_talk();
        policy_runs+=policy.solve(&b).runs;let matrix=policy.restricted_root_payoffs(&b).unwrap();
        assert_eq!(matrix.actions[side].len(),1);let own=matrix.actions[side][0].unwrap();
        assert_eq!(sample["fixed_action"],action_input(&dex,own));
        let cells=sample["cells"].as_array().unwrap();assert_eq!(cells.len(),matrix.actions[1-side].len());
        let mut result=sample.clone();let mut new_cells=Vec::new();let mut lo=1.0f64;let mut hi=1.0f64;
        for (saved,reply) in cells.iter().zip(&matrix.actions[1-side]){
            let reply=reply.unwrap();assert_eq!(saved["reply"],action_input(&dex,reply));
            let joint=if side==0 {[own,reply]} else {[reply,own]};
            let step=enumerate_step_partial(&dex,&b,joint.map(Some),p["probe_cell_cap"].as_u64().unwrap() as usize);
            let mut mass=[0.;4];for leaf in &step.leaves{
                let k=match leaf.battle.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3};mass[k]+=leaf.prob;
            }
            let pending=step.pending.iter().map(|x|x.mass).sum::<f64>();
            assert!((mass.iter().sum::<f64>()+pending-1.).abs()<1e-9);
            lo=lo.min(mass[side]+0.5*mass[2]);hi=hi.min(1.-mass[1-side]-0.5*mass[2]);
            new_cells.push(json!({"reply":saved["reply"],"mass":mass,"pending_mass":pending,
                "runs":step.runs,"exhausted":step.exhausted}));runs+=step.runs;
        }
        result["cells"]=json!(new_cells);result["one_step_score_bounds"]=json!([lo,hi]);samples.push(result);
    }
    refined["frontier_probe"]["samples"]=json!(samples);
    refined["frontier_probe"]["additional_runs"]=json!(runs);
    eprintln!("refined runs={runs} policy_validation_runs={policy_runs}");
    println!("{refined}");
}
