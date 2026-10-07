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
    let mut checked=0;let mut runs=0;let mut policy_runs=0;let mut max_error=0.0f64;
    for sample in r["frontier_probe"]["samples"].as_array().unwrap(){
        let raw=sample["battle_json"].as_str().unwrap();assert_eq!(sample["state_sha256"],sha(raw.as_bytes()));
        let b:Battle=serde_json::from_str(raw).unwrap();assert!(b.is_idle_move_request());
        assert_eq!(sample["state_key128"],format!("{:032x}",b.state_key128()));
        let priority=job["priority"].as_array().unwrap().iter().map(|m|SearchChoice::Move(dex.moves.id(m.as_str().unwrap()).unwrap())).collect();
        let mut policy=FixedPolicySolver::new(&dex,BoundConfig {work_budget:1,cell_cap:0,..Default::default()},side,priority).with_sleep_talk_until_wake();
        policy_runs+=policy.solve(&b).runs;let matrix=policy.restricted_root_payoffs(&b).unwrap();
        assert_eq!(matrix.actions[side].len(),1);let own=matrix.actions[side][0].unwrap();
        assert_eq!(sample["fixed_action"],action_input(&dex,own));
        let cells=sample["cells"].as_array().unwrap();assert_eq!(cells.len(),matrix.actions[1-side].len());
        for (saved,reply) in cells.iter().zip(&matrix.actions[1-side]){
            let reply=reply.unwrap();assert_eq!(saved["reply"],action_input(&dex,reply));
            let joint=if side==0 {[own,reply]} else {[reply,own]};
            let step=enumerate_step_partial(&dex,&b,joint.map(Some),p["probe_cell_cap"].as_u64().unwrap() as usize);
            let mut mass=[0.;4];for leaf in &step.leaves{
                let k=match leaf.battle.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3};mass[k]+=leaf.prob;
            }
            let pending=step.pending.iter().map(|x|x.mass).sum::<f64>();
            for (expected,actual) in saved["mass"].as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).zip(mass){max_error=max_error.max((expected-actual).abs());}
            max_error=max_error.max((saved["pending_mass"].as_f64().unwrap()-pending).abs());
            assert_eq!(saved["runs"],step.runs);assert_eq!(saved["exhausted"],step.exhausted);runs+=step.runs;checked+=1;
        }
    }
    assert!(max_error<1e-10);
    println!("{}",json!({"cells_checked":checked,"extra_engine_runs":runs,"policy_validation_runs":policy_runs,
        "max_probability_error":max_error,"records_sha256":sha(&std::fs::read(&a[3]).unwrap()),"performance_sample":false}));
}
