use nc2000_engine::{battle::{SearchChoice,Outcome,enumerate::{enumerate_step_probability_first,run_scripted}},dex::Dex,state::Battle};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn sha(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),5);
 let p:Value=serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
 let rb=std::fs::read(&a[2]).unwrap();let db=std::fs::read(&a[3]).unwrap();let raw=std::fs::read(&a[4]).unwrap();
 assert_eq!(p["roots_sha256"],sha(&rb));assert_eq!(p["dex_sha256"],sha(&db));
 let input:Value=std::str::from_utf8(&rb).unwrap().lines().map(|l|serde_json::from_str::<Value>(l).unwrap()).find(|r|r["record"]["game"]==62&&r["record"]["frame"]==92).unwrap();
 let state=serde_json::to_vec(&input["battle"]).unwrap();assert_eq!(input["record"]["state_sha256"],sha(&state));
 let mut b:Battle=serde_json::from_slice(&state).unwrap();let dex=Dex::from_json(std::str::from_utf8(&db).unwrap()).unwrap();
 let legal=[0,1].map(|s|b.legal_choices(&dex,s));let mut cells=0;let mut runs=0;let mut replays=0;let mut pending=0;
 for line in std::str::from_utf8(&raw).unwrap().lines(){
  let r:Value=serde_json::from_str(line).unwrap();let c=&r["candidate"];
  assert_eq!(r["state_sha256"],input["record"]["state_sha256"]);
  let choices=["own","reply"].map(|k|Some(SearchChoice::Move(dex.moves.id(r[k].as_str().unwrap().strip_prefix("move ").unwrap()).unwrap())));
  for side in 0..2{assert!(legal[side].contains(&choices[side].unwrap()));}
  let step=enumerate_step_probability_first(&dex,&b,choices,p["cell_cap"].as_u64().unwrap() as usize);
  assert_eq!(c["runs"],step.runs);assert_eq!(c["exhausted"],step.exhausted);
  assert_eq!(c["leaves"].as_array().unwrap().len(),step.leaves.len());assert_eq!(c["pending"].as_array().unwrap().len(),step.pending.len());
  for ((leaf,script),saved) in step.leaves.iter().zip(step.leaf_scripts.as_ref().unwrap()).zip(c["leaves"].as_array().unwrap()){
   assert_eq!(saved["mass"].as_f64().unwrap(),leaf.prob);assert_eq!(saved["script"],json!(script));
   let outcome=match leaf.battle.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3};assert_eq!(saved["outcome"],outcome);
   let (actual,trace)=run_scripted(&dex,&b,choices,script);assert_eq!(actual.state_key128(),leaf.battle.state_key128());assert_eq!(trace.len(),script.len());replays+=1;
  }
  for (prefix,saved) in step.pending.iter().zip(c["pending"].as_array().unwrap()){
   assert_eq!(saved["mass"].as_f64().unwrap(),prefix.mass);assert_eq!(saved["script"],json!(prefix.script));pending+=1;
  }
  runs+=step.runs;cells+=1;
 }
 println!("{}",json!({"cells_checked":cells,"enumeration_runs":runs,"leaf_state_replays":replays,"pending_prefixes_matched":pending,"total_extra_runs":runs+replays,"records_sha256":sha(&raw),"all_outputs_match":true,"performance_sample":false}));
}
