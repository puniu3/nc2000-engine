use nc2000_engine::{battle::{SearchChoice,Outcome,enumerate::run_scripted},dex::Dex,state::Battle};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn sha(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),5);
 let p:Value=serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
 let root_bytes=std::fs::read(&a[2]).unwrap();let db=std::fs::read(&a[3]).unwrap();let raw=std::fs::read(&a[4]).unwrap();
 assert_eq!(p["roots_sha256"],sha(&root_bytes));assert_eq!(p["dex_sha256"],sha(&db));
 let input:Value=std::str::from_utf8(&root_bytes).unwrap().lines().map(|l|serde_json::from_str::<Value>(l).unwrap()).find(|r|r["record"]["game"]==62&&r["record"]["frame"]==92).unwrap();
 let state=serde_json::to_vec(&input["battle"]).unwrap();assert_eq!(input["record"]["state_sha256"],sha(&state));
 let mut b:Battle=serde_json::from_slice(&state).unwrap();let dex=Dex::from_json(std::str::from_utf8(&db).unwrap()).unwrap();
 let legal=[0,1].map(|s|b.legal_choices(&dex,s));let mut checked=0;let mut leaves=0;let mut pending=0;let mut max_error=0.0f64;
 for line in std::str::from_utf8(&raw).unwrap().lines(){
  let r:Value=serde_json::from_str(line).unwrap();assert_eq!(r["state_sha256"],input["record"]["state_sha256"]);
  let choices=["own","reply"].map(|k|Some(SearchChoice::Move(dex.moves.id(r[k].as_str().unwrap().strip_prefix("move ").unwrap()).unwrap())));
  for side in 0..2{assert!(legal[side].contains(&choices[side].unwrap()));}
  for kind in ["leaves","pending"]{for entry in r["candidate"][kind].as_array().unwrap(){
   let script:Vec<_>=entry["script"].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as usize).collect();
   let (next,trace)=run_scripted(&dex,&b,choices,&script);
   assert!(trace.len()>=script.len());assert_eq!(script,trace.iter().take(script.len()).map(|d|d.chosen).collect::<Vec<_>>());
   let mass=trace.iter().take(script.len()).map(|d|d.prob()).product::<f64>();
   max_error=max_error.max((mass-entry["mass"].as_f64().unwrap()).abs());
   if kind=="leaves"{
    assert_eq!(trace.len(),script.len());let outcome=match next.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3};
    assert_eq!(entry["outcome"],outcome);leaves+=1;
   }else{pending+=1;}
  }}checked+=1;
 }
 assert!(max_error<1e-10);
 println!("{}",json!({"cells_checked":checked,"leaf_replays":leaves,"pending_replays":pending,"total_extra_runs":leaves+pending,"max_probability_error":max_error,"records_sha256":sha(&raw),"performance_sample":false}));
}
