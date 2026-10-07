use std::{cmp::Ordering,collections::BinaryHeap,io::Write};
use nc2000_bot::player::action_input;
use nc2000_engine::{battle::{Outcome,enumerate::{enumerate_step_partial,run_scripted}},dex::Dex,state::Battle};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn sha(b:&[u8])->String{format!("{:x}",Sha256::digest(b))}
#[derive(Clone)]
struct Prefix{mass:f64,script:Vec<usize>}
impl PartialEq for Prefix{fn eq(&self,o:&Self)->bool{self.mass==o.mass&&self.script==o.script}}
impl Eq for Prefix{}
impl PartialOrd for Prefix{fn partial_cmp(&self,o:&Self)->Option<Ordering>{Some(self.cmp(o))}}
impl Ord for Prefix{fn cmp(&self,o:&Self)->Ordering{self.mass.total_cmp(&o.mass).then_with(||o.script.cmp(&self.script))}}
fn outcome(b:&Battle)->usize{match b.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3}}
fn main(){
 let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),5);
 let p:Value=serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
 let raw=std::fs::read(&a[2]).unwrap();let db=std::fs::read(&a[3]).unwrap();
 assert_eq!(p["roots_sha256"],sha(&raw));assert_eq!(p["dex_sha256"],sha(&db));
 let input:Value=std::str::from_utf8(&raw).unwrap().lines().map(|l|serde_json::from_str::<Value>(l).unwrap()).find(|r|r["record"]["game"]==62&&r["record"]["frame"]==92).unwrap();
 let r=&input["record"];let state=serde_json::to_vec(&input["battle"]).unwrap();assert_eq!(r["state_sha256"],sha(&state));
 let mut b:Battle=serde_json::from_slice(&state).unwrap();assert_eq!(r["state_key"],b.state_key());
 let dex=Dex::from_json(std::str::from_utf8(&db).unwrap()).unwrap();let choices=[0,1].map(|s|b.legal_choices(&dex,s));
 assert_eq!(r["side"],0);assert_eq!(r["legal_actions"],json!(choices[0].iter().map(|&c|action_input(&dex,c)).collect::<Vec<_>>()));
 assert_eq!(r["opponent_legal_actions"],json!(choices[1].iter().map(|&c|action_input(&dex,c)).collect::<Vec<_>>()));
 let cap=p["cell_cap"].as_u64().unwrap() as usize;
 let mut out=std::io::BufWriter::new(std::fs::File::create_new(&a[4]).unwrap());
 for &own in &choices[0]{for &reply in &choices[1]{
  let joint=[Some(own),Some(reply)];let base=enumerate_step_partial(&dex,&b,joint,cap);
  let mut bm=[0.;4];for l in &base.leaves{bm[outcome(&l.battle)]+=l.prob;}
  let baseline=json!({"mass":bm,"pending_mass":base.pending.iter().map(|p|p.mass).sum::<f64>(),"runs":base.runs,"exhausted":base.exhausted});
  let mut heap=BinaryHeap::from([Prefix{mass:1.,script:vec![]}]);let mut leaves=Vec::new();let mut mass=[0.;4];let mut runs=0;
  while runs<cap&&!heap.is_empty(){
   let prefix=heap.pop().unwrap();let (leaf,trace)=run_scripted(&dex,&b,joint,&prefix.script);runs+=1;
   assert!(trace.len()>=prefix.script.len());
   assert_eq!(prefix.script,trace.iter().take(prefix.script.len()).map(|d|d.chosen).collect::<Vec<_>>());
   let mut prob=1.;let mut siblings=0.;
   for (i,d) in trace.iter().enumerate(){
    if i>=prefix.script.len(){for (class,&count) in d.counts.iter().enumerate(){if class!=d.chosen&&count>0{
     let m=prob*count as f64/4294967296.;let mut s:Vec<_>=trace[..i].iter().map(|d|d.chosen).collect();s.push(class);
     heap.push(Prefix{mass:m,script:s});siblings+=m;
    }}}
    prob*=d.prob();
   }
   assert!((prob+siblings-prefix.mass).abs()<1e-9);
   let k=outcome(&leaf);mass[k]+=prob;leaves.push(json!({"script":trace.iter().map(|d|d.chosen).collect::<Vec<_>>(),"mass":prob,"outcome":k}));
  }
  let mut pending=heap.into_vec();pending.sort_by(|a,b|a.script.cmp(&b.script));let pm=pending.iter().map(|p|p.mass).sum::<f64>();
  assert!((mass.iter().sum::<f64>()+pm-1.).abs()<1e-9);
  writeln!(out,"{}",json!({"game":62,"frame":92,"state_sha256":r["state_sha256"],"own":action_input(&dex,own),"reply":action_input(&dex,reply),"baseline":baseline,
    "candidate":{"mass":mass,"pending_mass":pm,"runs":runs,"exhausted":!pending.is_empty(),"leaves":leaves,"pending":pending.iter().map(|p|json!({"script":p.script,"mass":p.mass})).collect::<Vec<_>>()}})).unwrap();out.flush().unwrap();
 }}
}
