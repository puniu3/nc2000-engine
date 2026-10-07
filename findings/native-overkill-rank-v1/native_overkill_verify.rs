use std::{io::Write,time::Instant};
use nc2000_engine::{battle::{enumerate::enumerate_step_partial,Outcome},dex::Dex,state::Battle};
use nc2000_bot::player::action_input;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn main(){
    let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),5);
    let p:Value=serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
    let bytes=std::fs::read(&a[2]).unwrap();assert_eq!(p["roots_sha256"],format!("{:x}",Sha256::digest(&bytes)));
    let db=std::fs::read(&a[3]).unwrap();assert_eq!(p["dex_sha256"],format!("{:x}",Sha256::digest(&db)));
    let dex=Dex::from_json(std::str::from_utf8(&db).unwrap()).unwrap();
    let mut roots:Vec<Value>=std::str::from_utf8(&bytes).unwrap().lines().map(|l|serde_json::from_str(l).unwrap()).collect();
    roots.sort_by_key(|r|(r["record"]["game"].as_u64().unwrap(),r["record"]["frame"].as_u64().unwrap()));
    assert_eq!(roots.len(),p["source_roots"].as_u64().unwrap() as usize);
    let mut out=std::io::BufWriter::new(std::fs::File::create_new(&a[4]).unwrap());
    for (index,input) in roots.iter().enumerate(){
        let record=&input["record"];
        if !p["keys"].as_array().unwrap().iter().any(|key|key==&json!([record["game"],record["frame"]])) {continue;}
        let bytes=serde_json::to_vec(&input["battle"]).unwrap();
        assert_eq!(record["state_sha256"],format!("{:x}",Sha256::digest(&bytes)));
        let mut b:Battle=serde_json::from_slice(&bytes).unwrap();
        assert_eq!(record["state_key"],b.state_key());assert!(b.is_idle_move_request());
        assert!(b.sides.iter().all(|s|s.pokemon_left==1));
        let side=record["side"].as_u64().unwrap() as usize;
        let legal=[0,1].map(|s|b.legal_choices(&dex,s));
        let actions=legal.each_ref().map(|aa|aa.iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>());
        assert_eq!(record["legal_actions"],json!(actions[side]));assert_eq!(record["opponent_legal_actions"],json!(actions[1-side]));
        let selected=actions[side].iter().position(|a|a==record["current_response"]["action"].as_str().unwrap()).unwrap();
        let started=Instant::now();let mut cells=Vec::new();let mut runs=0;
        for &a in &legal[0]{for &b1 in &legal[1]{
            let step=enumerate_step_partial(&dex,&b,[Some(a),Some(b1)],p["cell_cap"].as_u64().unwrap() as usize);
            let mut masses=[0.;4];
            for leaf in &step.leaves{
                let k=match leaf.battle.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3};masses[k]+=leaf.prob;
            }
            let pending=step.pending.iter().map(|s|s.mass).sum::<f64>();
            assert!((masses.iter().sum::<f64>()+pending-1.).abs()<1e-9);
            runs+=step.runs;
            cells.push(json!({"outcome_mass_p0win_p1win_tie_live":masses,"pending_mass":pending,
                "pending_scripts":step.pending.len(),"leaves":step.leaves.len(),"runs":step.runs,"exhausted":step.exhausted}));
        }}
        let row=json!({"game":record["game"],"frame":record["frame"],"state_sha256":record["state_sha256"],
            "side":side,"selected":selected,"actions":actions,"cells":cells,"runs":runs,
            "elapsed_ms":started.elapsed().as_secs_f64()*1000.});
        writeln!(out,"{row}").unwrap();out.flush().unwrap();
        if index%20==0 || index+1==roots.len(){eprintln!("roots={}/{} game={} frame={} runs={}",index+1,roots.len(),record["game"],record["frame"],runs);}
    }
}
