use nc2000_engine::{battle::{enumerate::enumerate_step, SearchChoice, Outcome}, dex::Dex, state::Battle};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
fn main() {
    let a:Vec<_>=std::env::args().collect();assert_eq!(a.len(),4);
    let p:Value=serde_json::from_slice(&std::fs::read(&a[1]).unwrap()).unwrap();
    let bytes=std::fs::read(&a[2]).unwrap();assert_eq!(p["roots_sha256"],format!("{:x}",Sha256::digest(&bytes)));
    let dex_bytes=std::fs::read(&a[3]).unwrap();assert_eq!(p["dex_sha256"],format!("{:x}",Sha256::digest(&dex_bytes)));
    let dex=Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).unwrap();
    let r:Value=std::str::from_utf8(&bytes).unwrap().lines().map(|l|serde_json::from_str::<Value>(l).unwrap())
        .find(|r|r["record"]["game"]==10 && r["record"]["frame"]==32).unwrap();
    let mut b:Battle=serde_json::from_value(r["battle"].clone()).unwrap();assert!(b.is_idle_move_request());
    let side=r["record"]["side"].as_u64().unwrap() as usize;
    let ids=[0,1].map(|s|b.active_id(s).unwrap());
    let speeds=ids.map(|id|b.get_stat(&dex,id,4,false,false,false));
    let legal=[0,1].map(|s|b.legal_choices(&dex,s));let mut out=Vec::new();
    for name in ["earthquake","hiddenpower"] {
        let own=SearchChoice::Move(dex.moves.id(name).unwrap());assert!(legal[side].contains(&own));
        for &reply in &legal[1-side] {
            let choices=if side==0 {[own,reply]} else {[reply,own]};
            let step=enumerate_step(&dex,&b,choices.map(Some),100000).unwrap();
            let mut masses=[0.;4];let mut live=std::collections::BTreeMap::<String,f64>::new();
            for leaf in &step.leaves {
                let k=match leaf.battle.outcome(){Some(Outcome::P1Win)=>0,Some(Outcome::P2Win)=>1,Some(Outcome::Tie)=>2,None=>3};
                masses[k]+=leaf.prob;
                if k==3 {
                    let states=ids.map(|id|{let p=leaf.battle.poke(id);json!({"hp":p.hp,"status":format!("{:?}",p.status),"boosts":p.boosts,"item":p.item})});
                    *live.entry(serde_json::to_string(&states).unwrap()).or_default()+=leaf.prob;
                }
            }
            assert!((masses.iter().sum::<f64>()-1.).abs()<1e-10);
            out.push(json!({"own":name,"reply":nc2000_bot::player::action_input(&dex,reply),"runs":step.runs,
                "leaves":step.leaves.len(),"outcome_mass_p0win_p1win_tie_live":masses,"live_hp_status_boost_item_projection":live}));
        }
    }
    println!("{}",json!({"game":10,"frame":32,"state_sha256":r["record"]["state_sha256"],"side":side,"speeds":speeds,"records":out}));
}
