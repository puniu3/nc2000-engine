use std::io::Write;
use nc2000_engine::{battle::SearchChoice,dex::Dex,state::Battle};
use nc2000_bot::{eval::expected_hit_fraction,mcts::{playout_pick,Playout},player::action_input,rng::SplitMix64,RmConfig};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
fn main(){
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),5);
    let p:Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let roots=std::fs::read(&args[2]).unwrap();assert_eq!(p["roots_sha256"],format!("{:x}",Sha256::digest(&roots)));
    let db=std::fs::read(&args[3]).unwrap();assert_eq!(p["dex_sha256"],format!("{:x}",Sha256::digest(&db)));
    let dex=Dex::from_json(std::str::from_utf8(&db).unwrap()).unwrap();
    let mut out=std::io::BufWriter::new(std::fs::File::create_new(&args[4]).unwrap());
    let playout=Playout::Heavy{eps:0.0,turns:8,weights:Default::default()};
    let rules=RmConfig::default().rollout_rules();
    let mut count=0;
    for line in std::str::from_utf8(&roots).unwrap().lines(){
        let input:Value=serde_json::from_str(line).unwrap();let record=&input["record"];
        let bytes=serde_json::to_vec(&input["battle"]).unwrap();assert_eq!(record["state_sha256"],format!("{:x}",Sha256::digest(&bytes)));
        let mut b:Battle=serde_json::from_slice(&bytes).unwrap();assert_eq!(record["state_key"],b.state_key());
        assert!(b.is_idle_move_request() && b.sides.iter().all(|s|s.pokemon_left==1));
        let legal=[0,1].map(|s|b.legal_choices(&dex,s));
        for side in 0..2{
            let att=b.active_id(side).unwrap();let def=b.active_id(1-side).unwrap();
            let baseline=playout_pick(&b,&dex,&playout,side,&legal[side],&mut SplitMix64(0),rules);
            let mut best:Option<(SearchChoice,f64,f64)>=None;let mut raw_best:Option<(SearchChoice,f64)>=None;let mut scores=Vec::new();
            for &action in &legal[side]{
                let SearchChoice::Move(mid)=action else {panic!("last-mon move root")};
                let raw=expected_hit_fraction(&b,&dex,att,def,mid,true);
                let accuracy=b.hit_probability(&dex,att,def,mid);
                let capped=if accuracy>0.0 {(raw/accuracy).min(1.0)*accuracy} else {0.0};
                let suicidal=dex.move_static(mid).selfdestruct;
                scores.push(json!({"action":action_input(&dex,action),"raw":raw,"accuracy":accuracy,"capped":capped,"suicidal":suicidal}));
                if suicidal {continue;}
                if raw_best.is_none_or(|(_,r)|raw>r){raw_best=Some((action,raw));}
                if best.is_none_or(|(_,c,r)|capped>c || (capped==c && raw>r)){best=Some((action,capped,raw));}
            }
            let eligible=raw_best.is_some_and(|(a,s)|a==baseline && s>0.0);
            let candidate=if eligible {best.unwrap().0} else {baseline};
            let row=json!({"game":record["game"],"frame":record["frame"],"state_sha256":record["state_sha256"],"side":side,"root_side":record["side"],"baseline":action_input(&dex,baseline),"candidate":action_input(&dex,candidate),"eligible":eligible,"own_status":format!("{:?}",b.poke(att).status),"foe_status":format!("{:?}",b.poke(def).status),"scores":scores});
            writeln!(out,"{row}").unwrap();count+=1;
        }
    }
    assert_eq!(count,p["source_roots"].as_u64().unwrap()*2);
}
