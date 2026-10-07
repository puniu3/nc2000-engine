use std::{collections::BTreeMap,io::Write};
use nc2000_bot::{preview::load_meta_pool,player::action_input,SplitMix64};
use nc2000_engine::{battle::SearchChoice,dex::Dex,state::Battle};
use serde::{Serialize,Deserialize};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
#[derive(Clone,Serialize,Deserialize)]
struct Decision {game:usize,side:usize,ordinal:usize,key:String,action:String}
fn sha(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn main(){
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),5);
    let p:Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let bytes=std::fs::read(&args[2]).unwrap();assert_eq!(p["trace_sha256"],sha(&bytes));
    let mut records=BTreeMap::<(usize,usize),Vec<Decision>>::new();
    for line in std::str::from_utf8(&bytes).unwrap().lines(){let r:Decision=serde_json::from_str(line).unwrap();records.entry((r.game,r.side)).or_default().push(r);}
    let db=std::fs::read("data/gen2stadium2.json").unwrap();assert_eq!(p["dex_sha256"],sha(&db));let dex=Dex::from_json(std::str::from_utf8(&db).unwrap()).unwrap();
    assert_eq!(p["catalog_sha256"],sha(&std::fs::read("data/team-pool-v2/team-pool.json").unwrap()));
    let pool=load_meta_pool(std::path::Path::new("data/team-pool-v2/team-pool.json"));
    let results:Value=serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
    assert_eq!(p["result_sha256"],sha(&std::fs::read(&args[3]).unwrap()));
    let mut rng=SplitMix64::new(p["seed"].as_u64().unwrap());
    let mut out=std::io::BufWriter::new(std::fs::File::create_new(&args[4]).unwrap());
    for pair in 0..p["pairs"].as_u64().unwrap() as usize {
        let t0=rng.below(pool.teams.len());let t1=rng.below(pool.teams.len());let seed=rng.battle_seed();
        let mut b=Battle::from_fixture(&dex,&seed,&pool.teams[t0].sets,&pool.teams[t1].sets).unwrap();b.set_log_enabled(true);
        let mut cursors=[0,0];let mut frame=0;let mut found=false;
        loop {
            if b.outcome().is_some()||b.turn>p["max_turns"].as_u64().unwrap() as u16 {break;}
            let legal=[0,1].map(|s|b.legal_choices(&dex,s));let mut picks=[[None;2];2];let mut changed=Vec::new();
            for side in 0..2 {
                if legal[side].is_empty(){continue;}
                for game in 0..2 {
                    let r=&records[&(pair*2+game,side)][cursors[side]];assert_eq!(r.ordinal,cursors[side]);
                    assert_eq!(r.key,format!("{:032x}",b.state_key128()));
                    picks[game][side]=Some(*legal[side].iter().find(|&&a|action_input(&dex,a)==r.action).unwrap());
                }
                if picks[0][side]!=picks[1][side] {
                    let choices=(0..2).map(|g|action_input(&dex,picks[g][side].unwrap())).collect::<Vec<_>>();
                    changed.push(json!({"side":side,"candidate":choices[side],"control":choices[1-side]}));
                }
                cursors[side]+=1;
            }
            if !changed.is_empty(){
                let phase=if legal.iter().flatten().any(|a|matches!(a,SearchChoice::Team(_))){"preview"}else{"battle"};
                let bytes=serde_json::to_vec(&b).unwrap();
                let row=json!({"pair":pair,"frame":frame,"turn":b.turn,"phase":phase,"team_indices":[t0,t1],"state_key128":format!("{:032x}",b.state_key128()),"battle_sha256":sha(&bytes),"alive":[b.sides[0].pokemon_left,b.sides[1].pokemon_left],"changed":changed,"pair_score":results["pair_scores"][pair],"battle_json":std::str::from_utf8(&bytes).unwrap()});
                writeln!(out,"{row}").unwrap();found=true;break;
            }
            b.apply_choices(&dex,picks[0]).unwrap();frame+=1;
        }
        if !found {
            for game in 0..2 {for side in 0..2 {assert_eq!(cursors[side],records[&(pair*2+game,side)].len());}}
            assert_eq!(results["pair_scores"][pair],0.5);
            writeln!(out,"{}",json!({"pair":pair,"phase":"identical","frames":frame,"pair_score":results["pair_scores"][pair]})).unwrap();
        }
    }
}
