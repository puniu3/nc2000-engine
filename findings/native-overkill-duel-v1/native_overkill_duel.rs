use std::{collections::HashMap,path::Path,sync::{Arc,Mutex}};
use conformance::{fixture::repo_root,load_dex};
use nc2000_bot::{preview::{load_meta_pool,MetaPool},player::action_input,run_duel,Agent,BlindAgent,DuelSpec,RmConfig,SplitMix64,smmcts::SelRule};
use nc2000_engine::{battle::{Outcome,SearchChoice},dex::Dex,state::Battle};
use serde::{Serialize,Deserialize};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};

#[derive(Clone,Serialize,Deserialize)]
struct Decision {game:usize,side:usize,ordinal:usize,key:String,action:String}
struct Recorded {inner:BlindAgent,pair:usize,arm:usize,ordinal:usize,trace:Arc<Mutex<Vec<Decision>>>}
impl Agent for Recorded {
    fn name(&self)->String {self.inner.name()}
    fn choose(&mut self,b:&Battle,dex:&Dex,side:usize,choices:&[SearchChoice])->SearchChoice {
        let action=self.inner.choose(b,dex,side,choices);
        self.trace.lock().unwrap().push(Decision {game:2*self.pair+(side^self.arm),side,ordinal:self.ordinal,key:format!("{:032x}",b.state_key128()),action:action_input(dex,action)});
        self.ordinal+=1;action
    }
}
fn digest(path:impl AsRef<Path>)->String {format!("{:x}",Sha256::digest(std::fs::read(path).unwrap()))}
fn main(){
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),3);
    let p:Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    for (path,expected) in p["inputs"].as_object().unwrap(){assert_eq!(digest(path),expected.as_str().unwrap());}
    let dex=load_dex();let root=repo_root();
    let pool=Arc::new(load_meta_pool(&root.join("data/belief-pool-v3/belief-pool.json")));
    let catalog=load_meta_pool(&root.join("data/team-pool-v2/team-pool.json"));
    let teams:Vec<_>=catalog.teams.iter().map(|t|t.sets.clone()).collect();
    let games=p["games"].as_u64().unwrap() as usize;assert!(games%2==0);
    let seed=p["seed"].as_u64().unwrap();let null=p["null_control"].as_bool().unwrap();
    let iterations=p["iterations"].as_u64().unwrap() as u32;
    let max_turns=p["max_turns"].as_u64().unwrap() as u16;
    let pairs:HashMap<_,_>=(0..games/2).map(|i|(seed^(i as u64).wrapping_mul(0xA24B_AED4_963E_E407),i)).collect();
    let trace=Arc::new(Mutex::new(Vec::new()));
    let builder=|agent_seed,arm|->Box<dyn Agent>{
        Box::new(Recorded {inner:BlindAgent::new(RmConfig {iterations,c:0.4,rule:SelRule::Ucb,rollout_capped_damage:arm==0&&!null,..Default::default()},Arc::<MetaPool>::clone(&pool),None,agent_seed),pair:pairs[&agent_seed],arm,ordinal:0,trace:trace.clone()})
    };
    let result=run_duel(&dex,&teams,&|s|builder(s,0),&|s|builder(s,1),DuelSpec {games,base_seed:seed,threads:p["threads"].as_u64().unwrap() as usize,max_turns,progress:true,log_on:true,crn_agent_seeds:true});
    let mut trace=trace.lock().unwrap().clone();trace.sort_by_key(|d|(d.game,d.side,d.ordinal));
    let mut rng=SplitMix64::new(seed);let mut replayed=0;let mut replay_caps=0;
    for pair in 0..games/2 {
        let t0=rng.below(teams.len());let t1=rng.below(teams.len());let battle_seed=rng.battle_seed();
        for swapped in 0..2 {
            let game=pair*2+swapped;
            let records=[0,1].map(|s|trace.iter().filter(|d|d.game==game&&d.side==s).collect::<Vec<_>>());
            let mut b=Battle::from_fixture(&dex,&battle_seed,&teams[t0],&teams[t1]).unwrap();b.set_log_enabled(true);
            let mut cursors=[0,0];
            loop {
                if b.outcome().is_some()||b.turn>max_turns {break;}
                let mut picks=[None,None];
                for side in 0..2 {
                    let legal=b.legal_choices(&dex,side);
                    if !legal.is_empty(){
                        let r=records[side][cursors[side]];assert_eq!(r.ordinal,cursors[side]);
                        assert_eq!(r.key,format!("{:032x}",b.state_key128()));
                        picks[side]=Some(*legal.iter().find(|&&a|action_input(&dex,a)==r.action).unwrap());
                        cursors[side]+=1;replayed+=1;
                    }
                }
                assert_ne!(picks,[None,None]);b.apply_choices(&dex,picks).unwrap();
            }
            for s in 0..2 {assert_eq!(cursors[s],records[s].len());}
            let p0=match b.outcome(){Some(Outcome::P1Win)=>1.,Some(Outcome::P2Win)=>0.,Some(Outcome::Tie)=>0.5,None=>{replay_caps+=1;0.5}};
            assert_eq!(result.game_scores[game],if swapped==0 {p0}else{1.-p0});
        }
    }
    assert_eq!(replay_caps,result.turn_caps);
    if null {assert!(result.pair_scores.iter().all(|&s|s==0.5));}
    let text=trace.iter().map(|d|serde_json::to_string(d).unwrap()+"\n").collect::<String>();
    std::fs::write(&args[2],text).unwrap();
    println!("{}",json!({"protocol_sha256":digest(&args[1]),"trace_sha256":digest(&args[2]),"null_control":null,"iterations":iterations,"games":result.games,"pairs":result.pairs,"wins":result.wins,"losses":result.losses,"ties":result.ties,"caps":result.turn_caps,"score":result.score,"ci95_halfwidth":result.ci95,"pair_scores":result.pair_scores,"game_scores":result.game_scores,"seconds":result.secs,"mean_think_ms":[result.a_ms_per_move,result.b_ms_per_move],"replayed_games":games,"replayed_decisions":replayed}));
}
