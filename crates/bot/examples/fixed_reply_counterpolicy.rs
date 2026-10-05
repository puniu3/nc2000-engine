use std::time::Instant;
use nc2000_engine::battle::{Outcome,PokemonSet,SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle,Status,DK};
use serde::Deserialize;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};

#[derive(Deserialize)]
struct Decision {key:String,choices:[Option<usize>;2]}
#[derive(Deserialize)]
struct Game {game:usize,seed:String,teams:[Vec<PokemonSet>;2],decisions:Vec<Decision>}

fn reconstruct(dex:&Dex,game:&Game,decision:usize)->Battle {
    let mut b=Battle::from_fixture(dex,&game.seed,&game.teams[0],&game.teams[1]).unwrap();
    for record in &game.decisions[..decision] {
        assert_eq!(b.state_key128().to_string(),record.key);
        let legal=[0,1].map(|s|b.legal_choices(dex,s));
        b.apply_choices(dex,[0,1].map(|s|record.choices[s].map(|i|legal[s][i]))).unwrap();
    }
    assert_eq!(b.state_key128().to_string(),game.decisions[decision].key);b
}

fn action(dex:&Dex,legal:&[SearchChoice],name:&str)->Option<SearchChoice> {
    dex.moves.id(name).map(SearchChoice::Move).filter(|a|legal.contains(a))
}

fn rest_ready(b:&Battle,side:usize)->bool {
    b.active_id(side).is_some_and(|id| {
        let p=b.poke(id);
        !matches!(p.status,Status::Slp|Status::Frz) && i64::from(p.hp)*100<=i64::from(p.maxhp)*75
    })
}

fn choose(dex:&Dex,b:&mut Battle,side:usize,first:bool,own_rest:bool,opponent_rest:Option<u8>)->Option<SearchChoice> {
    if !b.needs_choice()[side] {return None;}
    let legal=b.legal_choices(dex,side);
    assert!(!legal.is_empty());
    if first {
        let choice=if side==0 {action(dex,&legal,"bodyslam").unwrap()} else {SearchChoice::Switch(2)};
        assert!(legal.contains(&choice));return Some(choice);
    }
    if side==0 {
        if own_rest && rest_ready(b,side) {
            if let Some(rest)=action(dex,&legal,"rest") {return Some(rest);}
        }
        return ["bodyslam","fireblast","rest","selfdestruct"].into_iter()
            .find_map(|name|action(dex,&legal,name)).or_else(||legal.first().copied());
    }
    if let Some(&switch)=legal.iter().find(|choice|match choice {
        SearchChoice::Switch(pos)=> {
            let team=&b.sides[side];
            dex.species.key(team.roster[team.party[(*pos-1) as usize] as usize].species)=="snorlax"
        },_=>false,
    }) {return Some(switch);}
    let id=b.active_id(side).unwrap();
    if dex.species.key(b.poke(id).species)=="snorlax" {
        return ["selfdestruct","doubleedge","earthquake","curse"].into_iter()
            .find_map(|name|action(dex,&legal,name)).or_else(||legal.first().copied());
    }
    let p=b.poke(id);
    if p.status==Status::Slp && p.status_state.get_int(DK::Time)>1 {
        if let Some(talk)=action(dex,&legal,"sleeptalk") {return Some(talk);}
    }
    if let Some(percent)=opponent_rest {
        let awake=p.status!=Status::Slp || (percent==99 && p.status_state.get_int(DK::Time)<=1);
        if awake && p.status!=Status::Frz && i64::from(p.hp)*100<=i64::from(p.maxhp)*i64::from(percent) {
            if let Some(rest)=action(dex,&legal,"rest") {return Some(rest);}
        }
    }
    ["hydropump","icebeam","rest","sleeptalk"].into_iter()
        .find_map(|name|action(dex,&legal,name)).or_else(||legal.first().copied())
}

fn seed(base:u64,index:usize)->u64 {
    let mut z=base.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
    z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)
}

fn run(dex:&Dex,root:&Battle,n:usize,base:u64,own_rest:bool,opponent_rest:Option<u8>)->Value {
    let started=Instant::now();let mut outcomes=[0usize;3];let mut steps_total=0;let mut max_steps=0;
    let mut example=Vec::new();
    for trial in 0..n {
        let mut b=root.clone();b.set_log_enabled(false);b.reseed(seed(base,trial));
        let mut steps=0;
        loop {
            if let Some(outcome)=b.outcome() {
                outcomes[match outcome {Outcome::P1Win=>0,Outcome::Tie=>1,Outcome::P2Win=>2}]+=1;break;
            }
            assert!(steps<2048,"every trial must reach an engine terminal state");
            let choices=[0,1].map(|side|choose(dex,&mut b,side,steps==0,own_rest,opponent_rest));
            if trial==0 && steps<16 {example.push(json!({"turn":b.turn,
                "hp":([0,1].map(|s|b.active_id(s).map(|id|b.poke(id).hp))),
                "choices":choices.map(|a|a.map(|a|nc2000_bot::player::action_input(dex,a)))}));}
            b.apply_choices(dex,choices).unwrap();steps+=1;
        }
        steps_total+=steps;max_steps=max_steps.max(steps);
    }
    json!({"own_rest_percent":own_rest.then_some(75),"opponent_rest_percent":opponent_rest,"opponent_waking_rest":opponent_rest==Some(99),
        "samples":n,"seed":base,"wins":outcomes[0],"ties":outcomes[1],"losses":outcomes[2],
        "mean_score":(outcomes[0] as f64+0.5*outcomes[1] as f64)/n as f64,
        "steps":steps_total,"max_steps":max_steps,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "first_trial_prefix":example})
}

fn sha(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),5,"corpus samples seed output");
    let raw=std::fs::read(&args[1]).unwrap();let n:usize=args[2].parse().unwrap();assert!(n>0);
    let base:u64=args[3].parse().unwrap();let dex=conformance::load_dex();
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==17).unwrap();
    let root=reconstruct(&dex,&game,21);assert_eq!(root.state_key128().to_string(),"199433906935569759004884428941975153695");
    let arms=std::thread::scope(|scope| {
        let handles=[(false,None),(true,None),(false,Some(75)),(true,Some(75)),(false,Some(99)),(true,Some(99))].map(|(own,opp)| {
            let dex=&dex;let root=&root;scope.spawn(move||run(dex,root,n,base,own,opp))
        });handles.map(|handle|handle.join().unwrap())
    });
    let paths=["crates/bot/examples/fixed_reply_counterpolicy.rs","crates/bot/src/bounds.rs",
        "crates/engine/src/state.rs","crates/engine/src/prng.rs","crates/engine/src/battle/search.rs",
        "crates/engine/src/battle/actions.rs","crates/engine/src/battle/conditions.rs",
        "crates/engine/src/battle/moveexec.rs","crates/engine/src/battle/turn.rs","crates/engine/src/battle/events.rs",
        "crates/engine/src/battle/mod.rs","crates/engine/src/battle/dmg.rs","crates/engine/src/battle/items.rs"];
    let sources=paths.map(|p|(p,sha(&std::fs::read(p).unwrap())));
    let revision=std::process::Command::new("git").args(["rev-parse","HEAD"]).output().unwrap();
    assert!(revision.status.success());
    let result=json!({"schema":"nc2000-fixed-reply-counterpolicy-v1","game":17,"decision":21,
        "state_key":root.state_key128().to_string(),"state":root.essence(&dex),"arms":arms,
        "manifest":{"revision":String::from_utf8(revision.stdout).unwrap().trim(),"corpus_sha256":sha(&raw),"source_hashes":sources,
            "dex_fingerprint":format!("{:?}",dex.replay_fingerprint),
            "binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap())},
        "interpretation":"Seeded full-state policy-pair rollouts after fixed first Body Slam / switch-2 actions. Side zero uses the existing attack priority, optionally Rest at <=75% HP while awake and unfrozen. Side one returns to any legal Snorlax, uses Selfdestruct, then Hydro Pump / Ice Beam on Vaporeon; Sleep Talk only when the remaining sleep counter exceeds one, with optional Rest at <=75% while awake or <=99% also on the waking turn. All trials run to a true engine terminal. Scores count ties as one half. This screens concrete counterpolicies, not human calibration, an optimal-opponent assumption, or an exact enumeration certificate."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    for arm in &result["arms"].as_array().unwrap()[..] {println!("rest={:?}/{:?} mean={} samples={} steps={}",arm["own_rest_percent"],arm["opponent_rest_percent"],arm["mean_score"],arm["samples"],arm["steps"]);}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mon(species:&str,moves:&[&str])->PokemonSet {
        PokemonSet {name:species.into(),species:species.into(),item:String::new(),ability:String::new(),
            moves:moves.iter().map(|name|(*name).into()).collect(),level:50,evs:None,ivs:None,gender:None,happiness:None}
    }
    fn battle(dex:&Dex,bench:bool)->Battle {
        let own=mon("Snorlax",&["Body Slam","Fire Blast","Rest","Self-Destruct"]);
        let mut opponents=vec![mon("Vaporeon",&["Hydro Pump","Ice Beam","Sleep Talk","Rest"])];
        if bench {opponents.push(mon("Snorlax",&["Self-Destruct","Double-Edge","Earthquake","Curse"]));}
        let mut b=Battle::from_fixture(dex,"1,2,3,4",&[own],&opponents).unwrap();
        let choices=[0,1].map(|s|b.legal_choices(dex,s)[0]);b.apply_choices(dex,choices.map(Some)).unwrap();b
    }
    #[test]
    fn counterpolicy_switches_to_snorlax_then_uses_selfdestruct() {
        let dex=conformance::load_dex();let mut b=battle(&dex,true);
        assert_eq!(choose(&dex,&mut b,1,false,false,None),Some(SearchChoice::Switch(2)));
        let a=choose(&dex,&mut b,0,false,false,None);
        b.apply_choices(&dex,[a,Some(SearchChoice::Switch(2))]).unwrap();
        assert_eq!(choose(&dex,&mut b,1,false,false,None),Some(SearchChoice::Move(dex.moves.id("selfdestruct").unwrap())));
    }
    #[test]
    fn rest_threshold_and_sleep_talk_respect_waking_turn_and_pp() {
        let dex=conformance::load_dex();let mut b=battle(&dex,false);
        let own=b.active_id(0).unwrap();let opp=b.active_id(1).unwrap();
        b.poke_mut(own).hp=b.poke(own).maxhp/2;
        assert_eq!(choose(&dex,&mut b,0,false,true,None),Some(SearchChoice::Move(dex.moves.id("rest").unwrap())));
        b.poke_mut(own).status=Status::Slp;
        assert_eq!(choose(&dex,&mut b,0,false,true,None),Some(SearchChoice::Move(dex.moves.id("bodyslam").unwrap())));
        b.poke_mut(opp).status=Status::Slp;b.poke_mut(opp).status_state.set_int(DK::Time,1);
        assert_eq!(choose(&dex,&mut b,1,false,false,None),Some(SearchChoice::Move(dex.moves.id("hydropump").unwrap())));
        b.poke_mut(opp).hp=b.poke(opp).maxhp/2;
        assert_eq!(choose(&dex,&mut b,1,false,false,Some(75)),Some(SearchChoice::Move(dex.moves.id("hydropump").unwrap())));
        assert_eq!(choose(&dex,&mut b,1,false,false,Some(99)),Some(SearchChoice::Move(dex.moves.id("rest").unwrap())));
        b.poke_mut(opp).status_state.set_int(DK::Time,2);
        assert_eq!(choose(&dex,&mut b,1,false,false,None),Some(SearchChoice::Move(dex.moves.id("sleeptalk").unwrap())));
        let talk=dex.moves.id("sleeptalk").unwrap();b.poke_mut(opp).get_move_slot_mut(talk).unwrap().pp=0;
        assert_eq!(choose(&dex,&mut b,1,false,false,None),Some(SearchChoice::Move(dex.moves.id("hydropump").unwrap())));
    }
    #[test]
    fn trial_seeds_do_not_alias_at_sixteen_bits() {
        let seeds=(0..200_000).map(|i|seed(2026100601,i)).collect::<std::collections::HashSet<_>>();
        assert_eq!(seeds.len(),200_000);
    }
}
