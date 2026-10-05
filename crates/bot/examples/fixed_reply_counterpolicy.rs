use std::time::Instant;
#[path="fixed_reply_counterpolicy/pair_bound.rs"]
mod pair_bound;
#[path="fixed_reply_counterpolicy/hp_storage.rs"]
mod hp_storage;
#[path="fixed_reply_counterpolicy/retained.rs"]
mod retained;
#[path="fixed_reply_counterpolicy/terminal_step.rs"]
mod terminal_step;
#[path="fixed_reply_counterpolicy/switch_kernel.rs"]
mod switch_kernel;
#[path="fixed_reply_counterpolicy/healing_kernel.rs"]
mod healing_kernel;
#[path="fixed_reply_counterpolicy/waking_continuation.rs"]
mod waking_continuation;
#[path="fixed_reply_counterpolicy/continuation_search.rs"]
mod continuation_search;
#[path="fixed_reply_counterpolicy/corpus_search.rs"]
mod corpus_search;
#[path="fixed_reply_counterpolicy/pick_census.rs"]
mod pick_census;
#[path="fixed_reply_counterpolicy/fork_corpus.rs"]
mod fork_corpus;
use nc2000_engine::battle::{Outcome,PokemonSet,SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle,Status,DK};
use serde::{Deserialize,Serialize};
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

fn rest_ready(b:&Battle,side:usize,waking:bool)->bool {
    b.active_id(side).is_some_and(|id| {
        let p=b.poke(id);
        p.status!=Status::Frz && (p.status!=Status::Slp || (waking && p.status_state.get_int(DK::Time)<=1))
            && i64::from(p.hp)*100<=i64::from(p.maxhp)*75
    })
}

#[derive(Clone,Copy,PartialEq,Eq)]
enum OwnPolicy { Attack,Rest75,AfterBoom75,AfterBoomWake75 }
impl OwnPolicy {
    fn name(self)->&'static str {match self {Self::Attack=>"attack",Self::Rest75=>"rest75",Self::AfterBoom75=>"after_boom75",Self::AfterBoomWake75=>"after_boom_wake75"}}
}


#[cfg(test)]
fn choose(dex:&Dex,b:&mut Battle,side:usize,first:bool,own_rest:bool,opponent_rest:Option<u8>)->Option<SearchChoice> {
    choose_rules(dex,b,side,first,if own_rest {OwnPolicy::Rest75} else {OwnPolicy::Attack},opponent_rest,true)
}

fn choose_rules(dex:&Dex,b:&mut Battle,side:usize,first:bool,own:OwnPolicy,opponent_rest:Option<u8>,return_boom:bool)->Option<SearchChoice> {
    if !b.needs_choice()[side] {return None;}
    let legal=b.legal_choices(dex,side);
    assert!(!legal.is_empty());
    if first {
        let choice=if side==0 {action(dex,&legal,"bodyslam").unwrap()} else {SearchChoice::Switch(2)};
        assert!(legal.contains(&choice));return Some(choice);
    }
    if side==0 {
        if own!=OwnPolicy::Attack && rest_ready(b,side,own==OwnPolicy::AfterBoomWake75)
            && (own==OwnPolicy::Rest75 || !nc2000_bot::bounds::has_live_boom_move(dex,b,1-side)) {
            if let Some(rest)=action(dex,&legal,"rest") {return Some(rest);}
        }
        return ["bodyslam","fireblast","rest","selfdestruct"].into_iter()
            .find_map(|name|action(dex,&legal,name)).or_else(||legal.first().copied());
    }
    if let Some(&switch)=legal.iter().find(|choice|return_boom && match choice {
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

#[derive(Clone,Copy,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(rename_all="snake_case")]
enum ReturnWhen { Always,Never,FoeLow,FoeFrozen,SelfLow }

#[derive(Clone,Copy,Debug,Serialize,Deserialize,PartialEq,Eq)]
#[serde(deny_unknown_fields)]
struct CounterResponse { return_when:ReturnWhen,ice_first:bool,finish_low:bool }

fn choose_response(dex:&Dex,b:&mut Battle,side:usize,first:bool,own:OwnPolicy,response:CounterResponse)->Option<SearchChoice> {
    if side==0 || first {return choose_rules(dex,b,side,first,own,Some(99),false);}
    if !b.needs_choice()[side] {return None;}
    let legal=b.legal_choices(dex,side);assert!(!legal.is_empty());
    let me=b.poke(b.active_id(side).unwrap());
    let foe=b.poke(b.active_id(1-side).unwrap());
    let return_now=match response.return_when {
        ReturnWhen::Always=>true,ReturnWhen::Never=>false,
        ReturnWhen::FoeLow=>i64::from(foe.hp)*2<=i64::from(foe.maxhp),
        ReturnWhen::FoeFrozen=>foe.status==Status::Frz,
        ReturnWhen::SelfLow=>i64::from(me.hp)*2<=i64::from(me.maxhp),
    };
    if return_now {
        if let Some(&switch)=legal.iter().find(|choice|match choice {
            SearchChoice::Switch(pos)=> {
                let team=&b.sides[side];dex.species.key(team.roster[team.party[(*pos-1) as usize] as usize].species)=="snorlax"
            },_=>false,
        }) {return Some(switch);}
    }
    if dex.species.key(me.species)=="vaporeon" && response.finish_low
        && i64::from(foe.hp)*100<=i64::from(foe.maxhp)*40
        && (me.status!=Status::Slp || me.status_state.get_int(DK::Time)<=1) {
        if let Some(hydro)=action(dex,&legal,"hydropump") {return Some(hydro);}
    }
    let chosen=choose_rules(dex,b,side,false,own,Some(99),false);
    if response.ice_first && chosen==action(dex,&legal,"hydropump") {
        if let Some(ice)=action(dex,&legal,"icebeam") {return Some(ice);}
    }
    chosen
}

fn seed(base:u64,index:usize)->u64 {
    let mut z=base.wrapping_add((index as u64).wrapping_mul(0x9e3779b97f4a7c15));
    z=(z^(z>>30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z=(z^(z>>27)).wrapping_mul(0x94d049bb133111eb);z^(z>>31)
}

fn run(dex:&Dex,root:&Battle,n:usize,base:u64,own:OwnPolicy,opponent_rest:Option<u8>,return_boom:bool,response:Option<CounterResponse>)->Value {
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
            let choices=[0,1].map(|side|if let Some(response)=response {
                choose_response(dex,&mut b,side,steps==0,own,response)
            } else {choose_rules(dex,&mut b,side,steps==0,own,opponent_rest,return_boom)});
            if trial==0 && steps<16 {example.push(json!({"turn":b.turn,
                "hp":([0,1].map(|s|b.active_id(s).map(|id|b.poke(id).hp))),
                "choices":choices.map(|a|a.map(|a|nc2000_bot::player::action_input(dex,a)))}));}
            b.apply_choices(dex,choices).unwrap();steps+=1;
        }
        steps_total+=steps;max_steps=max_steps.max(steps);
    }
    json!({"own_policy":own.name(),"counter_response":response,"opponent_return_boom":return_boom,"own_rest_percent":(own!=OwnPolicy::Attack).then_some(75),"opponent_rest_percent":opponent_rest,"opponent_waking_rest":opponent_rest==Some(99),
        "samples":n,"seed":base,"wins":outcomes[0],"ties":outcomes[1],"losses":outcomes[2],
        "mean_score":(outcomes[0] as f64+0.5*outcomes[1] as f64)/n as f64,
        "steps":steps_total,"max_steps":max_steps,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,
        "first_trial_prefix":example})
}

fn sha(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn main() {
    let args:Vec<_>=std::env::args().collect();
    if args.get(2).is_some_and(|mode|mode=="certify") {pair_bound::execute(&args);return;}
    if args.get(2).is_some_and(|mode|mode=="continuations") {waking_continuation::execute(&args);return;}
    if args.get(2).is_some_and(|mode|mode=="continuation-search") {continuation_search::execute(&args);return;}
    if args.get(2).is_some_and(|mode|mode=="corpus-search") {corpus_search::execute(&args);return;}
    if args.get(2).is_some_and(|mode|mode=="pick-census") {pick_census::execute(&args);return;}
    if args.get(2).is_some_and(|mode|mode=="fork-corpus") {fork_corpus::execute(&args);return;}
    assert!(matches!(args.len(),5|6),"corpus samples seed output [late-rest|counter-search|waking-rest|responses.json]");
    let extended=args.get(5).is_some_and(|mode|mode=="late-rest");
    let waking=args.get(5).is_some_and(|mode|mode=="waking-rest");
    let responses:Option<Vec<CounterResponse>>=args.get(5).filter(|mode|*mode!="late-rest").map(|mode| {
        if mode=="counter-search" || waking {
            [ReturnWhen::Always,ReturnWhen::Never,ReturnWhen::FoeLow,ReturnWhen::FoeFrozen,ReturnWhen::SelfLow].into_iter()
                .flat_map(|return_when|[false,true].into_iter().flat_map(move|ice_first|[false,true].map(|finish_low|CounterResponse {return_when,ice_first,finish_low}))).collect()
        } else {serde_json::from_slice(&std::fs::read(mode).unwrap()).unwrap()}
    });
    let raw=std::fs::read(&args[1]).unwrap();let n:usize=args[2].parse().unwrap();assert!(n>0);
    let base:u64=args[3].parse().unwrap();let dex=conformance::load_dex();
    let game=std::str::from_utf8(&raw).unwrap().lines().map(|line|serde_json::from_str::<Game>(line).unwrap())
        .find(|game|game.game==17).unwrap();
    let root=reconstruct(&dex,&game,21);assert_eq!(root.state_key128().to_string(),"199433906935569759004884428941975153695");
    let mut jobs=Vec::new();
    if let Some(responses)=&responses {
        assert!(!responses.is_empty());
        for &response in responses {
            for own in if waking {vec![OwnPolicy::AfterBoom75,OwnPolicy::AfterBoomWake75]} else {vec![OwnPolicy::AfterBoom75]} {
                jobs.push((own,Some(99),false,Some(response)));
            }
        }
    } else {
    for return_boom in if extended {vec![true,false]} else {vec![true]} {
        for opponent_rest in [None,Some(75),Some(99)] {
            for own in if extended {vec![OwnPolicy::Attack,OwnPolicy::Rest75,OwnPolicy::AfterBoom75]} else {vec![OwnPolicy::Attack,OwnPolicy::Rest75]} {
                jobs.push((own,opponent_rest,return_boom,None));
            }
        }
    }
    }
    let mut arms=Vec::new();
    for group in jobs.chunks(6) {
        arms.extend(std::thread::scope(|scope| {
            let handles=group.iter().map(|&(own,opp,return_boom,response)| {
                let dex=&dex;let root=&root;scope.spawn(move||run(dex,root,n,base,own,opp,return_boom,response))
            }).collect::<Vec<_>>();
            handles.into_iter().map(|handle|handle.join().unwrap()).collect::<Vec<_>>()
        }));
    }
    let result=json!({"schema":if waking {"nc2000-waking-rest-v1"} else if responses.is_some() {"nc2000-counterresponse-v1"} else if extended {"nc2000-late-rest-counterpolicy-v1"} else {"nc2000-fixed-reply-counterpolicy-v1"},"game":17,"decision":21,
        "state_key":root.state_key128().to_string(),"state":root.essence(&dex),"arms":arms,
        "manifest":manifest(&dex,&raw),
        "interpretation":"Seeded full-state policy-pair rollouts after fixed first Body Slam / switch-2 actions. Side zero uses the existing attack priority, optionally Rest at <=75% HP while awake and unfrozen. Side one returns to any legal Snorlax, uses Selfdestruct, then Hydro Pump / Ice Beam on Vaporeon; Sleep Talk only when the remaining sleep counter exceeds one, with optional Rest at <=75% while awake or <=99% also on the waking turn. All trials run to a true engine terminal. The optional late-rest study also keeps Vaporeon active as a counterpolicy and adds a side-zero rule withholding threshold Rest while a selected living opponent carries Selfdestruct or Explosion. The counter-search mode tests state-based return timing (always, never, foe HP <=50%, foe frozen, or own HP <=50%), optional Ice Beam priority, and optional Hydro Pump before Rest when foe HP <=40%. The waking-rest mode compares the existing after-boom policy with a variant allowing threshold Rest on the waking turn, against all twenty counterresponses. These rules read only the current battle state, not the opponent action or PRNG. Scores count ties as one half. This screens concrete counterpolicies, not human calibration, an optimal-opponent assumption, or an exact enumeration certificate."});
    std::fs::write(&args[4],serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    for arm in &result["arms"].as_array().unwrap()[..] {println!("own={} return={} rest={:?} response={} mean={} samples={} steps={}",arm["own_policy"],arm["opponent_return_boom"],arm["opponent_rest_percent"],arm["counter_response"],arm["mean_score"],arm["samples"],arm["steps"]);}
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mon(species:&str,moves:&[&str])->PokemonSet {
        PokemonSet {name:species.into(),species:species.into(),item:String::new(),ability:String::new(),
            moves:moves.iter().map(|name|(*name).into()).collect(),level:50,evs:None,ivs:None,gender:None,happiness:None}
    }
    pub(super) fn battle(dex:&Dex,bench:bool)->Battle {
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
    fn late_rest_waits_for_the_selected_boom_holder_to_faint() {
        let dex=conformance::load_dex();let mut b=battle(&dex,true);
        let own=b.active_id(0).unwrap();b.poke_mut(own).hp=b.poke(own).maxhp/2;
        assert!(nc2000_bot::bounds::has_live_boom_move(&dex,&b,1));
        assert_eq!(choose_rules(&dex,&mut b,0,false,OwnPolicy::AfterBoom75,None,true),Some(SearchChoice::Move(dex.moves.id("bodyslam").unwrap())));
        let slot=*b.sides[1].party.iter().find(|&&slot|dex.species.key(b.sides[1].roster[slot as usize].species)=="snorlax").unwrap();
        b.sides[1].roster[slot as usize].hp=0;b.sides[1].roster[slot as usize].fainted=true;b.sides[1].pokemon_left-=1;
        assert!(!nc2000_bot::bounds::has_live_boom_move(&dex,&b,1));
        assert_eq!(choose_rules(&dex,&mut b,0,false,OwnPolicy::AfterBoom75,None,true),Some(SearchChoice::Move(dex.moves.id("rest").unwrap())));
    }
    #[test]
    fn waking_rest_executes_after_cure_and_preserves_boom_and_pp_guards() {
        let dex=conformance::load_dex();let mut b=battle(&dex,false);
        let own=b.active_id(0).unwrap();let rest=dex.moves.id("rest").unwrap();
        let slam=Some(SearchChoice::Move(dex.moves.id("bodyslam").unwrap()));
        b.poke_mut(own).hp=b.poke(own).maxhp/2;
        b.poke_mut(own).status=Status::Slp;b.poke_mut(own).status_state.set_int(DK::Time,1);
        b.refresh_poke_mask(&dex,own);
        assert_eq!(choose_rules(&dex,&mut b,0,false,OwnPolicy::AfterBoom75,None,false),slam);
        let choice=choose_rules(&dex,&mut b,0,false,OwnPolicy::AfterBoomWake75,None,false);
        assert_eq!(choice,Some(SearchChoice::Move(rest)));
        let mut healed=b.clone();let pp=healed.poke(own).get_move_slot(rest).unwrap().pp;
        healed.apply_choices(&dex,[choice,Some(SearchChoice::Move(rest))]).unwrap();
        assert_eq!(healed.poke(own).hp,healed.poke(own).maxhp);
        assert_eq!(healed.poke(own).get_move_slot(rest).unwrap().pp,pp-1);
        assert_eq!(healed.poke(own).status,Status::Slp);
        assert!(healed.poke(own).status_state.get_int(DK::Time)>1);
        b.poke_mut(own).status_state.set_int(DK::Time,2);
        assert_eq!(choose_rules(&dex,&mut b,0,false,OwnPolicy::AfterBoomWake75,None,false),slam);
        b.poke_mut(own).status_state.set_int(DK::Time,1);b.poke_mut(own).get_move_slot_mut(rest).unwrap().pp=0;
        assert_eq!(choose_rules(&dex,&mut b,0,false,OwnPolicy::AfterBoomWake75,None,false),slam);
        let mut guarded=battle(&dex,true);let own=guarded.active_id(0).unwrap();
        guarded.poke_mut(own).hp=guarded.poke(own).maxhp/2;
        guarded.poke_mut(own).status=Status::Slp;guarded.poke_mut(own).status_state.set_int(DK::Time,1);
        assert_eq!(choose_rules(&dex,&mut guarded,0,false,OwnPolicy::AfterBoomWake75,None,false),slam);
    }
    #[test]
    fn stay_counterpolicy_does_not_voluntarily_return_to_snorlax() {
        let dex=conformance::load_dex();let mut b=battle(&dex,true);
        assert_eq!(choose_rules(&dex,&mut b,1,false,OwnPolicy::Attack,None,false),Some(SearchChoice::Move(dex.moves.id("hydropump").unwrap())));
    }
    #[test]
    fn response_switching_and_finishing_use_only_present_state() {
        let dex=conformance::load_dex();let mut b=battle(&dex,true);
        let own=b.active_id(0).unwrap();let foe=b.active_id(1).unwrap();
        let mut response=CounterResponse {return_when:ReturnWhen::FoeLow,ice_first:false,finish_low:false};
        assert_eq!(choose_response(&dex,&mut b,1,false,OwnPolicy::AfterBoom75,response),Some(SearchChoice::Move(dex.moves.id("hydropump").unwrap())));
        b.poke_mut(own).hp=b.poke(own).maxhp/2;
        assert_eq!(choose_response(&dex,&mut b,1,false,OwnPolicy::AfterBoom75,response),Some(SearchChoice::Switch(2)));
        response.return_when=ReturnWhen::Never;response.ice_first=true;
        assert_eq!(choose_response(&dex,&mut b,1,false,OwnPolicy::AfterBoom75,response),Some(SearchChoice::Move(dex.moves.id("icebeam").unwrap())));
        b.poke_mut(foe).hp=b.poke(foe).maxhp/2;b.poke_mut(own).hp=b.poke(own).maxhp/4;
        assert_eq!(choose_response(&dex,&mut b,1,false,OwnPolicy::AfterBoom75,response),Some(SearchChoice::Move(dex.moves.id("rest").unwrap())));
        response.finish_low=true;
        assert_eq!(choose_response(&dex,&mut b,1,false,OwnPolicy::AfterBoom75,response),Some(SearchChoice::Move(dex.moves.id("hydropump").unwrap())));
    }
    #[test]
    fn trial_seeds_do_not_alias_at_sixteen_bits() {
        let seeds=(0..200_000).map(|i|seed(2026100601,i)).collect::<std::collections::HashSet<_>>();
        assert_eq!(seeds.len(),200_000);
    }
}

fn manifest(dex:&Dex,raw:&[u8])->Value {
    let paths=["crates/bot/examples/fixed_reply_counterpolicy.rs","crates/bot/examples/fixed_reply_counterpolicy/pair_bound.rs","crates/bot/examples/fixed_reply_counterpolicy/best_first.rs","crates/bot/examples/fixed_reply_counterpolicy/hp_storage.rs","crates/bot/examples/fixed_reply_counterpolicy/retained.rs","crates/bot/examples/fixed_reply_counterpolicy/terminal_step.rs","crates/bot/examples/fixed_reply_counterpolicy/switch_kernel.rs","crates/bot/examples/fixed_reply_counterpolicy/healing_kernel.rs","crates/bot/examples/fixed_reply_counterpolicy/waking_continuation.rs","crates/bot/src/bounds.rs","crates/bot/src/hp_transition.rs",
        "crates/engine/src/state.rs","crates/engine/src/prng.rs","crates/engine/src/battle/enumerate.rs","crates/engine/src/battle/search.rs",
        "crates/engine/src/battle/actions.rs","crates/engine/src/battle/conditions.rs",
        "crates/engine/src/battle/moveexec.rs","crates/engine/src/battle/turn.rs","crates/engine/src/battle/events.rs",
        "crates/engine/src/battle/mod.rs","crates/engine/src/battle/dmg.rs","crates/engine/src/battle/items.rs"];
    let sources=paths.map(|p|(p,sha(&std::fs::read(p).unwrap())));
    let revision=std::process::Command::new("git").args(["rev-parse","HEAD"]).output().unwrap();
    assert!(revision.status.success());
    json!({"revision":String::from_utf8(revision.stdout).unwrap().trim(),"corpus_sha256":sha(raw),"source_hashes":sources,
        "dex_fingerprint":format!("{:?}",dex.replay_fingerprint),
        "binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap())})
}
