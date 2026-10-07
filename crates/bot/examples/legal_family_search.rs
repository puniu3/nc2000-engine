use nc2000_bot::{preview::load_meta_pool, rng, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::PokemonSet, dex::Dex, prng::BattleRng, state::Battle, validate::Learnsets};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

mod legal_move_models;
#[path = "../src/exp3.rs"]
mod exp3;

#[allow(dead_code)]
mod source {
    include!("hidden_family_search.rs");
    pub fn model(dex: &Dex, game: &Value, row: &Value, alternate: bool) -> (Battle, Observer) {
        replay(dex, game, row, alternate)
    }
}

#[allow(dead_code)]
mod certificate {
    include!("hidden_terminal.rs");
    pub fn solve(dex: &Dex, b: Battle, config: Value) -> Value { finite::solve(dex,b,config) }
}

fn model_key(dex: &Dex, b: &Battle) -> String {
    assert!(!b.quick_claw_roll_observable(dex));
    b.state_key128_without_quick_claw_roll(dex,false).to_string()
}

fn alias_checks(dex: &Dex, worlds: &[(Battle,Observer)], names: &[String], side: usize) -> Value {
    let mut rows=Vec::new();
    for (world,(base,_)) in worlds.iter().enumerate() {
        let replies=base.clone().legal_choices(dex,1-side);
        let mut reduced=replies.clone();
        nc2000_bot::smmcts::collapse_sleep_choices(base,dex,1-side,&mut reduced);
        for removed in replies.iter().filter(|r|!reduced.contains(r)) {
            let representative=*reduced.iter().find(|&&candidate| {
                let mut pair=vec![candidate,*removed];
                nc2000_bot::smmcts::collapse_sleep_choices(base,dex,1-side,&mut pair);
                pair==[candidate]
            }).unwrap();
            for own in base.clone().legal_choices(dex,side) {
                let mut pending=vec![Vec::<usize>::new()];let mut runs=0;let mut leaves=0;let mut mass=0.;
                while let Some(script)=pending.pop() {
                    let mut results=Vec::new();
                    for reply in [*removed,representative] {
                        let mut b=base.clone();b.set_log_enabled(false);b.set_log_enabled(true);
                        b.prng=BattleRng::enumerating(script.clone());
                        let choices=[0,1].map(|s|Some(if s==side {own} else {reply}));
                        b.apply_choices(dex,choices).unwrap();
                        let trace=b.prng.oracle.take().unwrap().trace;
                        results.push((b,trace));
                    }
                    assert_eq!(results[0].0.state_key128(),results[1].0.state_key128());
                    assert_eq!(results[0].0.log,results[1].0.log);
                    let trace=&results[0].1;
                    assert_eq!(format!("{:?}",trace),format!("{:?}",results[1].1));
                    runs+=2;assert!(runs<=100000);
                    if script.len()<trace.len() {
                        for (i,&count) in trace[script.len()].counts.iter().enumerate().rev() {
                            if count>0 {let mut next=script.clone();next.push(i);pending.push(next);}
                        }
                    } else {
                        assert_eq!(script.len(),trace.len());leaves+=1;
                        mass+=trace.iter().map(|d|d.prob()).product::<f64>();
                    }
                }
                assert!((mass-1.).abs()<1e-10);
                rows.push(json!({"world":names[world],"removed":removed.to_input(dex),
                    "representative":representative.to_input(dex),"own":own.to_input(dex),
                    "runs":runs,"leaves":leaves,"mass":mass}));
            }
        }
    }
    json!(rows)
}

fn empty_model(dex: &Dex, learnsets: &Learnsets, game: &Value, case: &Value) -> (Battle, Observer) {
    let side = case["side"].as_u64().unwrap() as usize;
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut altered = teams.clone();
    altered[1-side][case["foe_slot"].as_u64().unwrap() as usize].moves.remove(case["move_slot"].as_u64().unwrap() as usize);
    assert_eq!(nc2000_engine::validate::validate_team(dex,learnsets,&serde_json::to_string(&altered[1-side]).unwrap())["ok"],true);
    let mut battles = [&teams,&altered].map(|t| {
        let mut b=Battle::from_fixture(dex,game["seed"].as_str().unwrap(),&t[0],&t[1]).unwrap();
        b.set_log_enabled(true);b
    });
    let mut obs=battles.each_ref().map(|b|Observer::new(b,side));
    for d in game["decisions"].as_array().unwrap().iter().take(case["decision"].as_u64().unwrap() as usize) {
        assert_eq!(battles[0].state_key128().to_string(),d["key"].as_str().unwrap());
        assert_eq!(battles[0].log,battles[1].log);
        let actions=[0,1].map(|s|battles[0].legal_choices(dex,s));
        let choices=[0,1].map(|s|d["choices"][s].as_u64().map(|i|actions[s][i as usize]));
        for i in 0..2 {
            for s in 0..2 { if let Some(a)=choices[s] {assert!(battles[i].legal_choices(dex,s).contains(&a));} }
            battles[i].apply_choices(dex,choices).unwrap();
            obs[i].observe(&battles[i],dex);
        }
    }
    assert_eq!(battles[0].log,battles[1].log);
    assert_eq!(format!("{:?}",obs[0].mons()),format!("{:?}",obs[1].mons()));
    (battles[1].clone(),obs[1].clone())
}

fn main() {
    let args: Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),4,"corpus family output");
    let cases: Vec<Value>=serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let game: Value=std::fs::read_to_string(&args[1]).unwrap().lines().map(|l|serde_json::from_str::<Value>(l).unwrap())
        .find(|g|g["game"]==cases[0]["game"]).unwrap();
    let dex=conformance::load_dex();
    let learnsets=Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap()).unwrap();
    let routes: Value=serde_json::from_slice(&std::fs::read("data/legal-move-sources-gen2.json").unwrap()).unwrap();
    let pool=load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let side=cases[0]["side"].as_u64().unwrap() as usize;
    let mut actual=vec![source::model(&dex,&game,&cases[0],false)];
    let mut names=vec![cases[0]["original_move"].as_str().unwrap().to_string()];
    for case in &cases {actual.push(source::model(&dex,&game,case,true));names.push(case["replacement"].as_str().unwrap().into());}
    actual.push(empty_model(&dex,&learnsets,&game,&cases[0]));names.push("<empty>".into());
    let mut generation_checks=Vec::new();
    let expected: Vec<_>=actual.iter().map(|(b,_)|model_key(&dex,b)).collect();
    for (name,(b,obs)) in names.iter().zip(&actual) {
        let models=legal_move_models::generate(&dex,&learnsets,&routes,&pool,b,obs,0).unwrap();
        assert_eq!(models.names,names);
        assert!(models.excluded.is_empty());
        let keys: Vec<_>=models.worlds.iter().map(|b|model_key(&dex,b)).collect();
        assert_eq!(keys,expected,"actual template {name}");
        generation_checks.push(json!({"actual":name,"keys":keys}));
    }
    let aliases=alias_checks(&dex,&actual,&names,side);
    let config=json!({"game":cases[0]["game"],"decision":cases[0]["decision"],"target_side":side,"policy_side":side,
        "first_choice":"move rockslide","priority":["move rockslide"],"rest_at_percent":null,
        "horizon":3,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
    let empty_certificate=certificate::solve(&dex,actual.last().unwrap().0.clone(),config);
    eprintln!("empty runs={} bounds={}",empty_certificate["runs"],empty_certificate["bounds"]);
    let reflect=names.iter().position(|s|s=="reflect").unwrap();
    let mut rows=Vec::new();
    for timed in [false,true] {
        for i in 0..16 {
            let mut order=["catalog","uniform","robust","robust-merged"];
            order.rotate_left(i as usize%4);
            for mode in order {
                for template in if timed {vec![0]} else {vec![0,reflect]} {
                    let (base,obs)=&actual[template];
                    let seed=0x2026110501+i;
                    let start=Instant::now();
                    let models=if mode=="catalog" {None} else {Some(legal_move_models::generate(&dex,&learnsets,&routes,&pool,base,obs,0).unwrap())};
                    let beliefs: Vec<_>=if let Some(m)=&models {m.worlds.iter().map(|b|Belief::pinned_from_battle(b,obs)).collect()}
                        else {vec![Belief::new(&dex,&pool,obs)]};
                    let generation_seconds=start.elapsed().as_secs_f64();
                    let mut search=BlindSearch::new(base,&dex,RmConfig{c:0.4,rule:SelRule::Ucb,..Default::default()},side,seed);
                    if mode!="catalog" {search.enable_compact_history_continuation(true);}
                    let arms: Vec<_>=models.as_ref().map(|m|m.worlds.iter().enumerate().flat_map(|(i,b)| {
                        let mut choices=b.clone().legal_choices(&dex,1-side);
                        if mode=="robust-merged" {nc2000_bot::smmcts::collapse_sleep_choices(b,&dex,1-side,&mut choices);}
                        choices.into_iter().map(move |a|(i,a))
                    }).collect()).unwrap_or_default();
                    let mut own=exp3::Exp3Ix::new(search.actions().len());
                    let mut adversary=exp3::Exp3Ix::new(arms.len().max(2));
                    let mut own_rng=rng::SplitMix64::new(seed ^ 0xb1e54f83902ca761);
                    let mut world_rng=rng::SplitMix64::new(seed ^ 0x54bc170a2998d6e3);
                    let mut world_counts=vec![0u32;models.as_ref().map_or(0,|m|m.worlds.len())];
                    let mut arm_counts=vec![0u32;arms.len()];
                    let setup_seconds=start.elapsed().as_secs_f64();
                    let mut reward_sum=0.;
                    while if timed {start.elapsed()<Duration::from_millis(500)} else {search.iterations()<27000} {
                        let r=match mode {
                            "catalog"=>search.step_one(&dex,&beliefs[0],obs),
                            "uniform"=>{let w=world_rng.below(beliefs.len());world_counts[w]+=1;search.step_one(&dex,&beliefs[w],obs)},
                            "robust"|"robust-merged"=>{
                                let x=own.sample(&mut own_rng);let y=adversary.sample(&mut world_rng);
                                let (w,reply)=arms[y.action];world_counts[w]+=1;arm_counts[y.action]+=1;
                                let r=search.step_forced_joint(&dex,&beliefs[w],obs,x.action,reply);
                                let value=if side==0 {r} else {1.-r};own.observe(x,1.-value);adversary.observe(y,value);r
                            },
                            _=>unreachable!(),
                        };
                        reward_sum+=if side==0 {r} else {1.-r};
                    }
                    let seconds=start.elapsed().as_secs_f64();
                    let row=json!({"mode":mode,"seed":seed,"timed":timed,"template":template,"seconds":seconds,
                        "generation_seconds":generation_seconds,"setup_seconds":setup_seconds,"iterations":search.iterations(),
                        "best":search.best().unwrap().to_input(&dex),"visits":search.visits(),"means":search.means(),
                        "rng":search.search_rng_state(),"nodes":search.node_count(),"own_rng":own_rng.0,"world_rng":world_rng.0,
                        "world_counts":world_counts,"arm_counts":arm_counts,"reward_sum":reward_sum,
                        "arms":arms.iter().map(|(w,a)|json!({"world":w,"reply":a.to_input(&dex)})).collect::<Vec<_>>()});
                    eprintln!("{mode} seed={i} timed={timed} template={template} best={} iterations={}",row["best"],row["iterations"]);
                    rows.push(row);
                    std::fs::write(&args[3],serde_json::to_vec(&json!({"names":names,"generation_checks":generation_checks,"aliases":aliases,
                        "empty_state":actual.last().unwrap().0.essence(&dex),"empty_certificate":empty_certificate,"rows":rows})).unwrap()).unwrap();
                }
            }
        }
    }
}
