use std::collections::BTreeMap;
use std::time::Instant;
use nc2000_bot::hp_attack_kernel::HpAttackKernelCache;
use nc2000_bot::hp_transition::HpHealingCache;
use nc2000_bot::player::action_input;
use nc2000_engine::battle::enumerate::{enumerate_step,enumerate_step_with_runner,StepEnum};
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::{dex::Dex,state::Battle};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};

fn digest(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn probability_map(step:StepEnum)->BTreeMap<u128,f64> {
    let mut result=BTreeMap::new();
    for leaf in step.leaves {*result.entry(leaf.battle.state_key128()).or_default()+=leaf.prob;}
    assert!((result.values().sum::<f64>()-1.).abs()<1e-9);
    result
}
fn main() {
    let args:Vec<_>=std::env::args().collect();assert_eq!(args.len(),4,"PROTOCOL CENSUS DEX");
    let protocol:Value=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
    let bytes=std::fs::read(&args[2]).unwrap();assert_eq!(protocol["census_sha256"],digest(&bytes));
    let dex_bytes=std::fs::read(&args[3]).unwrap();assert_eq!(protocol["dex_sha256"],digest(&dex_bytes));
    let dex=Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).unwrap();
    let mut rows:Vec<Value>=std::str::from_utf8(&bytes).unwrap().lines().map(|l|serde_json::from_str(l).unwrap()).collect();
    rows.sort_by_key(|r|(r["game"].as_u64().unwrap(),r["frame"].as_u64().unwrap()));
    let mut cases=Vec::<(String,Vec<Battle>,[Option<SearchChoice>;2])>::new();
    let mut coverage=Vec::new();
    for r in &rows {
        let mut legal=0;let mut admitted=0;let mut states=0;
        for (family,f) in r["diagnostics"]["families"].as_array().unwrap().iter().enumerate() {
            legal+=f["legal_pairs"].as_u64().unwrap();
            states+=f["states"].as_array().unwrap().len();
            let battles:Vec<Battle>=f["states"].as_array().unwrap().iter().map(|b|serde_json::from_value(b.clone()).unwrap()).collect();
            let mut root=battles[0].clone();let actions=[0,1].map(|s|root.legal_choices(&dex,s));
            for pair in f["admitted_choices"].as_array().unwrap() {
                admitted+=1;
                let choices=[0,1].map(|s|Some(*actions[s].iter().find(|&&a|action_input(&dex,a)==pair[s]).unwrap()));
                cases.push((format!("{}/{}/{family}/{}:{}",r["game"],r["frame"],pair[0],pair[1]),battles.clone(),choices));
            }
        }
        coverage.push(json!({"game":r["game"],"frame":r["frame"],"families":r["diagnostics"]["families"].as_array().unwrap().len(),"states":states,"legal_pairs":legal,"admitted_pairs":admitted}));
    }
    let cap=protocol["benchmark_cell_cap"].as_u64().unwrap() as usize;
    let mut expected=BTreeMap::new();
    let mut records=Vec::new();
    for (pass,arm) in protocol["benchmark_order"].as_array().unwrap().iter().enumerate() {
        let arm=arm.as_str().unwrap();
        let mut kernels=HpAttackKernelCache::new(protocol["kernel_leaf_capacity"].as_u64().unwrap() as usize);
        let mut scripts=HpHealingCache::new(protocol["script_capacity"].as_u64().unwrap() as usize).with_attacks();
        let mut total_runs=0;let mut elapsed=0.;let mut calls=0;
        let mut map_hashes=Vec::new();
        for (case,states,choices) in &cases {
            for (index,b) in states.iter().enumerate() {
                let started=Instant::now();
                let step=match arm {
                    "control"=>enumerate_step(&dex,b,*choices,cap),
                    "scripts"=>{let mut runner=scripts.prepare(&dex,b,*choices);enumerate_step_with_runner(&dex,b,*choices,cap,&mut |script|runner.run(script))},
                    "kernel"=>kernels.enumerate(&dex,b,*choices,cap),
                    _=>panic!("unknown arm"),
                }.expect("registered cap must complete every admitted case");
                elapsed+=started.elapsed().as_secs_f64()*1000.;
                total_runs+=step.runs;calls+=1;
                let snapshots:Vec<_>=step.leaves.iter().map(|leaf|json!({"battle":leaf.battle,"prob":leaf.prob,"key":leaf.battle.state_key128().to_string()})).collect();
                let map=probability_map(step);
                let key=(case.clone(),index);
                if pass==0 {expected.insert(key.clone(),map.clone());}
                let old=&expected[&key];assert_eq!(old.len(),map.len());
                if map.iter().any(|(k,v)|old.get(k).is_none_or(|x|(x-v).abs()>1e-10)) {
                    let direct=enumerate_step(&dex,b,*choices,cap).unwrap();
                    let expected:Vec<_>=direct.leaves.iter().map(|leaf|json!({"battle":leaf.battle,"prob":leaf.prob,"key":leaf.battle.state_key128().to_string()})).collect();
                    println!("{}",json!({"pass":pass,"arm":arm,"case":case,"index":index,"actual":snapshots,"expected":expected}));
                    return;
                }
                map_hashes.push(digest(&serde_json::to_vec(&map.iter().map(|(k,v)|(k.to_string(),v)).collect::<Vec<_>>()).unwrap()));
            }
        }
        records.push(json!({"pass":pass,"arm":arm,"calls":calls,"runs":total_runs,"elapsed_ms":elapsed,
            "map_hashes":map_hashes,"kernel_stats":format!("{:?}",kernels.stats),"script_stats":scripts.stats()}));
        eprintln!("pass={pass} arm={arm} calls={calls} runs={total_runs} ms={elapsed:.0}");
    }
    println!("{}",json!({"schema":protocol["schema"],"coverage":coverage,"cases":cases.len(),"records":records}));
}
