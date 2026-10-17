//! Paired evaluation of unknown-party preview checkpoints. Same post-preview
//! agents, random seeds and opponent selection for every candidate choice.
use nc2000_bot::{preview::load_meta_pool, Agent, BlindAgent, RmConfig, SplitMix64};
use nc2000_engine::{
    battle::{Outcome, PokemonSet, SearchChoice},
    state::Battle,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};
fn main() {
    let a: Vec<_> = std::env::args().collect();
    assert!(
        (3..=4).contains(&a.len()),
        "preview_continuation UNKNOWN_DIR OUT_DIR [JOB_IDS.json]"
    );
    let jobs: Vec<usize> = a
        .get(3)
        .map(|p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap())
        .unwrap_or_else(|| (0..512).collect());
    assert!(jobs.iter().all(|&j| j < 512));
    assert_eq!(
        jobs.iter().collect::<std::collections::HashSet<_>>().len(),
        jobs.len()
    );
    let threads: usize = std::env::var("NC2000_CONTINUATION_THREADS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(8);
    assert!(threads > 0);
    let mut paths: Vec<_> = std::fs::read_dir(&a[1])
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|s| s == "json"))
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 16);
    let rows: Vec<Value> = paths
        .iter()
        .map(|p| serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap())
        .collect();
    let out = Path::new(&a[2]);
    std::fs::create_dir_all(out).unwrap();
    let dex = conformance::load_dex();
    let pool = Arc::new(load_meta_pool(Path::new(
        "data/belief-pool-v3/belief-pool.json",
    )));
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(||loop{
  let index=cursor.fetch_add(1,Ordering::Relaxed); let Some(&job)=jobs.get(index) else {break};
  let root=job/(2*16);let search_seed=(job/16)%2;let trial=job%16;
  let path=out.join(format!("r{root:02}-s{search_seed}-t{trial:02}.json"));if path.exists(){
   let old:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).expect("Incomplete existing result");
   assert_eq!(old["root"],root);assert_eq!(old["trial"],trial);
   assert_eq!(old["search_seed"],rows[root]["job"]["seeds"][search_seed]);continue
  }
  let row=&rows[root];let own:Vec<PokemonSet>=serde_json::from_value(row["job"]["own"].clone()).unwrap();let foe:Vec<PokemonSet>=serde_json::from_value(row["job"]["foe"].clone()).unwrap();
  let seed=row["job"]["seeds"][search_seed].as_u64().unwrap();let choices:Vec<_>=row["samples"].as_array().unwrap().iter().filter(|s|s["seed"]==seed && s["budget"].as_u64().unwrap()>=81000).map(|s|(s["budget"].as_u64().unwrap(),s["choice"].as_str().unwrap().to_string())).collect();assert_eq!(choices.len(),4);
  let mut unique:Vec<_>=choices.iter().map(|(_,s)|s.clone()).collect();unique.sort();unique.dedup();
  // Equal choices have exactly zero contrast under any continuation. Avoid
  // spending simulation budget proving identity.
  if unique.len()==1{write_result(&path,&json!({"root":root,"search_seed":seed,"trial":trial,"choices":choices,"equal":true,"scores":{}}));continue}
  let mut rng=SplitMix64::new(20261017011+job as u64);let battle_seed=rng.battle_seed();let agent_seeds=[rng.next(),rng.next()];
  let mut initial=Battle::from_fixture(&dex,&battle_seed,&own,&foe).unwrap();initial.set_log_enabled(true);
  let legal=initial.legal_choices(&dex,1);
  let mut opponent=BlindAgent::new(RmConfig{iterations:243000,c:0.4,..Default::default()},pool.clone(),None,rng.next());
  let foe_pick=opponent.choose(&initial,&dex,1,&legal);
  let checkpoint=path.with_extension("partial");
  let identity=json!({"build":nc2000_bot::compiled_preview::build_fingerprint(),"row_sha256":format!("{:x}",Sha256::digest(row.to_string().as_bytes())),"job":job,"experiment":"unknown-continuation-81k-v1"});
  let mut scores=serde_json::Map::new();let mut turns=serde_json::Map::new();
  if checkpoint.exists(){
   let old:Value=serde_json::from_slice(&std::fs::read(&checkpoint).unwrap()).unwrap();assert_eq!(old["identity"],identity);
   scores=old["scores"].as_object().unwrap().clone();turns=old["turns"].as_object().unwrap().clone();
   assert!(scores.keys().all(|k|unique.contains(k)&&turns.contains_key(k)));
  }
  for input in unique{
   if scores.contains_key(&input){continue}
   let mut b=initial.clone();let own_pick=b.legal_choices(&dex,0).into_iter().find(|c|c.to_input(&dex)==input).unwrap();assert!(matches!(own_pick,SearchChoice::Team(_)));
   let mut agents=[0,1].map(|s|BlindAgent::new(RmConfig{iterations:81000,c:0.4,..Default::default()},pool.clone(),None,agent_seeds[s]));
   // Observe the original preview, with the same one-iteration RNG advance.
   for s in 0..2{agents[s].set_iterations(1);let legal=b.legal_choices(&dex,s);agents[s].choose(&b,&dex,s,&legal);agents[s].set_iterations(81000);}
   b.apply_choices(&dex,[Some(own_pick),Some(foe_pick)]).unwrap();
   while b.outcome().is_none(){assert!(b.turn<=1001);let mut picks=[None,None];for s in 0..2{let legal=b.legal_choices(&dex,s);if !legal.is_empty(){picks[s]=Some(agents[s].choose(&b,&dex,s,&legal));}}b.apply_choices(&dex,picks).unwrap();}
   let score=match b.outcome().unwrap(){Outcome::P1Win=>1.,Outcome::P2Win=>0.,Outcome::Tie=>0.5};scores.insert(input.clone(),json!(score));turns.insert(input,json!(b.turn));
   write_result(&checkpoint,&json!({"identity":identity,"scores":scores,"turns":turns}));
  }
  let result=json!({"root":root,"search_seed":seed,"trial":trial,"choices":choices,"equal":false,"scores":scores,"turns":turns});write_result(&path,&result);let _=std::fs::remove_file(checkpoint);
  let n=done.fetch_add(1,Ordering::Relaxed)+1;if n%16==0{eprintln!("{n} distinct-choice trials complete");}
 });
        }
    });
}

fn write_result(path: &Path, result: &Value) {
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap().to_str().unwrap()
    ));
    std::fs::write(&tmp, result.to_string()).unwrap();
    std::fs::rename(tmp, path).unwrap();
}
