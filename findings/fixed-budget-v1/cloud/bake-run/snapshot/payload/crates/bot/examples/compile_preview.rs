//! Reproducible offline snapshots of the live blind preview policy.
//! compile_preview JOBS.json OUT_DIR [THREADS]; each job supplies legal teams,
//! nested budgets and seeds. Output is atomic and resumable by exact job match.
use nc2000_bot::{compiled_preview, preview::load_meta_pool, Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde::Deserialize;
use serde_json::{json,Value};
use std::{path::Path, sync::atomic::{AtomicUsize,Ordering}, time::Instant};
#[derive(Deserialize)]
struct Job { id: String, own: Vec<PokemonSet>, foe: Vec<PokemonSet>, seeds: Vec<u64>, budgets: Vec<u32> }
fn main() {
 let args:Vec<_>=std::env::args().collect(); assert!((3..=4).contains(&args.len()));
 if args[1] == "--prepare" { prepare(Path::new(&args[2])); return; }
 let raw:Vec<Value>=serde_json::from_slice(&std::fs::read(&args[1]).unwrap()).unwrap();
 let jobs:Vec<Job>=serde_json::from_value(json!(raw)).unwrap();
 let out=Path::new(&args[2]); std::fs::create_dir_all(out).unwrap();
 let threads:usize=args.get(3).map(|x|x.parse().unwrap()).unwrap_or(8);assert!(threads>0);
 let dex=conformance::load_dex(); let pool=load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
 let cursor=AtomicUsize::new(0);let done=AtomicUsize::new(0);let started=Instant::now();
 std::thread::scope(|scope| {for _ in 0..threads {scope.spawn(|| loop {
  let i=cursor.fetch_add(1,Ordering::Relaxed);if i>=jobs.len(){break} let j=&jobs[i];
  assert!(j.id.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'));
  assert!(!j.budgets.is_empty()&&j.budgets[0]>0&&j.budgets.windows(2).all(|x|x[0]<x[1]));
  let path=out.join(format!("{}.json",j.id));
  if path.exists(){let old:Value=serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();assert_eq!(old["job"],raw[i]);assert_eq!(old["build"],compiled_preview::build_fingerprint());continue}
  let mut b=Battle::from_fixture(&dex,"1,2,3,4",&j.own,&j.foe).unwrap();b.set_log_enabled(true);
  let mut obs=Observer::new(&b,0);obs.observe(&b,&dex);let mut belief=Belief::new(&dex,&pool,&obs);belief.sync(&dex,&obs);
  let mut samples=vec![];
  for &seed in &j.seeds {let mut search=BlindSearch::new(&b,&dex,RmConfig{c:0.4,..Default::default()},0,seed);
   for &budget in &j.budgets {let t=Instant::now();search.step(&dex,&belief,&obs,budget-search.iterations());
    samples.push(json!({"seed":seed,"budget":budget,"choice":search.best().unwrap().to_input(&dex),"increment_ns":t.elapsed().as_nanos() as u64}));}
  }
  let row=json!({"job":raw[i],"build":compiled_preview::build_fingerprint(),"key":compiled_preview::key(&b,0).unwrap(),"fallback":belief.candidate_count()==0,"samples":samples});
  let tmp=path.with_extension("tmp");std::fs::write(&tmp,serde_json::to_vec(&row).unwrap()).unwrap();std::fs::rename(tmp,path).unwrap();
  let n=done.fetch_add(1,Ordering::Relaxed)+1;if n%16==0||n==jobs.len(){eprintln!("{n}/{} {:.1}s",jobs.len(),started.elapsed().as_secs_f64());}
 });}});
}

fn prepare(out: &Path) {
 use nc2000_bot::{TeamGen, SplitMix64, to_sets};
 std::fs::create_dir_all(out).unwrap(); let dex=conformance::load_dex();
 let catalog=std::fs::read_to_string("data/team-pool-v2/team-pool.json").unwrap();
 let gen=TeamGen::new(&dex,&std::fs::read_to_string("data/learnsets-gen2.json").unwrap(),&catalog).unwrap();
 let teams:Vec<_>=(0..gen.teams().len()).map(|i|gen.canonize(&dex,&gen.team_json(i)).unwrap()).collect();
 let mut bake=vec![]; let mut seen=std::collections::HashSet::new();
 for (i,own) in teams.iter().enumerate() {for (j,foe) in teams.iter().enumerate() {
  let b=Battle::from_fixture(&dex,"1,2,3,4",&to_sets(own).unwrap(),&to_sets(foe).unwrap()).unwrap();
  if !seen.insert(compiled_preview::key(&b,0).unwrap()){continue}
  bake.push(json!({"id":format!("pair-{i:02}-{j:02}"),"own":own,"foe":foe,"seeds":[17011,29443,48179,60293],"budgets":[729000]}));
 }}
 let mut rng=SplitMix64::new(20261017001);let pool=load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));let mut unknown=vec![];
 while unknown.len()<16 {
  let i=unknown.len();let own=&teams[i%teams.len()];let mut foe=teams[(i*7+3)%teams.len()].clone();
  for _ in 0..3 { loop {if let Some(p)=gen.propose(&dex,&foe,&mut rng){foe=p.team;break}} }
  let b=Battle::from_fixture(&dex,"1,2,3,4",&to_sets(own).unwrap(),&to_sets(&foe).unwrap()).unwrap();
  let obs=Observer::new(&b,0);let belief=Belief::new(&dex,&pool,&obs);
  if belief.candidate_count()!=0 || seen.contains(&compiled_preview::key(&b,0).unwrap()){continue}
  unknown.push(json!({"id":format!("unknown-{i:02}"),"own":own,"foe":foe,"seeds":[94133,135721],"budgets":[27000,81000,243000,729000,2187000]}));
 }
 std::fs::write(out.join("bake-jobs.json"),serde_json::to_vec(&bake).unwrap()).unwrap();
 std::fs::write(out.join("unknown-jobs.json"),serde_json::to_vec(&unknown).unwrap()).unwrap();
 eprintln!("{} unique known matchups, {} unknown matchups",bake.len(),unknown.len());
}
