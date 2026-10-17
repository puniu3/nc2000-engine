//! Frozen-schedule evaluation of the distribution-first live policy.
//! prepare DIR [pilot] | run DIR THREADS | parity FIXTURE
#[path = "support/execution_agent.rs"]
mod execution_agent;
use execution_agent::{ExecutionAgent, Ensemble, Stats};
use nc2000_bot::{Agent, BrAgent, MctsConfig, SplitMix64, decision::PolicyMode, preview::MetaPool};
use nc2000_engine::{battle::{Outcome, PokemonSet}, dex::Dex, state::Battle};
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::{Arc, atomic::{AtomicUsize, Ordering}}, time::Instant};

const MODES: [&str; 3] = ["legacy", "visit-sample", "estimated-mix"];
#[derive(Clone, Serialize, Deserialize)]
struct Job {
    id: String, cell: String, block: usize, leg: usize, mode: String, opponent: String,
    teams: [usize; 2], battle_seed: String, agent_seeds: [u32; 2], model_seeds: [u32; 3],
}
#[derive(Serialize, Deserialize)]
struct Plan {
    schema: String, pilot: bool, budget: u32, attacker_budget: u32, max_turns: u16,
    source_sha256: Value, input_sha256: Value, jobs: Vec<Job>, protocol: Value,
}
fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn sources() -> Value {
    let paths = ["crates/bot/examples/mixed_policy_eval.rs", "crates/bot/examples/support/execution_agent.rs",
        "crates/bot/src/decision.rs", "crates/bot/src/blind.rs", "crates/bot/src/root_policy.rs",
        "crates/bot/src/exploit.rs", "crates/bot/src/smmcts.rs", "crates/bot/src/belief.rs",
        "crates/bot/src/observe.rs", "Cargo.lock"];
    let mut result: Value = paths.into_iter().map(|p| (p.to_string(), json!(sha(&fs::read(p).unwrap())))).collect();
    result["solver_build"] = json!(nc2000_bot::compiled_preview::build_fingerprint());
    result
}
fn inputs() -> Value {
    ["data/team-pool-v2/team-pool.json", "data/belief-pool-v3/belief-pool.json", "data/gen2stadium2.json"]
        .into_iter().map(|p| (p.to_string(), json!(sha(&fs::read(p).unwrap())))).collect()
}
fn catalog() -> Value { serde_json::from_slice(&fs::read("data/team-pool-v2/team-pool.json").unwrap()).unwrap() }
fn prepare(dir: &Path, pilot: bool) {
    fs::create_dir_all(dir).unwrap();
    assert!(!dir.join("plan.json").exists(), "never overwrite a preregistration");
    let cat = catalog();
    let teams = cat["teams"].as_array().unwrap();
    let weights: Vec<f64> = teams.iter().map(|t| t["drawWeight"].as_f64().unwrap()).collect();
    let sum: f64 = weights.iter().sum();
    let mut rng = SplitMix64::new(if pilot { 202610170101 } else { 202610170202 });
    let mut jobs = vec![];
    let n = if pilot { 1 } else { 128 };
    for block in 0..n {
        let selected = std::array::from_fn(|_| {
            let mut u = rng.next_f64() * sum;
            weights.iter().position(|w| { u -= w; u < 0. }).unwrap_or(weights.len()-1)
        });
        let battle_seed = rng.battle_seed();
        let seeds: [u32; 5] = std::array::from_fn(|_| rng.next() as u32);
        assert_eq!(seeds.iter().collect::<std::collections::HashSet<_>>().len(), 5);
        for mode in MODES {
            for opponent in ["direct", "oracle", "control"] {
                let eligible = match opponent {
                    "direct" => mode != "legacy" || block < 4,
                    "oracle" => block < 64,
                    "control" => block < 32,
                    _ => unreachable!(),
                };
                if !eligible { continue; }
                for leg in 0..2 {
                    let cell = format!("{opponent}-{mode}");
                    jobs.push(Job { id: format!("{cell}-{block:03}-{leg}"), cell, block, leg,
                        mode: mode.into(), opponent: opponent.into(), teams: selected,
                        battle_seed: battle_seed.clone(), agent_seeds: [seeds[0], seeds[1]],
                        model_seeds: [seeds[2], seeds[3], seeds[4]] });
                }
            }
        }
    }
    let plan = Plan { schema: "nc2000-mixed-policy-evaluation-v1".into(), pilot,
        budget: 81000, attacker_budget: 243000, max_turns: 1001,
        source_sha256: sources(), input_sha256: inputs(), jobs,
        protocol: json!({
            "population":"24 shipped catalog teams drawn independently by normalized drawWeight; shipped blind prior",
            "information":"defenders blind; attacker true state is a deliberately stronger stress test, same for oracle and control",
            "policy":"exact live search lifecycle, UCB c=0.4 buckets=16; no baked/compiled previews; all decisions including preview use requested mode",
            "oracle":"three independent persistent target-model seeds, never live target seed; root distribution only; deeper attacker UCB; approximate BR, not exact exploitability",
            "control":"same attacker search/rollout and own 243000 iterations, oracle disabled; total compute differs by three target searches",
            "schedule":"same teams, chance seeds and physical-side agent seeds across cells; both target sides per block; pilot seeds disjoint",
            "primary":"each candidate vs legacy: W/D/L and mean score, block-bootstrap confidence interval; against tailored oracle: paired target-score difference vs legacy, same blocks",
            "secondary":"paired pure-win difference; oracle-vs-control target score loss on common first 32 blocks; preview/battle mixing activation, legacy disagreement, latency, terminal coverage",
            "inference":"100000 block-bootstrap draws fixed seed 202610170303; 98.75% two-sided intervals for four primary contrasts (Bonferroni family 95%); other intervals descriptive 95%; block is unit",
            "criterion":"direct improvement: score-minus-0.5 interval lower >0; resistance improvement: oracle defender-score difference interval lower >0. Report pure-win estimates separately. Broad promotion requires both, no claim from significance in only one.",
            "stopping":"fixed 128 pairs direct per candidate, 64 pairs oracle per defender, 32 pairs control per defender, 4 legacy-null pairs. No outcome-driven stopping or parameter selection. Pilot for timing and correctness only.",
            "caps":"engine turn-1000 draw rule; operational max_turns=1001; any nonterminal cap reported as unresolved and bounded, not silently drawn",
            "scope":"live research path, not deployed compiled-preview behavior or formal Nash exploitability"
        }) };
    fs::write(dir.join("plan.json"), serde_json::to_vec_pretty(&plan).unwrap()).unwrap();
    eprintln!("{} jobs frozen", plan.jobs.len());
}

fn game(dex: &Dex, pool: Arc<MetaPool>, teams: &[Vec<PokemonSet>], plan: &Plan, job: &Job) -> Value {
    let target_side = job.leg;
    let mut battle = Battle::from_fixture(dex, &job.battle_seed, &teams[job.teams[0]], &teams[job.teams[1]]).unwrap();
    battle.set_log_enabled(true);
    let target = ExecutionAgent::new(pool.clone(), PolicyMode::parse(&job.mode).unwrap(), plan.budget, job.agent_seeds[target_side]);
    let target_stats = target.stats.clone();
    let model_stats: Vec<std::rc::Rc<std::cell::RefCell<Stats>>>;
    let opponent: Box<dyn Agent> = if job.opponent == "direct" {
        model_stats = vec![];
        Box::new(ExecutionAgent::new(pool.clone(), PolicyMode::Legacy, plan.budget, job.agent_seeds[1-target_side]))
    } else {
        let models: Vec<_> = job.model_seeds.iter().map(|&seed|
            ExecutionAgent::new(pool.clone(), PolicyMode::parse(&job.mode).unwrap(), plan.budget, seed)).collect();
        model_stats = models.iter().map(|m| m.stats.clone()).collect();
        let agent = BrAgent::new(Box::new(Ensemble(models)), 1,
            MctsConfig { iterations: plan.attacker_budget, ..Default::default() },
            u64::from(job.agent_seeds[1-target_side]));
        Box::new(if job.opponent == "control" { agent.without_oracle() } else { agent })
    };
    let mut agents: [Box<dyn Agent>; 2] = if target_side == 0 { [Box::new(target), opponent] } else { [opponent, Box::new(target)] };
    let mut ns = [0u64; 2]; let mut calls = [0u64; 2]; let mut trace = Sha256::new();
    let start = Instant::now();
    let mut capped = false;
    loop {
        if battle.outcome().is_some() { break; }
        if battle.turn > plan.max_turns { capped = true; break; }
        let mut picks = [None, None];
        for side in 0..2 {
            let choices = battle.legal_choices(dex, side);
            if choices.is_empty() { continue; }
            let t = Instant::now();
            let pick = agents[side].choose(&battle, dex, side, &choices);
            ns[side] += t.elapsed().as_nanos() as u64; calls[side] += 1;
            assert!(choices.contains(&pick));
            trace.update(format!("{}:{side}:{}\n", battle.turn, pick.to_input(dex)));
            picks[side] = Some(pick);
        }
        assert_ne!(picks, [None,None]);
        battle.apply_choices(dex, picks).unwrap();
    }
    let outcome = battle.outcome();
    let target_score = match outcome {
        Some(Outcome::P1Win) => Some(f64::from(target_side == 0)),
        Some(Outcome::P2Win) => Some(f64::from(target_side == 1)),
        Some(Outcome::Tie) => Some(0.5), None => None,
    };
    json!({"job":job,"score":target_score,"outcome":format!("{outcome:?}"),"capped":capped,
        "turns":battle.turn,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "target_ns":ns[target_side],"opponent_ns":ns[1-target_side],"target_calls":calls[target_side],"opponent_calls":calls[1-target_side],
        "trace_sha256":format!("{:x}",trace.finalize()),"target_stats":target_stats.borrow().clone(),
        "model_stats":model_stats.iter().map(|s|s.borrow().clone()).collect::<Vec<_>>()})
}
fn run(dir: &Path, threads: usize) {
    let raw = fs::read(dir.join("plan.json")).unwrap();
    let plan: Plan = serde_json::from_slice(&raw).unwrap();
    assert_eq!(plan.source_sha256,sources(),"sources changed after preregistration");
    assert_eq!(plan.input_sha256,inputs(),"inputs changed after preregistration");
    let plan_sha = sha(&raw);
    let dex = conformance::load_dex();
    let cat = catalog();
    let teams: Vec<Vec<PokemonSet>> = cat["teams"].as_array().unwrap().iter().map(|t|serde_json::from_value(t["sets"].clone()).unwrap()).collect();
    let pool = Arc::new(serde_json::from_slice::<MetaPool>(&fs::read("data/belief-pool-v3/belief-pool.json").unwrap()).unwrap());
    fs::create_dir_all(dir.join("rows")).unwrap();
    let cursor = AtomicUsize::new(0); let done = AtomicUsize::new(0);
    let start = Instant::now();
    std::thread::scope(|scope| { for _ in 0..threads {
        scope.spawn(|| loop {
            let i = cursor.fetch_add(1,Ordering::Relaxed); if i >= plan.jobs.len() { break; }
            let job = &plan.jobs[i]; let path = dir.join("rows").join(format!("{}.json",job.id));
            if path.exists() {
                let old: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
                assert_eq!(old["plan_sha256"],plan_sha); assert_eq!(old["job"],json!(job));
            } else {
                let mut row = game(&dex,pool.clone(),&teams,&plan,job);
                row["plan_sha256"] = json!(plan_sha);
                let temp = path.with_extension("tmp");
                fs::write(&temp,serde_json::to_vec(&row).unwrap()).unwrap(); fs::rename(temp,path).unwrap();
                eprintln!("{} turns={} score={} {:.1}s",job.id,row["turns"],row["score"],row["elapsed_seconds"].as_f64().unwrap());
            }
            let n = done.fetch_add(1,Ordering::Relaxed)+1;
            if n%10==0 || n==plan.jobs.len() { eprintln!("completed {n}/{} wall {:.1}s",plan.jobs.len(),start.elapsed().as_secs_f64()); }
        });
    }});
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "prepare" => prepare(Path::new(&args[2]), args.get(3).is_some_and(|s|s=="pilot")),
        "run" => run(Path::new(&args[2]), args[3].parse().unwrap()),
        _ => panic!("prepare DIR [pilot] | run DIR THREADS"),
    }
}
