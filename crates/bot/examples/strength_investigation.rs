//! Fixed, exploratory budget/solver audit. No production policy changes.
use nc2000_bot::{
    bounds::{BoundConfig, BoundSolver},
    eval::{eval01, EvalWeights},
    preview::load_meta_pool,
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{Outcome, PokemonSet},
    dex::Dex,
    prng::{BattleRng, Prng},
    state::Battle,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{
    io::Write,
    path::Path,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};
mod combo_certify;
#[path = "../tests/support/perish_deadline.rs"]
mod perish_deadline;
#[derive(Deserialize)]
struct Decision {
    key: String,
    choices: [Option<usize>; 2],
}
#[derive(Deserialize)]
struct Game {
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    decisions: Vec<Decision>,
    samples: Vec<usize>,
}
struct Task {
    id: String,
    b: Battle,
    obs: [Observer; 2],
    belief: [Belief; 2],
    side: usize,
}
fn cfg() -> RmConfig {
    RmConfig {
        c: 0.4,
        ..Default::default()
    }
}
fn snapshot(dex: &Dex, b: &Battle) -> Value {
    json!({"turn":b.turn,"key":b.state_key128().to_string(),"sides":([0,1].map(|s|b.sides[s].party.iter().map(|&slot|{let p=&b.sides[s].roster[slot as usize];json!({"species":dex.species.key(p.species),"active":b.sides[s].active==Some(slot),"hp":p.hp,"maxhp":p.maxhp,"status":p.status.as_str(),"trapped":p.trapped,"moves":p.move_slots.iter().map(|m|json!({"move":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})}).collect::<Vec<_>>()))})
}
fn tasks(dex: &Dex, path: &str) -> Vec<Task> {
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut tasks = vec![];
    let certified: Vec<Value> = serde_json::from_slice(
        &std::fs::read("findings/bot-strength-investigation-v1/certified-selection.json").unwrap(),
    )
    .unwrap();
    for line in std::fs::read_to_string(path).unwrap().lines() {
        let g: Game = serde_json::from_str(line).unwrap();
        let mut b = Battle::from_fixture(dex, &g.seed, &g.teams[0], &g.teams[1]).unwrap();
        b.set_log_enabled(true);
        let mut obs = [0, 1].map(|s| Observer::new(&b, s));
        let mut belief = [0, 1].map(|s| Belief::new(dex, &pool, &obs[s]));
        let mut selected = false;
        for (d, r) in g.decisions.iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), r.key);
            let legal = [0, 1].map(|s| b.legal_choices(dex, s));
            for s in 0..2 {
                if !legal[s].is_empty() {
                    obs[s].observe(&b, dex);
                    belief[s].sync(dex, &obs[s]);
                }
            }
            let preview = d == 0 && g.game < 6;
            let endgame = (!selected || g.samples.last() == Some(&d))
                && g.samples.contains(&d)
                && legal[0].len() > 1;
            if preview || endgame {
                tasks.push(Task {
                    id: format!("g{}-d{}", g.game, d),
                    b: b.clone(),
                    obs: obs.clone(),
                    belief: [0, 1].map(|s| Belief::new(dex, &pool, &obs[s])),
                    side: 0,
                });
                if endgame {
                    selected = true;
                }
            }
            for c in certified
                .iter()
                .filter(|c| c["game"] == g.game && c["decision"] == d)
            {
                let side = c["side"].as_u64().unwrap() as usize;
                let id = if side == 0 {
                    format!("g{}-d{}", g.game, d)
                } else {
                    format!("g{}-d{}-s1", g.game, d)
                };
                if !tasks.iter().any(|t| t.id == id) {
                    tasks.push(Task {
                        id,
                        b: b.clone(),
                        obs: obs.clone(),
                        belief: [0, 1].map(|s| Belief::new(dex, &pool, &obs[s])),
                        side,
                    });
                }
            }
            b.apply_choices(dex, [0, 1].map(|s| r.choices[s].map(|i| legal[s][i])))
                .unwrap();
        }
    }
    tasks
}
fn continuation(dex: &Dex, t: &Task, action: &str, seed: u64, iters: u32) -> Value {
    let mut b = t.b.clone();
    let mut obs = t.obs.clone();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut belief = [0, 1].map(|s| Belief::new(dex, &pool, &obs[s]));
    let mut rng = nc2000_bot::SplitMix64::new(seed);
    b.prng = BattleRng::seeded(Prng::from_seed_str(&rng.battle_seed()).unwrap());
    let mut count = 0;
    while b.outcome().is_none() && count < 1600 {
        let legal = [0, 1].map(|s| b.legal_choices(dex, s));
        let choices = [0, 1].map(|s| {
            if legal[s].is_empty() {
                return None;
            }
            obs[s].observe(&b, dex);
            belief[s].sync(dex, &obs[s]);
            if count == 0 && s == t.side {
                return Some(*legal[s].iter().find(|a| a.to_input(dex) == action).unwrap());
            }
            let seed = rng.next();
            if legal[s].len() == 1 {
                return Some(legal[s][0]);
            }
            let mut search = BlindSearch::new(&b, dex, cfg(), s, seed);
            search.step(dex, &belief[s], &obs[s], iters);
            search.best()
        });
        b.apply_choices(dex, choices).unwrap();
        count += 1;
    }
    let score = b.outcome().map(|o| match o {
        Outcome::P1Win => 1.,
        Outcome::P2Win => 0.,
        Outcome::Tie => 0.5,
    });
    json!({"score":score,"turn":b.turn,"decisions":count,"log":if std::env::var_os("NC2000_INVESTIGATION_TRACE").is_some(){Some(&b.log)}else{None}})
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = conformance::load_dex();
    let mode = &args[1];
    let out = Path::new(&args[2]);
    std::fs::create_dir_all(out).unwrap();
    if mode == "positions" {
        let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
        for name in [
            "turn-26-defense",
            "turn-27-offense",
            "turn-28-defense",
            "turn-30-offense",
        ] {
            let path = format!("data/perish-trap-audit-v1/{name}.json");
            let spec =
                nc2000_bot::PositionSpec::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let mut rows = vec![];
            for seed in 2026101901..2026101905 {
                let mut a =
                    nc2000_bot::ProtocolAgent::new(&dex, spec.side, pool.clone(), cfg(), seed);
                a.set_position(&dex, &spec).unwrap();
                let mut done = 0;
                for n in [27000, 81000, 243000] {
                    a.step(&dex, n - done).unwrap();
                    done = n;
                    let choice = a.best(&dex);
                    rows.push(json!({"seed":seed,"n":n,"choice":choice,"policy":serde_json::from_str::<Value>(&a.root_policy(&dex)).unwrap()}));
                }
            }
            let opposite = if name.ends_with("defense") {
                "data/perish-trap-audit-v1/turn-27-offense.json"
            } else {
                "data/perish-trap-audit-v1/turn-28-defense.json"
            };
            let _other =
                nc2000_bot::PositionSpec::parse(&std::fs::read_to_string(opposite).unwrap())
                    .unwrap();
            let b = nc2000_bot::position::synthesize_spec(&dex, &spec, &pool, None, 73).unwrap();
            let start = Instant::now();
            let mut solver = BoundSolver::new(
                &dex,
                BoundConfig {
                    time_budget: Some(Duration::from_secs(3)),
                    work_budget: 100000,
                    node_budget: 12000,
                    cell_cap: 0,
                    ..Default::default()
                },
            );
            let r = solver.solve(&b, None);
            std::fs::write(out.join(format!("{name}.json")),serde_json::to_vec_pretty(&json!({"name":name,"source":path,"scope":"historical reconstructed player information; solver on one prior-imputed world, not exact history or information-set value","rows":rows,"snapshot":snapshot(&dex,&b),"solver":{"value":[r.bounds.lo,r.bounds.hi],"stop":format!("{:?}",r.stop),"runs":r.runs,"ms":start.elapsed().as_secs_f64()*1000.}})).unwrap()).unwrap();
            eprintln!("{name}");
        }
        return;
    }
    if mode == "tactics" {
        for wait in [
            perish_deadline::WaitingMove::Return,
            perish_deadline::WaitingMove::Snore,
        ] {
            let (preview, b) = perish_deadline::position(&dex, wait);
            let mut obs = Observer::new(&preview, 0);
            let mut belief = Belief::pinned_from_battle(&preview, &obs);
            obs.observe(&b, &dex);
            belief.sync(&dex, &obs);
            let mut rows = vec![];
            for seed in 2026101701..2026101705 {
                let mut s = BlindSearch::new(&b, &dex, cfg(), 0, seed);
                for n in [27000, 81000, 243000] {
                    s.step(&dex, &belief, &obs, n - s.iterations());
                    rows.push(json!({"seed":seed,"n":n,"choice":s.best().unwrap().to_input(&dex),"actions":s.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"visits":s.visits(),"means":s.means()}));
                }
            }
            let started = Instant::now();
            let proof = combo_certify::certify(&dex, &b, 0, 100000, 4096);
            let mut solver = BoundSolver::new(
                &dex,
                BoundConfig {
                    time_budget: Some(Duration::from_secs(3)),
                    work_budget: 100000,
                    node_budget: 12000,
                    cell_cap: 0,
                    ..Default::default()
                },
            );
            let r = solver.solve(&b, None);
            let record = json!({"case":wait.key(),"snapshot":snapshot(&dex,&b),"eval":eval01(&b,&dex,&EvalWeights::default()),"search":rows,"proof":proof,"generic_solver":{"lo":r.bounds.lo,"hi":r.bounds.hi,"stop":format!("{:?}",r.stop),"runs":r.runs},"solver_total_ms":started.elapsed().as_secs_f64()*1000.});
            std::fs::write(
                out.join(format!("{}.json", wait.key())),
                serde_json::to_vec_pretty(&record).unwrap(),
            )
            .unwrap();
        }
        return;
    }
    let ts = tasks(&dex, "findings/bot-strength-investigation-v1/corpus.jsonl");
    if mode == "trace" {
        let t = ts.iter().find(|t| t.id == args[3]).unwrap();
        let r = continuation(
            &dex,
            t,
            &args[4],
            args[5].parse().unwrap(),
            args[6].parse().unwrap(),
        );
        std::fs::write(
            out.join("trace.json"),
            serde_json::to_vec_pretty(&r).unwrap(),
        )
        .unwrap();
        return;
    }
    if mode == "solver" {
        for t in ts.iter().filter(|t| t.b.turn > 0) {
            let start = Instant::now();
            let mut solver = BoundSolver::new(
                &dex,
                BoundConfig {
                    time_budget: Some(Duration::from_secs(2)),
                    work_budget: 100000,
                    node_budget: 12000,
                    cell_cap: 0,
                    ..Default::default()
                },
            );
            let r = solver.solve(&t.b, None);
            let one = nc2000_bot::stall::classify_one_sided_heal(&t.b, &dex);
            let two = nc2000_bot::stall::classify_two_sided_heal(&t.b, &dex);
            std::fs::write(out.join(format!("{}.json",t.id)),serde_json::to_vec_pretty(&json!({"id":t.id,"snapshot":snapshot(&dex,&t.b),"value":[r.bounds.lo,r.bounds.hi],"stop":format!("{:?}",r.stop),"runs":r.runs,"ms":start.elapsed().as_secs_f64()*1000.,"one_sided_heal":format!("{one:?}"),"two_sided_heal":format!("{two:?}")})).unwrap()).unwrap();
        }
        return;
    }
    if mode == "continuation" {
        let cases: Value = serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
        let trials: usize = args[4].parse().unwrap();
        let iters: u32 = args[5].parse().unwrap();
        let next = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(||loop{let j=next.fetch_add(1,Ordering::Relaxed);if j>=cases.as_array().unwrap().len()*trials{break;}let c=&cases[j/trials];let t=ts.iter().find(|t|t.id==c["id"].as_str().unwrap()).unwrap();let seed=2026101800+j as u64;let rows:Vec<_>=c["actions"].as_array().unwrap().iter().map(|a|json!({"action":a,"result":continuation(&dex,t,a.as_str().unwrap(),seed,iters)})).collect();std::fs::write(out.join(format!("{j:05}.json")),serde_json::to_vec(&json!({"id":t.id,"trial":j%trials,"seed":seed,"iters":iters,"rows":rows})).unwrap()).unwrap();});
            }
        });
        return;
    }
    assert_eq!(mode, "probe");
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(||loop{let index=next.fetch_add(1,Ordering::Relaxed);let Some(t)=ts.get(index) else{break;};let mut rows=vec![];
  for seed_index in 0..4 {let seed=2026101601+(index as u64)*100+seed_index;let mut s=BlindSearch::new(&t.b,&dex,cfg(),t.side,seed);let started=Instant::now();for n in [27000,81000] {s.step(&dex,&t.belief[t.side],&t.obs[t.side],n-s.iterations());rows.push(json!({"seed":seed,"n":n,"choice":s.best().unwrap().to_input(&dex),"actions":s.actions().iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"visits":s.visits(),"means":s.means(),"nodes":s.node_count(),"ms":started.elapsed().as_secs_f64()*1000.}));}}
  let record=json!({"id":t.id,"side":t.side,"snapshot":snapshot(&dex,&t.b),"rows":rows});std::fs::write(out.join(format!("{}.json",t.id)),serde_json::to_vec_pretty(&record).unwrap()).unwrap();eprintln!("{}/{} {}",index+1,ts.len(),t.id);std::io::stderr().flush().unwrap();
 });
        }
    });
}
