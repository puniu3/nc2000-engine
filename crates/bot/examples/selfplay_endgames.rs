use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use nc2000_bot::bounds::{BoundConfig, BoundSolver};
use nc2000_bot::eval::eval01;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::SelRule;
use nc2000_bot::{Agent, BlindAgent, EvalWeights, RmConfig, SkuctSearch, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
struct Decision {
    key: String,
    choices: [Option<usize>; 2],
}

#[derive(Serialize, Deserialize)]
struct Game {
    schema: String,
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    iterations: u32,
    decisions: Vec<Decision>,
    samples: Vec<usize>,
    outcome: String,
}

fn alive(b: &Battle) -> [usize; 2] {
    [0, 1].map(|s| {
        b.sides[s]
            .party
            .iter()
            .filter(|&&i| !b.sides[s].roster[i as usize].fainted)
            .count()
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = conformance::load_dex();
    match args.get(1).map(String::as_str) {
        Some("generate") | Some("generate-weighted") => {
            let path = &args[2];
            let games: usize = args[3].parse().unwrap();
            let iterations: u32 = args[4].parse().unwrap();
            let seed: u64 = args[5].parse().unwrap();
            let pool = Arc::new(load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json")));
            let catalog = load_meta_pool(Path::new("data/team-pool-v2/team-pool.json"));
            let weighted = args[1] == "generate-weighted";
            let weights: Vec<f64> = if weighted {
                let catalog_json: serde_json::Value = serde_json::from_slice(&std::fs::read("data/team-pool-v2/team-pool.json").unwrap()).unwrap();
                catalog_json["teams"].as_array().unwrap().iter().zip(&catalog.teams).map(|(row, team)| {
                    assert_eq!(row["id"].as_str().unwrap(), team.id);
                    row["drawWeight"].as_f64().unwrap()
                }).collect()
            } else { Vec::new() };
            assert!(!weighted || (weights.len() == catalog.teams.len() && weights.iter().all(|w| w.is_finite() && *w >= 0.0) && weights.iter().sum::<f64>() > 0.0));
            let mut rng = SplitMix64::new(seed);
            let mut out = std::io::BufWriter::new(std::fs::File::create_new(path).unwrap());
            for game in 0..games {
                let start = Instant::now();
                let teams = [0, 1].map(|_| {
                    let index = if weighted {
                        let mut draw = rng.next_f64() * weights.iter().sum::<f64>();
                        weights.iter().position(|w| { draw -= w; draw < 0.0 }).unwrap_or(weights.len() - 1)
                    } else { rng.below(catalog.teams.len()) };
                    catalog.teams[index].sets.clone()
                });
                let battle_seed = rng.battle_seed();
                let mut b = Battle::from_fixture(&dex, &battle_seed, &teams[0], &teams[1]).unwrap();
                let cfg = RmConfig { iterations, c: 0.4, rule: SelRule::Ucb, ..Default::default() };
                let mut agents = [0, 1].map(|_| BlindAgent::new(cfg.clone(), pool.clone(), None, rng.next()));
                let mut decisions = Vec::new();
                let mut classes: BTreeMap<[usize; 2], Vec<usize>> = BTreeMap::new();
                while b.outcome().is_none() && b.turn <= if weighted { 1000 } else { 300 } {
                    let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
                    let n = alive(&b);
                    if n.iter().all(|&n| (1..=2).contains(&n)) {
                        classes.entry(n).or_default().push(decisions.len());
                    }
                    let choices = [0, 1].map(|s| {
                        if acts[s].is_empty() { None } else {
                            let chosen = agents[s].choose(&b, &dex, s, &acts[s]);
                            Some(acts[s].iter().position(|&a| a == chosen).unwrap())
                        }
                    });
                    decisions.push(Decision { key: b.state_key128().to_string(), choices });
                    b.apply_choices(&dex, [0, 1].map(|s| choices[s].map(|i| acts[s][i]))).unwrap();
                }
                let mut samples: Vec<_> = classes.values().flat_map(|v| [v[0], *v.last().unwrap()]).collect();
                samples.sort_unstable();
                samples.dedup();
                let record = Game { schema: "nc2000-selfplay-endgames-v1".into(), game, seed: battle_seed, teams, iterations, decisions, samples, outcome: format!("{:?}", b.outcome()) };
                serde_json::to_writer(&mut out, &record).unwrap();
                writeln!(&mut out).unwrap();
                out.flush().unwrap();
                eprintln!("game {game}: {} samples, {} decisions, {}, {:.1}s", record.samples.len(), record.decisions.len(), record.outcome, start.elapsed().as_secs_f64());
            }
        }
        Some("measure") => {
            let file = std::io::BufReader::new(std::fs::File::open(&args[2]).unwrap());
            let work: usize = args[3].parse().unwrap();
            let cell_cap: usize = args[4].parse().unwrap();
            let iterations: u32 = args[5].parse().unwrap();
            let limit: usize = args.get(6).filter(|s| !s.starts_with("--")).map(|s| s.parse().unwrap()).unwrap_or(usize::MAX);
            let time_budget = args.iter().position(|a| a == "--milliseconds").map(|i| std::time::Duration::from_millis(args[i+1].parse().unwrap()));
            let reuse = args.iter().any(|a| a == "--reuse");
            let node_budget = args.iter().position(|a| a == "--nodes").map(|i|args[i+1].parse().unwrap()).unwrap_or(120_000);
            let cfg = BoundConfig { time_budget, work_budget: work, cell_cap, node_budget, reuse_unchanged_backups: !args.iter().any(|a| a == "--no-backup-cache"), ..Default::default() };
            let mut count = 0;
            for line in file.lines() {
                let game: Game = serde_json::from_str(&line.unwrap()).unwrap();
                assert_eq!(game.schema, "nc2000-selfplay-endgames-v1");
                let mut solver = BoundSolver::new(&dex, cfg.clone());
                let mut b = Battle::from_fixture(&dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
                for (decision, record) in game.decisions.iter().enumerate() {
                    assert_eq!(b.state_key128().to_string(), record.key, "replay mismatch game {} decision {decision}", game.game);
                    if game.samples.contains(&decision) {
                        if count == limit { return; }
                        count += 1;
                        let start = Instant::now();
                        if reuse { solver.retain_root(&b); } else { solver = BoundSolver::new(&dex, cfg.clone()); }
                        let before = solver.stats.clone();
                        let solve_started = Instant::now();
                        let report = solver.solve(&b, None);
                        let solve_seconds = solve_started.elapsed().as_secs_f64();
                        let seconds = start.elapsed().as_secs_f64();
                        let start = Instant::now();
                        let mut search = SkuctSearch::new(&b, &dex, RmConfig { iterations, c: 0.4, rule: SelRule::Ucb, ..Default::default() }, 71);
                        search.step(&dex, iterations);
                        let search_seconds = start.elapsed().as_secs_f64();
                        let certified_actions = [0, 1].map(|s| solver.action_bounds(&b,s).into_iter().map(|(a,v)| json!({"action":a.map(|a|a.to_input(&dex)),"lo":v.lo,"hi":v.hi})).collect::<Vec<_>>());
                        let policies = [0, 1].map(|s| {
                            let means = search.means(s);
                            let visits = search.visits(s);
                            search.actions(s).iter().enumerate().map(|(i, &a)| json!({"action": a.to_input(&dex), "visits": visits[i], "mean": means[i], "chosen": Some(a) == search.best(s)})).collect::<Vec<_>>()
                        });
                        let mons = [0, 1].map(|s| b.sides[s].party.iter().map(|&i| {
                            let p = &b.sides[s].roster[i as usize];
                            json!({"species":dex.species.key(p.species),"hp":p.hp,"maxhp":p.maxhp,"status":format!("{:?}",p.status),"moves":p.move_slots.iter().map(|m|json!({"id":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})
                        }).collect::<Vec<_>>());
                        println!("{}", json!({"game":game.game,"decision":decision,"key":record.key,"turn":b.turn,"alive":alive(&b),"mons":mons,"work_budget":work,"node_budget":node_budget,"time_budget_ms":time_budget.map(|t|t.as_millis()),"reuse":reuse,"cell_cap":cell_cap,"lo":report.bounds.lo,"hi":report.bounds.hi,"stop":format!("{:?}",report.stop),"runs":report.runs,"seconds":seconds,"solve_seconds":solve_seconds,"nodes":solver.node_count(),"closed":solver.closed_count(),"reused_backups":solver.stats.reused_backups-before.reused_backups,"eager_runs":solver.stats.eager_runs-before.eager_runs,"eager_failures":solver.stats.eager_failures-before.eager_failures,"reused_nodes":solver.stats.reused_nodes-before.reused_nodes,"certified_actions":certified_actions,"lp_solves":solver.stats.lp_solves-before.lp_solves,"trials":solver.stats.trials-before.trials,"dominated_rows":solver.stats.dominated_rows-before.dominated_rows,"dominated_cols":solver.stats.dominated_cols-before.dominated_cols,"eval":eval01(&b,&dex,&EvalWeights::default()),"search_iterations":iterations,"search_seconds":search_seconds,"policies":policies}));
                        eprintln!("g{} d{decision} {:?}: [{:.4},{:.4}] {:?} {} runs {:.2}s",game.game,alive(&b),report.bounds.lo,report.bounds.hi,report.stop,report.runs,seconds);
                    }
                    let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
                    b.apply_choices(&dex, [0, 1].map(|s| record.choices[s].map(|i| acts[s][i]))).unwrap();
                }
            }
        }
        _ => println!("selfplay_endgames generate PATH GAMES ITERATIONS SEED | measure PATH WORK CELL_CAP ITERATIONS [LIMIT] [--milliseconds MS] [--nodes N] [--reuse] [--no-backup-cache]"),
    }
}
