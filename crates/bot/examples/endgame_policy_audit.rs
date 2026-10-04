use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Bounds, RootPayoffs};
use nc2000_bot::player::action_input;
use nc2000_bot::{Agent, MaxDamageAgent, RmConfig, SkuctSearch};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    corpus: String,
    corpus_sha256: String,
    baseline: String,
    baseline_sha256: String,
    reference_ms: u64,
    node_budget: usize,
    work_budget: usize,
    cell_cap: usize,
    tolerance: f64,
    iterations: Vec<u32>,
    search_seeds: usize,
    seed: u64,
    threads: usize,
    limit: Option<usize>,
    interpretation: String,
}

#[derive(Deserialize)]
struct Decision {
    key: String,
    choices: [Option<usize>; 2],
}

#[derive(Deserialize)]
struct Game {
    schema: String,
    game: usize,
    seed: String,
    teams: [Vec<PokemonSet>; 2],
    decisions: Vec<Decision>,
    samples: Vec<usize>,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bound(b: Bounds) -> Value {
    json!({"lo": b.lo, "hi": b.hi})
}

fn assessment(matrix: &RootPayoffs, side: usize, policy: &[f64]) -> Value {
    json!({"policy":policy,"guarantee":bound(matrix.policy_value(side,policy)),
        "loss":bound(matrix.policy_loss(side,policy))})
}

fn reconstruct(dex: &Dex, game: &Game, decision: usize) -> Battle {
    let mut battle = Battle::from_fixture(dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
    for record in &game.decisions[..decision] {
        assert_eq!(battle.state_key128().to_string(), record.key);
        let choices = [0, 1].map(|side| battle.legal_choices(dex, side));
        battle
            .apply_choices(
                dex,
                [0, 1].map(|side| record.choices[side].map(|index| choices[side][index])),
            )
            .unwrap();
    }
    assert_eq!(
        battle.state_key128().to_string(),
        game.decisions[decision].key
    );
    battle
}

fn audit(
    dex: &Dex,
    game: &Game,
    decision: usize,
    archived: &[Value],
    protocol: &Protocol,
) -> Value {
    let mut battle = reconstruct(dex, game, decision);
    let legal = [0, 1].map(|side| battle.legal_choices(dex, side));
    let names = legal.clone().map(|aa| {
        aa.into_iter()
            .map(|a| action_input(dex, a))
            .collect::<Vec<_>>()
    });
    let mut targets = Vec::new();
    for side in 0..2 {
        if legal[side].is_empty() {
            continue;
        }
        for action in 0..legal[side].len() {
            let mut p = vec![0.0; legal[side].len()];
            p[action] = 1.0;
            targets.push((side, p));
        }
        targets.push((
            side,
            vec![1.0 / legal[side].len() as f64; legal[side].len()],
        ));
    }
    let started = Instant::now();
    let mut solver = BoundSolver::new(
        dex,
        BoundConfig {
            time_budget: Some(Duration::from_millis(protocol.reference_ms)),
            work_budget: protocol.work_budget,
            node_budget: protocol.node_budget,
            cell_cap: protocol.cell_cap,
            certified_action_pruning: false,
            ..BoundConfig::default()
        },
    );
    let solved = solver.solve_policy_losses(&battle, &targets, protocol.tolerance);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let matrix = solver.root_payoffs(&battle).unwrap();
    for side in 0..2 {
        let expected: Vec<_> = if legal[side].is_empty() {
            vec![None]
        } else {
            legal[side].iter().copied().map(Some).collect()
        };
        assert_eq!(matrix.actions[side], expected);
    }
    let mut endpoints = Vec::new();
    let mut discriminating = false;
    for side in 0..2 {
        if legal[side].is_empty() {
            continue;
        }
        let archived = archived
            .iter()
            .find(|r| r["game"] == game.game && r["decision"] == decision && r["side"] == side)
            .expect("every actionable endpoint must have an archived baseline");
        assert_eq!(archived["state_key"], battle.state_key128().to_string());
        let mut p = vec![0.0; legal[side].len()];
        for item in archived["policy"].as_array().unwrap() {
            let index = names[side]
                .iter()
                .position(|name| *name == item["action"].as_str().unwrap())
                .unwrap();
            p[index] = item["probability"].as_f64().unwrap();
        }
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        let pure: Vec<_> = (0..p.len())
            .map(|a| {
                let mut policy = vec![0.0; p.len()];
                policy[a] = 1.0;
                assessment(&matrix, side, &policy)
            })
            .collect();
        let all_good = pure
            .iter()
            .all(|r| r["loss"]["hi"].as_f64().unwrap() <= protocol.tolerance);
        let has_good = pure
            .iter()
            .any(|r| r["loss"]["hi"].as_f64().unwrap() <= protocol.tolerance);
        let has_bad = pure
            .iter()
            .any(|r| r["loss"]["lo"].as_f64().unwrap() > protocol.tolerance);
        discriminating |= has_good && has_bad;
        let greedy = MaxDamageAgent::conformant().choose(&battle, dex, side, &legal[side]);
        let greedy_index = legal[side].iter().position(|&a| a == greedy).unwrap();
        let oracle = solved.strategies.as_ref().map(|ss| {
            let p: Vec<_> = matrix.actions[side]
                .iter()
                .map(|a| {
                    ss[side]
                        .actions
                        .iter()
                        .find(|(b, _)| a == b)
                        .map_or(0.0, |(_, mass)| *mass)
                })
                .collect();
            assessment(&matrix, side, &p)
        });
        endpoints.push(
            json!({"side":side,"actions":names[side],"forced":p.len()==1,
            "all_actions_good":all_good,"has_good_action":has_good,"has_bad_action":has_bad,
            "archived":assessment(&matrix,side,&p),"archived_choice":archived["choice"],
            "archived_iterations":archived["iterations"],"pure_actions":pure,
            "uniform":assessment(&matrix,side,&vec![1.0/p.len() as f64;p.len()]),
            "max_damage":pure[greedy_index],"oracle":oracle}),
        );
    }
    let mut search_records = Vec::new();
    if discriminating {
        for seed_index in 0..protocol.search_seeds {
            let seed = protocol.seed
                ^ (game.game as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ (decision as u64).wrapping_mul(0xbf58476d1ce4e5b9)
                ^ (seed_index as u64).wrapping_mul(0x94d049bb133111eb);
            let mut search = SkuctSearch::new(
                &battle,
                dex,
                RmConfig {
                    c: 0.4,
                    ..Default::default()
                },
                seed,
            );
            let started = Instant::now();
            for &iterations in &protocol.iterations {
                search.step(dex, iterations - search.iterations());
                let choices: [Option<String>; 2] =
                    [0, 1].map(|s| search.best(s).map(|a| action_input(dex, a)));
                search_records.push(json!({"seed_index":seed_index,"seed":seed,"iterations":iterations,
                    "choices":choices,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"nodes":search.node_count()}));
            }
        }
        for endpoint in &mut endpoints {
            let side = endpoint["side"].as_u64().unwrap() as usize;
            let rows: Vec<_> = protocol
                .iterations
                .iter()
                .map(|&iterations| {
                    let mut p = vec![0.0; legal[side].len()];
                    for row in search_records
                        .iter()
                        .filter(|r| r["iterations"] == iterations)
                    {
                        let choice = row["choices"][side].as_str().unwrap();
                        p[names[side].iter().position(|a| a == choice).unwrap()] +=
                            1.0 / protocol.search_seeds as f64;
                    }
                    json!({"iterations":iterations,"empirical_mixture":assessment(&matrix,side,&p)})
                })
                .collect();
            endpoint["budget_ladder"] = json!(rows);
        }
    }
    json!({"game":game.game,"decision":decision,"state_key":battle.state_key128().to_string(),
        "turn":battle.turn,"alive":([0,1].map(|side|battle.sides[side].party.iter().filter(|&&i| !battle.sides[side].roster[i as usize].fainted).count())),
        "reference":{"value":bound(matrix.value),"cells":matrix.cells.iter().map(|b|bound(*b)).collect::<Vec<_>>(),
            "actions":matrix.actions.map(|aa|aa.into_iter().map(|a|a.map(|a|action_input(dex,a))).collect::<Vec<_>>()),
            "stop":format!("{:?}",solved.stop),"runs":solved.runs,"elapsed_ms":elapsed_ms,
            "nodes":solver.node_count(),"closed":solver.closed_count()},
        "discriminating":discriminating,"endpoints":endpoints,"search_records":search_records})
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "endgame_policy_audit PROTOCOL.json OUTPUT_DIRECTORY"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-policy-gap-v1");
    assert!(protocol.threads > 0 && protocol.search_seeds > 0 && protocol.reference_ms > 0);
    assert!(protocol.iterations.windows(2).all(|w| w[0] < w[1]));
    let corpus = std::fs::read(&protocol.corpus).unwrap();
    let baseline = std::fs::read(&protocol.baseline).unwrap();
    assert_eq!(sha(&corpus), protocol.corpus_sha256);
    assert_eq!(sha(&baseline), protocol.baseline_sha256);
    let games: Vec<Game> = BufReader::new(corpus.as_slice())
        .lines()
        .map(|s| serde_json::from_str(&s.unwrap()).unwrap())
        .collect();
    let archived: Vec<Value> = serde_json::from_slice(&baseline).unwrap();
    let mut jobs = Vec::new();
    for (i, game) in games.iter().enumerate() {
        assert_eq!(game.schema, "nc2000-selfplay-endgames-v1");
        for &decision in &game.samples {
            jobs.push((i, decision));
        }
    }
    if let Some(limit) = protocol.limit {
        jobs.truncate(limit);
    }
    let directory = Path::new(&args[2]);
    std::fs::create_dir(directory).unwrap();
    let git = |args: &[&str]| {
        String::from_utf8(
            std::process::Command::new("git")
                .args(args)
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string()
    };
    let sources = [
        "crates/bot/src/bounds.rs",
        "crates/bot/examples/endgame_policy_audit.rs",
    ];
    let manifest = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),
        "protocol_sha256":sha(&bytes),"revision":git(&["rev-parse","HEAD"]),"dirty":git(&["status","--porcelain"]),
        "binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),
        "source_hashes":sources.map(|p|(p,sha(&std::fs::read(p).unwrap()))),
        "expected_positions":jobs.len(),"source_games":games.len(),
        "interpretation":protocol.interpretation});
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    let dex = conformance::load_dex();
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(game, decision)) = jobs.get(index) else {
                    break;
                };
                let game = &games[game];
                let row = audit(&dex, game, decision, &archived, &protocol);
                std::fs::write(
                    directory.join(format!("game-{:03}-d{decision:04}.json", game.game)),
                    serde_json::to_vec_pretty(&row).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "{}/{} game={} decision={} stop={} reference_ms={:.0} ladder={}",
                    index + 1,
                    jobs.len(),
                    game.game,
                    decision,
                    row["reference"]["stop"],
                    row["reference"]["elapsed_ms"].as_f64().unwrap(),
                    row["discriminating"]
                );
            });
        }
    });
}
