use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};
use std::time::Instant;

use nc2000_bot::endgame::{EndgameAgent, EndgameConfig, EndgameSession};
use nc2000_bot::player::{action_input, PlayerChannel};
use nc2000_bot::position::PositionSpec;
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::{Agent, ProtocolAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    purpose: String,
    corpus: String,
    corpus_sha256: String,
    prior: String,
    per_class: usize,
    fractions: Vec<f64>,
    milliseconds: f64,
    threads: usize,
    seed: u64,
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

struct Task {
    game: usize,
    decision: usize,
    side: usize,
    alive: [usize; 2],
    battle: Battle,
    public: PositionSpec,
}

fn config() -> RmConfig {
    RmConfig {
        c: 0.4,
        ..Default::default()
    }
}

fn tasks(dex: &nc2000_engine::dex::Dex, protocol: &Protocol, pool: &MetaPool) -> Vec<Task> {
    let mut counts = BTreeMap::<[usize; 2], usize>::new();
    let mut tasks = Vec::new();
    for line in BufReader::new(std::fs::File::open(&protocol.corpus).unwrap()).lines() {
        let game: Game = serde_json::from_str(&line.unwrap()).unwrap();
        assert_eq!(game.schema, "nc2000-selfplay-endgames-v1");
        let mut battle =
            Battle::from_fixture(dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
        let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
        let mut agents = [0, 1].map(|side| {
            let mut agent = ProtocolAgent::new(
                dex,
                side,
                pool.clone(),
                config(),
                protocol.seed ^ side as u64,
            );
            agent.set_own_team(game.teams[side].clone());
            agent
        });
        for (decision, record) in game.decisions.iter().enumerate() {
            assert_eq!(
                battle.state_key128().to_string(),
                record.key,
                "replay mismatch: game {} decision {decision}",
                game.game
            );
            let alive = [0, 1].map(|side| {
                battle.sides[side]
                    .party
                    .iter()
                    .filter(|&&slot| !battle.sides[side].roster[slot as usize].fainted)
                    .count()
            });
            let selected = game.samples.contains(&decision)
                && counts.get(&alive).copied().unwrap_or(0) < protocol.per_class;
            for side in 0..2 {
                let frame = channels[side].frame(&mut battle, dex).unwrap();
                for line in frame.lines {
                    agents[side].push_line(dex, &line);
                }
                let actionable = agents[side]
                    .on_request(dex, &frame.request.to_string())
                    .unwrap();
                if selected && actionable {
                    tasks.push(Task {
                        game: game.game,
                        decision,
                        side,
                        alive,
                        battle: battle.clone(),
                        public: agents[side].to_position_spec(dex).unwrap(),
                    });
                }
            }
            if selected {
                *counts.entry(alive).or_default() += 1;
            }
            let actions = [0, 1].map(|side| battle.legal_choices(dex, side));
            battle
                .apply_choices(
                    dex,
                    [0, 1].map(|side| record.choices[side].map(|index| actions[side][index])),
                )
                .unwrap();
        }
    }
    eprintln!(
        "selected classes: {counts:?}; {} side decisions",
        tasks.len()
    );
    tasks
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: endgame_integration_audit PROTOCOL.json NEW-OUTPUT.jsonl");
        std::process::exit(2);
    }
    let protocol_bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&protocol_bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-endgame-integration-audit-v1");
    assert!(protocol.threads > 0 && protocol.per_class > 0);
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(std::fs::read(&protocol.corpus).unwrap())
        ),
        protocol.corpus_sha256
    );
    for &fraction in &protocol.fractions {
        EndgameSession::new(
            EndgameConfig {
                solver_fraction: fraction,
                ..Default::default()
            },
            0.0,
            protocol.milliseconds,
        )
        .unwrap();
    }
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new(&protocol.prior));
    let tasks = tasks(&dex, &protocol, &pool);
    let mut jobs = Vec::new();
    for task in 0..tasks.len() {
        for public in [false, true] {
            for &fraction in &protocol.fractions {
                jobs.push((task, public, fraction));
            }
        }
    }
    let mut rng = SplitMix64::new(protocol.seed);
    for i in (1..jobs.len()).rev() {
        let j = rng.below(i + 1);
        jobs.swap(i, j);
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(&args[2]).unwrap());
    let command = |args: &[&str]| {
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
    writeln!(out, "{}", json!({"type":"metadata","protocol":serde_json::from_slice::<serde_json::Value>(&protocol_bytes).unwrap(),"protocol_sha256":format!("{:x}",Sha256::digest(&protocol_bytes)),"revision":command(&["rev-parse","HEAD"]),"dirty":command(&["status","--porcelain"]),"binary_sha256":format!("{:x}",Sha256::digest(std::fs::read(std::env::current_exe().unwrap()).unwrap())),"prior_sha256":format!("{:x}",Sha256::digest(std::fs::read(&protocol.prior).unwrap())),"jobs":jobs.len(),"purpose":protocol.purpose,"interpretation":protocol.interpretation})).unwrap();
    out.flush().unwrap();
    let writer = Mutex::new(out);
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let job = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(task_index, public, fraction)) = jobs.get(job) else { break; };
                let task = &tasks[task_index];
                let seed = protocol.seed ^ (task_index as u64).wrapping_mul(0x9e3779b97f4a7c15);
                let cfg = EndgameConfig { solver_fraction: fraction, ..Default::default() };
                let started = Instant::now();
                let mut clock = || started.elapsed().as_secs_f64() * 1000.0;
                let (choice, policy, stats, iterations, nodes, legal) = if public {
                    let mut agent = ProtocolAgent::new(&dex, task.side, pool.clone(), config(), seed);
                    agent.set_position(&dex, &task.public).unwrap();
                    let mut session = EndgameSession::new(cfg, 0.0, protocol.milliseconds).unwrap();
                    while !session.expired(clock()) {
                        let previous = agent.iterations();
                        if agent.step_with_endgame(&dex, 1, &mut session, &mut clock).unwrap() == previous { break; }
                    }
                    let choice = agent.best(&dex).unwrap();
                    let mut state = task.battle.clone();
                    let legal = state.legal_choices(&dex, task.side).iter().any(|&a| action_input(&dex, a) == choice);
                    (choice, json!(null), session.stats, agent.iterations(), agent.search().unwrap().node_count(), legal)
                } else {
                    let mut state = task.battle.clone();
                    let choices = state.legal_choices(&dex, task.side);
                    let mut agent = EndgameAgent::new(cfg, config(), protocol.milliseconds, seed).unwrap();
                    let choice = agent.choose(&state, &dex, task.side, &choices);
                    let policy = agent.root_policy(&state, &dex, task.side, &choices);
                    (action_input(&dex, choice), json!(choices.iter().zip(policy).map(|(&a,p)|json!({"action":action_input(&dex,a),"probability":p})).collect::<Vec<_>>()), agent.stats, agent.search_iterations, agent.search_nodes, choices.contains(&choice))
                };
                assert!(legal, "illegal action at task {task_index}");
                let row = json!({"type":"decision","job":job,"task":task_index,"game":task.game,"decision":task.decision,"side":task.side,"alive":task.alive,"turn":task.battle.turn,"state_key":task.battle.state_key128().to_string(),"information":if public {"public-protocol"} else {"full"},"solver_fraction":fraction,"budget_ms":protocol.milliseconds,"elapsed_ms":clock(),"choice":choice,"policy":policy,"stats":stats,"search_iterations":iterations,"search_nodes":nodes,"legal":legal});
                let mut out = writer.lock().unwrap();
                writeln!(out, "{row}").unwrap();
                out.flush().unwrap();
                eprintln!("{}/{} g{} d{} p{} {} share={fraction} replacements={} overrides={}", job+1,jobs.len(),task.game,task.decision,task.side+1,if public {"public"}else{"full"},stats.value_replacements,stats.root_overrides);
            });
        }
    });
}
