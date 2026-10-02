use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Stop};
use nc2000_bot::endgame::{EndgameAgent, EndgameConfig, EndgameSession};
use nc2000_bot::player::{action_input, PlayerChannel};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::{Agent, ProtocolAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
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
    prior: String,
    prior_sha256: String,
    solver_fraction: f64,
    milliseconds: f64,
    reference_ms: f64,
    seed: u64,
    threads: usize,
    selection: String,
    interpretation: String,
    #[serde(default)]
    calibration: bool,
    candidate_selection: Option<String>,
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

enum Seat {
    Full(EndgameAgent),
    Public(Box<ProtocolAgent>, PlayerChannel, f64),
}

fn config() -> RmConfig {
    RmConfig {
        c: 0.4,
        ..Default::default()
    }
}

impl Seat {
    fn new(
        dex: &Dex,
        pool: &MetaPool,
        game: &Game,
        side: usize,
        public: bool,
        fraction: f64,
        protocol: &Protocol,
        seed: u64,
    ) -> Self {
        if public {
            let mut agent = ProtocolAgent::new(dex, side, pool.clone(), config(), seed);
            agent.set_own_team(game.teams[side].clone());
            Self::Public(Box::new(agent), PlayerChannel::new(side), fraction)
        } else {
            Self::Full(
                EndgameAgent::new(
                    EndgameConfig {
                        solver_fraction: fraction,
                        ..Default::default()
                    },
                    config(),
                    protocol.milliseconds,
                    seed,
                )
                .unwrap(),
            )
        }
    }

    fn observe(&mut self, dex: &Dex, battle: &mut Battle) -> bool {
        if let Self::Public(agent, channel, _) = self {
            let frame = channel.frame(battle, dex).unwrap();
            for line in frame.lines {
                agent.push_line(dex, &line);
            }
            agent.on_request(dex, &frame.request.to_string()).unwrap()
        } else {
            true
        }
    }

    fn choose(
        &mut self,
        dex: &Dex,
        battle: &mut Battle,
        side: usize,
        milliseconds: f64,
    ) -> (Option<SearchChoice>, Value) {
        let started = Instant::now();
        let mut clock = || started.elapsed().as_secs_f64() * 1000.0;
        let actionable = self.observe(dex, battle);
        let legal = battle.legal_choices(dex, side);
        if legal.is_empty() {
            assert!(!actionable || matches!(self, Self::Full(_)));
            return (None, json!({"wait":true}));
        }
        assert!(actionable);
        let (choice, policy, iterations, nodes, stats) = match self {
            Self::Full(agent) => {
                let choice = agent.choose(battle, dex, side, &legal);
                let policy = agent.root_policy(battle, dex, side, &legal);
                (
                    choice,
                    json!(legal
                        .iter()
                        .zip(policy)
                        .map(|(&a, p)| json!({"action":action_input(dex,a),"probability":p}))
                        .collect::<Vec<_>>()),
                    agent.search_iterations,
                    agent.search_nodes,
                    agent.stats.clone(),
                )
            }
            Self::Public(agent, _, fraction) => {
                let mut session = EndgameSession::new(
                    EndgameConfig {
                        solver_fraction: *fraction,
                        ..Default::default()
                    },
                    0.0,
                    milliseconds,
                )
                .unwrap();
                while !session.expired(clock()) {
                    let previous = agent.iterations();
                    if agent
                        .step_with_endgame(dex, 1, &mut session, &mut clock)
                        .unwrap()
                        == previous
                    {
                        break;
                    }
                }
                let input = agent.best(dex).unwrap();
                let choice = *legal
                    .iter()
                    .find(|&&a| action_input(dex, a) == input)
                    .unwrap_or_else(|| panic!("illegal holdout action {input}"));
                (
                    choice,
                    json!(null),
                    agent.iterations(),
                    agent.search().unwrap().node_count(),
                    session.stats,
                )
            }
        };
        assert!(legal.contains(&choice));
        (
            Some(choice),
            json!({"choice":action_input(dex,choice),"policy":policy,"iterations":iterations,"search_nodes":nodes,"stats":stats,"elapsed_ms":clock(),"legal":true,"forced":legal.len()==1}),
        )
    }
}

fn initial(
    dex: &Dex,
    pool: &MetaPool,
    game: &Game,
    decision: usize,
    public: bool,
    leg: usize,
    protocol: &Protocol,
    seed: u64,
) -> (Battle, [Seat; 2]) {
    let mut battle = Battle::from_fixture(dex, &game.seed, &game.teams[0], &game.teams[1]).unwrap();
    battle.log_enabled = true;
    let mut seats = [0, 1].map(|side| {
        Seat::new(
            dex,
            pool,
            game,
            side,
            public,
            if side == leg {
                protocol.solver_fraction
            } else {
                0.0
            },
            protocol,
            seed,
        )
    });
    for record in &game.decisions[..decision] {
        assert_eq!(battle.state_key128().to_string(), record.key);
        for seat in &mut seats {
            seat.observe(dex, &mut battle);
        }
        let legal = [0, 1].map(|side| battle.legal_choices(dex, side));
        battle
            .apply_choices(
                dex,
                [0, 1].map(|side| record.choices[side].map(|i| legal[side][i])),
            )
            .unwrap();
    }
    assert_eq!(
        battle.state_key128().to_string(),
        game.decisions[decision].key
    );
    (battle, seats)
}

fn reference(dex: &Dex, battle: &Battle, milliseconds: f64) -> Value {
    let started = Instant::now();
    let mut solver = BoundSolver::new(
        dex,
        BoundConfig {
            work_budget: 64,
            cell_cap: 0,
            node_budget: 120_000,
            dead_damage_quotient: false,
            certified_action_pruning: false,
            ..Default::default()
        },
    );
    let mut last = None;
    let mut actions = [Vec::new(), Vec::new()];
    while started.elapsed().as_secs_f64() * 1000.0 < milliseconds {
        solver.cfg.node_budget = 120_000usize.saturating_sub(solver.closed_count());
        let report = solver.solve(battle, None);
        for side in 0..2 {
            let next = solver.action_bounds(battle, side);
            if !next.is_empty() {
                actions[side] = next;
            }
        }
        let done =
            report.bounds.width() <= 0.02 || report.runs == 0 || report.stop == Stop::NodeBudget;
        last = Some(report);
        if done {
            break;
        }
    }
    json!({"bounds":last.as_ref().map(|r|json!({"lo":r.bounds.lo,"hi":r.bounds.hi})),"last_batch_stop":last.as_ref().map(|r|format!("{:?}",r.stop)),"external_time_exhausted":started.elapsed().as_secs_f64()*1000.0>=milliseconds,"elapsed_ms":started.elapsed().as_secs_f64()*1000.0,"entries":solver.node_count()+solver.closed_count(),"actions":actions.map(|row|row.into_iter().map(|(a,b)|json!({"action":a.map(|a|action_input(dex,a)),"lo":b.lo,"hi":b.hi})).collect::<Vec<_>>()),"scope":"full-information reference only; broad bounds are unresolved; action intervals may precede the last value update"})
}

fn play(
    dex: &Dex,
    pool: &MetaPool,
    game: &Game,
    decision: usize,
    public: bool,
    leg: usize,
    protocol: &Protocol,
    seed: u64,
    directory: &Path,
) {
    let info = if public { "public" } else { "full" };
    let path = directory.join(format!(
        "game-{:03}-d{decision:04}-{info}-{leg}.jsonl",
        game.game
    ));
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(path).unwrap());
    let (mut battle, mut seats) = initial(dex, pool, game, decision, public, leg, protocol, seed);
    writeln!(out,"{}",json!({"type":"continuation","source_game":game.game,"source_decision":decision,"information":info,"candidate_side":leg,"seed":seed,"state_key":battle.state_key128().to_string(),"turn":battle.turn})).unwrap();
    out.flush().unwrap();
    let mut step = 0;
    while battle.outcome().is_none() {
        let key = battle.state_key128().to_string();
        let turn = battle.turn;
        let mut choices = [None; 2];
        let mut reports = [json!(null), json!(null)];
        for side in [step % 2, 1 - step % 2] {
            (choices[side], reports[side]) =
                seats[side].choose(dex, &mut battle, side, protocol.milliseconds);
        }
        battle.apply_choices(dex, choices).unwrap();
        writeln!(
            out,
            "{}",
            json!({"type":"decision","step":step,"turn":turn,"state_key":key,"sides":reports})
        )
        .unwrap();
        out.flush().unwrap();
        step += 1;
        assert!(step <= 4000, "holdout emergency guard");
    }
    let score = match battle.outcome().unwrap() {
        Outcome::Tie => 0.5,
        Outcome::P1Win => {
            if leg == 0 {
                1.0
            } else {
                0.0
            }
        }
        Outcome::P2Win => {
            if leg == 1 {
                1.0
            } else {
                0.0
            }
        }
    };
    writeln!(out,"{}",json!({"type":"result","source_game":game.game,"source_decision":decision,"information":info,"candidate_side":leg,"candidate_score":score,"outcome":format!("{:?}",battle.outcome()),"decisions":step,"engine_cap":battle.turn>=1000})).unwrap();
    out.flush().unwrap();
    eprintln!("g{} d{decision} {info} p{} complete", game.game, leg + 1);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: endgame_holdout PROTOCOL.json NEW-DIRECTORY"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    if !protocol.calibration {
        let selection: Value = serde_json::from_slice(
            &std::fs::read(
                protocol
                    .candidate_selection
                    .as_ref()
                    .expect("holdout requires a frozen development selection"),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(selection["complete"], true);
        let chosen = selection["selected_for_confirmation"]
            .as_str()
            .expect("no selected candidate");
        let arm = &selection["arms"][chosen];
        assert_eq!(
            arm["protocol"]["candidate"]["solver_fraction"]
                .as_f64()
                .unwrap(),
            protocol.solver_fraction
        );
        assert_eq!(
            arm["protocol"]["milliseconds"].as_f64().unwrap(),
            protocol.milliseconds
        );
        assert!(
            std::process::Command::new("git")
                .args([
                    "diff",
                    "--quiet",
                    arm["revision"].as_str().unwrap(),
                    "--",
                    "crates/bot/src",
                    "crates/engine/src",
                    "data/gen2stadium2.json"
                ])
                .status()
                .unwrap()
                .success(),
            "solver or engine differs from the frozen candidate"
        );
    }
    assert_eq!(protocol.schema, "nc2000-endgame-holdout-v1");
    assert_eq!(protocol.selection, "all-recorded-samples");
    assert!(
        protocol.threads > 0
            && protocol.reference_ms.is_finite()
            && (0.0..=60000.0).contains(&protocol.reference_ms)
    );
    EndgameSession::new(
        EndgameConfig {
            solver_fraction: protocol.solver_fraction,
            ..Default::default()
        },
        0.0,
        protocol.milliseconds,
    )
    .unwrap();
    let sha = |b: &[u8]| format!("{:x}", Sha256::digest(b));
    assert_eq!(
        sha(&std::fs::read(&protocol.corpus).unwrap()),
        protocol.corpus_sha256
    );
    assert_eq!(
        sha(&std::fs::read(&protocol.prior).unwrap()),
        protocol.prior_sha256
    );
    let games: Vec<Game> = BufReader::new(std::fs::File::open(&protocol.corpus).unwrap())
        .lines()
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    let mut jobs = Vec::new();
    for (index, game) in games.iter().enumerate() {
        assert_eq!(game.schema, "nc2000-selfplay-endgames-v1");
        for &decision in &game.samples {
            assert!(decision < game.decisions.len());
            jobs.push((index, decision));
        }
    }
    let mut rng = SplitMix64::new(protocol.seed);
    for i in (1..jobs.len()).rev() {
        let j = rng.below(i + 1);
        jobs.swap(i, j);
    }
    let directory = PathBuf::from(&args[2]);
    std::fs::create_dir(&directory).unwrap();
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
    let metadata = json!({"protocol":serde_json::from_slice::<Value>(&bytes).unwrap(),"protocol_sha256":sha(&bytes),"revision":git(&["rev-parse","HEAD"]),"dirty":git(&["status","--porcelain"]),"binary_sha256":sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap()),"source_games":games.len(),"source_games_without_samples":games.iter().filter(|g|g.samples.is_empty()).count(),"positions":jobs.len(),"expected_continuations":jobs.len()*4,"interpretation":protocol.interpretation});
    std::fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec_pretty(&metadata).unwrap(),
    )
    .unwrap();
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new(&protocol.prior));
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let job = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(index, decision)) = jobs.get(job) else {
                    break;
                };
                let game = &games[index];
                let seed = protocol.seed
                    ^ (game.game as u64).wrapping_mul(0x9e3779b97f4a7c15)
                    ^ decision as u64;
                let (battle, _) = initial(&dex, &pool, game, decision, false, 0, &protocol, seed);
                let result = reference(&dex, &battle, protocol.reference_ms);
                std::fs::write(
                    directory.join(format!(
                        "game-{:03}-d{decision:04}-reference.json",
                        game.game
                    )),
                    serde_json::to_vec_pretty(&result).unwrap(),
                )
                .unwrap();
                for public in [job % 2 == 0, job % 2 != 0] {
                    for leg in [job % 2, 1 - job % 2] {
                        play(
                            &dex, &pool, game, decision, public, leg, &protocol, seed, &directory,
                        );
                    }
                }
            });
        }
    });
}
