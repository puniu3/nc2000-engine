use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use nc2000_bot::endgame::{EndgameConfig, EndgameSession, EndgameStats};
use nc2000_bot::player::{action_input, PlayerChannel};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::{Belief, BlindSearch, Observer, ProtocolAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Arm {
    information: String,
    solver_fraction: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Protocol {
    schema: String,
    purpose: String,
    catalog: String,
    catalog_sha256: String,
    prior: String,
    prior_sha256: String,
    pairs: usize,
    candidate: Arm,
    baseline: Arm,
    milliseconds: f64,
    calibration_iterations: Option<u32>,
    threads: usize,
    seed: u64,
    interpretation: String,
}

#[derive(Deserialize)]
struct Catalog {
    teams: Vec<Team>,
}

#[derive(Deserialize)]
struct Team {
    id: String,
    #[serde(rename = "drawWeight")]
    weight: f64,
    sets: Vec<PokemonSet>,
}

struct Pair {
    teams: [usize; 2],
    battle_seed: String,
    bot_seed: u64,
}

enum Seat {
    Public(Box<ProtocolAgent>, PlayerChannel, f64),
    Raw(Observer, Belief, SplitMix64),
    Empty(SplitMix64),
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
        battle: &Battle,
        side: usize,
        own: &[PokemonSet],
        arm: &Arm,
        seed: u64,
    ) -> Self {
        match arm.information.as_str() {
            "public" => {
                let mut agent = ProtocolAgent::new(dex, side, pool.clone(), config(), seed);
                agent.set_own_team(own.to_vec());
                Self::Public(
                    Box::new(agent),
                    PlayerChannel::new(side),
                    arm.solver_fraction,
                )
            }
            "raw" => {
                let observer = Observer::new(battle, side);
                let belief = Belief::new(dex, pool, &observer);
                Self::Raw(observer, belief, SplitMix64::new(seed))
            }
            "empty" => Self::Empty(SplitMix64::new(seed)),
            _ => panic!("unknown information policy"),
        }
    }

    fn choose(
        &mut self,
        dex: &Dex,
        battle: &mut Battle,
        side: usize,
        protocol: &Protocol,
    ) -> (Option<SearchChoice>, serde_json::Value) {
        let started = Instant::now();
        let mut clock = || started.elapsed().as_secs_f64() * 1000.0;
        let legal = battle.legal_choices(dex, side);
        let (choice, iterations, nodes, stats, drift, projections) = match self {
            Self::Public(agent, channel, fraction) => {
                let frame = channel.frame(battle, dex).unwrap();
                for line in frame.lines {
                    agent.push_line(dex, &line);
                }
                if !agent.on_request(dex, &frame.request.to_string()).unwrap() {
                    assert!(legal.is_empty());
                    return (None, json!({"wait":true}));
                }
                let mut session = EndgameSession::new(
                    EndgameConfig {
                        solver_fraction: *fraction,
                        ..Default::default()
                    },
                    0.0,
                    protocol.milliseconds,
                )
                .unwrap();
                if let Some(iterations) = protocol.calibration_iterations {
                    assert_eq!(*fraction, 0.0);
                    agent.step(dex, iterations).unwrap();
                } else {
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
                }
                let input = agent.best(dex).unwrap();
                let choice = *legal
                    .iter()
                    .find(|&&a| action_input(dex, a) == input)
                    .unwrap_or_else(|| panic!("illegal public choice: {input}"));
                (
                    choice,
                    agent.iterations(),
                    agent.search().unwrap().node_count(),
                    session.stats,
                    agent.legality_drift,
                    agent.projections,
                )
            }
            Self::Raw(observer, belief, rng) => {
                if legal.is_empty() {
                    return (None, json!({"wait":true}));
                }
                observer.observe(battle, dex);
                belief.sync_checked(dex, observer).unwrap();
                let mut search = BlindSearch::new(battle, dex, config(), side, rng.next());
                if let Some(iterations) = protocol.calibration_iterations {
                    search.step(dex, belief, observer, iterations);
                } else {
                    while clock() < protocol.milliseconds {
                        search.step(dex, belief, observer, 1);
                    }
                }
                (
                    search.best().unwrap_or(legal[0]),
                    search.iterations(),
                    search.node_count(),
                    EndgameStats::default(),
                    0,
                    0,
                )
            }
            Self::Empty(rng) => {
                if legal.is_empty() {
                    return (None, json!({"wait":true}));
                }
                (
                    legal[rng.below(legal.len())],
                    0,
                    0,
                    EndgameStats::default(),
                    0,
                    0,
                )
            }
        };
        assert!(legal.contains(&choice));
        (
            Some(choice),
            json!({"choice": action_input(dex,choice), "elapsed_ms":clock(), "iterations":iterations, "search_nodes":nodes, "stats":stats, "legality_drift":drift, "projections":projections, "legal":true, "forced":legal.len()==1}),
        )
    }
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn play(
    dex: &Dex,
    pool: &MetaPool,
    catalog: &Catalog,
    protocol: &Protocol,
    pair: &Pair,
    index: usize,
    leg: usize,
    directory: &Path,
) {
    let output = directory.join(format!("pair-{index:04}-{leg}.jsonl"));
    if output.exists() {
        let text = std::fs::read_to_string(&output).unwrap();
        if text
            .lines()
            .last()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
            .is_some_and(|v| v["type"] == "result")
        {
            return;
        }
        panic!(
            "incomplete game at {}; preserve it and run a separately recorded replacement",
            output.display()
        );
    }
    let teams = [0, 1].map(|side| &catalog.teams[pair.teams[side]]);
    let mut battle =
        Battle::from_fixture(dex, &pair.battle_seed, &teams[0].sets, &teams[1].sets).unwrap();
    battle.log_enabled = true;
    let mut seats = [0, 1].map(|side| {
        Seat::new(
            dex,
            pool,
            &battle,
            side,
            &teams[side].sets,
            if side == leg {
                &protocol.candidate
            } else {
                &protocol.baseline
            },
            pair.bot_seed,
        )
    });
    let mut out = std::io::BufWriter::new(std::fs::File::create_new(output).unwrap());
    writeln!(out,"{}",json!({"type":"game","pair":index,"leg":leg,"candidate_side":leg,"teams":[teams[0].id,teams[1].id],"battle_seed":pair.battle_seed,"bot_seed":pair.bot_seed})).unwrap();
    out.flush().unwrap();
    let mut decisions = 0;
    while battle.outcome().is_none() {
        let key = battle.state_key128().to_string();
        let turn = battle.turn;
        let alive = [0, 1].map(|s| {
            battle.sides[s]
                .party
                .iter()
                .filter(|&&i| !battle.sides[s].roster[i as usize].fainted)
                .count()
        });
        let mut choices = [None; 2];
        let mut reports = [json!(null), json!(null)];
        for side in [decisions % 2, 1 - decisions % 2] {
            (choices[side], reports[side]) = seats[side].choose(dex, &mut battle, side, protocol);
        }
        battle.apply_choices(dex, choices).unwrap();
        writeln!(out,"{}",json!({"type":"decision","decision":decisions,"turn":turn,"state_key":key,"alive":alive,"sides":reports})).unwrap();
        out.flush().unwrap();
        decisions += 1;
        assert!(
            decisions <= 4000,
            "nonterminal game exceeded emergency guard"
        );
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
    writeln!(out,"{}",json!({"type":"result","pair":index,"leg":leg,"candidate_score":score,"outcome":format!("{:?}",battle.outcome()),"turns":battle.turn,"decisions":decisions,"engine_cap":battle.turn>=1000})).unwrap();
    out.flush().unwrap();
    eprintln!(
        "pair {index} leg {leg}: score={score}, {} turns",
        battle.turn
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: endgame_matches PROTOCOL.json OUTPUT-DIRECTORY"
    );
    let bytes = std::fs::read(&args[1]).unwrap();
    let protocol: Protocol = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(protocol.schema, "nc2000-endgame-matches-v1");
    assert!(protocol.pairs > 0 && protocol.threads > 0);
    for arm in [&protocol.candidate, &protocol.baseline] {
        assert!(["public", "raw", "empty"].contains(&arm.information.as_str()));
        assert!(arm.information == "public" || arm.solver_fraction == 0.0);
        EndgameSession::new(
            EndgameConfig {
                solver_fraction: arm.solver_fraction,
                ..Default::default()
            },
            0.0,
            protocol.milliseconds,
        )
        .unwrap();
    }
    let catalog_bytes = std::fs::read(&protocol.catalog).unwrap();
    let prior_bytes = std::fs::read(&protocol.prior).unwrap();
    assert_eq!(sha(&catalog_bytes), protocol.catalog_sha256);
    assert_eq!(sha(&prior_bytes), protocol.prior_sha256);
    let catalog: Catalog = serde_json::from_slice(&catalog_bytes).unwrap();
    assert!(catalog
        .teams
        .iter()
        .all(|t| t.weight.is_finite() && t.weight >= 0.0));
    let weight: f64 = catalog.teams.iter().map(|t| t.weight).sum();
    assert!(weight > 0.0);
    let mut rng = SplitMix64::new(protocol.seed);
    let pairs: Vec<_> = (0..protocol.pairs)
        .map(|_| {
            let teams = [0, 1].map(|_| {
                let mut draw = rng.next_f64() * weight;
                catalog
                    .teams
                    .iter()
                    .position(|t| {
                        draw -= t.weight;
                        draw < 0.0
                    })
                    .unwrap_or(catalog.teams.len() - 1)
            });
            Pair {
                teams,
                battle_seed: rng.battle_seed(),
                bot_seed: rng.next(),
            }
        })
        .collect();
    let directory = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&directory).unwrap();
    let manifest_path = directory.join("manifest.json");
    let binary_sha = sha(&std::fs::read(std::env::current_exe().unwrap()).unwrap());
    if manifest_path.exists() {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(manifest["protocol_sha256"], sha(&bytes));
        assert_eq!(manifest["binary_sha256"], binary_sha);
    } else {
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
        let manifest = json!({"protocol":protocol,"protocol_sha256":sha(&bytes),"binary_sha256":binary_sha,"revision":git(&["rev-parse","HEAD"]),"dirty":git(&["status","--porcelain"]),"schedule":pairs.iter().enumerate().map(|(i,p)|json!({"pair":i,"teams":p.teams,"battle_seed":p.battle_seed,"bot_seed":p.bot_seed})).collect::<Vec<_>>()});
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new(&protocol.prior));
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..protocol.threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(pair) = pairs.get(index) else {
                    break;
                };
                for leg in [index % 2, 1 - index % 2] {
                    play(
                        &dex, &pool, &catalog, &protocol, pair, index, leg, &directory,
                    );
                }
            });
        }
    });
}
