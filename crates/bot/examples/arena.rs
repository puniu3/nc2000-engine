//! Agent-vs-agent evaluation arena. Teams come from the golden fixture
//! corpus (120 validator-clean teams). Games are seed-paired: each pairing
//! is played twice with sides swapped, same battle seed. Fully deterministic
//! for a given --seed regardless of --threads.
//!
//!   cargo run --release -p nc2000-bot --example arena -- \
//!       mcts:300 maxdamage --games 100 [--seed 1] [--threads N] [--max-turns 500]
//!
//! Agent specs: random | maxdamage | mcts[:ITERS[:C]]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::{Agent, MaxDamageAgent, MctsAgent, MctsConfig, RandomAgent, SplitMix64};
use nc2000_bot::{play_game, GameResult};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::state::Battle;

#[derive(Clone, Debug)]
enum AgentSpec {
    Random,
    MaxDamage,
    Mcts { iterations: u32, c: f64 },
}

impl AgentSpec {
    fn parse(s: &str) -> Result<AgentSpec, String> {
        let parts: Vec<&str> = s.split(':').collect();
        match parts[0] {
            "random" => Ok(AgentSpec::Random),
            "maxdamage" => Ok(AgentSpec::MaxDamage),
            "mcts" => {
                let iterations = parts
                    .get(1)
                    .map(|v| v.parse().map_err(|_| format!("bad iters: {v}")))
                    .transpose()?
                    .unwrap_or(1000);
                let c = parts
                    .get(2)
                    .map(|v| v.parse().map_err(|_| format!("bad c: {v}")))
                    .transpose()?
                    .unwrap_or(1.0);
                Ok(AgentSpec::Mcts { iterations, c })
            }
            other => Err(format!("unknown agent: {other}")),
        }
    }

    fn build(&self, seed: u64) -> Box<dyn Agent> {
        match self {
            AgentSpec::Random => Box::new(RandomAgent::new(seed)),
            AgentSpec::MaxDamage => Box::new(MaxDamageAgent::new()),
            AgentSpec::Mcts { iterations, c } => Box::new(MctsAgent::new(
                MctsConfig { iterations: *iterations, c: *c, ..Default::default() },
                seed,
            )),
        }
    }

    fn label(&self) -> String {
        match self {
            AgentSpec::Random => "random".into(),
            AgentSpec::MaxDamage => "maxdamage".into(),
            AgentSpec::Mcts { iterations, c } => format!("mcts:{iterations}:{c}"),
        }
    }
}

/// One scheduled game: pool team indices, battle seed, whether agent A is p1.
struct GameSpec {
    team_p1: usize,
    team_p2: usize,
    battle_seed: String,
    a_is_p1: bool,
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: arena <agentA> <agentB> [--games N] [--seed S] [--threads T] [--max-turns M]");
        std::process::exit(2);
    }
    let spec_a = AgentSpec::parse(&args[0]).unwrap();
    let spec_b = AgentSpec::parse(&args[1]).unwrap();
    let games: usize = flag(&args, "--games").map(|v| v.parse().unwrap()).unwrap_or(100);
    let games = games + games % 2; // paired
    let base_seed: u64 = flag(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(1);
    let max_turns: u16 = flag(&args, "--max-turns").map(|v| v.parse().unwrap()).unwrap_or(500);
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    let dex = load_dex();
    let teams = load_team_pool();
    eprintln!("{} teams in pool", teams.len());

    // schedule: pair k plays the same matchup twice with sides swapped
    let mut sched_rng = SplitMix64::new(base_seed);
    let mut specs = Vec::with_capacity(games);
    for _ in 0..games / 2 {
        let t1 = sched_rng.below(teams.len());
        let t2 = sched_rng.below(teams.len());
        let seed = sched_rng.battle_seed();
        specs.push(GameSpec { team_p1: t1, team_p2: t2, battle_seed: seed.clone(), a_is_p1: true });
        specs.push(GameSpec { team_p1: t1, team_p2: t2, battle_seed: seed, a_is_p1: false });
    }

    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();

    // per-game record: (a_score, turns)
    let mut results: Vec<(f64, u16)> = Vec::with_capacity(games);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (dex, teams, specs) = (&dex, &teams, &specs);
            let (cursor, done) = (&cursor, &done);
            let (spec_a, spec_b) = (spec_a.clone(), spec_b.clone());
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, f64, u16)> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= specs.len() {
                        break;
                    }
                    let g = &specs[i];
                    // agent seeds derive from game index only -> thread-count invariant
                    let sa = base_seed ^ (i as u64).wrapping_mul(0xA24B_AED4_963E_E407);
                    let sb = base_seed ^ (i as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25);
                    let mut agent_a = spec_a.build(sa);
                    let mut agent_b = spec_b.build(sb);
                    let mut battle = Battle::from_fixture(
                        dex,
                        &g.battle_seed,
                        &teams[g.team_p1],
                        &teams[g.team_p2],
                    )
                    .unwrap();
                    battle.set_log_enabled(false);
                    let (p1, p2): (&mut dyn Agent, &mut dyn Agent) = if g.a_is_p1 {
                        (agent_a.as_mut(), agent_b.as_mut())
                    } else {
                        (agent_b.as_mut(), agent_a.as_mut())
                    };
                    let res = play_game(dex, &mut battle, &mut [p1, p2], max_turns).unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let a_score = if g.a_is_p1 { p1_score } else { 1.0 - p1_score };
                    out.push((i, a_score, battle.turn));
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 10 == 0 || d == specs.len() {
                        eprintln!("  {d}/{} games ({:.0}s)", specs.len(), t0.elapsed().as_secs_f64());
                    }
                }
                out
            }));
        }
        let mut all: Vec<(usize, f64, u16)> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all.into_iter().map(|(_, s, t)| (s, t)).collect();
    });

    let dt = t0.elapsed().as_secs_f64();
    let n = results.len() as f64;
    let wins = results.iter().filter(|r| r.0 == 1.0).count();
    let losses = results.iter().filter(|r| r.0 == 0.0).count();
    let ties = results.len() - wins - losses;
    let score: f64 = results.iter().map(|r| r.0).sum::<f64>() / n;
    let var: f64 = results.iter().map(|r| (r.0 - score).powi(2)).sum::<f64>() / (n - 1.0);
    let ci = 1.96 * (var / n).sqrt();
    let avg_turns: f64 = results.iter().map(|r| r.1 as f64).sum::<f64>() / n;

    println!(
        "A={} vs B={}   {} games, seed {base_seed}, {threads} threads",
        spec_a.label(),
        spec_b.label(),
        results.len()
    );
    println!("A: {wins}W {losses}L {ties}T   score {score:.3} +/- {ci:.3} (95% CI)");
    println!("avg turns {avg_turns:.1}   {dt:.1}s total   {:.2} games/s", n / dt);
}

fn load_team_pool() -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fx = Fixture::load(&path).unwrap();
            teams.push(fx.p1team);
            teams.push(fx.p2team);
        }
    }
    teams
}
