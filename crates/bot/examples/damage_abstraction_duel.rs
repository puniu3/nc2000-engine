//! Seed-paired duel gate for the experimental damage abstraction.
//!
//! Both agents use identical SM-MCTS outside eligible small endgames. Inside
//! them they use `DamageSearchAgent`; incomplete fixed-work searches fall
//! back to the same SM-MCTS policy. Reported think time therefore includes
//! both the abstraction's speed and the coverage it buys.

use conformance::fixture::{corpus_files, repo_root, Fixture};
use nc2000_bot::damage_search::{DamageSearchAgent, DamageSearchAgentConfig, DamageSearchConfig};
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_bot::{run_duel, DuelSpec};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::prng::DamageRollMode;

fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|item| item == key)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn mode(value: &str) -> DamageRollMode {
    match value {
        "exact" => DamageRollMode::Exact,
        "mean" => DamageRollMode::Mean,
        "threshold1" | "t1" => DamageRollMode::Threshold1,
        "threshold2" | "t2" => DamageRollMode::Threshold2,
        _ => panic!("bad damage mode {value}"),
    }
}

fn config(
    mode: DamageRollMode,
    horizon: u16,
    work: usize,
    states: usize,
    leaf_cap: usize,
    alive_max: usize,
    hp_cap: u64,
    fallback_iters: u32,
) -> DamageSearchAgentConfig {
    DamageSearchAgentConfig {
        search: DamageSearchConfig {
            horizon,
            damage_mode: mode,
            state_budget: states,
            work_budget: work,
            leaf_cap,
            ..Default::default()
        },
        alive_max,
        hp_cap,
        fallback: RmConfig {
            iterations: fallback_iters,
            rule: SelRule::Ucb,
            c: 1.0,
            hp_buckets: 16,
            ..Default::default()
        },
    }
}

fn load_fixture_pool() -> Vec<Vec<PokemonSet>> {
    let root = repo_root().join("fixtures/corpus-v1");
    let mut teams = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            let fixture = Fixture::load(&path).unwrap();
            teams.push(fixture.p1team);
            teams.push(fixture.p2team);
        }
    }
    teams
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a_mode = mode(&arg(&args, "--a-mode", "threshold2"));
    let b_mode = mode(&arg(&args, "--b-mode", "exact"));
    let a_horizon: u16 = arg(&args, "--a-horizon", "0").parse().unwrap();
    let b_horizon: u16 = arg(&args, "--b-horizon", "0").parse().unwrap();
    let a_work: usize = arg(&args, "--a-work", "200000").parse().unwrap();
    let b_work: usize = arg(&args, "--b-work", "200000").parse().unwrap();
    let states: usize = arg(&args, "--states", "100000").parse().unwrap();
    let leaf_cap: usize = arg(&args, "--leaf-cap", "100000").parse().unwrap();
    let alive_max: usize = arg(&args, "--alive-max", "1").parse().unwrap();
    let hp_cap: u64 = arg(&args, "--hp-cap", "600").parse().unwrap();
    let fallback_iters: u32 = arg(&args, "--fallback-iters", "3000").parse().unwrap();
    let games: usize = arg(&args, "--games", "200").parse().unwrap();
    let seed: u64 = arg(&args, "--seed", "1").parse().unwrap();
    let threads: usize = arg(&args, "--threads", "12").parse().unwrap();
    let max_turns: u16 = arg(&args, "--max-turns", "500").parse().unwrap();
    let pool_spec = arg(&args, "--pool", "fixtures");

    let dex = conformance::load_dex();
    let teams = if let Some(range) = pool_spec.strip_prefix("meta") {
        let pool = load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"));
        let (lo, hi) = match range.strip_prefix(':') {
            None => (0, pool.teams.len() - 1),
            Some(value) => {
                let (lo, hi) = value.split_once('-').expect("--pool meta:LO-HI");
                (
                    lo.parse().unwrap(),
                    hi.parse::<usize>().unwrap().min(pool.teams.len() - 1),
                )
            }
        };
        pool.teams[lo..=hi]
            .iter()
            .map(|team| team.sets.clone())
            .collect()
    } else {
        load_fixture_pool()
    };

    let config_a = config(
        a_mode,
        a_horizon,
        a_work,
        states,
        leaf_cap,
        alive_max,
        hp_cap,
        fallback_iters,
    );
    let config_b = config(
        b_mode,
        b_horizon,
        b_work,
        states,
        leaf_cap,
        alive_max,
        hp_cap,
        fallback_iters,
    );
    eprintln!(
        "{} teams; A {:?} h{} w{} vs B {:?} h{} w{}; fallback {}",
        teams.len(),
        a_mode,
        a_horizon,
        a_work,
        b_mode,
        b_horizon,
        b_work,
        fallback_iters,
    );
    let stats = run_duel(
        &dex,
        &teams,
        &|agent_seed| Box::new(DamageSearchAgent::new(config_a.clone(), agent_seed)),
        &|agent_seed| Box::new(DamageSearchAgent::new(config_b.clone(), agent_seed)),
        DuelSpec {
            games,
            base_seed: seed,
            threads,
            max_turns,
            progress: true,
            log_on: false,
        },
    );

    println!(
        "A {:?}:h{}:w{} vs B {:?}:h{}:w{} — {} games seed {}",
        a_mode, a_horizon, a_work, b_mode, b_horizon, b_work, stats.games, seed,
    );
    println!(
        "A {}W {}L {}T score {:.4} +/- {:.4} (95% CI)",
        stats.wins, stats.losses, stats.ties, stats.score, stats.ci95,
    );
    println!(
        "avg turns {:.1}; wall {:.1}s; {:.3} games/s; think ms/move A {:.2} B {:.2}",
        stats.avg_turns,
        stats.secs,
        stats.games as f64 / stats.secs,
        stats.a_ms_per_move,
        stats.b_ms_per_move,
    );
}
