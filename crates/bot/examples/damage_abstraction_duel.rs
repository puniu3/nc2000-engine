//! Seed-paired duel gate for the experimental damage abstraction.
//!
//! Both agents use identical SM-MCTS outside eligible small endgames. Inside
//! them they use `DamageSearchAgent`; incomplete fixed-work searches fall
//! back to the same SM-MCTS policy. Reported think time therefore includes
//! both the abstraction's speed and the coverage it buys.

use std::path::{Path, PathBuf};

use conformance::fixture::{corpus_files, Fixture};
use nc2000_bot::damage_search::{
    DamageSearchAgent, DamageSearchAgentConfig, DamageSearchConfig, ProbeRefineAgent,
    ProbeRefineAgentConfig, ProbeRefineConfig,
};
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

fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|item| item == key)
}

fn mode(value: &str) -> DamageRollMode {
    match value {
        "exact" => DamageRollMode::Exact,
        "mean" => DamageRollMode::Mean,
        "threshold1" | "t1" => DamageRollMode::Threshold1,
        "threshold2" | "t2" => DamageRollMode::Threshold2,
        "lean" => DamageRollMode::ThresholdLean,
        "lean-no-counter" | "lnc" => DamageRollMode::ThresholdLeanNoCounter,
        "lean-no-drain" | "lnd" => DamageRollMode::ThresholdLeanNoDrainRecoil,
        "lean-no-multihit" | "lnm" => DamageRollMode::ThresholdLeanNoMultiHit,
        "lean-no-substitute" | "lns" => DamageRollMode::ThresholdLeanNoSubstitute,
        "lean-minimal" | "lmin" => DamageRollMode::ThresholdLeanMinimal,
        "lean-next" | "ln" => DamageRollMode::ThresholdLeanNext,
        "lean-residual" | "lr" => DamageRollMode::ThresholdLeanResidual,
        "lean-clock" | "lc" => DamageRollMode::ThresholdLeanClock,
        "heal-split" | "hs" => DamageRollMode::ThresholdHealSplit,
        "heal" => DamageRollMode::ThresholdHeal,
        _ => panic!("bad damage mode {value}"),
    }
}

fn repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("NC2000_REPO_ROOT") {
        return PathBuf::from(root);
    }
    let current = std::env::current_dir().unwrap();
    if current.join("data/gen2stadium2.json").is_file() {
        return current;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
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

fn probe_config(
    direct: &DamageSearchAgentConfig,
    probe_work_budget: usize,
    cell_threshold: f64,
) -> ProbeRefineAgentConfig {
    let mut approximate = direct.search.clone();
    approximate.damage_mode = DamageRollMode::ThresholdLeanMinimal;
    ProbeRefineAgentConfig {
        refine: ProbeRefineConfig {
            exact_work_budget: direct.search.work_budget,
            approximate,
            probe_work_budget,
            response_margin: 0.0,
            cell_threshold,
        },
        alive_max: direct.alive_max,
        hp_cap: direct.hp_cap,
        fallback: direct.fallback.clone(),
    }
}

fn load_fixture_pool(repo: &Path) -> Vec<Vec<PokemonSet>> {
    let root = repo.join("fixtures/corpus-v1");
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
    let a_probe = flag(&args, "--a-probe");
    let b_probe = flag(&args, "--b-probe");
    let probe_work: usize = arg(&args, "--probe-work", "200000").parse().unwrap();
    let probe_threshold: f64 = arg(&args, "--probe-threshold", "0.01").parse().unwrap();

    let root = repo_root();
    let dex_json = std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap();
    let dex = nc2000_engine::dex::Dex::from_json(&dex_json).unwrap();
    let teams = if let Some(range) = pool_spec.strip_prefix("meta") {
        let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
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
        load_fixture_pool(&root)
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
    let probe_config_a = probe_config(&config_a, probe_work, probe_threshold);
    let probe_config_b = probe_config(&config_b, probe_work, probe_threshold);
    eprintln!(
        "{} teams; A {}{:?} h{} w{} vs B {}{:?} h{} w{}; probe w{} t{}; fallback {}",
        teams.len(),
        if a_probe { "probe:" } else { "" },
        a_mode,
        a_horizon,
        a_work,
        if b_probe { "probe:" } else { "" },
        b_mode,
        b_horizon,
        b_work,
        probe_work,
        probe_threshold,
        fallback_iters,
    );
    let build_a = |agent_seed| -> Box<dyn nc2000_bot::Agent> {
        if a_probe {
            Box::new(ProbeRefineAgent::new(probe_config_a.clone(), agent_seed))
        } else {
            Box::new(DamageSearchAgent::new(config_a.clone(), agent_seed))
        }
    };
    let build_b = |agent_seed| -> Box<dyn nc2000_bot::Agent> {
        if b_probe {
            Box::new(ProbeRefineAgent::new(probe_config_b.clone(), agent_seed))
        } else {
            Box::new(DamageSearchAgent::new(config_b.clone(), agent_seed))
        }
    };
    let stats = run_duel(
        &dex,
        &teams,
        &build_a,
        &build_b,
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
