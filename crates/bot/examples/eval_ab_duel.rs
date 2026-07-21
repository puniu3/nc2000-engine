//! M17 gate: seed-paired direct duel between two eval-weight configurations
//! on the same skuct agent (the M6 lesson: compare variants head-to-head,
//! never through a third opponent). A = shipped weights, B = the
//! calibration-measured candidate (slp time-scaling + substitute bonus +
//! status penalties ×0.7 — eval_calibration --ab winner).
//!
//! Usage: eval_ab_duel [--games 200] [--iters 300] [--seed 1]

use nc2000_bot::duel::{run_duel, DuelSpec};
use nc2000_bot::eval::EvalWeights;
use nc2000_bot::mcts::Playout;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule};

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn candidate() -> EvalWeights {
    let mut w = EvalWeights::default();
    w.slp_time_scale = true;
    w.substitute = 0.5;
    w.slp *= 0.7;
    w.frz *= 0.7;
    w.tox *= 0.7;
    w
}

fn cfg_with(weights: EvalWeights, iters: u32) -> RmConfig {
    RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c: 1.0,
        hp_buckets: 16,
        playout: Playout::Heavy { eps: 0.2, turns: 8, weights },
        ..Default::default()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games = arg(&args, "--games", 200);
    let iters = arg(&args, "--iters", 300) as u32;
    let seed = arg(&args, "--seed", 1) as u64;

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<_> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    let a_cfg = cfg_with(EvalWeights::default(), iters);
    let b_cfg = cfg_with(candidate(), iters);
    let stats = run_duel(
        &dex,
        &teams,
        &|s| Box::new(RmAgent::new(a_cfg.clone(), s)),
        &|s| Box::new(RmAgent::new(b_cfg.clone(), s)),
        DuelSpec::new(games, seed),
    );
    println!(
        "A(shipped) vs B(candidate): {}W {}L {}T  A-score {:.3} +/- {:.3}  avg turns {:.1}  think A {:.0} B {:.0} ms",
        stats.wins, stats.losses, stats.ties, stats.score, stats.ci95, stats.avg_turns,
        stats.a_ms_per_move, stats.b_ms_per_move
    );
}
