//! M17 gate: seed-paired direct duel between two eval-weight configurations
//! on the same skuct agent (the M6 lesson: compare variants head-to-head,
//! never through a third opponent). A = the historical M6 leaf squash;
//! B = the M17c probability-backup candidate.
//!
//! Usage: eval_ab_duel [--games 200] [--iters 300] [--seed 1]
//!                     [--leaf-alpha 1.0]

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

fn weights(leaf_alpha: f64) -> EvalWeights {
    EvalWeights { leaf_alpha, ..EvalWeights::default() }
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
    let leaf_alpha: f64 = args
        .iter()
        .position(|a| a == "--leaf-alpha")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    // M17 tail: with --spikes W, A becomes the SHIPPED default (the M16-exit
    // bot, spikes off) and B the candidate weight. Without it the historical
    // M17c comparison stands: A = the M6 leaf squash, B = probability backup.
    let spikes: Option<f64> = args
        .iter()
        .position(|a| a == "--spikes")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let teams: Vec<_> = pool.teams.iter().map(|t| t.sets.clone()).collect();

    // M17 exchange term: A = shipped default, B = the candidate weight. The
    // corpus sweep put r at 0.585 -> 0.644 and Brier at 0.2116 -> 0.2047, but
    // the three criteria do NOT agree on one weight (r peaks at 0.75-1.0,
    // Brier/MSE at 0.5), and calibration is necessary-not-sufficient anyway --
    // M17c's heal-blind variant fit anchors twice as well and lost the duel at
    // 0.39. This is the sovereign reading.
    let exchange: Option<f64> = args
        .iter()
        .position(|a| a == "--exchange")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());
    if let Some(w) = exchange {
        let a_cfg = cfg_with(EvalWeights::default(), iters);
        let b_cfg =
            cfg_with(EvalWeights { exchange: w, ..EvalWeights::default() }, iters);
        let stats = run_duel(
            &dex,
            &teams,
            &|s| Box::new(RmAgent::new(a_cfg.clone(), s)),
            &|s| Box::new(RmAgent::new(b_cfg.clone(), s)),
            DuelSpec::new(games, seed),
        );
        println!(
            "A(shipped default) vs B(exchange {w}) @{iters}: {}W {}L {}T  A-score {:.3} +/- {:.3}  avg turns {:.1}  think A {:.0} B {:.0} ms",
            stats.wins, stats.losses, stats.ties, stats.score, stats.ci95, stats.avg_turns,
            stats.a_ms_per_move, stats.b_ms_per_move
        );
        return;
    }
    let (a_cfg, b_cfg, label) = match spikes {
        Some(w) => (
            cfg_with(EvalWeights::default(), iters),
            cfg_with(EvalWeights { spikes: w, ..EvalWeights::default() }, iters),
            format!("A(shipped default) vs B(spikes {w})"),
        ),
        None => (
            cfg_with(weights(0.5), iters),
            cfg_with(weights(leaf_alpha), iters),
            format!("A(leaf alpha 0.5) vs B(alpha {leaf_alpha})"),
        ),
    };
    let stats = run_duel(
        &dex,
        &teams,
        &|s| Box::new(RmAgent::new(a_cfg.clone(), s)),
        &|s| Box::new(RmAgent::new(b_cfg.clone(), s)),
        DuelSpec::new(games, seed),
    );
    println!(
        "{label}: {}W {}L {}T  A-score {:.3} +/- {:.3}  avg turns {:.1}  think A {:.0} B {:.0} ms",
        stats.wins, stats.losses, stats.ties, stats.score, stats.ci95, stats.avg_turns,
        stats.a_ms_per_move, stats.b_ms_per_move
    );
}
