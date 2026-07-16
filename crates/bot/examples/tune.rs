//! SPSA self-play tuning of the M6 eval weights.
//!
//! Each iteration perturbs the weight vector ±c_k along a random ±1 direction,
//! duels mcts(θ+) against mcts(θ-) (seed-paired games over the fixture team
//! pool), and steps along the estimated gradient of the duel score. Standard
//! SPSA gains (α=0.602, γ=0.101). `scale` is not tuned — a uniform scaling of
//! all weights is the same knob.
//!
//!   cargo run --release -p nc2000-bot --example tune -- \
//!       --iters 100 --games 48 --mcts-iters 300 [--eps 0.2] [--turns 8] \
//!       [--seed 1] [--threads N] [--out weights.txt] [--resume weights.txt] \
//!       [--final-games 200]
//!
//! The weights file is rewritten every iteration ("name value" lines), so an
//! interrupted run loses at most one iteration; --resume continues from it.
//! The end of the run duels tuned-vs-initial weights as a progress readout.

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::{
    run_duel, Agent, DuelSpec, EvalWeights, MctsAgent, MctsConfig, Playout, SplitMix64,
};
use nc2000_engine::battle::PokemonSet;

const N: usize = EvalWeights::N;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn write_weights(path: &str, iter: usize, score: f64, v: &[f64; N], scale: f64) {
    let mut s = format!("# iter {iter} score {score:.3}\n");
    for (name, val) in EvalWeights::NAMES.iter().zip(v.iter()) {
        s.push_str(&format!("{name} {val:.6}\n"));
    }
    s.push_str(&format!("scale {scale:.6}\n"));
    std::fs::write(path, s).expect("write weights file");
}

fn read_weights(path: &str) -> ([f64; N], f64) {
    let text = std::fs::read_to_string(path).expect("read weights file");
    let mut v = EvalWeights::default().to_vec();
    let mut scale = EvalWeights::default().scale;
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(name), Some(val)) = (it.next(), it.next()) else { continue };
        if name == "#" {
            continue;
        }
        let val: f64 = val.parse().expect("bad weight value");
        if name == "scale" {
            scale = val;
        } else if let Some(i) = EvalWeights::NAMES.iter().position(|n| *n == name) {
            v[i] = val;
        } else {
            panic!("unknown weight name in {path}: {name}");
        }
    }
    (v, scale)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iters: usize = flag(&args, "--iters").map(|v| v.parse().unwrap()).unwrap_or(100);
    let games: usize = flag(&args, "--games").map(|v| v.parse().unwrap()).unwrap_or(48);
    let mcts_iters: u32 = flag(&args, "--mcts-iters").map(|v| v.parse().unwrap()).unwrap_or(300);
    let ucb_c: f64 = flag(&args, "--c").map(|v| v.parse().unwrap()).unwrap_or(1.0);
    let eps: f64 = flag(&args, "--eps").map(|v| v.parse().unwrap()).unwrap_or(0.2);
    let turns: u16 = flag(&args, "--turns").map(|v| v.parse().unwrap()).unwrap_or(8);
    let base_seed: u64 = flag(&args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(1);
    let max_turns: u16 = flag(&args, "--max-turns").map(|v| v.parse().unwrap()).unwrap_or(500);
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let out = flag(&args, "--out").unwrap_or_else(|| "tuned-weights.txt".into());
    let final_games: usize =
        flag(&args, "--final-games").map(|v| v.parse().unwrap()).unwrap_or(200);

    let dex = load_dex();
    let teams = load_team_pool();

    let (theta0, scale) = match flag(&args, "--resume") {
        Some(path) => read_weights(&path),
        None => (EvalWeights::default().to_vec(), EvalWeights::default().scale),
    };
    let initial = theta0;

    // per-param normalization unit: initial magnitude, floored
    let unit: [f64; N] = theta0.map(|w| w.abs().max(0.1));
    // normalized coordinates
    let mut phi: [f64; N] = std::array::from_fn(|i| theta0[i] / unit[i]);

    // SPSA gains
    let alpha = 0.602;
    let gamma = 0.101;
    let c0: f64 = flag(&args, "--c0").map(|v| v.parse().unwrap()).unwrap_or(0.15);
    let a0: f64 = flag(&args, "--a0").map(|v| v.parse().unwrap()).unwrap_or(0.15);
    let stability = (iters as f64 * 0.1).max(1.0);

    let make_agent = |weights: EvalWeights, seed: u64| -> Box<dyn Agent> {
        Box::new(MctsAgent::new(
            MctsConfig {
                iterations: mcts_iters,
                c: ucb_c,
                playout: Playout::Heavy { eps, turns, weights },
                ..Default::default()
            },
            seed,
        ))
    };

    eprintln!(
        "SPSA: {iters} iters x {games} games, agent mcts:{mcts_iters}:{ucb_c}:{eps}:{turns}, \
         {threads} threads, out {out}"
    );
    let t0 = std::time::Instant::now();
    let mut rng = SplitMix64::new(base_seed ^ 0x5B5A_D1DE);

    for k in 0..iters {
        let ck = c0 / ((k + 1) as f64).powf(gamma);
        let ak = a0 / ((k + 1) as f64 + stability).powf(alpha);

        let delta: [f64; N] = std::array::from_fn(|_| if rng.next() & 1 == 1 { 1.0 } else { -1.0 });
        let theta_plus: [f64; N] = std::array::from_fn(|i| (phi[i] + ck * delta[i]) * unit[i]);
        let theta_minus: [f64; N] = std::array::from_fn(|i| (phi[i] - ck * delta[i]) * unit[i]);
        let w_plus = EvalWeights::from_vec(&theta_plus, scale);
        let w_minus = EvalWeights::from_vec(&theta_minus, scale);

        let stats = run_duel(
            &dex,
            &teams,
            &|seed| make_agent(w_plus.clone(), seed),
            &|seed| make_agent(w_minus.clone(), seed),
            DuelSpec {
                games,
                base_seed: base_seed ^ (k as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
                threads,
                max_turns,
                progress: false,
                log_on: false,
            },
        );

        // y(θ+) - y(θ-) = 2*score - 1 in a head-to-head duel
        let g = (2.0 * stats.score - 1.0) / (2.0 * ck);
        for i in 0..N {
            phi[i] = (phi[i] + ak * g * delta[i]).clamp(-6.0, 6.0);
        }

        let theta: [f64; N] = std::array::from_fn(|i| phi[i] * unit[i]);
        write_weights(&out, k + 1, stats.score, &theta, scale);
        eprintln!(
            "iter {:>3}/{iters}  score(θ+) {:.3}  ck {:.3} ak {:.4}  {:.0}s elapsed",
            k + 1,
            stats.score,
            ck,
            ak,
            t0.elapsed().as_secs_f64()
        );
    }

    let theta: [f64; N] = std::array::from_fn(|i| phi[i] * unit[i]);
    println!("final weights (also in {out}):");
    for (name, val) in EvalWeights::NAMES.iter().zip(theta.iter()) {
        println!("  {name}: {val:.4}");
    }
    println!("  scale: {scale:.4}");

    if final_games > 0 {
        eprintln!("holdout duel: tuned vs initial ({final_games} games)...");
        let w_tuned = EvalWeights::from_vec(&theta, scale);
        let w_init = EvalWeights::from_vec(&initial, scale);
        let stats = run_duel(
            &dex,
            &teams,
            &|seed| make_agent(w_tuned.clone(), seed),
            &|seed| make_agent(w_init.clone(), seed),
            DuelSpec {
                games: final_games,
                base_seed: base_seed ^ 0xF1A7_C0DE,
                threads,
                max_turns,
                progress: true,
                log_on: false,
            },
        );
        println!(
            "tuned vs initial: {}W {}L {}T  score {:.3} +/- {:.3}",
            stats.wins, stats.losses, stats.ties, stats.score, stats.ci95
        );
    }
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
