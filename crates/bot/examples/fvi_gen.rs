//! S1 fitted value iteration — self-play data generator.
//!
//! Both sides play the equilibrium of the depth-limited sampled Shapley
//! backup (`value::deep_backup`, `--ks` samples per depth) with ε-uniform
//! exploration; every decision state
//! is written with that backup's value as its TD target, and the game's
//! outcome is attached once it ends (the trainer forms λ-returns from both).
//! Leaves are `eval01` without `--model` (iteration 0) and the network with it.
//!
//!   fvi_gen --describe OUT.json
//!   fvi_gen --out DIR [--games 2000] [--seed 1] [--threads N] [--ks 16 | 8x4]
//!           [--eps 0.1] [--model MODEL.json] [--max-turns 300]
//!           [--pools data/team-pool-v2/team-pool.json,data/belief-pool-v3/belief-pool.json]
//!   fvi_gen --parity MODEL.json
//!
//! Output: DIR/records.f32 — little-endian f32 rows of
//! `value::RECORD + EXTRA` columns (record, then y, outcome, game, step),
//! appended one finished game at a time; DIR/meta.json — config and counts.

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::value::{
    deep_backup, outcome_value, sample_mixed, Leaf, ValueCodec, ValueNet, RECORD,
};
use nc2000_bot::eval::EvalWeights;
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;

const EXTRA: usize = 4;

fn arg_s(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let codec = ValueCodec::new(&dex);

    if let Some(i) = args.iter().position(|a| a == "--describe") {
        let path = &args[i + 1];
        std::fs::write(path, serde_json::to_string(&codec.describe()).unwrap()).unwrap();
        eprintln!("wrote {path} (record {RECORD})");
        return;
    }
    if let Some(i) = args.iter().position(|a| a == "--parity") {
        let net = ValueNet::load(std::path::Path::new(&args[i + 1])).unwrap();
        assert!(!net.parity.is_empty(), "model carries no parity rows");
        let mut worst = 0f32;
        for (rec, logit) in &net.parity {
            assert_eq!(rec.len(), RECORD, "parity row length");
            worst = worst.max((net.logit(&codec, rec) - logit).abs());
        }
        println!("parity rows {} worst |logit diff| {worst:.2e}", net.parity.len());
        assert!(worst < 1e-3, "Rust and trainer forward passes disagree");
        return;
    }

    let out_dir = root.join(arg_s(&args, "--out", "tmp/fvi/iter0"));
    let games: usize = arg_s(&args, "--games", "2000").parse().unwrap();
    let seed: u64 = arg_s(&args, "--seed", "1").parse().unwrap();
    let default_threads =
        std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(4);
    let threads: usize = arg_s(&args, "--threads", &default_threads.to_string()).parse().unwrap();
    let k: usize = arg_s(&args, "--k", "16").parse().unwrap();
    let ks: Vec<usize> = arg_s(&args, "--ks", &k.to_string())
        .split('x')
        .map(|x| x.parse().unwrap())
        .collect();
    let eps: f64 = arg_s(&args, "--eps", "0.1").parse().unwrap();
    let max_turns: u16 = arg_s(&args, "--max-turns", "300").parse().unwrap();
    let model_path = arg_s(&args, "--model", "");
    let pools = arg_s(
        &args,
        "--pools",
        "data/team-pool-v2/team-pool.json,data/belief-pool-v3/belief-pool.json",
    );
    let teams: Vec<Vec<PokemonSet>> = pools
        .split(',')
        .flat_map(|p| load_meta_pool(&root.join(p)).teams.into_iter().map(|t| t.sets))
        .collect();
    let codec = std::sync::Arc::new(codec);
    let net = (!model_path.is_empty())
        .then(|| std::sync::Arc::new(ValueNet::load(&root.join(&model_path)).expect("load model")));
    let leaf = match &net {
        Some(n) => Leaf::Net(n.clone(), codec.clone()),
        None => Leaf::Eval(EvalWeights::default()),
    };
    std::fs::create_dir_all(&out_dir).unwrap();
    let writer = Mutex::new(std::io::BufWriter::new(
        std::fs::File::create(out_dir.join("records.f32")).unwrap(),
    ));
    eprintln!(
        "fvi_gen: {games} games, {} teams, {threads} threads, K {k}, eps {eps}, leaf {}",
        teams.len(),
        if net.is_some() { model_path.as_str() } else { "eval01" }
    );

    let cursor = AtomicUsize::new(0);
    let records = AtomicUsize::new(0);
    let capped = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let mut scratch = Vec::with_capacity(RECORD);
                loop {
                    let g = cursor.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        return;
                    }
                    let mut rng = SplitMix64::new(seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let (ta, tb) = (rng.below(teams.len()), rng.below(teams.len()));
                    let mut b =
                        Battle::from_fixture(&dex, &rng.battle_seed(), &teams[ta], &teams[tb])
                            .expect("battle");
                    b.set_log_enabled(false);
                    let mut rows: Vec<f32> = Vec::new();
                    let mut step = 0usize;
                    let mut result = None;
                    loop {
                        if let Some(o) = b.outcome() {
                            result = Some(outcome_value(o));
                            break;
                        }
                        if b.turn > max_turns {
                            break;
                        }
                        if b.turn == 0 {
                            let mut picks = [None, None];
                            for s in 0..2 {
                                let cs = b.legal_choices(&dex, s);
                                if !cs.is_empty() {
                                    picks[s] = Some(cs[rng.below(cs.len())]);
                                }
                            }
                            b.apply_choices(&dex, picks).expect("preview");
                            continue;
                        }
                        let mut cache = std::collections::HashMap::new();
                        let Some(bk) =
                            deep_backup(&b, &dex, &leaf, &ks, &mut rng, &mut scratch, &mut cache)
                        else {
                            break;
                        };
                        codec.encode(&b, &dex, &mut rows);
                        rows.extend_from_slice(&[bk.value as f32, f32::NAN, g as f32, step as f32]);
                        step += 1;
                        let i = sample_mixed(&bk.mix[0], eps, &mut rng);
                        let j = sample_mixed(&bk.mix[1], eps, &mut rng);
                        b.apply_choices(&dex, [bk.actions[0][i], bk.actions[1][j]])
                            .expect("apply");
                    }
                    let outcome = result.unwrap_or_else(|| {
                        capped.fetch_add(1, Ordering::Relaxed);
                        0.5
                    });
                    for row in rows.chunks_exact_mut(RECORD + EXTRA) {
                        row[RECORD + 1] = outcome as f32;
                    }
                    {
                        let mut w = writer.lock().unwrap();
                        for x in &rows {
                            w.write_all(&x.to_le_bytes()).unwrap();
                        }
                    }
                    let n = records.fetch_add(step, Ordering::Relaxed) + step;
                    if (g + 1) % 100 == 0 {
                        eprintln!(
                            "  game {}/{games}: {n} records, {:.0}s",
                            g + 1,
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });
    writer.into_inner().unwrap().flush().unwrap();
    let meta = serde_json::json!({
        "schema": nc2000_bot::value::VALUE_SCHEMA,
        "record": RECORD,
        "extra": ["y", "outcome", "game", "step"],
        "games": games,
        "records": records.load(Ordering::Relaxed),
        "capped": capped.load(Ordering::Relaxed),
        "seed": seed,
        "ks": ks,
        "eps": eps,
        "max_turns": max_turns,
        "model": model_path,
        "pools": pools,
        "seconds": t0.elapsed().as_secs_f64(),
    });
    std::fs::write(out_dir.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap())
        .unwrap();
    println!("{}", serde_json::to_string(&meta).unwrap());
}
