//! S1 gate instrument: how close a value function is to V*, without playouts.
//!
//! 1. Corpus residual — on held-out human-corpus positions (a different
//!    distribution from the self-play training states), the one-step
//!    inconsistency `val(M[V]) − V(s)` with K sampled successors per cell.
//! 2. Certified anchors — endgame values bracketed by the M17e solver
//!    (`endgame_exactness_corpus` artifact rows). A row is used only when the
//!    reconstructed position's `state_key128` equals the certified one; the
//!    brackets are exact for the engine build that produced the artifact.
//!
//! Every metric is reported for `eval01` and, with `--model`, for the network
//! on the same positions.
//!
//!   fvi_eval [--model M.json] [--battles 0-569] [--per-battle 2] [--k 32]
//!            [--anchors tmp/m17e-merged-v4.json] [--eps 0.02] [--threads N]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::corpus::{
    corpus_files, load_battle, load_sources, reconstruct, reconstruct_context_with_pool,
};
use nc2000_bot::eval::EvalWeights;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::value::{sampled_backup, Leaf, ValueCodec, ValueNet};
use nc2000_bot::SplitMix64;
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side].party.iter().filter(|&&s| !b.sides[side].roster[s as usize].fainted).count()
}

struct Res {
    alive: usize,
    v: Vec<f64>,
    backup: Vec<f64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = arg_s(&args, "--model", "");
    let range = arg_s(&args, "--battles", "0-569");
    let (lo, hi) = {
        let mut it = range.split('-');
        (it.next().unwrap().parse::<usize>().unwrap(), it.next().unwrap().parse::<usize>().unwrap())
    };
    let per_battle: usize = arg_s(&args, "--per-battle", "2").parse().unwrap();
    let k: usize = arg_s(&args, "--k", "32").parse().unwrap();
    let eps: f64 = arg_s(&args, "--eps", "0.02").parse().unwrap();
    let anchors_path = arg_s(&args, "--anchors", "tmp/m17e-merged-v4.json");
    let default_threads =
        std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(4);
    let threads: usize = arg_s(&args, "--threads", &default_threads.to_string()).parse().unwrap();
    let seed = 20261002u64;

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let codec = std::sync::Arc::new(ValueCodec::new(&dex));
    let net = (!model_path.is_empty())
        .then(|| std::sync::Arc::new(ValueNet::load(&root.join(&model_path)).unwrap()));
    let mut leaves: Vec<(&str, Leaf)> = vec![("eval01", Leaf::Eval(EvalWeights::default()))];
    if let Some(n) = &net {
        leaves.push(("net", Leaf::Net(n.clone(), codec.clone())));
    }
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let pool = load_meta_pool(&pool_path);
    let files = corpus_files(&root.join("tmp/corpus-spectator"));

    // ---- 1. corpus residual
    let chosen: Vec<(usize, &std::path::PathBuf)> =
        files.iter().enumerate().filter(|(i, _)| *i >= lo && *i <= hi).collect();
    let results: Mutex<Vec<Res>> = Mutex::new(Vec::new());
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| {
                let mut scratch = Vec::new();
                loop {
                    let j = cursor.fetch_add(1, Ordering::Relaxed);
                    if j >= chosen.len() {
                        return;
                    }
                    let (bi, path) = chosen[j];
                    let cb = load_battle(path);
                    let mut rng = SplitMix64::new(seed ^ (bi as u64).wrapping_mul(0xD1B5_4A32));
                    let mut reservoir: Vec<Battle> = Vec::new();
                    let mut seen = 0usize;
                    for (di, d) in cb.decisions.iter().enumerate() {
                        let dseed = seed ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A) ^ (di as u64);
                        let Some(rec) = reconstruct_context_with_pool(
                            &dex, &src, pool.clone(), &cb.lines, &cb.evidence, d, dseed,
                        ) else {
                            continue;
                        };
                        let Some(b) = rec.agent.battle().cloned() else { continue };
                        if b.active_id(0).is_none() || b.active_id(1).is_none() {
                            continue;
                        }
                        let needs = b.needs_choice();
                        if !needs[0] && !needs[1] {
                            continue;
                        }
                        seen += 1;
                        if reservoir.len() < per_battle {
                            reservoir.push(b);
                        } else if rng.next_f64() < per_battle as f64 / seen as f64 {
                            let slot = rng.below(per_battle);
                            reservoir[slot] = b;
                        }
                    }
                    for (pi, b) in reservoir.into_iter().enumerate() {
                        let mut res = Res { alive: alive(&b, 0) + alive(&b, 1), v: vec![], backup: vec![] };
                        for (_, leaf) in &leaves {
                            let mut brng = SplitMix64::new(seed ^ ((bi * 16 + pi) as u64));
                            let Some(bk) = sampled_backup(&b, &dex, leaf, k, &mut brng, &mut scratch)
                            else {
                                continue;
                            };
                            res.v.push(leaf.value(&b, &dex, &mut scratch));
                            res.backup.push(bk.value);
                        }
                        if res.v.len() == leaves.len() {
                            results.lock().unwrap().push(res);
                        }
                    }
                }
            });
        }
    });
    let results = results.into_inner().unwrap();
    println!("corpus residual: {} positions, K {k}", results.len());
    for (name, pred) in [
        ("alive<=2", (|t| t <= 2) as fn(usize) -> bool),
        ("alive 3-4", |t| (3..=4).contains(&t)),
        ("alive 5-6", |t| t >= 5),
        ("all", |_| true),
    ] {
        let rs: Vec<&Res> = results.iter().filter(|r| pred(r.alive)).collect();
        if rs.is_empty() {
            continue;
        }
        let mut line = format!("  [{name:9}] n {:4}", rs.len());
        for (li, (lname, _)) in leaves.iter().enumerate() {
            let n = rs.len() as f64;
            let rms = (rs.iter().map(|r| (r.backup[li] - r.v[li]).powi(2)).sum::<f64>() / n).sqrt();
            let bias = rs.iter().map(|r| r.backup[li] - r.v[li]).sum::<f64>() / n;
            line += &format!(" | {lname} rms {rms:.4} bias {bias:+.4}");
        }
        println!("{line}");
    }

    // ---- 2. certified anchors
    let Ok(text) = std::fs::read_to_string(root.join(&anchors_path)) else {
        println!("anchors: {anchors_path} missing, skipped");
        return;
    };
    let doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    let rseed = doc["run"]["selection"]["reconstruction_seed"].as_u64().unwrap_or(1);
    let rows = doc["rows"].as_array().cloned().unwrap_or_default();
    let mut used = 0usize;
    let mut stale = 0usize;
    let mut stats: Vec<(usize, f64, f64, usize)> = vec![(0, 0.0, 0.0, 0); leaves.len()];
    let mut scratch = Vec::new();
    for row in &rows {
        let bi = row["battle"].as_u64().unwrap() as usize;
        let di = row["decision"].as_u64().unwrap() as usize;
        let cb = load_battle(&files[bi]);
        let Some(d) = cb.decisions.get(di) else {
            stale += 1;
            continue;
        };
        let Some(b) = reconstruct(&dex, &src, &pool_path, &cb.lines, &cb.evidence, d, rseed) else {
            stale += 1;
            continue;
        };
        let key = u128::from_str_radix(row["state_key128"].as_str().unwrap(), 16).unwrap();
        if b.state_key128() != key {
            stale += 1;
            continue;
        }
        used += 1;
        let exact = row["exact"].as_f64().unwrap();
        let width = row["width"].as_f64().unwrap();
        let (blo, bhi) = (exact - width / 2.0, exact + width / 2.0);
        for (li, (_, leaf)) in leaves.iter().enumerate() {
            let v = leaf.value(&b, &dex, &mut scratch);
            let viol = (v - bhi).max(blo - v).max(0.0);
            let st = &mut stats[li];
            if viol > eps {
                st.0 += 1;
                st.1 += viol;
            }
            if width <= 0.05 {
                st.2 += (v - exact).abs();
                st.3 += 1;
            }
        }
    }
    println!("anchors: {used} used, {stale} stale/unreconstructable of {} ({anchors_path})", rows.len());
    for (li, (lname, _)) in leaves.iter().enumerate() {
        let (nv, margin, abs, nt) = stats[li];
        println!(
            "  {lname:6} violations {nv} (eps {eps}) total margin {margin:.3} | tight MAE {:.4} (n {nt})",
            if nt > 0 { abs / nt as f64 } else { f64::NAN }
        );
    }
}
