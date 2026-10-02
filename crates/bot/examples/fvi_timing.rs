//! Time breakdown of self-play decisions as `fvi_gen` makes them: an
//! instrumented copy of `value::deep_backup` (same operations, same rng
//! consumption, so its values must equal the library's) timed per category,
//! with cache hit rates per depth. Every decision is also run through the
//! uninstrumented library `deep_backup` from the same rng state, so the
//! instrumentation overhead is measured, not assumed.
//!
//!   fvi_timing --model M.json --ks 6x3 --games 20 [--dense] [--seed 900]
//!              [--eps 0.1] [--max-turns 300]
//!
//! `--dense` clears the sparse-forward weight copy, so inference takes the
//! pre-optimization dense path. Prints one JSON object.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::value::{
    deep_backup, legal_actions, outcome_value, sample_mixed, solve, Leaf, ValueCodec, ValueNet,
};
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::json;

fn arg_s(args: &[String], k: &str, d: &str) -> String {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| d.to_string())
}

#[derive(Default)]
struct Times {
    legal: Duration,
    clone: Duration,
    step: Duration,
    key: Duration,
    cache: Duration,
    encode: Duration,
    forward: Duration,
    terminal: Duration,
    solve: Duration,
    accumulate: Duration,
}

#[derive(Default)]
struct Counts {
    backups: [u64; 4],
    cells: u64,
    samples: u64,
    lookups: [u64; 4],
    hits: [u64; 4],
    net_evals: u64,
    terminal_leaves: u64,
    recursions: u64,
    lp_solves: u64,
    degenerate_solves: u64,
    lp_cells: u64,
    max_cells: u64,
    /// Cells whose first sample consumed no randomness: the remaining k-1
    /// samples of such a cell repeat the same transition.
    det_cells: [u64; 4],
    det_redundant_samples: [u64; 4],
}

struct Ctx<'a> {
    dex: &'a Dex,
    net: &'a ValueNet,
    codec: &'a ValueCodec,
    scratch: Vec<f32>,
    t: Times,
    c: Counts,
}

type Key = (u64, usize);

fn timed_backup(
    x: &mut Ctx,
    b: &Battle,
    ks: &[usize],
    rng: &mut SplitMix64,
    cache: &mut HashMap<Key, f64>,
    depth: usize,
) -> Option<(f64, [Vec<f64>; 2], [Vec<Option<SearchChoice>>; 2])> {
    let t0 = Instant::now();
    let actions = legal_actions(b, x.dex);
    x.t.legal += t0.elapsed();
    let (n0, n1) = (actions[0].len(), actions[1].len());
    if n0 == 0 || n1 == 0 {
        return None;
    }
    x.c.backups[depth] += 1;
    x.c.cells += (n0 * n1) as u64;
    let k = ks[0];
    let mut matrix = vec![0.0; n0 * n1];
    for (i, &c0) in actions[0].iter().enumerate() {
        for (j, &c1) in actions[1].iter().enumerate() {
            let mut acc = 0.0;
            for si in 0..k {
                x.c.samples += 1;
                let t = Instant::now();
                let mut c = b.clone();
                c.set_log_enabled(false);
                let seed = rng.next();
                c.reseed(seed);
                let t1 = Instant::now();
                c.apply_choices(x.dex, [c0, c1]).expect("legal choices");
                let t2 = Instant::now();
                if si == 0 && c.prng.lcg == nc2000_engine::prng::Prng::new(seed) {
                    x.c.det_cells[depth] += 1;
                    x.c.det_redundant_samples[depth] += (k - 1) as u64;
                }
                let key = (c.state_key(), ks.len());
                let t3 = Instant::now();
                let hit = cache.get(&key).copied();
                let t4 = Instant::now();
                x.t.clone += t1 - t;
                x.t.step += t2 - t1;
                x.t.key += t3 - t2;
                x.t.cache += t4 - t3;
                x.c.lookups[depth] += 1;
                let v = match hit {
                    Some(v) => {
                        x.c.hits[depth] += 1;
                        v
                    }
                    None => {
                        let v = if let Some(o) = c.outcome() {
                            let t = Instant::now();
                            x.c.terminal_leaves += 1;
                            let v = outcome_value(o);
                            x.t.terminal += t.elapsed();
                            v
                        } else if ks.len() == 1 {
                            let t = Instant::now();
                            x.scratch.clear();
                            x.codec.encode(&c, x.dex, &mut x.scratch);
                            let t1 = Instant::now();
                            let v = x.net.value(x.codec, &x.scratch);
                            let t2 = Instant::now();
                            x.t.encode += t1 - t;
                            x.t.forward += t2 - t1;
                            x.c.net_evals += 1;
                            v
                        } else {
                            x.c.recursions += 1;
                            match timed_backup(x, &c, &ks[1..], rng, cache, depth + 1) {
                                Some((v, _, _)) => v,
                                None => {
                                    let t = Instant::now();
                                    x.scratch.clear();
                                    x.codec.encode(&c, x.dex, &mut x.scratch);
                                    let t1 = Instant::now();
                                    let v = x.net.value(x.codec, &x.scratch);
                                    x.t.encode += t1 - t;
                                    x.t.forward += t1.elapsed();
                                    x.c.net_evals += 1;
                                    v
                                }
                            }
                        };
                        let t = Instant::now();
                        cache.insert(key, v);
                        x.t.cache += t.elapsed();
                        v
                    }
                };
                let t = Instant::now();
                acc += v;
                x.t.accumulate += t.elapsed();
            }
            matrix[i * n1 + j] = acc / k as f64;
        }
    }
    let t = Instant::now();
    let (value, mix) = solve(&matrix, n0, n1);
    x.t.solve += t.elapsed();
    if n0 > 1 && n1 > 1 {
        x.c.lp_solves += 1;
        x.c.lp_cells += (n0 * n1) as u64;
        x.c.max_cells = x.c.max_cells.max((n0 * n1) as u64);
    } else {
        x.c.degenerate_solves += 1;
    }
    Some((value, mix, actions))
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let codec = std::sync::Arc::new(ValueCodec::new(&dex));
    let model = arg_s(&args, "--model", "tmp/fvi/model-d0.json");
    let mut net = ValueNet::load(&root.join(&model)).expect("model");
    let dense = args.iter().any(|a| a == "--dense");
    if dense {
        for d in [&mut net.move_layer, &mut net.mon1, &mut net.mon2, &mut net.head1, &mut net.head2, &mut net.head3] {
            d.wt.clear();
        }
    }
    let net = std::sync::Arc::new(net);
    let ks: Vec<usize> = arg_s(&args, "--ks", "6x3").split('x').map(|x| x.parse().unwrap()).collect();
    let games: usize = arg_s(&args, "--games", "20").parse().unwrap();
    let seed: u64 = arg_s(&args, "--seed", "900").parse().unwrap();
    let eps: f64 = arg_s(&args, "--eps", "0.1").parse().unwrap();
    let max_turns: u16 = arg_s(&args, "--max-turns", "300").parse().unwrap();
    let teams: Vec<Vec<PokemonSet>> =
        ["data/team-pool-v2/team-pool.json", "data/belief-pool-v3/belief-pool.json"]
            .iter()
            .flat_map(|p| load_meta_pool(&root.join(p)).teams.into_iter().map(|t| t.sets))
            .collect();
    let leaf = Leaf::Net(net.clone(), codec.clone());

    // Timer self-cost: one Instant::now() pair.
    let probe = Instant::now();
    let mut sink = Duration::ZERO;
    for _ in 0..1_000_000 {
        let a = Instant::now();
        sink += a.elapsed();
    }
    let timer_pair_ns = probe.elapsed().as_nanos() as f64 / 1e6;

    let mut x = Ctx { dex: &dex, net: &net, codec: &codec, scratch: Vec::new(), t: Times::default(), c: Counts::default() };
    let mut decisions = 0u64;
    let mut instrumented = Duration::ZERO;
    let mut plain = Duration::ZERO;
    let mut game_loop = Duration::ZERO;
    let mut mismatches = 0u64;
    let wall = Instant::now();
    for g in 0..games {
        let mut rng = SplitMix64::new(seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let (ta, tb) = (rng.below(teams.len()), rng.below(teams.len()));
        let mut b = Battle::from_fixture(&dex, &rng.battle_seed(), &teams[ta], &teams[tb]).unwrap();
        b.set_log_enabled(false);
        while b.outcome().is_none() && b.turn <= max_turns {
            if b.turn == 0 {
                let mut picks = [None, None];
                for s in 0..2 {
                    let cs = b.legal_choices(&dex, s);
                    if !cs.is_empty() {
                        picks[s] = Some(cs[rng.below(cs.len())]);
                    }
                }
                b.apply_choices(&dex, picks).unwrap();
                continue;
            }
            let mut r_plain = rng.clone();
            let t = Instant::now();
            let mut cache = HashMap::new();
            let Some((value, mix, actions)) = timed_backup(&mut x, &b, &ks, &mut rng, &mut cache, 0) else {
                break;
            };
            instrumented += t.elapsed();
            let t = Instant::now();
            let mut pcache = HashMap::new();
            let mut scratch = Vec::new();
            let pb = deep_backup(&b, &dex, &leaf, &ks, &mut r_plain, &mut scratch, &mut pcache).unwrap();
            plain += t.elapsed();
            if pb.value != value {
                mismatches += 1;
            }
            decisions += 1;
            let t = Instant::now();
            let i = sample_mixed(&mix[0], eps, &mut rng);
            let j = sample_mixed(&mix[1], eps, &mut rng);
            b.apply_choices(&dex, [actions[0][i], actions[1][j]]).unwrap();
            game_loop += t.elapsed();
        }
    }
    let wall_s = wall.elapsed().as_secs_f64();
    let t = &x.t;
    let c = &x.c;
    let parts = [
        ("state_clone_reseed", t.clone),
        ("rules_apply_choices", t.step),
        ("rules_legal_actions", t.legal),
        ("state_hash", t.key),
        ("cache_lookup_insert", t.cache),
        ("features_encode", t.encode),
        ("net_forward", t.forward),
        ("equilibrium_solve", t.solve),
        ("terminal_value", t.terminal),
        ("accumulate", t.accumulate),
    ];
    let measured: Duration = parts.iter().map(|p| p.1).sum();
    let other = instrumented.saturating_sub(measured);
    let total = us(instrumented);
    let mut breakdown = serde_json::Map::new();
    for (k, d) in parts {
        breakdown.insert(k.into(), json!({"us_per_decision": us(d) / decisions as f64, "share": us(d) / total}));
    }
    breakdown.insert(
        "other_residual".into(),
        json!({"us_per_decision": us(other) / decisions as f64, "share": us(other) / total,
               "note": "loop control, matrix/Vec allocation, recursion bookkeeping, timer calls"}),
    );
    let rate = |h: u64, l: u64| if l == 0 { serde_json::Value::Null } else { json!(h as f64 / l as f64) };
    let out = json!({
        "config": {"model": model, "ks": ks, "inference": if dense {"dense (pre-optimization)"} else {"sparse (current)"},
                    "games": games, "seed": seed, "eps": eps, "threads": 1},
        "decisions": decisions,
        "wall_s": wall_s,
        "per_decision_ms": {"instrumented": total / decisions as f64 / 1e3,
                            "uninstrumented_library": us(plain) / decisions as f64 / 1e3,
                            "game_loop_outside_backup": us(game_loop) / decisions as f64 / 1e3},
        "instrumentation_overhead": {"timer_pair_ns": timer_pair_ns,
                                     "instrumented_over_plain": us(instrumented) / us(plain) - 1.0,
                                     "value_mismatches_vs_library": mismatches},
        "breakdown": breakdown,
        "counts_per_decision": {
            "backups_by_depth": c.backups.iter().map(|&v| v as f64 / decisions as f64).collect::<Vec<_>>(),
            "matrix_cells": c.cells as f64 / decisions as f64,
            "successor_samples": c.samples as f64 / decisions as f64,
            "net_evals": c.net_evals as f64 / decisions as f64,
            "terminal_leaves": c.terminal_leaves as f64 / decisions as f64,
            "lp_solves": c.lp_solves as f64 / decisions as f64,
            "degenerate_solves": c.degenerate_solves as f64 / decisions as f64,
            "mean_lp_cells": if c.lp_solves > 0 { c.lp_cells as f64 / c.lp_solves as f64 } else { 0.0 },
            "max_lp_cells": c.max_cells,
            "deterministic_cells_by_depth": c.det_cells.iter().map(|&v| v as f64 / decisions as f64).collect::<Vec<_>>(),
            "redundant_samples_in_deterministic_cells_by_depth": c.det_redundant_samples.iter().map(|&v| v as f64 / decisions as f64).collect::<Vec<_>>(),
        },
        "deterministic_sample_fraction": c.det_redundant_samples.iter().sum::<u64>() as f64 / c.samples as f64,
        "cache_hit_rate_by_depth": c.lookups.iter().zip(&c.hits).map(|(&l, &h)| rate(h, l)).collect::<Vec<_>>(),
        "cache_hit_rate_all": rate(c.hits.iter().sum(), c.lookups.iter().sum()),
        "per_unit_us": {
            "clone_reseed_per_sample": us(t.clone) / c.samples as f64,
            "apply_choices_per_sample": us(t.step) / c.samples as f64,
            "state_hash_per_sample": us(t.key) / c.samples as f64,
            "encode_per_net_eval": us(t.encode) / c.net_evals.max(1) as f64,
            "forward_per_net_eval": us(t.forward) / c.net_evals.max(1) as f64,
            "solve_per_backup": us(t.solve) / c.backups.iter().sum::<u64>() as f64,
        },
        "sink": sink.as_nanos() % 2,
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
