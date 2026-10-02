//! S0 — cost of one Shapley backup, and how far `eval01` is from being a
//! fixed point of it.
//!
//! A backup at a decision state s is `val(M)`, where `M[i][j]` is the
//! expected value of the successor states after joint action (i, j) and
//! `val` is the zero-sum matrix-game value (row = side 0, maximizing). The
//! true game value V* satisfies `V*(s) = val(M[V*])` everywhere, so the
//! residual `val(M[eval01]) − eval01(s)` is a necessary-condition check on
//! the eval: zero residual does not prove correctness, a large one proves
//! inconsistency.
//!
//! Three ways to take the expectation over chance, each timed:
//!   exact  — `enumerate_step` (merged damage rolls; capped per cell),
//!   mean   — damage distribution collapsed to its mean (approximate),
//!   sample — K seeded engine runs per cell (unbiased, noisy).
//!
//! Positions are reconstructed from the human spectator corpus (imputed
//! hidden sets — each is a legal full-information state, which is what the
//! perfect-information core is solved over).
//!
//! Usage: s0_backup_probe [--battles 0-99] [--per-battle 2] [--threads N]
//!          [--cap 20000] [--samples 32] [--seed 1] [--out tmp/s0-backup.csv]

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, reconstruct_context_with_pool};
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::exact::solve_matrix_full;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::enumerate::enumerate_step_with_damage_mode;
use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::DamageRollMode;
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

fn leaf_value(b: &Battle, dex: &Dex, w: &EvalWeights) -> f64 {
    match b.outcome() {
        Some(Outcome::P1Win) => 1.0,
        Some(Outcome::P2Win) => 0.0,
        Some(Outcome::Tie) => 0.5,
        None => eval01(b, dex, w),
    }
}

fn game_value(m: &[f64], n0: usize, n1: usize) -> f64 {
    if n0 == 1 && n1 == 1 {
        m[0]
    } else if n1 == 1 {
        m.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    } else if n0 == 1 {
        m.iter().cloned().fold(f64::INFINITY, f64::min)
    } else {
        solve_matrix_full(m, n0, n1).value
    }
}

fn actions(b: &Battle, dex: &Dex, side: usize, need: bool) -> Vec<Option<SearchChoice>> {
    if need {
        b.clone().legal_choices(dex, side).into_iter().map(Some).collect()
    } else {
        vec![None]
    }
}

struct Mode {
    value: f64,
    runs: usize,
    leaves: usize,
    ms: f64,
    capped: usize,
}

fn enumerated(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
    a0: &[Option<SearchChoice>],
    a1: &[Option<SearchChoice>],
    cap: usize,
    mode: DamageRollMode,
) -> Mode {
    let t = Instant::now();
    let (n0, n1) = (a0.len(), a1.len());
    let mut m = vec![0.0; n0 * n1];
    let (mut runs, mut leaves, mut capped) = (0, 0, 0);
    for (i, &c0) in a0.iter().enumerate() {
        for (j, &c1) in a1.iter().enumerate() {
            match enumerate_step_with_damage_mode(dex, b, [c0, c1], cap, mode) {
                Some(step) => {
                    runs += step.runs;
                    leaves += step.leaves.len();
                    m[i * n1 + j] =
                        step.leaves.iter().map(|l| l.prob * leaf_value(&l.battle, dex, w)).sum();
                }
                None => {
                    runs += cap;
                    capped += 1;
                    m[i * n1 + j] = f64::NAN;
                }
            }
        }
    }
    let value = if capped > 0 { f64::NAN } else { game_value(&m, n0, n1) };
    Mode { value, runs, leaves, ms: t.elapsed().as_secs_f64() * 1e3, capped }
}

#[allow(clippy::too_many_arguments)]
fn sampled(
    b: &Battle,
    dex: &Dex,
    w: &EvalWeights,
    a0: &[Option<SearchChoice>],
    a1: &[Option<SearchChoice>],
    k: usize,
    seed: u64,
) -> Mode {
    let t = Instant::now();
    let (n0, n1) = (a0.len(), a1.len());
    let mut m = vec![0.0; n0 * n1];
    let mut rng = SplitMix64::new(seed);
    for (i, &c0) in a0.iter().enumerate() {
        for (j, &c1) in a1.iter().enumerate() {
            let mut acc = 0.0;
            for _ in 0..k {
                let mut c = b.clone();
                c.set_log_enabled(false);
                c.reseed(rng.next());
                c.apply_choices(dex, [c0, c1]).expect("legal choices");
                acc += leaf_value(&c, dex, w);
            }
            m[i * n1 + j] = acc / k as f64;
        }
    }
    Mode {
        value: game_value(&m, n0, n1),
        runs: n0 * n1 * k,
        leaves: n0 * n1 * k,
        ms: t.elapsed().as_secs_f64() * 1e3,
        capped: 0,
    }
}

struct Row {
    battle: usize,
    turn: u16,
    alive: [usize; 2],
    n: [usize; 2],
    eval: f64,
    exact: Mode,
    mean: Mode,
    sample: Mode,
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let range = arg_s(&args, "--battles", "0-99");
    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap().parse().unwrap();
        let hi: usize = it.next().unwrap().parse().unwrap();
        (lo, hi)
    };
    let per_battle: usize = arg_s(&args, "--per-battle", "2").parse().unwrap();
    let default_threads =
        std::thread::available_parallelism().map(|n| n.get().saturating_sub(1)).unwrap_or(4);
    let threads: usize =
        arg_s(&args, "--threads", &default_threads.to_string()).parse().unwrap();
    let cap: usize = arg_s(&args, "--cap", "20000").parse().unwrap();
    let k: usize = arg_s(&args, "--samples", "32").parse().unwrap();
    let seed: u64 = arg_s(&args, "--seed", "1").parse().unwrap();
    let out_path = arg_s(&args, "--out", "tmp/s0-backup.csv");

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let w = EvalWeights::default();
    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join("tmp/corpus-spectator"))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    eprintln!(
        "s0: {} battles, {per_battle}/battle, {threads} threads, cap {cap}, K {k}",
        files.len()
    );

    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (bi, path) = &files[j];
                let cb = load_battle(path);
                let mut rng = SplitMix64::new(seed ^ (*bi as u64).wrapping_mul(0xD1B5_4A32));
                let mut reservoir: Vec<Battle> = Vec::new();
                let mut seen = 0usize;
                for (di, d) in cb.decisions.iter().enumerate() {
                    let dseed = seed
                        ^ (*bi as u64).wrapping_mul(0x9E37_79B9_7F4A)
                        ^ (di as u64).wrapping_mul(0xBF58_476D)
                        ^ d.side as u64;
                    let Some(rec) = reconstruct_context_with_pool(
                        &dex,
                        &src,
                        pool.clone(),
                        &cb.lines,
                        &cb.evidence,
                        d,
                        dseed,
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
                    let needs = b.needs_choice();
                    let a0 = actions(&b, &dex, 0, needs[0]);
                    let a1 = actions(&b, &dex, 1, needs[1]);
                    if a0.is_empty() || a1.is_empty() {
                        continue;
                    }
                    let exact = enumerated(&b, &dex, &w, &a0, &a1, cap, DamageRollMode::Exact);
                    let mean = enumerated(&b, &dex, &w, &a0, &a1, cap, DamageRollMode::Mean);
                    let sseed = seed ^ (*bi as u64).wrapping_mul(0x2545_F491) ^ pi as u64;
                    let sample = sampled(&b, &dex, &w, &a0, &a1, k, sseed);
                    let row = Row {
                        battle: *bi,
                        turn: b.turn,
                        alive: [alive(&b, 0), alive(&b, 1)],
                        n: [a0.len(), a1.len()],
                        eval: eval01(&b, &dex, &w),
                        exact,
                        mean,
                        sample,
                    };
                    rows.lock().unwrap().push(row);
                }
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 10 == 0 {
                    eprintln!("  {n}/{} battles, {:.0}s", files.len(), t0.elapsed().as_secs_f64());
                }
            });
        }
    });

    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| (r.battle, r.turn));
    let mut f = std::fs::File::create(root.join(&out_path)).expect("create csv");
    writeln!(
        f,
        "battle,turn,alive0,alive1,n0,n1,eval,\
         v_exact,runs_exact,leaves_exact,ms_exact,capped_exact,\
         v_mean,runs_mean,leaves_mean,ms_mean,capped_mean,v_sample,ms_sample"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            f,
            "{},{},{},{},{},{},{:.6},{:.6},{},{},{:.2},{},{:.6},{},{},{:.2},{},{:.6},{:.2}",
            r.battle,
            r.turn,
            r.alive[0],
            r.alive[1],
            r.n[0],
            r.n[1],
            r.eval,
            r.exact.value,
            r.exact.runs,
            r.exact.leaves,
            r.exact.ms,
            r.exact.capped,
            r.mean.value,
            r.mean.runs,
            r.mean.leaves,
            r.mean.ms,
            r.mean.capped,
            r.sample.value,
            r.sample.ms,
        )
        .unwrap();
    }

    println!("positions {} ({:.0}s wall)", rows.len(), t0.elapsed().as_secs_f64());
    let strata: [(&str, fn(usize) -> bool); 3] =
        [("alive<=2", |t| t <= 2), ("alive 3-4", |t| (3..=4).contains(&t)), ("alive 5-6", |t| t >= 5)];
    for (name, pred) in strata {
        let rs: Vec<&Row> = rows.iter().filter(|r| pred(r.alive[0] + r.alive[1])).collect();
        if rs.is_empty() {
            continue;
        }
        let capped = rs.iter().filter(|r| r.exact.capped > 0).count();
        let mut ex_runs: Vec<f64> =
            rs.iter().filter(|r| r.exact.capped == 0).map(|r| r.exact.runs as f64).collect();
        let mut ex_ms: Vec<f64> =
            rs.iter().filter(|r| r.exact.capped == 0).map(|r| r.exact.ms).collect();
        let mut mn_runs: Vec<f64> = rs.iter().map(|r| r.mean.runs as f64).collect();
        let mut mn_ms: Vec<f64> = rs.iter().map(|r| r.mean.ms).collect();
        let mut sm_ms: Vec<f64> = rs.iter().map(|r| r.sample.ms).collect();
        let mut cells: Vec<f64> = rs.iter().map(|r| (r.n[0] * r.n[1]) as f64).collect();
        println!(
            "\n[{name}] n={} cells p50 {:.0} p90 {:.0} | exact capped {}/{} ({:.0}%)",
            rs.len(),
            pct(&mut cells, 0.5),
            pct(&mut cells, 0.9),
            capped,
            rs.len(),
            100.0 * capped as f64 / rs.len() as f64
        );
        println!(
            "  exact  runs p50 {:.0} p90 {:.0}  ms p50 {:.1} p90 {:.1}",
            pct(&mut ex_runs, 0.5),
            pct(&mut ex_runs, 0.9),
            pct(&mut ex_ms, 0.5),
            pct(&mut ex_ms, 0.9)
        );
        println!(
            "  mean   runs p50 {:.0} p90 {:.0}  ms p50 {:.1} p90 {:.1}",
            pct(&mut mn_runs, 0.5),
            pct(&mut mn_runs, 0.9),
            pct(&mut mn_ms, 0.5),
            pct(&mut mn_ms, 0.9)
        );
        println!("  sample ms p50 {:.1} p90 {:.1} (K={k})", pct(&mut sm_ms, 0.5), pct(&mut sm_ms, 0.9));
        let resid = |sel: &dyn Fn(&Row) -> f64| -> (usize, f64, f64, f64, f64) {
            let xs: Vec<(f64, f64)> =
                rs.iter().map(|r| (sel(r), r.eval)).filter(|(v, _)| v.is_finite()).collect();
            let n = xs.len() as f64;
            let bias = xs.iter().map(|(v, e)| v - e).sum::<f64>() / n;
            let rms = (xs.iter().map(|(v, e)| (v - e).powi(2)).sum::<f64>() / n).sqrt();
            let mut abs: Vec<f64> = xs.iter().map(|(v, e)| (v - e).abs()).collect();
            (xs.len(), bias, rms, pct(&mut abs.clone(), 0.5), pct(&mut abs, 0.9))
        };
        for (label, sel) in [
            ("exact", &(|r: &Row| r.exact.value) as &dyn Fn(&Row) -> f64),
            ("mean", &|r: &Row| r.mean.value),
            ("sample", &|r: &Row| r.sample.value),
        ] {
            let (n, bias, rms, p50, p90) = resid(sel);
            println!(
                "  residual[{label:6}] n {n} bias {bias:+.4} rms {rms:.4} |r| p50 {p50:.4} p90 {p90:.4}"
            );
        }
        let pairs = |sel: &dyn Fn(&Row) -> f64| -> (usize, f64, f64) {
            let xs: Vec<f64> = rs
                .iter()
                .filter(|r| r.exact.value.is_finite())
                .map(|r| (sel(r) - r.exact.value).abs())
                .collect();
            let n = xs.len();
            let mut v = xs.clone();
            (n, pct(&mut v.clone(), 0.5), pct(&mut v, 0.9))
        };
        let (n, p50, p90) = pairs(&|r: &Row| r.mean.value);
        println!("  |mean-exact|   n {n} p50 {p50:.4} p90 {p90:.4}");
        let (n, p50, p90) = pairs(&|r: &Row| r.sample.value);
        println!("  |sample-exact| n {n} p50 {p50:.4} p90 {p90:.4}");
    }
    println!("\ncsv: {out_path}");
}
