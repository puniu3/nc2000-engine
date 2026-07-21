//! M16a — equity calibration harness: does `eval01` measure equity?
//!
//! Pipeline: (1) generate decision-point positions by skuct self-play over
//! meta-pool pairs; (2) per position, estimate ground-truth equity as the
//! empirical side-0 score of K independent continuations (fresh battle
//! reseed + fresh searcher seeds per continuation = seed-marginal, both
//! sides skuct at --gt-iters); (3) compare eval01(position) against the
//! empirical rate: Pearson r, Brier vs outcomes (with the always-0.5
//! baseline), a 10-bucket calibration table, and per-feature-region slices
//! with ORIENTED bias — for a feature present on exactly one side the
//! position is mirrored so the feature side is side 0, making
//! mean(pred - emp) read directly as "the eval over/under-rates the side
//! that has <feature>" (the evasion blindness of 3629 would have shown here
//! as a large negative oriented bias on acc/eva stages).
//!
//! Ground-truth caveat (documented, deliberate): the empirical rate is
//! *policy-relative* equity — what skuct-vs-skuct play makes of the
//! position. A bias shared by the eval and the playout policy is invisible
//! here; the independent cross-check is M16b's human corpus.
//!
//! Positions from the same game are correlated; the per-game cap (default 3)
//! bounds it. Re-run after any eval/search change — this is a regression
//! net beside damage_conformance.
//!
//! Smoke:  cargo run --release -p nc2000-bot --example eval_calibration -- --smoke
//! Full:   ... --games 150 --per-game 4 --playouts 32 --gt-iters 300 (cx-scale)

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::agent::Agent;
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule};
use nc2000_engine::battle::Outcome;
use nc2000_engine::dex::{CondId, Dex};
use nc2000_engine::state::Battle;

fn skuct(iters: u32, c: f64, seed: u64) -> RmAgent {
    RmAgent::new(
        RmConfig { iterations: iters, rule: SelRule::Ucb, c, hp_buckets: 16, ..Default::default() },
        seed,
    )
}

// ---------------------------------------------------------------- features

/// (name, engine condition key, kind). Kind: how presence-per-side is read.
enum FeatKind {
    /// Side condition on side s's field.
    SideCond,
    /// Volatile on side s's active.
    ActiveVolatile,
    /// Status across side s's living mons.
    Status(nc2000_engine::state::Status),
    /// |stat boost| >= 2 on side s's active (indices 0..5).
    BigBoost,
    /// Nonzero accuracy/evasion stage on side s's active (indices 5, 6).
    AccEva,
}

struct Feat {
    name: &'static str,
    key: Option<CondId>,
    kind: FeatKind,
}

fn feats(dex: &Dex) -> Vec<Feat> {
    use FeatKind::*;
    use nc2000_engine::state::Status;
    let cond = |k: &str| dex.conds_id(k);
    vec![
        Feat { name: "spikes", key: cond("spikes"), kind: SideCond },
        Feat { name: "reflect", key: cond("reflect"), kind: SideCond },
        Feat { name: "lightscreen", key: cond("lightscreen"), kind: SideCond },
        Feat { name: "substitute", key: cond("substitute"), kind: ActiveVolatile },
        Feat { name: "confusion", key: cond("confusion"), kind: ActiveVolatile },
        Feat { name: "leechseed", key: cond("leechseed"), kind: ActiveVolatile },
        Feat { name: "curse", key: cond("curse"), kind: ActiveVolatile },
        Feat { name: "meanlook", key: cond("meanlook"), kind: ActiveVolatile },
        Feat { name: "partiallytrapped", key: cond("partiallytrapped"), kind: ActiveVolatile },
        Feat { name: "perishsong", key: cond("perishsong"), kind: ActiveVolatile },
        Feat { name: "encore", key: cond("encore"), kind: ActiveVolatile },
        Feat { name: "disable", key: cond("disable"), kind: ActiveVolatile },
        Feat { name: "slp", key: None, kind: Status(Status::Slp) },
        Feat { name: "frz", key: None, kind: Status(Status::Frz) },
        Feat { name: "tox", key: None, kind: Status(Status::Tox) },
        Feat { name: "bigboost", key: None, kind: BigBoost },
        Feat { name: "acceva", key: None, kind: AccEva },
    ]
}

/// Presence of each feature per side: bit 0 = side 0, bit 1 = side 1.
fn feat_presence(b: &Battle, fs: &[Feat]) -> Vec<u8> {
    fs.iter()
        .map(|f| {
            let mut m = 0u8;
            for s in 0..2 {
                let hit = match (&f.kind, f.key) {
                    (FeatKind::SideCond, Some(id)) => b.sides[s].has_side_condition(id),
                    (FeatKind::ActiveVolatile, Some(id)) => {
                        b.active_id(s).is_some_and(|p| b.poke(p).has_volatile(id))
                    }
                    (FeatKind::Status(st), _) => {
                        b.sides[s].roster.iter().any(|p| p.hp > 0 && p.status == *st)
                    }
                    (FeatKind::BigBoost, _) => b
                        .active_id(s)
                        .is_some_and(|p| b.poke(p).boosts[..5].iter().any(|&x| x.abs() >= 2)),
                    (FeatKind::AccEva, _) => b
                        .active_id(s)
                        .is_some_and(|p| b.poke(p).boosts[5..].iter().any(|&x| x != 0)),
                    _ => false,
                };
                if hit {
                    m |= 1 << s;
                }
            }
            m
        })
        .collect()
}

// ---------------------------------------------------------------- main

struct Pos {
    battle: Battle,
    game: usize,
    turn: u16,
    presence: Vec<u8>,
    weather: bool,
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let smoke = args.iter().any(|a| a == "--smoke");
    let games = arg(&args, "--games", if smoke { 12 } else { 40 });
    let per_game = arg(&args, "--per-game", 3);
    let gen_iters = arg(&args, "--gen-iters", if smoke { 100 } else { 300 }) as u32;
    let gt_iters = arg(&args, "--gt-iters", if smoke { 100 } else { 300 }) as u32;
    let playouts = arg(&args, "--playouts", if smoke { 8 } else { 16 });
    let seed = arg(&args, "--seed", 1) as u64;
    let threads = arg(
        &args,
        "--threads",
        std::thread::available_parallelism().map(|n| n.get().saturating_sub(1).max(1)).unwrap_or(4),
    );
    let max_turns: u16 = 500;

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let fs = feats(&dex);
    eprintln!(
        "games {games} per-game {per_game} gen {gen_iters} gt {gt_iters} playouts {playouts} \
         seed {seed} threads {threads} pool {} teams",
        pool.teams.len()
    );

    // ---- phase 1: self-play position generation (parallel by game)
    let positions: Mutex<Vec<Pos>> = Mutex::new(Vec::new());
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let g = cursor.fetch_add(1, Ordering::Relaxed);
                if g >= games {
                    return;
                }
                let mut rng = SplitMix64::new(seed ^ (g as u64).wrapping_mul(0x9E37_79B9));
                let a = rng.below(pool.teams.len());
                let mut bteam = rng.below(pool.teams.len() - 1);
                if bteam >= a {
                    bteam += 1;
                }
                let bs = rng.battle_seed();
                let mut b = Battle::from_fixture(&dex, &bs, &pool.teams[a].sets, &pool.teams[bteam].sets)
                    .unwrap();
                b.set_log_enabled(false);
                let mut agents: Vec<Box<dyn Agent>> = vec![
                    Box::new(skuct(gen_iters, 1.0, rng.next())),
                    Box::new(skuct(gen_iters, 1.0, rng.next())),
                ];
                // play_game inlined so every decision state is a snapshot
                // candidate; reservoir-sample per_game of them.
                let mut reservoir: Vec<(Battle, u16)> = Vec::new();
                let mut seen = 0usize;
                loop {
                    if b.outcome().is_some() || b.turn > 100 {
                        break;
                    }
                    if b.turn >= 1 && b.active_id(0).is_some() && b.active_id(1).is_some() {
                        seen += 1;
                        if reservoir.len() < per_game {
                            reservoir.push((b.clone(), b.turn));
                        } else if rng.next_f64() < per_game as f64 / seen as f64 {
                            let slot = rng.below(per_game);
                            reservoir[slot] = (b.clone(), b.turn);
                        }
                    }
                    let mut picks = [None, None];
                    for s in 0..2 {
                        let cs = b.legal_choices(&dex, s);
                        if !cs.is_empty() {
                            picks[s] = Some(agents[s].choose(&b, &dex, s, &cs));
                        }
                    }
                    if picks == [None, None] {
                        break;
                    }
                    if b.apply_choices(&dex, picks).is_err() {
                        break;
                    }
                }
                let mut out = positions.lock().unwrap();
                for (battle, turn) in reservoir {
                    let presence = feat_presence(&battle, &fs);
                    let weather = battle.field.weather.is_some();
                    out.push(Pos { battle, game: g, turn, presence, weather });
                }
            });
        }
    });
    let mut positions = positions.into_inner().unwrap();
    positions.sort_by_key(|p| (p.game, p.turn)); // deterministic order
    let n = positions.len();
    eprintln!("phase 1 done: {n} positions");

    // ---- phase 2: ground truth (parallel over position x playout jobs)
    let results: Mutex<Vec<Vec<f64>>> = Mutex::new(vec![Vec::new(); n]);
    let jobs = n * playouts;
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= jobs {
                    return;
                }
                let (pi, k) = (j / playouts, j % playouts);
                let mut rng = SplitMix64::new(
                    seed ^ 0xC0FF_EE00 ^ (pi as u64) << 20 ^ (k as u64).wrapping_mul(0x1234_5678_9ABC),
                );
                let mut b = positions[pi].battle.clone();
                b.reseed(rng.next());
                let mut a0 = skuct(gt_iters, 1.0, rng.next());
                let mut a1 = skuct(gt_iters, 1.0, rng.next());
                let score = match play_game(&dex, &mut b, &mut [&mut a0, &mut a1], max_turns) {
                    Ok(GameResult::Outcome(Outcome::P1Win)) => 1.0,
                    Ok(GameResult::Outcome(Outcome::P2Win)) => 0.0,
                    _ => 0.5,
                };
                results.lock().unwrap()[pi].push(score);
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 200 == 0 {
                    eprintln!("  gt {d}/{jobs}");
                }
            });
        }
    });
    let results = results.into_inner().unwrap();

    // ---- phase 3: report
    let w = EvalWeights::default();
    let preds: Vec<f64> = positions.iter().map(|p| eval01(&p.battle, &dex, &w)).collect();
    let emps: Vec<f64> =
        results.iter().map(|r| r.iter().sum::<f64>() / r.len().max(1) as f64).collect();

    // CSV
    let csv_path = root.join("tmp/eval-calibration.csv");
    let mut csv = String::from("game,turn,pred,emp,k,weather");
    for f in &fs {
        csv.push_str(&format!(",{}_s0,{}_s1", f.name, f.name));
    }
    csv.push('\n');
    for i in 0..n {
        csv.push_str(&format!(
            "{},{},{:.4},{:.4},{},{}",
            positions[i].game,
            positions[i].turn,
            preds[i],
            emps[i],
            results[i].len(),
            positions[i].weather as u8
        ));
        for m in &positions[i].presence {
            csv.push_str(&format!(",{},{}", m & 1, (m >> 1) & 1));
        }
        csv.push('\n');
    }
    std::fs::create_dir_all(root.join("tmp")).ok();
    std::fs::File::create(&csv_path).unwrap().write_all(csv.as_bytes()).unwrap();
    eprintln!("csv -> {}", csv_path.display());

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let pearson = |x: &[f64], y: &[f64]| {
        let (mx, my) = (mean(x), mean(y));
        let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let vx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
        let vy: f64 = y.iter().map(|b| (b - my) * (b - my)).sum();
        cov / (vx.sqrt() * vy.sqrt()).max(1e-12)
    };
    // Brier vs raw outcomes (and the 0.5 baseline on the same outcomes)
    let mut brier = 0.0;
    let mut brier05 = 0.0;
    let mut cnt = 0.0;
    for i in 0..n {
        for &o in &results[i] {
            brier += (preds[i] - o) * (preds[i] - o);
            brier05 += (0.5 - o) * (0.5 - o);
            cnt += 1.0;
        }
    }
    println!("\n== M16a eval01 calibration ==");
    println!(
        "positions {n}  playouts/pos {playouts}  gen skuct:{gen_iters}  gt skuct:{gt_iters}  seed {seed}"
    );
    println!(
        "pearson r {:.3}   MSE(pred,emp) {:.4}   brier {:.4} (always-0.5 baseline {:.4})",
        pearson(&preds, &emps),
        preds.iter().zip(&emps).map(|(p, e)| (p - e) * (p - e)).sum::<f64>() / n.max(1) as f64,
        brier / f64::max(cnt, 1.0),
        brier05 / f64::max(cnt, 1.0)
    );
    println!("\ncalibration (pred bucket -> empirical):");
    for bkt in 0..10 {
        let (lo, hi) = (bkt as f64 / 10.0, (bkt + 1) as f64 / 10.0);
        let idx: Vec<usize> =
            (0..n).filter(|&i| preds[i] >= lo && (preds[i] < hi || bkt == 9)).collect();
        if idx.is_empty() {
            continue;
        }
        let mp = mean(&idx.iter().map(|&i| preds[i]).collect::<Vec<_>>());
        let me = mean(&idx.iter().map(|&i| emps[i]).collect::<Vec<_>>());
        println!(
            "  [{lo:.1},{hi:.1})  n {:>4}  pred {mp:.3}  emp {me:.3}  gap {:+.3}",
            idx.len(),
            mp - me
        );
    }
    println!("\nfeature slices (oriented: feature side mirrored to side 0; bias = pred - emp):");
    println!(
        "  {:<18} {:>5} {:>5}  {:>9} {:>7}",
        "feature", "n_any", "n_1s", "bias(1s)", "MSE"
    );
    for (fi, f) in fs.iter().enumerate() {
        let any: Vec<usize> = (0..n).filter(|&i| positions[i].presence[fi] != 0).collect();
        // oriented subset: present on exactly one side
        let one: Vec<usize> = (0..n)
            .filter(|&i| {
                let m = positions[i].presence[fi];
                m == 1 || m == 2
            })
            .collect();
        let mut bias = f64::NAN;
        if !one.is_empty() {
            bias = mean(
                &one.iter()
                    .map(|&i| {
                        if positions[i].presence[fi] == 1 {
                            preds[i] - emps[i]
                        } else {
                            (1.0 - preds[i]) - (1.0 - emps[i])
                        }
                    })
                    .collect::<Vec<_>>(),
            );
        }
        let mse = if any.is_empty() {
            f64::NAN
        } else {
            mean(&any.iter().map(|&i| (preds[i] - emps[i]) * (preds[i] - emps[i])).collect::<Vec<_>>())
        };
        println!("  {:<18} {:>5} {:>5}  {:>+9.3} {:>7.4}", f.name, any.len(), one.len(), bias, mse);
    }
    // ---- --ab: paired eval-variant comparison on the SAME positions + GT
    if args.iter().any(|a| a == "--ab") {
        // M17c candidates: KO-race weight (rev-1 winners are the defaults now)
        let mut variants: Vec<(&str, EvalWeights)> = Vec::new();
        variants.push(("default", EvalWeights::default()));
        variants.push(("race2", EvalWeights { race: 2.0, ..EvalWeights::default() }));
        variants.push(("race3", EvalWeights { race: 3.0, ..EvalWeights::default() }));
        variants.push(("race4", EvalWeights { race: 4.0, ..EvalWeights::default() }));

        let feat_idx = |name: &str| fs.iter().position(|f| f.name == name).unwrap();
        let oriented_bias = |preds: &[f64], fi: usize| -> f64 {
            let one: Vec<usize> = (0..n)
                .filter(|&i| {
                    let m = positions[i].presence[fi];
                    m == 1 || m == 2
                })
                .collect();
            if one.is_empty() {
                return f64::NAN;
            }
            one.iter()
                .map(|&i| {
                    if positions[i].presence[fi] == 1 {
                        preds[i] - emps[i]
                    } else {
                        (1.0 - preds[i]) - (1.0 - emps[i])
                    }
                })
                .sum::<f64>()
                / one.len() as f64
        };
        println!("\n== --ab paired comparison (same positions, same GT; GT policy = default eval) ==");
        println!(
            "  {:<24} {:>6} {:>7} {:>7}  {:>8} {:>8} {:>8} {:>8}",
            "variant", "r", "brier", "MSE", "slp", "sub", "frz", "tox"
        );
        for (name, vw) in &variants {
            let vp: Vec<f64> = positions.iter().map(|p| eval01(&p.battle, &dex, vw)).collect();
            let mut vbrier = 0.0;
            let mut vcnt = 0.0;
            for i in 0..n {
                for &o in &results[i] {
                    vbrier += (vp[i] - o) * (vp[i] - o);
                    vcnt += 1.0;
                }
            }
            let vmse =
                vp.iter().zip(&emps).map(|(p, e)| (p - e) * (p - e)).sum::<f64>() / n.max(1) as f64;
            println!(
                "  {:<24} {:>6.3} {:>7.4} {:>7.4}  {:>+8.3} {:>+8.3} {:>+8.3} {:>+8.3}",
                name,
                pearson(&vp, &emps),
                vbrier / f64::max(vcnt, 1.0),
                vmse,
                oriented_bias(&vp, feat_idx("slp")),
                oriented_bias(&vp, feat_idx("substitute")),
                oriented_bias(&vp, feat_idx("frz")),
                oriented_bias(&vp, feat_idx("tox")),
            );
        }
    }

    let clean: Vec<usize> = (0..n)
        .filter(|&i| positions[i].presence.iter().all(|&m| m == 0) && !positions[i].weather)
        .collect();
    if !clean.is_empty() {
        let mse = mean(
            &clean.iter().map(|&i| (preds[i] - emps[i]) * (preds[i] - emps[i])).collect::<Vec<_>>(),
        );
        println!("  {:<18} {:>5} {:>5}  {:>9} {:>7.4}", "none(control)", clean.len(), "-", "-", mse);
    }
}
