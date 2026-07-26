//! Does the node key actually control tree shape?
//!
//! The threshold-preserving key (`smmcts::hp_class`) was built on the premise
//! that the uniform HP grid splits states that are the same decision, costing
//! transposition hits and therefore depth. That premise is testable in one
//! number: distinct nodes at a fixed iteration budget. The search expands at
//! most one node per iteration, so `nodes / iterations` is the miss rate — at
//! 1.0 the tree shares nothing and the key is irrelevant to depth.
//!
//! Full-information `SkuctSearch` is measured here rather than the blind
//! product path on purpose: a determinized search draws a fresh opponent set
//! every iteration, so its states cannot coincide for reasons that have
//! nothing to do with HP.
//!
//! Usage: cargo run --release -p nc2000-bot --example key_shape -- [iters] [games]

use conformance::fixture::{repo_root, Fixture};
use conformance::load_dex;
use nc2000_bot::smmcts::{RmConfig, SelRule, SkuctSearch};
use nc2000_bot::SplitMix64;
use nc2000_engine::state::Battle;

fn main() {
    let mut args = std::env::args().skip(1);
    let iters: u32 = args.next().and_then(|v| v.parse().ok()).unwrap_or(3000);
    let positions: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(6);

    let dex = load_dex();
    let root = repo_root();
    let files = conformance::fixture::corpus_files(&root.join("fixtures/corpus-v1/full"));
    let mut rng = SplitMix64::new(0xC0FFEE);
    let mut cases: Vec<Battle> = Vec::new();
    for f in files.iter() {
        if cases.len() >= positions {
            break;
        }
        let Ok(fx) = Fixture::load(f) else {
            eprintln!("skip {f:?}: load");
            continue;
        };
        let mut b = match Battle::from_fixture(&dex, "1,2,3,4", &fx.p1team, &fx.p2team) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip {f:?}: build {e:?}");
                continue;
            }
        };
        b.set_log_enabled(false);
        if let Err(e) = b.choose(&dex, 0, "team 1,2,3") {
            eprintln!("skip {f:?}: p1 team {e:?}");
            continue;
        }
        if let Err(e) = b.choose(&dex, 1, "team 1,2,3") {
            eprintln!("skip {f:?}: p2 team {e:?}");
            continue;
        }
        // Random legal play for a few turns: HP abstraction is supposed to
        // matter mid-game, not at a full-HP root.
        let mut ok = true;
        for _ in 0..6 {
            if b.outcome().is_some() {
                break;
            }
            let mut picks = [None, None];
            for s in 0..2 {
                let cs = b.legal_choices(&dex, s);
                if !cs.is_empty() {
                    picks[s] = Some(cs[rng.below(cs.len())]);
                }
            }
            if picks == [None, None] || b.apply_choices(&dex, picks).is_err() {
                ok = false;
                break;
            }
        }
        if ok && b.outcome().is_none() {
            cases.push(b);
        }
    }
    // Fan-out of ONE joint action under chance alone: if this is already in
    // the hundreds, the budget is spent before the tree finishes ply 1 and no
    // key abstraction can buy depth.
    println!("chance fan-out of a single joint action (200 samples each):");
    println!("  {:<14} {:>12} {:>12}", "key", "distinct", "of 200");
    for (label, threshold_key, key_no_damage) in [
        ("uniform-16", false, false),
        ("threshold", true, false),
        ("uni+nodmg", false, true),
        ("thr+nodmg", true, true),
    ] {
        let cfg = RmConfig {
            hp_buckets: 16,
            threshold_key,
            key_no_damage,
            ..RmConfig::default()
        };
        let mut total = 0usize;
        for (i, b) in cases.iter().enumerate() {
            let mut seen = std::collections::HashSet::new();
            let a0 = b.clone().legal_choices(&dex, 0);
            let a1 = b.clone().legal_choices(&dex, 1);
            if a0.is_empty() || a1.is_empty() {
                continue;
            }
            for k in 0..200u64 {
                let mut sim = b.clone();
                sim.reseed(0xA11CE ^ (k << 8) ^ i as u64);
                if sim.apply_choices(&dex, [Some(a0[0]), Some(a1[0])]).is_err() {
                    continue;
                }
                seen.insert(nc2000_bot::smmcts::key_for_test(&cfg, &dex, &mut sim));
            }
            total += seen.len();
        }
        println!(
            "  {:<14} {:>12.1} {:>12}",
            label,
            total as f64 / cases.len() as f64,
            200
        );
    }
    println!();
    println!("{} positions, {iters} iterations each\n", cases.len());
    println!(
        "  {:<14} {:>10} {:>11} {:>9}",
        "key", "nodes", "mean depth", "top1"
    );

    for (label, threshold_key, key_no_damage) in [
        ("uniform-16", false, false),
        ("threshold", true, false),
        ("uni+nodmg", false, true),
        ("thr+nodmg", true, true),
    ] {
        let mut nodes = 0usize;
        let mut top1 = 0.0;
        let mut depth = 0.0;
        for (i, b) in cases.iter().enumerate() {
            let cfg = RmConfig {
                iterations: iters,
                rule: SelRule::Ucb,
                c: 1.0,
                hp_buckets: 16,
                threshold_key,
                key_no_damage,
                ..RmConfig::default()
            };
            let mut s = SkuctSearch::new(b, &dex, cfg, 1 + i as u64);
            s.step(&dex, iters);
            nodes += s.node_count();
            depth += s.mean_depth();
            let v = s.visits(0);
            let total: u32 = v.iter().sum::<u32>().max(1);
            top1 += v.iter().copied().max().unwrap_or(0) as f64 / total as f64;
        }
        let n = cases.len() as f64;
        println!(
            "  {:<14} {:>10.0} {:>11.2} {:>9.3}",
            label,
            nodes as f64 / n,
            depth / n,
            top1 / n
        );
    }
}
