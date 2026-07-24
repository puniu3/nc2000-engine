//! M18 work item 1 — the reference counter for a community belief prior.
//!
//! Emits the `nc2000-belief-prior` format documented in
//! `docs/community-belief-prior-design.md`: per-`(species, move)` marginal
//! probabilities, plus optional item marginals, as plain declarative JSON a
//! non-technical human can edit.
//!
//! **Two sources, two different quantities.** The sampler consumes
//! `P(species carries move)`. Full-set sources give that directly; spectator
//! reveals give `P(species is seen using move)`, which is
//! `P(carries) x P(uses | carries)` — biased per move, since Rest is used
//! almost whenever carried and a situational coverage move often is not.
//! Rescaling cannot repair that, because it preserves the biased ratios. So
//! this tool counts both, labels which is which, and does **not** attempt a
//! reveal-rate correction: inferring `P(uses | carries)` is modelling this
//! scope excludes, and how to weigh the two sources is a community judgement
//! the developer cannot make for them.
//!
//! What the developer owns is this format and this reference count. What the
//! machine's owner owns is the content.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example count_belief_prior -- \
//!     [--source sets|reveals] [--corpus tmp/corpus-spectator] \
//!     [--battles 0-569] [--min-n 1] [--out data/belief-prior-v0.json]
//!   ... --check FILE     # report the per-species sum diagnostic for a table

use std::collections::BTreeMap;

use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, plain};
use nc2000_engine::dex::Dex;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// `species id -> (observations, move id -> count, item id -> count)`.
#[derive(Default)]
struct Counts {
    n: u64,
    moves: BTreeMap<String, u64>,
    items: BTreeMap<String, u64>,
}

fn to_id(dex: &Dex, name: &str) -> Option<String> {
    let key: String = name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    dex.moves.id(&key).map(|id| plain(dex.moves.key(id)))
}

/// Carry-marginals from complete 4-move sets (rentals + meta pool). Small
/// sample, but the quantity is the one the sampler actually wants.
fn count_sets(dex: &Dex, root: &std::path::Path) -> BTreeMap<String, Counts> {
    let src = load_sources(dex, root);
    let mut out: BTreeMap<String, Counts> = BTreeMap::new();
    for (sp, sets) in src.by_species.iter() {
        let key = dex.species.key(*sp).to_string();
        let e = out.entry(key).or_default();
        for s in sets {
            e.n += 1;
            if let Some(moves) = s["moves"].as_array() {
                for m in moves {
                    if let Some(id) = m.as_str().and_then(|m| to_id(dex, m)) {
                        *e.moves.entry(id).or_default() += 1;
                    }
                }
            }
            if let Some(item) = s["item"].as_str() {
                let id: String = item
                    .chars()
                    .filter(|c| c.is_ascii_alphanumeric())
                    .map(|c| c.to_ascii_lowercase())
                    .collect();
                if !id.is_empty() {
                    *e.items.entry(id).or_default() += 1;
                }
            }
        }
    }
    out
}

/// Reveal-marginals from the spectator corpus. Large sample of the WRONG
/// quantity — see the module doc. Emitted for drift detection and ranking.
fn count_reveals(
    dex: &Dex,
    root: &std::path::Path,
    corpus: &str,
    lo: usize,
    hi: usize,
) -> BTreeMap<String, Counts> {
    let mut out: BTreeMap<String, Counts> = BTreeMap::new();
    for (i, path) in corpus_files(&root.join(corpus)).into_iter().enumerate() {
        if i < lo || i > hi {
            continue;
        }
        let cb = load_battle(&path);
        for (species, moves) in cb.evidence.revealed_by_species() {
            let key: String = species
                .chars()
                .filter(|c| c.is_ascii_alphanumeric())
                .map(|c| c.to_ascii_lowercase())
                .collect();
            let Some(sp) = dex.species.id(&key) else { continue };
            let e = out.entry(dex.species.key(sp).to_string()).or_default();
            e.n += 1;
            for m in moves {
                if let Some(id) = to_id(dex, m) {
                    *e.moves.entry(id).or_default() += 1;
                }
            }
        }
    }
    out
}

fn emit(counts: &BTreeMap<String, Counts>, source: &str, min_n: u64) -> serde_json::Value {
    let mut species = serde_json::Map::new();
    for (sp, c) in counts {
        if c.n < min_n {
            continue;
        }
        let mut moves = serde_json::Map::new();
        for (m, k) in &c.moves {
            let p = (*k as f64 / c.n as f64 * 1000.0).round() / 1000.0;
            moves.insert(m.clone(), serde_json::json!(p));
        }
        let mut items = serde_json::Map::new();
        for (it, k) in &c.items {
            let p = (*k as f64 / c.n as f64 * 1000.0).round() / 1000.0;
            items.insert(it.clone(), serde_json::json!(p));
        }
        let mut entry = serde_json::Map::new();
        entry.insert("moves".into(), serde_json::Value::Object(moves));
        if !items.is_empty() {
            entry.insert("items".into(), serde_json::Value::Object(items));
        }
        entry.insert("n".into(), serde_json::json!(c.n));
        species.insert(sp.clone(), serde_json::Value::Object(entry));
    }
    serde_json::json!({
        "format": "nc2000-belief-prior",
        "version": 1,
        "note": format!(
            "reference count, source={source}. `n` is the observation count behind each \
             species and is informational. A species' move probabilities should sum to ~4.0 \
             when counted from complete sets; a sum well under 4 means the table was built \
             from reveals, which under-counts moves that are carried but not used."
        ),
        "species": species,
    })
}

/// The coverage diagnostic: per-species sum of move probabilities. ~4.0 =
/// carry-marginals from complete sets; well under 4 = reveal-derived or
/// hand-edited down; over 4 = the table claims more than four moves per set.
fn check(path: &str) {
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).expect("read prior")).expect("parse");
    let Some(species) = v["species"].as_object() else {
        println!("no `species` object — not a belief-prior table");
        return;
    };
    let mut sums: Vec<(String, f64, u64)> = Vec::new();
    for (sp, e) in species {
        let sum: f64 = e["moves"]
            .as_object()
            .map(|m| m.values().filter_map(|x| x.as_f64()).sum())
            .unwrap_or(0.0);
        sums.push((sp.clone(), sum, e["n"].as_u64().unwrap_or(0)));
    }
    sums.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mean = sums.iter().map(|s| s.1).sum::<f64>() / sums.len().max(1) as f64;
    println!("species {}  mean move-probability sum {mean:.2}  (complete sets => ~4.0)", sums.len());
    println!("\nfurthest from 4.0:");
    for (sp, sum, n) in sums.iter().take(5).chain(sums.iter().rev().take(5)) {
        let flag = if *sum > 4.5 {
            "  <- claims >4 moves per set"
        } else if *sum < 3.0 {
            "  <- under-counted (reveals?)"
        } else {
            ""
        };
        println!("  {sp:<16} sum {sum:>5.2}  n {n:>4}{flag}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let check_path = arg_s(&args, "--check", "");
    if !check_path.is_empty() {
        check(&check_path);
        return;
    }

    let source = arg_s(&args, "--source", "sets");
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-569");
    let min_n = arg(&args, "--min-n", 1) as u64;
    let out_path = arg_s(&args, "--out", "");
    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("569").parse().unwrap_or(569);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let counts = match source.as_str() {
        "sets" => count_sets(&dex, &root),
        "reveals" => count_reveals(&dex, &root, &corpus, lo, hi),
        other => {
            eprintln!("--source must be `sets` or `reveals` (got {other})");
            std::process::exit(2);
        }
    };

    let kept = counts.values().filter(|c| c.n >= min_n).count();
    let total_obs: u64 = counts.values().map(|c| c.n).sum();
    eprintln!("source {source}: {} species, {kept} kept at --min-n {min_n}, {total_obs} observations", counts.len());
    let mean_sum = counts
        .values()
        .filter(|c| c.n >= min_n)
        .map(|c| c.moves.values().sum::<u64>() as f64 / c.n as f64)
        .sum::<f64>()
        / kept.max(1) as f64;
    eprintln!("mean move-probability sum {mean_sum:.2} (complete sets => ~4.0)");

    let json = emit(&counts, &source, min_n);
    let text = serde_json::to_string_pretty(&json).unwrap();
    if out_path.is_empty() {
        println!("{text}");
    } else {
        std::fs::write(&out_path, text + "\n").expect("write");
        eprintln!("wrote {out_path}");
    }
}
