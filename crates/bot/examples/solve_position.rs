//! Solve a hand-entered position: the CLI twin of the browser's solver
//! screen. Same `PositionSpec` in, same `analysis::report` out, so a number
//! the screen shows can always be reproduced here — and a disagreement
//! between the two is a wasm bug, not a matter of opinion.
//!
//! ```text
//! cargo run --release -p nc2000-bot --example solve_position -- POSITION.json
//!     [--iters 30000] [--seed 1] [--plies 6] [--pool FILE] [--json]
//! ```
//!
//! The position's own side is analyzed under the product's information
//! structure: its sets exact, the opponent public-only, hidden fields left
//! to the belief.

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::analysis;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::position::PositionSpec;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::RmConfig;
use serde_json::Value;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path = None;
    let mut iters = 30_000u32;
    let mut seed = 1u64;
    let mut plies = 6usize;
    let mut pool_path = repo_root().join("data/belief-pool-v1/belief-pool.json");
    let mut as_json = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--iters" => {
                i += 1;
                iters = args[i].parse().expect("--iters");
            }
            "--seed" => {
                i += 1;
                seed = args[i].parse().expect("--seed");
            }
            "--plies" => {
                i += 1;
                plies = args[i].parse().expect("--plies");
            }
            "--pool" => {
                i += 1;
                pool_path = std::path::PathBuf::from(&args[i]);
            }
            "--json" => as_json = true,
            other => path = Some(other.to_string()),
        }
        i += 1;
    }
    let path = path.unwrap_or_else(|| {
        eprintln!("usage: solve_position POSITION.json [--iters N] [--seed N] [--plies N] [--pool FILE] [--json]");
        std::process::exit(2);
    });

    let dex = load_dex();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let spec = PositionSpec::parse(&text).unwrap_or_else(|e| {
        eprintln!("position rejected: {e}");
        std::process::exit(1);
    });
    let pool = load_meta_pool(&pool_path);
    let cfg = RmConfig { rule: nc2000_bot::smmcts::SelRule::Ucb, ..RmConfig::default() };
    let mut agent = ProtocolAgent::new(&dex, spec.side, pool, cfg, seed);
    if let Err(e) = agent.set_position(&dex, &spec) {
        eprintln!("position rejected: {e}");
        std::process::exit(1);
    }
    let t0 = std::time::Instant::now();
    agent.step(&dex, iters).unwrap();
    let ms = t0.elapsed().as_millis();
    let report = analysis::report(&agent, &dex, plies, seed);

    if as_json {
        println!("{report}");
        return;
    }
    print_report(&report, ms);
}

fn print_report(r: &Value, ms: u128) {
    let pct = |v: &Value| v.as_f64().map(|x| format!("{:.1}%", x * 100.0)).unwrap_or_default();
    println!(
        "turn {}  side {}  {} iterations in {ms} ms",
        r["turn"], r["side"], r["iterations"]
    );
    let belief = &r["belief"];
    println!(
        "belief: {} candidate(s){}",
        belief["count"],
        if belief["fallback"].as_bool() == Some(true) { " (off-pool fallback)" } else { "" }
    );

    println!(
        "\nposition value {} (both sides at equilibrium over the sampled matrix)",
        pct(&r["equilibrium"]["value"])
    );
    println!("\nactions (playouts / search share / vs their equilibrium / vs their best reply / equilibrium mix):");
    for a in r["actions"].as_array().into_iter().flatten() {
        let mark = if a["dominated"].as_bool() == Some(true) { " [dominated]" } else { "" };
        let why = a["reason"].as_str().map(|w| format!("  <- {w}")).unwrap_or_default();
        let worst = if a["worst"].is_null() { "   —".to_string() } else { pct(&a["worst"]) };
        println!(
            "  {:<22} {:>6}  {:>7}  {:>7}  {:>7}  {:>6}{mark}{why}",
            a["input"].as_str().unwrap_or(""),
            a["visits"].as_u64().unwrap_or(0),
            pct(&a["frac"]),
            pct(&a["equity"]),
            worst,
            pct(&a["mix"]),
        );
    }

    let cols = r["matrix"]["cols"].as_array().cloned().unwrap_or_default();
    if !cols.is_empty() {
        println!("\nroot matrix (win rate; '.' = never sampled):");
        print!("  {:<22}", "");
        for c in &cols {
            print!("{:>14}", short(c["input"].as_str().unwrap_or("")));
        }
        println!();
        print!("  {:<22}", "available in");
        for c in &cols {
            print!("{:>14}", pct(&c["available"]));
        }
        println!();
        for (i, a) in r["actions"].as_array().into_iter().flatten().enumerate() {
            print!("  {:<22}", a["input"].as_str().unwrap_or(""));
            for cell in r["matrix"]["cells"][i].as_array().into_iter().flatten() {
                if cell.is_null() {
                    print!("{:>14}", ".");
                } else {
                    print!(
                        "{:>14}",
                        format!("{} n{}", pct(&cell["mean"]), cell["n"].as_u64().unwrap_or(0))
                    );
                }
            }
            println!();
        }
    }

    for (label, key) in [("our damage", "mine"), ("their damage (assumed set)", "theirs")] {
        let rows = r["damage"][key].as_array().cloned().unwrap_or_default();
        if rows.is_empty() {
            continue;
        }
        println!("\n{label}:");
        for d in rows {
            let (min, max) = (d["min"].as_i64().unwrap_or(0), d["max"].as_i64().unwrap_or(0));
            let maxhp = d["maxhp"].as_i64().unwrap_or(1).max(1);
            println!(
                "  {:<16} {:>3}-{:<3} ({:>3}-{:>3}% of {})  crit {:<4} -> {}{}",
                d["move"].as_str().unwrap_or(""),
                min,
                max,
                100 * min / maxhp,
                100 * max / maxhp,
                maxhp,
                d["crit"].as_i64().unwrap_or(0),
                ko_text(&d),
                if d["revealed"].as_bool() == Some(true) { "" } else { "  (assumed)" },
            );
        }
    }

    if let Some(steps) = r["line"]["steps"].as_array() {
        println!("\nsearched line:");
        for (i, s) in steps.iter().enumerate() {
            println!(
                "  {}. us: {:<20} them: {:<20} ({} playouts)",
                i + 1,
                s["mine"].as_str().unwrap_or("-"),
                s["theirs"].as_str().unwrap_or("-"),
                s["visits"].as_u64().unwrap_or(0)
            );
            for l in s["log"].as_array().into_iter().flatten() {
                let l = l.as_str().unwrap_or("");
                if l.starts_with("|move|") || l.starts_with("|switch|") || l.starts_with("|-") {
                    println!("       {l}");
                }
            }
        }
    }
}

fn ko_text(d: &Value) -> String {
    let guaranteed = d["hitsGuaranteed"].as_i64();
    match (d["ko"].as_str().unwrap_or("never"), guaranteed) {
        ("always", _) => "KO".to_string(),
        ("possible", Some(n)) => format!("KO on a high roll ({n}HKO guaranteed)"),
        ("possible", None) => "KO on a high roll".to_string(),
        (_, Some(n)) => format!("{n}HKO"),
        (_, None) => "no damage".to_string(),
    }
}

fn short(input: &str) -> String {
    input.replace("move ", "").replace("switch ", "sw").chars().take(13).collect()
}
