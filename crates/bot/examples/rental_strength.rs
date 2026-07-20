//! Community-rentals strength test (throwaway harness): score the 34 meta-pool
//! teams AND the legality-clean community rental teams through the IDENTICAL
//! gauntlet (`teamgen::gauntlet_eval`) vs the T1 tournament field (pool 0-7),
//! skuct at a configurable budget. Apples-to-apples: every candidate faces the
//! SAME battle/agent seeds per (opponent, game), so an exact species+set twin
//! of a pool team reproduces its score bit-for-bit (harness sanity check).
//!
//!   cargo run --release -p nc2000-bot --example rental_strength -- \
//!       [--games N] [--iters N] [--seed S] [--threads T] [--max-turns M] \
//!       [--only LABEL,LABEL,...] [--out FILE.jsonl]
//!
//! Labels: pool-NN (pool index) / comm-CC (community cban). Illegal community
//! teams (canonicalize+validate fails: cban 16 item-clause, 26 Little Cup,
//! 28 non-team) are skipped automatically.

use std::io::Write;
use std::path::PathBuf;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::teamgen::{gauntlet_eval, to_sets, EvalCfg, TeamGen};
use nc2000_engine::battle::PokemonSet;
use serde_json::{json, Value};

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}
fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    flag(args, name).map(|v| v.parse().expect(name)).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let games: u32 = num(&args, "--games", 96);
    let iters: u32 = num(&args, "--iters", 1000);
    let seed: u64 = num(&args, "--seed", 1);
    let threads: usize = num(&args, "--threads", 6);
    let max_turns: u16 = num(&args, "--max-turns", 500);
    let only: Option<Vec<String>> =
        flag(&args, "--only").map(|s| s.split(',').map(|x| x.trim().to_string()).collect());
    let out_path = flag(&args, "--out").map(PathBuf::from);

    let dex = load_dex();
    let root = repo_root();
    let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
    let pool_text = std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let gen = TeamGen::new(&dex, &ls_text, &pool_text).unwrap();

    // ---- gauntlet: pool teams 0..=7 (the T1 tournament tier)
    let gauntlet: Vec<Vec<PokemonSet>> = (0..=7)
        .map(|i| to_sets(&gen.canonize(&dex, &gen.team_json(i)).unwrap()).unwrap())
        .collect();

    // ---- candidate list: all pool teams + legal community teams
    let mut candidates: Vec<(String, Vec<PokemonSet>)> = Vec::new();
    for i in 0..gen.teams().len() {
        let c = gen.canonize(&dex, &gen.team_json(i)).unwrap();
        candidates.push((format!("pool-{i:02}"), to_sets(&c).unwrap()));
    }
    let comm: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/community-rentals-v0/teams.json")).unwrap(),
    )
    .unwrap();
    let mut skipped: Vec<i64> = Vec::new();
    for t in comm["teams"].as_array().unwrap() {
        let cban = t["cban"].as_i64().unwrap();
        let raw: Vec<Value> = t["sets"].as_array().unwrap().clone();
        match gen.canonize(&dex, &raw) {
            Some(c) => candidates.push((format!("comm-{cban:02}"), to_sets(&c).unwrap())),
            None => skipped.push(cban),
        }
    }
    eprintln!(
        "candidates: {} pool + {} legal community (skipped illegal cban {:?})",
        gen.teams().len(),
        candidates.len() - gen.teams().len(),
        skipped
    );

    if let Some(only) = &only {
        candidates.retain(|(l, _)| only.contains(l));
        eprintln!("restricted to {}: {:?}", candidates.len(), only);
    }

    eprintln!(
        "eval: skuct:{iters}, {games} games/opp x 8 opp = {} games/team, seed {seed}, {threads} threads",
        (games + games % 2) * 8
    );

    let mut out_file = out_path.map(|p| std::fs::File::create(p).unwrap());
    println!(
        "{:<9} {:>7} {:>7}  {:>6}  per-opponent (vs pool 0-7)",
        "label", "score", "ci95", "games"
    );
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    for (label, sets) in &candidates {
        let t0 = std::time::Instant::now();
        let cfg = EvalCfg { games_per_opponent: games, agent_iters: iters, max_turns, threads, seed };
        let res = gauntlet_eval(&dex, sets, &gauntlet, &cfg);
        // Wald binomial 95% CI on the mean score (ties count as 0.5).
        let n = res.games as f64;
        let ci = 1.96 * (res.score * (1.0 - res.score) / n).sqrt();
        let per: Vec<String> = res.per_opponent.iter().map(|v| format!("{v:.2}")).collect();
        println!(
            "{:<9} {:>7.3} {:>7.3}  {:>6}  [{}]  ({:.0}s)",
            label,
            res.score,
            ci,
            res.games,
            per.join(" "),
            t0.elapsed().as_secs_f64()
        );
        rows.push((label.clone(), res.score, ci));
        if let Some(f) = out_file.as_mut() {
            writeln!(
                f,
                "{}",
                json!({
                    "label": label, "score": res.score, "ci95": ci, "games": res.games,
                    "per_opponent": res.per_opponent, "iters": iters, "eval_games": games,
                    "seed": seed,
                })
            )
            .unwrap();
            f.flush().unwrap();
        }
    }

    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\n=== ranking (desc) ===");
    for (i, (l, s, ci)) in rows.iter().enumerate() {
        println!("{:>2}. {:<9} {:.3} +/- {:.3}", i + 1, l, s, ci);
    }
}
