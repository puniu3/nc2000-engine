//! Native twin of `tests-node/bench.js`: drives the exact same deterministic
//! game through the same wasm-crate API (rlib, no JS boundary) and times the
//! same searcher work, so wasm/native throughput ratios compare identical
//! iteration streams.
//!
//!   cargo run --release -p nc2000-wasm --example native_bench

use std::time::Instant;

use nc2000_wasm::{derive_battle_seed, WasmBattle, WasmDex, WasmSearcher};

const BENCH_ITERS: u32 = 1000;
const PICK_ITERS: u32 = 100;
const MAX_DECISIONS: u32 = 200;

fn main() {
    let dex = WasmDex::new().map_err(|_| "dex").unwrap();
    let path = conformance::fixture::repo_root().join("fixtures/corpus-v1/full/battle-001.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let fx: serde_json::Value = serde_json::from_str(&text).unwrap();
    let (p1, p2) = (fx["p1team"].to_string(), fx["p2team"].to_string());

    let mut battle =
        WasmBattle::new(&dex, &p1, &p2, &derive_battle_seed(5)).map_err(|_| "battle").unwrap();

    let (mut prev_ns, mut prev_iters) = (0u128, 0u64); // preview decisions
    let (mut bat_ns, mut bat_iters) = (0u128, 0u64); // in-battle decisions
    let mut decisions = 0u32;
    while battle.outcome().is_none() && decisions < MAX_DECISIONS {
        let needs: [bool; 2] = serde_json::from_str(&battle.needs_choice()).unwrap();
        let mut picks: Vec<(usize, String)> = Vec::new();
        for side in 0..2 {
            if !needs[side] {
                continue;
            }
            // timed bench searcher (thrown away)
            let mut bench =
                WasmSearcher::new(&battle, side, 100_000 + decisions * 2 + side as u32, None, None);
            let preview = battle.turn() == 0;
            let t = Instant::now();
            bench.step(BENCH_ITERS);
            let ns = t.elapsed().as_nanos();
            if preview {
                prev_ns += ns;
                prev_iters += BENCH_ITERS as u64;
            } else {
                bat_ns += ns;
                bat_iters += BENCH_ITERS as u64;
            }
            // the game-driving pick (same seeds as bench.js)
            let mut picker =
                WasmSearcher::new(&battle, side, 42 + decisions * 2 + side as u32, None, None);
            picker.step(PICK_ITERS);
            picks.push((side, picker.best().unwrap()));
        }
        for (side, input) in picks {
            battle.apply_choice(side, &input).map_err(|_| "apply").unwrap();
        }
        decisions += 1;
    }

    let rate = |iters: u64, ns: u128| iters as f64 / (ns as f64 / 1e9);
    println!("native skuct throughput (BENCH_ITERS={BENCH_ITERS}/decision):");
    println!(
        "  preview roots: {prev_iters} iters, {:.2} s, {:.0} iters/s",
        prev_ns as f64 / 1e9,
        rate(prev_iters, prev_ns)
    );
    println!(
        "  battle roots:  {bat_iters} iters, {:.2} s, {:.0} iters/s",
        bat_ns as f64 / 1e9,
        rate(bat_iters, bat_ns)
    );
    println!(
        "  game: outcome {:?} in {} turns, {} decisions",
        battle.outcome(),
        battle.turn(),
        decisions
    );
}
