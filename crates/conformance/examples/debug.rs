//! Debug one fixture: replay choice by choice, printing our log lines and
//! PRNG seed after each input, plus the expected snapshot seed/log.
//! Usage: cargo run -p conformance --example debug -- <fixture.json> [from_snap]

use conformance::fixture::Fixture;
use conformance::load_dex;
use nc2000_engine::state::Battle;

fn main() {
    let path = std::env::args().nth(1).expect("fixture path");
    let from: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let dex = load_dex();
    let fx = Fixture::load(std::path::Path::new(&path)).unwrap();
    let mut battle = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
    let mut snap_idx = 1; // snapshot 0 is construction
    let mut log_pos = battle.log.len();
    for line in &fx.choices {
        let side_n = if line.side == "p1" { 0 } else { 1 };
        let before = battle.log.len();
        if let Err(e) = battle.choose(&dex, side_n, &line.choice) {
            println!("!! choice error {:?}: {e:?}", line.choice);
            return;
        }
        if battle.log.len() > before {
            let snap = &fx.snapshots[snap_idx];
            let ours = battle.prng.seed_str();
            let theirs = &snap.prng_seed;
            if snap_idx >= from {
                println!("--- snap {snap_idx} (turn {}) choice p{}:{:?}", snap.turn, side_n + 1, line.choice);
                let actual: Vec<&str> = battle.log[log_pos..].iter().map(|s| s.as_str()).collect();
                let max = actual.len().max(snap.log.len());
                for i in 0..max {
                    let a = actual.get(i).copied().unwrap_or("<none>");
                    let e = snap.log.get(i).map(|s| s.as_str()).unwrap_or("<none>");
                    let mark = if a == e { " " } else { "!" };
                    println!("{mark} ours: {a}");
                    if a != e {
                        println!("{mark} want: {e}");
                    }
                }
                println!("  seed ours={ours} want={theirs} {}", if &ours == theirs { "OK" } else { "MISMATCH" });
                if &ours != theirs {
                    return;
                }
            }
            log_pos = battle.log.len();
            snap_idx += 1;
        }
    }
    println!("replay finished, {} snapshots", snap_idx);
}
