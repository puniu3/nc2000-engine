//! Sampling profile of the search-mode hot path (log-off replay + random
//! playouts). Writes target/flamegraph.svg. Run:
//!   cargo run --release -p conformance --example profile

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_engine::state::Battle;

struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn main() {
    let dex = load_dex();
    let root = repo_root().join("fixtures/corpus-v1");
    let mut fixtures = Vec::new();
    for corpus in ["puredata", "full"] {
        for path in corpus_files(&root.join(corpus)) {
            fixtures.push(Fixture::load(&path).unwrap());
        }
    }

    let guard = pprof::ProfilerGuardBuilder::default().frequency(997).build().unwrap();

    // log-off replay
    for _ in 0..10 {
        for fx in &fixtures {
            let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            b.set_log_enabled(false);
            for line in &fx.choices {
                let side_n = if line.side == "p1" { 0 } else { 1 };
                b.choose(&dex, side_n, &line.choice).unwrap();
            }
        }
    }
    // random playouts
    let mut rng = TestRng(0xBADC_0DE);
    for (fi, fx) in fixtures.iter().enumerate() {
        for p in 0..15u64 {
            let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            b.set_log_enabled(false);
            b.reseed(0xFEED ^ ((fi as u64) << 20) ^ p);
            while b.outcome().is_none() {
                let picks = [0usize, 1].map(|side_n| {
                    let legal = b.legal_choices(&dex, side_n);
                    if legal.is_empty() {
                        None
                    } else {
                        Some(legal[(rng.next() % legal.len() as u64) as usize])
                    }
                });
                b.apply_choices(&dex, picks).unwrap();
            }
        }
    }

    let report = guard.report().build().unwrap();
    let path = repo_root().join("target/flamegraph.svg");
    let f = std::fs::File::create(&path).unwrap();
    report.flamegraph(f).unwrap();
    println!("wrote {}", path.display());

    // Also dump a top-N self-time table to stdout.
    let mut self_ms: std::collections::HashMap<String, isize> = Default::default();
    for (frames, n) in report.data.iter() {
        if let Some(top) = frames.frames.first().and_then(|f| f.first()) {
            *self_ms.entry(top.name()).or_default() += *n;
        }
    }
    let mut v: Vec<_> = self_ms.into_iter().collect();
    v.sort_by_key(|(_, n)| -*n);
    let total: isize = v.iter().map(|(_, n)| n).sum();
    println!("total samples: {total}");
    for (name, n) in v.iter().take(30) {
        println!("{:6.2}%  {}", *n as f64 / total as f64 * 100.0, name);
    }
}
