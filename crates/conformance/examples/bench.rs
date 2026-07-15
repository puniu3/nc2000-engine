//! M3 throughput benchmark. Run with:
//!   cargo run --release -p conformance --example bench
//!
//! Reference baseline (README, same machine): PS on Node does 6.5 battles/s,
//! 570 turns/s, 5.5 ms per clone.

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

// Allocator selection: default = counting wrapper around System (reports
// allocs/turn and allocs/clone). Build with `--features conformance/mimalloc`
// ... not wired; flip this flag manually to gauge allocator sensitivity.
const _: () = ();
#[cfg(not(feature = "mi"))]
#[global_allocator]
static A: Counting = Counting;
#[cfg(feature = "mi")]
#[global_allocator]
static A: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn alloc_snapshot() -> (u64, u64) {
    (ALLOCS.load(Ordering::Relaxed), ALLOC_BYTES.load(Ordering::Relaxed))
}

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
    println!("{} fixtures loaded", fixtures.len());

    // 1. fixture replay, log on (the conformance configuration)
    let reps = 20;
    let mut turns = 0u64;
    let t = Instant::now();
    for _ in 0..reps {
        for fx in &fixtures {
            let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            for line in &fx.choices {
                let side_n = if line.side == "p1" { 0 } else { 1 };
                b.choose(&dex, side_n, &line.choice).unwrap();
            }
            turns += b.turn as u64;
        }
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "replay (log on):   {:>9.0} turns/s   {:>7.0} battles/s   ({turns} turns, {:.2}s)",
        turns as f64 / dt,
        (reps * fixtures.len()) as f64 / dt,
        dt
    );

    // 2. fixture replay, log off (search stepping configuration)
    let mut turns = 0u64;
    let a0 = alloc_snapshot();
    let t = Instant::now();
    for _ in 0..reps {
        for fx in &fixtures {
            let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            b.set_log_enabled(false);
            for line in &fx.choices {
                let side_n = if line.side == "p1" { 0 } else { 1 };
                b.choose(&dex, side_n, &line.choice).unwrap();
            }
            turns += b.turn as u64;
        }
    }
    let dt = t.elapsed().as_secs_f64();
    let a1 = alloc_snapshot();
    println!(
        "replay (log off):  {:>9.0} turns/s   {:>7.0} battles/s   ({turns} turns, {:.2}s)",
        turns as f64 / dt,
        (reps * fixtures.len()) as f64 / dt,
        dt
    );
    println!(
        "  allocs/turn: {:.0}   bytes/turn: {:.0}",
        (a1.0 - a0.0) as f64 / turns as f64,
        (a1.1 - a0.1) as f64 / turns as f64
    );

    // 3. random playouts via the search API (enumerate + pick + apply)
    let playouts_per_fixture = 30u64;
    let mut rng = TestRng(0xBADC_0DE);
    let mut turns = 0u64;
    let mut battles = 0u64;
    let mut decisions = 0u64;
    let t = Instant::now();
    for (fi, fx) in fixtures.iter().enumerate() {
        for p in 0..playouts_per_fixture {
            let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            b.set_log_enabled(false);
            b.reseed(0xFEED ^ ((fi as u64) << 20) ^ p);
            while b.outcome().is_none() {
                let picks = [0usize, 1].map(|side_n| {
                    let legal = b.legal_choices(&dex, side_n);
                    if legal.is_empty() {
                        None
                    } else {
                        decisions += 1;
                        Some(legal[(rng.next() % legal.len() as u64) as usize])
                    }
                });
                b.apply_choices(&dex, picks).unwrap();
            }
            turns += b.turn as u64;
            battles += 1;
        }
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "random playouts:   {:>9.0} turns/s   {:>7.0} battles/s   ({battles} battles, {turns} turns, {decisions} decisions, {:.2}s)",
        turns as f64 / dt,
        battles as f64 / dt,
        dt
    );

    // 4. clone cost on a mid-battle state (log off → log stays small)
    let fx = &fixtures[0];
    let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
    b.set_log_enabled(false);
    let half = fx.choices.len() / 2;
    for line in &fx.choices[..half] {
        let side_n = if line.side == "p1" { 0 } else { 1 };
        b.choose(&dex, side_n, &line.choice).unwrap();
    }
    let n = 1_000_000u64;
    let a0 = alloc_snapshot();
    let t = Instant::now();
    for _ in 0..n {
        black_box(b.clone());
    }
    let dt = t.elapsed().as_secs_f64();
    let a1 = alloc_snapshot();
    println!("clone (mid-battle, log off): {:.0} ns/clone ({n} clones, {:.2}s)", dt / n as f64 * 1e9, dt);
    println!(
        "  allocs/clone: {:.0}   bytes/clone: {:.0}",
        (a1.0 - a0.0) as f64 / n as f64,
        (a1.1 - a0.1) as f64 / n as f64
    );

    // 5. clone cost with the protocol log still attached (for reference)
    let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
    for line in &fx.choices[..half] {
        let side_n = if line.side == "p1" { 0 } else { 1 };
        b.choose(&dex, side_n, &line.choice).unwrap();
    }
    let n = 100_000u64;
    let t = Instant::now();
    for _ in 0..n {
        black_box(b.clone());
    }
    let dt = t.elapsed().as_secs_f64();
    println!(
        "clone (mid-battle, log on, {} lines): {:.0} ns/clone ({n} clones, {:.2}s)",
        b.log.len(),
        dt / n as f64 * 1e9,
        dt
    );
}
