//! Where do the remaining allocs/bytes per clone come from?
//!   cargo run --release -p conformance --example clone_anatomy

use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static A: Counting = Counting;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::load_dex;
use nc2000_engine::state::Battle;

fn measure<T: Clone>(label: &str, v: &T) {
    let n = 1000u64;
    let a0 = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    let t = std::time::Instant::now();
    for _ in 0..n {
        black_box(v.clone());
    }
    let dt = t.elapsed().as_secs_f64();
    let a1 = (ALLOCS.load(Ordering::Relaxed), BYTES.load(Ordering::Relaxed));
    println!(
        "{label:22} {:6.0} ns  {:5.1} allocs  {:7.0} bytes",
        dt / n as f64 * 1e9,
        (a1.0 - a0.0) as f64 / n as f64,
        (a1.1 - a0.1) as f64 / n as f64
    );
}

fn main() {
    let dex = load_dex();
    let root = repo_root().join("fixtures/corpus-v1");
    let path = &corpus_files(&root.join("full"))[0];
    let fx = Fixture::load(path).unwrap();
    let mut b = Battle::from_fixture(&dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
    b.set_log_enabled(false);
    let half = fx.choices.len() / 2;
    for line in &fx.choices[..half] {
        let side_n = if line.side == "p1" { 0 } else { 1 };
        b.choose(&dex, side_n, &line.choice).unwrap();
    }

    println!("sizeof(Battle) = {}", std::mem::size_of::<Battle>());
    println!("sizeof(Pokemon) = {}", std::mem::size_of::<nc2000_engine::state::Pokemon>());
    println!("sizeof(EffectState) = {}", std::mem::size_of::<nc2000_engine::state::EffectState>());
    measure("battle", &b);
    measure("side0", &b.sides[0]);
    measure("side1", &b.sides[1]);
    measure("side0.roster", &b.sides[0].roster);
    for (i, p) in b.sides[0].roster.iter().enumerate() {
        measure(&format!("  poke0.{i} (v={})", p.volatiles.len()), p);
    }
    measure("queue", &b.queue);
    measure("log", &b.log);
    measure("field", &b.field);
    measure("format_data", &b.format_data);
    measure("faint_queue", &b.faint_queue);
    measure("winner", &b.winner);
}
