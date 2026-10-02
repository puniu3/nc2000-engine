use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{SelRule, SkuctSearch};
use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::state::Battle;
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

struct CountAlloc;
static ENABLED: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

fn record(bytes: usize) {
    if ENABLED.load(Relaxed) {
        CALLS.fetch_add(1, Relaxed);
        BYTES.fetch_add(bytes as u64, Relaxed);
    }
}

unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, old: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(p, old, size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
}

#[cfg(not(feature = "mi"))]
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn measure(name: &str, n: u32, mut f: impl FnMut()) -> Value {
    CALLS.store(0, Relaxed);
    BYTES.store(0, Relaxed);
    ENABLED.store(true, Relaxed);
    for _ in 0..n {
        f();
    }
    ENABLED.store(false, Relaxed);
    json!({"operation":name,"repetitions":n,"allocation_calls":CALLS.load(Relaxed),
           "requested_bytes":BYTES.load(Relaxed)})
}

fn main() {
    assert!(!cfg!(feature = "mi"), "run with --no-default-features");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(
        &conformance::fixture::repo_root().join("data/belief-pool-v3/belief-pool.json"),
    );
    let mut battle =
        Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[0].sets, &pool.teams[1].sets).unwrap();
    battle.set_log_enabled(false);
    let mut rows = Vec::new();
    for phase in ["preview", "battle"] {
        let choices = [0, 1].map(|side| battle.legal_choices(&dex, side).first().copied());
        let mut observer = Observer::new(&battle, 0);
        observer.observe(&battle, &dex);
        let mut belief = Belief::new(&dex, &pool, &observer);
        belief.sync(&dex, &observer);
        let cfg = RmConfig {
            c: 0.4,
            rule: SelRule::Ucb,
            ..Default::default()
        };
        let mut blind = BlindSearch::new(&battle, &dex, cfg.clone(), 0, 71);
        let mut skuct = SkuctSearch::new(&battle, &dex, cfg, 71);
        blind.step(&dex, &belief, &observer, 128);
        skuct.step(&dex, 128);
        let mut legal = battle.clone();
        let mut reusable = battle.clone();
        let measurements = [
            measure("clone", 128, || {
                black_box(battle.clone());
            }),
            measure("clone_from", 128, || {
                reusable.clone_from(&battle);
                black_box(&reusable);
            }),
            measure("legal_choices", 128, || {
                black_box(legal.legal_choices(&dex, 0));
            }),
            measure("clone_and_step", 128, || {
                let mut sim = battle.clone();
                sim.apply_choices(&dex, choices).unwrap();
                black_box(sim);
            }),
            measure("blind_iteration", 1000, || {
                blind.step(&dex, &belief, &observer, 1);
            }),
            measure("skuct_iteration", 1000, || {
                skuct.step(&dex, 1);
            }),
        ];
        rows.push(json!({"phase":phase,"measurements":measurements}));
        battle.apply_choices(&dex, choices).unwrap();
    }
    println!(
        "{}",
        json!({"allocator":"System with counters; timing is not measured","rows":rows})
    );
}
