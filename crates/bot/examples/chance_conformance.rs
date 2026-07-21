//! M17e step 1 — chance-node enumeration conformance.
//!
//! Verifies `BattleRng`'s Oracle mode (exhaustive chance enumeration,
//! `battle::enumerate::enumerate_step`) against the seeded engine it will
//! stand in for:
//!
//!   1. every recorded draw's outcome-class counts partition the u32 space
//!      exactly (hard assert inside `BattleRng::pick`);
//!   2. per enumerated step, leaf probabilities sum to 1;
//!   3. seeded replays of the same step land bit-exactly (by `state_key`)
//!      on an enumerated leaf — completeness — with empirical frequencies
//!      matching the enumerated probabilities — correct weighting.
//!
//! Positions come from meta-pool games driven by uniformly-random legal
//! choices, so team preview, switches, forced switches, and end-of-battle
//! steps are all crossed. Deterministic: one fixed meta-RNG seed drives
//! pairings, choices, and replication seeds.
//!
//! Usage: chance_conformance [games] [cap] [reps]
//!   games — meta-pool pairings to play (default 15)
//!   cap   — max leaves per enumerated step before skipping (default 20000)
//!   reps  — seeded replications per enumerated step (default 300)

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use conformance::load_dex;
use nc2000_bot::preview::load_meta_pool;
use nc2000_engine::battle::enumerate::enumerate_step;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::Prng;
use nc2000_engine::state::Battle;

const MAX_TURNS: u16 = 60;
const QUOTA_PER_GAME: usize = 12;

fn pick_choices(b: &mut Battle, dex: &Dex, mrng: &mut Prng) -> Option<[Option<SearchChoice>; 2]> {
    let needs = b.needs_choice();
    if !needs[0] && !needs[1] {
        return None;
    }
    let mut choices = [None, None];
    for side in 0..2 {
        if needs[side] {
            let ls = b.legal_choices(dex, side);
            if ls.is_empty() {
                return None;
            }
            choices[side] = Some(ls[mrng.sample_index(ls.len())]);
        }
    }
    Some(choices)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(15);
    let cap: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let reps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(300);

    let dex = load_dex();
    let pool = load_meta_pool(Path::new("data/meta-pool-v0/meta-pool.json"));
    println!("pool: {} teams; {games} games, cap {cap}, reps {reps}", pool.teams.len());

    let mut mrng = Prng::new(0x0123_4567_89ab_cdef);
    let t0 = Instant::now();

    let mut steps = 0usize;
    let mut skipped_cap = 0usize;
    let mut total_leaves = 0usize;
    let mut max_leaves = 0usize;
    let mut max_draws = 0usize;
    let mut worst_prob_sum_err = 0.0f64;
    let mut worst_z = 0.0f64;
    let mut z_checked = 0usize;
    let mut unmatched = 0usize;
    let mut replications = 0usize;

    for game in 0..games {
        let n = pool.teams.len();
        let (i, j) = loop {
            let (i, j) = (mrng.sample_index(n), mrng.sample_index(n));
            if i != j {
                break (i, j);
            }
        };
        let mut b = match Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[i].sets, &pool.teams[j].sets)
        {
            Ok(b) => b,
            Err(e) => {
                println!("game {game}: pair ({i},{j}) failed to start: {e:?}");
                continue;
            }
        };
        b.set_log_enabled(false);
        b.reseed(((mrng.next_u32() as u64) << 32) | mrng.next_u32() as u64);

        let mut enumerated = 0usize;
        while !b.ended && b.turn < MAX_TURNS {
            let choices = match pick_choices(&mut b, &dex, &mut mrng) {
                Some(c) => c,
                None => break,
            };

            if enumerated < QUOTA_PER_GAME {
                enumerated += 1;
                match enumerate_step(&dex, &b, choices, cap) {
                    None => skipped_cap += 1,
                    Some(leaves) => {
                        steps += 1;
                        total_leaves += leaves.len();
                        max_leaves = max_leaves.max(leaves.len());
                        max_draws = max_draws.max(leaves.iter().map(|l| l.draws).max().unwrap_or(0));

                        let sum: f64 = leaves.iter().map(|l| l.prob).sum();
                        worst_prob_sum_err = worst_prob_sum_err.max((sum - 1.0).abs());

                        let mut merged: HashMap<u64, f64> = HashMap::new();
                        for l in &leaves {
                            *merged.entry(l.battle.state_key()).or_default() += l.prob;
                        }

                        let mut obs: HashMap<u64, usize> = HashMap::new();
                        for _ in 0..reps {
                            let mut c = b.clone();
                            c.reseed(((mrng.next_u32() as u64) << 32) | mrng.next_u32() as u64);
                            c.apply_choices(&dex, choices).expect("legal choices");
                            let k = c.state_key();
                            replications += 1;
                            if merged.contains_key(&k) {
                                *obs.entry(k).or_default() += 1;
                            } else {
                                unmatched += 1;
                                if unmatched <= 5 {
                                    println!(
                                        "UNMATCHED: game {game} turn {} choices {choices:?} key {k:x}",
                                        b.turn
                                    );
                                }
                            }
                        }
                        for (k, p) in &merged {
                            let exp = p * reps as f64;
                            if exp >= 5.0 && *p < 1.0 {
                                let got = *obs.get(k).unwrap_or(&0) as f64;
                                let z = (got - exp) / (exp * (1.0 - p)).sqrt();
                                z_checked += 1;
                                worst_z = worst_z.max(z.abs());
                            }
                        }
                    }
                }
            }

            b.apply_choices(&dex, choices).expect("legal choices");
        }
    }

    println!(
        "\nsteps {steps} (skipped-by-cap {skipped_cap}); leaves total {total_leaves}, max {max_leaves}, max draws/path {max_draws}"
    );
    println!(
        "prob-sum worst |err| {worst_prob_sum_err:.3e}; replications {replications}, unmatched {unmatched}"
    );
    println!("z-checked cells {z_checked}, worst |z| {worst_z:.2}");
    println!("wall {:.1}s", t0.elapsed().as_secs_f64());

    let mut fail = false;
    if unmatched > 0 {
        println!("FAIL: {unmatched} seeded replications produced a state outside the enumeration");
        fail = true;
    }
    if worst_prob_sum_err > 1e-9 {
        println!("FAIL: leaf probabilities do not sum to 1 (worst err {worst_prob_sum_err:.3e})");
        fail = true;
    }
    // ~1.5k cells => expected max |z| ≈ 3.5 under the null; 5 is a real bug.
    if worst_z > 5.0 {
        println!("FAIL: empirical frequency deviates from enumerated probability (|z| {worst_z:.2})");
        fail = true;
    }
    if fail {
        std::process::exit(1);
    }
    println!("PASS");
}
