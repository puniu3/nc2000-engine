//! Direct micro-benchmark comparing the 256-byte branchless `CompactBattle`
//! kernel against the full `Battle` engine across:
//! 1. State clone (`Copy` 256B vs `Battle::clone_from`)
//! 2. Search state hashing (`CompactBattle::search_hash` vs `Battle::search_key`)
//! 3. 16-roll SIMD damage + Substitute pool evaluation (`evaluate_hit_simd16`)
//! 4. Full turn stepping (`CompactBattle::step_turn` vs `Battle::apply_choices`)
//!
//! Run with:
//!   cargo run --release -p nc2000-engine --example compact_bench

use std::hint::black_box;
use std::time::Instant;

use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::compact::{
    evaluate_hit_simd16, get_damage_base_branchless, CompactBattle, CompactWeather,
};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn mk(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": species, "species": species, "item": item, "ability": "No Ability",
        "moves": moves, "level": level,
        "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
        "happiness": 255
    }))
    .unwrap()
}

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/gen2stadium2.json");
    let json = std::fs::read_to_string(&path).unwrap();
    let dex = Dex::from_json(&json).unwrap();

    let p1 = vec![
        mk("Snorlax", 55, "Leftovers", &["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
        mk("Zapdos", 50, "Mint Berry", &["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
        mk("Cloyster", 50, "Leftovers", &["Surf", "Ice Beam", "Spikes", "Explosion"]),
        mk("Exeggutor", 50, "Miracle Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
        mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Destiny Bond"]),
        mk("Machamp", 50, "Scope Lens", &["Cross Chop", "Rock Slide", "Earthquake", "Curse"]),
    ];
    let p2 = vec![
        mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
        mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Curse", "Rest"]),
        mk("Raikou", 50, "Miracle Berry", &["Thunderbolt", "Crunch", "Reflect", "Rest"]),
        mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
        mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Roar"]),
        mk("Blissey", 50, "Leftovers", &["Ice Beam", "Thunderbolt", "Soft-Boiled", "Sing"]),
    ];

    let mut base_battle = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
    base_battle.set_log_enabled(false);
    base_battle
        .apply_choices(
            &dex,
            [
                Some(SearchChoice::Team([1, 2, 3])),
                Some(SearchChoice::Team([1, 2, 3])),
            ],
        )
        .unwrap();

    let (table, base_compact) = CompactBattle::from_battle(&base_battle, &dex);

    let iters = 200_000u64;

    // 1. Clone benchmark
    let mut battle_scratch = base_battle.clone();
    let t0 = Instant::now();
    for _ in 0..iters {
        battle_scratch.clone_from(black_box(&base_battle));
        black_box(&battle_scratch);
    }
    let battle_clone_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let t0 = Instant::now();
    for _ in 0..iters {
        let c = black_box(base_compact);
        black_box(c);
    }
    let compact_clone_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // 2. Hash benchmark
    let t0 = Instant::now();
    let mut h_acc = 0u64;
    for _ in 0..iters {
        h_acc ^= black_box(&base_battle).search_key(Some(32), true);
    }
    black_box(h_acc);
    let battle_hash_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    let t0 = Instant::now();
    let mut h_acc2 = 0u64;
    for _ in 0..iters {
        h_acc2 ^= black_box(&base_compact).search_hash();
    }
    black_box(h_acc2);
    let compact_hash_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // 3. 16-roll SIMD damage + Substitute pool evaluation
    let t0 = Instant::now();
    let mut ko_acc = 0u32;
    for i in 0..iters {
        let base_dmg = get_damage_base_branchless(
            55,
            120,
            black_box(198 + (i as i32 & 3)),
            165,
            true,
            false,
            false,
            true,
            CompactWeather::None,
            false,
            false,
            false,
            0,
            1 << 0,
            0,
            false,
        );
        let hit = evaluate_hit_simd16(base_dmg, false, 290, 65, false);
        ko_acc ^= (hit.ko_mask as u32) ^ (hit.sub_broke_mask as u32);
    }
    black_box(ko_acc);
    let simd16_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    // 4. Full turn step (clone + step + hash)
    let step_iters = 100_000u64;
    let p1_move = base_battle.legal_choices(&dex, 0)[0];
    let p2_move = base_battle.legal_choices(&dex, 1)[0];
    let t0 = Instant::now();
    let mut step_acc = 0u64;
    for i in 0..step_iters {
        battle_scratch.clone_from(black_box(&base_battle));
        battle_scratch.reseed(i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        battle_scratch
            .apply_choices(&dex, [Some(p1_move), Some(p2_move)])
            .unwrap();
        step_acc ^= battle_scratch.search_key(Some(32), true);
    }
    black_box(step_acc);
    let battle_step_ns = t0.elapsed().as_nanos() as f64 / step_iters as f64;

    let t0 = Instant::now();
    let mut c_step_acc = 0u64;
    for i in 0..step_iters {
        let mut c = black_box(base_compact);
        c.prng_seed = i.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        c.step_turn(&table, [0, 0]);
        c_step_acc ^= c.search_hash();
    }
    black_box(c_step_acc);
    let compact_step_ns = t0.elapsed().as_nanos() as f64 / step_iters as f64;

    println!(
        "=== CompactBattle ({}B, align(64)) vs Battle ({}B) ===",
        std::mem::size_of::<CompactBattle>(),
        std::mem::size_of::<Battle>()
    );
    println!(
        "1. State Clone : Battle = {:7.1} ns | CompactBattle = {:5.2} ns ({:6.1}x faster)",
        battle_clone_ns,
        compact_clone_ns,
        battle_clone_ns / compact_clone_ns.max(0.01)
    );
    println!(
        "2. Search Hash : Battle = {:7.1} ns | CompactBattle = {:5.2} ns ({:6.1}x faster)",
        battle_hash_ns,
        compact_hash_ns,
        battle_hash_ns / compact_hash_ns.max(0.01)
    );
    println!(
        "3. 16-Roll SIMD: base + 16 rolls + [hp, sub_hp] masks = {:5.2} ns ({:5.2} ns / roll)",
        simd16_ns,
        simd16_ns / 16.0
    );
    println!(
        "4. Clone+Step+Hash: Battle = {:7.1} ns | CompactBattle = {:5.2} ns ({:6.1}x faster)",
        battle_step_ns,
        compact_step_ns,
        battle_step_ns / compact_step_ns.max(0.01)
    );
}
