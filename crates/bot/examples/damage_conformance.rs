//! Differential harness: the bot eval's damage estimate vs the ENGINE's real
//! damage core, over real meta-pool positions.
//!
//! The eval (`eval::expected_hit_fraction`) maintains its own copy of the gen-2
//! damage formula. The engine's copy (`Battle::get_damage`, reachable read-only
//! via `get_damage_synthetic`) is the one bit-exactly conformed to PS. This
//! example measures where the two disagree, and how much the engine core costs
//! relative to the eval's copy — i.e. whether the eval could simply call it.
//!
//! Comparison is on MEAN damage with accuracy divided out and crits suppressed
//! on both sides, so a ratio != 1.0 is a genuine model divergence, not variance.
//!
//! NOTE: `get_damage` does not run `onModifyMove`, so Hidden Power's runtime
//! type/category must be planted by hand (as `replay_analysis.rs` does). That
//! planting is itself part of the answer: the engine core is not a drop-in
//! oracle — the move-modification callbacks live outside it.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use conformance::load_dex;
use nc2000_bot::eval;
use nc2000_bot::preview::load_meta_pool;
use nc2000_engine::battle::moveexec::{get_active_move, get_damage_synthetic};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::{Category, Dex, Multihit};
use nc2000_engine::state::{Battle, PokeId};

/// First 3-subset (1-indexed) whose level total is within the format cap.
fn legal_pick(sets: &[PokemonSet]) -> Option<String> {
    let cap = nc2000_engine::battle::MAX_TOTAL_LEVEL;
    let n = sets.len();
    for i in 0..n {
        for j in i + 1..n {
            for k in j + 1..n {
                let tot = sets[i].level as u32 + sets[j].level as u32 + sets[k].level as u32;
                if tot <= cap as u32 {
                    return Some(format!("team {},{},{}", i + 1, j + 1, k + 1));
                }
            }
        }
    }
    None
}

fn start(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Option<Battle> {
    let (c1, c2) = (legal_pick(p1)?, legal_pick(p2)?);
    let mut b = Battle::from_fixture(dex, "1,2,3,4", p1, p2).ok()?;
    b.set_log_enabled(false);
    b.choose(dex, 0, &c1).ok()?;
    b.choose(dex, 1, &c2).ok()?;
    Some(b)
}

/// Engine truth: mean damage of one use, no crit, no roll variance, with the
/// eval's multi-hit convention applied so the two are comparable.
fn engine_mean(b: &mut Battle, dex: &Dex, att: PokeId, def: PokeId, mv: nc2000_engine::dex::MoveId) -> Option<f64> {
    let ms = dex.move_static(mv);
    let mut fake = get_active_move(dex, mv);
    fake.no_damage_variance = true;
    fake.will_crit = Some(false);
    // onModifyMove is NOT run by get_damage — plant Hidden Power's real type
    // and gen-2 physical/special split from the attacker's rolled DVs.
    if dex.moves.key(mv) == "hiddenpower" {
        let a = b.poke(att);
        let (t, p) = (a.hp_type, a.hp_power);
        fake.move_type = t;
        fake.base_move_type = t;
        fake.base_power = p;
        let special = matches!(
            dex.type_name(t),
            "Fire" | "Water" | "Grass" | "Electric" | "Psychic" | "Ice" | "Dragon" | "Dark"
        );
        fake.category = if special { Category::Special } else { Category::Physical };
    }
    let top = get_damage_synthetic(b, dex, att, def, fake)?;
    if top <= 0.0 {
        return Some(0.0);
    }
    let hits = match &ms.multihit {
        Some(Multihit::Fixed(n)) => *n as f64,
        Some(Multihit::Range(2, 5)) => 3.0,
        Some(Multihit::Range(lo, hi)) => (*lo + *hi) as f64 / 2.0,
        None => 1.0,
    };
    Some(top * (236.0 / 255.0) * hits)
}

fn main() {
    let dex = load_dex();
    let pool = load_meta_pool(Path::new("data/meta-pool-v0/meta-pool.json"));
    println!("pool: {} teams", pool.teams.len());

    // Per-move divergence stats: (n, sum_ratio, worst_ratio, example)
    let mut stats: BTreeMap<String, (usize, f64, f64, String)> = BTreeMap::new();
    let mut eval_ns = 0u128;
    let mut engine_ns = 0u128;
    let mut eval_calls = 0u64;
    let mut engine_calls = 0u64;
    let mut compared = 0u64;
    let mut zero_both = 0u64;

    let n_teams = pool.teams.len();
    for i in 0..n_teams {
        for j in 0..n_teams {
            if i == j {
                continue;
            }
            let mut b = match start(&dex, &pool.teams[i].sets, &pool.teams[j].sets) {
                Some(b) => b,
                None => continue,
            };
            let (att, def) = match (b.active_id(0), b.active_id(1)) {
                (Some(a), Some(d)) => (a, d),
                _ => continue,
            };
            let slots: Vec<_> = b.poke(att).move_slots.iter().map(|m| m.id).collect();
            for mv in slots {
                let ms = dex.move_static(mv);
                if ms.category == Category::Status || ms.damage.is_some() {
                    continue; // status + fixed-damage: eval scores 0 by design
                }

                let t0 = Instant::now();
                let ehf = eval::expected_hit_fraction(&b, &dex, att, def, mv, true);
                eval_ns += t0.elapsed().as_nanos();
                eval_calls += 1;

                let t1 = Instant::now();
                let em = engine_mean(&mut b, &dex, att, def, mv);
                engine_ns += t1.elapsed().as_nanos();
                engine_calls += 1;

                let em = match em {
                    Some(v) => v,
                    None => continue,
                };
                let acc = b.hit_probability(&dex, att, def, mv);
                let hp = b.poke(def).hp.max(1) as f64;
                let eval_dmg = if acc > 0.0 { ehf * hp / acc } else { 0.0 };

                if em <= 0.0 && eval_dmg <= 0.0 {
                    zero_both += 1;
                    continue;
                }
                compared += 1;
                let ratio = if em > 0.0 { eval_dmg / em } else { f64::INFINITY };
                let key = dex.moves.key(mv).to_string();
                let e = stats.entry(key).or_insert((0, 0.0, 1.0, String::new()));
                e.0 += 1;
                e.1 += ratio;
                if (ratio - 1.0).abs() > (e.2 - 1.0).abs() {
                    e.2 = ratio;
                    e.3 = format!(
                        "{} -> {}",
                        dex.species.get(b.poke(att).species).name.clone(),
                        dex.species.get(b.poke(def).species).name.clone()
                    );
                }
            }
        }
    }

    println!("\ncompared {compared} (attacker,defender,move) triples; {zero_both} zero-on-both\n");
    println!("{:<22} {:>5} {:>9} {:>9}  {}", "move", "n", "mean", "worst", "worst case");
    let mut rows: Vec<_> = stats.iter().collect();
    rows.sort_by(|a, b| {
        let da = (a.1 .1 / a.1 .0 as f64 - 1.0).abs();
        let db = (b.1 .1 / b.1 .0 as f64 - 1.0).abs();
        db.partial_cmp(&da).unwrap()
    });
    for (mv, (n, sum, worst, ex)) in rows {
        let mean = sum / *n as f64;
        let flag = if (mean - 1.0).abs() > 0.02 { " <== DIVERGES" } else { "" };
        println!("{mv:<22} {n:>5} {mean:>9.4} {worst:>9.4}  {ex}{flag}");
    }

    println!(
        "\ncost per call:  eval {:.0} ns   engine core {:.0} ns   ratio {:.1}x",
        eval_ns as f64 / eval_calls as f64,
        engine_ns as f64 / engine_calls as f64,
        (engine_ns as f64 / engine_calls as f64) / (eval_ns as f64 / eval_calls as f64)
    );
}
