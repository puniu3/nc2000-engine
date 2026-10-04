use std::path::Path;

use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{Belief, FallbackPolicy, Observer, SplitMix64};
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn main() {
    assert!(!cfg!(feature = "hp-type-evidence"));
    let inventory_path = "data/team-inventory-v1/inventory.json";
    let prior_path = "data/belief-pool-v3/belief-pool.json";
    let inventory: Value = serde_json::from_slice(&std::fs::read(inventory_path).unwrap()).unwrap();
    let pool = load_meta_pool(Path::new(prior_path));
    let dex = Dex::from_json(&std::fs::read_to_string("data/gen2stadium2.json").unwrap()).unwrap();
    let mut rows = Vec::new();
    for team in inventory["teams"].as_array().unwrap() {
        if team["origin"] != "human" || team["eligibility"]["status"] != "eligible" {
            continue;
        }
        let sets: Vec<PokemonSet> = serde_json::from_value(team["sets"].clone()).unwrap();
        let battle = Battle::from_fixture(&dex, "1,2,3,4", &sets, &sets).unwrap();
        let obs = Observer::new(&battle, 0);
        let policies = [FallbackPolicy::EarlyPerish, FallbackPolicy::EarlyRest];
        let beliefs = policies.map(|policy| Belief::with_fallback_policy(&dex, &pool, &obs, policy));
        let candidates: Vec<_> = beliefs[0].alive().iter().map(|&i| beliefs[0].candidate_id(i)).collect();
        let full_support: Vec<_> = pool.teams.iter().filter(|t| candidates.contains(&t.id.as_str())).filter(|t| {
            let reference = Battle::from_fixture(&dex, "1,2,3,4", &t.sets, &t.sets).unwrap();
            battle.sides[1].roster.iter().all(|actual| {
                reference.sides[1].roster.iter().any(|r| {
                    r.species == actual.species && r.level == actual.level && r.gender == actual.gender
                    && r.item == actual.item && actual.base_move_slots.iter().all(|m| {
                        let key = dex.moves.key(m.id);
                        r.base_move_slots.iter().any(|n| {
                            let other = dex.moves.key(n.id);
                            key == other || key.starts_with("hiddenpower") && other.starts_with("hiddenpower")
                        })
                    })
                })
            })
        }).map(|t| t.id.as_str()).collect();
        let same_samples = (0..64).all(|seed| {
            beliefs[0].determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed)).state_key128()
                == beliefs[1].determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed)).state_key128()
        });
        rows.push(json!({"team":team["id"],"family":team["family"],
            "id_in_prior":pool.teams.iter().any(|t| team["id"] == t.id),
            "preview_candidates":candidates,"full_move_item_support":full_support,
            "fallback_at_preview":beliefs[0].is_fallback(),"paired_preview_samples_equal":same_samples}));
    }
    let inputs: serde_json::Map<String, Value> = [inventory_path, prior_path, "data/gen2stadium2.json", "crates/bot/src/belief.rs", "crates/bot/examples/rest_holdout_audit.rs"].into_iter().map(|path| {
        (path.into(),json!(format!("{:x}",Sha256::digest(std::fs::read(path).unwrap()))))
    }).collect();
    println!("{}",serde_json::to_string_pretty(&json!({"rows":rows,"inputs_sha256":inputs,
        "scope":"Eligible human-origin inventory only. Native preview filter; maximal move/item support is a set-compatibility check, not a legal game trajectory. Plain Hidden Power reveals do not identify type. No human performance or independent holdout claim."})).unwrap());
}
