use std::collections::{BTreeMap, BTreeSet};

use conformance::load_dex;
use nc2000_bot::pick_prior::PickPrior;
use nc2000_bot::preview::{MetaPool, MetaTeam};
use nc2000_bot::{Belief, Observer, SplitMix64};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::state::Battle;
use serde_json::json;

fn sets(levels: [u8; 6]) -> Vec<PokemonSet> {
    ["snorlax", "raikou", "marowak", "heracross", "cloyster", "machamp"]
        .into_iter().zip(levels).map(|(species, level)| serde_json::from_value(json!({
            "species": species, "name": species, "level": level, "moves": ["rest"],
        })).unwrap()).collect()
}

#[test]
fn hidden_selections_match_engine_legality_with_one_or_two_unknowns() {
    let dex = load_dex();
    let team = sets([55, 55, 50, 50, 50, 50]);
    let pool = MetaPool { teams: vec![MetaTeam { id: "test".into(), sets: team.clone(), weight: None }] };
    let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
    let legal: BTreeSet<u8> = battle.legal_choices(&dex, 1).iter().map(|choice| {
        let SearchChoice::Team(slots) = choice else { panic!() };
        slots.iter().fold(0u8, |mask, slot| mask | (1 << (slot-1)))
    }).collect();
    let mut obs = Observer::new(&battle, 0);
    let mut beliefs = [Belief::pinned_from_battle(&battle, &obs), Belief::new(&dex, &pool, &obs),
        Belief::new(&dex, &MetaPool { teams: vec![] }, &obs), Belief::new(&dex, &pool, &obs)];
    let prior = PickPrior::from_json(&json!({
        "schema": "nc2000-pick-prior-v1", "smoothing": 0.2, "source": {},
        "rows": [{"team": "test", "side": 1, "enemy_preview": [], "choices": [
            {"species": ["snorlax", "raikou", "marowak"], "count": 1000},
            {"species": ["snorlax", "marowak", "heracross"], "count": 1}]}],
    }).to_string()).unwrap();
    battle.apply_choices(&dex, [Some(SearchChoice::Team([1, 3, 4])); 2]).unwrap();
    for reveals in 1..=2 {
        if reveals == 2 {
            battle.apply_choices(&dex, [Some(SearchChoice::Switch(2)); 2]).unwrap();
        }
        obs.observe(&battle, &dex);
        let public = obs.mons().iter().enumerate().fold(0u8, |mask, (i, m)| mask | (u8::from(m.appeared) << i));
        let expected: BTreeSet<u8> = legal.iter().copied().filter(|mask| mask & public == public).collect();
        assert_eq!(expected.len(), if reveals == 1 { 6 } else { 3 });
        for (mode, belief) in beliefs.iter_mut().enumerate() {
            belief.sync(&dex, &obs);
            if mode == 3 { belief.condition_pick_prior(&dex, &prior, &[], &obs); }
            let mut counts = BTreeMap::new();
            for seed in 0..4096 {
                let det = belief.determinize(&dex, &battle, &obs, &mut SplitMix64::new(seed));
                let side = &det.sides[1];
                let mask = side.party.iter().fold(0u8, |mask, slot| mask | (1 << slot));
                assert!(expected.contains(&mask), "mode={mode} reveals={reveals} mask={mask}");
                assert_eq!(mask.count_ones(), 3);
                let total: u32 = side.party.iter().map(|&slot| u32::from(side.roster[slot as usize].level)).sum();
                assert!(total <= 155);
                for (position, &slot) in side.party.iter().enumerate() {
                    assert_eq!(side.roster[slot as usize].position as usize, position);
                }
                for (position, &slot) in battle.sides[1].party.iter().enumerate() {
                    if obs.mons()[slot as usize].appeared { assert_eq!(side.party[position], slot); }
                }
                *counts.entry(mask).or_insert(0) += 1;
            }
            assert_eq!(counts.keys().copied().collect::<BTreeSet<_>>(), expected);
            if mode < 3 {
                let uniform = 4096.0 / expected.len() as f64;
                assert!(counts.values().all(|&n| (n as f64 - uniform).abs() < uniform * 0.15));
            }
        }
    }
}

#[test]
fn nonbinding_cap_preserves_samples_and_rng() {
    let dex = load_dex();
    let team = sets([55, 50, 50, 50, 50, 50]);
    let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
    let mut obs = Observer::new(&battle, 0);
    let mut belief = Belief::pinned_from_battle(&battle, &obs);
    battle.apply_choices(&dex, [Some(SearchChoice::Team([1, 3, 4])); 2]).unwrap();
    obs.observe(&battle, &dex);
    belief.sync(&dex, &obs);
    let mut uncapped = battle.clone();
    uncapped.preview_level_caps[1] = None;
    for seed in 0..256 {
        let mut a = SplitMix64::new(seed);
        let mut b = a.clone();
        let mut bounded = belief.determinize(&dex, &battle, &obs, &mut a);
        let free = belief.determinize(&dex, &uncapped, &obs, &mut b);
        bounded.preview_level_caps[1] = None;
        assert_eq!(bounded.state_key128(), free.state_key128());
        assert_eq!(bounded.essence(&dex), free.essence(&dex));
        assert_eq!(a.next(), b.next());
    }
}
