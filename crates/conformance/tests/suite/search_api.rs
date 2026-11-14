//! M3 search-API conformance.

use crate::suite::common;
use conformance::ps_reference_battle;
use nc2000_engine::state::Battle;

fn side_index(side: &str) -> usize {
    if side == "p1" {
        0
    } else {
        1
    }
}

#[test]
fn legal_choices_cover_corpus() {
    let dex = common::dex();
    for fx in common::puredata_full_fixtures() {
        let mut battle = ps_reference_battle(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        for (i, line) in fx.choices.iter().enumerate() {
            let side_n = side_index(&line.side);
            let legal = battle.legal_choices(dex, side_n);
            let strings: Vec<String> = legal.iter().map(|c| c.to_input(dex)).collect();
            assert!(
                strings.iter().any(|s| s == &line.choice),
                "fixture {}-{:03}: choice {:?} (line {i}) not in legal set {strings:?}",
                fx.meta.pool,
                fx.meta.index,
                line.choice,
            );
            if i % 5 == 0 {
                for choice in &legal {
                    let mut probe = battle.clone();
                    probe.set_log_enabled(false);
                    let mut text = probe.clone();
                    text.choose(dex, side_n, &choice.to_input(dex)).unwrap();
                    if let Err(e) = probe.apply_choice(dex, side_n, *choice) {
                        panic!(
                            "fixture {}-{:03} line {i}: enumerated choice {:?} rejected: {e:?}",
                            fx.meta.pool,
                            fx.meta.index,
                            choice.to_input(dex),
                        );
                    }
                    assert_eq!(format!("{probe:?}"), format!("{text:?}"), "structured/text choice diverged: {choice:?}");
                }
            }
            battle.choose(dex, side_n, &line.choice).unwrap();
        }
    }
}

#[test]
fn nolog_replay_reaches_identical_state() {
    let dex = common::dex();
    for fx in common::puredata_full_fixtures() {
        let mut on = ps_reference_battle(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        let mut off = ps_reference_battle(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        off.set_log_enabled(false);
        let frozen_log_len = off.log.len();
        for line in &fx.choices {
            let side_n = side_index(&line.side);
            on.choose(dex, side_n, &line.choice).unwrap();
            off.choose(dex, side_n, &line.choice).unwrap();
        }
        let tag = format!("{}-{:03}", fx.meta.pool, fx.meta.index);
        assert_eq!(off.log.len(), frozen_log_len, "{tag}: log grew while disabled");
        assert_eq!(on.prng.seed_str(), off.prng.seed_str(), "{tag}: PRNG diverged");
        assert_eq!(on.turn, off.turn, "{tag}: turn diverged");
        assert_eq!(on.winner, off.winner, "{tag}: winner diverged");
        assert_eq!(on.essence(dex), off.essence(dex), "{tag}: state essence diverged");
    }
}

struct TestRng(u64);

impl TestRng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pick(&mut self, len: usize) -> usize {
        (self.next() % len as u64) as usize
    }
}

#[test]
fn random_playouts_terminate() {
    let dex = common::dex();
    for (fi, fx) in common::puredata_full_fixtures().iter().enumerate() {
        let mut results = Vec::with_capacity(2);
        for playout in 0..2u64 {
            let mut battle =
                Battle::from_fixture(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            battle.set_log_enabled(false);
            let pre_k = battle.state_key();
            let pre_k128 = battle.state_key128();
            let pre_s16 = battle.search_key(Some(16), true);
            let seed = 0xC0FFEE ^ (fi as u64) << 8 ^ playout;
            battle.reseed(seed);
            assert_eq!(battle.state_key(), pre_k, "reseed must not alter state_key");
            assert_eq!(battle.state_key128(), pre_k128, "reseed must not alter state_key128");
            assert_eq!(battle.search_key(Some(16), true), pre_s16, "reseed must not alter search_key");
            let mut rng = TestRng(0xDEAD_BEEF ^ (fi as u64) << 32 ^ playout);
            let mut steps = 0u32;
            while battle.outcome().is_none() {
                steps += 1;
                assert!(
                    steps < 10_000,
                    "{}-{:03} playout {playout}: no termination after {steps} decision points",
                    fx.meta.pool,
                    fx.meta.index,
                );
                let picks = [0usize, 1].map(|side_n| {
                    let legal = battle.legal_choices(dex, side_n);
                    if legal.is_empty() {
                        None
                    } else {
                        Some(legal[rng.pick(legal.len())])
                    }
                });
                assert!(
                    picks.iter().any(|p| p.is_some()),
                    "{}-{:03} playout {playout}: battle running but nobody owes a choice",
                    fx.meta.pool,
                    fx.meta.index,
                );
                battle.apply_choices(dex, picks).unwrap();
            }
            results.push((battle.outcome(), battle.state_key128()));
        }
        if fi == 0 {
            let playout = 0u64;
            let mut clone_b = Battle::from_fixture(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
            clone_b.set_log_enabled(false);
            clone_b.reseed(0xC0FFEE ^ (fi as u64) << 8 ^ playout);
            let mut rng = TestRng(0xDEAD_BEEF ^ (fi as u64) << 32 ^ playout);
            while clone_b.outcome().is_none() {
                let picks = [0usize, 1].map(|side_n| {
                    let legal = clone_b.legal_choices(dex, side_n);
                    (!legal.is_empty()).then(|| legal[rng.pick(legal.len())])
                });
                clone_b.apply_choices(dex, picks).unwrap();
            }
            assert_eq!((clone_b.outcome(), clone_b.state_key128()), results[0]);
        }
    }
}

#[test]
fn search_keys_preserve_state_partitions() {
    use nc2000_engine::battle::BattleSearchCache;
    use std::collections::HashMap;
    let dex = common::dex();
    let mut partitions = HashMap::new();
    let mut reverse = HashMap::new();
    for fx in common::puredata_full_fixtures() {
        let mut battle = ps_reference_battle(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        for line in &fx.choices {
            for (buckets, omit, stable) in [
                (None, false, battle.state_key128()),
                (None, true, battle.state_key128_without_damage_bookkeeping()),
                (Some(1), false, battle.state_key_bucketed(1) as u128),
                (Some(16), false, battle.state_key_bucketed(16) as u128),
                (Some(256), false, battle.state_key_bucketed(256) as u128),
            ] {
                let fast = battle.search_key(buckets, omit);
                if let Some(previous) = partitions.insert((buckets, omit, fast), stable) {
                    assert_eq!(previous, stable, "search key merged distinct states");
                }
                if let Some(previous) = reverse.insert((buckets, omit, stable), fast) {
                    assert_eq!(previous, fast, "search key split an existing state class");
                }
            }
            for buckets in [None, Some(1), Some(16), Some(256)] {
                for omit in [false, true] {
                    let cache = BattleSearchCache::new(&battle, dex, buckets, omit);
                    assert_eq!(
                        battle.search_key_with_cache(dex, buckets, omit, false, &cache),
                        battle.search_key(buckets, omit),
                    );
                    assert_eq!(
                        battle.search_key_with_cache(dex, buckets, omit, true, &cache),
                        battle.search_key_canonical_claw(dex, buckets, omit),
                    );
                }
            }
            battle.choose(dex, side_index(&line.side), &line.choice).unwrap();
        }
    }
    assert!(partitions.len() > 1000);
}

#[test]
fn reusable_battle_clone_resets_every_field() {
    let dex = common::dex();
    let mut reused = None;
    let mut cross_reused = None;
    for fx in common::puredata_full_fixtures() {
        let mut battle = ps_reference_battle(dex, &fx.seed, &fx.p1team, &fx.p2team).unwrap();
        let mut search_reused = None;
        for (step, line) in fx.choices.iter().enumerate() {
            battle.choose(dex, side_index(&line.side), &line.choice).unwrap();
            let target = reused.get_or_insert_with(|| battle.clone());
            target.clone_from(&battle);
            assert_eq!(target.state_key128(), battle.state_key128());
            let sim_target = search_reused.get_or_insert_with(|| battle.clone());
            sim_target.reset_search_sim(&battle, [true, true]);
            assert_eq!(sim_target.state_key128(), battle.state_key128());
            let cross_target = cross_reused.get_or_insert_with(|| battle.clone());
            cross_target.reset_search_sim(&battle, [false, false]);
            assert_eq!(cross_target.state_key128(), battle.state_key128());
            if step % 16 == 0 {
                assert_eq!(format!("{target:?}"), format!("{:?}", battle.clone()));
            }
        }
    }
}
