use conformance::load_dex;
use nc2000_bot::smmcts::collapse_sleep_choices;
use nc2000_engine::{
    battle::{enumerate::enumerate_step, PokemonSet, SearchChoice},
    dex::Dex,
    state::{Battle, Status, DK},
};
use std::collections::BTreeMap;

fn set(moves: &[&str], item: &str) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "species":"Smeargle","name":"Smeargle","gender":"M","level":50,
        "moves":moves,"item":item,
        "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
    }))
    .unwrap()
}

fn position(
    dex: &Dex,
    side: usize,
    moves: &[&str],
    replies: &[&str],
    item: &str,
    time: i64,
) -> Battle {
    let mut bench = set(&["Splash"], "");
    bench.species = "Magikarp".into();
    bench.name = "Magikarp".into();
    let own = vec![set(moves, item), bench.clone()];
    let foe = vec![set(replies, ""), bench];
    let teams = if side == 0 { [own, foe] } else { [foe, own] };
    let mut b = Battle::from_fixture(dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
    b.choose(dex, 0, "team 1,2").unwrap();
    b.choose(dex, 1, "team 1,2").unwrap();
    let id = b.active_id(side).unwrap();
    b.restore_status(dex, id, Status::Slp, None);
    b.poke_mut(id).status_state.set_int(DK::Time, time);
    b
}

fn choices(b: &mut Battle, dex: &Dex, side: usize) -> Vec<String> {
    let mut acts = b.legal_choices(dex, side);
    collapse_sleep_choices(b, dex, side, &mut acts);
    acts.iter().map(|a| a.to_input(dex)).collect()
}

fn distribution(
    b: &Battle,
    dex: &Dex,
    own: SearchChoice,
    reply: SearchChoice,
    side: usize,
) -> BTreeMap<u128, f64> {
    let pair = if side == 0 {
        [own, reply]
    } else {
        [reply, own]
    };
    let step = enumerate_step(dex, b, pair.map(Some), 200_000).expect("complete chance support");
    let mut out = BTreeMap::new();
    for leaf in step.leaves {
        *out.entry(leaf.battle.state_key128()).or_default() += leaf.prob;
    }
    assert!((out.values().sum::<f64>() - 1.0).abs() < 1e-10);
    out
}

#[test]
fn preserves_sleep_moves_priorities_pursuit_switches_and_cures() {
    let dex = load_dex();
    for side in 0..2 {
        for (moves, expected) in [
            (
                ["Scratch", "Growl", "Rest", "Whirlwind"],
                vec!["move scratch", "move whirlwind", "switch 2"],
            ),
            (
                ["Scratch", "Growl", "Sleep Talk", "Snore"],
                vec!["move scratch", "move sleeptalk", "move snore", "switch 2"],
            ),
            (
                ["Scratch", "Growl", "Pursuit", "Quick Attack"],
                vec![
                    "move scratch",
                    "move pursuit",
                    "move quickattack",
                    "switch 2",
                ],
            ),
        ] {
            let mut b = position(&dex, side, &moves, &["Splash"], "", 2);
            assert_eq!(choices(&mut b, &dex, side), expected);
        }
        for (item, time) in [("", 1), ("Mint Berry", 2), ("Miracle Berry", 2)] {
            let mut b = position(
                &dex,
                side,
                &["Scratch", "Growl", "Rest", "Splash"],
                &["Splash"],
                item,
                time,
            );
            let expected = b
                .legal_choices(&dex, side)
                .iter()
                .map(|a| a.to_input(&dex))
                .collect::<Vec<_>>();
            assert_eq!(choices(&mut b, &dex, side), expected);
        }
    }
}

#[test]
fn merged_moves_have_identical_successors_against_every_reply() {
    let dex = load_dex();
    let mut checked = 0;
    for side in 0..2 {
        for replies in [
            ["Rock Slide", "Encore", "Disable", "Haze"],
            ["Thief", "Hypnosis", "Dream Eater", "Nightmare"],
            ["Destiny Bond", "Pursuit", "Counter", "Mirror Coat"],
        ] {
            let mut b = position(
                &dex,
                side,
                &["Scratch", "Growl", "Rest", "Swords Dance"],
                &replies,
                "Leftovers",
                2,
            );
            let legal = b.legal_choices(&dex, side);
            let mut merged = legal.clone();
            collapse_sleep_choices(&b, &dex, side, &mut merged);
            assert_eq!(merged.len(), 2);
            for reply in b.legal_choices(&dex, 1 - side) {
                let expected = distribution(&b, &dex, legal[0], reply, side);
                for &own in &legal[1..4] {
                    assert_eq!(
                        distribution(&b, &dex, own, reply, side),
                        expected,
                        "side={side}, own={own:?}, reply={reply:?}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 90);
}

#[test]
fn waking_and_berry_cures_make_apparently_redundant_moves_different() {
    let dex = load_dex();
    for (item, time) in [("", 1), ("Mint Berry", 2), ("Miracle Berry", 2)] {
        let mut b = position(&dex, 0, &["Scratch", "Growl"], &["Splash"], item, time);
        let own = b.legal_choices(&dex, 0);
        let reply = b.legal_choices(&dex, 1)[0];
        assert_ne!(
            distribution(&b, &dex, own[0], reply, 0),
            distribution(&b, &dex, own[1], reply, 0)
        );
    }
}
