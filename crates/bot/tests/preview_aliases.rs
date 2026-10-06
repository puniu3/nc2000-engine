use nc2000_bot::{Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::Dex,
    prng::BattleRng,
    state::{Battle, PokeId, Status},
};
use std::collections::BTreeMap;

fn mon(species: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "species":species,"name":species,"level":50,"moves":moves
    }))
    .unwrap()
}

fn team() -> Vec<PokemonSet> {
    [
        "Smeargle", "Snorlax", "Machamp", "Zapdos", "Gengar", "Starmie",
    ]
    .map(|s| mon(s, &["Splash", "Heal Bell"]))
    .to_vec()
}

fn root(dex: &Dex, teams: &[Vec<PokemonSet>; 2]) -> Battle {
    Battle::from_fixture(dex, "1,2,3,4", &teams[0], &teams[1]).unwrap()
}

#[test]
fn preserves_indices_external_masks_and_small_teams() {
    let dex = conformance::load_dex();
    for size in 1..=6 {
        for side in 0..2 {
            let own = team()[..size].to_vec();
            let mut b = root(&dex, &[own.clone(), own]);
            let choices = b.legal_choices(&dex, side);
            let mut search = BlindSearch::new(&b, &dex, RmConfig::default(), side, 47);
            let expected = if size < 3 { 0 } else { choices.len() / 2 };
            assert_eq!(search.prune_preview_aliases(&dex), expected);
            assert_eq!(search.actions(), choices);
            assert_eq!(search.prune_preview_aliases(&dex), 0);
            let allowed = search.root_policy_evidence().0;
            assert_eq!(
                allowed.iter().filter(|&&a| a).count(),
                choices.len() - expected
            );
            if size >= 3 {
                let mut masked = BlindSearch::new(&b, &dex, RmConfig::default(), side, 47);
                let reverse: Vec<_> = choices
                    .iter()
                    .map(|a| matches!(a, SearchChoice::Team([_, b, c]) if b > c))
                    .collect();
                masked.mask_actions(&reverse);
                assert_eq!(masked.prune_preview_aliases(&dex), 0);
                assert_eq!(masked.root_policy_evidence().0, reverse);
            }
        }
    }
    let mut teams = [team(), team()];
    teams[0][0].level = 55;
    teams[0][1].level = 55;
    let mut b = root(&dex, &teams);
    let choices = b.legal_choices(&dex, 0);
    assert_eq!(choices.len(), 96);
    let mut search = BlindSearch::new(&b, &dex, RmConfig::default(), 0, 47);
    assert_eq!(search.prune_preview_aliases(&dex), 48);
    let observer = Observer::new(&b, 0);
    let belief = Belief::pinned_from_battle(&b, &observer);
    search.step(&dex, &belief, &observer, 256);
    let allowed = search.root_policy_evidence().0;
    assert!(search
        .visits()
        .iter()
        .zip(&allowed)
        .all(|(&n, &ok)| ok || n == 0));
    assert!(choices
        .iter()
        .zip(&allowed)
        .any(|(a, &ok)| ok && Some(*a) == search.best()));
    assert_eq!(search.actions(), choices);
}

#[test]
fn excludes_beat_up_and_every_move_acquisition_route_without_reading_foe() {
    let dex = conformance::load_dex();
    for unsafe_move in [
        "Beat Up",
        "Metronome",
        "Mimic",
        "Mirror Move",
        "Sketch",
        "Transform",
    ] {
        for side in 0..2 {
            let mut teams = [team(), team()];
            teams[side][5] = mon("Smeargle", &[unsafe_move]);
            let b = root(&dex, &teams);
            let mut own = BlindSearch::new(&b, &dex, RmConfig::default(), side, 1);
            assert_eq!(own.prune_preview_aliases(&dex), 0, "{unsafe_move}");
            let mut foe = BlindSearch::new(&b, &dex, RmConfig::default(), 1 - side, 1);
            assert_eq!(foe.prune_preview_aliases(&dex), 60, "{unsafe_move}");
        }
    }
    let mut b = root(&dex, &[team(), team()]);
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    let mut search = BlindSearch::new(&b, &dex, RmConfig::default(), 0, 1);
    assert_eq!(search.prune_preview_aliases(&dex), 0);
}

fn normalize(b: &mut Battle, side: usize) {
    b.sides[side].party[1..].sort_unstable();
    for position in 1..b.sides[side].party.len() {
        let slot = b.sides[side].party[position];
        b.sides[side].roster[slot as usize].position = position as u8;
    }
}

fn kernel(
    b: &Battle,
    dex: &Dex,
    joint: [SearchChoice; 2],
    side: usize,
) -> BTreeMap<(u128, Vec<String>), f64> {
    let mut out = BTreeMap::new();
    let mut pending = vec![Vec::new()];
    let mut runs = 0;
    while let Some(script) = pending.pop() {
        runs += 1;
        assert!(runs <= 50_000);
        let mut leaf = b.clone();
        leaf.set_log_enabled(true);
        leaf.prng = BattleRng::enumerating(script.clone());
        leaf.apply_choices(dex, joint.map(Some)).unwrap();
        let trace = &leaf.prng.oracle.as_ref().unwrap().trace;
        if let Some(draw) = trace.get(script.len()) {
            for (choice, &count) in draw.counts.iter().enumerate() {
                if count > 0 {
                    let mut next = script.clone();
                    next.push(choice);
                    pending.push(next);
                }
            }
        } else {
            let probability = trace.iter().map(|d| d.prob()).product::<f64>();
            normalize(&mut leaf, side);
            assert!(!leaf.log.is_empty());
            *out.entry((leaf.state_key128(), leaf.log)).or_default() += probability;
        }
    }
    assert!((out.values().sum::<f64>() - 1.0).abs() < 1e-10);
    out
}

#[test]
fn swaps_preserve_switch_heal_bell_and_phazing_kernels_including_logs() {
    let dex = conformance::load_dex();
    for side in 0..2 {
        let mut teams = [team(), team()];
        teams[1 - side][0] = mon("Skarmory", &["Whirlwind", "Splash"]);
        let b = root(&dex, &teams);
        let preview = [[1, 2, 3], [1, 3, 2]].map(|order| {
            let mut joint = [SearchChoice::Team([1, 2, 3]); 2];
            joint[side] = SearchChoice::Team(order);
            kernel(&b, &dex, joint, side)
        });
        assert_eq!(preview[0], preview[1]);
        let states = ["team 1,2,3", "team 1,3,2"].map(|order| {
            let mut b = root(&dex, &teams);
            b.choose(&dex, side, order).unwrap();
            b.choose(&dex, 1 - side, "team 1,2,3").unwrap();
            for slot in 1..=2 {
                b.restore_status(
                    &dex,
                    PokeId {
                        side: side as u8,
                        slot,
                    },
                    Status::Psn,
                    None,
                );
            }
            b
        });
        for own in ["move splash", "move healbell", "switch 2", "switch 3"] {
            for reply in ["move splash", "move whirlwind"] {
                let distributions = [0, 1].map(|i| {
                    let mut b = states[i].clone();
                    let own_input = match (i, own) {
                        (1, "switch 2") => "switch 3",
                        (1, "switch 3") => "switch 2",
                        _ => own,
                    };
                    let mut joint = [SearchChoice::Pass; 2];
                    joint[side] = b
                        .legal_choices(&dex, side)
                        .into_iter()
                        .find(|a| a.to_input(&dex) == own_input)
                        .unwrap();
                    joint[1 - side] = b
                        .legal_choices(&dex, 1 - side)
                        .into_iter()
                        .find(|a| a.to_input(&dex) == reply)
                        .unwrap();
                    kernel(&b, &dex, joint, side)
                });
                assert_eq!(
                    distributions[0], distributions[1],
                    "side={side} own={own} reply={reply}"
                );
            }
        }
    }
}

#[test]
fn beat_up_reveals_bench_members_in_different_orders() {
    let dex = conformance::load_dex();
    let mut teams = [team(), team()];
    teams[0][0] = mon("Smeargle", &["Beat Up"]);
    let orders = ["team 1,2,3", "team 1,3,2"].map(|order| {
        let mut b = root(&dex, &teams);
        b.set_log_enabled(true);
        b.choose(&dex, 0, order).unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();
        b.prng = BattleRng::enumerating(Vec::new());
        b.choose(&dex, 0, "move beatup").unwrap();
        b.choose(&dex, 1, "move splash").unwrap();
        let lines: Vec<_> = b
            .log
            .into_iter()
            .filter(|line| line.starts_with("|-activate|") && line.contains("Beat Up"))
            .collect();
        assert_eq!(lines.len(), 3);
        lines
    });
    assert_eq!(orders[0][0], orders[1][0]);
    assert_eq!(orders[0][1], orders[1][2]);
    assert_eq!(orders[0][2], orders[1][1]);
    assert_ne!(orders[0], orders[1]);
}
