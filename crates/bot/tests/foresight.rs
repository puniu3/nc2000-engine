use conformance::load_dex;
use nc2000_bot::{
    eval::expected_hit_fraction,
    mcts::{playout_pick, Playout, RolloutRules},
    rng::SplitMix64,
    preview::{MetaPool, MetaTeam},
    smmcts::dominated_actions,
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, EffectHandle, Outcome, PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};

fn set(species: &str, moves: &[&str], ev: u8) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "species": species, "name": species, "gender": "M", "level": 50, "moves": moves,
        "evs": {"hp":ev,"atk":ev,"def":ev,"spa":ev,"spd":ev,"spe":ev},
        "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
    }))
    .unwrap()
}

fn position(dex: &Dex, side: usize, own: &PokemonSet, foe: &PokemonSet) -> Battle {
    let teams = if side == 0 { [own, foe] } else { [foe, own] };
    let mut b = Battle::from_fixture(
        dex,
        "1,2,3,4",
        std::slice::from_ref(teams[0]),
        std::slice::from_ref(teams[1]),
    )
    .unwrap();
    b.choose(dex, 0, "team 1").unwrap();
    b.choose(dex, 1, "team 1").unwrap();
    b
}

fn foresight(b: &mut Battle, dex: &Dex, side: usize) {
    let source = b.active_id(side).unwrap();
    let target = b.active_id(1 - side).unwrap();
    assert!(b
        .add_volatile(dex, target, "foresight", Some(source), EffectHandle::None)
        .truthy());
    assert!(b.log.iter().any(|line| line.ends_with("|Foresight")));
}

fn masked(b: &Battle, dex: &Dex, side: usize, key: &str) -> bool {
    dominated_actions(b, dex, side)
        .iter()
        .any(|(action, _)| *action == SearchChoice::Move(dex.moves.id(key).unwrap()))
}

#[test]
fn foresight_restores_normal_and_fighting_only_while_active() {
    let dex = load_dex();
    let own = set(
        "Machamp",
        &["Foresight", "Return", "Cross Chop", "Rest"],
        255,
    );
    let foe = set("Gengar", &["Psychic"], 255);
    for side in 0..2 {
        let mut b = position(&dex, side, &own, &foe);
        for key in ["return", "crosschop"] {
            assert!(masked(&b, &dex, side, key));
        }
        foresight(&mut b, &dex, side);
        for key in ["return", "crosschop"] {
            assert!(!masked(&b, &dex, side, key));
        }
        let target = b.active_id(1 - side).unwrap();
        b.remove_volatile(&dex, target, "foresight");
        for key in ["return", "crosschop"] {
            assert!(masked(&b, &dex, side, key));
        }
    }
}

#[test]
fn foresight_preserves_unrelated_immunities() {
    let dex = load_dex();
    for (attacker, key, defender) in [
        ("Machamp", "earthquake", "Pidgeot"),
        ("Raichu", "thunderbolt", "Marowak"),
        ("Gengar", "shadowball", "Snorlax"),
    ] {
        for side in 0..2 {
            let mut b = position(
                &dex,
                side,
                &set(attacker, &[key, "Rest"], 255),
                &set(defender, &["Rest"], 255),
            );
            foresight(&mut b, &dex, side);
            assert!(masked(&b, &dex, side, key));
        }
    }
}

fn tactical_sets(ev: u8) -> [PokemonSet; 2] {
    [
        set(
            "Pidgeot",
            &["Foresight", "Quick Attack", "Agility", "Rest"],
            0,
        ),
        set(
            "Gengar",
            &["Psychic", "Thunderbolt", "Ice Punch", "Fire Punch"],
            ev,
        ),
    ]
}

fn tactical_position(dex: &Dex, side: usize, ev: u8) -> Battle {
    let sets = tactical_sets(ev);
    let mut b = position(dex, side, &sets[0], &sets[1]);
    foresight(&mut b, dex, side);
    for s in 0..2 {
        let id = b.active_id(s).unwrap();
        b.poke_mut(id).hp = 1;
    }
    assert!(
        b.get_pokemon_action_speed(dex, b.active_id(side).unwrap())
            < b.get_pokemon_action_speed(dex, b.active_id(1 - side).unwrap())
    );
    b
}

#[test]
fn priority_attack_has_separated_complete_support_bounds() {
    let dex = load_dex();
    let mut min_priority_lower = 1.0f64;
    let mut max_other_upper = 0.0f64;
    let mut runs = 0;
    for side in 0..2 {
        for ev in [0, 255] {
            let mut b = tactical_position(&dex, side, ev);
            let own = b.legal_choices(&dex, side);
            let replies = b.legal_choices(&dex, 1 - side);
            for action in own {
                for &reply in &replies {
                    let choices = if side == 0 {
                        [action, reply]
                    } else {
                        [reply, action]
                    };
                    let result = enumerate_step_partial(&dex, &b, choices.map(Some), 100_000);
                    runs += result.runs;
                    assert!(!result.exhausted && result.pending.is_empty());
                    let mut lower = 0.0;
                    let mut upper = 0.0;
                    for leaf in &result.leaves {
                        if let Some(outcome) = leaf.battle.outcome() {
                            let p0 = match outcome {
                                Outcome::P1Win => 1.0,
                                Outcome::P2Win => 0.0,
                                Outcome::Tie => 0.5,
                            };
                            let value = if side == 0 { p0 } else { 1.0 - p0 };
                            lower += leaf.prob * value;
                            upper += leaf.prob * value;
                        } else {
                            upper += leaf.prob;
                        }
                    }
                    assert!(
                        (result.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-10
                    );
                    let priority =
                        action == SearchChoice::Move(dex.moves.id("quickattack").unwrap());
                    let expected = if (side == 0) == priority {
                        Outcome::P1Win
                    } else {
                        Outcome::P2Win
                    };
                    assert!(result
                        .leaves
                        .iter()
                        .all(|leaf| leaf.battle.outcome() == Some(expected)));
                    if priority {
                        min_priority_lower = min_priority_lower.min(lower);
                    } else {
                        max_other_upper = max_other_upper.max(upper);
                    }
                    assert!(
                        if priority { lower > 0.99 } else { upper < 0.01 },
                        "{side} {ev} {action:?} {reply:?}: [{lower}, {upper}]"
                    );
                }
            }
        }
    }
    eprintln!("priority_lower={min_priority_lower} other_upper={max_other_upper} runs={runs}");
}

#[test]
fn blind_search_keeps_the_winning_priority_attack() {
    let dex = load_dex();
    let pool = MetaPool {
        teams: [0, 255]
            .into_iter()
            .map(|ev| MetaTeam {
                id: format!("gengar-{ev}"),
                sets: vec![tactical_sets(ev)[1].clone()],
                weight: None,
            })
            .collect(),
    };
    let mut all_choices = Vec::new();
    for side in 0..2 {
        let mut outputs = Vec::new();
        for ev in [0, 255] {
            let b = tactical_position(&dex, side, ev);
            let mut obs = Observer::new(&b, side);
            obs.observe(&b, &dex);
            let belief = Belief::new(&dex, &pool, &obs);
            assert_eq!(belief.candidate_count(), 2);
            let mut choices = Vec::new();
            for seed in 0..8 {
                let cfg = RmConfig {
                    c: 0.4,
                    iterations: 27_000,
                    ..Default::default()
                };
                let mut search = BlindSearch::new(&b, &dex, cfg, side, seed);
                search.step(&dex, &belief, &obs, 27_000);
                choices.push(search.best().unwrap());
            }
            eprintln!(
                "side={side} hidden_ev={ev} choices={:?}",
                choices.iter().map(|a| a.to_input(&dex)).collect::<Vec<_>>()
            );
            outputs.push(choices);
        }
        assert_eq!(outputs[0], outputs[1]);
        all_choices.extend(outputs.into_iter().flatten());
    }
    assert!(all_choices
        .iter()
        .all(|&a| a == SearchChoice::Move(dex.moves.id("quickattack").unwrap())));
}

#[test]
fn damage_estimate_respects_public_immunity_removal() {
    let dex = load_dex();
    for side in 0..2 {
        for (attacker, defender, key, restored) in [
            ("Machamp", "Gengar", "return", true),
            ("Machamp", "Gengar", "crosschop", true),
            ("Machamp", "Gengar", "seismictoss", true),
            ("Machamp", "Pidgeot", "earthquake", false),
            ("Raichu", "Marowak", "thunderbolt", false),
            ("Gengar", "Snorlax", "shadowball", false),
        ] {
            let mut b = position(&dex, side, &set(attacker, &[key], 255), &set(defender, &["splash"], 255));
            let att = b.active_id(side).unwrap();
            let def = b.active_id(1-side).unwrap();
            let mid = dex.moves.id(key).unwrap();
            assert_eq!(expected_hit_fraction(&b, &dex, att, def, mid, true), 0.0);
            foresight(&mut b, &dex, side);
            let choices = if side == 0 {
                [SearchChoice::Move(mid), SearchChoice::Move(dex.moves.id("splash").unwrap())]
            } else {
                [SearchChoice::Move(dex.moves.id("splash").unwrap()), SearchChoice::Move(mid)]
            };
            let outcomes = enumerate_step_partial(&dex, &b, choices.map(Some), 100_000);
            assert!(outcomes.pending.is_empty());
            let native_damage: f64 = outcomes.leaves.iter().map(|leaf| leaf.prob * f64::from(b.poke(def).hp-leaf.battle.poke(def).hp)).sum();
            assert_eq!(native_damage > 0.0, restored);
            let estimate = expected_hit_fraction(&b, &dex, att, def, mid, true);
            assert_eq!(estimate > 0.0, restored, "{key}, side {side}, native {native_damage}, estimate {estimate}");
            if key == "seismictoss" {
                assert!((estimate * f64::from(b.poke(def).hp) - native_damage).abs() < 1e-8);
            }
            b.remove_volatile(&dex, def, "foresight");
            assert_eq!(expected_hit_fraction(&b, &dex, att, def, mid, true), 0.0);
        }
    }
}

#[test]
fn greedy_rollout_uses_foresight_enabled_winning_attack() {
    let dex = load_dex();
    let playout = Playout::Heavy { eps: 0.0, turns: 8, weights: Default::default() };
    let attack = SearchChoice::Move(dex.moves.id("quickattack").unwrap());
    for side in 0..2 {
        for ev in [0, 255] {
            let mut b = tactical_position(&dex, side, ev);
            let actions = b.legal_choices(&dex, side);
            for seed in 0..32 {
                let picked = playout_pick(&b, &dex, &playout, side, &actions, &mut SplitMix64(seed), RolloutRules::default());
                assert_eq!(picked, attack, "side {side}, ev {ev}, seed {seed}");
            }
        }
    }
}
