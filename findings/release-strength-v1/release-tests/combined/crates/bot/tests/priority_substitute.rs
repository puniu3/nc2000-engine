use conformance::load_dex;
use nc2000_bot::smmcts::dominated_actions;
use nc2000_engine::{
    battle::{enumerate::enumerate_step, PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};

fn position(dex: &Dex, side: usize, species: &str, attack: &str, setup: &str) -> Battle {
    let set = |species: &str, moves: &[&str], defense: u8| -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "species":species,"name":species,"level":50,"gender":"M","moves":moves,
            "evs":{"hp":255,"atk":255,"def":defense,"spa":255,"spd":255,"spe":255},
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
        })).unwrap()
    };
    let own = [set("Jolteon", &["Substitute", "Thunderbolt", "Rest", "Agility"], 0)];
    let foe = [set(species, &[attack, setup], 255)];
    let teams = if side == 0 { [&own, &foe] } else { [&foe, &own] };
    let mut b = Battle::from_fixture(dex, "1,2,3,4", teams[0], teams[1]).unwrap();
    b.choose(dex, 0, "team 1").unwrap();
    b.choose(dex, 1, "team 1").unwrap();
    b.choose(dex, side, "move substitute").unwrap();
    b.choose(dex, 1 - side, &format!("move {setup}")).unwrap();
    let own = b.active_id(side).unwrap();
    let foe = b.active_id(1 - side).unwrap();
    assert!(b.get_pokemon_action_speed(dex, own) > b.get_pokemon_action_speed(dex, foe));
    assert!(b.poke(own).has_volatile(dex.conds_id("substitute").unwrap()));
    b
}

fn masked(b: &Battle, dex: &Dex, side: usize) -> bool {
    dominated_actions(b, dex, side).iter()
        .any(|(action, _)| *action == SearchChoice::Move(dex.moves.id("substitute").unwrap()))
}

#[test]
fn priority_damage_can_break_and_replace_an_existing_substitute() {
    let dex = load_dex();
    let mut incorrectly_masked = Vec::new();
    for side in 0..2 {
        for (species, attack, setup) in [
            ("Scizor", "quickattack", "swordsdance"),
            ("Hitmonchan", "machpunch", "detect"),
            ("Arcanine", "extremespeed", "leer"),
        ] {
            let b = position(&dex, side, species, attack, setup);
            let own = b.active_id(side).unwrap();
            let hp = b.poke(own).hp;
            let cost = b.poke(own).maxhp / 4;
            let mut joint = [None; 2];
            joint[side] = Some(SearchChoice::Move(dex.moves.id("substitute").unwrap()));
            joint[1 - side] = Some(SearchChoice::Move(dex.moves.id(attack).unwrap()));
            let step = enumerate_step(&dex, &b, joint, 4096).expect("complete chance enumeration");

            let mut rebuilt = 0.0;
            let mut unchanged = 0.0;
            for leaf in &step.leaves {
                assert!(leaf.battle.poke(own).has_volatile(dex.conds_id("substitute").unwrap()));
                match leaf.battle.poke(own).hp {
                    after if after == hp - cost => rebuilt += leaf.prob,
                    after if after == hp => unchanged += leaf.prob,
                    after => panic!("unexpected HP {after}, before {hp}, cost {cost}"),
                }
            }
            assert!((rebuilt + unchanged - 1.0).abs() < 1e-9);
            assert!(rebuilt > 0.99, "{species}: {rebuilt}");
            let excluded = masked(&b, &dex, side);
            println!("{}", serde_json::json!({"side":side,"species":species,"attack":attack,
                "runs":step.runs,"rebuilt_mass":rebuilt,"unchanged_mass":unchanged,"masked":excluded}));
            if excluded { incorrectly_masked.push((side, species)); }
        }
    }
    assert!(incorrectly_masked.is_empty(), "{incorrectly_masked:?}");
}

#[test]
fn existing_substitute_and_insufficient_hp_still_exclude_certain_failures() {
    let dex = load_dex();
    for side in 0..2 {
        assert!(masked(&position(&dex, side, "Snorlax", "return", "curse"), &dex, side));
        let mut b = position(&dex, side, "Scizor", "quickattack", "swordsdance");
        let own = b.active_id(side).unwrap();
        b.poke_mut(own).hp = b.poke(own).maxhp / 4;
        assert!(masked(&b, &dex, side));
        b.remove_volatile(&dex, own, "substitute");
        assert!(masked(&b, &dex, side));
    }
}

#[test]
fn hidden_move_identity_and_pp_do_not_control_substitute_exclusion() {
    let dex = load_dex();
    for side in 0..2 {
        for species in ["Scizor", "Snorlax"] {
            let mut b = position(&dex, side, species, "return", "swordsdance");
            let expected = masked(&b, &dex, side);
            assert_eq!(expected, species == "Snorlax");
            let foe = b.active_id(1 - side).unwrap();
            for key in ["quickattack", "machpunch", "extremespeed", "splash"] {
                for pp in [0, 1, 32] {
                    b.poke_mut(foe).move_slots[0].id = dex.moves.id(key).unwrap();
                    b.poke_mut(foe).move_slots[0].pp = pp;
                    assert_eq!(masked(&b, &dex, side), expected);
                }
            }
        }
    }
}

#[test]
fn copied_moves_unknown_learnsets_and_quick_claw_keep_substitute_available() {
    let dex = load_dex();
    for side in 0..2 {
        for case in ["transformed", "copied", "unknown", "quickclaw"] {
            let mut b = position(&dex, side, "Snorlax", "return", "curse");
            let foe = b.active_id(1 - side).unwrap();
            match case {
                "transformed" => b.poke_mut(foe).transformed = true,
                "copied" => b.poke_mut(foe).move_slots[0].shared = false,
                "unknown" => b.poke_mut(foe).species = dex.species.id("mewtwo").unwrap(),
                "quickclaw" => {
                    b.poke_mut(foe).item = dex.items.id("quickclaw");
                    b.quick_claw_roll = false;
                }
                _ => unreachable!(),
            }
            assert!(!masked(&b, &dex, side), "{case}");
        }
    }
}
