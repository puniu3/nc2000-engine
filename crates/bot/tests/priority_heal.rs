use conformance::load_dex;
use nc2000_bot::smmcts::{dominated_actions_with, MaskRules};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};

fn position(dex: &Dex, side: usize, foe: &str, reply: &str) -> Battle {
    let set = |species: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "species":species,"name":species,"level":50,"moves":moves,
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30}
        }))
        .unwrap()
    };
    let own = [set(
        "Alakazam",
        &["Recover", "Rest", "Psychic", "Explosion"],
    )];
    let opp = [set(foe, &[reply])];
    let teams = if side == 0 {
        [&own, &opp]
    } else {
        [&opp, &own]
    };
    let mut b = Battle::from_fixture(dex, "1,2,3,4", teams[0], teams[1]).unwrap();
    b.choose(dex, 0, "team 1").unwrap();
    b.choose(dex, 1, "team 1").unwrap();
    b
}

fn masked(b: &Battle, dex: &Dex, side: usize, key: &str, enabled: bool) -> bool {
    dominated_actions_with(
        b,
        dex,
        side,
        MaskRules {
            priority_heal: enabled,
            ..Default::default()
        },
    )
    .iter()
    .any(|(a, _)| a.to_input(dex) == format!("move {key}"))
}

#[test]
fn priority_damage_makes_full_hp_recovery_work_on_both_sides() {
    let dex = load_dex();
    for side in 0..2 {
        for (species, reply) in [
            ("Scizor", "quickattack"),
            ("Hitmonchan", "machpunch"),
            ("Arcanine", "extremespeed"),
        ] {
            for heal in ["recover", "rest"] {
                let mut b = position(&dex, side, species, reply);
                assert!(masked(&b, &dex, side, heal, false));
                assert!(!masked(&b, &dex, side, heal, true));
                assert!(masked(&b, &dex, side, "explosion", true));
                b.log.clear();
                b.choose(&dex, side, &format!("move {heal}")).unwrap();
                b.choose(&dex, 1 - side, &format!("move {reply}")).unwrap();
                assert!(
                    b.log
                        .iter()
                        .any(|line| line.starts_with(&format!("|-heal|p{}a:", side + 1))),
                    "{:?}",
                    b.log
                );
            }
        }
    }
}

#[test]
fn full_hp_exclusion_survives_when_the_species_cannot_preempt() {
    let dex = load_dex();
    for side in 0..2 {
        let mut b = position(&dex, side, "Snorlax", "return");
        for heal in ["recover", "rest"] {
            assert!(masked(&b, &dex, side, heal, true));
        }
        let id = b.active_id(side).unwrap();
        b.poke_mut(id).hp -= 1;
        assert!(!masked(&b, &dex, side, "recover", true));
    }
}

#[test]
fn hidden_moves_and_pp_do_not_change_exclusions() {
    let dex = load_dex();
    for species in ["Scizor", "Snorlax"] {
        let mut b = position(&dex, 0, species, "return");
        let expected = dominated_actions_with(&b, &dex, 0, MaskRules::default());
        let id = b.active_id(1).unwrap();
        for key in ["quickattack", "machpunch", "extremespeed", "splash"] {
            for pp in [0, 1, 32] {
                b.poke_mut(id).move_slots[0].id = dex.moves.id(key).unwrap();
                b.poke_mut(id).move_slots[0].pp = pp;
                assert_eq!(
                    dominated_actions_with(&b, &dex, 0, MaskRules::default()),
                    expected
                );
            }
        }
    }
}

#[test]
fn copied_moves_and_unknown_learnsets_keep_recovery_available() {
    let dex = load_dex();
    for case in ["transformed", "copied", "unknown", "quickclaw"] {
        let mut b = position(&dex, 0, "Snorlax", "return");
        let id = b.active_id(1).unwrap();
        match case {
            "transformed" => b.poke_mut(id).transformed = true,
            "copied" => b.poke_mut(id).move_slots[0].shared = false,
            "unknown" => b.poke_mut(id).species = dex.species.id("mewtwo").unwrap(),
            "quickclaw" => {
                b.poke_mut(id).item = dex.items.id("quickclaw");
                b.quick_claw_roll = false;
            }
            _ => unreachable!(),
        }
        assert!(!masked(&b, &dex, 0, "recover", true), "{case}");
    }
}
