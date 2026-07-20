//! Sleep Clause Mod conformance, hand-carried from a scripted PS battle
//! (RandomPlayerAI never produces the pattern, so the corpus can't cover it).
//!
//! PS reference (gen2nintendocup2000noohkostadium2strict, verified live
//! 2026-07-21): a second foe-sourced sleep on the same side is blocked with
//! `|-message|Sleep Clause Mod activated.`; a side whose only sleeper is
//! Rest-sleeping (ally-sourced) can still be slept.

use conformance::load_dex;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::{Battle, Status};

fn mk(name: &str, moves: &[&str]) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": name, "species": name, "item": "", "ability": "No Ability",
        "moves": moves, "level": 50,
        "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
        "happiness": 255
    }))
    .unwrap()
}

fn teams() -> (Vec<PokemonSet>, Vec<PokemonSet>) {
    let p1 = vec![
        mk("Parasect", &["Spore", "Slash"]),
        mk("Jynx", &["Lovely Kiss", "Psychic"]),
        mk("Snorlax", &["Body Slam", "Rest"]),
        mk("Gengar", &["Night Shade"]),
        mk("Exeggutor", &["Psychic"]),
        mk("Venomoth", &["Psychic"]),
    ];
    let p2 = vec![
        mk("Snorlax", &["Body Slam", "Rest"]),
        mk("Exeggutor", &["Psychic", "Rest"]),
        mk("Gengar", &["Night Shade", "Rest"]),
        mk("Jynx", &["Psychic"]),
        mk("Venomoth", &["Psychic"]),
        mk("Parasect", &["Slash"]),
    ];
    (p1, p2)
}

fn poke_status(b: &Battle, side: usize, species_key: &str, dex: &nc2000_engine::dex::Dex) -> Status {
    let side_ref = &b.sides[side];
    let slot = side_ref
        .roster
        .iter()
        .position(|p| dex.species.key(p.species) == species_key)
        .unwrap();
    side_ref.roster[slot].status
}

#[test]
fn second_foe_sleep_is_blocked() {
    let dex = load_dex();
    let (p1, p2) = teams();
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    b.choose(&dex, 0, "move spore").unwrap();
    b.choose(&dex, 1, "move bodyslam").unwrap();
    assert_eq!(poke_status(&b, 1, "snorlax", &dex), Status::Slp, "{:?}", b.log);
    b.choose(&dex, 0, "move slash").unwrap();
    b.choose(&dex, 1, "switch 2").unwrap();
    b.choose(&dex, 0, "move spore").unwrap();
    b.choose(&dex, 1, "move psychic").unwrap();
    assert!(
        b.log.iter().any(|l| l == "|-message|Sleep Clause Mod activated."),
        "clause message missing: {:?}",
        b.log
    );
    assert_eq!(poke_status(&b, 1, "exeggutor", &dex), Status::None);
}

#[test]
fn rest_sleep_does_not_engage_the_clause() {
    let dex = load_dex();
    let (p1, p2) = teams();
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    // damage Snorlax so Rest is usable, then Rest (ally-sourced sleep)
    b.choose(&dex, 0, "move slash").unwrap();
    b.choose(&dex, 1, "move bodyslam").unwrap();
    b.choose(&dex, 0, "move slash").unwrap();
    b.choose(&dex, 1, "move rest").unwrap();
    assert_eq!(poke_status(&b, 1, "snorlax", &dex), Status::Slp, "{:?}", b.log);
    b.choose(&dex, 0, "move slash").unwrap();
    b.choose(&dex, 1, "switch 2").unwrap();
    // foe sleep on Exeggutor must SUCCEED: the sleeping Snorlax is
    // Rest-sourced, which Sleep Clause Mod ignores (unlike Stadium Sleep
    // Clause, which the old format used)
    b.choose(&dex, 0, "move spore").unwrap();
    b.choose(&dex, 1, "move psychic").unwrap();
    assert!(
        !b.log.iter().any(|l| l == "|-message|Sleep Clause Mod activated."),
        "clause engaged wrongly: {:?}",
        b.log
    );
    assert_eq!(poke_status(&b, 1, "exeggutor", &dex), Status::Slp, "{:?}", b.log);
}
