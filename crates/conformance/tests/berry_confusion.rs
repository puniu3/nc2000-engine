//! Miracle Berry vs a mid-BeforeMove confusion cure — the one path with NO
//! PS oracle: the gen2stadium2nc2000 mod's onBeforeMove berry (priority 11)
//! removes the confusion volatile, and the already-collected confusion
//! handler then dereferences it — `volatiles["confusion"].time--` crashes
//! the reference core (reproduced on upstream master 393d5c867, 2026-07-21).
//! The community server runs the same combination, so the server itself
//! errors the battle here; there is no golden behavior to match.
//!
//! This test pins OUR engine's semantics (owner-accepted divergence): the
//! berry eats at BeforeMove, cures the confusion, the stale handler is
//! skipped, and the mon simply moves.

use conformance::load_dex;
use nc2000_engine::battle::PokemonSet;
use nc2000_engine::state::Battle;

fn mk(name: &str, moves: &[&str], item: &str) -> PokemonSet {
    serde_json::from_value(serde_json::json!({
        "name": name, "species": name, "item": item, "ability": "No Ability",
        "moves": moves, "level": 50,
        "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
        "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
        "happiness": 255
    }))
    .unwrap()
}

#[test]
fn miracle_berry_cures_confusion_at_before_move_without_crashing() {
    let dex = load_dex();
    let filler = |n: &str, m: &str| mk(n, &[m], "");
    let p1 = vec![
        mk("Gengar", &["Confuse Ray", "Night Shade"], ""),
        filler("Snorlax", "Body Slam"),
        filler("Marowak", "Earthquake"),
        filler("Exeggutor", "Psychic"),
        filler("Starmie", "Surf"),
        filler("Cloyster", "Surf"),
    ];
    let p2 = vec![
        mk("Jynx", &["Psychic", "Ice Beam"], "Miracle Berry"),
        filler("Snorlax", "Body Slam"),
        filler("Marowak", "Earthquake"),
        filler("Exeggutor", "Psychic"),
        filler("Starmie", "Surf"),
        filler("Cloyster", "Surf"),
    ];
    let mut b = Battle::from_fixture(&dex, "9,8,7,6", &p1, &p2).unwrap();
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    // T1: Gengar (faster) confuses Jynx; Jynx's BeforeMove fires the berry
    // (prio 11) before the confusion check — cure, then move normally.
    b.choose(&dex, 0, "move confuseray").unwrap();
    b.choose(&dex, 1, "move psychic").unwrap();
    let conf = dex.conds_id("confusion").unwrap();
    let jynx = b.sides[1]
        .roster
        .iter()
        .position(|p| dex.species.key(p.species) == "jynx")
        .unwrap();
    let jynx_id = nc2000_engine::state::PokeId { side: 1, slot: jynx as u8 };
    assert!(
        !b.poke(jynx_id).has_volatile(conf),
        "confusion should be cured by the berry: {:?}",
        b.log
    );
    assert!(b.poke(jynx_id).item.is_none(), "berry should be eaten: {:?}", b.log);
    assert!(
        b.log.iter().any(|l| l.contains("-enditem") && l.contains("Miracle Berry")),
        "{:?}",
        b.log
    );
    // Jynx still acted this turn (the cure precedes the confusion check)
    assert!(
        b.log.iter().any(|l| l.starts_with("|move|p2a: Jynx|Psychic")),
        "Jynx should move after the cure: {:?}",
        b.log
    );
    // the battle continues to be playable
    b.choose(&dex, 0, "move nightshade").unwrap();
    b.choose(&dex, 1, "move psychic").unwrap();
    assert!(!b.ended);
}
