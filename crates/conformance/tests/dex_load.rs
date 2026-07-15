//! The exported dex loads, has the measured shape, and every callback name in
//! the data maps onto an `events::Handler` variant (no silent enum drift).

use conformance::load_dex;
use nc2000_engine::events::Handler;

#[test]
fn dex_has_expected_counts() {
    let dex = load_dex();
    assert_eq!(dex.species.len(), 251, "gen2 species");
    assert_eq!(dex.moves.len(), 267, "gen2 moves (incl. 16 typed Hidden Powers)");
    assert_eq!(dex.items.len(), 62, "gen2 items");
    assert_eq!(dex.conditions.len(), 37, "conditions table");
    assert_eq!(dex.typechart.len(), 17, "gen2 type chart (17 types)");
}

#[test]
fn every_data_callback_maps_to_a_handler_variant() {
    let dex = load_dex();
    let mut unknown = Vec::new();
    let mut total = 0;
    let mut check = |entry: &str, names: &[String]| {
        for n in names {
            total += 1;
            if Handler::from_name(n).is_none() {
                unknown.push(format!("{entry}: {n}"));
            }
        }
    };
    for (i, m) in dex.moves.values.iter().enumerate() {
        check(&dex.moves.keys[i], &m.callbacks);
    }
    for (i, it) in dex.items.values.iter().enumerate() {
        check(&dex.items.keys[i], &it.callbacks);
    }
    for (i, c) in dex.conditions.values.iter().enumerate() {
        check(&dex.conditions.keys[i], &c.callbacks);
    }
    assert!(unknown.is_empty(), "callback names without a Handler variant:\n{}", unknown.join("\n"));
    assert!(total > 300, "expected the measured ~355 callbacks, saw {total}");
}

#[test]
fn typechart_is_integer_coded() {
    let dex = load_dex();
    let normal = &dex.typechart["normal"];
    assert_eq!(normal.damage_taken["Fighting"], 1, "Normal is weak to Fighting");
    assert_eq!(normal.damage_taken["Ghost"], 3, "Normal is immune to Ghost");
}
