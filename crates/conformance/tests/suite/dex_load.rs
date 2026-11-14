//! The exported dex loads, has the measured shape, and every callback name in
//! the data resolves in `dex.cb` (`KnownCallbacks`).

use crate::suite::common;

#[test]
fn dex_has_expected_counts() {
    let dex = common::dex().clone();
    assert_eq!(dex.species.len(), 251, "gen2 species");
    assert_eq!(dex.moves.len(), 268, "gen2 moves (incl. 16 typed Hidden Powers + synthetic recharge)");
    assert_eq!(dex.items.len(), 62, "gen2 items");
    assert_eq!(dex.conditions.len(), 37, "conditions table");
    assert_eq!(dex.typechart.len(), 17, "gen2 type chart (17 types)");
}

#[test]
fn every_data_callback_resolves_in_dex_cb_table() {
    let dex = common::dex().clone();
    let mut unknown = Vec::new();
    let mut total = 0;
    let mut check = |entry: &str, names: &[String], is_move: bool| {
        for n in names {
            total += 1;
            if is_move {
                if let Some((prefix, leaf)) = n.split_once('.') {
                    if !matches!(prefix, "condition" | "self" | "secondary") {
                        unknown.push(format!("{entry}: {n} (unexpected prefix)"));
                        continue;
                    }
                    let cb = dex.cb(leaf);
                    if !cb.exists() || dex.cb_key(cb) != leaf {
                        unknown.push(format!("{entry}: {n}"));
                    } else if prefix == "condition"
                        && !dex.conds_id(entry).is_some_and(|cid| dex.cond(cid).mask.has(cb))
                    {
                        unknown.push(format!("{entry}: {n} (missing from cond mask)"));
                    }
                    continue;
                }
            }
            let cb = dex.cb(n);
            if !cb.exists() || dex.cb_key(cb) != n {
                unknown.push(format!("{entry}: {n}"));
            }
        }
    };
    for (i, m) in dex.moves.values.iter().enumerate() {
        check(&dex.moves.keys[i], &m.callbacks, true);
    }
    for (i, it) in dex.items.values.iter().enumerate() {
        check(&dex.items.keys[i], &it.callbacks, false);
    }
    for (i, c) in dex.conditions.values.iter().enumerate() {
        check(&dex.conditions.keys[i], &c.callbacks, false);
    }
    assert!(unknown.is_empty(), "callback names missing from dex cb_index:\n{}", unknown.join("\n"));
    assert!(total > 300, "expected the measured ~355 callbacks, saw {total}");
}

#[test]
fn typechart_is_integer_coded() {
    let dex = common::dex().clone();
    let normal = &dex.typechart["normal"];
    assert_eq!(normal.damage_taken["Fighting"], 1, "Normal is weak to Fighting");
    assert_eq!(normal.damage_taken["Ghost"], 3, "Normal is immune to Ghost");
}
