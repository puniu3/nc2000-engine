//! The blind rebuild's shipped team files (docs/TEAM-POOL-REBUILD-PLAN.md
//! step 6) are what their versioned inputs say: the catalog is exactly the
//! selected parties with cluster-balanced draw weights, the Nash support is
//! inside the catalog, the frozen prior holds every measured eligible team
//! whatever its selection label, all three were measured on the same prior
//! file, catalog sets match the audited optimization of the inventory while
//! prior/Nash sets stay original, and every shipped team is legal and plays
//! a game to completion. The live search profile is blind at c = 0.4.

use std::collections::{BTreeMap, BTreeSet};

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{validate_team, Learnsets};
use serde_json::Value;

fn read(rel: &str) -> Value {
    let p = repo_root().join(rel);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{rel}: {e}")))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn fnv1a64(rel: &str) -> String {
    let bytes = std::fs::read(repo_root().join(rel)).unwrap();
    let h = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3));
    format!("fnv1a64:{h:016x}")
}

fn ids(v: &Value) -> BTreeSet<String> {
    v["teams"].as_array().unwrap().iter().map(|t| t["id"].as_str().unwrap().to_string()).collect()
}

const PRIOR: &str = "data/belief-pool-v3/belief-pool.json";

struct Products {
    inventory: BTreeMap<String, Value>,
    selection: Value,
    catalog: Value,
    prior: Value,
    nash: Value,
    solution: Value,
}

fn products() -> Products {
    let inv = read("data/team-inventory-v1/inventory.json");
    Products {
        inventory: inv["teams"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t["id"].as_str().unwrap().to_string(), t.clone()))
            .collect(),
        selection: read("data/team-selection-v2/selection.json"),
        catalog: read("data/team-pool-v2/team-pool.json"),
        prior: read(PRIOR),
        nash: read("data/meta-nash-v3/pool-artifact.json"),
        solution: read("data/meta-nash-v3/solution.json"),
    }
}

fn measured(p: &Products) -> BTreeSet<String> {
    p.inventory
        .iter()
        .filter(|(id, t)| t["eligibility"]["status"] == "eligible" && t["measuredAs"].as_str() == Some(id.as_str()))
        .map(|(id, _)| id.clone())
        .collect()
}

#[test]
fn memberships_follow_the_selection() {
    let p = products();
    let labels = p.selection["labels"].as_object().unwrap();
    let measured = measured(&p);
    assert_eq!(labels.keys().cloned().collect::<BTreeSet<_>>(), measured, "every measured team, and only those, carries a label");
    for (id, l) in labels {
        let label = l["label"].as_str().unwrap();
        assert!(["selected", "pending", "not-selected"].contains(&label), "{id}: {label}");
        if label != "selected" {
            assert!(l["reason"].is_string(), "{id}: {label} without a reason");
        }
    }
    let selected: BTreeSet<String> =
        labels.iter().filter(|(_, l)| l["label"] == "selected").map(|(id, _)| id.clone()).collect();
    assert_eq!(ids(&p.catalog), selected);
    for id in ids(&p.nash) {
        assert!(selected.contains(&id), "Nash support {id} is not a selected party");
    }
    assert_eq!(ids(&p.prior), measured, "the frozen prior holds every measured eligible team");
}

#[test]
fn everything_was_measured_on_the_frozen_prior() {
    let p = products();
    let fp = fnv1a64(PRIOR);
    assert_eq!(p.selection["prior"]["fingerprint"], fp.as_str());
    assert_eq!(p.solution["prior"]["fingerprint"], fp.as_str());
}

#[test]
fn shipped_sets_match_the_inventory_and_audited_catalog_optimization() {
    let p = products();
    for file in [&p.prior, &p.nash] {
        for t in file["teams"].as_array().unwrap() {
            let id = t["id"].as_str().unwrap();
            assert_eq!(t["sets"], p.inventory[id]["sets"], "{id}: sets differ from the inventory");
        }
    }
    let attack = read("findings/low-attack-catalog-v1/audit.json");
    let gender = read("findings/attract-gender-v1/mechanics.json");
    let mut applied = [0, 0];
    for t in p.catalog["teams"].as_array().unwrap() {
        let id = t["id"].as_str().unwrap();
        let mut expected = p.inventory[id]["sets"].clone();
        for (pass, audit) in [&attack, &gender].iter().enumerate() {
            for row in audit["rows"].as_array().unwrap().iter().filter(|r| r["baseId"] == id) {
                let set = &mut expected[row["slot"].as_u64().unwrap() as usize - 1];
                assert_eq!(set["species"], row["species"], "{id}: audited species");
                assert_eq!(set["ivs"]["atk"].as_u64().unwrap() / 2,
                    row["attackDvBefore"].as_u64().unwrap(), "{id}: audited baseline DV");
                set["ivs"]["atk"] = Value::from(row["attackDvAfter"].as_u64().unwrap() * 2);
                if pass == 0 {
                    assert_eq!(set["evs"]["atk"], row["attackEvBefore"], "{id}: audited baseline EV");
                    set["evs"]["atk"] = Value::from(0);
                } else {
                    assert_eq!(set["gender"], row["genderBefore"], "{id}: audited baseline gender");
                    set["gender"] = row["genderAfter"].clone();
                }
                applied[pass] += 1;
            }
        }
        assert_eq!(t["sets"], expected, "{id}: sets differ from the audited optimization");
    }
    assert_eq!(applied, [27, 6], "every audited adjustment applies exactly once");
}

#[test]
fn draw_weights_are_cluster_balanced() {
    let p = products();
    let cluster = |id: &str| p.inventory[id]["variantCluster"].as_str().unwrap_or(id).to_string();
    let teams = p.catalog["teams"].as_array().unwrap();
    let mut size: BTreeMap<String, usize> = BTreeMap::new();
    for t in teams {
        *size.entry(cluster(t["id"].as_str().unwrap())).or_default() += 1;
    }
    let total: f64 = teams.iter().map(|t| t["drawWeight"].as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-4, "draw weights sum to {total}");
    for t in teams {
        let id = t["id"].as_str().unwrap();
        let want = 1.0 / (size.len() as f64 * size[&cluster(id)] as f64);
        assert!((t["drawWeight"].as_f64().unwrap() - want).abs() < 1e-5, "{id}: drawWeight");
    }
    let nash: f64 = p.nash["teams"].as_array().unwrap().iter().map(|t| t["weight"].as_f64().unwrap()).sum();
    assert!((nash - 1.0).abs() < 5e-3, "nash weights sum to {nash}");
    let mut per_cluster: BTreeMap<String, f64> = BTreeMap::new();
    for t in p.prior["teams"].as_array().unwrap() {
        let w = t["weight"].as_f64().unwrap();
        assert!(w > 0.0 && w.is_finite(), "{}: prior weight {w}", t["id"]);
        *per_cluster.entry(cluster(t["id"].as_str().unwrap())).or_default() += w;
    }
    for (c, w) in per_cluster {
        assert!(w <= 1.0 + 1e-6, "prior cluster {c} carries weight {w}");
    }
}

#[test]
fn the_live_profile_is_blind_at_c_0_4() {
    let profiles = read("data/search-profiles.json");
    assert_eq!(profiles["blind"]["c"].as_f64(), Some(0.4));
    assert_eq!(profiles["blind"]["iterations"].as_u64(), Some(27000));
    assert!(profiles["open"]["retired"].is_string(), "the open profile only replays old records");
}

#[test]
fn every_shipped_team_is_legal_and_plays_out() {
    let p = products();
    let dex = load_dex();
    let ls = Learnsets::from_json(&std::fs::read_to_string(repo_root().join("data/learnsets-gen2.json")).unwrap())
        .unwrap();
    let mut seen = BTreeSet::new();
    let teams: Vec<&Value> = [&p.catalog, &p.prior, &p.nash]
        .iter()
        .flat_map(|f| f["teams"].as_array().unwrap().iter())
        .filter(|t| seen.insert(t["id"].as_str().unwrap().to_string()))
        .collect();
    for (i, t) in teams.iter().enumerate() {
        let id = t["id"].as_str().unwrap();
        let verdict = validate_team(&dex, &ls, &t["sets"].to_string());
        assert_eq!(verdict["ok"], true, "{id}: {verdict}");
        let sets: Vec<nc2000_engine::battle::PokemonSet> = serde_json::from_value(t["sets"].clone()).unwrap();
        let opp: Vec<nc2000_engine::battle::PokemonSet> =
            serde_json::from_value(teams[(i + 1) % teams.len()]["sets"].clone()).unwrap();
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &sets, &opp).unwrap();
        battle.set_log_enabled(false);
        let mut x: u64 = 0x5EED ^ i as u64;
        let mut steps = 0;
        while battle.outcome().is_none() {
            steps += 1;
            assert!(steps < 10_000, "{id}: no termination");
            let picks = [0usize, 1].map(|s| {
                let legal = battle.legal_choices(&dex, s);
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (!legal.is_empty()).then(|| legal[(x >> 33) as usize % legal.len()])
            });
            battle.apply_choices(&dex, picks).unwrap();
        }
    }
}
