//! Hand-entered positions: the spec's own contract.
//!
//! The heavy gate — that a position round-trips a real decision point
//! byte-for-byte — lives in `tests/import.rs`, where the corpus replay
//! already has real trackers to export. What is left here is what that gate
//! cannot see, because it only ever feeds itself specs the exporter wrote:
//! whether a position a HUMAN types is accepted, rejected for the right
//! reason, and analyzed as the thing they described.

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::analysis;
use nc2000_bot::import::ProtocolAgent;
use nc2000_bot::position::{synthesize_spec, PositionSpec};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::state::Status;
use serde_json::{json, Value};

fn pool() -> MetaPool {
    load_meta_pool(&repo_root().join("data/meta-pool-v0/meta-pool.json"))
}

fn pool_sets(id: &str) -> Vec<Value> {
    let text =
        std::fs::read_to_string(repo_root().join("data/meta-pool-v0/meta-pool.json")).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    v["teams"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"] == id)
        .unwrap_or_else(|| panic!("pool team {id}"))["sets"]
        .as_array()
        .unwrap()
        .clone()
}

fn mon(species: &str, level: u64) -> Value {
    json!({"species": species, "level": level})
}

/// Turn 9: our chipped Snorlax is out against their paralysed Zapdos, which
/// has shown Thunderbolt and Drill Peck; their Cloyster is down and Spikes
/// are on their side. Written the way the screen will write it — public
/// facts only for them, exact sets for us.
fn demo_position() -> Value {
    let mine = pool_sets("sample-07"); // Electrode/Tauros/Cloyster/Marowak/Snorlax/Zapdos
    json!({
        "schema": "nc2000-position-v1",
        "side": 0,
        "turn": 9,
        "own_sets": mine,
        "sides": [
            {
                "active": 4,
                "party": [4, 5, 1],
                "mons": [
                    mon("Electrode", 50),
                    json!({"species": "Tauros", "level": 55, "gender": "M", "appeared": true}),
                    mon("Cloyster", 50),
                    mon("Marowak", 55),
                    json!({"species": "Snorlax", "level": 50, "appeared": true, "active": true,
                           "hp_num": 62, "hp_den": 100,
                           "uses": [{"move": "bodyslam", "n": 3}]}),
                    json!({"species": "Zapdos", "level": 55, "appeared": true}),
                ],
            },
            {
                "active": 5,
                "mons": [
                    mon("Exeggutor", 50),
                    mon("Machamp", 50),
                    mon("Miltank", 55),
                    json!({"species": "Cloyster", "level": 50, "appeared": true,
                           "fainted": true, "hp_num": 0, "status": "fnt"}),
                    mon("Snorlax", 50),
                    json!({"species": "Zapdos", "level": 55, "appeared": true, "active": true,
                           "hp_num": 71, "hp_den": 100, "status": "par",
                           "revealed_moves": ["thunderbolt", "drillpeck"],
                           "uses": [{"move": "thunderbolt", "n": 2},
                                    {"move": "drillpeck", "n": 1}]}),
                ],
                "conditions": [{"key": "spikes", "start_turn": 4}],
            },
        ],
    })
}

#[test]
fn a_hand_written_position_becomes_the_battle_it_describes() {
    let dex = load_dex();
    let spec = PositionSpec::parse(&demo_position().to_string()).unwrap();
    let b = synthesize_spec(&dex, &spec, &pool(), None, 7).unwrap();

    assert_eq!(b.turn, 9);
    // us: exact, because our sets are ours
    let me = b.active_id(0).unwrap();
    assert_eq!(dex.species.key(b.poke(me).species), "snorlax");
    let (hp, maxhp) = (b.poke(me).hp, b.poke(me).maxhp);
    assert!(
        (hp as f64 / maxhp as f64 - 0.62).abs() < 0.01,
        "own HP {hp}/{maxhp} should sit at the stated 62%"
    );
    // Body Slam has been used three times and PP must show it
    let bs = dex.moves.id("bodyslam").unwrap();
    let slot = b.poke(me).move_slots.iter().find(|m| m.id == bs).unwrap();
    assert_eq!(slot.maxpp - slot.pp, 3, "three public uses of Body Slam");

    // them: public facts exact, HP inside the announced bucket
    let foe = b.active_id(1).unwrap();
    assert_eq!(dex.species.key(b.poke(foe).species), "zapdos");
    assert_eq!(b.poke(foe).status, Status::Par);
    let (fhp, fmax) = (b.poke(foe).hp, b.poke(foe).maxhp);
    assert!(
        (fhp as f64 / fmax as f64 - 0.71).abs() < 0.02,
        "foe HP {fhp}/{fmax} should sit in the announced 71% bucket"
    );
    // revealed moves must be in the imputed set — that is what a reveal MEANS
    for key in ["thunderbolt", "drillpeck"] {
        let id = dex.moves.id(key).unwrap();
        assert!(
            b.poke(foe).base_move_slots.iter().any(|m| m.id == id),
            "revealed move {key} missing from the imputed set"
        );
    }
    assert!(b.poke(foe).fainted.eq(&false));
    assert_eq!(b.sides[1].pokemon_left, 2, "their Cloyster is down");
    assert!(
        b.sides[1].side_conditions.iter().any(|(c, _)| dex.conds_key(*c) == "spikes"),
        "Spikes are on their side"
    );
}

#[test]
fn a_position_is_analyzed_and_every_legal_action_is_scored() {
    let dex = load_dex();
    let spec = PositionSpec::parse(&demo_position().to_string()).unwrap();
    let cfg = RmConfig { rule: SelRule::Ucb, ..RmConfig::default() };
    let mut agent = ProtocolAgent::new(&dex, 0, pool(), cfg, 11);
    agent.set_position(&dex, &spec).unwrap();
    agent.step(&dex, 600).unwrap();
    let r = analysis::report(&agent, &dex, 4, 11);

    let actions = r["actions"].as_array().unwrap();
    // four moves + two live benched mons
    assert_eq!(actions.len(), 6, "actions: {actions:#?}");
    let inputs: Vec<&str> = actions.iter().map(|a| a["input"].as_str().unwrap()).collect();
    for want in ["move bodyslam", "move selfdestruct", "switch 2", "switch 3"] {
        assert!(inputs.contains(&want), "missing {want} in {inputs:?}");
    }
    let total: f64 = actions.iter().map(|a| a["frac"].as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-6, "visit shares must sum to 1, got {total}");
    for a in actions {
        let mean = a["mean"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&mean), "win rate out of range: {a}");
    }
    // sorted by visits, descending
    let visits: Vec<u64> = actions.iter().map(|a| a["visits"].as_u64().unwrap()).collect();
    assert!(visits.windows(2).all(|w| w[0] >= w[1]), "not sorted by visits: {visits:?}");

    // the matrix is the joint the marginals hide: every sampled cell is a
    // real (ours, theirs) pair, and the row totals cannot exceed the visits
    let cols = r["matrix"]["cols"].as_array().unwrap();
    assert!(!cols.is_empty(), "no opponent replies were sampled");
    for (i, row) in r["matrix"]["cells"].as_array().unwrap().iter().enumerate() {
        let n: u64 = row
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c.get("n").and_then(|x| x.as_u64()))
            .sum();
        assert!(n <= visits[i], "row {i} has {n} joint samples but only {} visits", visits[i]);
    }

    // damage is engine truth: Earthquake cannot touch a Flying-type
    let eq = r["damage"]["mine"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["move"] == "earthquake")
        .expect("Earthquake is on this Snorlax");
    assert_eq!(eq["max"].as_i64(), Some(0), "Ground move vs Zapdos");
    let bs = r["damage"]["mine"]
        .as_array()
        .unwrap()
        .iter()
        .find(|d| d["move"] == "bodyslam")
        .unwrap();
    assert!(bs["min"].as_i64().unwrap() > 0);
    assert!(bs["min"].as_i64() <= bs["max"].as_i64());

    // their damage rows say which moves were actually shown
    let theirs = r["damage"]["theirs"].as_array().unwrap();
    assert!(
        theirs.iter().any(|d| d["move"] == "thunderbolt" && d["revealed"] == json!(true)),
        "Thunderbolt was revealed and must be marked so"
    );

    // the line is searched, not invented: every step names an action we own
    for step in r["line"]["steps"].as_array().unwrap() {
        assert!(step["mine"].is_string() || step["mine"].is_null());
    }
}

#[test]
fn a_position_that_cannot_be_true_is_refused_by_name() {
    let dex = load_dex();
    let bad = |edit: &dyn Fn(&mut Value)| -> String {
        let mut v = demo_position();
        edit(&mut v);
        PositionSpec::parse(&v.to_string())
            .err()
            .unwrap_or_else(|| panic!("expected a rejection"))
    };

    assert!(bad(&|v| v["schema"] = json!("nc2000-position-v99")).contains("schema"));
    assert!(bad(&|v| v["sides"][0]["mons"][4]["hp_num"] = json!(0)).contains("not fainted"));
    assert!(bad(&|v| v["sides"][0]["mons"][4]["hp_num"] = json!(140)).contains("out of"));
    assert!(bad(&|v| v["sides"][0]["mons"][0]["active"] = json!(true)).contains("active"));
    assert!(bad(&|v| v["sides"][1]["active"] = json!(9)).contains("out of range"));
    // an unknown field is a typo, not a fact to ignore
    assert!(bad(&|v| v["sides"][0]["mons"][0]["hp_percent"] = json!(50)).contains("hp_percent"));
    // and a species the dex never heard of fails at build time, not silently
    let mut v = demo_position();
    v["sides"][1]["mons"][0]["species"] = json!("Missingno");
    let spec = PositionSpec::parse(&v.to_string()).unwrap();
    assert!(synthesize_spec(&dex, &spec, &pool(), None, 3)
        .unwrap_err()
        .contains("Missingno"));
}

#[test]
fn a_forced_switch_position_offers_only_switches() {
    let dex = load_dex();
    let mut v = demo_position();
    v["force_switch"] = json!(true);
    v["sides"][0]["mons"][4]["hp_num"] = json!(0);
    v["sides"][0]["mons"][4]["fainted"] = json!(true);
    v["sides"][0]["mons"][4]["status"] = json!("fnt");
    let spec = PositionSpec::parse(&v.to_string()).unwrap();
    let cfg = RmConfig { rule: SelRule::Ucb, ..RmConfig::default() };
    let mut agent = ProtocolAgent::new(&dex, 0, pool(), cfg, 5);
    agent.set_position(&dex, &spec).unwrap();
    agent.step(&dex, 200).unwrap();
    let r = analysis::report(&agent, &dex, 0, 5);
    let inputs: Vec<&str> =
        r["actions"].as_array().unwrap().iter().map(|a| a["input"].as_str().unwrap()).collect();
    assert!(!inputs.is_empty());
    assert!(inputs.iter().all(|i| i.starts_with("switch")), "{inputs:?}");
}

#[test]
fn team_preview_is_a_position_too() {
    let dex = load_dex();
    let mine = pool_sets("sample-07");
    let mons: Vec<Value> = mine
        .iter()
        .map(|s| json!({"species": s["species"], "level": s["level"]}))
        .collect();
    let foe: Vec<Value> = pool_sets("sample-08")
        .iter()
        .map(|s| json!({"species": s["species"], "level": s["level"]}))
        .collect();
    let spec = PositionSpec::parse(
        &json!({
            "schema": "nc2000-position-v1",
            "side": 0,
            "turn": 0,
            "team_preview": true,
            "own_sets": mine,
            "sides": [{"mons": mons}, {"mons": foe}],
        })
        .to_string(),
    )
    .unwrap();
    let cfg = RmConfig { rule: SelRule::Ucb, ..RmConfig::default() };
    let mut agent = ProtocolAgent::new(&dex, 0, pool(), cfg, 3);
    agent.set_position(&dex, &spec).unwrap();
    agent.step(&dex, 200).unwrap();
    let r = analysis::report(&agent, &dex, 0, 3);
    let actions = r["actions"].as_array().unwrap();
    assert!(!actions.is_empty(), "team preview must offer picks");
    assert!(
        actions.iter().all(|a| a["kind"] == "team"),
        "preview actions are picks: {actions:#?}"
    );
    assert_eq!(r["preview"], json!(true));
}
