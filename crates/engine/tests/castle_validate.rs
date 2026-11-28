use std::path::{Path, PathBuf};

use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::Dex;
use nc2000_engine::replay::{Recorder, Replay};
use nc2000_engine::rules::{CastleOverlayConfig, LearnsetPolicy, Ruleset};
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{
    canonicalize_team, canonicalize_team_with_ruleset, castle_max_pp, validate_team,
    validate_team_with_ruleset, CastleRules, EffortVerdict, Learnsets,
};
use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn load_data() -> (Dex, Learnsets) {
    let root = repo_root();
    let dex = Dex::from_json(&std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap())
        .unwrap();
    let ls = Learnsets::from_json(
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    (dex, ls)
}

fn base_mon(species: &str, level: i64, moves: &[&str], item: &str) -> Value {
    json!({
        "name": species,
        "species": species,
        "item": item,
        "ability": "No Ability",
        "moves": moves,
        "nature": "Serious",
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "gender": match species {
            "Zapdos" | "Suicune" | "Starmie" | "Porygon" | "Raikou" => "N",
            _ => "M",
        },
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
        "level": level,
    })
}

fn pad_team(mut sets: Vec<Value>) -> String {
    let fillers = [
        base_mon("Snorlax", 50, &["Body Slam", "Rest"], ""),
        base_mon("Zapdos", 50, &["Thunderbolt", "Rest"], ""),
        base_mon("Suicune", 50, &["Surf", "Rest"], ""),
        base_mon("Cloyster", 50, &["Surf", "Spikes"], ""),
        base_mon("Marowak", 50, &["Earthquake", "Rest"], ""),
        base_mon("Umbreon", 50, &["Toxic", "Rest"], ""),
    ];
    for f in fillers {
        if sets.len() >= 6 {
            break;
        }
        let sp = f["species"].as_str().unwrap();
        if sets.iter().any(|s| s["species"].as_str() == Some(sp)) {
            continue;
        }
        sets.push(f);
    }
    serde_json::to_string(&sets).unwrap()
}

fn finding_codes(v: &Value) -> Vec<(String, String)> {
    v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            (
                f["severity"].as_str().unwrap().to_string(),
                f["code"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

#[test]
fn castle_ruleset_cross_check_against_belief_pools() {
    let (dex, ls) = load_data();
    let root = repo_root();
    let castle = Ruleset::castle_tournament();

    // 1. data/belief-pool-v3/belief-pool.json: expect 19 / 527 Castle-illegal teams
    let bp_text = std::fs::read_to_string(root.join("data/belief-pool-v3/belief-pool.json")).unwrap();
    let bp: Value = serde_json::from_str(&bp_text).unwrap();
    let bp_teams = bp["teams"].as_array().unwrap();
    assert_eq!(bp_teams.len(), 527);

    let mut bp_illegal_ids: Vec<String> = Vec::new();
    let mut bp_illegal_sets = 0usize;
    for t in bp_teams {
        let id = t["id"].as_str().unwrap().to_string();
        let sets_json = t["sets"].to_string();
        let v = validate_team_with_ruleset(&dex, &ls, &sets_json, castle);
        if v["ok"] != Value::Bool(true) {
            bp_illegal_ids.push(id);
            let mut bad_slots: Vec<u64> = v["findings"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|f| f["severity"] == "error")
                .filter_map(|f| f["slot"].as_u64())
                .collect();
            bad_slots.sort_unstable();
            bad_slots.dedup();
            bp_illegal_sets += bad_slots.len();
        }
    }
    assert_eq!(
        bp_illegal_ids.len(),
        19,
        "expected 19/527 Castle-illegal teams in belief-pool-v3, got {}: {bp_illegal_ids:?}",
        bp_illegal_ids.len()
    );
    assert_eq!(bp_illegal_sets, 25, "expected 25 Castle-illegal sets in belief-pool-v3");
    assert!(bp_illegal_ids.contains(&"pb-68".to_string()));
    assert!(bp_illegal_ids.contains(&"mjj-2017-night".to_string()));
    assert!(bp_illegal_ids.contains(&"supp-bp-jolteon-growth".to_string()));
    assert!(bp_illegal_ids.contains(&"supp-cov-extra-3".to_string()));
    // Jolteon Lv50 Agility + Baton Pass + Substitute must remain LEGAL (col5 <= 50)
    assert!(!bp_illegal_ids.contains(&"supp-bp-jolteon-marowak-1".to_string()));

    // 2. crates/bot/src/belief_supplement_v1.json: expect 17 / 43 Castle-illegal teams (23 sets)
    let sup_text =
        std::fs::read_to_string(root.join("crates/bot/src/belief_supplement_v1.json")).unwrap();
    let sup: Value = serde_json::from_str(&sup_text).unwrap();
    let sup_teams = sup["teams"].as_array().unwrap();
    assert_eq!(sup_teams.len(), 43);

    let mut sup_illegal_teams = 0usize;
    let mut sup_illegal_sets = 0usize;
    for t in sup_teams {
        let sets_json = t["sets"].to_string();
        let v = validate_team_with_ruleset(&dex, &ls, &sets_json, castle);
        if v["ok"] != Value::Bool(true) {
            sup_illegal_teams += 1;
            let mut bad_slots: Vec<u64> = v["findings"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|f| f["severity"] == "error")
                .filter_map(|f| f["slot"].as_u64())
                .collect();
            bad_slots.sort_unstable();
            bad_slots.dedup();
            sup_illegal_sets += bad_slots.len();
        }
    }
    assert_eq!(sup_illegal_teams, 17, "expected 17/43 Castle-illegal teams in belief_supplement_v1");
    assert_eq!(sup_illegal_sets, 23, "expected 23 Castle-illegal sets in belief_supplement_v1");
}

#[test]
fn blacklist_fifth_column_min_level_vs_permanent_ban() {
    let (dex, ls) = load_data();
    let castle = Ruleset::castle_tournament();

    // 1. Level-gated 51..55 boundary: Omastar Ancient Power + Seismic Toss requires Lv >= 54
    let omastar_53 = pad_team(vec![base_mon("Omastar", 53, &["Ancient Power", "Seismic Toss"], "")]);
    let v53 = validate_team_with_ruleset(&dex, &ls, &omastar_53, castle);
    assert_eq!(v53["ok"], false, "{v53}");
    assert!(
        finding_codes(&v53)
            .iter()
            .any(|(sev, c)| sev == "error" && c == "move-combo-level"),
        "{v53}"
    );

    let omastar_54 = pad_team(vec![base_mon("Omastar", 54, &["Ancient Power", "Seismic Toss"], "")]);
    let v54 = validate_team_with_ruleset(&dex, &ls, &omastar_54, castle);
    assert_eq!(v54["ok"], true, "{v54}");

    // 2. Level-gated > 55: Kabutops Ancient Power + Swords Dance requires Lv >= 65
    let kabutops_55 = pad_team(vec![base_mon("Kabutops", 55, &["Ancient Power", "Swords Dance"], "")]);
    let vk = validate_team_with_ruleset(&dex, &ls, &kabutops_55, castle);
    assert_eq!(vk["ok"], false, "{vk}");
    assert!(
        finding_codes(&vk)
            .iter()
            .any(|(sev, c)| sev == "error" && c == "move-combo-level"),
        "{vk}"
    );

    // 3. Level-gated <= 50: Jolteon Agility + Baton Pass + Substitute is LEGAL at Lv50
    let jolteon_50 = pad_team(vec![base_mon(
        "Jolteon",
        50,
        &["Agility", "Baton Pass", "Substitute", "Thunderbolt"],
        "",
    )]);
    let vj = validate_team_with_ruleset(&dex, &ls, &jolteon_50, castle);
    assert_eq!(vj["ok"], true, "{vj}");

    // 4. Permanently banned 2-move combo (col5 == 256): Machamp Encore + Rock Slide at Lv55
    let machamp_55 = pad_team(vec![base_mon("Machamp", 55, &["Encore", "Rock Slide"], "")]);
    let vm = validate_team_with_ruleset(&dex, &ls, &machamp_55, castle);
    assert_eq!(vm["ok"], false, "{vm}");
    assert!(
        finding_codes(&vm)
            .iter()
            .any(|(sev, c)| sev == "error" && c == "move-combo-illegal"),
        "{vm}"
    );

    // 5. Permanently banned 3-move combo (col5 == 256): Omastar Haze + Slam + Aurora Beam
    let omastar_3 = pad_team(vec![base_mon(
        "Omastar",
        55,
        &["Haze", "Slam", "Aurora Beam"],
        "",
    )]);
    let v3 = validate_team_with_ruleset(&dex, &ls, &omastar_3, castle);
    assert_eq!(v3["ok"], false, "{v3}");
    assert!(
        finding_codes(&v3)
            .iter()
            .any(|(sev, c)| sev == "error" && c == "move-combo-illegal"),
        "{v3}"
    );
}

#[test]
fn jp_domestic_events_nypc_exclusions_ohko_and_level_floors() {
    let (dex, ls) = load_data();
    let castle = Ruleset::castle_tournament();

    // JP domestic events: Castle accepts, PS rejects
    for (sp, lv, mv) in [
        ("Donphan", 50, "Encore"),
        ("Dragonite", 55, "Sunny Day"),
        ("Pikachu", 50, "Fly"),
        ("Porygon", 50, "Conversion 2"),
        ("Wartortle", 50, "Roar"),
        ("Rapidash", 50, "Pay Day"),
    ] {
        let t = pad_team(vec![base_mon(sp, lv, &[mv], "")]);
        let vc = validate_team_with_ruleset(&dex, &ls, &t, castle);
        let vp = validate_team(&dex, &ls, &t);
        assert_eq!(vc["ok"], true, "Castle should allow {sp} {mv}: {vc}");
        assert_eq!(vp["ok"], false, "PS should reject {sp} {mv}: {vp}");
    }

    // NYPC moves: PS accepts, Castle rejects
    for (sp, mv) in [
        ("Snorlax", "Lovely Kiss"),
        ("Espeon", "Growth"),
        ("Misdreavus", "Hypnosis"),
        ("Forretress", "Substitute"),
        ("Heracross", "Seismic Toss"),
    ] {
        let t = pad_team(vec![base_mon(sp, 50, &[mv], "")]);
        let vc = validate_team_with_ruleset(&dex, &ls, &t, castle);
        let vp = validate_team(&dex, &ls, &t);
        assert_eq!(vc["ok"], false, "Castle should reject NYPC {sp} {mv}: {vc}");
        assert!(
            finding_codes(&vc).iter().any(|(_, c)| c == "move-illegal"),
            "{vc}"
        );
        assert_eq!(vp["ok"], true, "PS should allow {sp} {mv}: {vp}");
    }

    // OHKO moves rejected in NC2000 Castle no-OHKO
    let ohko = pad_team(vec![base_mon("Snorlax", 55, &["Fissure", "Rest"], "")]);
    let vo = validate_team_with_ruleset(&dex, &ls, &ohko, castle);
    assert_eq!(vo["ok"], false, "{vo}");
    assert!(
        finding_codes(&vo).iter().any(|(s, c)| s == "error" && c == "move-ohko"),
        "{vo}"
    );

    // Evolution level floor (`levelLowerLimit.csv`): Tyranitar / Dragonite require Lv55
    let ttar_54 = pad_team(vec![base_mon("Tyranitar", 54, &["Rock Slide", "Crunch"], "")]);
    let vt = validate_team_with_ruleset(&dex, &ls, &ttar_54, castle);
    assert_eq!(vt["ok"], false, "{vt}");
    assert!(
        finding_codes(&vt)
            .iter()
            .any(|(s, c)| s == "error" && c == "species-underleveled"),
        "{vt}"
    );
}

#[test]
fn ev_rules_authoritative_checks_and_java_bug_fix() {
    let (dex, ls) = load_data();
    let castle = Ruleset::castle_tournament();
    let rules = CastleRules::embedded();

    // Direct unit checks against authoritative ev_rules.txt
    assert_eq!(
        rules.check_effort_levels("snorlax", &["bodyslam"], 55, [63, 63, 63, 63, 63]),
        EffortVerdict::Allowed
    );
    assert_eq!(
        rules.check_effort_levels("snorlax", &["bodyslam"], 55, [0, 0, 0, 0, 0]),
        EffortVerdict::Allowed
    );
    // Fixes EffortLevelCheck.java:164-171 copy-paste bug (`X-63-Y-63-63`, X>=30, Y>=21)
    assert_eq!(
        rules.check_effort_levels("snorlax", &["bodyslam"], 55, [30, 63, 21, 63, 63]),
        EffortVerdict::Allowed
    );
    // ev_rules.txt:649 (`X-63-63-Y-63`, X>=42, Y>=24)
    assert_eq!(
        rules.check_effort_levels("raikou", &["thunderbolt"], 55, [42, 63, 63, 24, 63]),
        EffortVerdict::Allowed
    );
    // ev_rules.txt:755 Wobbuffet spreads
    assert_eq!(
        rules.check_effort_levels("wobbuffet", &["counter"], 55, [62, 0, 15, 20, 17]),
        EffortVerdict::Allowed
    );
    assert_eq!(
        rules.check_effort_levels("wobbuffet", &["counter"], 55, [63, 0, 46, 20, 17]),
        EffortVerdict::Allowed
    );
    // Gen 1 vs Gen 2 breed split for 63-0-63-35-63 (Gen1 >= 33 OK, Gen2 >= 39 required)
    assert_eq!(
        rules.check_effort_levels("zapdos", &["thunderbolt"], 55, [63, 0, 63, 35, 63]),
        EffortVerdict::Allowed
    );
    assert!(matches!(
        rules.check_effort_levels("raikou", &["thunderbolt"], 55, [63, 0, 63, 35, 63]),
        EffortVerdict::Forbidden { min_effort: 39, .. }
    ));
    // Gen 2 move forces breedGen=2 on Gen 1 species (Gengar + Perish Song, Nidoking + Beat Up typo fix)
    assert_eq!(rules.breed_gen("gengar", &["perishsong"], 50), 2);
    assert_eq!(rules.breed_gen("nidoking", &["beatup"], 50), 2);
    // Poliwrath Mind Reader level exemption: Gen 1 legal at Lv51+, Gen 2 required below Lv51
    assert_eq!(rules.breed_gen("poliwrath", &["mindreader"], 50), 2);
    assert_eq!(rules.breed_gen("poliwrath", &["mindreader"], 51), 1);

    // End-to-end validator tests on EV spreads:
    // 1. Impossible Def effort 20 (`63-63-20-63-63`) -> error
    let mut bad_def = base_mon("Snorlax", 55, &["Body Slam", "Rest"], "");
    bad_def["evs"] = json!({"hp": 255, "atk": 255, "def": 80, "spa": 255, "spd": 255, "spe": 255});
    let r = validate_team_with_ruleset(&dex, &ls, &pad_team(vec![bad_def]), castle);
    assert_eq!(r["ok"], false, "{r}");
    assert!(
        finding_codes(&r)
            .iter()
            .any(|(s, c)| s == "error" && c == "ev-effort-impossible"),
        "{r}"
    );

    // 2. Bug-fixed HP+Def spread (`35-63-25-63-63`) -> clean OK (zero errors, zero warnings)
    let mut hp_def = base_mon("Snorlax", 55, &["Body Slam", "Rest"], "");
    hp_def["evs"] = json!({"hp": 140, "atk": 255, "def": 100, "spa": 255, "spd": 255, "spe": 255});
    let r = validate_team_with_ruleset(&dex, &ls, &pad_team(vec![hp_def]), castle);
    assert_eq!(r["ok"], true, "{r}");
    assert_eq!(r["warnings"], 0, "{r}");

    // 3. Unverified custom spread (`50-10-40-40-40`) -> warning, not error
    let mut custom_ev = base_mon("Snorlax", 55, &["Body Slam", "Rest"], "");
    custom_ev["evs"] = json!({"hp": 200, "atk": 40, "def": 160, "spa": 160, "spd": 160, "spe": 160});
    let r = validate_team_with_ruleset(&dex, &ls, &pad_team(vec![custom_ev]), castle);
    assert_eq!(r["ok"], true, "{r}");
    assert!(
        finding_codes(&r)
            .iter()
            .any(|(s, c)| s == "warning" && c == "ev-effort-unverified"),
        "{r}"
    );

    // 4. All-zero EVs (`0-0-0-0-0`) is legal in Castle and preserved by Castle canonicalization
    let mut zero_ev = base_mon("Snorlax", 55, &["Body Slam", "Rest"], "");
    zero_ev["evs"] = json!({"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0});
    let raw = pad_team(vec![zero_ev]);
    let vc = validate_team_with_ruleset(&dex, &ls, &raw, castle);
    assert_eq!(vc["ok"], true, "{vc}");
    let cc = canonicalize_team_with_ruleset(&dex, &ls, &raw, castle);
    assert_eq!(cc["ok"], true, "{cc}");
    assert_eq!(cc["team"][0]["evs"]["hp"], json!(0));
    // PS still rejects all-zero EVs
    let vp = validate_team(&dex, &ls, &raw);
    assert_eq!(vp["ok"], false, "{vp}");
}

#[test]
fn castle_inert_item_warnings_never_silently_rewrite_or_collide() {
    let (dex, ls) = load_data();
    let castle = Ruleset::castle_tournament();

    // Team holding BOTH Bitter Berry + Ice Berry AND Dragon Fang + Dragon Scale.
    // A buggy auto-rewrite would collide into duplicate Ice Berry & Dragon Scale!
    let sets = vec![
        base_mon("Snorlax", 55, &["Body Slam", "Rest"], "Bitter Berry"),
        base_mon("Zapdos", 50, &["Thunderbolt", "Rest"], "Ice Berry"),
        base_mon("Dragonite", 55, &["Dragon Breath", "Rest"], "Dragon Fang"),
        base_mon("Kingdra", 50, &["Surf", "Dragon Breath"], "Dragon Scale"),
        base_mon("Cloyster", 50, &["Surf", "Spikes"], "Leftovers"),
        base_mon("Umbreon", 50, &["Toxic", "Rest"], "Gold Berry"),
    ];
    let raw = serde_json::to_string(&sets).unwrap();

    let v = validate_team_with_ruleset(&dex, &ls, &raw, castle);
    assert_eq!(v["ok"], true, "inert items are warnings, not errors: {v}");
    assert_eq!(v["warnings"], 2, "{v}");
    let inert: Vec<&Value> = v["findings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["severity"] == "warning" && f["code"] == "item-castle-inert")
        .collect();
    assert_eq!(inert.len(), 2, "{v}");
    assert_eq!(inert[0]["item"], "bitterberry");
    assert_eq!(inert[0]["pbsId"], 84);
    assert_eq!(inert[0]["suggested"], "iceberry");
    assert_eq!(inert[1]["item"], "dragonfang");
    assert_eq!(inert[1]["pbsId"], 145);
    assert_eq!(inert[1]["suggested"], "dragonscale");

    // Canonicalization MUST preserve Bitter Berry & Dragon Fang untouched!
    let c = canonicalize_team_with_ruleset(&dex, &ls, &raw, castle);
    assert_eq!(c["ok"], true, "{c}");
    assert_eq!(c["team"][0]["item"], "Bitter Berry");
    assert_eq!(c["team"][1]["item"], "Ice Berry");
    assert_eq!(c["team"][2]["item"], "Dragon Fang");
    assert_eq!(c["team"][3]["item"], "Dragon Scale");
    assert_eq!(c["warnings"].as_array().unwrap().len(), 2);
}

#[test]
fn castle_default_canonicalization_and_pp_cap() {
    let (dex, ls) = load_data();
    let castle = Ruleset::castle_tournament();

    // Bare Lv50 Snorlax with unspecified EVs, IVs, gender
    let bare = json!({
        "species": "Snorlax",
        "level": 50,
        "moves": ["Body Slam", "Rest", "Curse", "Earthquake"]
    });
    // Mismatched HP DV & SpD != SpA + omitted gender -> derived HP DV, SpD:=SpA, Atk-DV gender
    let mismatched = json!({
        "species": "Zapdos",
        "level": 50,
        "moves": ["Thunderbolt", "Hidden Power Fire", "Rest"],
        "ivs": {"hp": 0, "atk": 28, "def": 24, "spa": 30, "spd": 10, "spe": 30}
    });
    let raw = pad_team(vec![bare, mismatched]);
    let c = canonicalize_team_with_ruleset(&dex, &ls, &raw, castle);
    assert_eq!(c["ok"], true, "{c}");
    let mon = &c["team"][0];
    assert_eq!(
        mon["evs"],
        json!({"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255})
    );
    assert_eq!(mon["ivs"]["hp"].as_i64().unwrap() / 2, 15);
    assert_eq!(mon["ivs"]["spa"], mon["ivs"]["spd"]);
    assert_eq!(mon["gender"], json!("M"));

    let mon1 = &c["team"][1];
    // Atk DV 14 (even), Def DV 12 (even), Spe DV 15 (odd), Spc DV 15 (odd) -> HP DV = 0+0+2+1 = 3 -> IV 6
    assert_eq!(mon1["ivs"]["hp"], json!(6));
    assert_eq!(mon1["ivs"]["spd"], json!(30));
    assert_eq!(mon1["gender"], json!("N"));

    // Castle max PP: base-40 PP moves reach 64 with 3 PP Ups; Sketch stays 1
    assert_eq!(castle_max_pp(40, 3, false), 64);
    assert_eq!(castle_max_pp(40, 0, false), 40);
    assert_eq!(castle_max_pp(15, 3, false), 24);
    assert_eq!(castle_max_pp(1, 3, true), 1);

    // PS canonicalize_team backward compatibility check
    let ps_c = canonicalize_team(&dex, &ls, &raw);
    assert_eq!(ps_c["ok"], true, "{ps_c}");
}

#[test]
fn replay_roundtrips_castle_and_preserves_ps_bit_exact() {
    let (dex, _) = load_data();
    let mon = |species: &str| PokemonSet {
        name: species.into(),
        species: species.into(),
        item: "Leftovers".into(),
        ability: String::new(),
        moves: vec!["Body Slam".into(), "Rest".into()],
        level: 50,
        evs: None,
        ivs: None,
        happiness: Some(255),
        gender: Some("M".into()),
    };
    let team = vec![mon("Snorlax"), mon("Cloyster"), mon("Marowak")];
    let seed = "12,34,56,78";

    for ruleset in [
        Ruleset::PsStadium2,
        Ruleset::castle_tournament(),
        Ruleset::castle_raw_cgi(false),
        Ruleset::castle_raw_cgi(true),
        Ruleset::castle_free(),
        Ruleset::CastlePbs(CastleOverlayConfig {
            recoil_1v1_attacker_wins: false,
            perish_dbond_lastmon_forfeit: true,
            turn_limit: Some(50),
            overflow_mode: true,
            learnset_policy: LearnsetPolicy::PsGen2Stadium2,
        }),
    ] {
        let mut battle =
            Battle::from_fixture_with_ruleset(&dex, seed, &team, &team, ruleset).unwrap();
        let mut rec = Recorder::new(&dex, &battle, [team.clone(), team.clone()], seed);
        assert_eq!(rec.replay.ruleset, ruleset);
        for _ in 0..4 {
            if battle.ended {
                break;
            }
            let needs = battle.needs_choice();
            for side in 0..2 {
                if needs[side] {
                    let pick = battle.legal_choices(&dex, side)[0].to_input(&dex);
                    rec.choose(&dex, &mut battle, side, &pick).unwrap();
                }
            }
        }
        let code = rec.replay.encode(&dex).unwrap();
        assert!(code.starts_with("NC2-"));
        let decoded = Replay::decode(&dex, &code).unwrap();
        assert_eq!(decoded.ruleset, ruleset);
        assert_eq!(decoded.encode(&dex).unwrap(), code);
        let replayed = decoded.at(&dex, decoded.rounds.len()).unwrap();
        assert_eq!(replayed.ruleset, ruleset);
        assert_eq!(replayed.state_key128(), battle.state_key128());
        assert_eq!(replayed.prng.seed_str(), battle.prng.seed_str());
        assert_eq!(replayed.log, battle.log);
    }
}

#[test]
fn castle_validate_audit_fixes_items_evs_tyrogue_level_sum_and_replay_bare() {
    let (dex, ls) = load_data();
    let castle = Ruleset::castle_tournament();

    // 1. Hitmonchan + High Jump Kick Tyrogue evolution warning (typo fix "hijumpkick" -> "highjumpkick").
    {
        let rules = CastleRules::embedded();
        // Atk DV 14 != Def DV 15 -> Hitmonchan + High Jump Kick must trigger Tyrogue warning!
        assert!(rules.needs_tyrogue_warning(
            "hitmonchan",
            &["highjumpkick"],
            50,
            14,
            15,
            [63, 63, 63, 63, 63]
        ));
        let mut chan = base_mon("Hitmonchan", 50, &["High Jump Kick", "Rest"], "");
        chan["ivs"] = json!({"hp": 14, "atk": 28, "def": 30, "spa": 30, "spd": 30, "spe": 30});
        let r = validate_team_with_ruleset(&dex, &ls, &pad_team(vec![chan]), castle);
        assert_eq!(r["ok"], true, "{r}");
        assert!(
            finding_codes(&r)
                .iter()
                .any(|(sev, c)| sev == "warning" && c == "ev-tyrogue-evolution"),
            "Hitmonchan + High Jump Kick must emit ev-tyrogue-evolution warning: {r}"
        );
    }

    // 2. Partial EV map in Castle defaults omitted keys to 255 (not 0).
    {
        let mut partial = base_mon("Snorlax", 55, &["Body Slam", "Rest"], "");
        partial["evs"] = json!({"hp": 140, "def": 100}); // omitted atk/spa/spd/spe default to 255 -> 35-63-25-63-63 (legal!)
        let raw = pad_team(vec![partial]);
        let v = validate_team_with_ruleset(&dex, &ls, &raw, castle);
        assert_eq!(v["ok"], true, "Partial EV map should default omitted keys to 255: {v}");
        assert_eq!(v["warnings"], 0, "{v}");
        let c = canonicalize_team_with_ruleset(&dex, &ls, &raw, castle);
        assert_eq!(c["ok"], true, "{c}");
        assert_eq!(
            c["team"][0]["evs"],
            json!({"hp": 140, "atk": 255, "def": 100, "spa": 255, "spd": 255, "spe": 255})
        );
    }

    // 3. `level-sum-highest` gated off in Castle (55 + five 51s: lowest-3 = 153 <= 155 -> legal in Castle, error in PS).
    {
        let sets = vec![
            base_mon("Snorlax", 55, &["Body Slam", "Rest"], ""),
            base_mon("Zapdos", 51, &["Thunderbolt", "Rest"], ""),
            base_mon("Suicune", 51, &["Surf", "Rest"], ""),
            base_mon("Cloyster", 51, &["Surf", "Spikes"], ""),
            base_mon("Marowak", 51, &["Earthquake", "Rest"], ""),
            base_mon("Umbreon", 51, &["Toxic", "Rest"], ""),
        ];
        let raw = serde_json::to_string(&sets).unwrap();
        let vc = validate_team_with_ruleset(&dex, &ls, &raw, castle);
        assert_eq!(vc["ok"], true, "55 + 5x51 must be legal in Castle: {vc}");
        let vp = validate_team(&dex, &ls, &raw);
        assert_eq!(vp["ok"], false, "55 + 5x51 must fail level-sum-highest in PS: {vp}");
        assert!(finding_codes(&vp).iter().any(|(_, c)| c == "level-sum-highest"));
    }

    // 4. `item-castle-inert` warnings fire for CastlePbs even under LearnsetPolicy::PsGen2Stadium2!
    {
        let castle_ps_ls = Ruleset::CastlePbs(CastleOverlayConfig {
            learnset_policy: LearnsetPolicy::PsGen2Stadium2,
            ..CastleOverlayConfig::tournament()
        });
        let raw = pad_team(vec![
            base_mon("Snorlax", 55, &["Lovely Kiss", "Rest"], "Bitter Berry"),
            base_mon("Dragonite", 55, &["Dragon Breath", "Rest"], "Dragon Fang"),
        ]);
        let v = validate_team_with_ruleset(&dex, &ls, &raw, castle_ps_ls);
        assert_eq!(v["ok"], true, "PsGen2Stadium2 learnset allows Lovely Kiss: {v}");
        assert_eq!(v["warnings"], 2, "Inert item warnings must still fire under CastlePbs+PsGen2Stadium2: {v}");
    }

    // 5. All 11 Castle inert items: alias dedup (`mail`+`flowermail`, `sportball`+`parkball`) vs distinct mails!
    {
        let distinct = vec![
            base_mon("Snorlax", 55, &["Body Slam", "Rest"], "Surf Mail"),
            base_mon("Zapdos", 50, &["Thunderbolt", "Rest"], "Eon Mail"),
            base_mon("Suicune", 50, &["Surf", "Rest"], "Flower Mail"),
            base_mon("Cloyster", 50, &["Surf", "Spikes"], "Park Ball"),
            base_mon("Marowak", 50, &["Earthquake", "Rest"], "Mirage Mail"),
            base_mon("Umbreon", 50, &["Toxic", "Rest"], "Blue Sky Mail"),
        ];
        let raw_ok = serde_json::to_string(&distinct).unwrap();
        let v_ok = validate_team_with_ruleset(&dex, &ls, &raw_ok, castle);
        assert_eq!(v_ok["ok"], true, "Distinct Castle inert items must validate cleanly: {v_ok}");
        let c_ok = canonicalize_team_with_ruleset(&dex, &ls, &raw_ok, castle);
        assert_eq!(c_ok["ok"], true, "{c_ok}");
        assert_eq!(c_ok["team"][0]["item"], "Surf Mail");
        assert_eq!(c_ok["team"][1]["item"], "Eon Mail");
        assert_eq!(c_ok["team"][2]["item"], "Mail");
        assert_eq!(c_ok["team"][3]["item"], "Sport Ball");

        // Duplicate aliases caught by Item Clause: mail + flowermail AND sportball + parkball!
        let dup_mail = pad_team(vec![
            base_mon("Snorlax", 55, &["Body Slam", "Rest"], "Mail"),
            base_mon("Zapdos", 50, &["Thunderbolt", "Rest"], "Flower Mail"),
        ]);
        let vm = validate_team_with_ruleset(&dex, &ls, &dup_mail, castle);
        assert_eq!(vm["ok"], false, "mail + flowermail must trigger item-clause: {vm}");
        assert!(finding_codes(&vm).iter().any(|(_, c)| c == "item-clause"));

        let dup_ball = pad_team(vec![
            base_mon("Snorlax", 55, &["Body Slam", "Rest"], "Sport Ball"),
            base_mon("Zapdos", 50, &["Thunderbolt", "Rest"], "Park Ball"),
        ]);
        let vb = validate_team_with_ruleset(&dex, &ls, &dup_ball, castle);
        assert_eq!(vb["ok"], false, "sportball + parkball must trigger item-clause: {vb}");
        assert!(finding_codes(&vb).iter().any(|(_, c)| c == "item-clause"));
    }

    // 6. Bare Castle Replay::encode (evs: None, ivs: None, typed HP Fire, extra mails) round-trips 100% identically!
    {
        let bare_team = vec![
            PokemonSet {
                name: "Snorlax".into(),
                species: "Snorlax".into(),
                item: "Surf Mail".into(),
                ability: String::new(),
                moves: vec!["Body Slam".into(), "Rest".into()],
                level: 50,
                evs: None,
                ivs: None,
                happiness: None,
                gender: None,
            },
            PokemonSet {
                name: "Zapdos".into(),
                species: "Zapdos".into(),
                item: "Eon Mail".into(),
                ability: String::new(),
                moves: vec!["Thunderbolt".into(), "Hidden Power Fire".into()],
                level: 50,
                evs: None,
                ivs: None,
                happiness: None,
                gender: None,
            },
            PokemonSet {
                name: "Cloyster".into(),
                species: "Cloyster".into(),
                item: "Park Ball".into(),
                ability: String::new(),
                moves: vec!["Surf".into(), "Spikes".into()],
                level: 50,
                evs: None,
                ivs: None,
                happiness: None,
                gender: None,
            },
        ];
        let replay = Replay {
            teams: [bare_team.clone(), bare_team],
            seed: "7,8,9,10".into(),
            caps: [None, None],
            open: true,
            bot_side: 1,
            ruleset: castle,
            rounds: Vec::new(),
        };
        let orig_b = replay.initial(&dex).unwrap();
        assert_eq!(orig_b.sides[0].roster[0].maxhp, 266);
        let code = replay.encode(&dex).unwrap();
        let decoded = Replay::decode(&dex, &code).unwrap();
        assert_eq!(decoded.teams[0][0].item, "surfmail");
        assert_eq!(decoded.teams[0][1].item, "eonmail");
        assert_eq!(decoded.teams[0][2].item, "sportball");
        let dec_b = decoded.initial(&dex).unwrap();
        assert_eq!(dec_b.sides[0].roster[0].maxhp, 266);
        assert_eq!(dec_b.sides[0].roster[1].set_ivs, orig_b.sides[0].roster[1].set_ivs);
        assert_eq!(dec_b.state_key128(), orig_b.state_key128());
        assert_eq!(decoded.encode(&dex).unwrap(), code);
    }

    // 7. Ruleset::canonical_id & Ruleset::parse_id custom round-trip (with & without -l{jp|ps})!
    {
        for cfg in [
            CastleOverlayConfig {
                recoil_1v1_attacker_wins: false,
                perish_dbond_lastmon_forfeit: true,
                turn_limit: Some(50),
                overflow_mode: true,
                learnset_policy: LearnsetPolicy::PsGen2Stadium2,
            },
            CastleOverlayConfig {
                recoil_1v1_attacker_wins: true,
                perish_dbond_lastmon_forfeit: false,
                turn_limit: None,
                overflow_mode: false,
                learnset_policy: LearnsetPolicy::CastleJpEvents,
            },
        ] {
            let r = Ruleset::CastlePbs(cfg);
            let cid = r.canonical_id();
            assert_eq!(Ruleset::parse_id(&cid), Some(r), "failed round-trip for {cid}");
        }
        // Backward-compatible parsing without -l...
        let legacy = Ruleset::parse_id("castle-custom-r0-p1-t50-o1").unwrap();
        assert_eq!(
            legacy,
            Ruleset::CastlePbs(CastleOverlayConfig {
                recoil_1v1_attacker_wins: false,
                perish_dbond_lastmon_forfeit: true,
                turn_limit: Some(50),
                overflow_mode: true,
                learnset_policy: LearnsetPolicy::CastleJpEvents,
            })
        );
    }
}
