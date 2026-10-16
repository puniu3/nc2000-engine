//! Offline proposal; never rewrites the shipped catalog, inventory or prior.
//! cargo run -p nc2000-engine --example low_attack_catalog_audit --
//!   data/team-pool-v2/team-pool.json NEW_OUTPUT_DIRECTORY
use nc2000_engine::{
    battle::{
        moveexec::{get_damage_synthetic, synthetic_move},
        PokemonSet,
    },
    dex::{toid, Dex},
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, path::Path};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn iv(set: &Value, key: &str) -> i64 {
    set["ivs"][key].as_i64().unwrap_or(31) & 30
}
fn shiny(set: &Value, atk: i64) -> bool {
    ["def", "spe", "spa"].iter().all(|k| iv(set, k) / 2 == 10) && atk % 4 >= 2
}
fn gender(species: &Value, atk: i64) -> String {
    species["gender"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| {
            let female = species["genderRatio"]["F"].as_f64().unwrap_or(0.5);
            if (atk as f64) < female * 16.0 {
                "F"
            } else {
                "M"
            }
            .to_owned()
        })
}
fn battle(dex: &Dex, sets: &Value, lead: usize) -> Battle {
    let team: Vec<PokemonSet> = serde_json::from_value(sets.clone()).unwrap();
    let pair = (0..team.len())
        .filter(|&i| i != lead)
        .flat_map(|i| {
            let team = &team;
            (i + 1..team.len())
                .filter(move |&j| {
                    j != lead
                        && u16::from(team[lead].level)
                            + u16::from(team[i].level)
                            + u16::from(team[j].level)
                            <= 155
                })
                .map(move |j| (i, j))
        })
        .next()
        .expect("legal lead selection");
    let mut b = Battle::from_fixture(dex, "1,2,3,4", &team, &team).unwrap();
    let choice = format!("team {},{},{}", lead + 1, pair.0 + 1, pair.1 + 1);
    b.choose(dex, 0, &choice).unwrap();
    b.choose(dex, 1, &choice).unwrap();
    b.set_log_enabled(false);
    b
}
fn self_hit(dex: &Dex, b: &mut Battle, move_name: &str, atk: i8, def: i8) -> i64 {
    let p = b.active_id(0).unwrap();
    b.poke_mut(p).boosts[0] = atk;
    b.poke_mut(p).boosts[1] = def;
    let id = dex.moves.id(&toid(move_name)).unwrap();
    let mut fake = synthetic_move(dex, 40);
    // Same item/base-type handling as conditions.rs's confusion handler.
    fake.base_move_type = dex.move_static(id).move_type;
    get_damage_synthetic(b, dex, p, p, fake).unwrap() as i64
}
fn write(dir: &Path, name: &str, value: &Value) {
    fs::write(
        dir.join(name),
        serde_json::to_string_pretty(value).unwrap() + "\n",
    )
    .unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "usage: low_attack_catalog_audit INPUT_POOL NEW_OUTPUT_DIRECTORY"
    );
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let input_path = root.join(&args[1]);
    let output = root.join(&args[2]);
    assert!(
        !output.exists(),
        "output must be a new directory; existing work is never overwritten"
    );
    let input = fs::read(&input_path).unwrap();
    let pool: Value = serde_json::from_slice(&input).unwrap();
    let dex_bytes = fs::read(root.join("data/gen2stadium2.json")).unwrap();
    let dex_json: Value = serde_json::from_slice(&dex_bytes).unwrap();
    let dex = Dex::from_json(std::str::from_utf8(&dex_bytes).unwrap()).unwrap();
    let ls =
        Learnsets::from_json(&fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap())
            .unwrap();
    let mut teams = pool["teams"].as_array().unwrap().clone();
    let mut rows = Vec::new();
    let mut excluded = BTreeMap::<String, usize>::new();
    let mut changed_teams = 0;
    let mut comparisons = 0;
    let mut already_low_dv = 0;
    for team in &mut teams {
        let original = team["sets"].clone();
        let check = validate_team(&dex, &ls, &original.to_string());
        assert_eq!(check["ok"], true, "source {}: {check}", team["id"]);
        let base_id = team["id"].as_str().unwrap().to_owned();
        let mut changed = false;
        for (slot, set) in original.as_array().unwrap().iter().enumerate() {
            let moves: Vec<&str> = set["moves"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| m.as_str().unwrap())
                .collect();
            let ids: Vec<String> = moves.iter().map(|m| toid(m)).collect();
            let reason = if ids.iter().any(|m| m.starts_with("hiddenpower")) {
                Some("hidden-power")
            } else if ids.iter().any(|m| dex_json["moves"][m].is_null()) {
                panic!("unknown move in {base_id}")
            } else if ids
                .iter()
                .any(|m| dex_json["moves"][m]["category"] == "Physical")
            {
                Some("physical-move")
            } else if ids.iter().any(|m| {
                matches!(
                    m.as_str(),
                    "metronome" | "mimic" | "mirrormove" | "transform"
                )
            }) {
                Some("external-move-access")
            } else {
                None
            };
            if let Some(reason) = reason {
                *excluded.entry(reason.to_owned()).or_default() += 1;
                continue;
            }
            let species = &dex_json["species"][toid(set["species"].as_str().unwrap())];
            assert!(!species.is_null());
            let old_dv = iv(set, "atk") / 2;
            let new_dv = (0..=old_dv)
                .find(|&a| {
                    a % 2 == old_dv % 2
                        && gender(species, a) == gender(species, old_dv)
                        && shiny(set, a) == shiny(set, old_dv)
                })
                .unwrap();
            if old_dv == new_dv {
                already_low_dv += 1;
            }
            let mut proposed = set.clone();
            // These catalog records carry complete maps; refuse ambiguous defaults.
            assert!(set["ivs"].is_object() && set["evs"].is_object());
            proposed["ivs"]["atk"] = json!(new_dv * 2);
            proposed["evs"]["atk"] = json!(0);
            if new_dv == old_dv && set["evs"]["atk"] == 0 {
                continue;
            }
            let mut one_changed = original.clone();
            one_changed[slot] = proposed.clone();
            let mut before = battle(&dex, &original, slot);
            let mut after = battle(&dex, &one_changed, slot);
            let a = before.poke(before.active_id(0).unwrap());
            let z = after.poke(after.active_id(0).unwrap());
            assert_eq!(a.maxhp, z.maxhp);
            assert_eq!(a.gender, z.gender);
            assert_eq!(a.stored_stats[1..], z.stored_stats[1..]);
            assert!(z.stored_stats[0] <= a.stored_stats[0]);
            let stats_before = a.stored_stats;
            let stats_after = z.stored_stats;
            let hp = a.maxhp;
            let mut damage = Vec::new();
            for m in &moves {
                damage.push(json!({"move":m,
                    "neutralBefore":self_hit(&dex, &mut before, m, 0, 0),
                    "neutralAfter":self_hit(&dex, &mut after, m, 0, 0),
                    "swaggerBefore":self_hit(&dex, &mut before, m, 2, 0),
                    "swaggerAfter":self_hit(&dex, &mut after, m, 2, 0)}));
                for atk in -6..=6 {
                    for def in -6..=6 {
                        let old = self_hit(&dex, &mut before, m, atk, def);
                        let new = self_hit(&dex, &mut after, m, atk, def);
                        assert!(
                            new <= old,
                            "self-hit increased: {base_id}/{slot}/{m}/{atk}/{def}"
                        );
                        comparisons += 1;
                    }
                }
            }
            rows.push(
                json!({"baseId":base_id,"slot":slot+1,"species":set["species"],
                "level":set["level"],"gender":set["gender"],"hp":hp,
                "attackDvBefore":old_dv,"attackDvAfter":new_dv,
                "attackEvBefore":set["evs"]["atk"],"attackEvAfter":0,
                "attackBefore":stats_before[0],"attackAfter":stats_after[0],"damage":damage}),
            );
            team["sets"][slot] = proposed;
            changed = true;
        }
        if changed {
            changed_teams += 1;
            team["baseId"] = json!(base_id);
            team["id"] = json!(format!("{base_id}-low-atk"));
        }
        // Selection scores were measured on the original sets, never these proposals.
        if let Some(score) = team.as_object_mut().unwrap().remove("score") {
            team["baselineScore"] = score;
        }
        let check = validate_team(&dex, &ls, &team["sets"].to_string());
        assert_eq!(check["ok"], true, "candidate {}: {check}", team["id"]);
    }
    // Fail if a concurrent session changed our input during the audit.
    assert_eq!(
        input,
        fs::read(&input_path).unwrap(),
        "source changed during audit"
    );
    let candidate = json!({"format":"nc2000-low-attack-candidate-v1", "sourcePool":args[1],
        "sourcePoolSha256":sha(&input),"measurementStatus":"unmeasured candidate; baselineScore is for original sets",
        "drawRule":pool["drawRule"],"teams":teams});
    let report = json!({"format":"nc2000-low-attack-audit-v1","sourcePool":args[1],
        "sourcePoolSha256":sha(&input),"dexSha256":sha(&dex_bytes),
        "teams":teams.len(),"changedTeams":changed_teams,"changedSets":rows.len(),
        "eligibleAlreadyAtMinimumDv":already_low_dv,"excluded":excluded,
        "checks":{"sourceAndCandidateTeamsValidated":teams.len(),"selfHitComparisons":comparisons,
            "selfHitIncreases":0,"preserved":"HP, gender, shiny DV predicate, all non-Attack stats, moves, items, levels, draw weights",
            "scope":"current native engine; each selected move's base type and held item; Attack and Defense stages -6 through +6; no burn, weather or screens"},
        "limitations":["Struggle damage also decreases: no unconditional dominance or win-rate claim",
            "Hidden Power and Physical-category moves conservatively excluded, including fixed damage",
            "Metronome, Mimic, Mirror Move and Transform excluded; Sleep Talk only accesses the unchanged own moves",
            "Changing gender could affect Attract: gender is preserved",
            "Built-in validator checks flat learnsets; it does not certify every event or egg combination",
            "Source catalog, inventory, prior, Nash mixture and compiled preview artifacts remain untouched"],
        "rows":rows});
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    fs::create_dir(&output).unwrap();
    write(&output, "candidate-pool.json", &candidate);
    write(&output, "audit.json", &report);
    println!(
        "{} teams, {} changed teams, {} changed sets, {} self-hit comparisons; source untouched",
        teams.len(),
        changed_teams,
        rows.len(),
        comparisons
    );
}
