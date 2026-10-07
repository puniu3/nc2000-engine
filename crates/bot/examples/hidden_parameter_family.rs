use nc2000_bot::Observer;
use nc2000_engine::{
    battle::PokemonSet,
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

#[allow(dead_code)]
mod certificate {
    include!("hidden_terminal.rs");
    pub fn solve(dex: &Dex, battle: Battle, config: Value) -> Value {
        finite::solve(dex, battle, config)
    }
}

fn digest(text: &str) -> String {
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus plan output");
    let plan: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let case = &plan["case"];
    let game: Value = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|g| g["game"] == case["game"])
        .unwrap();
    let side = case["side"].as_u64().unwrap() as usize;
    let slot = case["foe_slot"].as_u64().unwrap() as usize;
    let decision = case["decision"].as_u64().unwrap() as usize;
    let dex = conformance::load_dex();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let original: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let seed = game["seed"].as_str().unwrap();
    let mut battle = Battle::from_fixture(&dex, seed, &original[0], &original[1]).unwrap();
    battle.set_log_enabled(true);
    let mut obs = Observer::new(&battle, side);
    let mut prefixes = Vec::new();
    for step in 0..=decision {
        assert_eq!(
            battle.state_key128().to_string(),
            game["decisions"][step]["key"].as_str().unwrap()
        );
        let actions = [0, 1].map(|s| battle.legal_choices(&dex, s));
        prefixes.push((battle.clone(), obs.clone(), actions.clone()));
        if step < decision {
            battle
                .apply_choices(
                    &dex,
                    [0, 1].map(|s| {
                        game["decisions"][step]["choices"][s]
                            .as_u64()
                            .map(|i| actions[s][i as usize])
                    }),
                )
                .unwrap();
            obs.observe(&battle, &dex);
        }
    }
    let observation = &obs.mons()[slot];
    let observation = json!({"moves":observation.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),
        "item":format!("{:?}",observation.item),"preview_has_item":observation.preview_has_item,"hp_type_mask":observation.hp_type_mask});
    let mut rows = Vec::new();
    for request in plan["rows"].as_array().unwrap() {
        let mut teams = original.clone();
        let mut set = serde_json::to_value(&teams[1 - side][slot]).unwrap();
        for (field, value) in request["patch"].as_object().unwrap() {
            set[field] = value.clone();
        }
        teams[1 - side][slot] = serde_json::from_value(set).unwrap();
        if let Some(other) = request["swapped_slot"].as_u64() {
            assert_ne!(other as usize, slot);
            teams[1 - side][other as usize].item = original[1 - side][slot].item.clone();
        }
        let validity = validate_team(
            &dex,
            &learnsets,
            &serde_json::to_string(&teams[1 - side]).unwrap(),
        );
        let mut row = json!({"request":request,"validity":validity});
        if validity["ok"] != true {
            rows.push(row);
            continue;
        }
        let mut alternate = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
        alternate.set_log_enabled(true);
        let mut alternate_obs = Observer::new(&alternate, side);
        let mut checks = 0;
        for step in 0..=decision {
            let (base, base_obs, actions) = &prefixes[step];
            let reason = if alternate.log != base.log {
                Some("log")
            } else if format!("{:?}", alternate_obs.mons()) != format!("{:?}", base_obs.mons()) {
                Some("observer")
            } else if alternate.legal_choices(&dex, side) != actions[side] {
                Some("own_actions")
            } else {
                None
            };
            if let Some(reason) = reason {
                let line = base
                    .log
                    .iter()
                    .zip(&alternate.log)
                    .position(|(a, b)| a != b)
                    .unwrap_or(base.log.len().min(alternate.log.len()));
                row["mismatch"] = json!({"decision":step,"reason":reason,"line":line,"original":base.log.get(line),"alternate":alternate.log.get(line)});
                break;
            }
            checks += 1;
            if step == decision {
                break;
            }
            let chosen = [0, 1].map(|s| {
                game["decisions"][step]["choices"][s]
                    .as_u64()
                    .map(|i| actions[s][i as usize])
            });
            if [0, 1]
                .iter()
                .any(|&s| chosen[s].is_some_and(|a| !alternate.legal_choices(&dex, s).contains(&a)))
            {
                row["mismatch"] = json!({"decision":step,"reason":"recorded_action"});
                break;
            }
            alternate.apply_choices(&dex, chosen).unwrap();
            alternate_obs.observe(&alternate, &dex);
        }
        row["prefix_checks"] = json!(checks);
        if checks == decision + 1 && row.get("mismatch").is_none() {
            let id = alternate.active_id(1 - side).unwrap();
            let mon = alternate.poke(id);
            row["root"] = json!({"key":alternate.state_key128().to_string(),"log_hash":digest(&serde_json::to_string(&alternate.log).unwrap()),
                "observer_hash":digest(&format!("{:?}",alternate_obs.mons())),"stats":mon.base_stored_stats,"ivs":mon.set_ivs,"evs":mon.set_evs,
                "hp_type":dex.type_name(mon.base_hp_type),"hp_power":mon.base_hp_power,"hp":mon.hp,"maxhp":mon.maxhp,
                "item":mon.item.map(|i|dex.items.key(i)),"state":alternate.essence(&dex)});
            if request["prove"] == true {
                let mut ledgers = Vec::new();
                for (target, first, next) in [
                    (side, "rockslide", "rockslide"),
                    (side, "swordsdance", "rockslide"),
                    (1 - side, "sleeptalk", "hiddenpowerice"),
                ] {
                    let mut config = json!({"game":case["game"],"decision":case["decision"],"target_side":target,"policy_side":target,
                        "first_choice":format!("move {first}"),"priority":[format!("move {next}")],"rest_at_percent":null,
                        "horizon":3,"work_budget":200000,"cell_cap":10000,"guarded_keys":true,"support_certificate":true});
                    if first == "sleeptalk" {
                        config["opening"] =
                            json!(["move sleeptalk", "move sleeptalk", "move hiddenpowerice"]);
                    }
                    let ledger = certificate::solve(&dex, alternate.clone(), config);
                    eprintln!(
                        "{} {first}: {} {}",
                        request["id"], ledger["runs"], ledger["bounds"]
                    );
                    ledgers.push(json!({"first":first,"ledger":ledger}));
                }
                row["ledgers"] = json!(ledgers);
            }
        }
        rows.push(row);
    }
    std::fs::write(
        &args[3],
        serde_json::to_vec(
            &json!({"case":case,"observation":observation,"original_log":battle.log,"rows":rows}),
        )
        .unwrap(),
    )
    .unwrap();
}
