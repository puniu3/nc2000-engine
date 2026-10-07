use nc2000_bot::{
    player::{action_input, PlayerChannel, PlayerFrame},
    Observer,
};
use nc2000_engine::{
    battle::PokemonSet,
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn digest<T: serde::Serialize>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn frame(battle: &Battle, dex: &nc2000_engine::dex::Dex, side: usize) -> PlayerFrame {
    PlayerChannel::new(side)
        .frame(&mut battle.clone(), dex)
        .unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "export|screen corpus plan output");
    let plan: Value = serde_json::from_slice(&std::fs::read(&args[3]).unwrap()).unwrap();
    let case = &plan["case"];
    let game: Value = std::fs::read_to_string(&args[2])
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap())
        .find(|g| g["game"] == case["game"])
        .unwrap();
    let side = case["side"].as_u64().unwrap() as usize;
    let slot = case["foe_slot"].as_u64().unwrap() as usize;
    let decision = case["decision"].as_u64().unwrap() as usize;
    let dex = conformance::load_dex();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let seed = game["seed"].as_str().unwrap();
    let mut battle = Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
    battle.set_log_enabled(true);
    let mut obs = Observer::new(&battle, side);
    let mut prefixes = Vec::new();
    let mut tape = Vec::new();
    for step in 0..=decision {
        let actions = [0, 1].map(|s| battle.legal_choices(&dex, s));
        let inputs = actions
            .each_ref()
            .map(|a| a.iter().map(|&c| action_input(&dex, c)).collect::<Vec<_>>());
        if args[1] == "export" {
            assert_eq!(
                battle.state_key128().to_string(),
                game["decisions"][step]["key"].as_str().unwrap()
            );
        }
        let chosen: [Option<String>; 2] = if args[1] == "export" {
            [0, 1].map(|s| {
                game["decisions"][step]["choices"][s]
                    .as_u64()
                    .map(|i| inputs[s][i as usize].clone())
            })
        } else {
            serde_json::from_value(plan["tape"][step]["chosen"].clone()).unwrap()
        };
        tape.push(json!({"step":step,"key":battle.state_key128().to_string(),"chosen":chosen,"legal":inputs,"frame":frame(&battle,&dex,side),"full_log":battle.log}));
        prefixes.push((
            battle.clone(),
            frame(&battle, &dex, side),
            format!("{:?}", obs.mons()),
            actions.clone(),
        ));
        if step < decision {
            let selected = [0, 1].map(|s| {
                chosen[s].as_ref().map(|input| {
                    actions[s][inputs[s]
                        .iter()
                        .position(|v| v == input)
                        .expect("recorded action must remain legal")]
                })
            });
            battle.apply_choices(&dex, selected).unwrap();
            obs.observe(&battle, &dex);
        }
    }
    if args[1] == "export" {
        std::fs::write(
            &args[4],
            serde_json::to_vec(&json!({"case":case,"tape":tape})).unwrap(),
        )
        .unwrap();
        return;
    }
    assert_eq!(args[1], "screen");
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let mut rows = Vec::new();
    for request in plan["rows"].as_array().unwrap() {
        let mut alternate_teams = teams.clone();
        let mut set = serde_json::to_value(&alternate_teams[1 - side][slot]).unwrap();
        for (k, v) in request["patch"].as_object().unwrap() {
            set[k] = v.clone();
        }
        alternate_teams[1 - side][slot] = serde_json::from_value(set).unwrap();
        if let Some(other) = request["swapped_slot"].as_u64() {
            assert_ne!(other as usize, slot);
            alternate_teams[1 - side][other as usize].item = teams[1 - side][slot].item.clone();
        }
        let validity = validate_team(
            &dex,
            &learnsets,
            &serde_json::to_string(&alternate_teams[1 - side]).unwrap(),
        );
        let mut row = json!({"request":request,"valid":validity["ok"],"checks":0});
        if validity["ok"] != true {
            row["validity"] = validity;
            rows.push(row);
            continue;
        }
        let mut alternate =
            Battle::from_fixture(&dex, seed, &alternate_teams[0], &alternate_teams[1]).unwrap();
        alternate.set_log_enabled(true);
        let mut alternate_obs = Observer::new(&alternate, side);
        for step in 0..=decision {
            let (base, expected, expected_obs, actions) = &prefixes[step];
            let actual = frame(&alternate, &dex, side);
            if actual != *expected {
                let line = actual
                    .lines
                    .iter()
                    .zip(&expected.lines)
                    .position(|(a, b)| a != b)
                    .unwrap_or(actual.lines.len().min(expected.lines.len()));
                row["mismatch"] = json!({"step":step,"reason":if actual.lines!=expected.lines {"visible_log"} else if actual.request!=expected.request {"own_request"} else {"legal_actions"},"line":line,"expected":expected.lines.get(line),"actual":actual.lines.get(line)});
                break;
            }
            row["checks"] = json!(step + 1);
            if alternate.log != base.log && row.get("first_private_difference").is_none() {
                let line = alternate
                    .log
                    .iter()
                    .zip(&base.log)
                    .position(|(a, b)| a != b)
                    .unwrap_or(alternate.log.len().min(base.log.len()));
                row["first_private_difference"] = json!({"step":step,"line":line,"expected":base.log.get(line),"actual":alternate.log.get(line)});
            }
            if format!("{:?}", alternate_obs.mons()) != *expected_obs
                && row.get("first_observer_difference").is_none()
            {
                row["first_observer_difference"] = json!(step);
            }
            if step == decision {
                let mon = alternate.poke(alternate.active_id(1 - side).unwrap());
                row["root"] = json!({"key":alternate.state_key128().to_string(),"frame_hash":digest(&actual),"observer_hash":digest(&format!("{:?}",alternate_obs.mons())),"hp":mon.hp,"maxhp":mon.maxhp,"hp_type":dex.type_name(mon.hp_type),"hp_power":mon.hp_power,"stats":mon.base_stored_stats,"ivs":mon.set_ivs,"evs":mon.set_evs,"full_log":alternate.log});
                break;
            }
            let chosen: [Option<String>; 2] =
                serde_json::from_value(tape[step]["chosen"].clone()).unwrap();
            let selected = [0, 1].map(|s| {
                chosen[s].as_ref().map(|input| {
                    *actions[s]
                        .iter()
                        .find(|&&c| action_input(&dex, c) == *input)
                        .unwrap()
                })
            });
            if [0, 1].iter().any(|&s| {
                selected[s].is_some_and(|c| !alternate.legal_choices(&dex, s).contains(&c))
            }) {
                row["mismatch"] = json!({"step":step,"reason":"recorded_action"});
                break;
            }
            alternate.apply_choices(&dex, selected).unwrap();
            alternate_obs.observe(&alternate, &dex);
        }
        rows.push(row);
    }
    let observed = &obs.mons()[slot];
    let output = json!({"case":case,"tape":tape,"root":battle.essence(&dex),"observation":{"moves":observed.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),"hp_type_mask":observed.hp_type_mask},"rows":rows});
    std::fs::write(&args[4], serde_json::to_vec(&output).unwrap()).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ready() -> (nc2000_engine::dex::Dex, Battle) {
        let dex = conformance::load_dex();
        let mon: PokemonSet = serde_json::from_value(
            json!({"name":"Zapdos","species":"Zapdos","moves":["Thunderbolt"],"level":50}),
        )
        .unwrap();
        let team = vec![mon];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
        b.set_log_enabled(true);
        let choices = [0, 1].map(|s| Some(b.legal_choices(&dex, s)[0]));
        b.apply_choices(&dex, choices).unwrap();
        (dex, b)
    }
    fn health_line(b: &mut Battle) {
        let id = b.active_id(1).unwrap();
        let (private, public) = b.get_health(id);
        b.log.extend([
            "|split|p2".to_owned(),
            format!("|-damage|p2a: Zapdos|{private}"),
            format!("|-damage|p2a: Zapdos|{public}"),
        ]);
    }
    #[test]
    fn private_full_hp_difference_is_accepted_but_visible_damage_is_rejected() {
        let (dex, b) = ready();
        let mut a = b.clone();
        let mut other = b.clone();
        let id = other.active_id(1).unwrap();
        other.poke_mut(id).maxhp -= 1;
        other.poke_mut(id).hp -= 1;
        health_line(&mut a);
        health_line(&mut other);
        assert_ne!(a.log, other.log);
        assert_eq!(frame(&a, &dex, 0), frame(&other, &dex, 0));
        other.poke_mut(id).hp = 1;
        health_line(&mut other);
        assert_ne!(frame(&a, &dex, 0), frame(&other, &dex, 0));
    }
    #[test]
    fn own_pp_difference_is_visible_even_with_identical_logs() {
        let (dex, b) = ready();
        let mut other = b.clone();
        let id = other.active_id(0).unwrap();
        other.poke_mut(id).move_slots[0].pp -= 1;
        assert_eq!(b.log, other.log);
        assert_ne!(frame(&b, &dex, 0), frame(&other, &dex, 0));
    }
}
