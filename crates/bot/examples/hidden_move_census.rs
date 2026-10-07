use nc2000_bot::Observer;
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    dex::toid,
    state::Battle,
    validate::{validate_team, Learnsets},
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::BufRead;

fn digest(value: &str) -> String {
    format!("{:x}", Sha256::digest(value.as_bytes()))
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus tm-plan output");
    let dex = conformance::load_dex();
    let plan: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let learnsets =
        Learnsets::from_json(&std::fs::read_to_string("data/learnsets-gen2.json").unwrap())
            .unwrap();
    let mut rows = Vec::new();
    let mut attempted = 0;
    let mut rejected = std::collections::BTreeMap::<String, usize>::new();
    let mut eligible = 0;
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let original: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let seed = game["seed"].as_str().unwrap();
        let mut b = Battle::from_fixture(&dex, seed, &original[0], &original[1]).unwrap();
        b.set_log_enabled(true);
        let mut obs = [0, 1].map(|s| Observer::new(&b, s));
        let mut prefixes = Vec::new();
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
            prefixes.push((b.clone(), actions.clone(), obs.clone()));
            if b.is_idle_move_request() && b.sides.iter().all(|s| s.pokemon_left == 1) {
                for side in 0..2 {
                    if actions[side].len() < 2 {
                        continue;
                    }
                    eligible += 1;
                    let foe = b.active_id(1 - side).unwrap();
                    let slot = foe.slot as usize;
                    let set = &original[1 - side][slot];
                    let species = toid(&set.species);
                    let Some(options) = plan[&species].as_object() else {
                        continue;
                    };
                    for (move_slot, old) in set.moves.iter().enumerate() {
                        let old_id = dex.moves.id(&toid(old)).unwrap();
                        if obs[side].mons()[slot].revealed_moves.contains(&old_id) {
                            continue;
                        }
                        for (replacement, sources) in options {
                            if set.moves.iter().any(|m| toid(m) == *replacement) {
                                continue;
                            }
                            attempted += 1;
                            let mut teams = original.clone();
                            teams[1 - side][slot].moves[move_slot] = replacement.clone();
                            let validity = validate_team(
                                &dex,
                                &learnsets,
                                &serde_json::to_string(&teams[1 - side]).unwrap(),
                            );
                            if validity["ok"] != true {
                                *rejected.entry("validation".into()).or_default() += 1;
                                continue;
                            }
                            let mut alt =
                                Battle::from_fixture(&dex, seed, &teams[0], &teams[1]).unwrap();
                            alt.set_log_enabled(true);
                            let mut alt_obs = Observer::new(&alt, side);
                            let mut failed = None;
                            for step in 0..=decision {
                                let (base, aa, base_obs) = &prefixes[step];
                                if alt.log != base.log {
                                    failed = Some("log");
                                    break;
                                }
                                if format!("{:?}", alt_obs.mons())
                                    != format!("{:?}", base_obs[side].mons())
                                {
                                    failed = Some("observer");
                                    break;
                                }
                                if alt.legal_choices(&dex, side) != aa[side] {
                                    failed = Some("own_actions");
                                    break;
                                }
                                if step == decision {
                                    break;
                                }
                                let dd = &game["decisions"][step];
                                let mut choices = [0, 1]
                                    .map(|s| dd["choices"][s].as_u64().map(|i| aa[s][i as usize]));
                                if base.active_id(1 - side) == Some(foe)
                                    && choices[1 - side] == Some(SearchChoice::Move(old_id))
                                {
                                    choices[1 - side] =
                                        Some(SearchChoice::Move(dex.moves.id(replacement).unwrap()));
                                }
                                if [0, 1].iter().any(|&s| {
                                    choices[s]
                                        .is_some_and(|a| !alt.legal_choices(&dex, s).contains(&a))
                                }) {
                                    failed = Some("recorded_action");
                                    break;
                                }
                                alt.apply_choices(&dex, choices).unwrap();
                                alt_obs.observe(&alt, &dex);
                            }
                            if let Some(reason) = failed {
                                *rejected.entry(reason.into()).or_default() += 1;
                                continue;
                            }
                            assert_ne!(b.state_key128(), alt.state_key128());
                            rows.push(json!({"game":game["game"],"decision":decision,"side":side,"key":b.state_key128().to_string(),"alternate_key":alt.state_key128().to_string(),"foe_slot":slot,"species":species,"move_slot":move_slot,"original_move":toid(old),"replacement":replacement,"sources":sources,"prefix_checks":decision+1,"log_hash":digest(&serde_json::to_string(&b.log).unwrap()),"observer_hash":digest(&format!("{:?}",obs[side].mons())),"turn":b.turn,"actions":actions[side].iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"original_replies":actions[1-side].iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"alternate_replies":alt.legal_choices(&dex,1-side).iter().map(|a|a.to_input(&dex)).collect::<Vec<_>>(),"hp":[b.poke(b.active_id(0).unwrap()).hp,b.poke(b.active_id(1).unwrap()).hp]}));
                        }
                    }
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize])),
            )
            .unwrap();
            for o in &mut obs {
                o.observe(&b, &dex);
            }
        }
        eprintln!(
            "game={} eligible={eligible} attempted={attempted} accepted={}",
            game["game"],
            rows.len()
        );
        std::fs::write(
            &args[3],
            serde_json::to_vec(
                &json!({"eligible":eligible,"attempted":attempted,"rejected":rejected,"rows":rows}),
            )
            .unwrap(),
        )
        .unwrap();
    }
}
