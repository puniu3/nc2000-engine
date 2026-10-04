use std::path::Path;

use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::prng::{BattleRng, Prng};
use nc2000_engine::state::Battle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn read(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 2, "tactical_continuation_verify RUN_DIRECTORY");
    let directory = Path::new(&args[1]);
    let manifest = read(&directory.join("manifest.json"));
    let protocol = &manifest["protocol"];
    let dex = conformance::load_dex();
    let mut verified_games = 0;
    let mut verified_decisions = 0;
    for case in protocol["cases"].as_array().unwrap() {
        let bytes = std::fs::read(case["game"].as_str().unwrap()).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            case["game_sha256"].as_str().unwrap()
        );
        let game: Value = serde_json::from_slice(&bytes).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut root = Battle::from_fixture(
            &dex,
            game["battle_seed"].as_str().unwrap(),
            &teams[0],
            &teams[1],
        )
        .unwrap();
        let decision = case["decision"].as_u64().unwrap() as usize;
        for row in &game["records"].as_array().unwrap()[..decision] {
            assert_eq!(root.state_key128().to_string(), row["state_key"]);
            let choices = [0, 1].map(|side| {
                row["choices"][side].as_str().map(|input| {
                    *root
                        .legal_choices(&dex, side)
                        .iter()
                        .find(|a| a.to_input(&dex) == input)
                        .unwrap()
                })
            });
            root.apply_choices(&dex, choices).unwrap();
        }
        assert_eq!(
            root.state_key128().to_string(),
            game["records"][decision]["state_key"]
        );
        let id = case["id"].as_str().unwrap();
        let side = case["side"].as_u64().unwrap() as usize;
        for trial in 0..protocol["trials"].as_u64().unwrap() {
            for (arm, action) in case["arms"].as_array().unwrap().iter().enumerate() {
                let row = read(&directory.join(format!("{id}-t{trial:03}-a{arm}.json")));
                assert_eq!(row["arm"], *action);
                assert_eq!(row["case"], id);
                assert_eq!(row["trial"], trial);
                let mut battle = root.clone();
                battle.prng = BattleRng::seeded(Prng::new(row["battle_seed"].as_u64().unwrap()));
                let decisions = row["decisions"].as_array().unwrap();
                assert_eq!(decisions[0]["choices"][side], *action);
                for step in decisions {
                    assert_eq!(battle.state_key128().to_string(), step["key"]);
                    assert_eq!(battle.turn, step["turn"].as_u64().unwrap() as u16);
                    let choices = [0, 1].map(|s| {
                        let legal = battle.legal_choices(&dex, s);
                        assert_eq!(legal.is_empty(), step["choices"][s].is_null());
                        step["choices"][s]
                            .as_str()
                            .map(|input| *legal.iter().find(|a| a.to_input(&dex) == input).unwrap())
                    });
                    battle.apply_choices(&dex, choices).unwrap();
                    verified_decisions += 1;
                }
                let score = battle.outcome().map(|outcome| match outcome {
                    Outcome::Tie => 0.5,
                    Outcome::P1Win => {
                        if side == 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    Outcome::P2Win => {
                        if side == 1 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                });
                assert_eq!(json!(score), row["score"]);
                assert_eq!(json!(battle.outcome().is_some()), row["terminal"]);
                assert_eq!(format!("{:?}", battle.outcome()), row["outcome"]);
                assert_eq!(json!(battle.turn), row["turns"]);
                assert!(
                    battle.outcome().is_some()
                        || u64::from(battle.turn) > protocol["max_turns"].as_u64().unwrap()
                );
                verified_games += 1;
            }
        }
    }
    println!(
        "{}",
        json!({"verified_games":verified_games,"verified_decisions":verified_decisions,"state_keys":true,"legal_replay":true,"outcomes":true})
    );
}
