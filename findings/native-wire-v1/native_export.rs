use std::io::{BufRead, Write};

#[allow(dead_code)]
mod worker {
    include!("native_worker.rs");
}

use nc2000_bot::player::{action_input, PlayerChannel};
use nc2000_bot::preview::load_meta_pool;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 { return Err("native_export GAMES.jsonl OUTPUT.jsonl BELIEF_POOL".into()); }
    let mut lines = std::io::BufReader::new(std::fs::File::open(&args[1])?).lines();
    let manifest: Value = serde_json::from_str(&lines.next().ok_or("missing manifest")??)?;
    let dex = Dex::from_json(&std::fs::read_to_string(manifest["config"]["dex"].as_str().ok_or("dex")?)?)?;
    let pool = load_meta_pool(std::path::Path::new(manifest["config"]["pool"].as_str().ok_or("pool")?));
    let belief = load_meta_pool(std::path::Path::new(&args[3]));
    let mut output = std::io::BufWriter::new(std::fs::OpenOptions::new().write(true).create_new(true).open(&args[2])?);
    for line in lines {
        let game: Value = serde_json::from_str(&line?)?;
        let ids = game["team_ids"].as_array().ok_or("team ids")?;
        let mut battle = Battle::from_fixture(&dex, game["battle_seed"].as_str().ok_or("battle seed")?,
            &pool.teams[ids[0].as_u64().ok_or("team id")? as usize].sets,
            &pool.teams[ids[1].as_u64().ok_or("team id")? as usize].sets).map_err(|e| format!("init: {e:?}"))?;
        let mut sessions = [
            worker::NativeSession::new(&dex, &belief, &battle, 0, 19),
            worker::NativeSession::new(&dex, &belief, &battle, 1, 23),
        ];
        let mut channels = [PlayerChannel::new(0), PlayerChannel::new(1)];
        let frames = game["frames"].as_array().ok_or("frames")?;
        let mut frame = 0;
        let mut round = 0;
        loop {
            let state = serde_json::to_value(&battle)?;
            let restored: Battle = serde_json::from_value(state.clone())?;
            assert_eq!(format!("{:?}", battle.clone()), format!("{:?}", restored.clone()));
            assert_eq!(battle.state_key(), restored.state_key());
            let terminal = battle.outcome().map(|o| format!("{o:?}"));
            let mut expected = [None, None];
            for side in 0..2 {
                if terminal.is_none() && !battle.legal_choices(&dex, side).is_empty() {
                    expected[side] = Some(sessions[side].choose(&dex, &battle, 128)?);
                }
            }
            serde_json::to_writer(&mut output, &json!({"game":game["game"],"round":round,
                "key":battle.state_key(), "debug":format!("{:?}",battle.clone()),
                "terminal":terminal,"battle":state,"expected":expected}))?;
            writeln!(output)?;
            if battle.outcome().is_some() {
                assert_eq!(frame, frames.len());
                assert_eq!(battle.turn as u64, game["turns"].as_u64().unwrap());
                break;
            }
            let mut choices = [None, None];
            for side in 0..2 {
                let legal = battle.legal_choices(&dex, side);
                if legal.is_empty() { continue; }
                let saved = &frames[frame];
                assert_eq!(saved["side"], side);
                assert_eq!(serde_json::to_value(channels[side].frame(&mut battle, &dex)?)?, saved["frame"]);
                choices[side] = Some(*legal.iter().find(|&&a| action_input(&dex, a) == saved["response"]["action"]).ok_or("illegal recorded action")?);
                frame += 1;
            }
            battle.apply_choices(&dex, choices).map_err(|e| format!("apply: {e:?}"))?;
            round += 1;
        }
    }
    output.flush()?;
    Ok(())
}
