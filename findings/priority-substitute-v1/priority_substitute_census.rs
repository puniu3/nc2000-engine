use std::io::{BufRead, BufReader};

use nc2000_bot::player::action_input;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::SplitMix64;
use nc2000_engine::battle::Outcome;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 2 { return Err("native_replay GAMES.jsonl".into()); }
    let mut lines = BufReader::new(std::fs::File::open(&args[1])?).lines();
    let manifest: Value = serde_json::from_str(&lines.next().ok_or("missing manifest")??)?;
    let cfg = &manifest["config"];
    let dex = Dex::from_json(&std::fs::read_to_string(cfg["dex"].as_str().ok_or("dex")?)?)?;
    let pool = load_meta_pool(std::path::Path::new(cfg["pool"].as_str().ok_or("pool")?));
    let mut games = std::collections::BTreeSet::new();
    let mut decisions = 0;
    let mut states = 0;
    let mut hp_uses = 0;
    let mut hp_games = 0;
    let mut audit = Vec::new();
    for line in lines {
        let row: Value = serde_json::from_str(&line?)?;
        let game = row["game"].as_u64().ok_or("game")?;
        assert!(games.insert(game));
        let mut schedule = SplitMix64(SplitMix64(cfg["seed"].as_u64().unwrap()
            .wrapping_add((game / 2).wrapping_mul(0x9E3779B97F4A7C15))).next());
        let teams = [schedule.below(pool.teams.len()), schedule.below(pool.teams.len())];
        let seed = schedule.battle_seed();
        let agent_seeds = [schedule.next() as u32; 2];
        assert_eq!(row["team_ids"], json!(teams));
        assert_eq!(row["battle_seed"], seed);
        assert_eq!(row["agent_seeds"], json!(agent_seeds));
        let mut battle = Battle::from_fixture(&dex, &seed, &pool.teams[teams[0]].sets,
            &pool.teams[teams[1]].sets).map_err(|e| format!("init: {e:?}"))?;
        let frames = row["frames"].as_array().ok_or("frames")?;
        let mut cursor = 0;
        while battle.outcome().is_none() {
            let state = serde_json::to_value(&battle)?;
            let hash = format!("{:x}", Sha256::digest(serde_json::to_vec(&state)?));
            let mut picks = [None, None];
            for side in 0..2 {
                let choices = battle.legal_choices(&dex, side);
                if choices.is_empty() { continue; }
                let frame = frames.get(cursor).ok_or("incomplete frames")?;
                assert_eq!(frame["side"], side);
                assert_eq!(frame["agent"], side ^ (game as usize % 2));
                assert_eq!(frame["turn"], battle.turn);
                assert_eq!(frame["state_sha256"], hash);
                assert_eq!(frame["state_key"], battle.state_key());
                assert_eq!(frame["response"]["key"], battle.state_key());
                assert_eq!(frame["legal_actions"], json!(choices.iter().map(|&a| action_input(&dex,a)).collect::<Vec<_>>()));
                picks[side] = Some(*choices.iter().find(|&&a| action_input(&dex,a)==frame["response"]["action"]).ok_or("illegal action")?);
                if frame["agent"] == 0 && battle.is_idle_move_request() {
                    if let Some(own) = battle.active_id(side) {
                        let substitute = dex.moves.id("substitute").unwrap();
                        if choices.contains(&nc2000_engine::battle::SearchChoice::Move(substitute)) {
                            let masks = nc2000_bot::smmcts::dominated_actions(&battle, &dex, side);
                            let reason = masks.iter().find(|(a, _)| *a == nc2000_engine::battle::SearchChoice::Move(substitute)).map(|(_, why)| *why);
                            let foe = battle.active_id(1-side).unwrap();
                            audit.push(json!({"game":game,"frame":cursor,"side":side,"state_sha256":hash,
                                "species":dex.species.key(battle.poke(own).species),"foe":dex.species.key(battle.poke(foe).species),
                                "substitute_up":battle.poke(own).has_volatile(dex.conds_id("substitute").unwrap()),
                                "hp":battle.poke(own).hp,"maxhp":battle.poke(own).maxhp,"reason":reason,
                                "chosen":frame["response"]["action"]}));
                        }
                    }
                }
                cursor += 1;
            }
            assert_ne!(picks, [None, None]);
            battle.apply_choices(&dex,picks).map_err(|e| format!("apply: {e:?}"))?;
            states += 1;
        }
        let outcome = match battle.outcome().unwrap() {
            Outcome::P1Win => "p1", Outcome::P2Win => "p2", Outcome::Tie => "tie",
        };
        assert_eq!(row["outcome"], outcome);
        assert_eq!(row["turns"], battle.turn);
        assert_eq!(row["terminal_key"], battle.state_key());
        assert_eq!(row["terminal_seed"], battle.prng.seed_str());
        assert_eq!(frames.len(), cursor);
        decisions += cursor;
        let hp = battle.log.iter().filter(|l| l.starts_with("|move|") && l.contains("|Hidden Power|")).count();
        hp_uses += hp;
        hp_games += usize::from(hp > 0);
    }
    assert_eq!(games.len() as u64, cfg["games"].as_u64().unwrap());
    assert_eq!(games.into_iter().collect::<Vec<_>>(), (0..cfg["games"].as_u64().unwrap()).collect::<Vec<_>>());
    println!("{}", json!({"games":cfg["games"],"replayed_decisions":decisions,"replayed_states":states,
        "games_with_hidden_power":hp_games,"hidden_power_uses":hp_uses,"substitute_roots":audit}));
    Ok(())
}
