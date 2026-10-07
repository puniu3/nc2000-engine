use std::io::{BufRead, BufReader, Write};

use nc2000_bot::player::action_input;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::{SplitMix64, Observer, Belief};
use nc2000_bot::observe::move_matches;
use nc2000_engine::battle::Outcome;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 { return Err("native_asymmetric_census GAMES.jsonl ROOTS.jsonl".into()); }
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    let mut selected_games = std::collections::BTreeSet::new();
    let mut census = Vec::new();
    let mut lines = BufReader::new(std::fs::File::open(&args[1])?).lines();
    let manifest: Value = serde_json::from_str(&lines.next().ok_or("missing manifest")??)?;
    let cfg = &manifest["config"];
    let dex = Dex::from_json(&std::fs::read_to_string(cfg["dex"].as_str().ok_or("dex")?)?)?;
    let pool = load_meta_pool(std::path::Path::new(cfg["pool"].as_str().ok_or("pool")?));
    let belief_pool = load_meta_pool(std::path::Path::new(cfg["agents"][0]["args"][1].as_str().ok_or("belief pool")?));
    let mut games = std::collections::BTreeSet::new();
    let mut decisions = 0;
    let mut states = 0;
    let mut hp_uses = 0;
    let mut hp_games = 0;
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
        let current_side = game as usize % 2;
        let mut observer = Observer::new(&battle, current_side);
        let mut belief = Belief::new(&dex, &belief_pool, &observer);
        let mut bot_rng = SplitMix64::new(agent_seeds[0] as u64);
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
                if side == current_side {
                    observer.observe(&battle, &dex);
                    belief.sync_checked(&dex, &observer)?;
                    assert_eq!(frame["response"]["search_seed"], bot_rng.next());
                    *counts.entry("current_decisions".into()).or_default() += 1;
                    let idle = battle.is_idle_move_request();
                    let alive = battle.sides.each_ref().map(|s| s.pokemon_left);
                    let asymmetric = (alive[0] == 1 && alive[1] > 1) || (alive[1] == 1 && alive[0] > 1);
                    if idle { *counts.entry("idle_current_decisions".into()).or_default() += 1; }
                    if asymmetric { *counts.entry("asymmetric_current_decisions".into()).or_default() += 1; }
                    if asymmetric && idle { *counts.entry("asymmetric_idle_decisions".into()).or_default() += 1; }
                    if asymmetric && idle && choices.len() > 1 {
                        selected_games.insert(game);
                        let opp = 1-side;
                        let oid = battle.active_id(opp).ok_or("missing opponent")?;
                        let sid = battle.active_id(side).ok_or("missing current")?;
                        let enemy = &battle.sides[opp].roster[oid.slot as usize];
                        let own = &battle.sides[side].roster[sid.slot as usize];
                        let obs = &observer.mons()[oid.slot as usize];
                        let unrevealed = enemy.move_slots.iter().filter(|m|
                            !obs.revealed_moves.iter().any(|&r| move_matches(&dex,m.id,r)))
                            .map(|m| dex.moves.key(m.id)).collect::<Vec<_>>();
                        let item_known = obs.item.current.is_some() || !obs.preview_has_item;
                        let active = |s: usize| {
                            let id = battle.active_id(s).unwrap();
                            let mon = &battle.sides[s].roster[id.slot as usize];
                            json!({"slot":id.slot,"species":dex.species.key(mon.species),
                                "hp":mon.hp,"maxhp":mon.maxhp,"status":mon.status.as_str(),
                                "transformed":mon.transformed,"stats":mon.stored_stats,
                                "item":mon.item.map(|i|dex.items.key(i)),
                                "hp_type":format!("{:?}",mon.hp_type),"hp_power":mon.hp_power,
                                "moves":mon.move_slots.iter().map(|m|json!({"id":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})
                        };
                        let record = json!({"game":game,"pair":game/2,"frame":cursor,"side":side,
                            "turn":battle.turn,"alive":alive,"state_sha256":hash,"state_key":battle.state_key(),
                            "current_response":frame["response"],"legal_actions":frame["legal_actions"],
                            "opponent_legal_actions":battle.clone().legal_choices(&dex,opp).iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>(),
                            "observation":{"revealed_moves":obs.revealed_moves.iter().map(|&m|dex.moves.key(m)).collect::<Vec<_>>(),
                                "hp_type_mask":obs.hp_type_mask,"item_current":format!("{:?}",obs.item.current),
                                "item_known":item_known,"preview_has_item":obs.preview_has_item},
                            "belief":{"candidate_count":belief.candidate_count(),"fallback":belief.is_fallback(),
                                "candidate_ids":belief.alive().iter().map(|&i|belief.candidate_id(i)).collect::<Vec<_>>()},
                            "diagnostic_truth":{"own":active(side),"opponent":active(opp),
                                "unrevealed_current_moves":unrevealed,"all_current_moves_revealed":unrevealed.is_empty(),
                                "either_transformed":own.transformed || enemy.transformed}});
                        let restored: Battle = serde_json::from_value(state.clone())?;
                        assert_eq!(format!("{:?}", restored.clone()), format!("{:?}", battle.clone()));
                        assert_eq!(restored.state_key(),battle.state_key());
                        serde_json::to_writer(&mut output, &json!({"record":record,"battle":state}))?;
                        writeln!(output)?;
                        census.push(record);
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
    output.flush()?;
    census.sort_by_key(|r|(r["game"].as_u64().unwrap(),r["frame"].as_u64().unwrap()));
    println!("{}", json!({"games":cfg["games"],"replayed_decisions":decisions,"replayed_states":states,
        "games_with_hidden_power":hp_games,"hidden_power_uses":hp_uses,"counts":counts,"selected_games":selected_games.len(),"selected_roots":census.len(),"roots":census}));
    Ok(())
}
