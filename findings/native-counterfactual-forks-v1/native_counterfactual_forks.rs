use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Write};

use nc2000_bot::{Belief, Observer, RmConfig, SplitMix64};
use nc2000_bot::blind::BlindSearch;
use nc2000_bot::player::action_input;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::SelRule;
use nc2000_engine::battle::Outcome;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn hash(v: &Value) -> String { format!("{:x}",Sha256::digest(serde_json::to_vec(v).unwrap())) }
fn outcome(b: &Battle) -> Option<&'static str> {
    b.outcome().map(|o|match o {Outcome::P1Win=>"p1",Outcome::P2Win=>"p2",Outcome::Tie=>"tie"})
}
fn belief_info(b: &Belief) -> Value {
    json!({"candidate_count":b.candidate_count(),"fallback":b.is_fallback(),
        "candidate_ids":b.alive().iter().map(|&i|b.candidate_id(i)).collect::<Vec<_>>()})
}
fn observe_info(dex:&Dex,obs:&Observer)->Value {
    json!(obs.mons().iter().map(|m|json!({"species":dex.species.key(m.species),
        "appeared":m.appeared,"preview_has_item":m.preview_has_item,
        "revealed_moves":m.revealed_moves.iter().map(|&id|dex.moves.key(id)).collect::<Vec<_>>(),
        "hp_type_mask":m.hp_type_mask,"item_original":format!("{:?}",m.item.original),
        "item_current":format!("{:?}",m.item.current)})).collect::<Vec<_>>())
}
fn choose(dex:&Dex,battle:&Battle,side:usize,belief:&Belief,observer:&Observer,seed:u64)->Value {
    let cfg=RmConfig{rule:SelRule::Ucb,c:0.4,hp_buckets:16,..RmConfig::default()};
    let mut search=BlindSearch::new(battle,dex,cfg,side,seed);
    search.step(dex,belief,observer,27000);
    let action=search.best().expect("choice");
    let actions=search.actions().iter().zip(search.visits()).zip(search.means())
        .map(|((a,visits),mean)|json!({"input":a.to_input(dex),"visits":visits,"mean":mean})).collect::<Vec<_>>();
    json!({"action":action.to_input(dex),"actions":actions,"iterations":search.iterations(),
        "search_seed":seed,"key":battle.state_key()})
}
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    assert_eq!(args.len(),4,"native_counterfactual_forks PROTOCOL GAMES OUTPUT");
    let protocol:Value=serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let input=std::fs::read(&args[2])?;
    assert_eq!(protocol["games_sha256"],format!("{:x}",Sha256::digest(&input)));
    let measure=protocol["measure_search"].as_bool().unwrap();
    let parents:BTreeMap<u64,usize>=protocol["parents"].as_array().unwrap().iter()
        .map(|p|(p[0].as_u64().unwrap(),p[1].as_u64().unwrap() as usize)).collect();
    assert_eq!(parents.len(),82);
    let mut lines=BufReader::new(input.as_slice()).lines();
    let manifest:Value=serde_json::from_str(&lines.next().ok_or("manifest")??)?;
    let cfg=&manifest["config"];
    let dex_raw=std::fs::read(cfg["dex"].as_str().unwrap())?;
    assert_eq!(protocol["dex_sha256"],format!("{:x}",Sha256::digest(&dex_raw)));
    let dex=Dex::from_json(std::str::from_utf8(&dex_raw)?)?;
    let pool_path=cfg["pool"].as_str().unwrap();
    let belief_path=cfg["agents"][0]["args"][1].as_str().unwrap();
    assert_eq!(protocol["pool_sha256"],format!("{:x}",Sha256::digest(std::fs::read(pool_path)?)));
    assert_eq!(protocol["belief_sha256"],format!("{:x}",Sha256::digest(std::fs::read(belief_path)?)));
    let pool=load_meta_pool(std::path::Path::new(pool_path));
    let belief_pool=load_meta_pool(std::path::Path::new(belief_path));
    let mut output=std::io::BufWriter::new(std::fs::File::create_new(&args[3])?);
    let mut games=BTreeSet::new();let mut completed_parents=BTreeSet::new();
    let mut decisions=0;let mut states=0;let mut records=Vec::new();
    for line in lines {
        let row:Value=serde_json::from_str(&line?)?;
        let game=row["game"].as_u64().unwrap();assert!(games.insert(game));
        let mut schedule=SplitMix64(SplitMix64(cfg["seed"].as_u64().unwrap()
            .wrapping_add((game/2).wrapping_mul(0x9E3779B97F4A7C15))).next());
        let teams=[schedule.below(pool.teams.len()),schedule.below(pool.teams.len())];
        let seed=schedule.battle_seed();let agent_seeds=[schedule.next() as u32;2];
        assert_eq!(row["team_ids"],json!(teams));assert_eq!(row["battle_seed"],seed);
        assert_eq!(row["agent_seeds"],json!(agent_seeds));
        let mut battle=Battle::from_fixture(&dex,&seed,&pool.teams[teams[0]].sets,&pool.teams[teams[1]].sets)
            .map_err(|e|format!("init {e:?}"))?;
        let side=game as usize%2;let opp=1-side;
        let mut observer=Observer::new(&battle,side);
        let mut belief=Belief::new(&dex,&belief_pool,&observer);
        let mut fork_beliefs:Vec<_>=(0..4).map(|_|Belief::new(&dex,&belief_pool,&observer)).collect();
        let mut rng=SplitMix64::new(agent_seeds[0] as u64);
        let frames=row["frames"].as_array().unwrap();let mut cursor=0;
        while battle.outcome().is_none() {
            let state=serde_json::to_value(&battle)?;let state_hash=hash(&state);
            let mut picks=[None,None];let mut parent_frame=None;
            for s in 0..2 {
                let choices=battle.legal_choices(&dex,s);if choices.is_empty(){continue;}
                let f=frames.get(cursor).ok_or("incomplete frames")?;
                assert_eq!(f["side"],s);assert_eq!(f["agent"],s^side);assert_eq!(f["turn"],battle.turn);
                assert_eq!(f["state_sha256"],state_hash);assert_eq!(f["state_key"],battle.state_key());
                assert_eq!(f["response"]["key"],battle.state_key());
                assert_eq!(f["legal_actions"],json!(choices.iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>()));
                picks[s]=Some(*choices.iter().find(|&&a|action_input(&dex,a)==f["response"]["action"]).ok_or("illegal")?);
                if s==side {
                    observer.observe(&battle,&dex);belief.sync_checked(&dex,&observer)?;
                    for fork in &mut fork_beliefs {fork.sync_checked(&dex,&observer)?;}
                    assert_eq!(f["response"]["search_seed"],rng.next());
                    if parents.get(&game)==Some(&cursor) {
                        assert!(battle.is_idle_move_request() && choices.len()>1);
                        assert!(battle.sides.iter().all(|s|s.pokemon_left==1));
                        parent_frame=Some(cursor);
                    }
                }
                cursor+=1;
            }
            if let Some(parent_frame)=parent_frame {
                assert!(completed_parents.insert(game));
                let opponent_choices=battle.clone().legal_choices(&dex,opp);
                assert!(opponent_choices.len()<=fork_beliefs.len());
                let next_seed=rng.clone().next();
                for (branch,(&opponent_choice,mut fork_belief)) in opponent_choices.iter().zip(fork_beliefs.drain(..)).enumerate() {
                    assert_eq!(belief_info(&belief),belief_info(&fork_belief));
                    let is_control=Some(opponent_choice)==picks[opp];
                    let mut fork=battle.clone();let mut choices=picks;choices[opp]=Some(opponent_choice);
                    fork.apply_choices(&dex,choices).map_err(|e|format!("fork {e:?}"))?;
                    let fork_state=serde_json::to_value(&fork)?;let fork_hash=hash(&fork_state);
                    let restored:Battle=serde_json::from_value(fork_state.clone())?;
                    assert_eq!(format!("{:?}",restored.clone()),format!("{:?}",fork.clone()));
                    let legal=[0,1].map(|s|fork.clone().legal_choices(&dex,s).iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>());
                    let eligible=fork.is_idle_move_request() && legal[side].len()>1
                        && fork.sides.iter().all(|s|s.pokemon_left==1);
                    let mut fork_observer=observer.clone();
                    let mut response=Value::Null;let mut expected=Value::Null;
                    if !legal[side].is_empty() {
                        fork_observer.observe(&fork,&dex);fork_belief.sync_checked(&dex,&fork_observer)?;
                        if is_control {
                            let next=frames[cursor..].iter().find(|f|f["side"]==side).ok_or("control next decision")?;
                            assert_eq!(next["state_sha256"],fork_hash);assert_eq!(next["state_key"],fork.state_key());
                            assert_eq!(next["legal_actions"],json!(legal[side]));
                            assert_eq!(next["response"]["search_seed"],next_seed);
                            expected=next["response"].clone();
                        }
                        if measure {
                            response=choose(&dex,&fork,side,&fork_belief,&fork_observer,next_seed);
                            if is_control {assert_eq!(response,expected);}
                        }
                    } else if is_control && fork.outcome().is_some() {
                        assert_eq!(row["outcome"],json!(outcome(&fork)));
                        assert_eq!(row["terminal_key"],fork.state_key());
                        assert_eq!(row["terminal_seed"],fork.prng.seed_str());
                    }
                    let record=json!({"fork_id":format!("{game}:{parent_frame}:{branch}"),"source_game":game,
                        "source_frame":parent_frame,"branch":branch,"side":side,"parent_sha256":state_hash,
                        "parent_key":battle.state_key(),"own_action":action_input(&dex,picks[side].unwrap()),
                        "opponent_action":action_input(&dex,opponent_choice),"original_opponent_action":action_input(&dex,picks[opp].unwrap()),
                        "opponent_actions":opponent_choices.iter().map(|&a|action_input(&dex,a)).collect::<Vec<_>>(),
                        "is_control":is_control,"child_sha256":fork_hash,"child_key":fork.state_key(),
                        "child_turn":fork.turn,"outcome":outcome(&fork),"eligible":eligible,
                        "legal_actions":legal,"next_bot_seed":next_seed,"response":response,"expected_control_response":expected,
                        "observation":observe_info(&dex,&fork_observer),"belief":belief_info(&fork_belief)});
                    serde_json::to_writer(&mut output,&json!({"record":record,"battle":fork_state}))?;
                    writeln!(output)?;output.flush()?;records.push(record);
                }
                eprintln!("parents={}/82 game={game} branches={}",completed_parents.len(),opponent_choices.len());
            }
            assert_ne!(picks,[None,None]);
            battle.apply_choices(&dex,picks).map_err(|e|format!("apply {e:?}"))?;states+=1;
        }
        assert_eq!(row["outcome"],json!(outcome(&battle)));assert_eq!(row["turns"],battle.turn);
        assert_eq!(row["terminal_key"],battle.state_key());assert_eq!(row["terminal_seed"],battle.prng.seed_str());
        assert_eq!(frames.len(),cursor);decisions+=cursor;
    }
    assert_eq!(games.into_iter().collect::<Vec<_>>(),(0..cfg["games"].as_u64().unwrap()).collect::<Vec<_>>());
    assert_eq!(completed_parents,parents.keys().copied().collect());
    records.sort_by_key(|r|(r["source_game"].as_u64().unwrap(),r["branch"].as_u64().unwrap()));
    println!("{}",json!({"games":cfg["games"],"replayed_states":states,"replayed_decisions":decisions,
        "parents":parents.len(),"branches":records.len(),"measure_search":measure,"records":records}));
    Ok(())
}
