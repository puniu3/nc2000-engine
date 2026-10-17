//! Read-only replay of a saved fixed-budget trial, without running search.
use nc2000_engine::{battle::PokemonSet, state::Battle};
use serde_json::{json, Value};
use std::{env, fs};
fn main() {
    let args: Vec<_> = env::args().collect();
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let trace: Value = serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    let pair = trace["pair"].as_u64().map(|n| n as usize).unwrap_or_else(|| {
        trace["context"]["job"].as_str().unwrap().split("-p").nth(1).unwrap()[..3].parse().unwrap()
    });
    let schedule = &manifest["schedules"][pair];
    let catalog: Value = serde_json::from_slice(&fs::read("data/team-pool-v2/team-pool.json").unwrap()).unwrap();
    let teams: [Vec<PokemonSet>; 2] = [0, 1].map(|s| {
        serde_json::from_value(catalog["teams"].as_array().unwrap().iter()
            .find(|t| t["id"] == schedule["teams"][s]).unwrap()["sets"].clone()).unwrap()
    });
    let dex = conformance::load_dex();
    let mut b = Battle::from_fixture(&dex, schedule["battle_seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(true);
    let decisions = trace["decisions"].as_array().or_else(|| trace["result"]["decisions"].as_array()).unwrap();
    let limit = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(decisions.len());
    let mut cursor = 0;
    while cursor < limit {
        let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
        if cursor + legal.iter().filter(|c| !c.is_empty()).count() > limit { break; }
        let mut choices = [None, None];
        for s in 0..2 {
            if legal[s].is_empty() { continue; }
            let row = &decisions[cursor];
            assert_eq!(row["side"], s);
            assert_eq!(row["key"], b.state_key128().to_string());
            assert_eq!(row["turn"], b.turn);
            choices[s] = Some(*legal[s].iter().find(|c| c.to_input(&dex) == row["choice"].as_str().unwrap()).unwrap());
            cursor += 1;
        }
        b.apply_choices(&dex, choices).unwrap();
    }
    if cursor == decisions.len() && trace.get("next_key").is_some() {
        assert_eq!(trace["next_key"], b.state_key128().to_string());
    }
    let sides = [0, 1].map(|s| b.sides[s].party.iter().map(|&slot| {
        let p = &b.sides[s].roster[slot as usize];
        json!({"species":dex.species.key(p.species),"active":b.sides[s].active==Some(slot),
            "hp":p.hp,"maxhp":p.maxhp,"status":p.status.as_str(),
            "moves":p.move_slots.iter().map(|m|json!({"move":dex.moves.key(m.id),"pp":m.pp})).collect::<Vec<_>>()})
    }).collect::<Vec<_>>());
    println!("{}", json!({"pair":pair,"teams":schedule["teams"],"turn":b.turn,"key":b.state_key128().to_string(),"sides":sides}));
}
