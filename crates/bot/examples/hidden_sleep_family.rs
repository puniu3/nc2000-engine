use nc2000_bot::Observer;
use nc2000_engine::{battle::PokemonSet, state::{Battle, DK, PokeId, Status}};
use serde_json::{json, Value};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus case output");
    let row: Value = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    let game: Value = std::fs::read_to_string(&args[1]).unwrap().lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap()).find(|g| g["game"] == row["game"]).unwrap();
    let dex = conformance::load_dex();
    let side = row["side"].as_u64().unwrap() as usize;
    let foe = PokeId { side: (1-side) as u8, slot: row["foe_slot"].as_u64().unwrap() as u8 };
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut b = Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(true);
    let mut obs = Observer::new(&b, side);
    let mut timeline = Vec::new();
    for (i,d) in game["decisions"].as_array().unwrap().iter().take(row["decision"].as_u64().unwrap() as usize).enumerate() {
        assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
        let before = b.clone();
        let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
        let choices = [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize]));
        b.apply_choices(&dex, choices).unwrap();
        obs.observe(&b, &dex);
        if before.poke(foe).status == Status::Slp || b.poke(foe).status == Status::Slp {
            timeline.push(json!({"decision":i,"before_status":format!("{:?}",before.poke(foe).status),
                "before_time":before.poke(foe).status_state.get_int(DK::Time),
                "after_status":format!("{:?}",b.poke(foe).status),"after_time":b.poke(foe).status_state.get_int(DK::Time),
                "log":&b.log[before.log.len()..]}));
        }
    }
    assert_eq!(b.state_key128().to_string(),row["key"].as_str().unwrap());
    std::fs::write(&args[3], serde_json::to_vec_pretty(&json!({"case":row,"timeline":timeline,"state":b.essence(&dex)})).unwrap()).unwrap();
}
