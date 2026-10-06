use nc2000_bot::{belief::Belief, observe::Observer, rng::SplitMix64};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::{json, Value};
use std::io::BufRead;

fn key(dex: &Dex, b: &Battle) -> String {
    assert!(!b.damage_bookkeeping_observable(dex) && !b.quick_claw_roll_observable(dex));
    b.state_key128_without_quick_claw_roll(dex, true)
        .to_string()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "corpus game decision output");
    let game_id: usize = args[2].parse().unwrap();
    let decision: usize = args[3].parse().unwrap();
    let dex = conformance::load_dex();
    let game = std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap())
        .lines()
        .map(|l| serde_json::from_str::<Value>(&l.unwrap()).unwrap())
        .find(|g| g["game"] == game_id)
        .unwrap();
    let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    let mut b =
        Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(true);
    let mut observers = [0, 1].map(|s| Observer::new(&b, s));
    let mut beliefs = observers
        .each_ref()
        .map(|obs| Belief::pinned_from_battle(&b, obs));
    for d in game["decisions"].as_array().unwrap().iter().take(decision) {
        assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
        let aa = [0, 1].map(|s| b.legal_choices(&dex, s));
        b.apply_choices(
            &dex,
            [0, 1].map(|s| d["choices"][s].as_u64().map(|i| aa[s][i as usize])),
        )
        .unwrap();
        for s in [0, 1] {
            observers[s].observe(&b, &dex);
            beliefs[s].sync(&dex, &observers[s]);
        }
    }
    assert_eq!(
        b.state_key128().to_string(),
        game["decisions"][decision]["key"].as_str().unwrap()
    );
    let mut rows = Vec::new();
    for history in [false, true] {
        for side in [0, 1] {
            let fresh = Observer::new(&b, side);
            let pinned = Belief::pinned_from_battle(&b, &fresh);
            let (obs, belief) = if history {
                (&observers[side], &beliefs[side])
            } else {
                (&fresh, &pinned)
            };
            for seed in 0..32 {
                let sampled = belief.determinize(&dex, &b, obs, &mut SplitMix64::new(seed));
                let same_party = (0..2).all(|s| sampled.sides[s].party == b.sides[s].party);
                let restored_key = same_party.then(|| {
                    let mut restored = sampled.clone();
                    for s in [0, 1] {
                        for slot in 0..b.sides[s].roster.len() {
                            if !b.sides[s].party.contains(&(slot as u8)) {
                                restored.sides[s].roster[slot].position =
                                    b.sides[s].roster[slot].position;
                            }
                        }
                    }
                    key(&dex, &restored)
                });
                let mut changed = Vec::new();
                for s in [0, 1] {
                    for (slot, (before, after)) in b.sides[s]
                        .roster
                        .iter()
                        .zip(&sampled.sides[s].roster)
                        .enumerate()
                    {
                        let before = format!("{before:#?}");
                        let after = format!("{after:#?}");
                        if before != after {
                            changed.push(
                                json!({"side": s, "slot": slot, "before": before, "after": after}),
                            );
                        }
                    }
                }
                rows.push(json!({"history": history, "side": side, "seed": seed, "key": key(&dex, &sampled),
                "party": sampled.sides.each_ref().map(|s| s.party.to_vec()), "nonparty_positions_restored_key": restored_key,
                "essence": sampled.essence(&dex), "changed": changed}));
            }
        }
    }
    std::fs::write(&args[4], serde_json::to_vec(&json!({"game": game_id, "decision": decision,
        "state_key": b.state_key128().to_string(), "key": key(&dex, &b), "party": b.sides.each_ref().map(|s| s.party.to_vec()),
        "essence": b.essence(&dex), "rows": rows})).unwrap()).unwrap();
}
