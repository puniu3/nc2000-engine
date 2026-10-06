use nc2000_bot::{
    preview::load_meta_pool, smmcts::dominated_actions, Belief, Observer, SplitMix64,
};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::{json, Value};
use std::{collections::BTreeMap, io::BufRead, path::Path};

fn masks(b: &Battle, dex: &Dex, side: usize) -> Vec<(String, String)> {
    dominated_actions(b, dex, side)
        .into_iter()
        .map(|(a, reason)| (a.to_input(dex), reason.to_string()))
        .collect()
}

fn active(b: &Battle, dex: &Dex, side: usize) -> Value {
    let p = b.poke(b.active_id(side).unwrap());
    json!({"species":dex.species.key(p.species),"item":p.item.map(|id|dex.items.key(id)),
        "speed":b.get_pokemon_action_speed(dex,b.active_id(side).unwrap()),"stats":p.stored_stats,
        "hp":p.hp,"maxhp":p.maxhp})
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut decisions = 0;
    let mut draws = 0;
    let mut changed_draws = 0;
    let mut changed = Vec::new();
    let mut reasons = BTreeMap::<String, usize>::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        b.set_log_enabled(true);
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let legal = [0, 1].map(|s| b.legal_choices(&dex, s));
            for side in 0..2 {
                if legal[side].len() < 2 || !b.is_idle_move_request() {
                    continue;
                }
                decisions += 1;
                let obs = &observers[side];
                let belief = Belief::new(&dex, &pool, obs);
                let mut base = b.clone();
                base.quick_claw_roll = false;
                base.update_all_speeds(&dex);
                let before = masks(&base, &dex, side);
                let mut variants = BTreeMap::new();
                for sample in 0..32 {
                    let seed =
                        0x2026102700 ^ b.state_key128() as u64 ^ ((side as u64) << 32) ^ sample;
                    let mut sim = belief.determinize(&dex, &b, obs, &mut SplitMix64::new(seed));
                    sim.quick_claw_roll = false;
                    sim.update_all_speeds(&dex);
                    let after = masks(&sim, &dex, side);
                    draws += 1;
                    if before != after {
                        changed_draws += 1;
                        for a in before
                            .iter()
                            .filter(|a| !after.contains(a))
                            .chain(after.iter().filter(|a| !before.contains(a)))
                        {
                            *reasons.entry(a.1.clone()).or_default() += 1;
                        }
                        variants.entry(format!("{after:?}")).or_insert_with(
                            || json!({"sample":sample,"mask":after,"foe":active(&sim,&dex,1-side)}),
                        );
                    }
                }
                if !variants.is_empty() {
                    let foe = b.active_id(1 - side).unwrap();
                    changed.push(json!({"game":game["game"],"decision":decision,"side":side,"key":d["key"],
                        "candidates":belief.candidate_count(),"item_known":obs.mons()[foe.slot as usize].item.current.is_some(),
                        "own":active(&base,&dex,side),"foe":active(&base,&dex,1-side),"mask":before,
                        "variants":variants.into_values().collect::<Vec<_>>()}));
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| legal[s][i as usize])),
            )
            .unwrap();
            for obs in &mut observers {
                obs.observe(&b, &dex);
            }
        }
    }
    std::fs::write(&args[2],serde_json::to_vec_pretty(&json!({"decisions":decisions,"draws":draws,
        "changed_draws":changed_draws,"changed_decisions":changed.len(),"reasons":reasons,"cases":changed})).unwrap()).unwrap();
}
