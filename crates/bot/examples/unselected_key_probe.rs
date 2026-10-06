use nc2000_bot::{preview::load_meta_pool, Belief, Observer, SplitMix64};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, PokemonSet},
    state::Battle,
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io::BufRead,
};

fn project(b: &mut Battle) -> usize {
    assert!(b.started && b.turn > 0);
    let marker = b.sides[0].roster[b.sides[0].party[0] as usize].clone();
    let mut omitted = 0;
    for side in &mut b.sides {
        for (slot, mon) in side.roster.iter_mut().enumerate() {
            if !side.party.contains(&(slot as u8)) {
                *mon = marker.clone();
                omitted += 1;
            }
        }
    }
    omitted
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert!(
        args.len() == 3 || (args.len() == 4 && args[3] == "--verify"),
        "corpus output [--verify]"
    );
    let verify = args.len() == 4;
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    let mut rows = Vec::new();
    let mut cases = Vec::new();
    for line in std::io::BufReader::new(std::fs::File::open(&args[1]).unwrap()).lines() {
        let game: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
        let mut b =
            Battle::from_fixture(&dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1])
                .unwrap();
        let mut observers = [0, 1].map(|s| Observer::new(&b, s));
        for (decision, d) in game["decisions"].as_array().unwrap().iter().enumerate() {
            assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap());
            let acts = [0, 1].map(|s| b.legal_choices(&dex, s));
            for side in 0..2 {
                if acts[side].len() < 2 || b.active_id(side).is_none() || b.turn == 0 {
                    continue;
                }
                let belief = Belief::new(&dex, &pool, &observers[side]);
                let mut rng = SplitMix64::new(0x2026101000 ^ b.state_key128() as u64 ^ side as u64);
                let mut exact = HashSet::new();
                let mut projected_exact = HashSet::new();
                let mut bucketed = HashSet::new();
                let mut projected_bucketed = HashSet::new();
                let mut coin_bucketed = HashSet::new();
                let mut combined_bucketed = HashSet::new();
                let mut coin_eligible_samples = 0;
                let mut omitted = 0;
                let mut seen = HashMap::<u128, (u128, Battle)>::new();
                let mut pair = None;
                for _ in 0..256 {
                    let pick = belief.sample(&mut rng);
                    let mut sim =
                        belief.determinize_with(&dex, &b, &observers[side], pick, &mut rng);
                    let key = sim.state_key128();
                    let original = (verify && pair.is_none()).then(|| sim.clone());
                    exact.insert(key);
                    bucketed.insert(sim.search_key(Some(16), false));
                    let omit_coin = !sim.quick_claw_roll_observable(&dex);
                    coin_eligible_samples += usize::from(omit_coin);
                    let coin = sim.quick_claw_roll;
                    if omit_coin {
                        sim.quick_claw_roll = false;
                    }
                    coin_bucketed.insert(sim.search_key(Some(16), false));
                    sim.quick_claw_roll = coin;
                    omitted += project(&mut sim);
                    let projected = sim.state_key128();
                    if let Some(original) = original {
                        if let Some((other_key, other)) = seen.get(&projected) {
                            if *other_key != key {
                                pair = Some((other.clone(), original));
                            }
                        } else {
                            seen.insert(projected, (key, original));
                        }
                    }
                    projected_exact.insert(projected);
                    projected_bucketed.insert(sim.search_key(Some(16), false));
                    if omit_coin {
                        sim.quick_claw_roll = false;
                    }
                    combined_bucketed.insert(sim.search_key(Some(16), false));
                }
                assert!(projected_exact.len() <= exact.len());
                assert!(projected_bucketed.len() <= bucketed.len());
                assert!(coin_bucketed.len() <= bucketed.len());
                assert!(combined_bucketed.len() <= coin_bucketed.len());
                if let Some((a, other)) = pair {
                    cases.push((
                        b.state_key128(),
                        game["game"].clone(),
                        decision,
                        side,
                        a,
                        other,
                    ));
                }
                rows.push(json!({"game":game["game"],"decision":decision,"side":side,
                    "key":b.state_key128().to_string(),"pool_candidates":belief.candidate_count(),
                    "samples":256,"omitted_mons":omitted,"exact":exact.len(),"projected_exact":projected_exact.len(),
                    "bucketed":bucketed.len(),"projected_bucketed":projected_bucketed.len(),
                    "coin_eligible_samples":coin_eligible_samples,"coin_bucketed":coin_bucketed.len(),"combined_bucketed":combined_bucketed.len()}));
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| acts[s][i as usize])),
            )
            .unwrap();
            for observer in &mut observers {
                observer.observe(&b, &dex);
            }
        }
        eprintln!("key census game {} complete", game["game"]);
    }
    let mut kernels = Vec::new();
    if verify {
        cases.sort_by_key(|c| (c.0, c.3));
        cases.truncate(5);
        assert_eq!(cases.len(), 5);
        for (key, game, decision, side, mut a, mut b) in cases {
            let actions = [0, 1].map(|s| a.legal_choices(&dex, s));
            assert_eq!(actions, [0, 1].map(|s| b.legal_choices(&dex, s)));
            let mut cells = Vec::new();
            for &x in &actions[0] {
                for &y in &actions[1] {
                    let mut maps = Vec::new();
                    let mut runs = 0;
                    let mut pending = Vec::new();
                    for original in [&a, &b] {
                        let result =
                            enumerate_step_partial(&dex, original, [Some(x), Some(y)], 50000);
                        runs += result.runs;
                        pending.push(result.pending.iter().map(|p| p.mass).sum::<f64>());
                        let mut map = BTreeMap::<u128, f64>::new();
                        for leaf in result.leaves {
                            let mut copy = leaf.battle.clone();
                            project(&mut copy);
                            *map.entry(copy.state_key128()).or_default() += leaf.prob;
                        }
                        assert!(
                            (map.values().sum::<f64>() + pending.last().unwrap() - 1.0).abs()
                                < 1e-8
                        );
                        maps.push(map);
                    }
                    let complete = pending.iter().all(|&p| p == 0.0);
                    let equal = complete
                        && maps[0].len() == maps[1].len()
                        && maps[0]
                            .iter()
                            .all(|(k, p)| maps[1].get(k).is_some_and(|q| (p - q).abs() < 1e-12));
                    cells.push(json!({"actions":[x.to_input(&dex),y.to_input(&dex)],"runs":runs,"pending":pending,"equal":equal,"successors":[maps[0].len(),maps[1].len()]}));
                }
            }
            kernels.push(json!({"game":game,"decision":decision,"side":side,"key":key.to_string(),"full_keys":[a.state_key128().to_string(),b.state_key128().to_string()],"cells":cells}));
            eprintln!("kernel {}/5 complete", kernels.len());
        }
    }
    std::fs::write(&args[2], serde_json::to_vec(&json!({"rows":rows,"kernels":kernels,
        "scope":"Hypothetical identity projection: unselected roster entries are overwritten on hash-only copies. No projected state is simulated. Census measures possible key sharing. Optional full-support one-step comparisons cover five sampled state pairs only; they do not establish a universal quotient, speed or strength."})).unwrap()).unwrap();
}
