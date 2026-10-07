use nc2000_bot::{
    preview::{load_meta_pool, MetaPool}, rng, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[path = "../src/exp3.rs"]
mod exp3;

fn replay(dex: &Dex, game: &Value, row: &Value, alternate: bool) -> (Battle, Observer) {
    let side = row["side"].as_u64().unwrap() as usize;
    let mut teams: [Vec<PokemonSet>; 2] = serde_json::from_value(game["teams"].clone()).unwrap();
    if alternate {
        teams[1 - side][row["foe_slot"].as_u64().unwrap() as usize].moves
            [row["move_slot"].as_u64().unwrap() as usize] =
            row["replacement"].as_str().unwrap().into();
    }
    let mut b = Battle::from_fixture(dex, game["seed"].as_str().unwrap(), &teams[0], &teams[1]).unwrap();
    b.set_log_enabled(true);
    let mut obs = Observer::new(&b, side);
    for d in game["decisions"].as_array().unwrap().iter().take(row["decision"].as_u64().unwrap() as usize) {
        if !alternate { assert_eq!(b.state_key128().to_string(), d["key"].as_str().unwrap()); }
        let actions = [0, 1].map(|s| b.legal_choices(dex, s));
        let choices = [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize]));
        b.apply_choices(dex, choices).unwrap();
        obs.observe(&b, dex);
    }
    assert_eq!(b.state_key128().to_string(), row[if alternate { "alternate_key" } else { "key" }].as_str().unwrap());
    (b, obs)
}

fn active_model(dex: &Dex, b: &Battle) -> Value {
    let essence = b.essence(dex);
    json!([0, 1].map(|side| {
        assert_eq!(b.sides[side].pokemon_left, 1);
        let p = b.poke(b.active_id(side).unwrap());
        let mut active = essence["sides"][side]["pokemon"].as_array().unwrap().iter()
            .find(|v| v["active"] == true).unwrap().clone();
        active["moves"].as_array_mut().unwrap().sort_by_key(|v| v["id"].as_str().unwrap().to_string());
        json!({"active":active,"stats":p.stored_stats,"happiness":p.happiness,
            "hp_type":format!("{:?}",p.hp_type),"hp_power":p.hp_power})
    }))
}

fn check_model(dex: &Dex, worlds: &[(Battle, Observer)], side: usize, reflect: usize) -> Value {
    let mut models = 0;
    let mut joints = 0;
    for (w, (b, obs)) in worlds.iter().enumerate() {
        let belief = Belief::pinned_from_battle(b, obs);
        let expected = active_model(dex, b);
        let replies = b.clone().legal_choices(dex, 1 - side);
        for template in [0, reflect] {
            for seed in 0..8 {
                let mut rng = rng::SplitMix64::new(seed);
                let mut sim = belief.determinize(dex, &worlds[template].0, &worlds[template].1, &mut rng);
                assert_eq!(active_model(dex, &sim), expected, "world={w} template={template}");
                let actual = sim.legal_choices(dex, 1 - side);
                assert_eq!(actual.len(), replies.len());
                assert!(replies.iter().all(|r| actual.contains(r)));
                models += 1;
            }
        }
        let mut search = BlindSearch::new(&worlds[0].0, dex,
            RmConfig { c: 0.4, rule: SelRule::Ucb, ..Default::default() }, side, 0x4b3900 + w as u64);
        search.enable_compact_history_continuation(true);
        let actions = search.actions().to_vec();
        for (a, action) in actions.iter().enumerate() {
            for &reply in &replies {
                let mut checked = false;
                search.step_forced_joint_observed(dex, &belief, &worlds[0].1, a, reply, &mut |event| {
                    if let nc2000_bot::smmcts::SearchTrace::Choice { battle, chosen, .. } = event {
                        if !checked {
                            assert_eq!(active_model(dex, battle), expected);
                            assert_eq!(chosen[side], Some(*action));
                            assert_eq!(chosen[1 - side], Some(reply));
                            checked = true;
                        }
                    }
                });
                assert!(checked);
                joints += 1;
            }
        }
    }
    json!({"active_model_checks":models,"forced_joint_checks":joints})
}

fn run(
    dex: &Dex,
    worlds: &[(Battle, Observer)],
    pool: &MetaPool,
    side: usize,
    template: usize,
    mode: &str,
    seed: u64,
    timed: bool,
) -> Value {
    let start = Instant::now();
    let beliefs: Vec<_> = if mode == "catalog" { vec![Belief::new(dex, pool, &worlds[template].1)] } else {
        worlds.iter().map(|(b, obs)| Belief::pinned_from_battle(b, obs)).collect()
    };
    let mut search = BlindSearch::new(&worlds[template].0, dex,
        RmConfig { c: 0.4, rule: SelRule::Ucb, ..Default::default() }, side, seed);
    if mode != "catalog" { search.enable_compact_history_continuation(true); }
    let arms: Vec<_> = worlds.iter().enumerate().flat_map(|(i, (b, _))|
        b.clone().legal_choices(dex, 1 - side).into_iter().map(move |reply| (i, reply))).collect();
    let mut own = exp3::Exp3Ix::new(search.actions().len());
    let mut adversary = exp3::Exp3Ix::new(arms.len());
    let mut own_rng = rng::SplitMix64::new(seed ^ 0xb1e54f83902ca761);
    let mut world_rng = rng::SplitMix64::new(seed ^ 0x54bc170a2998d6e3);
    let mut world_n = vec![0u32; worlds.len()];
    let mut arm_n = vec![0u32; arms.len()];
    let mut reward_sum = 0.;
    let setup_seconds = start.elapsed().as_secs_f64();
    while if timed { start.elapsed() < Duration::from_millis(500) } else { search.iterations() < 27000 } {
        let reward = match mode {
            "catalog" => search.step_one(dex, &beliefs[0], &worlds[template].1),
            "uniform" => {
                let w = world_rng.below(worlds.len());
                world_n[w] += 1;
                search.step_one(dex, &beliefs[w], &worlds[template].1)
            },
            "robust" => {
                let x = own.sample(&mut own_rng);
                let y = adversary.sample(&mut world_rng);
                let (w, reply) = arms[y.action];
                world_n[w] += 1;
                arm_n[y.action] += 1;
                let r = search.step_forced_joint(dex, &beliefs[w], &worlds[template].1, x.action, reply);
                let own_reward = if side == 0 { r } else { 1. - r };
                own.observe(x, 1. - own_reward);
                adversary.observe(y, own_reward);
                r
            },
            _ => unreachable!(),
        };
        reward_sum += if side == 0 { reward } else { 1. - reward };
    }
    let seconds = start.elapsed().as_secs_f64();
    assert_eq!(search.visits().iter().sum::<u32>(), search.iterations());
    if mode != "catalog" { assert_eq!(world_n.iter().sum::<u32>(), search.iterations()); }
    if mode == "robust" { assert_eq!(arm_n.iter().sum::<u32>(), search.iterations()); }
    json!({"mode":mode,"seed":seed,"template":template,"timed":timed,"iterations":search.iterations(),
        "seconds":seconds,"setup_seconds":setup_seconds,"best":search.best().unwrap().to_input(dex),
        "visits":search.visits(),"means":search.means(),"rng":search.search_rng_state(),
        "nodes":search.node_count(),"world_counts":world_n,"arm_counts":arm_n,
        "own_rng":own_rng.0,"world_rng":world_rng.0,"reward_sum":reward_sum})
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "corpus family output");
    let cases: Vec<Value> = serde_json::from_slice(&std::fs::read(&args[2]).unwrap()).unwrap();
    assert_eq!(cases.len(), 38);
    let game: Value = std::fs::read_to_string(&args[1]).unwrap().lines()
        .map(|s| serde_json::from_str::<Value>(s).unwrap()).find(|g| g["game"] == cases[0]["game"]).unwrap();
    let dex = conformance::load_dex();
    let side = cases[0]["side"].as_u64().unwrap() as usize;
    let mut worlds = vec![replay(&dex, &game, &cases[0], false)];
    let mut names = vec![cases[0]["original_move"].as_str().unwrap().to_string()];
    for row in &cases {
        assert_eq!(row["game"], cases[0]["game"]);
        assert_eq!(row["decision"], cases[0]["decision"]);
        let mut world = replay(&dex, &game, row, true);
        assert_eq!(world.0.log, worlds[0].0.log);
        assert_eq!(format!("{:?}", world.1.mons()), format!("{:?}", worlds[0].1.mons()));
        assert_eq!(world.0.legal_choices(&dex, side), worlds[0].0.legal_choices(&dex, side));
        worlds.push(world);
        names.push(row["replacement"].as_str().unwrap().into());
    }
    let reflect = names.iter().position(|s| s == "reflect").unwrap();
    let checks = check_model(&dex, &worlds, side, reflect);
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
    
    let arms: Vec<_> = worlds.iter_mut().enumerate().flat_map(|(i,(b,_))|
        b.legal_choices(&dex, 1-side).into_iter().map(move |r| (i,r))).collect();
    let arm_labels: Vec<_> = arms.iter().map(|(w,r)| json!({"world":names[*w],"reply":r.to_input(&dex)})).collect();
    let actions: Vec<_> = worlds[0].0.legal_choices(&dex, side).iter().map(|a| a.to_input(&dex)).collect();
    let mut rows = Vec::new();
    for timed in [false, true] {
        for seed_index in 0..16 {
            let order = if seed_index % 2 == 0 { ["catalog", "uniform", "robust"] } else { ["robust", "uniform", "catalog"] };
            for mode in order {
                for template in if timed { vec![0] } else { vec![0, reflect] } {
                    let row = run(&dex, &worlds, &pool, side, template, mode, 0x2026110301 + seed_index, timed);
                    eprintln!("{mode} timed={timed} seed={seed_index} template={template} best={} iterations={} seconds={}",row["best"],row["iterations"],row["seconds"]);
                    rows.push(row);
                    std::fs::write(&args[3],serde_json::to_vec(&json!({"cases":cases,"worlds":names,"arms":arm_labels,"actions":actions,"checks":checks,"rows":rows})).unwrap()).unwrap();
                }
            }
        }
    }
}
