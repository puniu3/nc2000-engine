use nc2000_bot::{
    preview::load_meta_pool, smmcts::SelRule, Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{battle::PokemonSet, dex::Dex, state::Battle};
use serde_json::{json, Value};
use std::{io::BufRead, time::Instant};

fn snapshot(search: &BlindSearch, dex: &Dex) -> Value {
    json!({"iterations":search.iterations(),"best":search.best().unwrap().to_input(dex),
        "visits":search.visits(),"means":search.means(),"nodes":search.node_count()})
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 3, "corpus output");
    let dex = conformance::load_dex();
    let pool = load_meta_pool(std::path::Path::new("data/belief-pool-v3/belief-pool.json"));
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
            let actions = [0, 1].map(|s| b.legal_choices(&dex, s));
            if b.turn > 0 && actions.iter().all(|a| a.len() > 1) {
                let active_claw = [0, 1].iter().any(|&s| {
                    b.active_id(s).is_some_and(|id| {
                        b.poke(id).item.is_some() && b.poke(id).item == dex.known_items.quickclaw
                    })
                });
                if active_claw || !b.quick_claw_roll_observable(&dex) {
                    for side in 0..2 {
                        let order = b
                            .state_key128()
                            .wrapping_mul(0x9e3779b97f4a7c15)
                            .wrapping_add(side as u128);
                        cases.push((
                            order,
                            active_claw,
                            game["game"].clone(),
                            decision,
                            side,
                            b.clone(),
                            observers[side].clone(),
                        ));
                    }
                }
            }
            b.apply_choices(
                &dex,
                [0, 1].map(|s| d["choices"][s].as_u64().map(|i| actions[s][i as usize])),
            )
            .unwrap();
            for obs in &mut observers {
                obs.observe(&b, &dex);
            }
        }
    }
    let available = [false, true].map(|active| cases.iter().filter(|c| c.1 == active).count());
    cases.sort_by_key(|c| c.0);
    let mut retained = [0, 0];
    cases.retain(|c| {
        let i = usize::from(c.1);
        retained[i] += 1;
        c.1 || retained[i] <= 12
    });
    assert_eq!(available, [734, 44]);
    assert_eq!(cases.len(), 56);
    let cfg = RmConfig {
        c: 0.4,
        rule: SelRule::Ucb,
        ..Default::default()
    };
    let mut rows = Vec::new();
    for (index, (_, active_claw, game, decision, side, battle, observer)) in
        cases.into_iter().enumerate()
    {
        let belief = Belief::new(&dex, &pool, &observer);
        let seed = 0x2026102300 ^ battle.state_key128() as u64 ^ side as u64;
        BlindSearch::new(&battle, &dex, cfg.clone(), side, seed)
            .step(&dex, &belief, &observer, 1000);
        let mut fixed = Vec::new();
        let mut baseline_times = Vec::new();
        for repeat in 0..3 {
            for enabled in if (index + repeat) % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let start = Instant::now();
                let mut search = BlindSearch::new(
                    &battle,
                    &dex,
                    RmConfig {
                        unseen_claw_root: enabled,
                        ..cfg.clone()
                    },
                    side,
                    seed,
                );
                search.step(&dex, &belief, &observer, 27000);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                if !enabled {
                    baseline_times.push(elapsed_ms);
                }
                fixed.push(json!({"enabled":enabled,"repeat":repeat,"elapsed_ms":elapsed_ms,"snapshot":snapshot(&search,&dex)}));
            }
        }
        baseline_times.sort_by(f64::total_cmp);
        let budget_ms = baseline_times[1];
        let mut timed = Vec::new();
        for enabled in if index % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            let start = Instant::now();
            let mut search = BlindSearch::new(
                &battle,
                &dex,
                RmConfig {
                    unseen_claw_root: enabled,
                    ..cfg.clone()
                },
                side,
                seed,
            );
            let mut elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
            let mut last_batch_ms = 0.;
            while elapsed_ms < budget_ms {
                let before = elapsed_ms;
                search.step(&dex, &belief, &observer, 64);
                elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                last_batch_ms = elapsed_ms - before;
            }
            assert!(elapsed_ms - budget_ms <= last_batch_ms);
            timed.push(
                json!({"enabled":enabled,"budget_ms":budget_ms,"elapsed_ms":elapsed_ms,
                "last_batch_ms":last_batch_ms,"snapshot":snapshot(&search,&dex)}),
            );
        }
        rows.push(json!({"game":game,"decision":decision,"side":side,"seed":seed,"key":battle.state_key128().to_string(),
            "active_claw":active_claw,"idle":battle.is_idle_move_request(),"fixed":fixed,"timed":timed}));
        eprintln!("claw budget {}/56 complete", index + 1);
    }
    std::fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"available":available,"iterations":27000,"fixed_repeats":3,
        "timed_batch":64,"rows":rows}),
        )
        .unwrap(),
    )
    .unwrap();
}
