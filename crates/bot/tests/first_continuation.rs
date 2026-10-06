use nc2000_bot::{smmcts::SearchTrace, Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{battle::PokemonSet, state::Battle};

#[test]
fn first_continuation_counts_match_trace_and_survive_chunking_on_both_sides() {
    let dex = conformance::load_dex();
    let teams = ["Snorlax", "Porygon2"].map(|species| {
        vec![serde_json::from_value::<PokemonSet>(serde_json::json!({
            "species":species,"name":species,"level":50,"moves":["Curse","Tackle","Rest"]
        }))
        .unwrap()]
    });
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
    let mut observers = [0, 1].map(|s| Observer::new(&b, s));
    b.choose(&dex, 0, "team 1").unwrap();
    b.choose(&dex, 1, "team 1").unwrap();
    for obs in &mut observers {
        obs.observe(&b, &dex);
    }
    for side in 0..2 {
        let obs = &observers[side];
        let belief = Belief::pinned_from_battle(&b, obs);
        for shared in [false, true] {
            let cfg = RmConfig {
                horizon: 2,
                c: 0.4,
                ..Default::default()
            };
            let mut bulk = BlindSearch::new(&b, &dex, cfg.clone(), side, 71);
            bulk.enable_first_continuation(shared);
            bulk.step(&dex, &belief, obs, 1024);
            let mut traced = BlindSearch::new(&b, &dex, cfg, side, 71);
            traced.enable_first_continuation(shared);
            let mut total = 0;
            let mut reward_sum = 0.;
            let mut depth = 0;
            let mut eligible = false;
            for count in [31, 993] {
                traced.step_observed(&dex, &belief, obs, count, &mut |event| match event {
                    SearchTrace::Choice {
                        battle, actions, ..
                    } => {
                        if depth == 1 && battle.is_idle_move_request() && actions[side].len() > 1 {
                            eligible = true;
                        }
                        depth += 1;
                    }
                    SearchTrace::Result { reward0, .. } => {
                        if eligible {
                            total += 1;
                            reward_sum += if side == 0 { reward0 } else { 1. - reward0 };
                        }
                        eligible = false;
                        depth = 0;
                    }
                    _ => {}
                });
            }
            assert_eq!(bulk.visits(), traced.visits());
            assert_eq!(bulk.means(), traced.means());
            assert_eq!(bulk.search_rng_state(), traced.search_rng_state());
            assert_eq!(bulk.best(), traced.best());
            let groups = traced.first_continuation().unwrap().groups();
            assert!(total > 100);
            assert_eq!(
                groups.iter().flat_map(|g| g.visits.iter()).sum::<u32>(),
                total
            );
            assert!(
                (groups.iter().flat_map(|g| g.rewards.iter()).sum::<f64>() - reward_sum).abs()
                    < 1e-9
            );
            assert!(groups.iter().all(|g| !g.log.is_empty()));
            assert_eq!(
                bulk.first_continuation().unwrap().groups().len(),
                groups.len()
            );
        }
    }
}
