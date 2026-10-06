use nc2000_bot::{smmcts::SearchTrace, Belief, BlindSearch, Observer, RmConfig};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    state::{Battle, RequestState},
};
use std::collections::HashMap;

type Step = (Option<SearchChoice>, RequestState, Vec<String>);
type Key = (Vec<Step>, Vec<SearchChoice>);

#[test]
fn every_history_group_matches_independent_replay_on_both_sides() {
    let dex = conformance::load_dex();
    let teams=["Snorlax","Porygon2"].map(|bench| {
        ["Smeargle",bench,"Starmie"].map(|species|serde_json::from_value::<PokemonSet>(serde_json::json!({
            "name":species,"species":species,"level":50,
            "moves":if species=="Smeargle" {vec!["Baton Pass","Tackle","Curse"]} else {vec!["Tackle","Rest","Curse"]}
        })).unwrap()).to_vec()
    });
    let mut b = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
    let mut obs = [0, 1].map(|side| Observer::new(&b, side));
    b.choose(&dex, 0, "team 1,2,3").unwrap();
    b.choose(&dex, 1, "team 1,2,3").unwrap();
    for observer in &mut obs {
        observer.observe(&b, &dex);
    }
    for side in 0..2 {
        let belief = Belief::pinned_from_battle(&b, &obs[side]);
        for shared in [false, true] {
            let cfg = RmConfig {
                horizon: 3,
                c: 0.4,
                ..Default::default()
            };
            let mut bulk = BlindSearch::new(&b, &dex, cfg.clone(), side, 73);
            bulk.enable_history_continuation(shared);
            bulk.step(&dex, &belief, &obs[side], 2048);
            let mut traced = BlindSearch::new(&b, &dex, cfg.clone(), side, 73);
            traced.enable_history_continuation(shared);
            let mut expected = HashMap::<Key, (Vec<u32>, Vec<f64>)>::new();
            let mut shadow: Option<Battle> = None;
            let mut history = Vec::new();
            let mut pending = Vec::new();
            let mut switches = 0;
            for n in [31, 2017] {
                traced.step_observed(&dex, &belief, &obs[side], n, &mut |event| match event {
                    SearchTrace::Choice {
                        battle,
                        actions,
                        chosen,
                        ..
                    } => {
                        if shadow.is_none() {
                            let mut copy = battle.clone();
                            copy.set_log_enabled(true);
                            shadow = Some(copy);
                        } else {
                            assert_eq!(
                                shadow.as_ref().unwrap().state_key128(),
                                battle.state_key128()
                            );
                            if actions[side].len() > 1 {
                                switches +=
                                    usize::from(battle.request_state == RequestState::Switch);
                                let key = (history.clone(), actions[side].clone());
                                let action = actions[side]
                                    .iter()
                                    .position(|&a| Some(a) == chosen[side])
                                    .unwrap();
                                expected
                                    .entry(key.clone())
                                    .or_insert_with(|| {
                                        (
                                            vec![0; actions[side].len()],
                                            vec![0.; actions[side].len()],
                                        )
                                    })
                                    .0[action] += 1;
                                pending.push((key, action));
                            }
                        }
                        let copy = shadow.as_mut().unwrap();
                        let start = copy.log.len();
                        copy.apply_choices(&dex, chosen).unwrap();
                        history.push((
                            chosen[side],
                            copy.request_state,
                            copy.log[start..].to_vec(),
                        ));
                    }
                    SearchTrace::Result { reward0, .. } => {
                        for (key, action) in pending.drain(..) {
                            expected.get_mut(&key).unwrap().1[action] +=
                                if side == 0 { reward0 } else { 1. - reward0 };
                        }
                        shadow = None;
                        history.clear();
                    }
                    _ => {}
                });
            }
            assert_eq!(bulk.visits(), traced.visits());
            assert_eq!(bulk.means(), traced.means());
            assert_eq!(bulk.search_rng_state(), traced.search_rng_state());
            if !shared {
                let mut plain = BlindSearch::new(&b, &dex, cfg, side, 73);
                plain.step(&dex, &belief, &obs[side], 2048);
                assert_eq!(plain.visits(), bulk.visits());
                assert_eq!(plain.means(), bulk.means());
                assert_eq!(plain.search_rng_state(), bulk.search_rng_state());
            }
            let policy = traced.history_continuation().unwrap();
            assert!(policy.histories().iter().any(|h| h.depth >= 3));
            assert_eq!(policy.groups().len(), expected.len());
            for g in policy.groups() {
                let mut id = g.history;
                let mut steps = Vec::new();
                while id != 0 {
                    let h = &policy.histories()[id - 1];
                    steps.push((h.own_action, h.request, h.log.clone()));
                    id = h.parent;
                }
                steps.reverse();
                let (visits, rewards) = expected.remove(&(steps, g.actions.clone())).unwrap();
                assert_eq!(g.visits, visits);
                assert_eq!(g.rewards, rewards);
            }
            assert!(expected.is_empty());
            assert!(switches > 0);
        }
    }
}
