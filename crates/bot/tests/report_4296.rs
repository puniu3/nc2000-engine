use conformance::{fixture::repo_root, load_dex};
use nc2000_bot::{
    position::{synthesize_spec, PositionSpec},
    preview::load_meta_pool,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, PokemonSet},
    dex::Dex,
    state::Battle,
    validate::{validate_team, Learnsets},
};

fn position(dex: &Dex, turn: u16) -> Battle {
    let root = repo_root();
    let dir = root.join("data/report-4296");
    let spec = PositionSpec::parse(
        &std::fs::read_to_string(dir.join(format!("turn-{turn}.json"))).unwrap(),
    )
    .unwrap();
    let sets: Vec<PokemonSet> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("opponent-team.json")).unwrap())
            .unwrap();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    synthesize_spec(dex, &spec, &pool, Some(&sets), 61001).unwrap()
}

fn ko_probabilities(dex: &Dex, b: &Battle, action: &str, reply: &str) -> (f64, f64) {
    let mut b = b.clone();
    let actor = b.active_id(1).unwrap();
    let target = b.active_id(0).unwrap();
    let mut joint = [None, None];
    for (side, input) in [(0, reply), (1, action)] {
        let canonical = input
            .strip_prefix("move ")
            .and_then(|m| dex.moves.id(m))
            .map(|id| format!("move {}", dex.moves.key(dex.canonical_move_id(id))));
        joint[side] = Some(
            *b.legal_choices(dex, side)
                .iter()
                .find(|a| {
                    let s = a.to_input(dex);
                    s == input || canonical.as_deref() == Some(s.as_str())
                })
                .unwrap(),
        );
    }
    let step = enumerate_step(dex, &b, joint, 200_000).unwrap();
    assert!((step.leaves.iter().map(|l| l.prob).sum::<f64>() - 1.0).abs() < 1e-9);
    let ko = |id| {
        step.leaves
            .iter()
            .filter(|l| l.battle.poke(id).fainted)
            .map(|l| l.prob)
            .sum::<f64>()
    };
    (ko(actor), ko(target))
}

#[test]
fn frozen_evaluation_preserves_training_and_simultaneous_root_information() {
    use nc2000_bot::{frozen::FrozenPolicy, import::ProtocolAgent, smmcts::RmConfig};
    let dex = load_dex();
    let root = repo_root();
    let spec = PositionSpec::parse(&std::fs::read_to_string(root.join("data/report-4296/turn-11.json")).unwrap()).unwrap();
    let sets: Vec<PokemonSet> = serde_json::from_str(&std::fs::read_to_string(root.join("data/report-4296/opponent-team.json")).unwrap()).unwrap();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    for pinned in [false, true] {
        let [mut evaluated, mut ordinary] = std::array::from_fn(|_| {
            let mut agent = ProtocolAgent::new(&dex, spec.side, pool.clone(), RmConfig::default(), 61001);
            if pinned { agent.pin_opponent(sets.clone()); }
            agent.set_position(&dex, &spec).unwrap();
            agent.step(&dex, 600).unwrap();
            agent
        });
        let search = evaluated.search().unwrap();
        if pinned {
            let earthquake = *search.actions().iter().find(|c| c.to_input(&dex) == "move earthquake").unwrap();
            let mut base = evaluated.resynthesize(&dex, 991).unwrap();
            let replies: Vec<_> = base.legal_choices(&dex, 1-spec.side).into_iter()
                .filter(|c| matches!(c, nc2000_engine::battle::SearchChoice::Move(_))).collect();
            assert_eq!(replies.len(), 4);
            for seed in 991..1023 {
                let results: Vec<_> = replies.iter().map(|&reply| {
                    let mut forced = [None, None];
                    forced[spec.side] = Some(earthquake);
                    forced[1-spec.side] = Some(reply);
                    search.evaluate_frozen_joint(&dex, evaluated.belief().unwrap(), evaluated.observer().unwrap(),
                        forced, [FrozenPolicy::MostVisited; 2], seed, &mut |_| {}).reward0
                }).collect();
                assert!(results.iter().all(|r| *r == results[0]));
            }
        }
        let mut reference_root = None;
        for policy in [FrozenPolicy::MostVisited, FrozenPolicy::SampleVisits] {
            for &action in search.actions() {
                let mut repeated = Vec::new();
                for _ in 0..2 {
                    let mut seen_root = false;
                    let result = search.evaluate_frozen(&dex, evaluated.belief().unwrap(), evaluated.observer().unwrap(),
                        action, [policy; 2], 991, &mut |event| {
                            let mut copy = event.battle.clone();
                            for side in 0..2 {
                                if let Some(choice) = event.chosen[side] {
                                    assert!(copy.legal_choices(&dex, side).contains(&choice));
                                }
                            }
                            if !seen_root {
                                assert_eq!(event.chosen[spec.side], Some(action));
                                let root = (event.battle.state_key_bucketed(0), event.chosen[1-spec.side]);
                                assert_eq!(*reference_root.get_or_insert(root), root);
                                seen_root = true;
                            }
                        });
                    assert!(seen_root);
                    assert!((0.0..=1.0).contains(&result.reward0));
                    repeated.push((result.reward0, result.terminal, result.prefix_choices));
                }
                assert_eq!(repeated[0], repeated[1]);
            }
        }
        for extra in [0, 100] {
            evaluated.step(&dex, extra).unwrap();
            ordinary.step(&dex, extra).unwrap();
            let a = evaluated.search().unwrap();
            let b = ordinary.search().unwrap();
            assert_eq!(a.iterations(), b.iterations());
            assert_eq!(a.visits(), b.visits());
            assert_eq!(a.means(), b.means());
            assert_eq!(a.root_matrix(), b.root_matrix());
            assert_eq!(a.node_count(), b.node_count());
        }
    }
}

#[test]
fn observing_search_preserves_statistics_and_subsequent_rng() {
    use nc2000_bot::{import::ProtocolAgent, mcts::Playout, smmcts::{RmConfig, SearchTrace}};
    let dex = load_dex();
    let root = repo_root();
    let spec = PositionSpec::parse(&std::fs::read_to_string(root.join("data/report-4296/turn-11.json")).unwrap()).unwrap();
    let sets: Vec<PokemonSet> = serde_json::from_str(&std::fs::read_to_string(root.join("data/report-4296/opponent-team.json")).unwrap()).unwrap();
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    for playout in [Playout::heavy(), Playout::Uniform] {
        let cfg = RmConfig { playout, ..RmConfig::default() };
        let [mut observed, mut ordinary] = std::array::from_fn(|_| {
            let mut agent = ProtocolAgent::new(&dex, 1, pool.clone(), cfg.clone(), 61001);
            agent.pin_opponent(sets.clone());
            agent.set_position(&dex, &spec).unwrap();
            agent
        });
        let mut results = 0;
        let mut rewards = 0.0;
        let mut choices = 0;
        observed.step_observed(&dex, 1200, &mut |event| match event {
            SearchTrace::Choice { battle, chosen, .. } => {
                choices += 1;
                assert_ne!(chosen, [None, None]);
                let mut copy = battle.clone();
                copy.apply_choices(&dex, chosen).unwrap();
            },
            SearchTrace::Leaf { rng, .. } => {
                let mut copy = rng.clone();
                std::hint::black_box(copy.next());
            },
            SearchTrace::Result { reward0, .. } => {
                results += 1;
                rewards += 1.0-reward0;
            },
        }).unwrap();
        ordinary.step(&dex, 1200).unwrap();
        assert_eq!(results, 1200);
        assert!(choices >= results);
        let search = observed.search().unwrap();
        let backed_up: f64 = search.visits().iter().zip(search.means()).map(|(n,w)|*n as f64*w).sum();
        assert!((rewards-backed_up).abs()<1e-9);
        for extra in [0, 100] {
            observed.step(&dex, extra).unwrap();
            ordinary.step(&dex, extra).unwrap();
            let a = observed.search().unwrap();
            let b = ordinary.search().unwrap();
            assert_eq!(a.visits(), b.visits());
            assert_eq!(a.means(), b.means());
            assert_eq!(a.root_matrix(), b.root_matrix());
            assert_eq!(a.node_count(), b.node_count());
        }
    }
}

#[test]
fn submitted_sheet_is_legal_and_pinned_with_consumed_items_preserved() {
    let dex = load_dex();
    let root = repo_root();
    let ls = Learnsets::from_json(
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    let text = std::fs::read_to_string(root.join("data/report-4296/opponent-team.json")).unwrap();
    assert_eq!(validate_team(&dex, &ls, &text)["ok"], true);
    let canonical: serde_json::Value = serde_json::from_str(&text).unwrap();
    let submitted: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/report-4296/opponent-submitted.json")).unwrap(),
    )
    .unwrap();
    for slot in 0..6 {
        for key in ["species", "level", "item", "moves", "evs", "ivs"] {
            assert_eq!(canonical[slot][key], submitted[slot][key]);
        }
    }
    let b = position(&dex, 26);
    let starmie = &b.sides[0].roster[0];
    assert_eq!(dex.items.key(starmie.item.unwrap()), "miracleberry");
    assert_eq!(
        starmie
            .base_move_slots
            .iter()
            .map(|m| dex.moves.key(m.id))
            .collect::<Vec<_>>(),
        ["surf", "psychic", "confuseray", "recover"]
    );
    assert!(b.sides[0].roster[2].item.is_none());
    assert!(b.sides[1].roster[0].item.is_none());
    assert_eq!(
        dex.items.key(b.sides[0].roster[5].item.unwrap()),
        "quickclaw"
    );
    assert_eq!(b.sides[0].pokemon_left, 3);
    assert_eq!(b.sides[1].pokemon_left, 1);
}

#[test]
fn earthquake_prevents_shuckles_recorded_rest_at_turn_11() {
    let dex = load_dex();
    let mut b = position(&dex, 11);
    let target = b.active_id(0).unwrap();
    for hp in [70, 71] {
        b.poke_mut(target).hp = hp;
        assert_eq!(
            ko_probabilities(&dex, &b, "move earthquake", "move rest"),
            (0.0, 1.0)
        );
        let (_, hp_ko) = ko_probabilities(&dex, &b, "move hiddenpowerbug", "move rest");
        assert!((hp_ko - 1.0 / 16.0).abs() < 1e-9);
    }
}

#[test]
fn a_critical_ko_can_save_poisoned_marowak_at_turn_24() {
    let dex = load_dex();
    let mut b = position(&dex, 24);
    let actor = b.active_id(1).unwrap();
    for hp in [6, 7] {
        b.poke_mut(actor).hp = hp;
        let (actor_ko, _) = ko_probabilities(&dex, &b, "move earthquake", "switch 2");
        assert!((actor_ko - 15.0 / 16.0).abs() < 1e-9);
        let (actor_ko, _) = ko_probabilities(&dex, &b, "move swordsdance", "switch 2");
        assert_eq!(actor_ko, 1.0);
    }
}

#[test]
fn confusion_does_not_make_fire_punch_as_likely_to_ko_as_thunderbolt() {
    let dex = load_dex();
    let mut b = position(&dex, 26);
    let target = b.active_id(0).unwrap();
    for hp in [22, 23] {
        b.poke_mut(target).hp = hp;
        assert_eq!(
            nc2000_bot::import::announce_hp(hp, b.poke(target).maxhp, 100),
            13
        );
        let (_, thunderbolt) = ko_probabilities(&dex, &b, "move thunderbolt", "move confuseray");
        let (_, fire_punch) = ko_probabilities(&dex, &b, "move firepunch", "move confuseray");
        assert!((thunderbolt - 0.5).abs() < 1e-9);
        assert!(fire_punch < thunderbolt);
        if hp == 23 {
            assert!((fire_punch - 1.0 / 32.0).abs() < 1e-9);
        } else {
            assert!(fire_punch > 1.0 / 32.0);
        }
    }
}
