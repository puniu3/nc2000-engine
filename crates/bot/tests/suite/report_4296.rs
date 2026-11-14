use super::common;
use conformance::fixture::repo_root;
use nc2000_bot::{
    decision::PolicyMode,
    import::ProtocolAgent,
    mcts::Playout,
    position::{synthesize_spec, PositionSpec},
    smmcts::{RmConfig, SearchTrace},
};
use nc2000_engine::{
    battle::PokemonSet,
    dex::Dex,
    state::Battle,
    validate::validate_team,
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
    let pool = common::meta_pool_v0();
    synthesize_spec(dex, &spec, pool, Some(&sets), 61001).unwrap()
}

#[test]
fn observing_search_preserves_statistics_and_subsequent_rng() {
    let dex = common::dex();
    let root = repo_root();
    let spec = PositionSpec::parse(&std::fs::read_to_string(root.join("data/report-4296/turn-11.json")).unwrap()).unwrap();
    let sets: Vec<PokemonSet> = serde_json::from_str(&std::fs::read_to_string(root.join("data/report-4296/opponent-team.json")).unwrap()).unwrap();
    let pool = common::meta_pool_v0();
    for playout in [Playout::heavy(), Playout::Uniform] {
        let cfg = RmConfig { playout, ..RmConfig::default() };
        let [mut observed, mut ordinary] = std::array::from_fn(|_| {
            let mut agent = ProtocolAgent::new(dex, 1, pool.clone(), cfg.clone(), 61001);
            agent.pin_opponent(dex, sets.clone()).unwrap();
            agent.set_position(dex, &spec).unwrap();
            agent
        });
        let mut results = 0;
        let mut choices = 0;
        observed.step_observed(dex, 200, &mut |event| match event {
            SearchTrace::Choice { battle, chosen, .. } => {
                choices += 1;
                assert_ne!(chosen, [None, None]);
                let mut copy = battle.clone();
                copy.apply_choices(dex, chosen).unwrap();
            },
            SearchTrace::Leaf { rng, .. } => {
                let mut copy = rng.clone();
                std::hint::black_box(copy.next());
            },
            SearchTrace::Result { .. } => {
                results += 1;
            },
        }).unwrap();
        ordinary.step(dex, 200).unwrap();
        assert_eq!(results, 200);
        assert!(choices >= results);
        for extra in [0, 50] {
            observed.step(dex, extra).unwrap();
            ordinary.step(dex, extra).unwrap();
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
    let dex = common::dex();
    let root = repo_root();
    let ls = common::learnsets();
    let text = std::fs::read_to_string(root.join("data/report-4296/opponent-team.json")).unwrap();
    assert_eq!(validate_team(dex, ls, &text)["ok"], true);
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
fn protocol_agent_makes_deterministic_legal_decision_on_report_4296() {
    let dex = common::dex();
    let root = repo_root();
    let dir = root.join("data/report-4296");
    let sets: Vec<PokemonSet> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("opponent-team.json")).unwrap())
            .unwrap();
    let pool = common::meta_pool_v0();
    for turn in [11u16, 24, 26] {
        let spec = PositionSpec::parse(
            &std::fs::read_to_string(dir.join(format!("turn-{turn}.json"))).unwrap(),
        )
        .unwrap();
        let [mut a, mut b] = std::array::from_fn(|_| {
            let mut agent = ProtocolAgent::new(dex, 1, pool.clone(), RmConfig::default(), 61001);
            agent.pin_opponent(dex, sets.clone()).unwrap();
            agent.set_position(dex, &spec).unwrap();
            agent
        });
        assert_eq!(a.step(dex, 200).unwrap(), 200);
        assert_eq!(b.step(dex, 200).unwrap(), 200);
        let best_a = a.best(dex).unwrap();
        let best_b = b.best(dex).unwrap();
        assert_eq!(best_a, best_b, "turn {turn} search must be seed-deterministic");
        assert_eq!(a.legality_drift, 0, "turn {turn} had legality drift");
        let policy = a.decision_policy(dex, PolicyMode::EstimatedMix).unwrap();
        let total_prob: f64 = policy.actions().iter().map(|row| row.probability).sum();
        assert!((total_prob - 1.0).abs() < 1e-6);
        assert!(policy.actions().iter().any(|row| row.input == best_a || best_a.starts_with("move ")));
    }
}

#[test]
fn default_search_uses_eight_turn_rollout_without_experimental_bundle() {
    let cfg = RmConfig::default();
    assert_eq!(
        cfg.settlement,
        nc2000_bot::accounting::LeafSettlementMode::BlindSpotLeaf
    );
    assert!(matches!(
        cfg.playout,
        Playout::Heavy { turns: 8, .. }
    ));
    let dex = common::dex();
    let team = vec![serde_json::from_value::<PokemonSet>(serde_json::json!({
        "name": "Starmie", "species": "Starmie", "item": "Leftovers", "ability": "",
        "level": 50, "moves": ["Recover"], "happiness": 255,
        "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
        "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30}
    }))
    .unwrap()];
    let mut battle = Battle::from_fixture(dex, "1,2,3,4", &team, &team).unwrap();
    let picks = [0, 1].map(|side| battle.legal_choices(dex, side).into_iter().next());
    battle.apply_choices(dex, picks).unwrap();
    let turn = battle.turn;
    let value = nc2000_bot::mcts::playout_value(
        &mut battle,
        dex,
        &cfg.playout,
        turn + cfg.horizon,
        &mut nc2000_bot::SplitMix64::new(777),
        cfg.rollout_rules(),
    );
    assert!(value.is_finite() && (0.0..=1.0).contains(&value));
    let end = nc2000_bot::mcts::last_sim_end();
    assert_eq!(end.outcome, 0);
    assert_eq!(end.turn, turn + 8);
}
