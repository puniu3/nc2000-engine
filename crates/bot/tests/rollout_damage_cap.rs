use conformance::load_dex;
use nc2000_bot::{
    mcts::{playout_pick, Playout, RolloutRules},
    rng::SplitMix64,
    RmConfig,
};
use nc2000_engine::{
    battle::{enumerate::enumerate_step_partial, Outcome, PokemonSet, SearchChoice},
    dex::Dex,
    state::Battle,
};

fn fixture(dex: &Dex, side: usize, low_hp: bool, moves: &[&str]) -> Battle {
    let set = |species: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "species":species,"name":species,"level":50,"moves":moves,
            "ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
            "evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}
        }))
        .unwrap()
    };
    let own = [set("Raikou", moves)];
    let foe = [set("Snorlax", &["Body Slam"])];
    let teams = if side == 0 {
        [&own, &foe]
    } else {
        [&foe, &own]
    };
    let mut b = Battle::from_fixture(dex, "1,2,3,4", teams[0], teams[1]).unwrap();
    b.choose(dex, 0, "team 1").unwrap();
    b.choose(dex, 1, "team 1").unwrap();
    if low_hp {
        for s in 0..2 {
            let id = b.active_id(s).unwrap();
            b.poke_mut(id).hp = 1;
        }
    }
    b
}

#[test]
fn capped_ranking_prefers_certified_winning_hit_over_overkill() {
    let dex = load_dex();
    let playout = Playout::Heavy {
        eps: 0.0,
        turns: 8,
        weights: Default::default(),
    };
    for side in 0..2 {
        let mut b = fixture(&dex, side, true, &["Thunder", "Hidden Power"]);
        let choices = b.legal_choices(&dex, side);
        let baseline = playout_pick(
            &b,
            &dex,
            &playout,
            side,
            &choices,
            &mut SplitMix64(0),
            RmConfig::default().rollout_rules(),
        );
        let candidate = playout_pick(
            &b,
            &dex,
            &playout,
            side,
            &choices,
            &mut SplitMix64(0),
            RmConfig {
                rollout_capped_damage: true,
                ..Default::default()
            }
            .rollout_rules(),
        );
        assert_eq!(
            baseline,
            SearchChoice::Move(dex.moves.id("thunder").unwrap())
        );
        assert_eq!(
            candidate,
            SearchChoice::Move(dex.moves.id("hiddenpower").unwrap())
        );
        let reply = SearchChoice::Move(dex.moves.id("bodyslam").unwrap());
        let mut bounds = Vec::new();
        for action in [baseline, candidate] {
            let joint = if side == 0 {
                [action, reply]
            } else {
                [reply, action]
            };
            let step = enumerate_step_partial(&dex, &b, joint.map(Some), 100_000);
            assert!(step.pending.is_empty());
            let mut lower = 0.;
            let mut upper = 0.;
            for leaf in step.leaves {
                let value = match leaf.battle.outcome() {
                    Some(Outcome::P1Win) => Some(f64::from(side == 0)),
                    Some(Outcome::P2Win) => Some(f64::from(side == 1)),
                    Some(Outcome::Tie) => Some(0.5),
                    None => None,
                };
                lower += leaf.prob * value.unwrap_or(0.);
                upper += leaf.prob * value.unwrap_or(1.);
            }
            bounds.push((lower, upper));
        }
        assert!(bounds[1].0 > 1. - 1e-10);
        assert!(bounds[0].1 < 0.9);
    }
}

#[test]
fn inactive_cap_and_equal_accuracy_preserve_choices_and_rng() {
    let dex = load_dex();
    assert!(!RmConfig::default().rollout_capped_damage);
    for side in 0..2 {
        for (low_hp, moves) in [
            (false, vec!["Thunder", "Hidden Power"]),
            (true, vec!["Thunderbolt", "Hidden Power"]),
        ] {
            let mut b = fixture(&dex, side, low_hp, &moves);
            let choices = b.legal_choices(&dex, side);
            for seed in 0..128 {
                let mut old_rng = SplitMix64(seed);
                let mut new_rng = SplitMix64(seed);
                let old = playout_pick(
                    &b,
                    &dex,
                    &Playout::heavy(),
                    side,
                    &choices,
                    &mut old_rng,
                    RolloutRules::default(),
                );
                let new = playout_pick(
                    &b,
                    &dex,
                    &Playout::heavy(),
                    side,
                    &choices,
                    &mut new_rng,
                    RolloutRules {
                        capped_damage: true,
                        ..Default::default()
                    },
                );
                assert_eq!(old, new);
                assert_eq!(old_rng.next(), new_rng.next());
            }
        }
    }
}
