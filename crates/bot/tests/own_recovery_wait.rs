use conformance::load_dex;
use nc2000_bot::{
    smmcts::{dominated_actions_with, MaskRules, SelRule},
    Belief, BlindSearch, Observer, RmConfig,
};
use nc2000_engine::{
    battle::{PokemonSet, SearchChoice},
    state::{Battle, PokeId, Status},
};

#[path = "../examples/own_recovery_support/mod.rs"]
mod fixture;

#[test]
fn recovery_wait_preserves_search_and_removes_retaliation_loss() {
    let dex = load_dex();
    let (preview, b, _) = fixture::position(&dex);
    let mut obs = Observer::new(&preview, 0);
    let mut belief = Belief::pinned_from_battle(&preview, &obs);
    obs.observe(&b, &dex);
    belief.sync(&dex, &obs);
    let searches: Vec<_> = [false, true]
        .into_iter()
        .map(|enabled| {
            let mut s = BlindSearch::new(
                &b,
                &dex,
                RmConfig {
                    c: 0.4,
                    rule: SelRule::Ucb,
                    mask_rules: MaskRules {
                        own_recovery_wait: enabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                0,
                91001,
            );
            s.step(&dex, &belief, &obs, 27000);
            s
        })
        .collect();
    assert_eq!(searches[0].visits(), searches[1].visits());
    assert_eq!(searches[0].means(), searches[1].means());
    assert!(matches!(
        searches[0].best().unwrap().to_input(&dex).as_str(),
        "move return" | "move bodyslam"
    ));
    assert!(matches!(
        searches[1].best().unwrap().to_input(&dex).as_str(),
        "move toxic" | "move thunderwave"
    ));
}

#[test]
fn only_a_living_injured_active_holder_relaxes_noops_and_never_forfeit_guards() {
    let dex = load_dex();
    let set = |species: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(
            serde_json::json!({"species":species,"name":species,"level":50,"moves":moves}),
        )
        .unwrap()
    };
    for side in 0..2 {
        let own = vec![
            set("Smeargle", &["Snore", "Sleep Talk", "Explosion", "Spore"]),
            set("Magikarp", &["Splash"]),
        ];
        let foe = vec![set("Snorlax", &["Rest"]), set("Magikarp", &["Splash"])];
        let teams = if side == 0 { [own, foe] } else { [foe, own] };
        let mut base = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
        base.choose(&dex, 0, "team 1,2").unwrap();
        base.choose(&dex, 1, "team 1,2").unwrap();
        let me = base.active_id(side).unwrap();
        let own_bench = PokeId {
            side: side as u8,
            slot: base.sides[side].party[1],
        };
        let foe_bench = PokeId {
            side: (1 - side) as u8,
            slot: base.sides[1 - side].party[1],
        };
        base.poke_mut(own_bench).hp = 0;
        base.poke_mut(own_bench).fainted = true;
        base.sides[side].pokemon_left = 1;
        base.restore_status(&dex, foe_bench, Status::Slp, Some(me));
        for case in [
            "injured",
            "full",
            "empty",
            "other_item",
            "bench_only",
            "fainted",
            "zero_hp",
        ] {
            let mut b = base.clone();
            b.poke_mut(me).item = dex.items.id("leftovers");
            b.poke_mut(me).hp -= 1;
            match case {
                "full" => b.poke_mut(me).hp = b.poke(me).maxhp,
                "empty" => b.poke_mut(me).item = None,
                "other_item" => b.poke_mut(me).item = dex.items.id("berry"),
                "bench_only" => {
                    b.poke_mut(me).item = None;
                    b.poke_mut(own_bench).item = dex.items.id("leftovers");
                }
                "fainted" => b.poke_mut(me).fainted = true,
                "zero_hp" => b.poke_mut(me).hp = 0,
                _ => {}
            }
            let old = dominated_actions_with(
                &b,
                &dex,
                side,
                MaskRules {
                    own_recovery_wait: false,
                    ..Default::default()
                },
            );
            let new = dominated_actions_with(&b, &dex, side, MaskRules::default());
            if case != "injured" {
                assert_eq!(old, new, "{side} {case}");
                continue;
            }
            for key in ["snore", "sleeptalk"] {
                let a = SearchChoice::Move(dex.moves.id(key).unwrap());
                assert!(old.iter().any(|(c, _)| *c == a));
                assert!(!new.iter().any(|(c, _)| *c == a));
            }
            for key in ["explosion", "spore"] {
                let a = SearchChoice::Move(dex.moves.id(key).unwrap());
                assert!(new.iter().any(|(c, _)| *c == a), "{side} {key}");
            }
        }
    }
}
