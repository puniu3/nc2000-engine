use conformance::load_dex;
use nc2000_bot::smmcts::{dominated_actions_with, MaskRules, SelRule};
use nc2000_bot::{Agent, Belief, BlindSearch, MaxDamageAgent, Observer, RmConfig};
use nc2000_engine::{
    battle::{enumerate::enumerate_step, EffectHandle, Outcome, PokemonSet, SearchChoice},
    state::{Battle, PokeId, Status},
};
use serde_json::json;

#[path = "support/destiny_bond_wait.rs"]
mod fixture;

fn config(enabled: bool) -> RmConfig {
    RmConfig {
        rule: SelRule::Ucb,
        c: 0.4,
        mask_rules: MaskRules {
            destiny_bond_wait: enabled,
            residual_damage_wait: false,
            ..MaskRules::default()
        },
        ..RmConfig::default()
    }
}

#[test]
fn final_mask_preserves_the_winning_wait_without_changing_search() {
    let dex = load_dex();
    for (plain, blind) in [(false, false), (true, false), (false, true), (true, true)] {
        let (preview, b) = fixture::position(&dex, plain, true);
        let mut obs = Observer::new(&preview, 0);
        let mut belief = if blind {
            Belief::new(
                &dex,
                &nc2000_bot::preview::load_meta_pool(
                    &conformance::fixture::repo_root().join("data/belief-pool-v3/belief-pool.json"),
                ),
                &obs,
            )
        } else {
            Belief::pinned_from_battle(&preview, &obs)
        };
        obs.observe(&b, &dex);
        belief.sync(&dex, &obs);
        let searches: Vec<_> = [false, true]
            .into_iter()
            .map(|enabled| {
                let mut s = BlindSearch::new(&b, &dex, config(enabled), 0, 61001);
                s.step(&dex, &belief, &obs, 3000);
                s
            })
            .collect();
        assert_eq!(searches[0].visits(), searches[1].visits());
        assert_eq!(searches[0].means(), searches[1].means());
        let before = searches[0].best().unwrap();
        let after = searches[1].best().unwrap();
        assert!(matches!(
            before.to_input(&dex).as_str(),
            "move shadowball" | "move psychic"
        ));
        assert!(
            after.to_input(&dex) == "move return"
                || (!plain && after.to_input(&dex) == "move curse")
        );
        let reply = SearchChoice::Move(dex.moves.id("haze").unwrap());
        let attack = enumerate_step(&dex, &b, [Some(before), Some(reply)], 4096).unwrap();
        assert!(attack
            .leaves
            .iter()
            .all(|l| l.battle.outcome() == Some(Outcome::P2Win)));
        let wait = enumerate_step(&dex, &b, [Some(after), Some(reply)], 4096).unwrap();
        assert!(wait
            .leaves
            .iter()
            .all(|l| l.battle.outcome() != Some(Outcome::P2Win)));
        for seed in 62001..62009 {
            for (first, expected) in [(before, Outcome::P2Win), (after, Outcome::P1Win)] {
                let mut game = b.clone();
                game.reseed(seed);
                game.apply_choices(&dex, [Some(first), Some(reply)])
                    .unwrap();
                let mut greedy = MaxDamageAgent::conformant();
                for _ in 0..100 {
                    if game.outcome().is_some() {
                        break;
                    }
                    let mut picks = [None, None];
                    for (side, pick) in picks.iter_mut().enumerate() {
                        let choices = game.legal_choices(&dex, side);
                        if !choices.is_empty() {
                            *pick = Some(greedy.choose(&game, &dex, side, &choices));
                        }
                    }
                    game.apply_choices(&dex, picks).unwrap();
                }
                assert_eq!(game.outcome(), Some(expected), "plain={plain}, seed={seed}");
            }
        }
    }
}

#[test]
fn only_the_opposing_active_bond_relaxes_noop_exclusions() {
    let dex = load_dex();
    let set = |species: &str, moves: &[&str]| -> PokemonSet {
        serde_json::from_value(json!({"species":species,"name":species,"level":50,
            "moves":moves,"item":"","ability":"No Ability"}))
        .unwrap()
    };
    for side in 0..2 {
        let own = vec![
            set("Smeargle", &["Snore", "Explosion", "Spore"]),
            set("Magikarp", &["Splash"]),
        ];
        let foe = vec![set("Snorlax", &["Rest"]), set("Magikarp", &["Splash"])];
        let teams = if side == 0 { [own, foe] } else { [foe, own] };
        let mut base = Battle::from_fixture(&dex, "1,2,3,4", &teams[0], &teams[1]).unwrap();
        base.choose(&dex, 0, "team 1,2").unwrap();
        base.choose(&dex, 1, "team 1,2").unwrap();
        let me = base.active_id(side).unwrap();
        let them = base.active_id(1 - side).unwrap();
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
        for target in [None, Some(me), Some(them), Some(foe_bench)] {
            let mut b = base.clone();
            if let Some(target) = target {
                b.add_volatile(&dex, target, "destinybond", None, EffectHandle::None);
            }
            for enabled in [false, true] {
                let excluded = dominated_actions_with(&b, &dex, side, config(enabled).mask_rules);
                let masked = |name| {
                    excluded
                        .iter()
                        .any(|(a, _)| *a == SearchChoice::Move(dex.moves.id(name).unwrap()))
                };
                assert_eq!(masked("snore"), !(enabled && target == Some(them)));
                assert!(masked("explosion"));
                assert!(masked("spore"));
            }
        }
    }
}

#[test]
fn fixture_sets_are_legal_on_complete_team_sheets() {
    use nc2000_engine::validate::{validate_team, Learnsets};
    let dex = load_dex();
    let learnsets = Learnsets::from_json(
        &std::fs::read_to_string(
            conformance::fixture::repo_root().join("data/learnsets-gen2.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for plain in [false, true] {
        let (preview, _) = fixture::position(&dex, plain, true);
        for side in &preview.sides {
            let mut sheet: Vec<_> = side.roster.iter().map(|p| {
                let keys = ["hp", "atk", "def", "spa", "spd", "spe"];
                let ivs: std::collections::BTreeMap<_, _> = keys.into_iter().zip(p.set_ivs).collect();
                let evs: std::collections::BTreeMap<_, _> = keys.into_iter().zip(p.set_evs).collect();
                json!({"name":dex.species.key(p.species),"species":dex.species.key(p.species),
                    "level":p.level,"item":p.item.map(|i|dex.items.key(i)).unwrap_or(""),
                    "ability":"No Ability","happiness":p.happiness,"ivs":ivs,"evs":evs,
                    "moves":p.base_move_slots.iter().map(|m|dex.moves.key(m.id)).collect::<Vec<_>>()})
            }).collect();
            for (species, attack) in [
                ("Bulbasaur", "Tackle"),
                ("Charmander", "Scratch"),
                ("Squirtle", "Tackle"),
                ("Pikachu", "Thunder Shock"),
            ] {
                sheet.push(json!({"species":species,"name":species,"level":50,
                    "moves":[attack],"item":"","ability":"No Ability"}));
            }
            let result = validate_team(&dex, &learnsets, &serde_json::to_string(&sheet).unwrap());
            assert_eq!(result["ok"], true, "{result}");
        }
    }
}
