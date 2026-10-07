use nc2000_engine::battle::enumerate::enumerate_step;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{Dex, ItemId};
use nc2000_engine::state::{Battle, EffId, EffectState, PokeId};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};

fn passive(dex: &Dex, item: ItemId) -> bool {
    dex.items.get(item).callbacks.is_empty()
        && !matches!(
            dex.items.key(item),
            "thickclub" | "lightball" | "metalpowder" | "quickclaw"
        )
}

fn normalize(dex: &Dex, battle: &Battle, inactive: &[PokeId]) -> Battle {
    let canonical = dex.items.id("pokeball").unwrap();
    assert!(passive(dex, canonical));
    let mut result = battle.clone();
    for side in &mut result.sides {
        for mon in &mut side.roster {
            if mon.item.is_some_and(|item| passive(dex, item)) {
                mon.item = Some(canonical);
                if let EffId::Item(item) = mon.item_state.id {
                    if passive(dex, item) {
                        mon.item_state.id = EffId::Item(canonical);
                    }
                }
            }
            if mon.last_item.is_some_and(|item| passive(dex, item)) {
                mon.last_item = Some(canonical);
            }
        }
    }
    for &id in inactive {
        let mon = result.poke_mut(id);
        mon.item = None;
        mon.last_item = None;
        mon.item_state = EffectState::default();
        result.refresh_poke_mask(dex, id);
    }
    result
}

fn inactive(battle: &Battle) -> Option<Vec<PokeId>> {
    if !battle.is_idle_move_request() || battle.sides.iter().any(|side| side.pokemon_left != 1) {
        return None;
    }
    let mut result = Vec::new();
    for (s, side) in battle.sides.iter().enumerate() {
        let active = side.active?;
        if side.roster[active as usize].hp <= 0 || side.roster[active as usize].fainted {
            return None;
        }
        for (slot, mon) in side.roster.iter().enumerate() {
            if slot as u8 == active {
                continue;
            }
            if side.party.contains(&(slot as u8)) && (!mon.fainted || mon.hp > 0) {
                return None;
            }
            result.push(PokeId {
                side: s as u8,
                slot: slot as u8,
            });
        }
    }
    Some(result)
}

fn kernel(
    dex: &Dex,
    battle: &Battle,
    choices: [Option<SearchChoice>; 2],
    ignored: &[PokeId],
) -> (BTreeMap<u128, f64>, usize) {
    let step = enumerate_step(dex, battle, choices, 200_000).expect("item-model audit cap");
    let mut result = BTreeMap::<u128, f64>::new();
    for leaf in step.leaves {
        let key = normalize(dex, &leaf.battle, ignored).state_key128();
        *result.entry(key).or_default() += leaf.prob;
    }
    assert!((result.values().sum::<f64>() - 1.).abs() < 1e-12);
    (result, step.runs)
}

pub fn group(dex: &Dex, worlds: &[Battle], audit: bool) -> (Vec<Vec<usize>>, Value) {
    let mut by_key = HashMap::<u128, usize>::new();
    let mut groups = Vec::<Vec<usize>>::new();
    let mut roots = Vec::new();
    let mut ignored = Vec::new();
    for (index, battle) in worlds.iter().enumerate() {
        let ids = inactive(battle);
        let key = ids.as_ref().map_or_else(
            || battle.state_key128(),
            |ids| normalize(dex, battle, ids).state_key128(),
        );
        let group = *by_key.entry(key).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[group].push(index);
        roots.push(json!({"raw":battle.state_key128().to_string(),"key":key.to_string(),"eligible":ids.is_some()}));
        ignored.push(ids);
    }
    let mut audit_runs = 0;
    let mut audit_cells = 0;
    let mut maximum_error = 0f64;
    if audit {
        for members in &groups {
            if members.len() == 1 {
                continue;
            }
            let first = members[0];
            let ids = ignored[first].as_ref().unwrap();
            let mut battle = worlds[first].clone();
            let legal = [0, 1].map(|side| battle.legal_choices(dex, side));
            for member in &members[1..] {
                let mut other = worlds[*member].clone();
                for side in 0..2 {
                    assert_eq!(other.legal_choices(dex, side), legal[side]);
                }
                assert_eq!(ignored[*member].as_ref().unwrap(), ids);
            }
            for &a in &legal[0] {
                for &b in &legal[1] {
                    let choices = [Some(a), Some(b)];
                    let (expected, runs) = kernel(dex, &worlds[first], choices, ids);
                    audit_runs += runs;
                    for &member in &members[1..] {
                        let (actual, runs) = kernel(dex, &worlds[member], choices, ids);
                        audit_runs += runs;
                        audit_cells += 1;
                        assert_eq!(actual.len(), expected.len(), "item-model kernel support");
                        for (key, mass) in &expected {
                            let error =
                                (actual.get(key).expect("item-model state mismatch") - mass).abs();
                            maximum_error = maximum_error.max(error);
                            assert!(error < 1e-12, "item-model kernel mass mismatch");
                        }
                    }
                }
            }
        }
    }
    (
        groups,
        json!({"roots":roots,"audit_runs":audit_runs,"audit_cells":audit_cells,"maximum_error":maximum_error}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;

    fn fixture(dex: &Dex, species: &str, action: &str) -> Battle {
        let mon = |species: &str, item: &str, moves: Vec<&str>| {
            serde_json::from_value::<PokemonSet>(
                json!({"species":species,"name":species,"level":50,"item":item,"moves":moves}),
            )
            .unwrap()
        };
        let own = vec![
            mon(species, "", vec![action]),
            mon("Snorlax", "", vec!["Tackle"]),
            mon("Tauros", "", vec!["Tackle"]),
        ];
        let foe = vec![
            mon("Magikarp", "Dragon Scale", vec!["Splash"]),
            mon("Miltank", "", vec!["Tackle"]),
            mon("Starmie", "", vec!["Tackle"]),
        ];
        let mut battle = Battle::from_fixture(dex, "1,2,3,4", &own, &foe).unwrap();
        let choices = [0, 1].map(|s| Some(battle.legal_choices(dex, s)[0]));
        battle.apply_choices(dex, choices).unwrap();
        for side in &mut battle.sides {
            for (slot, mon) in side.roster.iter_mut().enumerate() {
                if Some(slot as u8) != side.active {
                    mon.hp = 0;
                    mon.fainted = true;
                }
            }
            side.pokemon_left = 1;
        }
        battle
    }

    #[test]
    fn passive_identity_survives_theft_and_transformation() {
        let dex = conformance::load_dex();
        for (species, action) in [("Jynx", "thief"), ("Ditto", "transform")] {
            let battle = fixture(&dex, species, action);
            let ids = inactive(&battle).unwrap();
            let canonical = normalize(&dex, &battle, &ids);
            assert_ne!(battle.state_key128(), canonical.state_key128());
            let choices = [
                Some(SearchChoice::Move(dex.moves.id(action).unwrap())),
                Some(SearchChoice::Move(dex.moves.id("splash").unwrap())),
            ];
            let (a, _) = kernel(&dex, &battle, choices, &ids);
            let (b, _) = kernel(&dex, &canonical, choices, &ids);
            assert_eq!(a, b);
            let first = enumerate_step(&dex, &battle, choices, 200_000).unwrap();
            let mut triggered = false;
            let mut continuations = 0;
            for mut leaf in first.leaves {
                let active = leaf.battle.sides[0].active.unwrap() as usize;
                let mon = &leaf.battle.sides[0].roster[active];
                triggered |= if action == "thief" {
                    mon.item.is_some()
                } else {
                    mon.transformed
                };
                if leaf.battle.ended {
                    continue;
                }
                let second = [0, 1].map(|s| Some(leaf.battle.legal_choices(&dex, s)[0]));
                let canonical = normalize(&dex, &leaf.battle, &ids);
                let (a, _) = kernel(&dex, &leaf.battle, second, &ids);
                let (b, _) = kernel(&dex, &canonical, second, &ids);
                assert_eq!(a, b);
                continuations += 1;
            }
            assert!(triggered && continuations > 0, "{action}");
        }
    }

    #[test]
    fn live_bench_and_pending_turn_reject_inactive_item_removal() {
        let dex = conformance::load_dex();
        let mut battle = fixture(&dex, "Jynx", "Thief");
        assert!(inactive(&battle).is_some());
        battle.mid_turn = true;
        assert!(inactive(&battle).is_none());
        battle.mid_turn = false;
        let active = battle.sides[0].active.unwrap() as usize;
        let bench = battle.sides[0]
            .party
            .iter()
            .copied()
            .find(|&slot| slot as usize != active)
            .unwrap() as usize;
        battle.sides[0].roster[bench].hp = 1;
        battle.sides[0].roster[bench].fainted = false;
        assert!(inactive(&battle).is_none());
    }

    #[test]
    fn explicit_item_effects_remain_distinct() {
        let dex = conformance::load_dex();
        for name in [
            "quickclaw",
            "thickclub",
            "lightball",
            "metalpowder",
            "brightpowder",
            "focusband",
            "berry",
            "mail",
        ] {
            assert!(!passive(&dex, dex.items.id(name).unwrap()), "{name}");
        }
        for name in [
            "pokeball",
            "greatball",
            "dragonscale",
            "upgrade",
            "waterstone",
        ] {
            assert!(passive(&dex, dex.items.id(name).unwrap()), "{name}");
        }
    }
}
