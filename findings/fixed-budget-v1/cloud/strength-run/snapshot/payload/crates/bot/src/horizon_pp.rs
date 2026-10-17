use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HorizonPpKey {
    turns_left: u16,
    state: u128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PpGuardError {
    NoTurns,
    OutsideHealClass,
    UnsupportedMove(String),
    RequiresLeftovers,
    UnsupportedVolatile(String),
    InconsistentSlots,
    DamageObserver,
}

/// Equivalent only through `turns_left` complete turns, with all nonterminal
/// continuations at that boundary assigned the unknown interval [0, 1].
#[derive(Debug)]
pub struct HorizonPpState {
    battle: Battle,
    key: HorizonPpKey,
}

impl HorizonPpState {
    pub fn new(battle: &Battle, dex: &Dex, turns_left: u16) -> Result<Self, PpGuardError> {
        if turns_left == 0 {
            return Err(PpGuardError::NoTurns);
        }
        if crate::stall::classify_two_sided_heal(battle, dex).is_err() {
            return Err(PpGuardError::OutsideHealClass);
        }
        for side in 0..2 {
            let p = battle.poke(battle.active_id(side).unwrap());
            if p.item.map(|item| dex.items.key(item)) != Some("leftovers") {
                return Err(PpGuardError::RequiresLeftovers);
            }
            for (condition, _) in &p.volatiles {
                let key = dex.conds.key(*condition);
                if !matches!(
                    key,
                    "confusion" | "substitute" | "curse" | "parspeeddrop" | "brnattackdrop"
                ) {
                    return Err(PpGuardError::UnsupportedVolatile(key.into()));
                }
            }
            for (slot, base) in p.move_slots.iter().zip(p.base_move_slots.iter()) {
                if slot.pp < 0 || slot.pp != base.pp || !slot.shared || slot.id != base.id {
                    return Err(PpGuardError::InconsistentSlots);
                }
                let key = dex.moves.key(slot.id);
                if !matches!(
                    key,
                    "bodyslam"
                        | "earthquake"
                        | "curse"
                        | "selfdestruct"
                        | "thunder"
                        | "swagger"
                        | "substitute"
                        | "recover"
                        | "seismictoss"
                        | "splash"
                        | "tackle"
                ) && !key.starts_with("hiddenpower")
                {
                    return Err(PpGuardError::UnsupportedMove(key.into()));
                }
            }
        }
        let mut capped = battle.clone();
        for side in 0..2 {
            let id = capped.active_id(side).unwrap();
            let p = capped.poke_mut(id);
            for slot in p.move_slots.iter_mut() {
                slot.pp = slot.pp.min(i32::from(turns_left));
            }
            for slot in p.base_move_slots.iter_mut() {
                slot.pp = slot.pp.min(i32::from(turns_left));
            }
        }
        if capped.damage_bookkeeping_observable(dex) {
            return Err(PpGuardError::DamageObserver);
        }
        let state = capped.state_key128_without_damage_bookkeeping();
        Ok(Self {
            battle: capped,
            key: HorizonPpKey { turns_left, state },
        })
    }

    pub fn key(&self) -> HorizonPpKey {
        self.key
    }

    pub fn battle(&self) -> &Battle {
        &self.battle
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::{enumerate::enumerate_step, PokemonSet};
    use std::collections::HashMap;

    fn root(dex: &Dex, moves: &[&str]) -> Battle {
        let p = PokemonSet {
            name: "Pikachu".into(),
            species: "Pikachu".into(),
            item: "Leftovers".into(),
            ability: String::new(),
            moves: moves.iter().map(|m| (*m).into()).collect(),
            level: 50,
            evs: None,
            ivs: None,
            happiness: None,
            gender: None,
        };
        let mut foe = p.clone();
        foe.name = "Raichu".into();
        foe.species = "Raichu".into();
        let mut b = Battle::from_fixture(dex, "1,2,3,4", &[p], &[foe]).unwrap();
        let choices = [0, 1].map(|side| b.legal_choices(dex, side)[0]);
        b.apply_choices(dex, choices.map(Some)).unwrap();
        b
    }

    fn set_pp(b: &mut Battle, pp: i32) {
        for side in 0..2 {
            let id = b.active_id(side).unwrap();
            let p = b.poke_mut(id);
            for slot in p.move_slots.iter_mut() {
                slot.pp = pp;
            }
            for slot in p.base_move_slots.iter_mut() {
                slot.pp = pp;
            }
        }
    }

    #[test]
    fn horizon_keys_merge_only_surplus_pp_and_keep_horizons_separate() {
        let dex = conformance::load_dex();
        let b = root(&dex, &["Recover", "Seismic Toss"]);
        let full = HorizonPpState::new(&b, &dex, 3).unwrap();
        let mut low = b.clone();
        set_pp(&mut low, 3);
        assert_ne!(b.state_key128(), low.state_key128());
        assert_eq!(
            full.key(),
            HorizonPpState::new(&low, &dex, 3).unwrap().key()
        );
        assert_ne!(
            full.key(),
            HorizonPpState::new(&low, &dex, 2).unwrap().key()
        );
        for pp in [0, 1, 2] {
            set_pp(&mut low, pp);
            assert_ne!(
                full.key(),
                HorizonPpState::new(&low, &dex, 3).unwrap().key()
            );
        }
    }

    #[test]
    fn unsafe_pp_paths_and_boundary_are_rejected() {
        let dex = conformance::load_dex();
        for moves in [
            vec!["Spite"],
            vec!["Metronome"],
            vec!["Transform"],
            vec!["Sleep Talk"],
            vec!["Encore"],
        ] {
            assert!(HorizonPpState::new(&root(&dex, &moves), &dex, 3).is_err());
        }
        let mut b = root(&dex, &["Recover"]);
        assert_eq!(
            HorizonPpState::new(&b, &dex, 0).unwrap_err(),
            PpGuardError::NoTurns
        );
        let id = b.active_id(0).unwrap();
        b.poke_mut(id).item = dex.items.id("mysteryberry");
        assert!(HorizonPpState::new(&b, &dex, 3).is_err());
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    enum Leaf {
        Win,
        Loss,
        Tie,
        Frontier,
        State(HorizonPpKey),
    }

    fn leaf(b: &Battle, dex: &Dex, turns_left: u16) -> Leaf {
        match b.outcome() {
            Some(nc2000_engine::battle::Outcome::P1Win) => Leaf::Win,
            Some(nc2000_engine::battle::Outcome::P2Win) => Leaf::Loss,
            Some(nc2000_engine::battle::Outcome::Tie) => Leaf::Tie,
            None if turns_left == 0 => Leaf::Frontier,
            None => Leaf::State(HorizonPpState::new(b, dex, turns_left).unwrap().key()),
        }
    }

    #[test]
    fn capped_pp_preserves_legal_choices_and_complete_successor_distributions() {
        let dex = conformance::load_dex();
        for moves in [
            vec!["Recover", "Seismic Toss"],
            vec!["Body Slam", "Substitute"],
        ] {
            let mut b = root(&dex, &moves);
            for side in 0..2 {
                let id = b.active_id(side).unwrap();
                b.poke_mut(id).hp = 60;
            }
            for turns in [1, 2, 3] {
                let mut representative = HorizonPpState::new(&b, &dex, turns)
                    .unwrap()
                    .battle()
                    .clone();
                let actions = [0, 1].map(|side| b.legal_choices(&dex, side));
                for side in 0..2 {
                    assert_eq!(actions[side], representative.legal_choices(&dex, side));
                }
                for &a in &actions[0] {
                    for &c in &actions[1] {
                        let mut distributions = Vec::new();
                        for state in [&b, &representative] {
                            let step =
                                enumerate_step(&dex, state, [Some(a), Some(c)], 100_000).unwrap();
                            let mut distribution = HashMap::<_, f64>::new();
                            for next in step.leaves {
                                *distribution
                                    .entry(leaf(&next.battle, &dex, turns - 1))
                                    .or_default() += next.prob;
                            }
                            assert!((distribution.values().sum::<f64>() - 1.0).abs() < 1e-10);
                            distributions.push(distribution);
                        }
                        assert_eq!(distributions[0].len(), distributions[1].len());
                        for (key, mass) in &distributions[0] {
                            assert!((mass - distributions[1][key]).abs() < 1e-10);
                        }
                    }
                }
            }
        }
    }
}
