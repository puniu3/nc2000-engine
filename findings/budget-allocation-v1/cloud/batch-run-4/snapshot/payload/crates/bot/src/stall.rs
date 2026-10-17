//! Conservative classification and monotone ranks for last-mon heal endings.
//!
//! This module only recognizes quiescent, nonterminal, last-mon 1v1 move
//! requests.  The returned class is a proof obligation for a later solver:
//! callers must run the certificate's `check_edge` on every nonterminal child
//! before scheduling or freeing resource generations.

use nc2000_engine::dex::{Dex, MoveId};
use nc2000_engine::state::{Battle, PokeId, RequestState};

const TURN_LIMIT_RANK_BASE: i64 = 1001;

/// A narrowly certified last-mon ending in which only `healer` can regain HP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneSidedHeal {
    pub healer: usize,
    pub non_healer: usize,
    active: [PokeId; 2],
    move_ids: [Vec<MoveId>; 2],
    max_hp: [i32; 2],
}

/// A fixed-roster last-mon ending in which both sides can currently heal.
///
/// Unlike [`OneSidedHeal`], this certificate makes no HP-monotonicity claim:
/// its well-founded rank is the lexicographic pair (total PP, turns left).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoSidedHeal {
    active: [PokeId; 2],
    move_ids: [Vec<MoveId>; 2],
    max_hp: [i32; 2],
    healing_slots: [Vec<usize>; 2],
}

/// Components of the well-founded scheduling rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct MonotoneRank {
    pub value: i64,
    pub pp_total: i64,
    pub non_healer_hp: i32,
    pub turns_remaining: i64,
}

/// Resource components for a two-sided-heal ending.
///
/// `healing_pp` is exposed as a scheduling signal, but the proof order is
/// [`ResourceRank::lexicographic_key`]: total PP, then turns remaining.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceRank {
    pub healing_pp: i64,
    pub pp_total: i64,
    pub turns_remaining: i64,
}

impl ResourceRank {
    pub fn lexicographic_key(self) -> (i64, i64) {
        (self.pp_total, self.turns_remaining)
    }
}

/// Why a state was not admitted to the v1 one-sided-heal domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClassifyError {
    NotQuiescent,
    NotLastMon { side: usize },
    InvalidActive { side: usize },
    TurnPastLimit { turn: u16 },
    ChangedMoveSlots { side: usize },
    UnsafeMove { side: usize, move_key: String },
    MysteryBerry { side: usize },
    NotExactlyOneHealer,
    NotBothHealers,
    NonHealerHealingItem { side: usize, item_key: String },
}

/// A violated invariant on a parent -> nonterminal-child decision edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EdgeError {
    NotQuiescent,
    TurnPastLimit {
        turn: u16,
    },
    ActiveChanged {
        side: usize,
    },
    MoveSlotsChanged {
        side: usize,
    },
    MaxHpChanged {
        side: usize,
    },
    PpIncreased {
        side: usize,
        slot: usize,
        parent: i32,
        child: i32,
    },
    NonHealerHpIncreased {
        parent: i32,
        child: i32,
    },
    TurnDidNotIncrease {
        parent: u16,
        child: u16,
    },
    RankDidNotDecrease {
        parent: i64,
        child: i64,
    },
    ResourceRankDidNotDecrease {
        parent: (i64, i64),
        child: (i64, i64),
    },
}

struct RootAudit {
    active: [PokeId; 2],
    move_ids: [Vec<MoveId>; 2],
    max_hp: [i32; 2],
    repeatable_heal: [bool; 2],
    healing_slots: [Vec<usize>; 2],
}

/// Classify a state for the conservative v1 one-sided-heal scheduler.
pub fn classify_one_sided_heal(b: &Battle, dex: &Dex) -> Result<OneSidedHeal, ClassifyError> {
    let audit = audit_root(b, dex)?;
    let healer = match audit.repeatable_heal {
        [true, false] => 0,
        [false, true] => 1,
        _ => return Err(ClassifyError::NotExactlyOneHealer),
    };
    let non_healer = 1 - healer;
    if let Some(item) = b.poke(audit.active[non_healer]).item {
        let key = dex.items.key(item);
        if is_hp_healing_item(key) {
            return Err(ClassifyError::NonHealerHealingItem {
                side: non_healer,
                item_key: key.to_string(),
            });
        }
    }

    Ok(OneSidedHeal {
        healer,
        non_healer,
        active: audit.active,
        move_ids: audit.move_ids,
        max_hp: audit.max_hp,
    })
}

/// Classify a fixed-roster last-mon ending where both sides can heal.
pub fn classify_two_sided_heal(b: &Battle, dex: &Dex) -> Result<TwoSidedHeal, ClassifyError> {
    let audit = audit_root(b, dex)?;
    if audit.repeatable_heal != [true, true] {
        return Err(ClassifyError::NotBothHealers);
    }
    Ok(TwoSidedHeal {
        active: audit.active,
        move_ids: audit.move_ids,
        max_hp: audit.max_hp,
        healing_slots: audit.healing_slots,
    })
}

fn audit_root(b: &Battle, dex: &Dex) -> Result<RootAudit, ClassifyError> {
    if !is_quiescent_move_request(b) {
        return Err(ClassifyError::NotQuiescent);
    }
    if b.turn > 1000 {
        return Err(ClassifyError::TurnPastLimit { turn: b.turn });
    }

    let mut active = [PokeId { side: 0, slot: 0 }; 2];
    let mut move_ids: [Vec<MoveId>; 2] = std::array::from_fn(|_| Vec::new());
    let mut max_hp = [0; 2];
    let mut repeatable_heal = [false; 2];
    let mut healing_slots: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());
    let mut seeded_healers = Vec::new();

    for side in 0..2 {
        if b.sides[side].pokemon_left != 1 {
            return Err(ClassifyError::NotLastMon { side });
        }
        let Some(id) = b.active_id(side) else {
            return Err(ClassifyError::InvalidActive { side });
        };
        let p = b.poke(id);
        if p.fainted || p.hp <= 0 || !p.is_active {
            return Err(ClassifyError::InvalidActive { side });
        }

        let current: Vec<MoveId> = p.move_slots.iter().map(|m| m.id).collect();
        let base: Vec<MoveId> = p.base_move_slots.iter().map(|m| m.id).collect();
        if p.transformed || current != base || p.move_slots.iter().any(|slot| !slot.shared) {
            // Reject roots already transformed or carrying a Mimic-created
            // slot; the edge checker can then require immutable slot ids.
            return Err(ClassifyError::ChangedMoveSlots { side });
        }
        for (slot_index, slot) in p.move_slots.iter().enumerate() {
            let key = dex.moves.key(slot.id);
            if unsafe_move(key) {
                return Err(ClassifyError::UnsafeMove {
                    side,
                    move_key: key.to_string(),
                });
            }
            if slot.pp > 0 && move_can_heal(dex, slot.id) {
                repeatable_heal[side] = true;
                healing_slots[side].push(slot_index);
            }
        }

        if let Some(item) = p.item {
            let key = dex.items.key(item);
            if key == "mysteryberry" {
                return Err(ClassifyError::MysteryBerry { side });
            }
            if key == "leftovers" {
                repeatable_heal[side] = true;
            }
        }
        if let Some(leech_seed) = dex.conds_id("leechseed") {
            if let Some(seed) = p.volatile(leech_seed) {
                // Its source, not its holder, receives HP. In a last-mon
                // ending a certified active source cannot switch away, so it
                // is another statically known healing path.
                // Mechanics resolve the leecher solely through source_slot;
                // an absent/stale slot means this seed can no longer heal.
                if let Some(source) = seed
                    .source_slot
                    .and_then(|slot| b.poke_at_slot_pos(slot))
                    .filter(|&source| !b.poke(source).fainted && b.poke(source).hp > 0)
                {
                    seeded_healers.push(source.side as usize);
                }
            }
        }

        active[side] = id;
        move_ids[side] = current;
        max_hp[side] = p.maxhp;
    }
    for side in seeded_healers {
        repeatable_heal[side] = true;
    }

    Ok(RootAudit {
        active,
        move_ids,
        max_hp,
        repeatable_heal,
        healing_slots,
    })
}

impl OneSidedHeal {
    /// Compute the scheduling rank after checking root identity invariants.
    pub fn rank(&self, b: &Battle) -> Result<MonotoneRank, EdgeError> {
        self.check_identity(b)?;
        if b.turn > 1000 {
            return Err(EdgeError::TurnPastLimit { turn: b.turn });
        }
        let pp_total = self
            .active
            .iter()
            .map(|&id| {
                b.poke(id)
                    .move_slots
                    .iter()
                    .map(|m| i64::from(m.pp))
                    .sum::<i64>()
            })
            .sum::<i64>();
        let non_healer_hp = b.poke(self.active[self.non_healer]).hp;
        let turns_remaining = TURN_LIMIT_RANK_BASE - i64::from(b.turn);
        Ok(MonotoneRank {
            value: pp_total + i64::from(non_healer_hp) + turns_remaining,
            pp_total,
            non_healer_hp,
            turns_remaining,
        })
    }

    /// Check all invariants needed to schedule `child` below `parent`.
    ///
    /// Terminal children do not need a rank and should bypass this API.
    pub fn check_edge(&self, parent: &Battle, child: &Battle) -> Result<(), EdgeError> {
        self.check_identity(parent)?;
        self.check_identity(child)?;
        check_pp_nonincrease(&self.active, parent, child)?;

        let parent_hp = parent.poke(self.active[self.non_healer]).hp;
        let child_hp = child.poke(self.active[self.non_healer]).hp;
        if child_hp > parent_hp {
            return Err(EdgeError::NonHealerHpIncreased {
                parent: parent_hp,
                child: child_hp,
            });
        }
        if child.turn <= parent.turn {
            return Err(EdgeError::TurnDidNotIncrease {
                parent: parent.turn,
                child: child.turn,
            });
        }

        let parent_rank = self.rank(parent)?;
        let child_rank = self.rank(child)?;
        if child_rank.value >= parent_rank.value {
            return Err(EdgeError::RankDidNotDecrease {
                parent: parent_rank.value,
                child: child_rank.value,
            });
        }
        Ok(())
    }

    fn check_identity(&self, b: &Battle) -> Result<(), EdgeError> {
        check_identity(&self.active, &self.move_ids, &self.max_hp, b)
    }
}

impl TwoSidedHeal {
    /// Compute the resource rank after checking fixed-roster invariants.
    pub fn rank(&self, b: &Battle) -> Result<ResourceRank, EdgeError> {
        self.check_identity(b)?;
        let pp_total = self
            .active
            .iter()
            .map(|&id| {
                b.poke(id)
                    .move_slots
                    .iter()
                    .map(|slot| i64::from(slot.pp))
                    .sum::<i64>()
            })
            .sum();
        let healing_pp = (0..2)
            .flat_map(|side| {
                self.healing_slots[side]
                    .iter()
                    .map(move |&slot| i64::from(b.poke(self.active[side]).move_slots[slot].pp))
            })
            .sum();
        Ok(ResourceRank {
            healing_pp,
            pp_total,
            turns_remaining: TURN_LIMIT_RANK_BASE - i64::from(b.turn),
        })
    }

    /// Verify the fixed-resource proof obligations on a nonterminal edge.
    pub fn check_edge(&self, parent: &Battle, child: &Battle) -> Result<(), EdgeError> {
        self.check_identity(parent)?;
        self.check_identity(child)?;
        check_pp_nonincrease(&self.active, parent, child)?;
        if child.turn <= parent.turn {
            return Err(EdgeError::TurnDidNotIncrease {
                parent: parent.turn,
                child: child.turn,
            });
        }

        let parent_rank = self.rank(parent)?.lexicographic_key();
        let child_rank = self.rank(child)?.lexicographic_key();
        if child_rank >= parent_rank {
            return Err(EdgeError::ResourceRankDidNotDecrease {
                parent: parent_rank,
                child: child_rank,
            });
        }
        Ok(())
    }

    fn check_identity(&self, b: &Battle) -> Result<(), EdgeError> {
        check_identity(&self.active, &self.move_ids, &self.max_hp, b)
    }
}

fn check_identity(
    active: &[PokeId; 2],
    move_ids: &[Vec<MoveId>; 2],
    max_hp: &[i32; 2],
    b: &Battle,
) -> Result<(), EdgeError> {
    if !is_quiescent_move_request(b) {
        return Err(EdgeError::NotQuiescent);
    }
    if b.turn > 1000 {
        return Err(EdgeError::TurnPastLimit { turn: b.turn });
    }
    for side in 0..2 {
        if b.active_id(side) != Some(active[side]) {
            return Err(EdgeError::ActiveChanged { side });
        }
        let p = b.poke(active[side]);
        if p.maxhp != max_hp[side] {
            return Err(EdgeError::MaxHpChanged { side });
        }
        let ids: Vec<MoveId> = p.move_slots.iter().map(|m| m.id).collect();
        if ids != move_ids[side] {
            return Err(EdgeError::MoveSlotsChanged { side });
        }
    }
    Ok(())
}

fn check_pp_nonincrease(
    active: &[PokeId; 2],
    parent: &Battle,
    child: &Battle,
) -> Result<(), EdgeError> {
    for side in 0..2 {
        let p = parent.poke(active[side]);
        let c = child.poke(active[side]);
        for (slot, (pm, cm)) in p.move_slots.iter().zip(c.move_slots.iter()).enumerate() {
            if cm.pp > pm.pp {
                return Err(EdgeError::PpIncreased {
                    side,
                    slot,
                    parent: pm.pp,
                    child: cm.pp,
                });
            }
        }
    }
    Ok(())
}

fn is_quiescent_move_request(b: &Battle) -> bool {
    !b.ended
        && b.winner.is_none()
        && b.request_state == RequestState::Move
        && b.needs_choice() == [true, true]
        && !b.mid_turn
        && b.queue.is_empty()
        && b.faint_queue.is_empty()
        && b.event_stack.is_empty()
        && b.effect_stack.is_empty()
        && b.active_move.is_none()
        && b.active_pokemon.is_none()
        && b.active_target.is_none()
        && b.pending_boosts.is_none()
        && b.sides.iter().all(|s| {
            s.choice.actions.is_empty()
                && s.choice.switch_ins.is_empty()
                && s.choice.forced_switches_left == 0
                && s.choice.forced_passes_left == 0
        })
}

fn unsafe_move(key: &str) -> bool {
    matches!(
        key,
        // HP can increase on the eventual non-healer.
        "painsplit" | "present" |
        // Move slots or their PP can be rewritten.
        "mimic" | "sketch" | "transform" | "spite" |
        // Arbitrary/copy calls invalidate the static moveset audit. Sleep
        // Talk is deliberately allowed: it calls only a current slot and
        // does not deduct that called slot's PP.
        "metronome" | "mirrormove" |
        // Can move a healing item across the monotone boundary.
        "thief"
    )
}

fn move_can_heal(dex: &Dex, id: MoveId) -> bool {
    let m = dex.move_static(id);
    m.heal.is_some()
        || m.drain.is_some()
        || matches!(
            dex.moves.key(id),
            "rest" | "moonlight" | "morningsun" | "synthesis" | "leechseed"
        )
}

fn is_hp_healing_item(key: &str) -> bool {
    matches!(key, "leftovers" | "berryjuice" | "berry" | "goldberry")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;

    fn set(name: &str, species: &str, item: &str, moves: [&str; 4]) -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "name": name,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "nature": "Serious",
            "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
            "gender": "M",
            "level": 50
        }))
        .unwrap()
    }

    fn battle(moves0: [&str; 4], item0: &str, moves1: [&str; 4], item1: &str) -> (Dex, Battle) {
        let dex = conformance::load_dex();
        let filler = || {
            set(
                "Pikachu",
                "Pikachu",
                "",
                ["Thunderbolt", "Toxic", "Protect", "Roar"],
            )
        };
        let t0 = vec![set("Snorlax", "Snorlax", item0, moves0), filler(), filler()];
        let t1 = vec![
            set("Skarmory", "Skarmory", item1, moves1),
            filler(),
            filler(),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t0, &t1).unwrap();
        b.set_log_enabled(false);
        b.choose(&dex, 0, "team 1, 2, 3").unwrap();
        b.choose(&dex, 1, "team 1, 2, 3").unwrap();
        for side in 0..2 {
            let active = b.active_id(side).unwrap();
            b.sides[side].pokemon_left = 1;
            for slot in 0..b.sides[side].roster.len() {
                let id = PokeId {
                    side: side as u8,
                    slot: slot as u8,
                };
                if id != active {
                    let p = b.poke_mut(id);
                    p.hp = 0;
                    p.fainted = true;
                }
            }
        }
        (dex, b)
    }

    fn normal() -> (Dex, Battle) {
        battle(
            ["Rest", "Double-Edge", "Earthquake", "Curse"],
            "",
            ["Toxic", "Whirlwind", "Drill Peck", "Protect"],
            "",
        )
    }

    fn double_rest() -> (Dex, Battle) {
        battle(
            ["Rest", "Double-Edge", "Earthquake", "Curse"],
            "",
            ["Rest", "Toxic", "Drill Peck", "Protect"],
            "",
        )
    }

    #[test]
    fn classifies_last_mon_one_sided_rest_and_builds_rank() {
        let (dex, b) = normal();
        let class = classify_one_sided_heal(&b, &dex).unwrap();
        assert_eq!((class.healer, class.non_healer), (0, 1));
        let rank = class.rank(&b).unwrap();
        let expected_pp: i64 = (0..2)
            .map(|s| {
                b.poke(b.active_id(s).unwrap())
                    .move_slots
                    .iter()
                    .map(|m| i64::from(m.pp))
                    .sum::<i64>()
            })
            .sum();
        assert_eq!(rank.pp_total, expected_pp);
        assert_eq!(
            rank.value,
            expected_pp + i64::from(rank.non_healer_hp) + 1001 - i64::from(b.turn)
        );
    }

    #[test]
    fn classifies_rest_vs_rest_and_leftovers_vs_rest() {
        let (dex, b) = double_rest();
        let class = classify_two_sided_heal(&b, &dex).unwrap();
        let rank = class.rank(&b).unwrap();
        assert!(rank.healing_pp > 0);
        assert_eq!(
            rank.lexicographic_key(),
            (rank.pp_total, rank.turns_remaining)
        );

        // b455 shape: Snorlax heals from Leftovers while only Skarmory has
        // a healing move slot.
        let (dex, b) = battle(
            ["Double-Edge", "Earthquake", "Self-Destruct", "Curse"],
            "Leftovers",
            ["Toxic", "Whirlwind", "Drill Peck", "Rest"],
            "",
        );
        let class = classify_two_sided_heal(&b, &dex).unwrap();
        let rank = class.rank(&b).unwrap();
        let skarmory = b.active_id(1).unwrap();
        let rest = dex.moves.id("rest").unwrap();
        let expected_healing_pp = b
            .poke(skarmory)
            .move_slots
            .iter()
            .find(|slot| slot.id == rest)
            .unwrap()
            .pp;
        assert_eq!(rank.healing_pp, i64::from(expected_healing_pp));
    }

    #[test]
    fn two_sided_edge_allows_both_hp_to_rise_and_rank_decreases() {
        let (dex, mut parent) = double_rest();
        for side in 0..2 {
            let active = parent.active_id(side).unwrap();
            parent.poke_mut(active).hp -= 20;
        }
        let class = classify_two_sided_heal(&parent, &dex).unwrap();
        let mut child = parent.clone();
        child.turn += 1;
        for side in 0..2 {
            let active = child.active_id(side).unwrap();
            child.poke_mut(active).hp += 10;
        }

        assert!(class.check_edge(&parent, &child).is_ok());
        let parent_rank = class.rank(&parent).unwrap();
        let child_rank = class.rank(&child).unwrap();
        assert_eq!(child_rank.healing_pp, parent_rank.healing_pp);
        assert!(child_rank.lexicographic_key() < parent_rank.lexicographic_key());
    }

    #[test]
    fn two_sided_edge_rejects_pp_increase_and_same_turn() {
        let (dex, parent) = double_rest();
        let class = classify_two_sided_heal(&parent, &dex).unwrap();

        let mut pp_increase = parent.clone();
        pp_increase.turn += 1;
        let active = pp_increase.active_id(0).unwrap();
        pp_increase.poke_mut(active).move_slots[0].pp += 1;
        assert!(matches!(
            class.check_edge(&parent, &pp_increase),
            Err(EdgeError::PpIncreased {
                side: 0,
                slot: 0,
                ..
            })
        ));

        let mut same_turn = parent.clone();
        let active = same_turn.active_id(0).unwrap();
        same_turn.poke_mut(active).move_slots[0].pp -= 1;
        assert!(matches!(
            class.check_edge(&parent, &same_turn),
            Err(EdgeError::TurnDidNotIncrease { .. })
        ));
    }

    #[test]
    fn two_sided_classifier_rejects_pp_and_slot_mutation_roots() {
        let (dex, b) = battle(
            ["Rest", "Sleep Talk", "Earthquake", "Curse"],
            "Mystery Berry",
            ["Rest", "Toxic", "Drill Peck", "Protect"],
            "",
        );
        assert_eq!(
            classify_two_sided_heal(&b, &dex),
            Err(ClassifyError::MysteryBerry { side: 0 })
        );

        let (dex, b) = battle(
            ["Rest", "Mimic", "Earthquake", "Curse"],
            "",
            ["Rest", "Toxic", "Drill Peck", "Protect"],
            "",
        );
        assert!(matches!(
            classify_two_sided_heal(&b, &dex),
            Err(ClassifyError::UnsafeMove { side: 0, .. })
        ));
    }

    #[test]
    fn rejects_two_healers_and_nonhealer_healing_item() {
        let (dex, b) = battle(
            ["Rest", "Double-Edge", "Earthquake", "Curse"],
            "",
            ["Recover", "Toxic", "Drill Peck", "Protect"],
            "",
        );
        assert_eq!(
            classify_one_sided_heal(&b, &dex),
            Err(ClassifyError::NotExactlyOneHealer)
        );

        let (dex, b) = battle(
            ["Rest", "Double-Edge", "Earthquake", "Curse"],
            "",
            ["Toxic", "Whirlwind", "Drill Peck", "Protect"],
            "Gold Berry",
        );
        assert!(matches!(
            classify_one_sided_heal(&b, &dex),
            Err(ClassifyError::NonHealerHealingItem { side: 1, .. })
        ));
    }

    #[test]
    fn rejects_nonquiescent_and_non_last_mon_states() {
        let (dex, mut b) = normal();
        b.sides[1].pokemon_left = 2;
        assert_eq!(
            classify_one_sided_heal(&b, &dex),
            Err(ClassifyError::NotLastMon { side: 1 })
        );

        b.sides[1].pokemon_left = 1;
        b.mid_turn = true;
        assert_eq!(
            classify_one_sided_heal(&b, &dex),
            Err(ClassifyError::NotQuiescent)
        );
    }

    #[test]
    fn rejects_mutation_call_copy_and_hp_transfer_moves() {
        for key in [
            "Present",
            "Pain Split",
            "Mimic",
            "Sketch",
            "Transform",
            "Spite",
            "Metronome",
            "Mirror Move",
            "Thief",
        ] {
            let (dex, b) = battle(
                ["Rest", "Double-Edge", "Earthquake", "Curse"],
                "",
                [key, "Toxic", "Drill Peck", "Protect"],
                "",
            );
            assert!(
                matches!(
                    classify_one_sided_heal(&b, &dex),
                    Err(ClassifyError::UnsafeMove { side: 1, .. })
                ),
                "{key}"
            );
        }
    }

    #[test]
    fn rejects_mystery_berry_but_allows_sleep_talk() {
        let (dex, b) = battle(
            ["Rest", "Sleep Talk", "Earthquake", "Curse"],
            "Mystery Berry",
            ["Toxic", "Whirlwind", "Drill Peck", "Protect"],
            "",
        );
        assert_eq!(
            classify_one_sided_heal(&b, &dex),
            Err(ClassifyError::MysteryBerry { side: 0 })
        );

        let (dex, b) = battle(
            ["Rest", "Sleep Talk", "Earthquake", "Curse"],
            "",
            ["Toxic", "Whirlwind", "Drill Peck", "Protect"],
            "",
        );
        assert!(classify_one_sided_heal(&b, &dex).is_ok());
    }

    #[test]
    fn existing_leech_seed_assigns_healing_to_its_active_source() {
        let (dex, mut b) = battle(
            ["Double-Edge", "Earthquake", "Curse", "Protect"],
            "",
            ["Toxic", "Whirlwind", "Drill Peck", "Protect"],
            "",
        );
        let seeded = b.active_id(0).unwrap();
        let source = b.active_id(1).unwrap();
        b.add_volatile(
            &dex,
            seeded,
            "leechseed",
            Some(source),
            nc2000_engine::battle::EffectHandle::None,
        );
        // Import reconstruction can leave the object reference stale while
        // source_slot (the field mechanics use) still names the active mon.
        let leech_seed = dex.conds_id("leechseed").unwrap();
        b.poke_mut(seeded).volatile_mut(leech_seed).unwrap().source =
            Some(PokeId { side: 1, slot: 2 });
        let class = classify_one_sided_heal(&b, &dex).unwrap();
        assert_eq!((class.healer, class.non_healer), (1, 0));
    }

    #[test]
    fn edge_requires_componentwise_monotonicity_and_strict_rank_progress() {
        let (dex, parent) = normal();
        let class = classify_one_sided_heal(&parent, &dex).unwrap();
        let mut child = parent.clone();
        child.turn += 1;
        child.poke_mut(class.active[0]).move_slots[0].pp -= 1;
        child.poke_mut(class.active[1]).hp -= 5;
        assert!(class.check_edge(&parent, &child).is_ok());
        assert!(class.rank(&child).unwrap().value < class.rank(&parent).unwrap().value);

        let mut bad_pp = child.clone();
        bad_pp.poke_mut(class.active[1]).move_slots[0].pp += 1;
        assert!(matches!(
            class.check_edge(&child, &bad_pp),
            Err(EdgeError::PpIncreased { .. })
        ));

        let mut bad_hp = child.clone();
        bad_hp.turn += 1;
        bad_hp.poke_mut(class.active[1]).hp += 1;
        assert!(matches!(
            class.check_edge(&child, &bad_hp),
            Err(EdgeError::NonHealerHpIncreased { .. })
        ));

        let mut same_turn = child.clone();
        same_turn.poke_mut(class.active[1]).hp -= 1;
        assert!(matches!(
            class.check_edge(&child, &same_turn),
            Err(EdgeError::TurnDidNotIncrease { .. })
        ));
    }

    #[test]
    fn edge_rejects_identity_drift() {
        let (dex, parent) = normal();
        let class = classify_one_sided_heal(&parent, &dex).unwrap();

        let mut slots = parent.clone();
        slots.turn += 1;
        slots.poke_mut(class.active[1]).move_slots[0].id = dex.moves.id("surf").unwrap();
        assert_eq!(
            class.check_edge(&parent, &slots),
            Err(EdgeError::MoveSlotsChanged { side: 1 })
        );

        let mut maxhp = parent.clone();
        maxhp.turn += 1;
        maxhp.poke_mut(class.active[0]).maxhp += 1;
        assert_eq!(
            class.check_edge(&parent, &maxhp),
            Err(EdgeError::MaxHpChanged { side: 0 })
        );
    }
}
