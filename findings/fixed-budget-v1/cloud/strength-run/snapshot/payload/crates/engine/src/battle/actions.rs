//! Switch machinery: base `switchIn`/`dragIn` + gen4 `runSwitch`.

use crate::dex::Dex;
use crate::state::*;

use super::events::{ev, EvTarget};
use super::EffectHandle;

impl Battle {
    /// battle.canSwitch(side) — count of bench non-fainted.
    pub fn can_switch(&self, side_n: u8) -> bool {
        self.possible_switches(side_n).next().is_some()
    }

    fn possible_switches(&self, side_n: u8) -> impl Iterator<Item = PokeId> + '_ {
        let side = &self.sides[side_n as usize];
        side.party
            .iter()
            .enumerate()
            .filter(move |(i, _)| *i >= 1) // singles: active.length == 1
            .map(move |(_, &slot)| PokeId { side: side_n, slot })
            .filter(move |id| !self.poke(*id).fainted && side.pokemon_left > 0)
    }

    /// battle.getRandomSwitchable.
    pub fn get_random_switchable(&mut self, side_n: u8) -> Option<PokeId> {
        let can: Vec<PokeId> = self.possible_switches(side_n).collect();
        if can.is_empty() {
            None
        } else {
            let idx = self.prng.sample_index(can.len());
            Some(can[idx])
        }
    }

    /// actions.switchIn(pokemon, pos, sourceEffect, isDrag). Returns
    /// Ok(true)=switched, Ok(false)=failed, Err(())='pursuitfaint'.
    pub fn switch_in(
        &mut self,
        dex: &Dex,
        pokemon: PokeId,
        pos: u8,
        source_effect: Option<EffectHandle>,
        is_drag: bool,
    ) -> Result<bool, ()> {
        if self.poke(pokemon).is_active {
            return Ok(false);
        }
        let side_n = pokemon.side;
        let old_active = self.active_id(side_n as usize);
        let unfainted_old = old_active.filter(|&o| self.poke(o).hp > 0);

        if let Some(old) = unfainted_old {
            self.poke_mut(old).being_called_back = true;
            let switch_copy = matches!(
                source_effect,
                Some(EffectHandle::MoveEff(m))
                    if dex.move_static(m).self_switch.as_deref() == Some("copyvolatile")
            );
            if !self.poke(old).skip_before_switch_out && !is_drag {
                self.run_event(dex, &ev::BeforeSwitchOut, EvTarget::Poke(old), None, EffectHandle::None, None, false, false);
            }
            self.poke_mut(old).skip_before_switch_out = false;
            if !self
                .run_event(dex, &ev::SwitchOut, EvTarget::Poke(old), None, EffectHandle::None, None, false, false)
                .truthy()
            {
                return Ok(false);
            }
            if self.poke(old).hp <= 0 {
                return Err(()); // pursuitfaint
            }
            // End events for ability/item: no gen2 handlers.
            self.queue_cancel_action(old);
            if switch_copy {
                self.copy_volatile_from(dex, pokemon, old);
            }
            self.clear_volatile(dex, old, true);
        }
        if let Some(old) = old_active {
            let p = self.poke_mut(old);
            p.is_active = false;
            p.is_started = false;
            p.used_item_this_turn = false;
            p.stats_raised_this_turn = false;
            p.stats_lowered_this_turn = false;
            // position swap
            let new_pos = self.poke(pokemon).position;
            self.poke_mut(old).position = new_pos;
            if self.poke(old).fainted {
                self.poke_mut(old).status = Status::None;
                self.refresh_poke_mask(dex, old);
            }
            // gen <= 4: incoming takes outgoing's lastItem
            let old_last_item = self.poke(old).last_item;
            self.poke_mut(pokemon).last_item = old_last_item;
            self.poke_mut(old).last_item = None;
            self.poke_mut(pokemon).position = pos;
            // sync party display order
            let old_slot = old.slot;
            let side = &mut self.sides[side_n as usize];
            side.party[pos as usize] = pokemon.slot;
            side.party[new_pos as usize] = old_slot;
        } else {
            self.poke_mut(pokemon).position = pos;
            let side = &mut self.sides[side_n as usize];
            side.party[pos as usize] = pokemon.slot;
        }
        self.poke_mut(pokemon).is_active = true;
        self.sides[side_n as usize].active = Some(pokemon.slot);
        {
            let p = self.poke_mut(pokemon);
            p.active_turns = 0;
            p.active_move_actions = 0;
            for slot in &mut p.move_slots {
                slot.used = false;
            }
            for slot in &mut p.base_move_slots {
                slot.used = false;
            }
        }
        // abilityState/itemState re-init: itemState gets a fresh effectOrder
        // when the holder is now active and has an item.
        let has_item = self.poke(pokemon).item.is_some();
        let state = EffectState {
            id: self.poke(pokemon).item.map(crate::state::EffId::Item).unwrap_or_default(),
            ..Default::default()
        };
        let state = self.init_effect_state(state, has_item);
        self.poke_mut(pokemon).item_state = state;

        self.run_event(dex, &ev::BeforeSwitchIn, EvTarget::Poke(pokemon), None, EffectHandle::None, None, false, false);
        // |switch| / |drag|
        let ps = self.poke_str(pokemon);
        let details = self.details(dex, pokemon);
        let (secret, shared) = self.get_health(pokemon);
        let side_id = format!("p{}", side_n + 1);
        let tag = if is_drag { "drag" } else { "switch" };
        match source_effect {
            Some(se) if !se.is_none() => {
                // PS: '[from] ' + sourceEffect (toString = plain name)
                let from = format!("[from] {}", self.effect_name(dex, se));
                self.add_split(
                    &side_id,
                    &[tag, &ps, &details, &secret, &from],
                    &[tag, &ps, &details, &shared, &from],
                );
            }
            _ => {
                self.add_split(
                    &side_id,
                    &[tag, &ps, &details, &secret],
                    &[tag, &ps, &details, &shared],
                );
            }
        }
        if is_drag {
            // gen2: draggedIn = turn
            self.poke_mut(pokemon).dragged_in = Some(self.turn);
        }
        self.poke_mut(pokemon).previously_switched_in += 1;

        if is_drag {
            // gen < 5 → still insertChoice (only gen >= 5 runs runSwitch
            // immediately)
            self.insert_choice_run_switch(dex, pokemon);
        } else {
            self.insert_choice_run_switch(dex, pokemon);
        }
        Ok(true)
    }

    /// queue.insertChoice({choice:'runSwitch', pokemon}).
    fn insert_choice_run_switch(&mut self, dex: &Dex, pokemon: PokeId) {
        self.update_speed(dex, pokemon);
        let action = Action {
            choice: ActionKind::RunSwitch,
            order: 101,
            priority: 0.0,
            fractional_priority: 0.0,
            speed: self.poke(pokemon).speed as f64,
            pokemon: Some(pokemon),
        };
        self.insert_action_sorted(action);
    }

    /// queue.insertChoice: insert by comparePriority without re-sorting,
    /// random position among full ties.
    pub fn insert_action_sorted(&mut self, action: Action) {
        let mut first_index: Option<usize> = None;
        let mut last_index: Option<usize> = None;
        for (i, cur) in self.queue.iter().enumerate() {
            let compared = compare_action_priority(&action, cur);
            if compared <= 0.0 && first_index.is_none() {
                first_index = Some(i);
            }
            if compared < 0.0 {
                last_index = Some(i);
                break;
            }
        }
        match first_index {
            None => self.queue.push(action),
            Some(fi) => {
                let li = last_index.unwrap_or(self.queue.len());
                let index = if fi == li {
                    fi
                } else {
                    self.prng.random_range(fi as u32, li as u32 + 1) as usize
                };
                self.queue.insert(index, action);
            }
        }
    }

    /// queue.cancelAction(pokemon): remove all queued actions by this pokemon.
    pub fn queue_cancel_action(&mut self, pokemon: PokeId) -> bool {
        let before = self.queue.len();
        self.queue.retain(|a| a.pokemon != Some(pokemon));
        self.queue.len() != before
    }

    /// queue.cancelMove(pokemon): remove the first queued 'move' action.
    pub fn queue_cancel_move(&mut self, pokemon: PokeId) -> bool {
        if let Some(i) = self
            .queue
            .iter()
            .position(|a| matches!(a.choice, ActionKind::Move { .. }) && a.pokemon == Some(pokemon))
        {
            self.queue.remove(i);
            true
        } else {
            false
        }
    }

    /// **gen4** actions.runSwitch (the override that applies to gen2) — NOT
    /// the modern base version: no allActive speedSort, no fieldEvent.
    pub fn run_switch(&mut self, dex: &Dex, pokemon: PokeId) -> bool {
        self.run_event(dex, &ev::EntryHazard, EvTarget::Poke(pokemon), None, EffectHandle::None, None, false, false);
        self.run_event(dex, &ev::SwitchIn, EvTarget::Poke(pokemon), None, EffectHandle::None, None, false, false);
        // gen <= 2: pokemon.lastMove is reset for ALL actives (Mirror Move)
        for active in self.get_all_active(false) {
            self.poke_mut(active).last_move = None;
        }
        let side_fainted = self.sides[pokemon.side as usize].fainted_this_turn.is_some();
        let dragged_this_turn = self.poke(pokemon).dragged_in == Some(self.turn);
        if !side_fainted && !dragged_this_turn {
            self.run_event(dex, &ev::AfterSwitchInSelf, EvTarget::Poke(pokemon), None, EffectHandle::None, None, false, false);
        }
        if self.poke(pokemon).hp <= 0 {
            return false;
        }
        self.poke_mut(pokemon).is_started = true;
        if !self.poke(pokemon).fainted {
            // singleEvent Start for ability (none) and item (M2: berserkgene…)
            if let Some(item) = self.poke(pokemon).item {
                if dex.items.get(item).mask.has(dex.known.on_start) {
                    self.single_event(
                        dex,
                        &ev::Start,
                        EffectHandle::Item(item),
                        StateLoc::None,
                        EvTarget::Poke(pokemon),
                        None,
                        EffectHandle::None,
                        None,
                    );
                }
            }
        }
        self.poke_mut(pokemon).dragged_in = None;
        true
    }

    /// actions.dragIn.
    pub fn drag_in(&mut self, dex: &Dex, side_n: u8, pos: u8) -> bool {
        let Some(pokemon) = self.get_random_switchable(side_n) else { return false };
        if self.poke(pokemon).is_active {
            return false;
        }
        let Some(old_active) = self.active_id(side_n as usize) else { return false };
        if self.poke(old_active).hp <= 0 {
            return false;
        }
        if !self
            .run_event(dex, &ev::DragOut, EvTarget::Poke(old_active), None, EffectHandle::None, None, false, false)
            .truthy()
        {
            return false;
        }
        matches!(self.switch_in(dex, pokemon, pos, None, true), Ok(true))
    }
}

/// comparePriority for queue actions.
pub fn compare_action_priority(a: &Action, b: &Action) -> f64 {
    let d = a.order as f64 - b.order as f64;
    if d != 0.0 {
        return d;
    }
    let d = b.priority - a.priority;
    if d != 0.0 {
        return d;
    }
    let d = b.speed - a.speed;
    if d != 0.0 {
        return d;
    }
    0.0
}
