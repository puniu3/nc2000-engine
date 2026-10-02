//! Choice handling: side.choose (team/move/switch/pass), commitChoices,
//! queue.resolveAction + battle.getActionSpeed.

use crate::choice::Choice as ParsedChoice;
use crate::dex::{toid, Dex, MoveId};
use crate::state::*;

use super::actions::compare_action_priority;
use super::events::{ev, EvTarget};
use super::{EffectHandle, EngineError, RV};

/// NC2000's `Max Total Level = 155` preview cap (PS rulesets.ts
/// `maxtotallevel`, bound in config/formats.ts). Single-format port:
/// hardcoded like the rest of the format's rule table.
///
/// Certified against the live PS clone 2026-07-17
/// (`tools/probe-max-total-level.js`):
/// - the rule binds on the PICKED mons' level sum, threshold INCLUSIVE
///   (sum 155 accepted, sum 156 rejected);
/// - rejection is a CHOICE error on the player stream
///   (`|error|[Invalid choice] Can't choose for Team Preview: Your selected
///   team has a total level of N, but it can't be above 155.`) — the
///   request stays open and a corrected pick is accepted;
/// - lead ordering is irrelevant (the rule sums the picked set);
/// - a team with NO legal triple cannot exist: TeamValidator rejects any
///   team whose 3 lowest levels sum over 155 (`maxtotallevel.onValidateTeam`
///   check 1; mirrored by validate.rs `level-sum`), so filtering the
///   preview enumeration never empties it for a validator-legal team.
pub const MAX_TOTAL_LEVEL: u32 = 155;

impl Battle {
    pub fn choose(&mut self, dex: &Dex, side_n: usize, input: &str) -> Result<(), EngineError> {
        let parsed = ParsedChoice::parse(input)
            .map_err(|e| EngineError::InvalidChoice(format!("p{}: {e}", side_n + 1)))?;
        self.submit_choice(dex, side_n, || input.to_owned(), |battle| {
            for part in &parsed {
                match part {
                    ParsedChoice::Team(slots) => battle.choose_team(dex, side_n, slots)?,
                    ParsedChoice::MoveById(id) => battle.choose_move(dex, side_n, &toid(id))?,
                    ParsedChoice::MoveBySlot(slot) => battle.choose_move_slot(dex, side_n, *slot)?,
                    ParsedChoice::Switch(slot) => battle.choose_switch(dex, side_n, *slot)?,
                    ParsedChoice::Pass => battle.choose_pass(side_n)?,
                    ParsedChoice::Default => return Err(EngineError::Unimplemented("choice: default")),
                    ParsedChoice::Undo => return Err(EngineError::Unimplemented("choice: undo")),
                }
            }
            Ok(())
        })
    }

    pub(crate) fn choose_search(
        &mut self,
        dex: &Dex,
        side_n: usize,
        choice: super::SearchChoice,
    ) -> Result<(), EngineError> {
        use super::SearchChoice;
        if choice == SearchChoice::Team([0; 3]) {
            return self.choose(dex, side_n, &choice.to_input(dex));
        }
        self.submit_choice(dex, side_n, || choice.to_input(dex), |battle| {
            match choice {
                SearchChoice::Team(slots) => {
                    let mut nonzero = [0; 3];
                    let mut n = 0;
                    for slot in slots {
                        if slot != 0 { nonzero[n] = slot; n += 1; }
                    }
                    battle.choose_team(dex, side_n, &nonzero[..n])
                }
                SearchChoice::Move(id) => battle.choose_move_id(dex, side_n, dex.moves.key(id), Some(id)),
                SearchChoice::Switch(slot) => battle.choose_switch(dex, side_n, slot),
                SearchChoice::Pass => battle.choose_pass(side_n),
            }
        })
    }

    fn submit_choice(
        &mut self,
        dex: &Dex,
        side_n: usize,
        input: impl FnOnce() -> String,
        choose: impl FnOnce(&mut Self) -> Result<(), EngineError>,
    ) -> Result<(), EngineError> {
        if self.sides[side_n].request_state().is_none() {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: no request pending for choice {:?}", side_n + 1, input()
            )));
        }
        self.clear_choice(side_n);
        choose(self)?;
        if !self.is_choice_done(side_n) {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: incomplete choice {:?}", side_n + 1, input()
            )));
        }
        if self.all_choices_done() {
            self.commit_choices(dex);
        }
        Ok(())
    }

    /// side.isChoiceDone.
    fn is_choice_done(&self, side_n: usize) -> bool {
        let side = &self.sides[side_n];
        let Some(request) = side.request_state() else { return true };
        if side.choice.forced_switches_left > 0 {
            return false;
        }
        if request == RequestKind::TeamPreview {
            return side.choice.actions.len() >= self.picked_team_size(side_n);
        }
        // singles: one action needed
        !side.choice.actions.is_empty()
    }

    pub(crate) fn picked_team_size(&self, side_n: usize) -> usize {
        self.sides[side_n].party.len().min(3)
    }

    /// Level sum of a team-preview pick given as display positions
    /// (0-based). Shared by `choose_team` validation and the `search.rs`
    /// enumeration — one rule, one code path.
    pub(crate) fn picked_total_level(&self, side_n: usize, positions: &[usize]) -> u32 {
        let side = &self.sides[side_n];
        positions
            .iter()
            .map(|&pos| side.roster[side.party[pos] as usize].level as u32)
            .sum()
    }

    fn all_choices_done(&self) -> bool {
        (0..2).all(|n| self.is_choice_done(n))
    }

    // ------------------------------------------------------------ choosers

    /// side.chooseTeam (positions are 1-based in the choice string).
    fn choose_team(&mut self, dex: &Dex, side_n: usize, slots: &[u8]) -> Result<(), EngineError> {
        let _ = dex;
        if self.sides[side_n].request_state() != Some(RequestKind::TeamPreview) {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: team choice outside team preview",
                side_n + 1
            )));
        }
        let picked = self.picked_team_size(side_n);
        let mut positions: Vec<usize> = slots.iter().map(|&s| s as usize - 1).collect();
        positions.truncate(picked);
        let mut i = 0;
        while positions.len() < picked && i < picked {
            if !positions.contains(&i) {
                positions.push(i);
            }
            i += 1;
        }
        for (index, &pos) in positions.iter().enumerate() {
            if pos >= self.sides[side_n].party.len() {
                return Err(EngineError::InvalidChoice(format!(
                    "p{}: no pokemon in slot {}",
                    side_n + 1,
                    pos + 1
                )));
            }
            if positions.iter().position(|&p| p == pos) != Some(index) {
                return Err(EngineError::InvalidChoice(format!(
                    "p{}: duplicate team slot {}",
                    side_n + 1,
                    pos + 1
                )));
            }
        }
        let total = self.picked_total_level(side_n, &positions);
        if self.preview_level_caps[side_n].is_some_and(|cap| total > cap) {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: selected team has a total level of {}, above the {} cap",
                side_n + 1,
                total,
                self.preview_level_caps[side_n].unwrap()
            )));
        }
        for (index, &pos) in positions.iter().enumerate() {
            let slot = self.sides[side_n].party[pos];
            self.sides[side_n].choice.switch_ins.push(pos as u8);
            self.sides[side_n].choice.actions.push(ChosenAction::Team {
                pokemon: PokeId { side: side_n as u8, slot },
                index: index as u8,
                priority: -(index as i32),
            });
        }
        Ok(())
    }

    /// side.chooseMove by id.
    fn choose_move(&mut self, dex: &Dex, side_n: usize, move_id_str: &str) -> Result<(), EngineError> {
        self.choose_move_id(dex, side_n, move_id_str, None)
    }

    fn choose_move_id(&mut self, dex: &Dex, side_n: usize, move_id_str: &str, resolved: Option<MoveId>) -> Result<(), EngineError> {
        if self.sides[side_n].request_state() != Some(RequestKind::Move) {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: move choice without move request",
                side_n + 1
            )));
        }
        let pokemon = self
            .active_id(side_n)
            .ok_or_else(|| EngineError::InvalidChoice("no active pokemon".into()))?;

        // locked moves take precedence
        let locked = self
            .get_locked_move(dex, pokemon)
            .or_else(|| self.get_semi_locked_move(dex, pokemon));
        if let Some(locked_id) = locked {
            let target_loc = self.poke(pokemon).last_move_target_loc.unwrap_or(0);
            let lm = dex
                .moves
                .id(&locked_id)
                .or_else(|| dex.moves.id("recharge"))
                .ok_or_else(|| EngineError::InvalidChoice(format!("locked move {locked_id}")))?;
            self.sides[side_n].choice.actions.push(ChosenAction::Move {
                pokemon,
                target_loc,
                move_id: lm,
                move_slot: None,
            });
            return Ok(());
        }

        // no valid moves → Struggle
        let moves = &self.poke(pokemon).move_slots;
        if !moves.iter().any(|slot| !slot.disabled && slot.pp > 0) {
            let struggle = dex.moves.id("struggle").unwrap();
            self.sides[side_n].choice.actions.push(ChosenAction::Move {
                pokemon,
                target_loc: 0,
                move_id: struggle,
                move_slot: None,
            });
            return Ok(());
        }

        let move_id = resolved.or_else(|| dex.moves.id(move_id_str))
            .ok_or_else(|| EngineError::InvalidChoice(format!("unknown move {move_id_str}")))?;
        let mut move_slot = None;
        let mut enabled = false;
        for (i, slot) in moves.iter().enumerate() {
            if slot.id == move_id {
                if move_slot.is_none() {
                    move_slot = Some(i);
                }
                if !slot.disabled && slot.pp > 0 {
                    enabled = true;
                    break;
                }
            }
        }
        if move_slot.is_none() {
            // struggle chosen explicitly (not in move slots)
            if move_id_str == "struggle" {
                self.sides[side_n].choice.actions.push(ChosenAction::Move {
                    pokemon,
                    target_loc: 0,
                    move_id,
                    move_slot: None,
                });
                return Ok(());
            }
            return Err(EngineError::InvalidChoice(format!(
                "p{}: doesn't have {move_id_str}",
                side_n + 1
            )));
        }
        if !enabled {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: {move_id_str} is disabled",
                side_n + 1
            )));
        }
        self.sides[side_n].choice.actions.push(ChosenAction::Move {
            pokemon,
            target_loc: 0,
            move_id,
            move_slot,
        });
        Ok(())
    }

    fn choose_move_slot(&mut self, dex: &Dex, side_n: usize, slot: u8) -> Result<(), EngineError> {
        let pokemon = self
            .active_id(side_n)
            .ok_or_else(|| EngineError::InvalidChoice("no active pokemon".into()))?;
        let moves = self.pokemon_choosable_moves(pokemon);
        let idx = slot as usize - 1;
        if idx >= moves.len() {
            return Err(EngineError::InvalidChoice(format!("no move slot {slot}")));
        }
        let id = moves[idx].0;
        self.choose_move_id(dex, side_n, dex.moves.key(id), Some(id))
    }

    /// pokemon.getMoves() reduced to (id, disabled) pairs; empty if no valid.
    pub(crate) fn pokemon_choosable_moves(&self, pokemon: PokeId) -> Vec<(MoveId, bool)> {
        let p = self.poke(pokemon);
        let mut out = Vec::new();
        let mut has_valid = false;
        for slot in &p.move_slots {
            let disabled = slot.disabled || slot.pp <= 0;
            if !disabled {
                has_valid = true;
            }
            out.push((slot.id, disabled));
        }
        if has_valid {
            out
        } else {
            Vec::new()
        }
    }

    /// side.chooseSwitch.
    fn choose_switch(&mut self, dex: &Dex, side_n: usize, slot_1based: u8) -> Result<(), EngineError> {
        let _ = dex;
        let request = self.sides[side_n].request_state();
        if request != Some(RequestKind::Move) && request != Some(RequestKind::Switch) {
            return Err(EngineError::InvalidChoice(format!(
                "p{}: switch choice without move/switch request",
                side_n + 1
            )));
        }
        let pokemon = self
            .active_id(side_n)
            .ok_or_else(|| EngineError::InvalidChoice("no active pokemon".into()))?;
        let slot = slot_1based as usize - 1;
        if slot >= self.sides[side_n].party.len() {
            return Err(EngineError::InvalidChoice(format!("no pokemon in slot {slot_1based}")));
        }
        if slot < 1 {
            return Err(EngineError::InvalidChoice("can't switch to an active pokemon".into()));
        }
        if self.sides[side_n].choice.switch_ins.contains(&(slot as u8)) {
            return Err(EngineError::InvalidChoice("already switching in".into()));
        }
        let target_slot = self.sides[side_n].party[slot];
        let target = PokeId { side: side_n as u8, slot: target_slot };
        if self.poke(target).fainted {
            return Err(EngineError::InvalidChoice("can't switch to a fainted pokemon".into()));
        }
        if request == Some(RequestKind::Move) {
            if self.poke(pokemon).trapped {
                return Err(EngineError::InvalidChoice("trapped".into()));
            }
        } else if request == Some(RequestKind::Switch) {
            if self.sides[side_n].choice.forced_switches_left == 0 {
                return Err(EngineError::InvalidChoice("switched too many".into()));
            }
            self.sides[side_n].choice.forced_switches_left -= 1;
        }
        self.sides[side_n].choice.switch_ins.push(slot as u8);
        let insta = request == Some(RequestKind::Switch);
        self.sides[side_n].choice.actions.push(ChosenAction::Switch { insta, pokemon, target });
        Ok(())
    }

    fn choose_pass(&mut self, side_n: usize) -> Result<(), EngineError> {
        match self.sides[side_n].request_state() {
            Some(RequestKind::Switch) => {
                let needs_switch = self
                    .active_id(side_n)
                    .map(|a| self.poke(a).switch_flag.is_set())
                    .unwrap_or(false);
                if needs_switch {
                    if self.sides[side_n].choice.forced_passes_left == 0 {
                        return Err(EngineError::InvalidChoice("can't pass".into()));
                    }
                    self.sides[side_n].choice.forced_passes_left -= 1;
                }
            }
            Some(RequestKind::Move) => {
                let fainted = self
                    .active_id(side_n)
                    .map(|a| self.poke(a).fainted)
                    .unwrap_or(true);
                if !fainted {
                    return Err(EngineError::InvalidChoice("must make a move".into()));
                }
            }
            _ => return Err(EngineError::InvalidChoice("can't pass".into())),
        }
        self.sides[side_n].choice.actions.push(ChosenAction::Pass);
        Ok(())
    }

    // -------------------------------------------------------------- commit

    /// battle.commitChoices.
    pub fn commit_choices(&mut self, dex: &Dex) {
        self.update_all_speeds(dex);
        let old_queue = std::mem::take(&mut self.queue);

        // side.commitChoices → queue.addChoice(actions)
        for side_n in 0..2usize {
            let actions = std::mem::take(&mut self.sides[side_n].choice.actions);
            for action in actions {
                // resolveAction unshifts a beforeTurnMove for moves with a
                // beforeTurnCallback (pursuit) when !midTurn.
                if let ChosenAction::Move { pokemon, target_loc, move_id, .. } = &action {
                    if dex
                        .move_static(*move_id)
                        .callbacks
                        .iter()
                        .any(|c| c == "beforeTurnCallback")
                    {
                        let speed = self.get_pokemon_action_speed(dex, *pokemon) as f64;
                        let mut btm_loc = *target_loc;
                        if btm_loc == 0 {
                            let ms = dex.move_static(*move_id);
                            if let Some(t) = self.get_random_target(&ms.target, *pokemon) {
                                btm_loc = if t.side == pokemon.side { -1 } else { 1 };
                            }
                        }
                        self.queue.push(Action {
                            choice: ActionKind::BeforeTurnMove { move_id: *move_id, target_loc: btm_loc },
                            order: 5,
                            priority: 0.0,
                            fractional_priority: 0.0,
                            speed,
                            pokemon: Some(*pokemon),
                        });
                    }
                }
                let resolved = self.resolve_action(dex, action, false);
                if let Some(a) = resolved {
                    self.queue.push(a);
                }
            }
        }
        // clearRequest
        self.request_state = RequestState::None;
        for side_n in 0..2usize {
            self.sides[side_n].request = None;
            self.clear_choice(side_n);
        }

        // queue.sort() — speedSort with PRNG ties
        let mut queue = std::mem::take(&mut self.queue);
        self.speed_sort(&mut queue, compare_action_priority);
        self.queue = queue;
        self.queue.extend(old_queue);

        self.turn_loop(dex);
    }

    /// queue.resolveAction (midTurn unused in M1 paths that call this).
    fn resolve_action(&mut self, dex: &Dex, action: ChosenAction, _mid_turn: bool) -> Option<Action> {
        match action {
            ChosenAction::Pass => None,
            ChosenAction::Team { pokemon, index, priority } => {
                self.update_speed(dex, pokemon);
                let speed = self.get_pokemon_action_speed(dex, pokemon) as f64;
                Some(Action {
                    choice: ActionKind::Team { index },
                    order: 1,
                    priority: priority as f64,
                    fractional_priority: 0.0,
                    speed,
                    pokemon: Some(pokemon),
                })
            }
            ChosenAction::Switch { insta, pokemon, target } => {
                // switchFlag string → sourceEffect
                let source_effect = match &self.poke(pokemon).switch_flag {
                    SwitchFlag::Move(m) => Some(*m),
                    _ => None,
                };
                self.poke_mut(pokemon).switch_flag = SwitchFlag::No;
                let speed = self.get_pokemon_action_speed(dex, pokemon) as f64;
                Some(Action {
                    choice: ActionKind::Switch { insta, target, source_effect },
                    order: if insta { 3 } else { 103 },
                    priority: 0.0,
                    fractional_priority: 0.0,
                    speed,
                    pokemon: Some(pokemon),
                })
            }
            ChosenAction::Move { pokemon, target_loc, move_id, move_slot: _ } => {
                // fractionalPriority = runEvent('FractionalPriority', pokemon, null, move, 0)
                let frac = self
                    .run_event(
                        dex,
                        &ev::FractionalPriority,
                        EvTarget::Poke(pokemon),
                        None,
                        EffectHandle::MoveEff(move_id),
                        Some(RV::Num(0.0)),
                        false,
                        false,
                    )
                    .as_num();
                // targetLoc resolution
                let ms = dex.move_static(move_id);
                let mut target_loc = target_loc;
                if target_loc == 0 {
                    if let Some(t) = self.get_random_target(&ms.target, pokemon) {
                        target_loc = if t.side == pokemon.side { -1 } else { 1 };
                    }
                }
                let original_target = if target_loc == 1 {
                    self.active_id(1 - pokemon.side as usize)
                } else {
                    self.active_id(pokemon.side as usize)
                };
                // getActionSpeed: priority + ModifyPriority events
                let mut priority = ms.priority as f64;
                // singleEvent('ModifyPriority', move...): no gen2 move has it.
                let rv = self.run_event(
                    dex,
                    &ev::ModifyPriority,
                    EvTarget::Poke(pokemon),
                    original_target,
                    EffectHandle::MoveEff(move_id),
                    Some(RV::Num(priority)),
                    false,
                    false,
                );
                priority = rv.as_num();
                let speed = self.get_pokemon_action_speed(dex, pokemon) as f64;
                Some(Action {
                    choice: ActionKind::Move {
                        move_id,
                        target_loc,
                        original_target,
                        source_effect: None,
                    },
                    order: 200,
                    priority: priority + frac,
                    fractional_priority: frac,
                    speed,
                    pokemon: Some(pokemon),
                })
            }
        }
    }
}
