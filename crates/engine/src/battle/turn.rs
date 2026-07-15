//! Turn processing: turnLoop / runAction / endTurn / checkFainted +
//! gen2stadium2 faintMessages (self-KO clause) + win + makeRequest.

use crate::dex::Dex;
use crate::state::*;

use super::events::{ev, EvTarget};
use super::EffectHandle;

impl Battle {
    /// battle.turnLoop.
    pub fn turn_loop(&mut self, dex: &Dex) {
        self.add(&[""]);
        // (|t:| timestamp line is stripped from fixtures — not emitted)
        self.request_state = RequestState::None;

        if !self.mid_turn {
            // insertChoice({choice:'beforeTurn'}) + addChoice({choice:'residual'})
            let before_turn = Action {
                choice: ActionKind::BeforeTurn,
                order: 4,
                priority: 0.0,
                fractional_priority: 0.0,
                speed: 1.0,
                pokemon: None,
            };
            self.insert_action_sorted(before_turn);
            self.queue.push(Action {
                choice: ActionKind::Residual,
                order: 300,
                priority: 0.0,
                fractional_priority: 0.0,
                speed: 1.0,
                pokemon: None,
            });
            self.mid_turn = true;
        }

        while !self.queue.is_empty() {
            let action = self.queue.remove(0);
            self.run_action(dex, action);
            if self.request_state != RequestState::None || self.ended {
                return;
            }
        }

        self.end_turn(dex);
        self.mid_turn = false;
        self.queue.clear();
    }

    /// battle.runAction (gen2 singles slice).
    pub fn run_action(&mut self, dex: &Dex, action: Action) {
        match &action.choice {
            ActionKind::Team { index } => {
                let pokemon = action.pokemon.unwrap();
                if *index == 0 {
                    self.sides[pokemon.side as usize].party.clear();
                }
                self.sides[pokemon.side as usize].party.push(pokemon.slot);
                self.poke_mut(pokemon).position = *index;
                // early return: no Update event, no faint checks
                return;
            }
            ActionKind::Start => {
                for side_n in 0..2 {
                    let len = self.sides[side_n].party.len() as i32;
                    if self.sides[side_n].pokemon_left > 0 {
                        self.sides[side_n].pokemon_left = len;
                    }
                    let side_id = format!("p{}", side_n + 1);
                    let len_str = len.to_string();
                    self.add(&["teamsize", &side_id, &len_str]);
                }
                self.add(&["start"]);
                // BattleStart species/format/rule singleEvents: none in NC2000.
                for side_n in 0..2u8 {
                    if self.sides[side_n as usize].pokemon_left > 0 {
                        let first = PokeId { side: side_n, slot: self.sides[side_n as usize].party[0] };
                        let _ = self.switch_in(dex, first, 0, None, false);
                    }
                }
                self.mid_turn = true;
            }
            ActionKind::Move { move_id, target_loc, source_effect, .. } => {
                let pokemon = action.pokemon.unwrap();
                if !self.poke(pokemon).is_active {
                    return;
                }
                if self.poke(pokemon).fainted {
                    return;
                }
                self.run_move(dex, *move_id, pokemon, *target_loc, *source_effect);
            }
            ActionKind::BeforeTurnMove { move_id, target_loc } => {
                let pokemon = action.pokemon.unwrap();
                if !self.poke(pokemon).is_active || self.poke(pokemon).fainted {
                    return;
                }
                let mv = super::moveexec::get_active_move(dex, *move_id);
                let Some(target) = self.get_target(&mv, pokemon, *target_loc) else { return };
                self.before_turn_callback(dex, *move_id, pokemon, target);
            }
            ActionKind::Switch { insta: _, target, source_effect } => {
                let pokemon = action.pokemon.unwrap();
                let se = source_effect.map(EffectHandle::MoveEff);
                let pos = self.poke(pokemon).position;
                if self.switch_in(dex, *target, pos, se, false) == Err(()) {
                    // pursuitfaint: in gen 2-4 the switch still happens
                    let mut requeued = action.clone();
                    requeued.priority = -101.0;
                    self.queue.insert(0, requeued);
                    return self.after_action(dex, None);
                }
            }
            ActionKind::RunSwitch => {
                let pokemon = action.pokemon.unwrap();
                self.run_switch(dex, pokemon);
            }
            ActionKind::BeforeTurn => {
                self.each_event(dex, &ev::BeforeTurn, None);
            }
            ActionKind::Residual => {
                self.add(&[""]);
                self.clear_active_move(true);
                self.update_all_speeds(dex);
                self.field_event(dex, &ev::Residual, None);
                if !self.ended {
                    self.add(&["upkeep"]);
                }
            }
        }
        self.after_action(dex, action.pokemon)
    }

    /// The tail of runAction shared by all action kinds (phazing, faints,
    /// forced switches, Update events).
    fn after_action(&mut self, dex: &Dex, _pokemon: Option<PokeId>) {
        // phazing (Roar etc.)
        for side_n in 0..2u8 {
            if let Some(active) = self.active_id(side_n as usize) {
                if self.poke(active).force_switch_flag {
                    if self.poke(active).hp > 0 {
                        let pos = self.poke(active).position;
                        self.drag_in(dex, side_n, pos);
                    }
                    self.poke_mut(active).force_switch_flag = false;
                }
            }
        }

        self.clear_active_move(false);

        // fainting
        self.faint_messages(dex, false);
        if self.ended {
            return;
        }

        // gen <= 3: switching in fainted pokemon after every move
        let next_choice = self.queue.first().map(|a| a.choice.clone());
        let queue_empty = next_choice.is_none();
        if queue_empty
            || matches!(next_choice, Some(ActionKind::Move { .. }) | Some(ActionKind::Residual))
        {
            self.check_fainted(dex);
        } else if matches!(next_choice, Some(ActionKind::Switch { insta: true, .. })) {
            return;
        }

        let switches: Vec<bool> = (0..2)
            .map(|side_n| {
                self.active_id(side_n)
                    .map(|a| self.poke(a).switch_flag.is_set())
                    .unwrap_or(false)
            })
            .collect();

        let mut switches = switches;
        for side_n in 0..2usize {
            if switches[side_n] && !self.can_switch(side_n as u8) {
                if let Some(active) = self.active_id(side_n) {
                    self.poke_mut(active).switch_flag = SwitchFlag::No;
                }
                switches[side_n] = false;
            } else if switches[side_n] {
                if let Some(active) = self.active_id(side_n) {
                    if self.poke(active).hp > 0
                        && self.poke(active).switch_flag.is_set()
                        && self.poke(active).switch_flag != SwitchFlag::Yes // not revivalblessing
                        && !self.poke(active).skip_before_switch_out
                    {
                        // switchFlag is a move id (selfSwitch) or plain true;
                        // BeforeSwitchOut runs for both (PS checks !== 'revivalblessing').
                    }
                    // PS runs BeforeSwitchOut for hp+switchFlag actives
                    if self.poke(active).hp > 0
                        && self.poke(active).switch_flag.is_set()
                        && !self.poke(active).skip_before_switch_out
                    {
                        self.run_event(dex, &ev::BeforeSwitchOut, EvTarget::Poke(active), None, EffectHandle::None, None, false, false);
                        self.poke_mut(active).skip_before_switch_out = true;
                        self.faint_messages(dex, false);
                        if self.ended {
                            return;
                        }
                        if self.poke(active).fainted {
                            switches[side_n] = self
                                .active_id(side_n)
                                .map(|a| self.poke(a).switch_flag.is_set())
                                .unwrap_or(false);
                        }
                    }
                }
            }
        }

        for &player_switch in &switches {
            if player_switch {
                self.make_request(dex, RequestState::Switch);
                return;
            }
        }

        // gen < 5
        self.each_event(dex, &ev::Update, None);
    }

    /// beforeTurnCallback dispatch (pursuit).
    pub fn before_turn_callback(&mut self, dex: &Dex, move_id: crate::dex::MoveId, pokemon: PokeId, target: PokeId) {
        match dex.moves.key(move_id) {
            "pursuit" => {
                self.add_volatile(
                    dex,
                    pokemon,
                    "pursuit",
                    Some(pokemon),
                    super::EffectHandle::MoveEff(move_id),
                );
                let loc: i64 = if target.side == pokemon.side { -1 } else { 1 };
                let pu = crate::cond_id!(dex, "pursuit").unwrap();
                if let Some(vs) = self.poke_mut(pokemon).volatile_mut(pu) {
                    vs.set_int("targetLoc", loc);
                }
            }
            other => panic!("unported beforeTurnCallback: {other}"),
        }
    }

    /// battle.checkFainted.
    pub fn check_fainted(&mut self, dex: &Dex) {
        for side_n in 0..2usize {
            if let Some(active) = self.active_id(side_n) {
                if self.poke(active).fainted {
                    self.poke_mut(active).status = Status::Fnt;
                    self.poke_mut(active).switch_flag = SwitchFlag::Yes;
                    self.refresh_poke_mask(dex, active);
                }
            }
        }
    }

    /// gen2stadium2 Battle.faintMessages (self-KO clause).
    pub fn faint_messages(&mut self, dex: &Dex, last_first: bool) -> bool {
        if self.ended {
            return false;
        }
        let length = self.faint_queue.len();
        if length == 0 {
            return false;
        }
        if last_first {
            let last = self.faint_queue.pop().unwrap();
            self.faint_queue.insert(0, last);
        }
        let mut faint_data: Option<FaintEntry> = None;
        while !self.faint_queue.is_empty() {
            let entry = self.faint_queue.remove(0);
            let pokemon = entry.target;
            faint_data = Some(entry.clone());
            if !self.poke(pokemon).fainted
                && self
                    .run_event(
                        dex,
                        &ev::BeforeFaint,
                        EvTarget::Poke(pokemon),
                        entry.source,
                        entry.effect.unwrap_or(EffectHandle::None),
                        None,
                        false,
                        false,
                    )
                    .truthy()
            {
                let ps = self.poke_str(pokemon);
                self.add(&["faint", &ps]);
                self.sides[pokemon.side as usize].pokemon_left -= 1;
                if self.sides[pokemon.side as usize].total_fainted < 100 {
                    self.sides[pokemon.side as usize].total_fainted += 1;
                }
                self.run_event(
                    dex,
                    &ev::Faint,
                    EvTarget::Poke(pokemon),
                    entry.source,
                    entry.effect.unwrap_or(EffectHandle::None),
                    None,
                    false,
                    false,
                );
                // singleEvent End ability: none.
                self.clear_volatile(dex, pokemon, false);
                let p = self.poke_mut(pokemon);
                p.fainted = true;
                p.is_active = false;
                p.is_started = false;
                self.sides[pokemon.side as usize].fainted_this_turn = Some(pokemon.slot);
            }
        }

        // gen2: fainting skips moves only
        for active in self.get_all_active(true) {
            self.queue_cancel_move(active);
        }

        let p1_left = self.sides[0].pokemon_left;
        let p2_left = self.sides[1].pokemon_left;
        if p1_left == 0 && p2_left == 0 {
            // Self-KO clause
            let p1_last = self.sides[0].last_move.is_some();
            let p2_last = self.sides[1].last_move.is_some();
            if p1_last && !p2_last {
                self.win(Some(1));
                return true;
            } else if p2_last && !p1_last {
                self.win(Some(0));
                return true;
            }
            let winner = faint_data.as_ref().map(|f| 1 - f.target.side as usize);
            self.win(winner);
            return true;
        }
        if p1_left == 0 {
            self.win(Some(1));
            return true;
        }
        if p2_left == 0 {
            self.win(Some(0));
            return true;
        }
        if let Some(fd) = faint_data {
            self.run_event(
                dex,
                &ev::AfterFaint,
                EvTarget::Poke(fd.target),
                fd.source,
                fd.effect.unwrap_or(EffectHandle::None),
                Some(super::RV::Num(length as f64)),
                false,
                false,
            );
        }
        false
    }

    /// battle.win.
    pub fn win(&mut self, side: Option<usize>) -> bool {
        if self.ended {
            return false;
        }
        self.winner = Some(match side {
            Some(n) => self.sides[n].name.clone(),
            None => String::new(),
        });
        self.add(&[""]);
        match side {
            Some(n) => {
                let name = self.sides[n].name.clone();
                self.add(&["win", &name]);
            }
            None => self.add(&["tie"]),
        }
        self.ended = true;
        self.request_state = RequestState::None;
        for side in &mut self.sides {
            side.request = None;
        }
        true
    }

    /// battle.endTurn.
    pub fn end_turn(&mut self, dex: &Dex) {
        self.turn += 1;
        self.last_successful_move_this_turn = None;

        for side_n in 0..2usize {
            if let Some(active) = self.active_id(side_n) {
                {
                    let turn = self.turn;
                    let p = self.poke_mut(active);
                    p.move_this_turn = None;
                    p.newly_switched = false;
                    p.move_last_turn_result = p.move_this_turn_result;
                    p.move_this_turn_result = MoveResult::Undef;
                    if turn != 1 {
                        p.used_item_this_turn = false;
                        p.stats_raised_this_turn = false;
                        p.stats_lowered_this_turn = false;
                        p.hurt_this_turn = None;
                    }
                    for slot in &mut p.move_slots {
                        slot.disabled = false;
                        if slot.shared {
                            // shared objects: base slot too
                        }
                    }
                    for slot in &mut p.base_move_slots {
                        slot.disabled = false;
                    }
                }
                self.run_event(dex, &ev::DisableMove, EvTarget::Poke(active), None, EffectHandle::None, None, false, false);
                // per-slot singleEvent('DisableMove', activeMove): no gen2 M1
                // move carries onDisableMove.

                // attackedBy pruning
                {
                    // PS iterates backwards, marking thisTurn=false for active
                    // sources and removing gone ones.
                    let sources: Vec<(usize, bool)> = self
                        .poke(active)
                        .attacked_by
                        .iter()
                        .enumerate()
                        .map(|(i, a)| (i, self.poke(a.source).is_active))
                        .collect();
                    let p = self.poke_mut(active);
                    let mut remove: Vec<usize> = Vec::new();
                    for (i, src_active) in sources.into_iter().rev() {
                        if src_active {
                            p.attacked_by[i].this_turn = false;
                        } else {
                            remove.push(i);
                        }
                    }
                    for i in remove {
                        p.attacked_by.remove(i);
                    }
                }

                self.poke_mut(active).trapped = false;
                self.poke_mut(active).maybe_trapped = false;
                self.run_event(dex, &ev::TrapPokemon, EvTarget::Poke(active), None, EffectHandle::None, None, false, false);
                // knownType always true in gen2 → MaybeTrapPokemon if immune
                // to 'trapped'... PS: if (!knownType || getImmunity('trapped'))
                let types = self.poke(active).types.clone();
                if dex.get_immunity("trapped", &types) {
                    self.run_event(dex, &ev::MaybeTrapPokemon, EvTarget::Poke(active), None, EffectHandle::None, None, false, false);
                }

                if !self.poke(active).fainted {
                    self.poke_mut(active).active_turns += 1;
                }
            }
            self.sides[side_n].fainted_last_turn = self.sides[side_n].fainted_this_turn;
            self.sides[side_n].fainted_this_turn = None;
        }

        // maybeTriggerEndlessBattleClause: turn limits only matter past 100
        if self.turn > 1000 {
            self.add(&["message", "It is turn 1000. You have hit the turn limit!"]);
            self.win(None);
            return;
        }
        if self.turn > 100 {
            if (self.turn >= 500 && self.turn % 100 == 0)
                || (self.turn >= 900 && self.turn % 10 == 0)
                || self.turn >= 990
            {
                let left = 1000 - self.turn;
                let text = if left == 1 { "1 turn".to_string() } else { format!("{left} turns") };
                let msg =
                    format!("You will auto-tie if the battle doesn't end in {text} (on turn 1000).");
                self.add(&["bigerror", &msg]);
            }
            // full staleness tracking not ported (unreachable in fixtures)
        }

        let turn_str = self.turn.to_string();
        self.add(&["turn", &turn_str]);

        // gen2 quick claw roll — every endTurn, RNG parity critical
        self.quick_claw_roll = self.prng.random_chance(60, 256);

        self.make_request(dex, RequestState::Move);
    }

    /// battle.makeRequest.
    pub fn make_request(&mut self, dex: &Dex, kind: RequestState) {
        self.request_state = kind;
        for side_n in 0..2usize {
            self.clear_choice(side_n);
            self.sides[side_n].request = None;
        }
        if kind == RequestState::TeamPreview {
            self.add(&["teampreview", "3"]);
        }
        let requests = self.get_requests(kind);
        for (side_n, req) in requests.into_iter().enumerate() {
            self.sides[side_n].request = Some(req);
        }
        let _ = dex;
    }

    /// battle.getRequests.
    fn get_requests(&self, kind: RequestState) -> Vec<RequestKind> {
        let mut requests = vec![RequestKind::Wait, RequestKind::Wait];
        match kind {
            RequestState::Switch => {
                for side_n in 0..2usize {
                    if self.sides[side_n].pokemon_left == 0 {
                        continue;
                    }
                    let needs_switch = self
                        .active_id(side_n)
                        .map(|a| self.poke(a).switch_flag.is_set())
                        .unwrap_or(false);
                    if needs_switch {
                        requests[side_n] = RequestKind::Switch;
                    }
                }
            }
            RequestState::TeamPreview => {
                for req in requests.iter_mut() {
                    *req = RequestKind::TeamPreview;
                }
            }
            _ => {
                for side_n in 0..2usize {
                    if self.sides[side_n].pokemon_left == 0 {
                        continue;
                    }
                    requests[side_n] = RequestKind::Move;
                }
            }
        }
        requests
    }

    /// side.clearChoice.
    pub fn clear_choice(&mut self, side_n: usize) {
        let mut forced_switches = 0;
        let mut forced_passes = 0;
        if self.request_state == RequestState::Switch {
            let can_switch_out = self
                .active_id(side_n)
                .map(|a| self.poke(a).switch_flag.is_set() as u32)
                .unwrap_or(0);
            let can_switch_in = self.sides[side_n]
                .party
                .iter()
                .skip(1)
                .filter(|&&slot| !self.sides[side_n].roster[slot as usize].fainted)
                .count() as u32;
            forced_switches = can_switch_out.min(can_switch_in);
            forced_passes = can_switch_out - forced_switches;
        }
        self.sides[side_n].choice = Choice {
            cant_undo: false,
            error: false,
            actions: Vec::new(),
            forced_switches_left: forced_switches,
            forced_passes_left: forced_passes,
            switch_ins: Vec::new(),
        };
    }
}
