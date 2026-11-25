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
                    if self.log_enabled {
                        let side_id = format!("p{}", side_n + 1);
                        let len_str = len.to_string();
                        self.add(&["teamsize", &side_id, &len_str]);
                    }
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
                let Some(target) = self.get_target(dex, &mv, pokemon, *target_loc) else { return };
                self.before_turn_callback(dex, *move_id, pokemon, target);
            }
            ActionKind::Switch { insta: _, target, source_effect } => {
                let pokemon = action.pokemon.unwrap();
                let se = source_effect.map(EffectHandle::MoveEff);
                let pos = self.poke(pokemon).position;
                if self.switch_in(dex, *target, pos, se, false) == Err(()) {
                    if !self.is_castle() {
                        // pursuitfaint: in gen 2-4 the switch still happens
                        let mut requeued = action.clone();
                        requeued.priority = -101.0;
                        self.queue.insert(0, requeued);
                    }
                    // #4 Castle PBS: Pursuit KO cancels the planned switch so a free
                    // replacement request (CHANGE) opens in after_action.
                    return self.after_action(dex, None, false);
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
                if self.is_castle() {
                    self.castle_residual(dex);
                } else {
                    self.field_event(dex, &ev::Residual, None);
                }
                if !self.ended {
                    self.add(&["upkeep"]);
                }
            }
        }
        let is_run_switch = matches!(action.choice, ActionKind::RunSwitch);
        self.after_action(dex, action.pokemon, is_run_switch)
    }

    /// The tail of runAction shared by all action kinds (phazing, faints,
    /// forced switches, Update events).
    fn after_action(&mut self, dex: &Dex, _pokemon: Option<PokeId>, is_run_switch: bool) {
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

        // #2 Castle PBS: defer faint_messages across simultaneous RunSwitch batches
        // (e.g. dual replacement switches onto Spikes) so 1v1 dual Spikes KO is DRAW.
        let defer_faint = self.is_castle()
            && is_run_switch
            && matches!(self.queue.first().map(|a| &a.choice), Some(ActionKind::RunSwitch));

        // fainting
        if !defer_faint {
            self.faint_messages(dex, false);
            if self.ended {
                return;
            }
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

        let mut switches: [bool; 2] = [0, 1].map(|side_n| {
            self.active_id(side_n)
                .map(|a| self.poke(a).switch_flag.is_set())
                .unwrap_or(false)
        });
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

    #[cold]
    fn unported_before_turn(dex: &Dex, move_id: crate::dex::MoveId) -> ! {
        panic!("unported beforeTurnCallback: {}", dex.moves.key(move_id));
    }

    /// beforeTurnCallback dispatch (pursuit).
    pub fn before_turn_callback(&mut self, dex: &Dex, move_id: crate::dex::MoveId, pokemon: PokeId, target: PokeId) {
        if dex.move_static(move_id).is_pursuit {
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
                vs.set_int(crate::state::DK::TargetLoc, loc);
            }
        } else {
            Self::unported_before_turn(dex, move_id);
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
        let db_cond = crate::cond_id!(dex, "destinybond");
        let ps_cond = crate::cond_id!(dex, "perishsong");
        let mut dbond_caster: Option<u8> = None;
        let mut perish_caster_mask: u8 = 0;
        while !self.faint_queue.is_empty() {
            let entry = self.faint_queue.remove(0);
            let pokemon = entry.target;
            if let Some(EffectHandle::Cond(c)) = entry.effect {
                if Some(c) == db_cond {
                    dbond_caster = entry.source.map(|s| s.side);
                } else if Some(c) == ps_cond {
                    if let Some(src) = entry.source {
                        perish_caster_mask |= 1u8 << (src.side & 1);
                    }
                }
            }
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

        if self.is_castle() {
            // #3 Castle PBS Turn-Cut: any faint skips all remaining queued moves
            // AND voluntary switches (warping straight to replacement + EOT).
            self.queue.retain(|a| {
                !matches!(
                    a.choice,
                    ActionKind::Move { .. }
                        | ActionKind::Switch { insta: false, .. }
                        | ActionKind::BeforeTurnMove { .. }
                )
            });
        } else {
            // gen2: fainting skips moves only
            for active in self.get_all_active(true) {
                self.queue_cancel_move(active);
            }
        }

        let p1_left = self.sides[0].pokemon_left;
        let p2_left = self.sides[1].pokemon_left;
        if p1_left == 0 && p2_left == 0 {
            // Self-KO clause (Explosion / Self-Destruct)
            let p1_last = self.sides[0].last_move.is_some();
            let p2_last = self.sides[1].last_move.is_some();
            if p1_last && !p2_last {
                self.win(Some(1));
                return true;
            } else if p2_last && !p1_last {
                self.win(Some(0));
                return true;
            }
            if let Some(overlay) = self.ruleset.castle_overlay() {
                // #1 Layer 2: 1v1 recoil mutual KO
                if let Some(attacker_side) = self.recoil_ko_side {
                    if overlay.recoil_1v1_attacker_wins {
                        self.win(Some(attacker_side as usize));
                    } else {
                        self.win(None);
                    }
                    return true;
                }
                // Layer 2: 1v1 Perish Song / Destiny Bond mutual KO
                let perish_caster = match perish_caster_mask {
                    0b01 => Some(0u8),
                    0b10 => Some(1u8),
                    _ => None,
                };
                if let Some(caster_side) = dbond_caster.or(perish_caster) {
                    if overlay.perish_dbond_lastmon_forfeit {
                        self.win(Some(1 - caster_side as usize));
                    } else {
                        self.win(None);
                    }
                    return true;
                }
                // #2 Castle PBS: 1v1 Sandstorm / simultaneous Spikes mutual KO -> DRAW
                self.win(None);
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
        // Clear Stadium 2 Self-KO tracker once the current faint batch resolves
        // without ending the battle so an early-game Explosion/Self-Destruct
        // cannot mis-adjudicate a later 1v1 mutual faint (e.g. Sandstorm/Perish).
        self.sides[0].last_move = None;
        self.sides[1].last_move = None;
        self.recoil_ko_side = None;
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
            Some(n) => self.sides[n].name.to_string(),
            None => String::new(),
        });
        self.add(&[""]);
        match side {
            Some(n) => {
                let name = self.sides[n].name;
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
        self.turn = self.turn.saturating_add(1);
        self.last_successful_move_this_turn = None;
        self.triple_kick_mult = 0;
        self.recoil_ko_side = None;
        self.bp_switched_this_turn = [false, false];

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
                    let mut i = self.poke(active).attacked_by.len();
                    while i > 0 {
                        i -= 1;
                        let src = self.poke(active).attacked_by[i].source;
                        let src_active = self.poke(src).is_active;
                        let p = self.poke_mut(active);
                        if src_active {
                            p.attacked_by[i].this_turn = false;
                        } else {
                            p.attacked_by.remove(i);
                        }
                    }
                }

                self.poke_mut(active).trapped = false;
                self.poke_mut(active).maybe_trapped = false;
                self.run_event(dex, &ev::TrapPokemon, EvTarget::Poke(active), None, EffectHandle::None, None, false, false);
                // knownType always true in gen2 → MaybeTrapPokemon if immune
                // to 'trapped'... PS: if (!knownType || getImmunity('trapped'))
                let types = self.poke(active).types;
                if !types.iter().any(|d| dex.status_key_immune("trapped", d)) {
                    self.run_event(dex, &ev::MaybeTrapPokemon, EvTarget::Poke(active), None, EffectHandle::None, None, false, false);
                }

                if !self.poke(active).fainted {
                    self.poke_mut(active).active_turns += 1;
                }
            }
            self.sides[side_n].fainted_last_turn = self.sides[side_n].fainted_this_turn;
            self.sides[side_n].fainted_this_turn = None;
            self.sides[side_n].last_move = None;
        }

        // #8 Castle PBS turn limit (`L_TURN`): judge by alive count -> sum of alive hp/maxhp ratios -> tie.
        if self.is_castle() {
            if let Some(limit) = self.ruleset.turn_limit() {
                if self.turn > limit {
                    self.judge_castle_turn_limit();
                    return;
                }
            }
        } else {
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
                    let msg = format!(
                        "You will auto-tie if the battle doesn't end in {text} (on turn 1000)."
                    );
                    self.add(&["bigerror", &msg]);
                }
                // full staleness tracking not ported (unreachable in fixtures)
            }
        }

        if self.log_enabled {
            let turn_str = self.turn.to_string();
            self.add(&["turn", &turn_str]);
        }

        // gen2 quick claw roll — every endTurn, RNG parity critical
        self.roll_quick_claw();

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
    fn get_requests(&self, kind: RequestState) -> [RequestKind; 2] {
        let mut requests = [RequestKind::Wait, RequestKind::Wait];
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
        let choice = &mut self.sides[side_n].choice;
        choice.cant_undo = false;
        choice.error = false;
        choice.actions.clear();
        choice.forced_switches_left = forced_switches;
        choice.forced_passes_left = forced_passes;
        choice.switch_ins.clear();
    }

    /// #5, #6, #7 Castle PBS end-of-turn residual resolution in strict fixed `1P -> 2P` order:
    /// 1. Future Sight (`1P -> 2P`)
    /// 2. Screens & Safeguard expiry (`1P -> 2P`)
    /// 3. Weather duration / upkeep & Sandstorm damage (`1P -> 2P`)
    /// 4. Perish Song countdown & simultaneous faint (`1P -> 2P`)
    /// 5. Per-side `[Active Item -> Active Freeze Thaw]` (`1P -> 2P`)
    /// 6. Remaining volatile expirations (Encore, Disable, Protect, Endure, Flinch, Pursuit,
    ///    Stall, FuryCutter, Bide, Rollout, TwoTurnMove; skipping PartiallyTrapped / LockOn /
    ///    LockedMove / PerishSong)
    fn castle_residual(&mut self, dex: &Dex) {
        // 1. Future Sight (1P -> 2P)
        if let Some(fm) = crate::cond_id!(dex, "futuremove") {
            for side_n in 0..2u8 {
                let due = self.sides[side_n as usize]
                    .slot_conditions
                    .iter()
                    .find(|(c, _)| *c == fm)
                    .map(|(_, st)| (self.turn as i64 - 1) >= st.get_int(DK::EndingTurn))
                    .unwrap_or(false);
                if due {
                    if let Some(active) = self.active_id(side_n as usize) {
                        self.remove_slot_condition(dex, active, fm);
                        self.faint_messages(dex, false);
                        if self.ended {
                            return;
                        }
                    }
                }
            }
        }

        // 2. Screens & Safeguard expiration (1P -> 2P)
        for side_n in 0..2u8 {
            let side_conds: Vec<(crate::dex::CondId, u32)> = self.sides[side_n as usize]
                .side_conditions
                .iter()
                .filter_map(|(c, st)| st.duration.map(|_| (*c, st.effect_order)))
                .collect();
            for (c, tok) in side_conds {
                let expired = if let Some((_, st)) = self.sides[side_n as usize]
                    .side_conditions
                    .iter_mut()
                    .find(|(k, s)| *k == c && s.effect_order == tok)
                {
                    if let Some(d) = st.duration {
                        if d > 0 {
                            st.duration = Some(d - 1);
                            d - 1 == 0
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if expired {
                    self.remove_side_condition(dex, side_n, c);
                }
            }
        }

        // 3. Weather duration / upkeep & Sandstorm damage (1P -> 2P)
        if let Some(w) = self.field.weather {
            let expired = if let Some(d) = self.field.weather_state.duration {
                if d > 0 {
                    self.field.weather_state.duration = Some(d - 1);
                    d - 1 == 0
                } else {
                    false
                }
            } else {
                false
            };
            if expired {
                self.clear_weather(dex);
            } else {
                match dex.conds_key(w) {
                    "raindance" => self.add(&["-weather", "RainDance", "[upkeep]"]),
                    "sunnyday" => self.add(&["-weather", "SunnyDay", "[upkeep]"]),
                    "sandstorm" => {
                        self.add(&["-weather", "Sandstorm", "[upkeep]"]);
                        let eff = EffectHandle::Cond(w);
                        for side_n in 0..2usize {
                            if let Some(id) = self.active_id(side_n) {
                                if !self.poke(id).fainted && self.poke(id).hp > 0 {
                                    let dmg = self.poke(id).base_maxhp as f64 / 8.0;
                                    self.damage(
                                        dex,
                                        dmg,
                                        Some(id),
                                        None,
                                        super::conditions::DamageEffect::Effect(eff),
                                        false,
                                    );
                                }
                            }
                        }
                        self.faint_messages(dex, false);
                        if self.ended {
                            return;
                        }
                    }
                    _ => {}
                }
            }
        }

        // 4. Perish Song countdown & fainting (1P -> 2P; asymmetric 0-remaining loser queued first)
        if let Some(ps_cond) = crate::cond_id!(dex, "perishsong") {
            let mut perish_faints: Vec<(PokeId, Option<PokeId>)> = Vec::new();
            for side_n in 0..2usize {
                let Some(id) = self.active_id(side_n) else { continue };
                if self.poke(id).fainted || self.poke(id).hp <= 0 {
                    continue;
                }
                let (new_dur, src) = match self.poke_mut(id).volatile_mut(ps_cond) {
                    Some(vs) => {
                        let d = vs.duration.unwrap_or(0);
                        if d > 0 {
                            vs.duration = Some(d - 1);
                            (Some(d - 1), vs.source)
                        } else {
                            (None, vs.source)
                        }
                    }
                    None => (None, None),
                };
                if let Some(d) = new_dur {
                    let ts = self.poke_str(id);
                    let tag = format!("perish{d}");
                    self.add(&["-start", &ts, &tag]);
                    if d == 0 {
                        self.poke_mut(id).volatiles.retain(|(k, _)| *k != ps_cond);
                        self.refresh_poke_mask(dex, id);
                        perish_faints.push((id, src));
                    }
                }
            }
            if !perish_faints.is_empty() {
                // #7 Castle PBS: when both active mons perish simultaneously and only one
                // side has 0 bench remaining, queue the 0-remaining loser first in faint_queue.
                if perish_faints.len() == 2 {
                    let p1_last = self.sides[0].pokemon_left <= 1;
                    let p2_last = self.sides[1].pokemon_left <= 1;
                    if !p1_last && p2_last {
                        perish_faints.swap(0, 1);
                    }
                }
                for (id, src) in perish_faints {
                    self.pokemon_faint(id, src, EffectHandle::Cond(ps_cond));
                }
                self.faint_messages(dex, false);
                if self.ended {
                    return;
                }
            }
        }

        // 5. Per-side [Active Item -> Natural Freeze Thaw] (1P -> 2P)
        let ki = &dex.known_items;
        let conf_cond = crate::cond_id!(dex, "confusion");
        for side_n in 0..2usize {
            let Some(id) = self.active_id(side_n) else { continue };
            if self.poke(id).fainted || self.poke(id).hp <= 0 {
                continue;
            }
            if let Some(item) = self.poke(id).item {
                let it = Some(item);
                if it == ki.leftovers {
                    let amount = self.poke(id).base_maxhp as f64 / 16.0;
                    self.heal(
                        dex,
                        amount,
                        Some(id),
                        None,
                        super::dmg::HealEffect::Effect(EffectHandle::Item(item)),
                    );
                } else if it == ki.berryjuice {
                    if self.poke(id).hp < self.poke(id).maxhp / 2 {
                        let ok = self
                            .run_event(
                                dex,
                                &ev::TryHeal,
                                EvTarget::Poke(id),
                                None,
                                EffectHandle::Item(item),
                                Some(super::RV::Num(20.0)),
                                false,
                                false,
                            )
                            .truthy();
                        if ok && self.use_item(dex, id, None, EffectHandle::Item(item)) {
                            self.heal(
                                dex,
                                20.0,
                                Some(id),
                                None,
                                super::dmg::HealEffect::Effect(EffectHandle::Item(item)),
                            );
                        }
                    }
                } else if it == ki.berry || it == ki.goldberry {
                    if self.poke(id).hp < self.poke(id).maxhp / 2 {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.mintberry {
                    if self.poke(id).status == Status::Slp {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.miracleberry {
                    let has_conf = conf_cond.is_some_and(|c| self.poke(id).has_volatile(c));
                    if self.poke(id).status != Status::None || has_conf {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.iceberry {
                    let has_conf = conf_cond.is_some_and(|c| self.poke(id).has_volatile(c));
                    if self.poke(id).status == Status::Brn || has_conf {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.burntberry {
                    if self.poke(id).status == Status::Frz {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.przcureberry {
                    if self.poke(id).status == Status::Par {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.psncureberry {
                    if matches!(self.poke(id).status, Status::Psn | Status::Tox) {
                        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
                    }
                } else if it == ki.mysteryberry {
                    if let Some(slot_idx) = self.poke(id).move_slots.iter().position(|s| s.pp == 0) {
                        self.castle_eat_mystery_berry_at(dex, id, slot_idx);
                    }
                }
            }
            if !self.poke(id).fainted && self.poke(id).status == Status::Frz {
                if self.prng.random_chance(25, 256) {
                    self.cure_status(dex, id, false);
                }
            }
        }

        // 6. Remaining volatile expirations (1P -> 2P)
        let pt_cond = crate::cond_id!(dex, "partiallytrapped");
        let lo_cond = crate::cond_id!(dex, "lockon");
        let lm_cond = crate::cond_id!(dex, "lockedmove");
        let ps_cond = crate::cond_id!(dex, "perishsong");
        let en_cond = crate::cond_id!(dex, "encore");
        let ro_cond = crate::cond_id!(dex, "rollout");
        for side_n in 0..2usize {
            let Some(id) = self.active_id(side_n) else { continue };
            if self.poke(id).fainted || self.poke(id).hp <= 0 {
                continue;
            }
            let vols: Vec<(crate::dex::CondId, u32)> = self
                .poke(id)
                .volatiles
                .iter()
                .filter_map(|(c, st)| {
                    if Some(*c) == pt_cond
                        || Some(*c) == lo_cond
                        || Some(*c) == lm_cond
                        || Some(*c) == ps_cond
                    {
                        return None;
                    }
                    if st.duration.is_some() || Some(*c) == en_cond || Some(*c) == ro_cond {
                        Some((*c, st.effect_order))
                    } else {
                        None
                    }
                })
                .collect();
            for (c, tok) in vols {
                if self.poke(id).fainted {
                    break;
                }
                let still_same = self
                    .poke(id)
                    .volatile(c)
                    .is_some_and(|st| st.effect_order == tok);
                if !still_same {
                    continue;
                }
                // Rollout: clear silently if Struggle was used
                if Some(c) == ro_cond {
                    let is_struggle = self
                        .poke(id)
                        .last_move
                        .is_some_and(|lm| Some(lm) == dex.known_moves.struggle);
                    if is_struggle {
                        self.poke_mut(id).volatiles.retain(|(k, _)| *k != c);
                        self.refresh_poke_mask(dex, id);
                        continue;
                    }
                }
                let expired = if let Some(st) = self.poke_mut(id).volatile_mut(c) {
                    if let Some(d) = st.duration {
                        if d > 0 {
                            st.duration = Some(d - 1);
                            d - 1 == 0
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };
                if expired {
                    self.remove_volatile_id(dex, id, c);
                    continue;
                }
                // Encore: also ends at EOT if Encored move has 0 PP
                if Some(c) == en_cond {
                    let pp_left = self
                        .poke(id)
                        .volatile(c)
                        .and_then(|s| s.get_move())
                        .and_then(|mid| self.poke(id).get_move_slot(mid).map(|s| s.pp))
                        .unwrap_or(0);
                    if pp_left <= 0 {
                        self.remove_volatile_id(dex, id, c);
                    }
                }
            }
        }
        self.faint_messages(dex, false);
    }

    /// #36 Castle PBS helper: consume Mystery Berry to restore +5 PP (unclamped) to `slot_idx`.
    pub(crate) fn castle_eat_mystery_berry_at(&mut self, dex: &Dex, id: PokeId, slot_idx: usize) {
        let Some(item) = self.poke(id).item else { return };
        if Some(item) != dex.known_items.mysteryberry {
            return;
        }
        self.add_volatile(dex, id, "leppaberry", None, EffectHandle::None);
        if let Some(lb) = crate::cond_id!(dex, "leppaberry") {
            if let Some(vs) = self.poke_mut(id).volatile_mut(lb) {
                vs.slot_ref = Some(slot_idx as u8);
            }
        }
        self.eat_item(dex, id, false, None, EffectHandle::Item(item));
    }

    /// #32 Castle PBS helper: trigger Mint/Miracle/Ice/status berries immediately on `id`
    /// (called for foe right after actor's move and before actor's AfterMoveSelf poison/burn).
    pub(crate) fn castle_check_immediate_status_berry(&mut self, dex: &Dex, id: PokeId) {
        if self.poke(id).fainted || self.poke(id).hp <= 0 {
            return;
        }
        let Some(item) = self.poke(id).item else { return };
        let ki = &dex.known_items;
        let it = Some(item);
        let conf_cond = crate::cond_id!(dex, "confusion");
        let has_conf = conf_cond.is_some_and(|c| self.poke(id).has_volatile(c));
        let st = self.poke(id).status;
        let should_eat = (it == ki.mintberry && st == Status::Slp)
            || (it == ki.miracleberry && (st != Status::None || has_conf))
            || (it == ki.iceberry && (st == Status::Brn || has_conf))
            || (it == ki.burntberry && st == Status::Frz)
            || (it == ki.przcureberry && st == Status::Par)
            || (it == ki.psncureberry && matches!(st, Status::Psn | Status::Tox));
        if should_eat {
            self.eat_item(dex, id, false, None, EffectHandle::Item(item));
        }
    }

    /// #8 Castle PBS `L_TURN` judging: alive count -> sum of alive `hp/maxhp` ratios -> tie.
    fn judge_castle_turn_limit(&mut self) {
        let p1_alive = self.sides[0].pokemon_left;
        let p2_alive = self.sides[1].pokemon_left;
        if p1_alive != p2_alive {
            self.win(Some(if p1_alive > p2_alive { 0 } else { 1 }));
            return;
        }
        let ratio_sum = |side: &Side| -> f64 {
            side.party
                .iter()
                .map(|&slot| &side.roster[slot as usize])
                .filter(|p| !p.fainted && p.hp > 0 && p.maxhp > 0)
                .map(|p| p.hp as f64 / p.maxhp as f64)
                .sum()
        };
        let r1 = ratio_sum(&self.sides[0]);
        let r2 = ratio_sum(&self.sides[1]);
        if (r1 - r2).abs() <= 1e-12 {
            self.win(None);
        } else if r1 > r2 {
            self.win(Some(0));
        } else {
            self.win(Some(1));
        }
    }
}
