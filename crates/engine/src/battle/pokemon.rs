//! Pokemon-level methods (sim/pokemon.ts + gen2/gen2stadium2 overrides),
//! implemented on `Battle` because everything needs battle context.

use crate::dex::{Dex, MoveId};
use crate::state::*;

use super::events::{ev, EvTarget};
use super::{clamp_int_range, EffectHandle, RV};

pub const STAT_KEYS: [&str; 5] = ["atk", "def", "spa", "spd", "spe"];

impl Battle {
    // ------------------------------------------------------------- stats

    /// gen2stadium2 `pokemon.getStat(statName, unboosted, unmodified, fastReturn)`.
    /// `stat` is an index into stored_stats (0=atk..4=spe).
    pub fn get_stat(&self, dex: &Dex, id: PokeId, stat: usize, unboosted: bool, unmodified: bool, fast_return: bool) -> i32 {
        let p = self.poke(id);
        let mut value = p.stored_stats[stat] as f64;

        if !unboosted {
            let mut boost = p.boosts[stat] as i32;
            boost = boost.clamp(-6, 6);
            if boost >= 0 {
                const BOOST_TABLE: [f64; 7] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
                value = (value * BOOST_TABLE[boost as usize]).floor();
            } else {
                const NUMERATORS: [f64; 7] = [100.0, 66.0, 50.0, 40.0, 33.0, 28.0, 25.0];
                value = (value * NUMERATORS[(-boost) as usize] / 100.0).floor();
            }
        }
        let parspeeddrop = crate::cond_id!(dex, "parspeeddrop").map(|c| p.has_volatile(c)).unwrap_or(false);
        if p.status == Status::Par && stat == 4 && parspeeddrop {
            value = (value / 4.0).floor();
        }
        if !unmodified {
            let brnattackdrop =
                crate::cond_id!(dex, "brnattackdrop").map(|c| p.has_volatile(c)).unwrap_or(false);
            if p.status == Status::Brn && stat == 0 && brnattackdrop {
                value = (value / 2.0).floor();
            }
        }

        let mut value = clamp_int_range(value, Some(1.0), Some(999.0));
        if fast_return {
            return value as i32;
        }

        // Screens
        if !unboosted {
            let side = &self.sides[id.side as usize];
            let reflect = crate::cond_id!(dex, "reflect").map(|c| side.has_side_condition(c)).unwrap_or(false);
            let lightscreen =
                crate::cond_id!(dex, "lightscreen").map(|c| side.has_side_condition(c)).unwrap_or(false);
            if (stat == 1 && reflect) || (stat == 3 && lightscreen) {
                value *= 2.0;
            }
        }

        // Boosting items (thickclub/lightball/metalpowder). PS compares
        // `species.name` (the CURRENT species, transform included) — the
        // interned id compare is equivalent.
        if let Some(item) = p.item {
            let item = Some(item);
            let species = Some(p.species);
            let ki = &dex.known_items;
            let ks = &dex.known_species;
            if (item == ki.thickclub
                && (species == ks.cubone || species == ks.marowak)
                && stat == 0)
                || (item == ki.lightball && species == ks.pikachu && stat == 2)
            {
                value *= 2.0;
            } else if item == ki.metalpowder && species == ks.ditto && (stat == 1 || stat == 3) {
                value = (value * 1.5).floor();
            }
        }

        value as i32
    }

    /// gen3 `pokemon.getActionSpeed()`.
    pub fn get_pokemon_action_speed(&self, dex: &Dex, id: PokeId) -> i32 {
        let speed = self.get_stat(dex, id, 4, false, false, false);
        if self.quick_claw_roll
            && self.poke(id).item.is_some()
            && self.poke(id).item == dex.known_items.quickclaw
        {
            return 65535;
        }
        speed
    }

    pub fn update_speed(&mut self, dex: &Dex, id: PokeId) {
        self.poke_mut(id).speed = self.get_pokemon_action_speed(dex, id);
    }

    pub fn update_all_speeds(&mut self, dex: &Dex) {
        for side in 0..2 {
            if let Some(id) = self.active_id(side) {
                if !self.poke(id).fainted {
                    self.update_speed(dex, id);
                }
            }
        }
    }

    /// gen2 `pokemon.boostBy(boost)` — returns the last delta applied.
    pub fn pokemon_boost_by(&mut self, dex: &Dex, id: PokeId, boosts: &[(usize, i8)]) -> i8 {
        let mut delta: i8 = 0;
        for &(stat, amount) in boosts {
            delta = amount;
            if amount > 0 && stat < 5 && self.get_stat(dex, id, stat, false, true, false) == 999 {
                delta = 0;
                continue;
            }
            let p = self.poke_mut(id);
            p.boosts[stat] = p.boosts[stat].saturating_add(delta);
            if p.boosts[stat] > 6 {
                delta -= p.boosts[stat] - 6;
                p.boosts[stat] = 6;
            }
            if p.boosts[stat] < -6 {
                delta -= p.boosts[stat] + 6;
                p.boosts[stat] = -6;
            }
        }
        delta
    }

    // ------------------------------------------------------------ status

    /// pokemon.trySetStatus.
    pub fn try_set_status(
        &mut self,
        dex: &Dex,
        id: PokeId,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> RV {
        let current = self.poke(id).status;
        let status_to_set = if current != Status::None { current.as_str().to_string() } else { status.to_string() };
        self.set_status(dex, id, &status_to_set, source, source_effect, false)
    }

    /// pokemon.setStatus.
    pub fn set_status(
        &mut self,
        dex: &Dex,
        id: PokeId,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        ignore_immunities: bool,
    ) -> RV {
        if self.poke(id).hp <= 0 {
            return RV::False;
        }
        let mut source = source;
        let mut source_effect = source_effect;
        if let Some(frame) = self.event_stack.last() {
            if source.is_none() {
                source = frame.source;
            }
        }
        if source_effect.is_none() {
            source_effect = self.current_effect();
        }
        if source.is_none() {
            source = Some(id);
        }

        // sourceEffect move `status` field checks (fail messages)
        let se_status: Option<String> = match source_effect {
            EffectHandle::MoveEff(m) => dex.move_static(m).status.clone(),
            _ => None,
        };

        if self.poke(id).status.as_str() == status {
            if se_status.as_deref() == Some(status) {
                let ts = self.poke_str(id);
                b_add_fail_status(self, &ts, status);
            } else if se_status.is_some() {
                let ss = self.poke_str(source.unwrap());
                self.add(&["-fail", &ss]);
                self.attr_last_move(&["[still]"]);
            }
            return RV::False;
        }

        if !ignore_immunities && !status.is_empty() {
            let check = if status == "tox" { "psn" } else { status };
            if !self.run_status_immunity(dex, id, check, false) {
                if se_status.is_some() {
                    let ts = self.poke_str(id);
                    self.add(&["-immune", &ts]);
                }
                return RV::False;
            }
        }

        let prev_status = self.poke(id).status;
        let prev_state = self.poke(id).status_state.clone();

        if !status.is_empty() {
            let result = self.run_event(
                dex,
                &ev::SetStatus,
                EvTarget::Poke(id),
                source,
                source_effect,
                Some(RV::Str(status.to_string())),
                false,
                false,
            );
            if !result.truthy() {
                return result;
            }
        }

        self.poke_mut(id).status = Status::from_str(status);
        self.refresh_poke_mask(dex, id);
        let mut state = EffectState { id: status.to_string(), ..Default::default() };
        state.source = source;
        let cond = if status.is_empty() { None } else { dex.conds_id(status) };
        if let Some(c) = cond {
            if let Some(d) = dex.cond(c).duration {
                state.duration = Some(d);
            }
        }
        let target_active = self.poke(id).is_active;
        let state = self.init_effect_state(state, target_active && !status.is_empty());
        self.poke_mut(id).status_state = state;
        // (statuses in gen2 have no durationCallback)

        if !status.is_empty() {
            let c = cond.expect("status condition must exist");
            let started = self.single_event(
                dex,
                &ev::Start,
                EffectHandle::Cond(c),
                StateLoc::Status(id),
                EvTarget::Poke(id),
                source,
                source_effect,
                None,
            );
            if !started.truthy() {
                self.poke_mut(id).status = prev_status;
                self.poke_mut(id).status_state = prev_state;
                self.refresh_poke_mask(dex, id);
                return RV::False;
            }
            let after = self.run_event(
                dex,
                &ev::AfterSetStatus,
                EvTarget::Poke(id),
                source,
                source_effect,
                Some(RV::Str(status.to_string())),
                false,
                false,
            );
            if !after.truthy() {
                return RV::False;
            }
        }
        RV::True
    }

    /// pokemon.cureStatus (with message).
    pub fn cure_status(&mut self, dex: &Dex, id: PokeId, silent: bool) -> bool {
        let p = self.poke(id);
        if p.hp <= 0 || p.status == Status::None {
            return false;
        }
        let ts = self.poke_str(id);
        let status = self.poke(id).status.as_str().to_string();
        self.add(&["-curestatus", &ts, &status, if silent { "[silent]" } else { "[msg]" }]);
        if self.poke(id).status == Status::Slp && self.remove_volatile(dex, id, "nightmare") {
            let ts = self.poke_str(id);
            self.add(&["-end", &ts, "Nightmare", "[silent]"]);
        }
        self.set_status(dex, id, "", None, EffectHandle::None, false);
        true
    }

    /// pokemon.clearStatus (no message).
    pub fn pokemon_clear_status(&mut self, dex: &Dex, id: PokeId) -> bool {
        let p = self.poke(id);
        if p.hp <= 0 || p.status == Status::None {
            return false;
        }
        if self.poke(id).status == Status::Slp && self.remove_volatile(dex, id, "nightmare") {
            let ts = self.poke_str(id);
            self.add(&["-end", &ts, "Nightmare", "[silent]"]);
        }
        self.set_status(dex, id, "", None, EffectHandle::None, false);
        true
    }

    // --------------------------------------------------------- volatiles

    /// pokemon.addVolatile.
    pub fn add_volatile(
        &mut self,
        dex: &Dex,
        id: PokeId,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> RV {
        let cond = dex
            .conds_id(status)
            .unwrap_or_else(|| panic!("addVolatile of unknown condition {status}"));
        if self.poke(id).hp <= 0 {
            return RV::False;
        }
        let mut source = source;
        let mut source_effect = source_effect;
        if !self.event_stack.is_empty() {
            if source.is_none() {
                source = self.event_stack.last().unwrap().source;
            }
            if source_effect.is_none() {
                source_effect = self.current_effect();
            }
        }
        if source.is_none() {
            source = Some(id);
        }

        if self.poke(id).has_volatile(cond) {
            if !dex.cond(cond).has_callback("onRestart") {
                return RV::False;
            }
            return self.single_event(
                dex,
                &ev::Restart,
                EffectHandle::Cond(cond),
                StateLoc::Volatile(id, cond),
                EvTarget::Poke(id),
                source,
                source_effect,
                None,
            );
        }
        if !self.run_status_immunity(dex, id, status, false) {
            let se_status = match source_effect {
                EffectHandle::MoveEff(m) => dex.move_static(m).status.is_some(),
                _ => false,
            };
            if se_status {
                let ts = self.poke_str(id);
                self.add(&["-immune", &ts]);
            }
            return RV::False;
        }
        let result = self.run_event(
            dex,
            &ev::TryAddVolatile,
            EvTarget::Poke(id),
            source,
            source_effect,
            Some(RV::Str(status.to_string())),
            false,
            false,
        );
        if !result.truthy() {
            return result;
        }

        let mut state = EffectState {
            id: status.to_string(),
            name: Some(dex.cond_display_name(cond).to_string()),
            ..Default::default()
        };
        if let Some(src) = source {
            state.source = Some(src);
            state.source_slot = Some(self.slot_str(src));
        }
        if !source_effect.is_none() {
            state.source_effect = Some(self.effect_id(dex, source_effect).to_string());
        }
        if let Some(d) = dex.cond(cond).duration {
            state.duration = Some(d);
        }
        let target_active = self.poke(id).is_active;
        let state = self.init_effect_state(state, target_active);
        self.poke_mut(id).volatiles.push((cond, state));
        self.refresh_poke_mask(dex, id);
        if dex.cond(cond).has_callback("durationCallback") {
            let dur = super::conditions::duration_callback(self, dex, status, Some(id), source, source_effect);
            if let Some(d) = dur {
                if let Some(vs) = self.poke_mut(id).volatile_mut(cond) {
                    vs.duration = Some(d);
                }
            }
        }
        let started = self.single_event(
            dex,
            &ev::Start,
            EffectHandle::Cond(cond),
            StateLoc::Volatile(id, cond),
            EvTarget::Poke(id),
            source,
            source_effect,
            None,
        );
        if !started.truthy() {
            self.poke_mut(id).volatiles.retain(|(k, _)| *k != cond);
            self.refresh_poke_mask(dex, id);
            return started;
        }
        RV::True
    }

    /// pokemon.addVolatile with a linkedStatus (meanlook/spiderweb 'trapper').
    pub fn add_volatile_linked(
        &mut self,
        dex: &Dex,
        id: PokeId,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        linked_status: &str,
    ) -> RV {
        if let Some(src) = source {
            if self.poke(src).hp <= 0 {
                return RV::False;
            }
        }
        let result = self.add_volatile(dex, id, status, source, source_effect);
        if result != RV::True {
            return result;
        }
        let Some(src) = source else { return result };
        let cond = dex.conds_id(status).unwrap();
        let linked_cond = dex.conds_id(linked_status).unwrap();
        if !self.poke(src).has_volatile(linked_cond) {
            self.add_volatile(dex, src, linked_status, Some(id), source_effect);
            if let Some(vs) = self.poke_mut(src).volatile_mut(linked_cond) {
                vs.linked_pokemon = vec![id];
                // PS stores the Condition OBJECT here — essence-invisible.
                vs.linked_status = Some(status.to_string());
            }
        } else if let Some(vs) = self.poke_mut(src).volatile_mut(linked_cond) {
            vs.linked_pokemon.push(id);
        }
        if let Some(vs) = self.poke_mut(id).volatile_mut(cond) {
            vs.linked_pokemon = vec![src];
            vs.linked_status = Some(linked_status.to_string());
            // PS stores the STRING here — it lands in the essence.
            vs.set(
                "linkedStatus",
                crate::state::Scalar::Str(linked_status.to_string()),
            );
        }
        result
    }

    /// pokemon.removeVolatile.
    pub fn remove_volatile(&mut self, dex: &Dex, id: PokeId, status: &str) -> bool {
        let Some(cond) = dex.conds_id(status) else { return false };
        self.remove_volatile_id(dex, id, cond)
    }

    pub fn remove_volatile_id(&mut self, dex: &Dex, id: PokeId, cond: crate::dex::CondId) -> bool {
        if self.poke(id).hp <= 0 {
            return false;
        }
        if !self.poke(id).has_volatile(cond) {
            return false;
        }
        let (linked_pokemon, linked_status) = {
            let vs = self.poke(id).volatile(cond).unwrap();
            (vs.linked_pokemon.clone(), vs.linked_status.clone())
        };
        self.single_event(
            dex,
            &ev::End,
            EffectHandle::Cond(cond),
            StateLoc::Volatile(id, cond),
            EvTarget::Poke(id),
            None,
            EffectHandle::None,
            None,
        );
        self.poke_mut(id).volatiles.retain(|(k, _)| *k != cond);
        self.refresh_poke_mask(dex, id);
        if !linked_pokemon.is_empty() {
            if let Some(ls) = linked_status {
                self.remove_linked_volatiles(dex, id, &ls, &linked_pokemon);
            }
        }
        true
    }

    /// pokemon.removeLinkedVolatiles(linkedStatus, linkedPokemon), self = `id`.
    fn remove_linked_volatiles(
        &mut self,
        dex: &Dex,
        id: PokeId,
        linked_status: &str,
        linked_pokemon: &[PokeId],
    ) {
        let Some(cond) = dex.conds_id(linked_status) else { return };
        for &linked in linked_pokemon {
            let Some(vs) = self.poke_mut(linked).volatile_mut(cond) else { continue };
            vs.linked_pokemon.retain(|&p| p != id);
            if vs.linked_pokemon.is_empty() {
                self.remove_volatile_id(dex, linked, cond);
            }
        }
    }

    pub fn try_trap(&mut self, dex: &Dex, id: PokeId) -> bool {
        // runStatusImmunity('trapped') without message
        // (dex lookup needs dex; trapped immunity: Ghost)
        // NOTE: callers pass through run_status_immunity when needed; PS
        // tryTrap checks it — we do too via types directly.
        let p = self.poke(id);
        let ghost = p.types.has(dex.known_types.ghost);
        if ghost {
            return false;
        }
        if p.fainted {
            return false;
        }
        self.poke_mut(id).trapped = true;
        true
    }

    // --------------------------------------------------------- hp change

    /// pokemon.damage (the low-level one).
    pub fn pokemon_damage(
        &mut self,
        id: PokeId,
        d: f64,
        source: Option<PokeId>,
        effect: EffectHandle,
    ) -> f64 {
        let p = self.poke(id);
        if p.hp <= 0 || d.is_nan() || d <= 0.0 {
            return 0.0;
        }
        let mut d = d;
        if d < 1.0 && d > 0.0 {
            d = 1.0;
        }
        let mut d = super::tr(d);
        self.poke_mut(id).hp -= d as i32;
        if self.poke(id).hp <= 0 {
            d += self.poke(id).hp as f64;
            self.pokemon_faint(id, source, effect);
        }
        d
    }

    /// pokemon.heal (low-level). Returns healed amount or None (false).
    pub fn pokemon_heal(&mut self, id: PokeId, d: f64) -> Option<f64> {
        let p = self.poke(id);
        if p.hp <= 0 {
            return None;
        }
        let d = super::tr(d);
        if d <= 0.0 {
            return None;
        }
        if p.hp >= p.maxhp {
            return None;
        }
        let mut d = d;
        let p = self.poke_mut(id);
        p.hp += d as i32;
        if p.hp > p.maxhp {
            d -= (p.hp - p.maxhp) as f64;
            p.hp = p.maxhp;
        }
        Some(d)
    }

    /// pokemon.faint: queue only.
    pub fn pokemon_faint(&mut self, id: PokeId, source: Option<PokeId>, effect: EffectHandle) -> f64 {
        let p = self.poke(id);
        if p.fainted || p.faint_queued {
            return 0.0;
        }
        let d = p.hp as f64;
        let p = self.poke_mut(id);
        p.hp = 0;
        p.switch_flag = SwitchFlag::No;
        p.faint_queued = true;
        self.faint_queue.push(FaintEntry {
            target: id,
            source,
            effect: if effect.is_none() { None } else { Some(effect) },
        });
        d
    }

    // --------------------------------------------------------- pp / moves

    /// pokemon.deductPP. Writes mirror into base_move_slots for shared slots
    /// (PS shares the slot objects).
    pub fn deduct_pp(&mut self, id: PokeId, move_id: MoveId, amount: i32) -> i32 {
        let Some(slot) = self.poke_mut(id).get_move_slot_mut(move_id) else { return 0 };
        slot.used = true;
        let mut deducted = 0;
        if slot.pp > 0 {
            deducted = amount;
            slot.pp -= amount;
            if slot.pp < 0 {
                deducted += slot.pp;
                slot.pp = 0;
            }
        }
        let (pp, used, shared) = (slot.pp, slot.used, slot.shared);
        if shared {
            if let Some(base) = self.poke_mut(id).base_move_slots.iter_mut().find(|m| m.id == move_id) {
                base.pp = pp;
                base.used = used;
            }
        }
        deducted
    }

    /// pokemon.moveUsed (gen2 rules).
    pub fn move_used(&mut self, dex: &Dex, id: PokeId, move_id: MoveId, target_loc: Option<i8>) {
        let key = dex.moves.key(move_id);
        let p = self.poke_mut(id);
        if matches!(key, "metronome" | "mimic" | "mirrormove" | "sketch" | "sleeptalk" | "transform") {
            p.last_move = None;
            p.last_move_encore = None;
        } else {
            p.last_move = Some(move_id);
            p.last_move_encore = Some(move_id);
        }
        p.last_move_target_loc = target_loc;
        p.move_this_turn = Some(move_id);
    }

    /// pokemon.gotAttacked.
    pub fn got_attacked(&mut self, id: PokeId, move_id: Option<MoveId>, damage: Option<f64>, source: PokeId) {
        let damage_number = damage.unwrap_or(0.0) as i64;
        self.poke_mut(id).attacked_by.push(Attacker {
            source,
            damage: damage_number,
            move_id: move_id.unwrap_or(MoveId(u16::MAX)),
            this_turn: true,
            damage_value: damage.map(|d| d as i64),
        });
        self.poke_mut(id).times_attacked += 1;
    }

    /// pokemon.getLockedMove: priorityEvent('LockMove').
    pub fn get_locked_move(&mut self, dex: &Dex, id: PokeId) -> Option<String> {
        let rv = self.priority_event(dex, &ev::LockMove, EvTarget::Poke(id), None, EffectHandle::None, None);
        match rv {
            RV::Str(s) => Some(s),
            _ => None,
        }
    }

    /// pokemon.disableMove.
    pub fn pokemon_disable_move(&mut self, id: PokeId, move_id: MoveId) {
        let mut shared = false;
        if let Some(slot) = self.poke_mut(id).get_move_slot_mut(move_id) {
            if !slot.disabled {
                slot.disabled = true;
                shared = slot.shared;
            }
        }
        if shared {
            if let Some(base) = self.poke_mut(id).base_move_slots.iter_mut().find(|m| m.id == move_id) {
                base.disabled = true;
            }
        }
    }

    /// pokemon.copyVolatileFrom (Baton Pass): boosts + non-noCopy volatiles.
    pub fn copy_volatile_from(&mut self, dex: &Dex, id: PokeId, from: PokeId) {
        // this.clearVolatile() — incoming bench mon is already clean.
        self.poke_mut(id).boosts = self.poke(from).boosts;
        let copied: Vec<(crate::dex::CondId, EffectState)> = self
            .poke(from)
            .volatiles
            .iter()
            .filter(|(c, _)| !dex.cond(*c).no_copy)
            .map(|(c, s)| (*c, s.clone()))
            .collect();
        for (c, state) in copied {
            // linked volatiles: re-point the partners' links at the new mon,
            // and strip the links from the passer so its clearVolatile()
            // doesn't tear them down.
            if !state.linked_pokemon.is_empty() {
                if let Some(vs) = self.poke_mut(from).volatile_mut(c) {
                    vs.linked_pokemon.clear();
                    vs.linked_status = None;
                }
                if let Some(ls) = &state.linked_status {
                    if let Some(lc) = dex.conds_id(ls) {
                        for &linked in &state.linked_pokemon {
                            if let Some(lv) = self.poke_mut(linked).volatile_mut(lc) {
                                for p in lv.linked_pokemon.iter_mut() {
                                    if *p == from {
                                        *p = id;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            self.poke_mut(id).volatiles.push((c, state));
        }
        self.refresh_poke_mask(dex, id);
        self.clear_volatile(dex, from, true);
        // singleEvent('Copy') per volatile: no gen2 condition has onCopy.
    }

    // ------------------------------------------------------- clearVolatile

    /// pokemon.clearVolatile(includeSwitchFlags).
    pub fn clear_volatile(&mut self, dex: &Dex, id: PokeId, include_switch_flags: bool) {
        // linked volatiles first (PS iterates this.volatiles)
        let linked: Vec<(String, Vec<PokeId>)> = self
            .poke(id)
            .volatiles
            .iter()
            .filter_map(|(_, vs)| {
                vs.linked_status
                    .as_ref()
                    .map(|ls| (ls.clone(), vs.linked_pokemon.clone()))
            })
            .collect();
        for (ls, lp) in linked {
            self.remove_linked_volatiles(dex, id, &ls, &lp);
        }
        let p = self.poke_mut(id);
        p.boosts = [0; 7];
        p.move_slots = p.base_move_slots.clone();
        p.transformed = false;
        p.hp_type = p.base_hp_type.clone();
        p.hp_power = p.base_hp_power;
        p.volatiles.clear();
        if include_switch_flags {
            p.switch_flag = SwitchFlag::No;
            p.force_switch_flag = false;
        }
        p.last_move = None;
        p.last_move_encore = None;
        p.last_move_used = None;
        p.move_this_turn = None;
        p.move_last_turn_result = MoveResult::Undef;
        p.move_this_turn_result = MoveResult::Undef;
        p.last_damage = 0;
        p.attacked_by.clear();
        p.hurt_this_turn = None;
        p.newly_switched = true;
        p.being_called_back = false;
        // setSpecies(baseSpecies): restores species/types/stats (transform).
        p.species = p.base_species;
        p.stored_stats = p.base_stored_stats;
        p.types = dex.species_types(p.base_species);
        p.speed = p.stored_stats[4];
        self.refresh_poke_mask(dex, id);
    }

    // ---------------------------------------------------------- immunity

    /// pokemon.runStatusImmunity(type, message?).
    pub fn run_status_immunity(&mut self, dex: &Dex, id: PokeId, ty: &str, message: bool) -> bool {
        if self.poke(id).fainted {
            return false;
        }
        if ty.is_empty() {
            return true;
        }
        let types = self.poke(id).types;
        let immune = match dex.type_id(ty) {
            Some(att) => types.iter().any(|d| dex.type_immune(att, d)),
            None => types.iter().any(|d| dex.status_key_immune(ty, d)),
        };
        if immune {
            if message {
                let ts = self.poke_str(id);
                self.add(&["-immune", &ts]);
            }
            return false;
        }
        let immunity = self.run_event(
            dex,
            &ev::Immunity,
            EvTarget::Poke(id),
            None,
            EffectHandle::None,
            Some(RV::Str(ty.to_string())),
            false,
            false,
        );
        if !immunity.truthy() {
            if message && immunity != RV::Null {
                let ts = self.poke_str(id);
                self.add(&["-immune", &ts]);
            }
            return false;
        }
        true
    }

    /// pokemon.runImmunity(move, message) — false = immune.
    pub fn run_move_immunity(&mut self, dex: &Dex, id: PokeId, message: bool) -> bool {
        let (ty, ignore) = {
            let m = self.active_move.as_ref().expect("runImmunity without active move");
            (m.move_type, m.ignore_immunity)
        };
        if ignore {
            return true;
        }
        if ty == dex.known_types.unknown {
            return true;
        }
        let negate = !self
            .run_event(
                dex,
                &ev::NegateImmunity,
                EvTarget::Poke(id),
                None,
                EffectHandle::None,
                Some(RV::Num(ty.0 as f64)),
                false,
                false,
            )
            .truthy();
        let not_immune = if ty == dex.known_types.ground {
            // gen2 isGrounded: Flying-types are airborne, nothing else.
            let flying = self.poke(id).has_type(dex.known_types.flying);
            if negate {
                true
            } else {
                !flying
            }
        } else {
            let types = self.poke(id).types;
            negate || !types.iter().any(|d| dex.type_immune(ty, d))
        };
        if not_immune {
            return true;
        }
        if message {
            let ts = self.poke_str(id);
            self.add(&["-immune", &ts]);
        }
        false
    }

    /// pokemon.runImmunity(type-string) — false = immune (jumpkick crash).
    pub fn run_type_immunity(&mut self, dex: &Dex, id: PokeId, ty: crate::dex::TypeId) -> bool {
        let negate = !self
            .run_event(
                dex,
                &ev::NegateImmunity,
                EvTarget::Poke(id),
                None,
                EffectHandle::None,
                Some(RV::Num(ty.0 as f64)),
                false,
                false,
            )
            .truthy();
        if ty == dex.known_types.ground {
            let flying = self.poke(id).has_type(dex.known_types.flying);
            return negate || !flying;
        }
        let types = self.poke(id).types;
        negate || !types.iter().any(|d| dex.type_immune(ty, d))
    }

    /// pokemon.runEffectiveness(move) — total type mod.
    pub fn run_effectiveness(&mut self, dex: &Dex, id: PokeId) -> i32 {
        let move_type = self.active_move.as_ref().unwrap().move_type;
        let move_eff = self
            .active_move
            .as_ref()
            .and_then(|m| m.id)
            .map(EffectHandle::MoveEff)
            .unwrap_or(EffectHandle::None);
        let types = self.poke(id).types;
        let mut total = 0i32;
        for ty in types.iter() {
            let type_mod = dex.eff(move_type, ty);
            // singleEvent('Effectiveness', move, ...) — only M2 moves have it.
            let rv = self.run_event(
                dex,
                &ev::Effectiveness,
                EvTarget::Poke(id),
                None,
                move_eff,
                Some(RV::Num(type_mod as f64)),
                false,
                false,
            );
            total += rv.as_num() as i32;
        }
        total
    }

    // ------------------------------------------------------------ weather

    /// field.effectiveWeather (no suppression in gen2).
    pub fn field_effective_weather<'d>(&self, dex: &'d Dex) -> &'d str {
        match self.field.weather {
            Some(w) => dex.conds_key(w),
            None => "",
        }
    }

    // ------------------------------------------------------------- items

    /// pokemon.useItem(source?, sourceEffect?).
    pub fn use_item(
        &mut self,
        dex: &Dex,
        id: PokeId,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> bool {
        if self.poke(id).hp <= 0 || !self.poke(id).is_active {
            return false;
        }
        let Some(item) = self.poke(id).item else { return false };
        let mut source_effect = source_effect;
        if source_effect.is_none() {
            source_effect = self.current_effect();
        }
        let mut source = source;
        if source.is_none() {
            source = self.event_stack.last().and_then(|f| f.target);
        }
        if let EffectHandle::Item(se_item) = source_effect {
            if se_item != item && source == Some(id) {
                return false;
            }
        }
        let rv = self.run_event(
            dex,
            &ev::UseItem,
            EvTarget::Poke(id),
            None,
            EffectHandle::None,
            Some(RV::True),
            false,
            false,
        );
        if !rv.truthy() {
            return false;
        }
        let ps = self.poke_str(id);
        let item_name = dex.items.get(item).name.clone();
        self.add(&["-enditem", &ps, &item_name]);
        let boosts = dex.items.get(item).boosts();
        if !boosts.is_empty() {
            self.boost(dex, &boosts, Some(id), source, EffectHandle::Item(item));
        }
        self.single_event(
            dex,
            &ev::Use,
            EffectHandle::Item(item),
            StateLoc::None,
            EvTarget::Poke(id),
            source,
            source_effect,
            None,
        );
        let p = self.poke_mut(id);
        p.last_item = Some(item);
        p.item = None;
        p.item_state = EffectState::default();
        p.used_item_this_turn = true;
        self.refresh_poke_mask(dex, id);
        self.run_event(
            dex,
            &ev::AfterUseItem,
            EvTarget::Poke(id),
            None,
            EffectHandle::None,
            Some(RV::True),
            false,
            false,
        );
        true
    }

    /// pokemon.eatItem(force?, source?, sourceEffect?).
    pub fn eat_item(
        &mut self,
        dex: &Dex,
        id: PokeId,
        force: bool,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> bool {
        let Some(item) = self.poke(id).item else { return false };
        if self.poke(id).hp <= 0 || !self.poke(id).is_active {
            return false;
        }
        let mut source_effect = source_effect;
        if source_effect.is_none() {
            source_effect = self.current_effect();
        }
        let mut source = source;
        if source.is_none() {
            source = self.event_stack.last().and_then(|f| f.target);
        }
        if let EffectHandle::Item(se_item) = source_effect {
            if se_item != item && source == Some(id) {
                return false;
            }
        }
        let use_ok = self
            .run_event(dex, &ev::UseItem, EvTarget::Poke(id), None, EffectHandle::None, Some(RV::True), false, false)
            .truthy();
        let eat_ok = use_ok
            && (force
                || self
                    .run_event(
                        dex,
                        &ev::TryEatItem,
                        EvTarget::Poke(id),
                        None,
                        EffectHandle::None,
                        Some(RV::True),
                        false,
                        false,
                    )
                    .truthy());
        if !eat_ok {
            return false;
        }
        let ps = self.poke_str(id);
        let item_name = dex.items.get(item).name.clone();
        self.add(&["-enditem", &ps, &item_name, "[eat]"]);
        self.single_event(
            dex,
            &ev::Eat,
            EffectHandle::Item(item),
            StateLoc::None,
            EvTarget::Poke(id),
            source,
            source_effect,
            None,
        );
        self.run_event(
            dex,
            &ev::EatItem,
            EvTarget::Poke(id),
            source,
            source_effect,
            Some(RV::True),
            false,
            false,
        );
        let p = self.poke_mut(id);
        p.last_item = Some(item);
        p.item = None;
        p.item_state = EffectState::default();
        p.used_item_this_turn = true;
        self.refresh_poke_mask(dex, id);
        self.run_event(
            dex,
            &ev::AfterUseItem,
            EvTarget::Poke(id),
            None,
            EffectHandle::None,
            Some(RV::True),
            false,
            false,
        );
        true
    }

    /// pokemon.takeItem(source?). Returns the taken item.
    pub fn take_item(&mut self, dex: &Dex, id: PokeId, source: Option<PokeId>) -> Option<crate::dex::ItemId> {
        let source = source.unwrap_or(id);
        let item = self.poke(id).item?;
        let rv = self.run_event(
            dex,
            &ev::TakeItem,
            EvTarget::Poke(id),
            Some(source),
            EffectHandle::None,
            Some(RV::True),
            false,
            false,
        );
        if !rv.truthy() {
            return None;
        }
        let p = self.poke_mut(id);
        p.item = None;
        p.item_state = EffectState::default();
        self.refresh_poke_mask(dex, id);
        // singleEvent('End', item): no gen2 item has onEnd.
        self.run_event(
            dex,
            &ev::AfterTakeItem,
            EvTarget::Poke(id),
            None,
            EffectHandle::None,
            Some(RV::True),
            false,
            false,
        );
        Some(item)
    }

    /// pokemon.setItem(item, source?, effect?).
    pub fn set_item(&mut self, dex: &Dex, id: PokeId, item: crate::dex::ItemId) -> bool {
        if self.poke(id).hp <= 0 || !self.poke(id).is_active {
            return false;
        }
        let item_key = dex.items.key(item).to_string();
        let state = EffectState { id: item_key, ..Default::default() };
        let state = self.init_effect_state(state, true);
        let p = self.poke_mut(id);
        p.item = Some(item);
        p.item_state = state;
        self.refresh_poke_mask(dex, id);
        // singleEvent('End', oldItem) / ('Start', item): no gen2 handlers.
        true
    }

    // ------------------------------------------------------------ various

    /// pokemon.getLastAttackedBy().
    pub fn get_last_attacked_by(&self, id: PokeId) -> Option<&Attacker> {
        self.poke(id).attacked_by.last()
    }

    /// pokemon.sethp(d) (Pain Split).
    pub fn set_hp(&mut self, id: PokeId, d: f64) -> f64 {
        if self.poke(id).hp <= 0 {
            return 0.0;
        }
        let mut d = super::tr(d);
        if d < 1.0 {
            d = 1.0;
        }
        let p = self.poke_mut(id);
        let mut delta = d - p.hp as f64;
        p.hp += delta as i32;
        if p.hp > p.maxhp {
            delta -= (p.hp - p.maxhp) as f64;
            p.hp = p.maxhp;
        }
        delta
    }

    /// pokemon.setType(newType).
    pub fn set_type(&mut self, id: PokeId, new_types: crate::dex::TypeList) -> bool {
        self.poke_mut(id).types = new_types;
        true
    }

    /// pokemon.transformInto(target) — gen2 slice.
    pub fn transform_into(&mut self, dex: &Dex, id: PokeId, target: PokeId) -> bool {
        if self.poke(target).fainted || self.poke(target).transformed {
            return false;
        }
        let (t_species, t_types, t_stats, t_boosts, t_hp_type, t_hp_power, t_times_attacked) = {
            let t = self.poke(target);
            (
                t.species,
                t.types.clone(),
                t.stored_stats,
                t.boosts,
                t.hp_type.clone(),
                t.hp_power,
                t.times_attacked,
            )
        };
        let t_move_ids: Vec<crate::dex::MoveId> =
            self.poke(target).move_slots.iter().map(|m| m.id).collect();
        // setSpecies caches speed from spreadModify(newSpecies.baseStats,
        // this.set) — the USER's level/DVs/stat exp on the TARGET's base spe.
        // (transformInto then overwrites storedStats with the target's copies
        // WITHOUT refreshing the cached speed.)
        let own_spe_on_target_base = {
            let p = self.poke(id);
            let base = dex.species.get(t_species).base_stats.spe as f64;
            let iv = p.set_ivs[5] as f64;
            let ev_term = super::tr(p.set_evs[5] as f64 / 4.0);
            super::tr(super::tr(2.0 * base + iv + ev_term) * p.level as f64 / 100.0 + 5.0) as i32
        };
        {
            let p = self.poke_mut(id);
            p.species = t_species;
            p.transformed = true;
            p.types = t_types;
            p.stored_stats = t_stats;
            p.speed = own_spe_on_target_base;
            p.hp_type = t_hp_type;
            p.hp_power = t_hp_power;
            p.times_attacked = t_times_attacked;
            p.boosts = t_boosts;
            p.move_slots = MoveSlots::default();
        }
        for mid in t_move_ids {
            let ms = dex.move_static(mid);
            let pp = ms.pp.min(5);
            let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
            let mut maxpp = ms.pp * (5 + pp_ups) / 5;
            if ms.pp == 40 {
                maxpp -= pp_ups;
            }
            self.poke_mut(id).move_slots.push(MoveSlot {
                id: mid,
                pp,
                maxpp,
                disabled: false,
                used: false,
                shared: false,
            });
        }
        let ps = self.poke_str(id);
        let ts = self.poke_str(target);
        self.add(&["-transform", &ps, &ts]);
        true
    }

    /// pokemon.effectiveWeather (no utility umbrella in gen2).
    pub fn effective_weather(&self, id: PokeId) -> String {
        let _ = id;
        // resolved via field only; dex key lookup needs dex, so store id here
        self.field_weather_key.clone()
    }

    pub fn field_is_weather(&self, key: &str) -> bool {
        self.field_weather_key == key
    }
}

fn b_add_fail_status(b: &mut Battle, target_str: &str, status: &str) {
    b.add(&["-fail", target_str, status]);
}
