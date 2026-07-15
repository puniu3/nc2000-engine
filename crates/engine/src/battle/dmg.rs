//! battle.spreadDamage / damage / directDamage / heal + the gen2stadium2
//! Battle.boost override.

use crate::dex::{Dex, EffectType};
use crate::state::*;

use super::conditions::DamageEffect;
use super::events::{ev, EvTarget};
use super::{clamp_int_range, EffectHandle, RV};

impl Battle {
    /// battle.damage(damage, target, source, effect, instafaint). `None` args
    /// default from the current event frame (PS `||=` semantics).
    pub fn damage(
        &mut self,
        dex: &Dex,
        damage: f64,
        target: Option<PokeId>,
        source: Option<PokeId>,
        effect: DamageEffect,
        instafaint: bool,
    ) -> Option<f64> {
        let mut target = target;
        let mut source = source;
        let mut effect = effect;
        if let Some(frame) = self.event_stack.last() {
            if target.is_none() {
                target = frame.target;
            }
            if source.is_none() {
                source = frame.source;
            }
            if effect == DamageEffect::Effect(EffectHandle::None) {
                effect = DamageEffect::Effect(self.current_effect());
            }
        }
        self.spread_damage(dex, damage, target, source, effect, instafaint)
    }

    /// battle.spreadDamage collapsed to the singles case. Returns
    /// Some(dealt) or None (PS false/undefined).
    pub fn spread_damage(
        &mut self,
        dex: &Dex,
        damage: f64,
        target: Option<PokeId>,
        source: Option<PokeId>,
        effect: DamageEffect,
        instafaint: bool,
    ) -> Option<f64> {
        let effect_handle = effect.to_handle(dex);
        let Some(target) = target else { return Some(0.0) };
        if self.poke(target).hp <= 0 {
            return Some(0.0);
        }
        if !self.poke(target).is_active {
            return None; // PS false
        }
        let mut target_damage = damage;
        if target_damage != 0.0 {
            target_damage = clamp_int_range(target_damage, Some(1.0), None);
        }

        let effect_id = self.effect_id(dex, effect_handle).to_string();
        if effect_id != "struggle-recoil" {
            // weather immunity
            if self.effect_type(dex, effect_handle) == EffectType::Weather
                && !self.run_status_immunity(dex, target, &effect_id, false)
            {
                return Some(0.0);
            }
            let rv = self.run_event(
                dex,
                &ev::Damage,
                EvTarget::Poke(target),
                source,
                effect_handle,
                Some(RV::Num(target_damage)),
                true,
                false,
            );
            match rv {
                RV::Num(n) => target_damage = n,
                RV::True => {}
                _ => {
                    // damage event failed
                    return None;
                }
            }
        }
        if target_damage != 0.0 {
            target_damage = clamp_int_range(target_damage, Some(1.0), None);
        }

        let dealt = self.pokemon_damage(target, target_damage, source, effect_handle);
        let target_damage = dealt;
        if target_damage != 0.0 {
            let hp = self.poke(target).hp;
            self.poke_mut(target).hurt_this_turn = Some(hp);
        }
        if let Some(src) = source {
            if self.effect_type(dex, effect_handle) == EffectType::Move {
                self.poke_mut(src).last_damage = target_damage as i64;
            }
        }

        let name = {
            let full = self.effect_fullname(dex, effect_handle);
            if full == "tox" {
                "psn".to_string()
            } else {
                full
            }
        };
        let ts = self.poke_str(target);
        let (secret, shared) = self.get_health(target);
        let side_id = format!("p{}", target.side + 1);
        match effect_id.as_str() {
            "partiallytrapped" => {
                let pt = dex.conds_id("partiallytrapped").unwrap();
                let src_eff = self
                    .poke(target)
                    .volatile(pt)
                    .and_then(|v| v.source_effect.clone())
                    .and_then(|id| dex.moves.id(&id))
                    .map(|m| format!("move: {}", dex.move_static(m).name))
                    .unwrap_or_default();
                let from = format!("[from] {src_eff}");
                self.add_split(
                    &side_id,
                    &["-damage", &ts, &secret, &from, "[partiallytrapped]"],
                    &["-damage", &ts, &shared, &from, "[partiallytrapped]"],
                );
            }
            "confused" => {
                self.add_split(
                    &side_id,
                    &["-damage", &ts, &secret, "[from] confusion"],
                    &["-damage", &ts, &shared, "[from] confusion"],
                );
            }
            _ => {
                if self.effect_type(dex, effect_handle) == EffectType::Move || name.is_empty() {
                    self.add_split(&side_id, &["-damage", &ts, &secret], &["-damage", &ts, &shared]);
                } else if source.is_some() && source != Some(target) {
                    let of = format!("[of] {}", self.poke_str(source.unwrap()));
                    let from = format!("[from] {name}");
                    self.add_split(
                        &side_id,
                        &["-damage", &ts, &secret, &from, &of],
                        &["-damage", &ts, &shared, &from, &of],
                    );
                } else {
                    let from = format!("[from] {name}");
                    self.add_split(
                        &side_id,
                        &["-damage", &ts, &secret, &from],
                        &["-damage", &ts, &shared, &from],
                    );
                }
            }
        }

        if target_damage != 0.0 && self.effect_type(dex, effect_handle) == EffectType::Move {
            // gen <= 4 drain handling lives here
            let drain = match effect_handle {
                EffectHandle::MoveEff(m) => dex.move_static(m).drain,
                _ => None,
            };
            if let (Some((num, den)), Some(src)) = (drain, source) {
                let amount = clamp_int_range(
                    (target_damage * num as f64 / den as f64).floor(),
                    Some(1.0),
                    None,
                );
                self.heal(dex, amount, Some(src), Some(target), HealEffect::Drain);
            }
        }

        if instafaint && target_damage != 0.0 && self.poke(target).hp <= 0 {
            self.faint_messages(dex, true);
            // gen <= 2
            self.pokemon_faint(target, None, EffectHandle::None);
        }

        Some(target_damage)
    }

    /// battle.directDamage.
    pub fn direct_damage(
        &mut self,
        dex: &Dex,
        damage: f64,
        target: Option<PokeId>,
        source: Option<PokeId>,
        effect: EffectHandle,
    ) -> f64 {
        let mut target = target;
        let mut source = source;
        let mut effect = effect;
        if let Some(frame) = self.event_stack.last() {
            if target.is_none() {
                target = frame.target;
            }
            if source.is_none() {
                source = frame.source;
            }
            if effect.is_none() {
                effect = self.current_effect();
            }
        }
        let Some(target) = target else { return 0.0 };
        if self.poke(target).hp <= 0 || damage == 0.0 {
            return 0.0;
        }
        let damage = clamp_int_range(damage, Some(1.0), None);
        let dealt = self.pokemon_damage(target, damage, source, effect);
        let ts = self.poke_str(target);
        let (secret, shared) = self.get_health(target);
        let side_id = format!("p{}", target.side + 1);
        let effect_id = self.effect_id(dex, effect).to_string();
        match effect_id.as_str() {
            "strugglerecoil" => {
                self.add_split(
                    &side_id,
                    &["-damage", &ts, &secret, "[from] recoil"],
                    &["-damage", &ts, &shared, "[from] recoil"],
                );
            }
            "confusion" => {
                self.add_split(
                    &side_id,
                    &["-damage", &ts, &secret, "[from] confusion"],
                    &["-damage", &ts, &shared, "[from] confusion"],
                );
            }
            _ => {
                self.add_split(&side_id, &["-damage", &ts, &secret], &["-damage", &ts, &shared]);
            }
        }
        if self.poke(target).fainted {
            // battle.faint() — already queued by pokemon_damage's faint path;
            // PS calls this.faint(target) which is idempotent (faintQueued).
            self.pokemon_faint(target, None, EffectHandle::None);
        }
        dealt
    }

    /// battle.heal. Returns Some(healed) / None (false).
    pub fn heal(
        &mut self,
        dex: &Dex,
        damage: f64,
        target: Option<PokeId>,
        source: Option<PokeId>,
        effect: HealEffect,
    ) -> Option<f64> {
        let mut target = target;
        let mut source = source;
        let mut effect_handle = match effect {
            HealEffect::Drain => EffectHandle::Cond(dex.conds_id("drain").unwrap()),
            HealEffect::Effect(e) => e,
        };
        if let Some(frame) = self.event_stack.last() {
            if target.is_none() {
                target = frame.target;
            }
            if source.is_none() {
                source = frame.source;
            }
            if effect_handle.is_none() {
                effect_handle = self.current_effect();
            }
        }
        let mut damage = damage;
        if damage != 0.0 && damage <= 1.0 {
            damage = 1.0;
        }
        damage = super::tr(damage);
        // TryHeal event
        let rv = self.run_event(
            dex,
            &ev::TryHeal,
            target.map(EvTarget::Poke).unwrap_or(EvTarget::Battle),
            source,
            effect_handle,
            Some(RV::Num(damage)),
            false,
            false,
        );
        let damage = match rv {
            RV::Num(n) => n,
            RV::True => damage,
            _ => return None,
        };
        if damage == 0.0 {
            return None;
        }
        let target = target?;
        if self.poke(target).hp <= 0 || !self.poke(target).is_active {
            return None;
        }
        if self.poke(target).hp >= self.poke(target).maxhp {
            return None;
        }
        let final_damage = self.pokemon_heal(target, damage)?;
        let ts = self.poke_str(target);
        let (secret, shared) = self.get_health(target);
        let side_id = format!("p{}", target.side + 1);
        let effect_id = self.effect_id(dex, effect_handle).to_string();
        match (effect, effect_id.as_str()) {
            (HealEffect::Drain, _) => {
                let of = format!("[of] {}", self.poke_str(source.unwrap()));
                self.add_split(
                    &side_id,
                    &["-heal", &ts, &secret, "[from] drain", &of],
                    &["-heal", &ts, &shared, "[from] drain", &of],
                );
            }
            (_, "leechseed") | (_, "rest") => {
                self.add_split(
                    &side_id,
                    &["-heal", &ts, &secret, "[silent]"],
                    &["-heal", &ts, &shared, "[silent]"],
                );
            }
            _ => {
                if self.effect_type(dex, effect_handle) == EffectType::Move
                    || effect_handle.is_none()
                {
                    self.add_split(&side_id, &["-heal", &ts, &secret], &["-heal", &ts, &shared]);
                } else if source.is_some() && source != Some(target) {
                    let fullname = self.effect_fullname(dex, effect_handle);
                    let from = format!("[from] {fullname}");
                    let of = format!("[of] {}", self.poke_str(source.unwrap()));
                    self.add_split(
                        &side_id,
                        &["-heal", &ts, &secret, &from, &of],
                        &["-heal", &ts, &shared, &from, &of],
                    );
                } else {
                    let fullname = self.effect_fullname(dex, effect_handle);
                    let from = format!("[from] {fullname}");
                    self.add_split(
                        &side_id,
                        &["-heal", &ts, &secret, &from],
                        &["-heal", &ts, &shared, &from],
                    );
                }
            }
        }
        self.run_event(
            dex,
            &ev::Heal,
            EvTarget::Poke(target),
            source,
            effect_handle,
            Some(RV::Num(final_damage)),
            false,
            false,
        );
        Some(final_damage)
    }

    /// gen2stadium2 Battle.boost override. Returns Some(true) on any success,
    /// None if nothing happened (PS null), Some(false) unreachable.
    pub fn boost(
        &mut self,
        dex: &Dex,
        boosts: &[(usize, i8)],
        target: Option<PokeId>,
        source: Option<PokeId>,
        effect: EffectHandle,
    ) -> Option<bool> {
        let mut target = target;
        let mut source = source;
        let mut effect = effect;
        if let Some(frame) = self.event_stack.last() {
            if target.is_none() {
                target = frame.target;
            }
            if source.is_none() {
                source = frame.source;
            }
            if effect.is_none() {
                effect = self.current_effect();
            }
        }
        let target = target?;
        if self.poke(target).hp <= 0 {
            return Some(false); // PS returns 0
        }
        // TryBoost event — relayVar is the boost table (mist deletes
        // negative entries in place).
        let saved_pending = self.pending_boosts.take();
        self.pending_boosts = Some(boosts.to_vec());
        let rv = self.run_event(
            dex,
            &ev::TryBoost,
            EvTarget::Poke(target),
            source,
            effect,
            Some(RV::True),
            false,
            false,
        );
        let boosts = self.pending_boosts.take().unwrap_or_default();
        self.pending_boosts = saved_pending;
        if rv == RV::False || rv == RV::Null {
            return None;
        }
        let mut success = None;
        for &(stat, amount) in &boosts {
            let mut boost_by = self.pokemon_boost_by(dex, target, &[(stat, amount)]);
            let mut msg = "-boost";
            if amount < 0 {
                msg = "-unboost";
                boost_by = -boost_by;
            }
            if boost_by != 0 {
                success = Some(true);
                // brn/par drop removal on boost
                if stat == 0 && self.poke(target).status == Status::Brn {
                    let c = dex.conds_id("brnattackdrop").unwrap();
                    if self.poke(target).has_volatile(c) {
                        self.remove_volatile(dex, target, "brnattackdrop");
                    }
                }
                if stat == 4 && self.poke(target).status == Status::Par {
                    let c = dex.conds_id("parspeeddrop").unwrap();
                    if self.poke(target).has_volatile(c) {
                        self.remove_volatile(dex, target, "parspeeddrop");
                    }
                }
                let ts = self.poke_str(target);
                let stat_name = crate::dex::BOOST_KEYS[stat];
                let amount_str = boost_by.to_string();
                let is_move_or_none = effect.is_none()
                    || self.effect_type(dex, effect) == EffectType::Move;
                if is_move_or_none {
                    self.add(&[msg, &ts, stat_name, &amount_str]);
                } else {
                    let from = format!("[from] {}", self.effect_fullname(dex, effect));
                    self.add(&[msg, &ts, stat_name, &amount_str, &from]);
                }
                self.run_event(
                    dex,
                    &ev::AfterEachBoost,
                    EvTarget::Poke(target),
                    source,
                    effect,
                    Some(RV::True),
                    false,
                    false,
                );
            }
        }
        self.run_event(
            dex,
            &ev::AfterBoost,
            EvTarget::Poke(target),
            source,
            effect,
            Some(RV::True),
            false,
            false,
        );
        success
    }
}

/// battle.heal effect argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealEffect {
    Effect(EffectHandle),
    Drain,
}
