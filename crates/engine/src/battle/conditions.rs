//! Ported condition callbacks (merged gen2stadium2 semantics — see
//! reference/merged-conditions.txt, dumped from the live PS dex).
//!
//! Dispatch is by (condition id, callback name). Anything not ported panics
//! loudly so the conformance harness reports exactly what is missing.

use crate::dex::{Dex, MoveId};
use crate::state::*;

use super::events::EvTarget;
use super::{EffectHandle, RV};

/// Markers for callbacks implemented in code but absent from the data's
/// callback lists. PS allows non-function callbacks (constants); the export
/// tool only records function-valued ones, so constants are listed here.
pub fn has_builtin(cond: &str, callback: &str) -> bool {
    matches!((cond, callback), ("mustrecharge", "onLockMove"))
}

/// durationCallback for conditions that define one. Returns None if the
/// condition has no durationCallback.
pub fn duration_callback(
    b: &mut Battle,
    dex: &Dex,
    cond: &str,
    _target: Option<PokeId>,
    _source: Option<PokeId>,
    _source_effect: EffectHandle,
) -> Option<i32> {
    let _ = dex;
    match cond {
        // gen2 partiallytrapped: random(3,6)
        "partiallytrapped" => Some(b.prng.random_range(3, 6) as i32),
        // gen2 lockedmove: random(2,4)
        "lockedmove" => Some(b.prng.random_range(2, 4) as i32),
        // weathers: 5 unless rock items (M2)
        "raindance" | "sunnyday" | "sandstorm" => Some(5),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    b: &mut Battle,
    dex: &Dex,
    effect: EffectHandle,
    callback_name: &str,
    state: StateLoc,
    target: EvTarget,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
    _has_relay: bool,
) -> RV {
    match effect {
        EffectHandle::Cond(c) => {
            let key = dex.conds_key(c).to_string();
            dispatch_cond(b, dex, &key, callback_name, state, target, source, source_effect, relay)
        }
        EffectHandle::MoveEff(m) => {
            super::moveexec::dispatch_move_callback(b, dex, m, callback_name, target, source, relay)
        }
        EffectHandle::Item(i) => {
            panic!("unported item callback: {} {}", dex.items.key(i), callback_name)
        }
        _ => RV::Undef,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_cond(
    b: &mut Battle,
    dex: &Dex,
    cond: &str,
    cb: &str,
    state: StateLoc,
    target: EvTarget,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
) -> RV {
    let tpoke = target.poke();
    match (cond, cb) {
        // ------------------------------------------------------------- brn
        ("brn", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "brn"]);
            b.add_volatile(dex, t, "brnattackdrop", None, EffectHandle::None);
            RV::Undef
        }
        ("brn", "onAfterMoveSelf") | ("brn", "onAfterSwitchInSelf") => {
            residualdmg(b, dex, tpoke.unwrap());
            RV::Undef
        }
        ("brn", "onSwitchIn") => {
            let t = tpoke.unwrap();
            b.add_volatile(dex, t, "brnattackdrop", None, EffectHandle::None);
            RV::Undef
        }
        // ------------------------------------------------------------- par
        ("par", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "par"]);
            b.add_volatile(dex, t, "parspeeddrop", None, EffectHandle::None);
            RV::Undef
        }
        ("par", "onBeforeMove") => {
            let t = tpoke.unwrap();
            if b.prng.random_chance(1, 4) {
                let ts = b.poke_str(t);
                b.add(&["cant", &ts, "par"]);
                return RV::False;
            }
            RV::Undef
        }
        ("par", "onSwitchIn") => {
            let t = tpoke.unwrap();
            b.add_volatile(dex, t, "parspeeddrop", None, EffectHandle::None);
            RV::Undef
        }
        // ------------------------------------------------------------- slp
        ("slp", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            if let EffectHandle::MoveEff(m) = source_effect {
                let from = format!("[from] move: {}", dex.move_static(m).name);
                b.add(&["-status", &ts, "slp", &from]);
            } else {
                b.add(&["-status", &ts, "slp"]);
            }
            // 1-4 turns, guaranteed 1 turn of sleep (stadium2)
            let time = b.prng.random_range(2, 5) as i64;
            if let Some(st) = b.state_at_mut(state) {
                st.set_int("time", time);
            }
            if b.remove_volatile(dex, t, "nightmare") {
                let ts = b.poke_str(t);
                b.add(&["-end", &ts, "Nightmare", "[silent]"]);
            }
            RV::Undef
        }
        ("slp", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let loc = StateLoc::Status(t);
            let time = b.state_at(loc).map(|s| s.get_int("time")).unwrap_or(0) - 1;
            if let Some(st) = b.state_at_mut(loc) {
                st.set_int("time", time);
            }
            if time <= 0 {
                b.cure_status(dex, t, false);
                return RV::Undef;
            }
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "slp"]);
            let sleep_usable = b
                .active_move
                .as_ref()
                .map(|m| m.sleep_usable)
                .unwrap_or(false);
            if sleep_usable {
                return RV::Undef;
            }
            RV::False
        }
        // ------------------------------------------------------------- frz
        ("frz", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "frz"]);
            RV::Undef
        }
        ("frz", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let defrost = b.active_move.as_ref().map(|m| m.has_flag("defrost")).unwrap_or(false);
            if defrost {
                return RV::Undef;
            }
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "frz"]);
            RV::False
        }
        ("frz", "onAfterMoveSecondary") => {
            // (move.secondary?.status === 'brn' || move.statusRoll === 'brn') → cure
            let t = tpoke.unwrap();
            let is_brn = b
                .active_move
                .as_ref()
                .map(|m| {
                    m.secondaries
                        .iter()
                        .any(|s| s.status.as_deref() == Some("brn"))
                })
                .unwrap_or(false);
            if is_brn {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        ("frz", "onAfterMoveSecondarySelf") => {
            let t = tpoke.unwrap();
            let defrost = b.active_move.as_ref().map(|m| m.has_flag("defrost")).unwrap_or(false);
            if defrost {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        ("frz", "onResidual") => {
            let t = tpoke.unwrap();
            if b.prng.random_chance(25, 256) {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        // ------------------------------------------------------------- psn
        ("psn", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "psn"]);
            RV::Undef
        }
        ("psn", "onAfterMoveSelf") | ("psn", "onAfterSwitchInSelf") => {
            residualdmg(b, dex, tpoke.unwrap());
            RV::Undef
        }
        // ------------------------------------------------------------- tox
        ("tox", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "tox"]);
            let rd = dex.conds_id("residualdmg").unwrap();
            if !b.poke(t).has_volatile(rd) {
                b.add_volatile(dex, t, "residualdmg", None, EffectHandle::None);
            }
            if let Some(vs) = b.poke_mut(t).volatile_mut(rd) {
                vs.set_int("counter", 0);
            }
            RV::Undef
        }
        ("tox", "onAfterMoveSelf") => {
            let t = tpoke.unwrap();
            let rd = dex.conds_id("residualdmg").unwrap();
            let counter = b.poke(t).volatile(rd).map(|v| v.get_int("counter")).unwrap_or(0);
            let maxhp = b.poke(t).maxhp as f64;
            let dmg = super::clamp_int_range((maxhp / 16.0).floor(), Some(1.0), None) * counter as f64;
            let eff = EffectHandle::Cond(dex.conds_id("tox").unwrap());
            b.damage(dex, dmg, Some(t), Some(t), DamageEffect::Effect(eff), false);
            RV::Undef
        }
        ("tox", "onSwitchIn") => {
            let t = tpoke.unwrap();
            b.poke_mut(t).status = Status::Psn;
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "psn", "[silent]"]);
            RV::Undef
        }
        ("tox", "onAfterSwitchInSelf") => {
            let t = tpoke.unwrap();
            let maxhp = b.poke(t).maxhp as f64;
            let dmg = super::clamp_int_range((maxhp / 16.0).floor(), Some(1.0), None);
            // this.damage(...) with implicit target/effect from the event
            let eff = EffectHandle::Cond(dex.conds_id("tox").unwrap());
            b.damage(dex, dmg, Some(t), Some(t), DamageEffect::Effect(eff), false);
            RV::Undef
        }
        // ------------------------------------------------------- confusion
        ("confusion", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            let from_locked = matches!(
                source_effect,
                EffectHandle::Cond(c) if dex.conds_key(c) == "lockedmove"
            );
            if from_locked {
                b.add(&["-start", &ts, "confusion", "[silent]"]);
            } else {
                b.add(&["-start", &ts, "confusion"]);
            }
            let time = b.prng.random_range(2, 6) as i64;
            if let Some(st) = b.state_at_mut(state) {
                st.set_int("time", time);
            }
            RV::Undef
        }
        ("confusion", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "confusion"]);
            RV::Undef
        }
        ("confusion", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let cid = dex.conds_id("confusion").unwrap();
            let time = b.poke(t).volatile(cid).map(|v| v.get_int("time")).unwrap_or(0) - 1;
            if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                vs.set_int("time", time);
            }
            if time == 0 {
                b.remove_volatile(dex, t, "confusion");
                return RV::Undef;
            }
            let ts = b.poke_str(t);
            b.add(&["-activate", &ts, "confusion"]);
            if b.prng.random_chance(1, 2) {
                return RV::Undef;
            }
            // 40 BP typeless physical self-hit
            let base_move_type = b
                .active_move
                .as_ref()
                .map(|m| m.move_type.clone())
                .unwrap_or_else(|| "Normal".to_string());
            let selfdestruct =
                b.active_move.as_ref().map(|m| m.selfdestruct).unwrap_or(false);
            let mut fake = super::moveexec::synthetic_move(40);
            fake.base_move_type = base_move_type;
            fake.is_confusion_self_hit = true;
            fake.no_damage_variance = true;
            fake.will_crit = Some(false);
            fake.selfdestruct = selfdestruct;
            let damage = super::moveexec::get_damage_synthetic(b, dex, t, t, fake);
            let Some(damage) = damage else {
                panic!("Confusion damage not dealt");
            };
            let eff = EffectHandle::Cond(dex.conds_id("confusion").unwrap());
            b.direct_damage(dex, damage as f64, Some(t), None, eff);
            RV::False
        }
        // ---------------------------------------------------------- flinch
        ("flinch", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "flinch"]);
            b.run_event(dex, "Flinch", EvTarget::Poke(t), None, EffectHandle::None, None, false, false);
            RV::False
        }
        // ------------------------------------------------ partiallytrapped
        ("partiallytrapped", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            let src_move_name = b
                .state_at(state)
                .and_then(|s| s.source_effect.clone())
                .map(|id| {
                    dex.moves
                        .id(&id)
                        .map(|m| dex.move_static(m).name.clone())
                        .unwrap_or(id)
                })
                .unwrap_or_default();
            let of = format!("[of] {}", b.poke_str(source.unwrap()));
            let activate = format!("move: {src_move_name}");
            b.add(&["-activate", &ts, &activate, &of]);
            // gen5+ merged data: boundDivisor = bindingband ? 8 : 16 (no items in M1)
            if let Some(st) = b.state_at_mut(state) {
                st.set_int("boundDivisor", 16);
            }
            RV::Undef
        }
        ("partiallytrapped", "onResidual") => {
            let t = tpoke.unwrap();
            let (trapper, divisor, src_move) = {
                let st = b.state_at(state).unwrap();
                (st.source, st.get_int("boundDivisor").max(1), st.source_effect.clone())
            };
            let trapper_gone = match trapper {
                Some(tr_id) => {
                    let tp = b.poke(tr_id);
                    !tp.is_active || tp.hp <= 0 || tp.active_turns == 0
                }
                None => true,
            };
            if trapper_gone {
                let cid = dex.conds_id("partiallytrapped").unwrap();
                b.poke_mut(t).volatiles.retain(|(k, _)| *k != cid);
                let ts = b.poke_str(t);
                let name = src_move_name(dex, &src_move);
                b.add(&["-end", &ts, &name, "[partiallytrapped]", "[silent]"]);
                return RV::Undef;
            }
            let dmg = b.poke(t).base_maxhp as f64 / divisor as f64;
            let eff = EffectHandle::Cond(dex.conds_id("partiallytrapped").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        ("partiallytrapped", "onEnd") => {
            let t = tpoke.unwrap();
            let src_move = b.state_at(state).and_then(|s| s.source_effect.clone());
            let ts = b.poke_str(t);
            let name = src_move_name(dex, &src_move);
            b.add(&["-end", &ts, &name, "[partiallytrapped]"]);
            RV::Undef
        }
        ("partiallytrapped", "onTrapPokemon") => {
            let t = tpoke.unwrap();
            let source_active = b
                .state_at(state)
                .and_then(|s| s.source)
                .map(|s| b.poke(s).is_active)
                .unwrap_or(false);
            if source_active {
                b.try_trap(t);
            }
            RV::Undef
        }
        // ----------------------------------------------------- residualdmg
        ("residualdmg", "onStart") => {
            let t = tpoke.unwrap();
            let cid = dex.conds_id("residualdmg").unwrap();
            if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                vs.set_int("counter", 0);
            }
            RV::Undef
        }
        ("residualdmg", "onAfterMoveSelf") | ("residualdmg", "onAfterSwitchInSelf") => {
            let t = tpoke.unwrap();
            if matches!(b.poke(t).status, Status::Brn | Status::Psn | Status::Tox) {
                let cid = dex.conds_id("residualdmg").unwrap();
                if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                    let c = vs.get_int("counter");
                    vs.set_int("counter", c + 1);
                }
            }
            RV::Undef
        }
        // ---------------------------------------------------- mustrecharge
        ("mustrecharge", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "recharge"]);
            b.remove_volatile(dex, t, "mustrecharge");
            b.remove_volatile(dex, t, "truant");
            RV::Null
        }
        ("mustrecharge", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-mustrecharge", &ts]);
            RV::Undef
        }
        ("mustrecharge", "onLockMove") => RV::Str("recharge".to_string()),
        // --------------------------------------------------------- trapped
        ("trapped", "onTrapPokemon") => {
            let t = tpoke.unwrap();
            b.try_trap(t);
            RV::Undef
        }
        ("trapped", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-activate", &ts, "trapped"]);
            RV::Undef
        }
        // -------------------------------------------------------- weathers
        ("raindance", "onFieldStart") => {
            b.add(&["-weather", "RainDance"]);
            RV::Undef
        }
        ("raindance", "onFieldResidual") => {
            b.add(&["-weather", "RainDance", "[upkeep]"]);
            b.each_event(dex, "Weather", None);
            RV::Undef
        }
        ("raindance", "onFieldEnd") => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        ("sunnyday", "onFieldStart") => {
            b.add(&["-weather", "SunnyDay"]);
            RV::Undef
        }
        ("sunnyday", "onFieldResidual") => {
            b.add(&["-weather", "SunnyDay", "[upkeep]"]);
            b.each_event(dex, "Weather", None);
            RV::Undef
        }
        ("sunnyday", "onFieldEnd") => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        ("sunnyday", "onImmunity") => {
            // onImmunity(type, pokemon): frz → false (can't freeze in sun)
            let t = tpoke.unwrap();
            if b.effective_weather(t) != "sunnyday" {
                return RV::Undef;
            }
            if relay == RV::Str("frz".to_string()) {
                return RV::False;
            }
            RV::Undef
        }
        ("sandstorm", "onFieldStart") => {
            b.add(&["-weather", "Sandstorm"]);
            RV::Undef
        }
        ("sandstorm", "onFieldResidual") => {
            b.add(&["-weather", "Sandstorm", "[upkeep]"]);
            if b.field_is_weather("sandstorm") {
                b.each_event(dex, "Weather", None);
            }
            RV::Undef
        }
        ("sandstorm", "onWeather") => {
            // gen2: this.damage(target.baseMaxhp / 8)
            let t = tpoke.unwrap();
            let dmg = b.poke(t).base_maxhp as f64 / 8.0;
            let eff = EffectHandle::Cond(dex.conds_id("sandstorm").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        ("sandstorm", "onFieldEnd") => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        // ----------------------------------------------------------- rules
        ("stadiumsleepclause", "onSetStatus") => {
            // (status, target, source): ally source → undefined
            let t = tpoke.unwrap();
            if let Some(src) = source {
                if src.side == t.side {
                    return RV::Undef;
                }
            }
            if relay == RV::Str("slp".to_string()) {
                let side = t.side as usize;
                let any_slp = b.sides[side]
                    .party
                    .iter()
                    .map(|&slot| &b.sides[side].roster[slot as usize])
                    .any(|p| p.hp > 0 && p.status == Status::Slp);
                if any_slp {
                    b.add(&["-message", "Sleep Clause activated. (In official formats, Sleep Clause activates if any of the opponent's Pokemon are asleep, even if self-inflicted from Rest)"]);
                    return RV::False;
                }
            }
            RV::Undef
        }
        ("freezeclausemod", "onSetStatus") => {
            let t = tpoke.unwrap();
            if let Some(src) = source {
                if src.side == t.side {
                    return RV::Undef;
                }
            }
            if relay == RV::Str("frz".to_string()) {
                let side = t.side as usize;
                let any_frz = b.sides[side]
                    .party
                    .iter()
                    .map(|&slot| &b.sides[side].roster[slot as usize])
                    .any(|p| p.status == Status::Frz);
                if any_frz {
                    b.add(&["-message", "Freeze Clause activated."]);
                    return RV::False;
                }
            }
            RV::Undef
        }
        // ------------------------------------------------- marker volatiles
        ("brnattackdrop", _) | ("parspeeddrop", _) => RV::Undef,
        _ => panic!("unported condition callback: {cond} {cb}"),
    }
}

/// `${effect}` (Effect.toString()) — the plain name ("Wrap").
fn src_move_name(dex: &Dex, src_move: &Option<String>) -> String {
    match src_move {
        Some(id) => dex
            .moves
            .id(id)
            .map(|m| dex.move_static(m).name.clone())
            .unwrap_or_else(|| id.clone()),
        None => String::new(),
    }
}

/// The shared gen2 residualdmg helper (brn/psn tick).
fn residualdmg(b: &mut Battle, dex: &Dex, pokemon: PokeId) {
    let rd = dex.conds_id("residualdmg").unwrap();
    let status = b.poke(pokemon).status;
    let eff = EffectHandle::Cond(
        dex.conds_id(status.as_str()).unwrap_or_else(|| dex.conds_id("brn").unwrap()),
    );
    if b.poke(pokemon).has_volatile(rd) {
        let counter = b.poke(pokemon).volatile(rd).map(|v| v.get_int("counter")).unwrap_or(0);
        let maxhp = b.poke(pokemon).maxhp as f64;
        let dmg = super::clamp_int_range((maxhp / 16.0).floor() * counter as f64, Some(1.0), None);
        b.damage(dex, dmg, Some(pokemon), None, DamageEffect::Effect(eff), false);
        b.hint(
            "In Gen 2, Toxic's counter is retained through Baton Pass/Heal Bell and applies to PSN/BRN.",
            true,
        );
    } else {
        let maxhp = b.poke(pokemon).maxhp as f64;
        let dmg = super::clamp_int_range((maxhp / 8.0).floor(), Some(1.0), None);
        b.damage(dex, dmg, Some(pokemon), None, DamageEffect::Effect(eff), false);
    }
}

/// battle.damage effect argument: an effect handle or the special string
/// forms 'recoil' / 'drain' (PS resolves those via conditions.getByID).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DamageEffect {
    Effect(EffectHandle),
    Recoil,
    Drain,
}

impl DamageEffect {
    pub fn to_handle(self, dex: &Dex) -> EffectHandle {
        match self {
            DamageEffect::Effect(e) => e,
            DamageEffect::Recoil => EffectHandle::Cond(dex.conds_id("recoil").unwrap()),
            DamageEffect::Drain => EffectHandle::Cond(dex.conds_id("drain").unwrap()),
        }
    }
}

/// Convenience for moveexec: move id by name key.
pub fn move_id(dex: &Dex, key: &str) -> Option<MoveId> {
    dex.moves.id(key)
}
