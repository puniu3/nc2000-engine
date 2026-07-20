//! Ported condition callbacks (merged gen2stadium2 semantics — see
//! reference/merged-conditions.txt, dumped from the live PS dex).
//!
//! Dispatch is by (condition id, callback name). Anything not ported panics
//! loudly so the conformance harness reports exactly what is missing.

use crate::dex::{Cb, Dex, MoveId};
use crate::state::*;

use super::events::{ev, EvTarget};
use super::{EffectHandle, RV};

/// Markers for callbacks implemented in code but absent from the data's
/// callback lists. PS allows non-function callbacks (constants); the export
/// tool only records function-valued ones, so constants are listed here.
pub fn has_builtin(cond: &str, callback: &str) -> bool {
    matches!(
        (cond, callback),
        ("mustrecharge", "onLockMove") | ("rollout", "onLockMove") | ("bide", "onSemiLockMove")
    )
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
        "bide" => Some(b.prng.random_range(3, 5) as i32),
        "disable" => Some(b.prng.random_range(2, 6) as i32),
        "encore" => Some(b.prng.random_range(3, 7) as i32),
        "safeguard" => Some(5),
        // weathers: 5 unless rock items (not in NC2000's item pool)
        "raindance" | "sunnyday" | "sandstorm" => Some(5),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    b: &mut Battle,
    dex: &Dex,
    effect: EffectHandle,
    cb: Cb,
    state: StateLoc,
    target: EvTarget,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
    _has_relay: bool,
) -> RV {
    let callback_name = dex.cb_key(cb);
    match effect {
        EffectHandle::Cond(c) => {
            let key = dex.conds_key(c);
            dispatch_cond(b, dex, key, callback_name, state, target, source, source_effect, relay)
        }
        EffectHandle::MoveEff(m) => {
            super::moveexec::dispatch_move_callback(b, dex, m, callback_name, target, source, relay)
        }
        EffectHandle::Item(i) => super::items::dispatch_item(
            b,
            dex,
            i,
            callback_name,
            target,
            source,
            source_effect,
            relay,
        ),
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
                st.set_int(crate::state::DK::Time, time);
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
            let time = b.state_at(loc).map(|s| s.get_int(crate::state::DK::Time)).unwrap_or(0) - 1;
            if let Some(st) = b.state_at_mut(loc) {
                st.set_int(crate::state::DK::Time, time);
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
            let defrost = b.active_move.as_ref().map(|m| m.has_flag(dex, "defrost")).unwrap_or(false);
            if defrost {
                return RV::Undef;
            }
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "frz"]);
            RV::False
        }
        ("frz", "onAfterMoveSecondary") => {
            // (move.secondary?.status === 'brn' || move.statusRoll === 'brn') → cure
            // statusRoll is tri attack's dynamically rolled status.
            let t = tpoke.unwrap();
            let is_brn = b
                .active_move
                .as_ref()
                .map(|m| {
                    m.secondaries
                        .iter()
                        .any(|s| s.status.as_deref() == Some("brn"))
                        || m.status_roll.as_deref() == Some("brn")
                })
                .unwrap_or(false);
            if is_brn {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        ("frz", "onAfterMoveSecondarySelf") => {
            let t = tpoke.unwrap();
            let defrost = b.active_move.as_ref().map(|m| m.has_flag(dex, "defrost")).unwrap_or(false);
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
            let rd = crate::cond_id!(dex, "residualdmg").unwrap();
            if !b.poke(t).has_volatile(rd) {
                b.add_volatile(dex, t, "residualdmg", None, EffectHandle::None);
            }
            if let Some(vs) = b.poke_mut(t).volatile_mut(rd) {
                vs.set_int(crate::state::DK::Counter, 0);
            }
            RV::Undef
        }
        ("tox", "onAfterMoveSelf") => {
            let t = tpoke.unwrap();
            let rd = crate::cond_id!(dex, "residualdmg").unwrap();
            let counter = b.poke(t).volatile(rd).map(|v| v.get_int(crate::state::DK::Counter)).unwrap_or(0);
            let maxhp = b.poke(t).maxhp as f64;
            let dmg = super::clamp_int_range((maxhp / 16.0).floor(), Some(1.0), None) * counter as f64;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "tox").unwrap());
            b.damage(dex, dmg, Some(t), Some(t), DamageEffect::Effect(eff), false);
            RV::Undef
        }
        ("tox", "onSwitchIn") => {
            let t = tpoke.unwrap();
            b.poke_mut(t).status = Status::Psn;
            b.refresh_poke_mask(dex, t);
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "psn", "[silent]"]);
            RV::Undef
        }
        ("tox", "onAfterSwitchInSelf") => {
            let t = tpoke.unwrap();
            let maxhp = b.poke(t).maxhp as f64;
            let dmg = super::clamp_int_range((maxhp / 16.0).floor(), Some(1.0), None);
            // this.damage(...) with implicit target/effect from the event
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "tox").unwrap());
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
                st.set_int(crate::state::DK::Time, time);
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
            let cid = crate::cond_id!(dex, "confusion").unwrap();
            let time = b.poke(t).volatile(cid).map(|v| v.get_int(crate::state::DK::Time)).unwrap_or(0) - 1;
            if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                vs.set_int(crate::state::DK::Time, time);
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
                .map(|m| m.move_type)
                .unwrap_or(dex.known_types.normal);
            let selfdestruct =
                b.active_move.as_ref().map(|m| m.selfdestruct).unwrap_or(false);
            let mut fake = super::moveexec::synthetic_move(dex, 40);
            fake.base_move_type = base_move_type;
            fake.is_confusion_self_hit = true;
            fake.no_damage_variance = true;
            fake.will_crit = Some(false);
            fake.selfdestruct = selfdestruct;
            let damage = super::moveexec::get_damage_synthetic(b, dex, t, t, fake);
            let Some(damage) = damage else {
                panic!("Confusion damage not dealt");
            };
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "confusion").unwrap());
            b.direct_damage(dex, damage as f64, Some(t), None, eff);
            RV::False
        }
        // ---------------------------------------------------------- flinch
        ("flinch", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "flinch"]);
            b.run_event(dex, &ev::Flinch, EvTarget::Poke(t), None, EffectHandle::None, None, false, false);
            RV::False
        }
        // ------------------------------------------------ partiallytrapped
        ("partiallytrapped", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            let src_move_name = b
                .state_at(state)
                .and_then(|s| s.source_effect)
                .map(|e| src_move_name(dex, &Some(e)))
                .unwrap_or_default();
            let of = format!("[of] {}", b.poke_str(source.unwrap()));
            let activate = format!("move: {src_move_name}");
            b.add(&["-activate", &ts, &activate, &of]);
            // gen5+ merged data: boundDivisor = bindingband ? 8 : 16 (no items in M1)
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::BoundDivisor, 16);
            }
            RV::Undef
        }
        ("partiallytrapped", "onResidual") => {
            let t = tpoke.unwrap();
            let (trapper, divisor, src_move) = {
                let st = b.state_at(state).unwrap();
                (st.source, st.get_int(crate::state::DK::BoundDivisor).max(1), st.source_effect)
            };
            let trapper_gone = match trapper {
                Some(tr_id) => {
                    let tp = b.poke(tr_id);
                    !tp.is_active || tp.hp <= 0 || tp.active_turns == 0
                }
                None => true,
            };
            if trapper_gone {
                let cid = crate::cond_id!(dex, "partiallytrapped").unwrap();
                b.poke_mut(t).volatiles.retain(|(k, _)| *k != cid);
                b.refresh_poke_mask(dex, t);
                let ts = b.poke_str(t);
                let name = src_move_name(dex, &src_move);
                b.add(&["-end", &ts, &name, "[partiallytrapped]", "[silent]"]);
                return RV::Undef;
            }
            let dmg = b.poke(t).base_maxhp as f64 / divisor as f64;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "partiallytrapped").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        ("partiallytrapped", "onEnd") => {
            let t = tpoke.unwrap();
            let src_move = b.state_at(state).and_then(|s| s.source_effect);
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
                b.try_trap(dex, t);
            }
            RV::Undef
        }
        // ----------------------------------------------------- residualdmg
        ("residualdmg", "onStart") => {
            let t = tpoke.unwrap();
            let cid = crate::cond_id!(dex, "residualdmg").unwrap();
            if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                vs.set_int(crate::state::DK::Counter, 0);
            }
            RV::Undef
        }
        ("residualdmg", "onAfterMoveSelf") | ("residualdmg", "onAfterSwitchInSelf") => {
            let t = tpoke.unwrap();
            if matches!(b.poke(t).status, Status::Brn | Status::Psn | Status::Tox) {
                let cid = crate::cond_id!(dex, "residualdmg").unwrap();
                if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                    let c = vs.get_int(crate::state::DK::Counter);
                    vs.set_int(crate::state::DK::Counter, c + 1);
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
            b.try_trap(dex, t);
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
            b.each_event(dex, &ev::Weather, None);
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
            b.each_event(dex, &ev::Weather, None);
            RV::Undef
        }
        ("sunnyday", "onFieldEnd") => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        ("sunnyday", "onImmunity") => {
            // onImmunity(type, pokemon): frz → false (can't freeze in sun)
            let t = tpoke.unwrap();
            if b.effective_weather(t) != crate::cond_id!(dex, "sunnyday") {
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
            if b.field.weather.is_some() && b.field.weather == crate::cond_id!(dex, "sandstorm") {
                b.each_event(dex, &ev::Weather, None);
            }
            RV::Undef
        }
        ("sandstorm", "onWeather") => {
            // gen2: this.damage(target.baseMaxhp / 8)
            let t = tpoke.unwrap();
            let dmg = b.poke(t).base_maxhp as f64 / 8.0;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "sandstorm").unwrap());
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
        ("sleepclausemod", "onSetStatus") => {
            // Unlike Stadium Sleep Clause, self-/ally-inflicted sleep (Rest)
            // does not engage the clause: only foe-sourced sleep counts.
            let t = tpoke.unwrap();
            if let Some(src) = source {
                if src.side == t.side {
                    return RV::Undef;
                }
            }
            if relay == RV::Str("slp".to_string()) {
                let side = t.side as usize;
                let foe_sourced_slp = b.sides[side]
                    .party
                    .iter()
                    .map(|&slot| &b.sides[side].roster[slot as usize])
                    .any(|p| {
                        p.hp > 0
                            && p.status == Status::Slp
                            && p.status_state
                                .source
                                .map(|s| s.side != t.side)
                                .unwrap_or(true)
                    });
                if foe_sourced_slp {
                    b.add(&["-message", "Sleep Clause Mod activated."]);
                    b.hint("Sleep Clause Mod prevents players from putting more than one of their opponent's Pok\u{e9}mon to sleep at a time", false);
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
        // --------------------------------------------- lightscreen/reflect
        ("lightscreen", "onSideStart") | ("reflect", "onSideStart") => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            let label = if cond == "lightscreen" { "move: Light Screen" } else { "Reflect" };
            b.add(&["-sidestart", &ss, label]);
            RV::Undef
        }
        ("lightscreen", "onSideEnd") | ("reflect", "onSideEnd") => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            let label = if cond == "lightscreen" { "move: Light Screen" } else { "Reflect" };
            b.add(&["-sideend", &ss, label]);
            RV::Undef
        }
        // ------------------------------------------------------- safeguard
        ("safeguard", "onSideStart") => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            b.add(&["-sidestart", &ss, "Safeguard"]);
            RV::Undef
        }
        ("safeguard", "onSideEnd") => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            b.add(&["-sideend", &ss, "Safeguard"]);
            RV::Undef
        }
        ("safeguard", "onSetStatus") | ("safeguard", "onTryAddVolatile") => {
            let t = tpoke.unwrap();
            let Some(src) = source else { return RV::Undef };
            if source_effect.is_none() {
                return RV::Undef;
            }
            let is_move = matches!(source_effect, EffectHandle::MoveEff(_));
            let infiltrates = b
                .active_move
                .as_ref()
                .map(|am| {
                    matches!(source_effect, EffectHandle::MoveEff(m) if am.id == Some(m))
                        && am.infiltrates
                })
                .unwrap_or(false);
            if is_move && infiltrates && t.side != src.side {
                return RV::Undef;
            }
            if cb == "onSetStatus" {
                if t != src {
                    if is_move && !move_eff_has_secondaries(b, dex, source_effect) {
                        let ts = b.poke_str(t);
                        b.add(&["-activate", &ts, "move: Safeguard"]);
                    }
                    return RV::Null;
                }
            } else {
                let is_confusion = relay == RV::Str("confusion".to_string());
                if is_confusion && t != src {
                    if is_move && !move_eff_has_secondaries(b, dex, source_effect) {
                        let ts = b.poke_str(t);
                        b.add(&["-activate", &ts, "move: Safeguard"]);
                    }
                    return RV::Null;
                }
            }
            RV::Undef
        }
        // ---------------------------------------------------------- spikes
        ("spikes", "onSideStart") => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            b.add(&["-sidestart", &ss, "Spikes"]);
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Layers, 1);
            }
            RV::Undef
        }
        ("spikes", "onEntryHazard") => {
            let t = tpoke.unwrap();
            if b.poke(t).has_type(dex.known_types.flying) {
                return RV::Undef;
            }
            let layers = b.state_at(state).map(|s| s.get_int(crate::state::DK::Layers)).unwrap_or(1);
            const AMOUNTS: [i64; 4] = [0, 3, 4, 6];
            let dmg = AMOUNTS[layers.clamp(0, 3) as usize] as f64 * b.poke(t).maxhp as f64 / 24.0;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "spikes").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        // ------------------------------------------------------- leechseed
        ("leechseed", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "move: Leech Seed"]);
            RV::Undef
        }
        ("leechseed", "onAfterMoveSelf") => {
            let t = tpoke.unwrap();
            if b.poke(t).hp <= 0 {
                return RV::Undef;
            }
            let leecher = b
                .state_at(state)
                .and_then(|s| s.source_slot)
                .and_then(|slot| b.poke_at_slot_pos(slot));
            let Some(leecher) = leecher else { return RV::Undef };
            if b.poke(leecher).fainted || b.poke(leecher).hp <= 0 {
                return RV::Undef;
            }
            let to_leech =
                super::clamp_int_range(b.poke(t).maxhp as f64 / 8.0, Some(1.0), None);
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "leechseed").unwrap());
            let dealt = b.damage(dex, to_leech, Some(t), Some(leecher), DamageEffect::Effect(eff), false);
            if let Some(d) = dealt {
                if d != 0.0 {
                    b.heal(dex, d, Some(leecher), Some(t), super::dmg::HealEffect::Effect(eff));
                }
            }
            RV::Undef
        }
        // ------------------------------------------------------- nightmare
        ("nightmare", "onStart") => {
            let t = tpoke.unwrap();
            if b.poke(t).status != Status::Slp {
                return RV::False;
            }
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Nightmare"]);
            RV::Undef
        }
        ("nightmare", "onAfterMoveSelf") => {
            let t = tpoke.unwrap();
            if b.poke(t).status == Status::Slp {
                let dmg = b.poke(t).base_maxhp as f64 / 4.0;
                let eff = EffectHandle::Cond(crate::cond_id!(dex, "nightmare").unwrap());
                b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            }
            RV::Undef
        }
        // ----------------------------------------------------------- curse
        ("curse", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            let of = format!("[of] {}", b.poke_str(source.unwrap()));
            b.add(&["-start", &ts, "Curse", &of]);
            RV::Undef
        }
        ("curse", "onAfterMoveSelf") => {
            let t = tpoke.unwrap();
            let dmg = b.poke(t).base_maxhp as f64 / 4.0;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "curse").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        // ------------------------------------------------------- foresight
        ("foresight", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Foresight"]);
            RV::Undef
        }
        ("foresight", "onNegateImmunity") => {
            let t = tpoke.unwrap();
            if b.poke(t).has_type(dex.known_types.ghost) {
                if let RV::Num(tyv) = relay {
                    let ty = crate::dex::TypeId(tyv as u8);
                    if ty == dex.known_types.normal || ty == dex.known_types.fighting {
                        return RV::False;
                    }
                }
            }
            RV::Undef
        }
        // ---------------------------------------------------------- lockon
        ("lockon", "onSourceAccuracy") => {
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            let locked_onto = b.state_at(state).and_then(|s| s.source);
            if source == Some(holder) && tpoke.is_some() && tpoke == locked_onto {
                return RV::True;
            }
            RV::Undef
        }
        // ----------------------------------------------------- destinybond
        ("destinybond", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singlemove", &ts, "Destiny Bond"]);
            RV::Undef
        }
        ("destinybond", "onFaint") => {
            let t = tpoke.unwrap();
            let Some(src) = source else { return RV::Undef };
            if source_effect.is_none() || src.side == t.side {
                return RV::Undef;
            }
            if let EffectHandle::MoveEff(m) = source_effect {
                if !dex.move_static(m).has_flag(dex, "futuremove") {
                    let ts = b.poke_str(t);
                    b.add(&["-activate", &ts, "move: Destiny Bond"]);
                    b.pokemon_faint(src, None, EffectHandle::None);
                }
            }
            RV::Undef
        }
        ("destinybond", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let is_db = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(|m| dex.moves.key(m) == "destinybond")
                .unwrap_or(false);
            if !is_db {
                b.remove_volatile(dex, t, "destinybond");
            }
            RV::Undef
        }
        ("destinybond", "onMoveAborted") => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "destinybond");
            RV::Undef
        }
        // gen2stadium2nc2000: Destiny Bond protects only until the foe next
        // acts — expire after the foe's own move, and on foe switch-out.
        ("destinybond", "onFoeAfterMoveSelf") => {
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            if b.poke(holder).hp > 0 {
                b.remove_volatile(dex, holder, "destinybond");
            }
            RV::Undef
        }
        ("destinybond", "onFoeSwitchOut") => {
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            b.remove_volatile(dex, holder, "destinybond");
            RV::Undef
        }
        // ------------------------------------------------------ perishsong
        ("perishsong", "onResidual") => {
            let t = tpoke.unwrap();
            let dur = b.state_at(state).and_then(|s| s.duration).unwrap_or(0);
            let ts = b.poke_str(t);
            let tag = format!("perish{dur}");
            b.add(&["-start", &ts, &tag]);
            RV::Undef
        }
        ("perishsong", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "perish0"]);
            b.pokemon_faint(t, None, EffectHandle::None);
            RV::Undef
        }
        // ---------------------------------------------------------- encore
        ("encore", "onStart") => {
            let t = tpoke.unwrap();
            let locked = b.poke(t).last_move_encore;
            let Some(locked) = locked else { return RV::False };
            let has_slot = b.poke(t).get_move_slot(locked).map(|s| s.pp).unwrap_or(0) > 0;
            if dex.move_static(locked).has_flag(dex, "failencore") || !has_slot {
                return RV::False;
            }
            let locked_key = dex.moves.key(locked).to_string();
            if let Some(st) = b.state_at_mut(state) {
                st.set(crate::state::DK::Move, Scalar::MoveK(locked));
            }
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Encore"]);
            if locked_key == "pursuit" {
                b.add_volatile(dex, t, "pursuit", Some(t), EffectHandle::MoveEff(locked));
            }
            RV::Undef
        }
        ("encore", "onOverrideAction") => {
            let move_key = match source_effect {
                EffectHandle::MoveEff(m) => dex.moves.key(m).to_string(),
                _ => String::new(),
            };
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            if !locked.is_empty() && move_key != locked {
                return RV::Str(locked);
            }
            RV::Undef
        }
        ("encore", "onResidual") => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            let pp_left = dex
                .moves
                .id(&locked)
                .and_then(|mid| b.poke(t).get_move_slot(mid).map(|s| s.pp))
                .unwrap_or(0);
            if pp_left <= 0 {
                b.remove_volatile(dex, t, "encore");
            }
            RV::Undef
        }
        ("encore", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Encore"]);
            RV::Undef
        }
        ("encore", "onDisableMove") => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            let Some(mid) = dex.moves.id(&locked) else { return RV::Undef };
            if b.poke(t).get_move_slot(mid).is_none() {
                return RV::Undef;
            }
            let others: Vec<crate::dex::MoveId> = b
                .poke(t)
                .move_slots
                .iter()
                .map(|s| s.id)
                .filter(|&id| id != mid)
                .collect();
            for other in others {
                b.pokemon_disable_move(t, other);
            }
            RV::Undef
        }
        // --------------------------------------------------------- disable
        ("disable", "onStart") => {
            let t = tpoke.unwrap();
            if !b.queue_will_move(t) {
                if let Some(st) = b.state_at_mut(state) {
                    if let Some(d) = st.duration {
                        st.duration = Some(d + 1);
                    }
                }
            }
            let Some(last) = b.poke(t).last_move else { return RV::False };
            let Some(slot) = b.poke(t).get_move_slot(last) else { return RV::False };
            if slot.pp == 0 {
                return RV::False;
            }
            let ts = b.poke_str(t);
            let name = dex.move_static(last).name.clone();
            b.add(&["-start", &ts, "Disable", &name]);
            if let Some(st) = b.state_at_mut(state) {
                st.set(crate::state::DK::Move, Scalar::MoveK(last));
            }
            RV::Undef
        }
        ("disable", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Disable"]);
            RV::Undef
        }
        ("disable", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            let cur = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            if !locked.is_empty() && cur == locked {
                let ts = b.poke_str(t);
                let name = b.active_move_name(dex);
                b.add(&["cant", &ts, "Disable", &name]);
                return RV::False;
            }
            RV::Undef
        }
        ("disable", "onDisableMove") => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            if let Some(mid) = dex.moves.id(&locked) {
                if b.poke(t).get_move_slot(mid).is_some() {
                    b.pokemon_disable_move(t, mid);
                }
            }
            RV::Undef
        }
        // ------------------------------------------------------------ mist
        ("mist", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Mist"]);
            RV::Undef
        }
        ("mist", "onTryBoost") => {
            let t = tpoke.unwrap();
            let Some(src) = source else { return RV::Undef };
            if t == src {
                return RV::Undef;
            }
            let mut show_msg = false;
            if let Some(table) = b.pending_boosts.as_mut() {
                let before = table.len();
                table.retain(|&(_, amt)| amt >= 0);
                show_msg = table.len() != before;
            }
            if show_msg && !move_eff_has_secondaries(b, dex, source_effect) {
                let ts = b.poke_str(t);
                b.add(&["-activate", &ts, "move: Mist"]);
            }
            RV::Undef
        }
        // ------------------------------------------------------ substitute
        ("substitute", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Substitute"]);
            let hp = (b.poke(t).maxhp as f64 / 4.0).floor() as i64;
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Hp, hp);
            }
            let pt = crate::cond_id!(dex, "partiallytrapped").unwrap();
            if b.poke(t).has_volatile(pt) {
                let src_move = b.poke(t).volatile(pt).and_then(|v| v.source_effect);
                let name = src_move_name(dex, &src_move);
                let ts = b.poke_str(t);
                b.add(&["-end", &ts, &name, "[partiallytrapped]", "[silent]"]);
                b.poke_mut(t).volatiles.retain(|(k, _)| *k != pt);
                b.refresh_poke_mask(dex, t);
            }
            RV::Undef
        }
        ("substitute", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Substitute"]);
            RV::Undef
        }
        ("substitute", "onTryPrimaryHit") => {
            let t = tpoke.unwrap();
            let src = source.unwrap();
            let (stalling, is_drain, category, move_key, has_status, has_boosts, vs, has_recoil) = {
                let am = b.active_move.as_ref().unwrap();
                (
                    am.stalling_move,
                    am.drain.is_some(),
                    am.category,
                    am.id.map(|m| dex.moves.key(m).to_string()).unwrap_or_default(),
                    am.status.is_some(),
                    am.has_boosts,
                    am.volatile_status.clone(),
                    am.recoil.is_some(),
                )
            };
            if stalling {
                let ss = b.poke_str(src);
                b.add(&["-fail", &ss]);
                return RV::Null;
            }
            if t == src {
                return RV::Undef;
            }
            if move_key == "twineedle" {
                if let Some(am) = b.active_move.as_mut() {
                    am.secondaries.retain(|s| !s.kingsrock);
                }
            }
            if is_drain {
                let ss = b.poke_str(src);
                b.add(&["-miss", &ss]);
                b.hint("In Gen 2, draining moves always miss against Substitute.", false);
                return RV::Null;
            }
            if category == crate::dex::Category::Status {
                const SUB_BLOCKED: [&str; 6] =
                    ["leechseed", "lockon", "mindreader", "nightmare", "painsplit", "sketch"];
                let mut vs = vs;
                if move_key == "swagger" {
                    if let Some(am) = b.active_move.as_mut() {
                        am.volatile_status = None;
                    }
                    vs = None;
                }
                if has_status
                    || (has_boosts && move_key != "swagger")
                    || vs.as_deref() == Some("confusion")
                    || SUB_BLOCKED.contains(&move_key.as_str())
                {
                    let ts = b.poke_str(t);
                    let name = b.active_move_name(dex);
                    let block = format!("[block] {name}");
                    b.add(&["-activate", &ts, "Substitute", &block]);
                    return RV::Null;
                }
                return RV::Undef;
            }
            let calc = b.get_damage(dex, src, t, false);
            let mut damage = match calc {
                super::moveexec::DamageResult::Damage(d) if d > 0.0 => d,
                _ => return RV::Null,
            };
            let sub = crate::cond_id!(dex, "substitute").unwrap();
            let sub_hp = b.poke(t).volatile(sub).map(|v| v.get_int(crate::state::DK::Hp)).unwrap_or(0);
            if damage > sub_hp as f64 {
                damage = sub_hp as f64;
            }
            if let Some(vsq) = b.poke_mut(t).volatile_mut(sub) {
                let hp = vsq.get_int(crate::state::DK::Hp);
                vsq.set_int(crate::state::DK::Hp, hp - damage as i64);
            }
            b.poke_mut(src).last_damage = damage as i64;
            let left = b.poke(t).volatile(sub).map(|v| v.get_int(crate::state::DK::Hp)).unwrap_or(0);
            if left <= 0 {
                b.remove_volatile(dex, t, "substitute");
            } else {
                let ts = b.poke_str(t);
                b.add(&["-activate", &ts, "Substitute", "[damage]"]);
            }
            if has_recoil {
                b.damage(dex, 1.0, Some(src), Some(t), DamageEffect::Recoil, false);
            }
            let move_eff = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(EffectHandle::MoveEff)
                .unwrap_or(EffectHandle::None);
            b.run_event(
                dex,
                &ev::AfterSubDamage,
                EvTarget::Poke(t),
                Some(src),
                move_eff,
                Some(RV::Num(damage)),
                false,
                false,
            );
            RV::Num(0.0) // HIT_SUBSTITUTE
        }
        // ------------------------------------------------------------ bide
        ("bide", "onStart") => {
            let t = tpoke.unwrap();
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::TotalDamage, 0);
            }
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "move: Bide"]);
            RV::Undef
        }
        ("bide", "onDamage") => {
            if !matches!(source_effect, EffectHandle::MoveEff(_)) || source.is_none() {
                return RV::Undef;
            }
            let dmg = relay.as_num() as i64;
            if let Some(st) = b.state_at_mut(state) {
                let cur = st.get_int(crate::state::DK::TotalDamage);
                st.set_int(crate::state::DK::TotalDamage, cur + dmg);
                st.last_damage_source = source;
            }
            RV::Undef
        }
        ("bide", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let (duration, total, last_src) = {
                let st = b.state_at(state).unwrap();
                (st.duration.unwrap_or(0), st.get_int(crate::state::DK::TotalDamage), st.last_damage_source)
            };
            if duration == 1 {
                let ts = b.poke_str(t);
                b.add(&["-end", &ts, "move: Bide"]);
                if total == 0 {
                    let ts = b.poke_str(t);
                    b.add(&["-fail", &ts]);
                    return RV::False;
                }
                let mut bide_target = match last_src {
                    Some(s) => s,
                    None => {
                        let ts = b.poke_str(t);
                        b.add(&["-fail", &ts]);
                        return RV::False;
                    }
                };
                if !b.poke(bide_target).is_active {
                    match b.get_random_target("normal", t) {
                        Some(nt) => bide_target = nt,
                        None => {
                            let ts = b.poke_str(t);
                            b.add(&["-miss", &ts]);
                            return RV::False;
                        }
                    }
                }
                // synthetic 'bide' unleash move
                let bide_id = dex.moves.id("bide").unwrap();
                let mut fake = super::moveexec::get_active_move(dex, bide_id);
                fake.accuracy = crate::dex::Accuracy::Pct(100);
                fake.base_power = 0;
                fake.damage = Some(crate::dex::FixedDamage::Amount((total * 2) as i32));
                fake.category = crate::dex::Category::Physical;
                fake.move_type = dex.known_types.normal;
                fake.crit_ratio = 0;
                fake.will_crit = None;
                fake.priority = 0;
                fake.target = "normal".into();
                fake.volatile_status = None;
                fake.secondaries = Vec::new();
                fake.self_effect = None;
                fake.flags = dex.flag_bit("contact") | dex.flag_bit("protect");
                fake.cb_mask = crate::dex::CbMask::EMPTY;
                // the synthetic moveData has no ignoreImmunity: Physical → false
                fake.ignore_immunity = false;
                let saved_move = b.active_move.take();
                let saved_pokemon = b.active_pokemon;
                let saved_target = b.active_target;
                b.set_active_move(fake, Some(t), Some(bide_target));
                b.try_move_hit(dex, bide_target, t);
                b.active_move = saved_move;
                b.active_pokemon = saved_pokemon;
                b.active_target = saved_target;
                b.remove_volatile(dex, t, "bide");
                return RV::False;
            }
            let ts = b.poke_str(t);
            b.add(&["-activate", &ts, "move: Bide"]);
            RV::Undef
        }
        ("bide", "onMoveAborted") => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "bide");
            RV::Undef
        }
        ("bide", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "move: Bide", "[silent]"]);
            RV::Undef
        }
        ("bide", "onSemiLockMove") => RV::Str("bide".to_string()),
        // ------------------------------------------------------------ rage
        ("rage", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singlemove", &ts, "Rage"]);
            RV::Undef
        }
        ("rage", "onHit") => {
            let t = tpoke.unwrap();
            let category = b.active_move.as_ref().map(|m| m.category);
            if source != Some(t) && category != Some(crate::dex::Category::Status) {
                let eff = EffectHandle::Cond(crate::cond_id!(dex, "rage").unwrap());
                b.boost(dex, &[(0, 1)], Some(t), source, eff);
            }
            RV::Undef
        }
        ("rage", "onBeforeMove") => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "rage");
            RV::Undef
        }
        // --------------------------------------------------------- rollout
        ("rollout", "onStart") => {
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::HitCount, 0);
                st.set_int(crate::state::DK::ContactHitCount, 0);
            }
            RV::Undef
        }
        ("rollout", "onResidual") => {
            let t = tpoke.unwrap();
            let is_struggle = b
                .poke(t)
                .last_move
                .map(|lm| dex.moves.key(lm) == "struggle")
                .unwrap_or(false);
            if is_struggle {
                let ro = crate::cond_id!(dex, "rollout").unwrap();
                b.poke_mut(t).volatiles.retain(|(k, _)| *k != ro);
                b.refresh_poke_mask(dex, t);
            }
            RV::Undef
        }
        ("rollout", "onLockMove") => RV::Str("rollout".to_string()),
        // ------------------------------------------------------ furycutter
        ("furycutter", "onStart") => {
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Multiplier, 1);
            }
            RV::Undef
        }
        ("furycutter", "onRestart") => {
            if let Some(st) = b.state_at_mut(state) {
                let m = st.get_int(crate::state::DK::Multiplier);
                if m < 16 {
                    st.set_int(crate::state::DK::Multiplier, m << 1);
                }
                st.duration = Some(2);
            }
            RV::Undef
        }
        // ------------------------------------------- defensecurl/minimize
        ("defensecurl", "onRestart") | ("minimize", "onRestart") => RV::Null,
        ("minimize", "onSourceModifyDamage") => {
            let has_flag = b
                .active_move
                .as_ref()
                .map(|m| m.has_flag(dex, "minimize"))
                .unwrap_or(false);
            if has_flag {
                b.chain_modify(2.0, 1.0);
            }
            RV::Undef
        }
        // --------------------------------------------------------- attract
        ("attract", "onStart") => {
            let t = tpoke.unwrap();
            let src = source.unwrap();
            let tg = b.poke(t).gender;
            let sg = b.poke(src).gender;
            if !((tg == Gender::M && sg == Gender::F) || (tg == Gender::F && sg == Gender::M)) {
                return RV::False;
            }
            if !b
                .run_event(dex, &ev::Attract, EvTarget::Poke(t), Some(src), EffectHandle::None, None, false, false)
                .truthy()
            {
                return RV::False;
            }
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Attract"]);
            RV::Undef
        }
        ("attract", "onUpdate") => {
            let t = tpoke.unwrap();
            let src = b.state_at(state).and_then(|s| s.source);
            if let Some(src) = src {
                if !b.poke(src).is_active {
                    b.remove_volatile(dex, t, "attract");
                }
            }
            RV::Undef
        }
        ("attract", "onBeforeMove") => {
            let t = tpoke.unwrap();
            let src = b.state_at(state).and_then(|s| s.source);
            let ts = b.poke_str(t);
            let of = format!("[of] {}", src.map(|s| b.poke_str(s)).unwrap_or_default());
            b.add(&["-activate", &ts, "move: Attract", &of]);
            if b.prng.random_chance(1, 2) {
                let ts = b.poke_str(t);
                b.add(&["cant", &ts, "Attract"]);
                return RV::False;
            }
            RV::Undef
        }
        ("attract", "onEnd") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Attract", "[silent]"]);
            RV::Undef
        }
        // ----------------------------------------------------- focusenergy
        ("focusenergy", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            let silent = matches!(
                source_effect,
                EffectHandle::MoveEff(m) if matches!(dex.moves.key(m), "psychup" | "transform")
            );
            if silent {
                b.add(&["-start", &ts, "move: Focus Energy", "[silent]"]);
            } else {
                b.add(&["-start", &ts, "move: Focus Energy"]);
            }
            RV::Undef
        }
        ("focusenergy", "onModifyCritRatio") => RV::Num(relay.as_num() + 1.0),
        // --------------------------------------------------------- protect
        ("protect", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singleturn", &ts, "Protect"]);
            RV::Undef
        }
        ("protect", "onTryHit") => {
            let t = tpoke.unwrap();
            let has_protect_flag = b
                .active_move
                .as_ref()
                .map(|m| m.has_flag(dex, "protect"))
                .unwrap_or(false);
            if !has_protect_flag {
                return RV::Undef; // move bypasses protect
            }
            let src = source.unwrap();
            if !b
                .run_event(dex, &ev::HitProtect, EvTarget::Poke(src), tpoke, EffectHandle::None, None, false, false)
                .truthy()
            {
                return RV::Undef;
            }
            let ts = b.poke_str(t);
            b.add(&["-activate", &ts, "Protect"]);
            RV::Null
        }
        // ---------------------------------------------------------- endure
        ("endure", "onStart") => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singleturn", &ts, "move: Endure"]);
            RV::Undef
        }
        ("endure", "onDamage") => {
            let t = tpoke.unwrap();
            if matches!(source_effect, EffectHandle::MoveEff(_)) {
                let damage = relay.as_num();
                if damage >= b.poke(t).hp as f64 {
                    let ts = b.poke_str(t);
                    b.add(&["-activate", &ts, "move: Endure"]);
                    return RV::Num(b.poke(t).hp as f64 - 1.0);
                }
            }
            RV::Undef
        }
        // ----------------------------------------------------------- stall
        ("stall", "onStart") => {
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Counter, 127);
            }
            RV::Undef
        }
        ("stall", "onStallMove") => {
            let counter = b
                .state_at(state)
                .and_then(|s| s.get(crate::state::DK::Counter).cloned())
                .map(|v| v.as_f64())
                .unwrap_or(0.0);
            let mut c = counter.floor() as u32;
            if c == 0 {
                c = 127;
            }
            RV::from_bool(b.prng.random_chance(c, 255))
        }
        ("stall", "onRestart") => {
            if let Some(st) = b.state_at_mut(state) {
                let c = st.get(crate::state::DK::Counter).map(|v| v.as_f64()).unwrap_or(127.0);
                let half = c / 2.0;
                if half.fract() == 0.0 {
                    st.set(crate::state::DK::Counter, Scalar::Int(half as i64));
                } else {
                    st.set(crate::state::DK::Counter, Scalar::Float(half));
                }
                st.duration = Some(2);
            }
            RV::Undef
        }
        // ----------------------------------------------------- twoturnmove
        ("twoturnmove", "onStart") => {
            let t = tpoke.unwrap();
            let EffectHandle::MoveEff(mv) = source_effect else { return RV::Undef };
            let move_key = dex.moves.key(mv).to_string();
            if let Some(st) = b.state_at_mut(state) {
                st.set(crate::state::DK::Move, Scalar::MoveK(mv));
            }
            b.add_volatile(dex, t, &move_key, None, EffectHandle::None);
            // gen2 runMove's moveUsed() never passes targetLoc, so this is
            // None for directly-used charge moves...
            let mut move_target_loc = b.poke(t).last_move_target_loc;
            // ...but metronome/mirror-move-called charge moves (sourceEffect
            // set) retarget explicitly.
            let has_source_effect = b
                .active_move
                .as_ref()
                .map(|am| am.source_effect.is_some())
                .unwrap_or(false);
            if has_source_effect && dex.move_static(mv).target != "self" {
                let mut defender = source;
                if defender.map(|d| b.poke(d).fainted).unwrap_or(false) {
                    // this.sample(attacker.foes(true)) — consumes PRNG
                    let foes = b.foes_of(t, true);
                    if !foes.is_empty() {
                        defender = Some(foes[b.prng.sample_index(foes.len())]);
                    }
                }
                if let Some(d) = defender {
                    move_target_loc = Some(if d.side == t.side { -1 } else { 1 });
                }
            }
            let mvc = dex.conds_id(&move_key).unwrap();
            if let Some(loc) = move_target_loc {
                if let Some(vs) = b.poke_mut(t).volatile_mut(mvc) {
                    vs.set_int(crate::state::DK::TargetLoc, loc as i64);
                }
            }
            b.attr_last_move(&["[still]"]);
            b.run_event(dex, &ev::PrepareHit, EvTarget::Poke(t), source, source_effect, None, false, false);
            RV::Undef
        }
        ("twoturnmove", "onEnd") => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            if !locked.is_empty() {
                b.remove_volatile(dex, t, &locked);
            }
            RV::Undef
        }
        ("twoturnmove", "onLockMove") => {
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            RV::Str(locked)
        }
        ("twoturnmove", "onMoveAborted") => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "twoturnmove");
            RV::Undef
        }
        // ---------------------------------------------- dig / fly volatiles
        ("dig", "onImmunity") => {
            if relay == RV::Str("sandstorm".to_string()) || relay == RV::Str("hail".to_string()) {
                return RV::False;
            }
            RV::Undef
        }
        ("dig", "onInvulnerability") | ("fly", "onInvulnerability") => {
            let move_key = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            if cond == "dig" {
                if matches!(move_key.as_str(), "earthquake" | "magnitude" | "fissure") {
                    return RV::Undef;
                }
            } else {
                if matches!(move_key.as_str(), "gust" | "twister" | "thunder" | "whirlwind") {
                    return RV::Undef;
                }
                if matches!(move_key.as_str(), "earthquake" | "magnitude" | "fissure") {
                    return RV::False;
                }
            }
            if matches!(
                move_key.as_str(),
                "attract" | "curse" | "foresight" | "meanlook" | "mimic" | "nightmare"
                    | "spiderweb" | "transform"
            ) {
                return RV::False;
            }
            // lock-on: source has lockon volatile targeting this pokemon
            if let (Some(t), Some(src)) = (tpoke, source) {
                let lo = crate::cond_id!(dex, "lockon").unwrap();
                let locked = b
                    .poke(src)
                    .volatile(lo)
                    .and_then(|v| v.source)
                    .map(|locked| locked == t)
                    .unwrap_or(false);
                if b.poke(src).has_volatile(lo) && locked {
                    return RV::Undef;
                }
            }
            RV::False
        }
        ("dig", "onSourceBasePower") | ("fly", "onSourceBasePower") => {
            let move_key = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            let doubled = if cond == "dig" {
                matches!(move_key.as_str(), "earthquake" | "magnitude")
            } else {
                matches!(move_key.as_str(), "gust" | "twister")
            };
            if doubled {
                b.chain_modify(2.0, 1.0);
            }
            RV::Undef
        }
        // ------------------------------------------------------ lockedmove
        ("lockedmove", "onStart") => {
            let val = match source_effect {
                EffectHandle::MoveEff(m) => Some(Scalar::MoveK(m)),
                EffectHandle::Cond(c) => Some(Scalar::CondK(c)),
                _ => None,
            };
            if let (Some(v), Some(st)) = (val, b.state_at_mut(state)) {
                st.set(crate::state::DK::Move, v);
            }
            RV::Undef
        }
        ("lockedmove", "onResidual") => {
            let t = tpoke.unwrap();
            let is_struggle = b
                .poke(t)
                .last_move
                .map(|lm| dex.moves.key(lm) == "struggle")
                .unwrap_or(false);
            if is_struggle || b.poke(t).status == Status::Slp {
                // direct delete (no End event → no confusion)
                let lm = crate::cond_id!(dex, "lockedmove").unwrap();
                b.poke_mut(t).volatiles.retain(|(k, _)| *k != lm);
                b.refresh_poke_mask(dex, t);
            }
            RV::Undef
        }
        ("lockedmove", "onEnd") => {
            let t = tpoke.unwrap();
            // silently delete confusion, then re-add unless safeguard
            let conf = crate::cond_id!(dex, "confusion").unwrap();
            b.poke_mut(t).volatiles.retain(|(k, _)| *k != conf);
            b.refresh_poke_mask(dex, t);
            let sg = crate::cond_id!(dex, "safeguard").unwrap();
            if !b.sides[t.side as usize].has_side_condition(sg) {
                let lm_eff = EffectHandle::Cond(crate::cond_id!(dex, "lockedmove").unwrap());
                b.add_volatile(dex, t, "confusion", None, lm_eff);
            }
            RV::Undef
        }
        ("lockedmove", "onLockMove") => {
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m).to_string())
                .unwrap_or_default();
            RV::Str(locked)
        }
        ("lockedmove", "onMoveAborted") => {
            let t = tpoke.unwrap();
            let lm = crate::cond_id!(dex, "lockedmove").unwrap();
            b.poke_mut(t).volatiles.retain(|(k, _)| *k != lm);
            b.refresh_poke_mask(dex, t);
            RV::Undef
        }
        // --------------------------------------------------------- pursuit
        ("pursuit", "onFoeBeforeSwitchOut") => {
            let switching = tpoke.unwrap();
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            let (target_loc, src_effect) = {
                let st = b.state_at(state).unwrap();
                (st.get_int(crate::state::DK::TargetLoc), st.source_effect)
            };
            let expected_loc: i64 = if switching.side == holder.side { -1 } else { 1 };
            if target_loc != expected_loc {
                return RV::Undef;
            }
            if b.poke(holder).hp <= 0 {
                return RV::Undef;
            }
            let en = crate::cond_id!(dex, "encore").unwrap();
            if let Some(enc) = b.poke(holder).volatile(en) {
                if enc.get_move() != dex.moves.id("pursuit") {
                    return RV::Undef;
                }
            }
            if !b.queue_cancel_move(holder) {
                return RV::Undef;
            }
            let holder_speed = b.poke(holder).speed;
            let switching_speed = b.poke(switching).speed;
            if holder_speed < switching_speed
                || (holder_speed == switching_speed && b.prng.random_chance(1, 2))
            {
                b.remove_volatile(dex, switching, "destinybond");
            }
            let pursuit_id = dex.moves.id("pursuit").unwrap();
            let se = src_effect.and_then(|e| match e {
                EffectHandle::MoveEff(m) => Some(m),
                _ => None,
            });
            b.run_move(dex, pursuit_id, holder, expected_loc as i8, se);
            RV::Undef
        }
        // ------------------------------------------------------ futuremove
        ("futuremove", "onStart") => {
            let t = tpoke.unwrap();
            let slot = b.slot_of(t);
            let ending = b.turn as i64 + 1;
            if let Some(st) = b.state_at_mut(state) {
                st.set(crate::state::DK::TargetSlot, Scalar::Slot(slot.0, slot.1));
                st.set_int(crate::state::DK::EndingTurn, ending);
            }
            RV::Undef
        }
        ("futuremove", "onResidual") => {
            let (ending, slot) = {
                let st = b.state_at(state).unwrap();
                (
                    st.get_int(crate::state::DK::EndingTurn),
                    st.get(crate::state::DK::TargetSlot).and_then(|v| match v {
                        Scalar::Slot(a, b) => Some((*a, *b)),
                        _ => None,
                    }),
                )
            };
            // getOverflowedTurnCount() = turn - 1 for gen < 8
            if (b.turn as i64 - 1) < ending {
                return RV::Undef;
            }
            if let Some(at_slot) = slot.and_then(|sl| b.poke_at_slot_pos(sl)) {
                let fm = crate::cond_id!(dex, "futuremove").unwrap();
                b.remove_slot_condition(dex, at_slot, fm);
            }
            RV::Undef
        }
        ("futuremove", "onEnd") => {
            let t = tpoke.unwrap();
            super::moveexec::resolve_future_move(b, dex, state, t);
            RV::Undef
        }
        // ---------------------------------------------------------- trapper
        ("trapper", _) => RV::Undef,
        // ------------------------------------------------- marker volatiles
        ("brnattackdrop", _) | ("parspeeddrop", _) | ("leppaberry", _) => RV::Undef,
        _ => panic!("unported condition callback: {cond} {cb}"),
    }
}

/// Whether the effect (a move) has secondaries — mist/safeguard message gate.
fn move_eff_has_secondaries(b: &Battle, dex: &Dex, effect: EffectHandle) -> bool {
    match effect {
        EffectHandle::MoveEff(m) => {
            if let Some(am) = &b.active_move {
                if am.id == Some(m) {
                    return !am.secondaries.is_empty();
                }
            }
            !dex.move_static(m).secondaries.is_empty()
        }
        _ => false,
    }
}

/// `${effect}` (Effect.toString()) — the plain name ("Wrap").
fn src_move_name(dex: &Dex, src_move: &Option<EffectHandle>) -> String {
    match src_move {
        Some(EffectHandle::MoveEff(m)) => dex.move_static(*m).name.clone(),
        Some(EffectHandle::Cond(c)) => dex.conds_key(*c).to_string(),
        _ => String::new(),
    }
}

/// The shared gen2 residualdmg helper (brn/psn tick).
fn residualdmg(b: &mut Battle, dex: &Dex, pokemon: PokeId) {
    let rd = crate::cond_id!(dex, "residualdmg").unwrap();
    let status = b.poke(pokemon).status;
    let eff = EffectHandle::Cond(
        dex.conds_id(status.as_str()).unwrap_or_else(|| crate::cond_id!(dex, "brn").unwrap()),
    );
    if b.poke(pokemon).has_volatile(rd) {
        let counter = b.poke(pokemon).volatile(rd).map(|v| v.get_int(crate::state::DK::Counter)).unwrap_or(0);
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
            DamageEffect::Recoil => EffectHandle::Cond(crate::cond_id!(dex, "recoil").unwrap()),
            DamageEffect::Drain => EffectHandle::Cond(crate::cond_id!(dex, "drain").unwrap()),
        }
    }
}

/// Convenience for moveexec: move id by name key.
pub fn move_id(dex: &Dex, key: &str) -> Option<MoveId> {
    dex.moves.id(key)
}
