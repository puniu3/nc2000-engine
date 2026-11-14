use crate::dex::{Cb, CondCbOp, CondId, Dex, DurationCbKind, MoveId};
use crate::state::*;

use super::events::{ev, EvTarget};
use super::{EffectHandle, RV};

pub fn has_builtin(cond: &str, callback: &str) -> bool {
    matches!(
        (cond, callback),
        ("mustrecharge", "onLockMove") | ("rollout", "onLockMove") | ("bide", "onSemiLockMove")
    )
}

#[inline]
pub fn eval_duration_cb(prng: &mut crate::prng::BattleRng, kind: DurationCbKind) -> Option<i32> {
    match kind {
        DurationCbKind::None => None,
        DurationCbKind::Rand3To6 => Some(prng.random_range(3, 6) as i32),
        DurationCbKind::Rand2To4 => Some(prng.random_range(2, 4) as i32),
        DurationCbKind::Rand3To5 => Some(prng.random_range(3, 5) as i32),
        DurationCbKind::Rand2To6 => Some(prng.random_range(2, 6) as i32),
        DurationCbKind::Rand3To7 => Some(prng.random_range(3, 7) as i32),
        DurationCbKind::Const5 => Some(5),
    }
}

#[cold]
fn unported_cond_cb(dex: &Dex, cond: CondId, cb: Cb) -> ! {
    panic!("unported condition callback: {} {}", dex.conds_key(cond), dex.cb_key(cb))
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
    match effect {
        EffectHandle::Cond(c) => {
            dispatch_cond(b, dex, c, cb, state, target, source, source_effect, relay)
        }
        EffectHandle::MoveEff(m) => {
            super::moveexec::dispatch_move_callback(b, dex, m, cb, target, source, relay)
        }
        EffectHandle::Item(i) => {
            super::items::dispatch_item(b, dex, i, cb, target, source, source_effect, relay)
        }
        _ => RV::Undef,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_cond(
    b: &mut Battle,
    dex: &Dex,
    cond: CondId,
    cb: Cb,
    state: StateLoc,
    target: EvTarget,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
) -> RV {
    let tpoke = target.poke();
    match dex.cond(cond).cb_ops[cb.0 as usize] {
        // ------------------------------------------------------------- brn
        CondCbOp::BrnStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "brn"]);
            b.add_volatile(dex, t, "brnattackdrop", None, EffectHandle::None);
            RV::Undef
        }
        CondCbOp::BrnAfterMoveSelf => {
            residualdmg(b, dex, tpoke.unwrap());
            RV::Undef
        }
        CondCbOp::BrnSwitchIn => {
            let t = tpoke.unwrap();
            b.add_volatile(dex, t, "brnattackdrop", None, EffectHandle::None);
            RV::Undef
        }
        // ------------------------------------------------------------- par
        CondCbOp::ParStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "par"]);
            b.add_volatile(dex, t, "parspeeddrop", None, EffectHandle::None);
            RV::Undef
        }
        CondCbOp::ParBeforeMove => {
            let t = tpoke.unwrap();
            if b.prng.random_chance(1, 4) {
                let ts = b.poke_str(t);
                b.add(&["cant", &ts, "par"]);
                return RV::False;
            }
            RV::Undef
        }
        CondCbOp::ParSwitchIn => {
            let t = tpoke.unwrap();
            b.add_volatile(dex, t, "parspeeddrop", None, EffectHandle::None);
            RV::Undef
        }
        // ------------------------------------------------------------- slp
        CondCbOp::SlpStart => {
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
        CondCbOp::SlpBeforeMove => {
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
        CondCbOp::FrzStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "frz"]);
            RV::Undef
        }
        CondCbOp::FrzBeforeMove => {
            let t = tpoke.unwrap();
            let defrost = b.active_move.as_ref().map(|m| m.has_flag(dex, "defrost")).unwrap_or(false);
            if defrost {
                return RV::Undef;
            }
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "frz"]);
            RV::False
        }
        CondCbOp::FrzAfterMoveSecondary => {
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
        CondCbOp::FrzAfterMoveSecondarySelf => {
            let t = tpoke.unwrap();
            let defrost = b.active_move.as_ref().map(|m| m.has_flag(dex, "defrost")).unwrap_or(false);
            if defrost {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        CondCbOp::FrzResidual => {
            let t = tpoke.unwrap();
            if b.prng.random_chance(25, 256) {
                b.cure_status(dex, t, false);
            }
            RV::Undef
        }
        // ------------------------------------------------------------- psn
        CondCbOp::PsnStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "psn"]);
            RV::Undef
        }
        CondCbOp::PsnAfterMoveSelf => {
            residualdmg(b, dex, tpoke.unwrap());
            RV::Undef
        }
        // ------------------------------------------------------------- tox
        CondCbOp::ToxStart => {
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
        CondCbOp::ToxAfterMoveSelf => {
            let t = tpoke.unwrap();
            let rd = crate::cond_id!(dex, "residualdmg").unwrap();
            let counter = b.poke(t).volatile(rd).map(|v| v.get_int(crate::state::DK::Counter)).unwrap_or(0);
            let maxhp = b.poke(t).maxhp as f64;
            let dmg = super::clamp_int_range((maxhp / 16.0).floor(), Some(1.0), None) * counter as f64;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "tox").unwrap());
            b.damage(dex, dmg, Some(t), Some(t), DamageEffect::Effect(eff), false);
            RV::Undef
        }
        CondCbOp::ToxSwitchIn => {
            let t = tpoke.unwrap();
            b.poke_mut(t).status = Status::Psn;
            b.refresh_poke_mask(dex, t);
            let ts = b.poke_str(t);
            b.add(&["-status", &ts, "psn", "[silent]"]);
            RV::Undef
        }
        CondCbOp::ToxAfterSwitchInSelf => {
            let t = tpoke.unwrap();
            let maxhp = b.poke(t).maxhp as f64;
            let dmg = super::clamp_int_range((maxhp / 16.0).floor(), Some(1.0), None);
            // this.damage(...) with implicit target/effect from the event
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "tox").unwrap());
            b.damage(dex, dmg, Some(t), Some(t), DamageEffect::Effect(eff), false);
            RV::Undef
        }
        // ------------------------------------------------------- confusion
        CondCbOp::ConfusionStart => {
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
        CondCbOp::ConfusionEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "confusion"]);
            RV::Undef
        }
        CondCbOp::ConfusionBeforeMove => {
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
        CondCbOp::FlinchBeforeMove => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "flinch"]);
            b.run_event(dex, &ev::Flinch, EvTarget::Poke(t), None, EffectHandle::None, None, false, false);
            RV::False
        }
        // ------------------------------------------------ partiallytrapped
        CondCbOp::PartiallyTrappedStart => {
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
        CondCbOp::PartiallyTrappedResidual => {
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
        CondCbOp::PartiallyTrappedEnd => {
            let t = tpoke.unwrap();
            let src_move = b.state_at(state).and_then(|s| s.source_effect);
            let ts = b.poke_str(t);
            let name = src_move_name(dex, &src_move);
            b.add(&["-end", &ts, &name, "[partiallytrapped]"]);
            RV::Undef
        }
        CondCbOp::PartiallyTrappedTrapPokemon => {
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
        CondCbOp::ResidualDmgStart => {
            let t = tpoke.unwrap();
            let cid = crate::cond_id!(dex, "residualdmg").unwrap();
            if let Some(vs) = b.poke_mut(t).volatile_mut(cid) {
                vs.set_int(crate::state::DK::Counter, 0);
            }
            RV::Undef
        }
        CondCbOp::ResidualDmgAfterMoveSelf => {
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
        CondCbOp::MustRechargeBeforeMove => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["cant", &ts, "recharge"]);
            b.remove_volatile(dex, t, "mustrecharge");
            b.remove_volatile(dex, t, "truant");
            RV::Null
        }
        CondCbOp::MustRechargeStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-mustrecharge", &ts]);
            RV::Undef
        }
        CondCbOp::MustRechargeLockMove => RV::Str("recharge"),
        // --------------------------------------------------------- trapped
        CondCbOp::TrappedTrapPokemon => {
            let t = tpoke.unwrap();
            b.try_trap(dex, t);
            RV::Undef
        }
        CondCbOp::TrappedStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-activate", &ts, "trapped"]);
            RV::Undef
        }
        // -------------------------------------------------------- weathers
        CondCbOp::RainDanceFieldStart => {
            b.add(&["-weather", "RainDance"]);
            RV::Undef
        }
        CondCbOp::RainDanceFieldResidual => {
            b.add(&["-weather", "RainDance", "[upkeep]"]);
            b.each_event(dex, &ev::Weather, None);
            RV::Undef
        }
        CondCbOp::RainDanceFieldEnd => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        CondCbOp::SunnyDayFieldStart => {
            b.add(&["-weather", "SunnyDay"]);
            RV::Undef
        }
        CondCbOp::SunnyDayFieldResidual => {
            b.add(&["-weather", "SunnyDay", "[upkeep]"]);
            b.each_event(dex, &ev::Weather, None);
            RV::Undef
        }
        CondCbOp::SunnyDayFieldEnd => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        CondCbOp::SunnyDayImmunity => {
            // onImmunity(type, pokemon): frz → false (can't freeze in sun)
            let t = tpoke.unwrap();
            if b.effective_weather(t) != crate::cond_id!(dex, "sunnyday") {
                return RV::Undef;
            }
            if relay == RV::Str("frz") {
                return RV::False;
            }
            RV::Undef
        }
        CondCbOp::SandstormFieldStart => {
            b.add(&["-weather", "Sandstorm"]);
            RV::Undef
        }
        CondCbOp::SandstormFieldResidual => {
            b.add(&["-weather", "Sandstorm", "[upkeep]"]);
            if b.field.weather.is_some() && b.field.weather == crate::cond_id!(dex, "sandstorm") {
                b.each_event(dex, &ev::Weather, None);
            }
            RV::Undef
        }
        CondCbOp::SandstormWeather => {
            // gen2: this.damage(target.baseMaxhp / 8)
            let t = tpoke.unwrap();
            let dmg = b.poke(t).base_maxhp as f64 / 8.0;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "sandstorm").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        CondCbOp::SandstormFieldEnd => {
            b.add(&["-weather", "none"]);
            RV::Undef
        }
        // ----------------------------------------------------------- rules
        // SetStatus listener order is part of the PS PRNG contract; set_status adjudicates after Start.
        CondCbOp::SleepClauseModSetStatus => {
            // Unlike Stadium Sleep Clause, self-/ally-inflicted sleep (Rest)
            // does not engage the clause: only foe-sourced sleep counts.
            let t = tpoke.unwrap();
            if let Some(src) = source {
                if src.side == t.side {
                    return RV::Undef;
                }
            }
            if relay == RV::Str("slp") {
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
        CondCbOp::FreezeClauseModSetStatus => {
            let t = tpoke.unwrap();
            if let Some(src) = source {
                if src.side == t.side {
                    return RV::Undef;
                }
            }
            if relay == RV::Str("frz") {
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
        CondCbOp::LightScreenSideStart => screen_side_event(b, target, "-sidestart", "move: Light Screen"),
        CondCbOp::ReflectSideStart => screen_side_event(b, target, "-sidestart", "Reflect"),
        CondCbOp::LightScreenSideEnd => screen_side_event(b, target, "-sideend", "move: Light Screen"),
        CondCbOp::ReflectSideEnd => screen_side_event(b, target, "-sideend", "Reflect"),
        // ------------------------------------------------------- safeguard
        CondCbOp::SafeguardSideStart => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            b.add(&["-sidestart", &ss, "Safeguard"]);
            RV::Undef
        }
        CondCbOp::SafeguardSideEnd => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            b.add(&["-sideend", &ss, "Safeguard"]);
            RV::Undef
        }
        CondCbOp::SafeguardSetStatus => safeguard_block(b, dex, tpoke, source, source_effect, relay, true),
        CondCbOp::SafeguardTryAddVolatile => safeguard_block(b, dex, tpoke, source, source_effect, relay, false),
        // ---------------------------------------------------------- spikes
        CondCbOp::SpikesSideStart => {
            let EvTarget::Side(n) = target else { return RV::Undef };
            let ss = b.side_str(n);
            b.add(&["-sidestart", &ss, "Spikes"]);
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Layers, 1);
            }
            RV::Undef
        }
        CondCbOp::SpikesEntryHazard => {
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
        CondCbOp::LeechSeedStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "move: Leech Seed"]);
            RV::Undef
        }
        CondCbOp::LeechSeedAfterMoveSelf => {
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
        CondCbOp::NightmareStart => {
            let t = tpoke.unwrap();
            if b.poke(t).status != Status::Slp {
                return RV::False;
            }
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Nightmare"]);
            RV::Undef
        }
        CondCbOp::NightmareAfterMoveSelf => {
            let t = tpoke.unwrap();
            if b.poke(t).status == Status::Slp {
                let dmg = b.poke(t).base_maxhp as f64 / 4.0;
                let eff = EffectHandle::Cond(crate::cond_id!(dex, "nightmare").unwrap());
                b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            }
            RV::Undef
        }
        // ----------------------------------------------------------- curse
        CondCbOp::CurseStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            let of = format!("[of] {}", b.poke_str(source.unwrap()));
            b.add(&["-start", &ts, "Curse", &of]);
            RV::Undef
        }
        CondCbOp::CurseAfterMoveSelf => {
            let t = tpoke.unwrap();
            let dmg = b.poke(t).base_maxhp as f64 / 4.0;
            let eff = EffectHandle::Cond(crate::cond_id!(dex, "curse").unwrap());
            b.damage(dex, dmg, Some(t), None, DamageEffect::Effect(eff), false);
            RV::Undef
        }
        // ------------------------------------------------------- foresight
        CondCbOp::ForesightStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Foresight"]);
            RV::Undef
        }
        CondCbOp::ForesightNegateImmunity => {
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
        CondCbOp::LockOnSourceAccuracy => {
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            let locked_onto = b.state_at(state).and_then(|s| s.source);
            if source == Some(holder) && tpoke.is_some() && tpoke == locked_onto {
                return RV::True;
            }
            RV::Undef
        }
        // ----------------------------------------------------- destinybond
        CondCbOp::DestinyBondStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singlemove", &ts, "Destiny Bond"]);
            RV::Undef
        }
        CondCbOp::DestinyBondFaint => {
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
        CondCbOp::DestinyBondBeforeMove => {
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
        CondCbOp::DestinyBondMoveAborted => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "destinybond");
            RV::Undef
        }
        // gen2stadium2nc2000: Destiny Bond protects only until the foe next
        // acts — expire after the foe's own move, and on foe switch-out.
        CondCbOp::DestinyBondFoeAfterMoveSelf => {
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            if b.poke(holder).hp > 0 {
                b.remove_volatile(dex, holder, "destinybond");
            }
            RV::Undef
        }
        CondCbOp::DestinyBondFoeSwitchOut => {
            let StateLoc::Volatile(holder, _) = state else { return RV::Undef };
            b.remove_volatile(dex, holder, "destinybond");
            RV::Undef
        }
        // ------------------------------------------------------ perishsong
        CondCbOp::PerishSongResidual => {
            let t = tpoke.unwrap();
            let dur = b.state_at(state).and_then(|s| s.duration).unwrap_or(0);
            let ts = b.poke_str(t);
            let tag = format!("perish{dur}");
            b.add(&["-start", &ts, &tag]);
            RV::Undef
        }
        CondCbOp::PerishSongEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "perish0"]);
            b.pokemon_faint(t, None, EffectHandle::None);
            RV::Undef
        }
        // ---------------------------------------------------------- encore
        CondCbOp::EncoreStart => {
            let t = tpoke.unwrap();
            let locked = b.poke(t).last_move_encore;
            let Some(locked) = locked else { return RV::False };
            let has_slot = b.poke(t).get_move_slot(locked).map(|s| s.pp).unwrap_or(0) > 0;
            if dex.move_static(locked).has_flag(dex, "failencore") || !has_slot {
                return RV::False;
            }
            let locked_key = dex.moves.key(locked);
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
        CondCbOp::EncoreOverrideAction => {
            let move_key = match source_effect {
                EffectHandle::MoveEff(m) => dex.moves.key(m),
                _ => "",
            };
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            if !locked.is_empty() && move_key != locked {
                return RV::Str(locked);
            }
            RV::Undef
        }
        CondCbOp::EncoreResidual => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            let pp_left = dex
                .moves
                .id(locked)
                .and_then(|mid| b.poke(t).get_move_slot(mid).map(|s| s.pp))
                .unwrap_or(0);
            if pp_left <= 0 {
                b.remove_volatile(dex, t, "encore");
            }
            RV::Undef
        }
        CondCbOp::EncoreEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Encore"]);
            RV::Undef
        }
        CondCbOp::EncoreDisableMove => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            let Some(mid) = dex.moves.id(locked) else { return RV::Undef };
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
        CondCbOp::DisableStart => {
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
        CondCbOp::DisableEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Disable"]);
            RV::Undef
        }
        CondCbOp::DisableBeforeMove => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            let cur = b
                .active_move
                .as_ref()
                .and_then(|m| m.id)
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            if !locked.is_empty() && cur == locked {
                let ts = b.poke_str(t);
                let name = b.active_move_name(dex);
                b.add(&["cant", &ts, "Disable", &name]);
                return RV::False;
            }
            RV::Undef
        }
        CondCbOp::DisableDisableMove => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            if let Some(mid) = dex.moves.id(locked) {
                if b.poke(t).get_move_slot(mid).is_some() {
                    b.pokemon_disable_move(t, mid);
                }
            }
            RV::Undef
        }
        // ------------------------------------------------------------ mist
        CondCbOp::MistStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "Mist"]);
            RV::Undef
        }
        CondCbOp::MistTryBoost => {
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
        CondCbOp::SubstituteStart => {
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
        CondCbOp::SubstituteEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Substitute"]);
            RV::Undef
        }
        CondCbOp::SubstituteTryPrimaryHit => {
            let t = tpoke.unwrap();
            let src = source.unwrap();
            let (stalling, is_drain, category, move_key, has_status, has_boosts, vs, has_recoil) = {
                let am = b.active_move.as_ref().unwrap();
                (
                    am.stalling_move,
                    am.drain.is_some(),
                    am.category,
                    am.id.map(|m| dex.moves.key(m)).unwrap_or(""),
                    am.status.is_some(),
                    am.has_boosts,
                    am.volatile_status,
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
                    || vs == Some("confusion")
                    || SUB_BLOCKED.contains(&move_key)
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
        CondCbOp::BideStart => {
            let t = tpoke.unwrap();
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::TotalDamage, 0);
            }
            let ts = b.poke_str(t);
            b.add(&["-start", &ts, "move: Bide"]);
            RV::Undef
        }
        CondCbOp::BideDamage => {
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
        CondCbOp::BideBeforeMove => {
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
                fake.secondaries = smallvec::SmallVec::new();
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
        CondCbOp::BideMoveAborted => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "bide");
            RV::Undef
        }
        CondCbOp::BideEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "move: Bide", "[silent]"]);
            RV::Undef
        }
        CondCbOp::BideSemiLockMove => RV::Str("bide"),
        // ------------------------------------------------------------ rage
        CondCbOp::RageStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singlemove", &ts, "Rage"]);
            RV::Undef
        }
        CondCbOp::RageHit => {
            let t = tpoke.unwrap();
            let category = b.active_move.as_ref().map(|m| m.category);
            if source != Some(t) && category != Some(crate::dex::Category::Status) {
                let eff = EffectHandle::Cond(crate::cond_id!(dex, "rage").unwrap());
                b.boost(dex, &[(0, 1)], Some(t), source, eff);
            }
            RV::Undef
        }
        CondCbOp::RageBeforeMove => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "rage");
            RV::Undef
        }
        // --------------------------------------------------------- rollout
        CondCbOp::RolloutStart => {
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::HitCount, 0);
                st.set_int(crate::state::DK::ContactHitCount, 0);
            }
            RV::Undef
        }
        CondCbOp::RolloutResidual => {
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
        CondCbOp::RolloutLockMove => RV::Str("rollout"),
        // ------------------------------------------------------ furycutter
        CondCbOp::FuryCutterStart => {
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Multiplier, 1);
            }
            RV::Undef
        }
        CondCbOp::FuryCutterRestart => {
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
        CondCbOp::DefenseCurlMinimizeRestart => RV::Null,
        CondCbOp::MinimizeSourceModifyDamage => {
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
        CondCbOp::AttractStart => {
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
        CondCbOp::AttractUpdate => {
            let t = tpoke.unwrap();
            let src = b.state_at(state).and_then(|s| s.source);
            if let Some(src) = src {
                if !b.poke(src).is_active {
                    b.remove_volatile(dex, t, "attract");
                }
            }
            RV::Undef
        }
        CondCbOp::AttractBeforeMove => {
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
        CondCbOp::AttractEnd => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-end", &ts, "Attract", "[silent]"]);
            RV::Undef
        }
        // ----------------------------------------------------- focusenergy
        CondCbOp::FocusEnergyStart => {
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
        CondCbOp::FocusEnergyModifyCritRatio => RV::Num(relay.as_num() + 1.0),
        // --------------------------------------------------------- protect
        CondCbOp::ProtectStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singleturn", &ts, "Protect"]);
            RV::Undef
        }
        CondCbOp::ProtectTryHit => {
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
        CondCbOp::EndureStart => {
            let t = tpoke.unwrap();
            let ts = b.poke_str(t);
            b.add(&["-singleturn", &ts, "move: Endure"]);
            RV::Undef
        }
        CondCbOp::EndureDamage => {
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
        CondCbOp::StallStart => {
            if let Some(st) = b.state_at_mut(state) {
                st.set_int(crate::state::DK::Counter, 127);
            }
            RV::Undef
        }
        CondCbOp::StallMove => {
            let counter = b
                .state_at(state)
                .and_then(|s| s.get(crate::state::DK::Counter))
                .map(|v| v.as_f64())
                .unwrap_or(0.0);
            let mut c = counter.floor() as u32;
            if c == 0 {
                c = 127;
            }
            RV::from_bool(b.prng.random_chance(c, 255))
        }
        CondCbOp::StallRestart => {
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
        CondCbOp::TwoTurnMoveStart => {
            let t = tpoke.unwrap();
            let EffectHandle::MoveEff(mv) = source_effect else { return RV::Undef };
            let move_key = dex.moves.key(mv);
            if let Some(st) = b.state_at_mut(state) {
                st.set(crate::state::DK::Move, Scalar::MoveK(mv));
            }
            b.add_volatile(dex, t, move_key, None, EffectHandle::None);
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
            let mvc = dex.conds_id(move_key).unwrap();
            if let Some(loc) = move_target_loc {
                if let Some(vs) = b.poke_mut(t).volatile_mut(mvc) {
                    vs.set_int(crate::state::DK::TargetLoc, loc as i64);
                }
            }
            b.attr_last_move(&["[still]"]);
            b.run_event(dex, &ev::PrepareHit, EvTarget::Poke(t), source, source_effect, None, false, false);
            RV::Undef
        }
        CondCbOp::TwoTurnMoveEnd => {
            let t = tpoke.unwrap();
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            if !locked.is_empty() {
                b.remove_volatile(dex, t, locked);
            }
            RV::Undef
        }
        CondCbOp::TwoTurnMoveLockMove => {
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            RV::Str(locked)
        }
        CondCbOp::TwoTurnMoveAborted => {
            let t = tpoke.unwrap();
            b.remove_volatile(dex, t, "twoturnmove");
            RV::Undef
        }
        // ---------------------------------------------- dig / fly volatiles
        CondCbOp::DigImmunity => {
            if relay == RV::Str("sandstorm") || relay == RV::Str("hail") {
                return RV::False;
            }
            RV::Undef
        }
        CondCbOp::DigInvulnerability => two_turn_invuln(b, dex, tpoke, source, true),
        CondCbOp::FlyInvulnerability => two_turn_invuln(b, dex, tpoke, source, false),
        CondCbOp::DigSourceBasePower => two_turn_source_bp(b, dex, true),
        CondCbOp::FlySourceBasePower => two_turn_source_bp(b, dex, false),
        // ------------------------------------------------------ lockedmove
        CondCbOp::LockedMoveStart => {
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
        CondCbOp::LockedMoveResidual => {
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
        CondCbOp::LockedMoveEnd => {
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
        CondCbOp::LockedMoveLockMove => {
            let locked = b
                .state_at(state)
                .and_then(|s| s.get_move())
                .map(|m| dex.moves.key(m))
                .unwrap_or("");
            RV::Str(locked)
        }
        CondCbOp::LockedMoveAborted => {
            let t = tpoke.unwrap();
            let lm = crate::cond_id!(dex, "lockedmove").unwrap();
            b.poke_mut(t).volatiles.retain(|(k, _)| *k != lm);
            b.refresh_poke_mask(dex, t);
            RV::Undef
        }
        // --------------------------------------------------------- pursuit
        CondCbOp::PursuitFoeBeforeSwitchOut => {
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
        CondCbOp::FutureMoveStart => {
            let t = tpoke.unwrap();
            let slot = b.slot_of(t);
            let ending = b.turn as i64 + 1;
            if let Some(st) = b.state_at_mut(state) {
                st.set(crate::state::DK::TargetSlot, Scalar::Slot(slot.0, slot.1));
                st.set_int(crate::state::DK::EndingTurn, ending);
            }
            RV::Undef
        }
        CondCbOp::FutureMoveResidual => {
            let (ending, slot) = {
                let st = b.state_at(state).unwrap();
                (
                    st.get_int(crate::state::DK::EndingTurn),
                    st.get(crate::state::DK::TargetSlot).and_then(|v| match v {
                        Scalar::Slot(a, b) => Some((a, b)),
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
        CondCbOp::FutureMoveEnd => {
            let t = tpoke.unwrap();
            super::moveexec::resolve_future_move(b, dex, state, t);
            RV::Undef
        }
        // ---------------------------------------------------------- trapper
        // trapper, stadiumsleepclause onSetStatus, marker volatiles (brnattackdrop, parspeeddrop, leppaberry)
        CondCbOp::Noop => RV::Undef,
        CondCbOp::Unported => unported_cond_cb(dex, cond, cb),
    }
}

#[inline]
fn screen_side_event(b: &mut Battle, target: EvTarget, tag: &str, label: &str) -> RV {
    let EvTarget::Side(n) = target else { return RV::Undef };
    let ss = b.side_str(n);
    b.add(&[tag, &ss, label]);
    RV::Undef
}

fn safeguard_block(
    b: &mut Battle,
    dex: &Dex,
    tpoke: Option<PokeId>,
    source: Option<PokeId>,
    source_effect: EffectHandle,
    relay: RV,
    is_set_status: bool,
) -> RV {
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
            matches!(source_effect, EffectHandle::MoveEff(m) if am.id == Some(m)) && am.infiltrates
        })
        .unwrap_or(false);
    if is_move && infiltrates && t.side != src.side {
        return RV::Undef;
    }
    if t != src && (is_set_status || relay == RV::Str("confusion")) {
        if is_move && !move_eff_has_secondaries(b, dex, source_effect) {
            let ts = b.poke_str(t);
            b.add(&["-activate", &ts, "move: Safeguard"]);
        }
        return RV::Null;
    }
    RV::Undef
}

fn two_turn_invuln(
    b: &Battle,
    dex: &Dex,
    tpoke: Option<PokeId>,
    source: Option<PokeId>,
    is_dig: bool,
) -> RV {
    let move_key = b.active_move.as_ref().and_then(|m| m.id).map(|m| dex.moves.key(m)).unwrap_or("");
    if is_dig {
        if matches!(move_key, "earthquake" | "magnitude" | "fissure") {
            return RV::Undef;
        }
    } else {
        if matches!(move_key, "gust" | "twister" | "thunder" | "whirlwind") {
            return RV::Undef;
        }
        if matches!(move_key, "earthquake" | "magnitude" | "fissure") {
            return RV::False;
        }
    }
    if matches!(
        move_key,
        "attract" | "curse" | "foresight" | "meanlook" | "mimic" | "nightmare" | "spiderweb"
            | "transform"
    ) {
        return RV::False;
    }
    if let (Some(t), Some(src)) = (tpoke, source) {
        let lo = crate::cond_id!(dex, "lockon").unwrap();
        let locked =
            b.poke(src).volatile(lo).and_then(|v| v.source).map(|l| l == t).unwrap_or(false);
        if b.poke(src).has_volatile(lo) && locked {
            return RV::Undef;
        }
    }
    RV::False
}

fn two_turn_source_bp(b: &mut Battle, dex: &Dex, is_dig: bool) -> RV {
    let move_key = b.active_move.as_ref().and_then(|m| m.id).map(|m| dex.moves.key(m)).unwrap_or("");
    let doubled = if is_dig {
        matches!(move_key, "earthquake" | "magnitude")
    } else {
        matches!(move_key, "gust" | "twister")
    };
    if doubled {
        b.chain_modify(2.0, 1.0);
    }
    RV::Undef
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
