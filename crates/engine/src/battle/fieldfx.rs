//! Field + side condition management (sim/field.ts, side.ts).

use crate::dex::{CondId, Dex};
use crate::state::*;

use super::events::EvTarget;
use super::{EffectHandle, RV};

impl Battle {
    /// field.setWeather.
    pub fn set_weather(
        &mut self,
        dex: &Dex,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> RV {
        let cond = dex.conds_id(status).unwrap_or_else(|| panic!("unknown weather {status}"));
        let mut source_effect = source_effect;
        if source_effect.is_none() {
            source_effect = self.current_effect();
        }
        let mut source = source;
        if source.is_none() {
            source = self.event_stack.last().and_then(|f| f.target);
        }

        if self.field.weather == Some(cond) {
            // gen2: only sandstorm re-set fails... (gen > 2 || id === 'sandstorm' → false)
            if status == "sandstorm" {
                return RV::False;
            }
        }
        if let Some(src) = source {
            let result = self.run_event(
                dex,
                "SetWeather",
                EvTarget::Poke(src),
                Some(src),
                EffectHandle::Cond(cond),
                Some(RV::Str(status.to_string())),
                false,
                false,
            );
            if !result.truthy() {
                if result == RV::False {
                    let is_weather_move = match source_effect {
                        EffectHandle::MoveEff(m) => dex.move_static(m).weather.is_some(),
                        _ => false,
                    };
                    if is_weather_move {
                        let ss = self.poke_str(src);
                        let se_name = self.effect_name(dex, source_effect);
                        let from = format!("[from] {}", self.field_weather_key);
                        self.add(&["-fail", &ss, &se_name, &from]);
                    }
                }
                return RV::Null;
            }
        }
        let prev_weather = self.field.weather;
        let prev_state = self.field.weather_state.clone();
        let prev_key = self.field_weather_key.clone();

        self.field.weather = Some(cond);
        self.field_weather_key = status.to_string();
        let mut state = EffectState { id: status.to_string(), ..Default::default() };
        if let Some(src) = source {
            state.source = Some(src);
            state.source_slot = Some(self.slot_str(src));
        }
        if let Some(d) = dex.cond(cond).duration {
            state.duration = Some(d);
        }
        let state = self.init_effect_state(state, true);
        self.field.weather_state = state;
        if dex.cond(cond).has_callback("durationCallback") {
            let dur =
                super::conditions::duration_callback(self, dex, status, source, source, source_effect);
            if let Some(d) = dur {
                self.field.weather_state.duration = Some(d);
            }
        }
        let started = self.single_event(
            dex,
            "FieldStart",
            EffectHandle::Cond(cond),
            StateLoc::Weather,
            EvTarget::Battle,
            source,
            source_effect,
            None,
        );
        if !started.truthy() {
            self.field.weather = prev_weather;
            self.field.weather_state = prev_state;
            self.field_weather_key = prev_key;
            return RV::False;
        }
        self.each_event(dex, "WeatherChange", Some(source_effect));
        RV::True
    }

    /// field.clearWeather.
    pub fn clear_weather(&mut self, dex: &Dex) -> bool {
        let Some(cond) = self.field.weather else { return false };
        self.single_event(
            dex,
            "FieldEnd",
            EffectHandle::Cond(cond),
            StateLoc::Weather,
            EvTarget::Battle,
            None,
            EffectHandle::None,
            None,
        );
        self.field.weather = None;
        self.field_weather_key.clear();
        // clearEffectState: keep effectOrder 0, drop the rest
        self.field.weather_state = EffectState::default();
        self.each_event(dex, "WeatherChange", None);
        true
    }

    /// field.addPseudoWeather.
    pub fn add_pseudo_weather(
        &mut self,
        dex: &Dex,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> RV {
        let cond = dex.conds_id(status).unwrap_or_else(|| panic!("unknown pseudo weather {status}"));
        if self.field.has_pseudo_weather(cond) {
            if !dex.cond(cond).has_callback("onFieldRestart") {
                return RV::False;
            }
            return self.single_event(
                dex,
                "FieldRestart",
                EffectHandle::Cond(cond),
                StateLoc::PseudoWeather(cond),
                EvTarget::Battle,
                source,
                source_effect,
                None,
            );
        }
        let mut state = EffectState { id: status.to_string(), ..Default::default() };
        if let Some(src) = source {
            state.source = Some(src);
            state.source_slot = Some(self.slot_str(src));
        }
        if let Some(d) = dex.cond(cond).duration {
            state.duration = Some(d);
        }
        let state = self.init_effect_state(state, true);
        self.field.pseudo_weather.push((cond, state));
        if dex.cond(cond).has_callback("durationCallback") {
            let dur =
                super::conditions::duration_callback(self, dex, status, source, source, source_effect);
            if let Some(d) = dur {
                if let Some(st) = self.state_at_mut(StateLoc::PseudoWeather(cond)) {
                    st.duration = Some(d);
                }
            }
        }
        let started = self.single_event(
            dex,
            "FieldStart",
            EffectHandle::Cond(cond),
            StateLoc::PseudoWeather(cond),
            EvTarget::Battle,
            source,
            source_effect,
            None,
        );
        if !started.truthy() {
            self.field.pseudo_weather.retain(|(k, _)| *k != cond);
            return RV::False;
        }
        RV::True
    }

    pub fn remove_pseudo_weather(&mut self, dex: &Dex, cond: CondId) -> bool {
        if !self.field.has_pseudo_weather(cond) {
            return false;
        }
        self.single_event(
            dex,
            "FieldEnd",
            EffectHandle::Cond(cond),
            StateLoc::PseudoWeather(cond),
            EvTarget::Battle,
            None,
            EffectHandle::None,
            None,
        );
        self.field.pseudo_weather.retain(|(k, _)| *k != cond);
        true
    }

    /// side.addSideCondition.
    pub fn add_side_condition(
        &mut self,
        dex: &Dex,
        side_n: u8,
        status: &str,
        source: Option<PokeId>,
        source_effect: EffectHandle,
    ) -> RV {
        let cond = dex.conds_id(status).unwrap_or_else(|| panic!("unknown side condition {status}"));
        let mut source = source;
        if source.is_none() {
            source = self.event_stack.last().and_then(|f| f.target);
        }
        let source = source.expect("setting sidecond without a source");

        if self.sides[side_n as usize].has_side_condition(cond) {
            if !dex.cond(cond).has_callback("onSideRestart") {
                return RV::False;
            }
            return self.single_event(
                dex,
                "SideRestart",
                EffectHandle::Cond(cond),
                StateLoc::SideCond(side_n, cond),
                EvTarget::Side(side_n),
                Some(source),
                source_effect,
                None,
            );
        }
        let mut state = EffectState { id: status.to_string(), ..Default::default() };
        state.source = Some(source);
        state.source_slot = Some(self.slot_str(source));
        if let Some(d) = dex.cond(cond).duration {
            state.duration = Some(d);
        }
        let state = self.init_effect_state(state, true);
        self.sides[side_n as usize].side_conditions.push((cond, state));
        if dex.cond(cond).has_callback("durationCallback") {
            let dur = super::conditions::duration_callback(
                self,
                dex,
                status,
                self.active_id(side_n as usize),
                Some(source),
                source_effect,
            );
            if let Some(d) = dur {
                if let Some(st) = self.state_at_mut(StateLoc::SideCond(side_n, cond)) {
                    st.duration = Some(d);
                }
            }
        }
        let started = self.single_event(
            dex,
            "SideStart",
            EffectHandle::Cond(cond),
            StateLoc::SideCond(side_n, cond),
            EvTarget::Side(side_n),
            Some(source),
            source_effect,
            None,
        );
        if !started.truthy() {
            self.sides[side_n as usize].side_conditions.retain(|(k, _)| *k != cond);
            return RV::False;
        }
        self.run_event(
            dex,
            "SideConditionStart",
            EvTarget::Side(side_n),
            Some(source),
            EffectHandle::Cond(cond),
            None,
            false,
            false,
        );
        RV::True
    }

    pub fn remove_side_condition(&mut self, dex: &Dex, side_n: u8, cond: CondId) -> bool {
        if !self.sides[side_n as usize].has_side_condition(cond) {
            return false;
        }
        self.single_event(
            dex,
            "SideEnd",
            EffectHandle::Cond(cond),
            StateLoc::SideCond(side_n, cond),
            EvTarget::Side(side_n),
            None,
            EffectHandle::None,
            None,
        );
        self.sides[side_n as usize].side_conditions.retain(|(k, _)| *k != cond);
        true
    }
}
