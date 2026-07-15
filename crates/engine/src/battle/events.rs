//! The event system: a faithful port of PS `runEvent`/`singleEvent`/
//! `fieldEvent`/`eachEvent` + `findEventHandlers`/`resolvePriority`/
//! `speedSort`. Handler ordering (including PRNG-consuming tie shuffles) is
//! the load-bearing part — treat every deviation as a conformance bug.

use crate::dex::{CondId, Dex, EffectType};
use crate::state::*;

use super::{EffectHandle, RV};

/// runEvent/singleEvent target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvTarget {
    Poke(PokeId),
    Side(u8),
    Battle,
}

impl EvTarget {
    pub fn poke(self) -> Option<PokeId> {
        match self {
            EvTarget::Poke(p) => Some(p),
            _ => None,
        }
    }
}

/// What fieldEvent's `handler.end` does when a duration expires.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndKind {
    RemoveVolatile,
    ClearStatus,
    RemoveSideCond,
    ClearWeather,
    RemovePseudoWeather,
    None,
}

#[derive(Clone, Debug)]
pub struct Listener {
    pub effect: EffectHandle,
    pub has_callback: bool,
    pub state: StateLoc,
    pub holder: Holder,
    pub end_kind: EndKind,
    pub order: Option<i64>,
    pub priority: f64,
    pub sub_order: f64,
    pub speed: f64,
    pub effect_order: f64,
    /// Captured `state.effect_order` for the fieldEvent staleness check.
    pub state_token: u32,
}

/// comparePriority as a signed delta (PS returns a number; only the sign and
/// zero-ness matter). Returns <0 if a first, >0 if b first, 0 tie.
pub fn compare_priority(a: &Listener, b: &Listener) -> f64 {
    let ao = a.order.unwrap_or(4294967296) as f64;
    let bo = b.order.unwrap_or(4294967296) as f64;
    let d = ao - bo;
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
    let d = a.sub_order - b.sub_order;
    if d != 0.0 {
        return d;
    }
    a.effect_order - b.effect_order
}

/// compareLeftToRightOrder (order asc, priority desc, index asc). We have no
/// spread targets in singles, so index is always equal.
pub fn compare_left_to_right(a: &Listener, b: &Listener) -> f64 {
    let ao = a.order.unwrap_or(4294967296) as f64;
    let bo = b.order.unwrap_or(4294967296) as f64;
    let d = ao - bo;
    if d != 0.0 {
        return d;
    }
    b.priority - a.priority
}

/// compareRedirectOrder (priority desc, speed desc, ability effectOrder asc —
/// no abilities here, so 0).
pub fn compare_redirect_order(a: &Listener, b: &Listener) -> f64 {
    let d = b.priority - a.priority;
    if d != 0.0 {
        return d;
    }
    b.speed - a.speed
}

impl Battle {
    /// PS speedSort: selection sort; full ties shuffled with the battle PRNG.
    pub fn speed_sort<T>(&mut self, list: &mut [T], cmp: impl Fn(&T, &T) -> f64) {
        if list.len() < 2 {
            return;
        }
        let mut sorted = 0;
        while sorted + 1 < list.len() {
            let mut next_indexes = vec![sorted];
            for i in (sorted + 1)..list.len() {
                let delta = cmp(&list[next_indexes[0]], &list[i]);
                if delta < 0.0 {
                    continue;
                }
                if delta > 0.0 {
                    next_indexes = vec![i];
                } else {
                    next_indexes.push(i);
                }
            }
            for (i, &index) in next_indexes.iter().enumerate() {
                if index != sorted + i {
                    list.swap(sorted + i, index);
                }
            }
            if next_indexes.len() > 1 {
                if std::env::var_os("NC_TRACE").is_some() {
                    eprintln!("[trace] speed_sort shuffle: ties={} seed_before={}", next_indexes.len(), self.prng.seed_str());
                }
                self.prng.shuffle(list, sorted, sorted + next_indexes.len());
            }
            sorted += next_indexes.len();
        }
    }

    // ------------------------------------------------------ event frames

    pub fn current_event(&self) -> Option<&EventFrame> {
        self.event_stack.last()
    }

    pub fn current_effect(&self) -> EffectHandle {
        self.effect_stack.last().map(|f| f.effect).unwrap_or(EffectHandle::None)
    }

    pub fn current_effect_state(&self) -> StateLoc {
        self.effect_stack.last().map(|f| f.state).unwrap_or(StateLoc::None)
    }

    /// Access an EffectState by location (None if it no longer exists).
    pub fn state_at(&self, loc: StateLoc) -> Option<&EffectState> {
        match loc {
            StateLoc::Status(p) => Some(&self.poke(p).status_state),
            StateLoc::Volatile(p, c) => self.poke(p).volatile(c),
            StateLoc::SideCond(s, c) => self.sides[s as usize].side_condition(c),
            StateLoc::SlotCond(s, slot, c) => self.sides[s as usize]
                .slot_conditions
                .iter()
                .find(|(k, _)| *k == c)
                .map(|(_, v)| v)
                .filter(|_| slot == 0),
            StateLoc::Weather => Some(&self.field.weather_state),
            StateLoc::PseudoWeather(c) => {
                self.field.pseudo_weather.iter().find(|(k, _)| *k == c).map(|(_, v)| v)
            }
            StateLoc::Format => Some(&self.format_data),
            StateLoc::None => None,
        }
    }

    pub fn state_at_mut(&mut self, loc: StateLoc) -> Option<&mut EffectState> {
        match loc {
            StateLoc::Status(p) => Some(&mut self.poke_mut(p).status_state),
            StateLoc::Volatile(p, c) => self.poke_mut(p).volatile_mut(c),
            StateLoc::SideCond(s, c) => self.sides[s as usize]
                .side_conditions
                .iter_mut()
                .find(|(k, _)| *k == c)
                .map(|(_, v)| v),
            StateLoc::SlotCond(s, slot, c) => self.sides[s as usize]
                .slot_conditions
                .iter_mut()
                .find(|(k, _)| *k == c)
                .map(|(_, v)| v)
                .filter(|_| slot == 0),
            StateLoc::Weather => Some(&mut self.field.weather_state),
            StateLoc::PseudoWeather(c) => {
                self.field.pseudo_weather.iter_mut().find(|(k, _)| *k == c).map(|(_, v)| v)
            }
            StateLoc::Format => Some(&mut self.format_data),
            StateLoc::None => None,
        }
    }

    // ------------------------------------------------- handler collection

    /// resolvePriority: fill sort metadata for a listener.
    fn resolve_priority(&self, dex: &Dex, mut h: Listener, callback_name: &str) -> Listener {
        if let EffectHandle::Cond(c) = h.effect {
            let entry = dex.cond(c);
            h.order = entry.num(&format!("{callback_name}Order")).map(|v| v as i64);
            h.priority = entry.num(&format!("{callback_name}Priority")).unwrap_or(0) as f64;
            h.sub_order = entry.num(&format!("{callback_name}SubOrder")).unwrap_or(0) as f64;
            if h.sub_order == 0.0 {
                // effectTypeOrder: Condition 2 (3/4/5 by state target), Weather
                // 5, Format/Rule 5, Item 8. 'Status' is NOT in the table → 0.
                h.sub_order = match dex.cond_effect_type(c) {
                    EffectType::Condition => match h.state {
                        StateLoc::SlotCond(..) => 3.0,
                        StateLoc::SideCond(..) => 4.0,
                        StateLoc::PseudoWeather(_) => 5.0,
                        _ => 2.0,
                    },
                    EffectType::Weather => 5.0,
                    EffectType::Rule | EffectType::Format => 5.0,
                    EffectType::Item => 8.0,
                    _ => 0.0,
                };
            }
        } else if let EffectHandle::Item(_) = h.effect {
            if h.sub_order == 0.0 {
                h.sub_order = 8.0;
            }
        }
        if callback_name.ends_with("SwitchIn") || callback_name.ends_with("RedirectTarget") {
            h.effect_order = self.state_at(h.state).map(|s| s.effect_order as f64).unwrap_or(0.0);
        }
        if let Holder::Poke(p) = h.holder {
            h.speed = self.poke(p).speed as f64;
            if callback_name.ends_with("SwitchIn") {
                let fpv = self.field_position_value(p);
                let idx = self.speed_order.iter().position(|&v| v == fpv).map(|i| i as f64).unwrap_or(-1.0);
                h.speed -= idx / 2.0;
            }
        }
        h
    }

    fn cond_has_callback(&self, dex: &Dex, c: CondId, callback_name: &str) -> bool {
        dex.cond(c).has_callback(callback_name)
            || super::conditions::has_builtin(dex.conds_key(c), callback_name)
    }

    /// findPokemonEventHandlers: status → volatiles → (ability) → item →
    /// (species) → slot conditions.
    pub fn find_pokemon_event_handlers(
        &self,
        dex: &Dex,
        pokemon: PokeId,
        callback_name: &str,
        get_key_duration: bool,
    ) -> Vec<Listener> {
        let mut handlers = Vec::new();
        let p = self.poke(pokemon);

        // status
        if p.status != Status::None && p.status != Status::Fnt {
            if let Some(c) = dex.conds_id(p.status.as_str()) {
                let has_cb = self.cond_has_callback(dex, c, callback_name);
                let dur = p.status_state.duration.is_some();
                if has_cb || (get_key_duration && dur) {
                    handlers.push(self.resolve_priority(
                        dex,
                        Listener {
                            effect: EffectHandle::Cond(c),
                            has_callback: has_cb,
                            state: StateLoc::Status(pokemon),
                            holder: Holder::Poke(pokemon),
                            end_kind: EndKind::ClearStatus,
                            order: None,
                            priority: 0.0,
                            sub_order: 0.0,
                            speed: 0.0,
                            effect_order: 0.0,
                            state_token: p.status_state.effect_order,
                        },
                        callback_name,
                    ));
                }
            }
        }
        // volatiles (insertion order)
        for (c, state) in &p.volatiles {
            let has_cb = self.cond_has_callback(dex, *c, callback_name);
            if has_cb || (get_key_duration && state.duration.is_some()) {
                handlers.push(self.resolve_priority(
                    dex,
                    Listener {
                        effect: EffectHandle::Cond(*c),
                        has_callback: has_cb,
                        state: StateLoc::Volatile(pokemon, *c),
                        holder: Holder::Poke(pokemon),
                        end_kind: EndKind::RemoveVolatile,
                        order: None,
                        priority: 0.0,
                        sub_order: 0.0,
                        speed: 0.0,
                        effect_order: 0.0,
                        state_token: state.effect_order,
                    },
                    callback_name,
                ));
            }
        }
        // ability: 'No Ability' has no handlers — skip.
        // item (milestone 2: item callbacks)
        if let Some(item) = p.item {
            let has_cb = dex.items.get(item).callbacks.iter().any(|c| c == callback_name);
            if has_cb || (get_key_duration && p.item_state.duration.is_some()) {
                handlers.push(self.resolve_priority(
                    dex,
                    Listener {
                        effect: EffectHandle::Item(item),
                        has_callback: has_cb,
                        state: StateLoc::None,
                        holder: Holder::Poke(pokemon),
                        end_kind: EndKind::None,
                        order: None,
                        priority: 0.0,
                        sub_order: 0.0,
                        speed: 0.0,
                        effect_order: 0.0,
                        state_token: p.item_state.effect_order,
                    },
                    callback_name,
                ));
            }
        }
        // species: no runtime species conditions in gen2.
        // slot conditions (futuremove — milestone 2)
        let side = &self.sides[pokemon.side as usize];
        if p.position == 0 {
            for (c, state) in &side.slot_conditions {
                let has_cb = self.cond_has_callback(dex, *c, callback_name);
                if has_cb || (get_key_duration && state.duration.is_some()) {
                    handlers.push(self.resolve_priority(
                        dex,
                        Listener {
                            effect: EffectHandle::Cond(*c),
                            has_callback: has_cb,
                            state: StateLoc::SlotCond(pokemon.side, 0, *c),
                            holder: Holder::Poke(pokemon),
                            end_kind: EndKind::None,
                            order: None,
                            priority: 0.0,
                            sub_order: 0.0,
                            speed: 0.0,
                            effect_order: 0.0,
                            state_token: state.effect_order,
                        },
                        callback_name,
                    ));
                }
            }
        }
        handlers
    }

    pub fn find_side_event_handlers(
        &self,
        dex: &Dex,
        side_n: u8,
        callback_name: &str,
        get_key_duration: bool,
        custom_holder: Option<PokeId>,
    ) -> Vec<Listener> {
        let mut handlers = Vec::new();
        for (c, state) in &self.sides[side_n as usize].side_conditions {
            let has_cb = self.cond_has_callback(dex, *c, callback_name);
            if has_cb || (get_key_duration && state.duration.is_some()) {
                handlers.push(self.resolve_priority(
                    dex,
                    Listener {
                        effect: EffectHandle::Cond(*c),
                        has_callback: has_cb,
                        state: StateLoc::SideCond(side_n, *c),
                        holder: match custom_holder {
                            Some(p) => Holder::Poke(p),
                            None => Holder::Side(side_n),
                        },
                        end_kind: if custom_holder.is_some() { EndKind::None } else { EndKind::RemoveSideCond },
                        order: None,
                        priority: 0.0,
                        sub_order: 0.0,
                        speed: 0.0,
                        effect_order: 0.0,
                        state_token: state.effect_order,
                    },
                    callback_name,
                ));
            }
        }
        handlers
    }

    pub fn find_field_event_handlers(
        &self,
        dex: &Dex,
        callback_name: &str,
        get_key_duration: bool,
        custom_holder: Option<PokeId>,
    ) -> Vec<Listener> {
        let mut handlers = Vec::new();
        for (c, state) in &self.field.pseudo_weather {
            let has_cb = self.cond_has_callback(dex, *c, callback_name);
            if has_cb || (get_key_duration && state.duration.is_some()) {
                handlers.push(self.resolve_priority(
                    dex,
                    Listener {
                        effect: EffectHandle::Cond(*c),
                        has_callback: has_cb,
                        state: StateLoc::PseudoWeather(*c),
                        holder: match custom_holder {
                            Some(p) => Holder::Poke(p),
                            None => Holder::Field,
                        },
                        end_kind: if custom_holder.is_some() { EndKind::None } else { EndKind::RemovePseudoWeather },
                        order: None,
                        priority: 0.0,
                        sub_order: 0.0,
                        speed: 0.0,
                        effect_order: 0.0,
                        state_token: state.effect_order,
                    },
                    callback_name,
                ));
            }
        }
        if let Some(w) = self.field.weather {
            let has_cb = self.cond_has_callback(dex, w, callback_name);
            if has_cb || (get_key_duration && self.field.weather_state.duration.is_some()) {
                handlers.push(self.resolve_priority(
                    dex,
                    Listener {
                        effect: EffectHandle::Cond(w),
                        has_callback: has_cb,
                        state: StateLoc::Weather,
                        holder: match custom_holder {
                            Some(p) => Holder::Poke(p),
                            None => Holder::Field,
                        },
                        end_kind: if custom_holder.is_some() { EndKind::None } else { EndKind::ClearWeather },
                        order: None,
                        priority: 0.0,
                        sub_order: 0.0,
                        speed: 0.0,
                        effect_order: 0.0,
                        state_token: self.field.weather_state.effect_order,
                    },
                    callback_name,
                ));
            }
        }
        // terrain: none in gen2.
        handlers
    }

    /// findBattleEventHandlers: the format itself has no runtime handlers in
    /// NC2000 (rules act via pseudo-weathers), so this is empty.
    pub fn find_battle_event_handlers(&self, _callback_name: &str) -> Vec<Listener> {
        Vec::new()
    }

    /// findEventHandlers (the full prefix dance).
    pub fn find_event_handlers(
        &self,
        dex: &Dex,
        target: EvTarget,
        event_name: &str,
        source: Option<PokeId>,
    ) -> Vec<Listener> {
        let mut handlers: Vec<Listener> = Vec::new();
        let prefixed = !matches!(
            event_name,
            "BeforeTurn" | "Update" | "Weather" | "WeatherChange" | "TerrainChange"
        );
        let should_bubble_down = matches!(target, EvTarget::Side(_));
        let mut side_target: Option<u8> = match target {
            EvTarget::Side(s) => Some(s),
            _ => None,
        };

        if let EvTarget::Poke(p) = target {
            let target_active = self.poke(p).is_active;
            let source_active = source.map(|s| self.poke(s).is_active).unwrap_or(false);
            if target_active || source_active {
                handlers = self.find_pokemon_event_handlers(dex, p, &format!("on{event_name}"), false);
                if prefixed {
                    // allies incl. self (singles: self only, if hp)
                    for ally in self.allies_and_self(p) {
                        handlers.extend(self.find_pokemon_event_handlers(
                            dex,
                            ally,
                            &format!("onAlly{event_name}"),
                            false,
                        ));
                        handlers.extend(self.find_pokemon_event_handlers(
                            dex,
                            ally,
                            &format!("onAny{event_name}"),
                            false,
                        ));
                    }
                    for foe in self.foes_of(p, false) {
                        handlers.extend(self.find_pokemon_event_handlers(
                            dex,
                            foe,
                            &format!("onFoe{event_name}"),
                            false,
                        ));
                        handlers.extend(self.find_pokemon_event_handlers(
                            dex,
                            foe,
                            &format!("onAny{event_name}"),
                            false,
                        ));
                    }
                }
                side_target = Some(p.side);
            }
        }
        if let Some(src) = source {
            if prefixed {
                handlers.extend(self.find_pokemon_event_handlers(
                    dex,
                    src,
                    &format!("onSource{event_name}"),
                    false,
                ));
            }
        }
        if let Some(t_side) = side_target {
            for side_n in 0..2u8 {
                if should_bubble_down {
                    if let Some(active) = self.active_id(side_n as usize) {
                        if side_n == t_side {
                            handlers.extend(self.find_pokemon_event_handlers(
                                dex,
                                active,
                                &format!("on{event_name}"),
                                false,
                            ));
                        } else if prefixed {
                            handlers.extend(self.find_pokemon_event_handlers(
                                dex,
                                active,
                                &format!("onFoe{event_name}"),
                                false,
                            ));
                        }
                        if prefixed {
                            handlers.extend(self.find_pokemon_event_handlers(
                                dex,
                                active,
                                &format!("onAny{event_name}"),
                                false,
                            ));
                        }
                    }
                }
                if side_n == t_side {
                    handlers.extend(self.find_side_event_handlers(
                        dex,
                        side_n,
                        &format!("on{event_name}"),
                        false,
                        None,
                    ));
                } else if prefixed {
                    handlers.extend(self.find_side_event_handlers(
                        dex,
                        side_n,
                        &format!("onFoe{event_name}"),
                        false,
                        None,
                    ));
                }
                if prefixed {
                    handlers.extend(self.find_side_event_handlers(
                        dex,
                        side_n,
                        &format!("onAny{event_name}"),
                        false,
                        None,
                    ));
                }
            }
        }
        handlers.extend(self.find_field_event_handlers(dex, &format!("on{event_name}"), false, None));
        handlers.extend(self.find_battle_event_handlers(&format!("on{event_name}")));
        handlers
    }

    // ----------------------------------------------------------- helpers

    /// side.allies() incl. self (hp only): singles → [self] if hp.
    pub fn allies_and_self(&self, p: PokeId) -> Vec<PokeId> {
        match self.active_id(p.side as usize) {
            Some(a) if self.poke(a).hp > 0 => vec![a],
            _ => vec![],
        }
    }

    /// pokemon.foes(): foe actives with hp (all=false).
    pub fn foes_of(&self, p: PokeId, all: bool) -> Vec<PokeId> {
        match self.active_id(1 - p.side as usize) {
            Some(a) if all || self.poke(a).hp > 0 => vec![a],
            _ => vec![],
        }
    }

    // ------------------------------------------------------- singleEvent

    #[allow(clippy::too_many_arguments)]
    pub fn single_event(
        &mut self,
        dex: &Dex,
        event_id: &str,
        effect: EffectHandle,
        state: StateLoc,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
    ) -> RV {
        if self.event_depth >= 8 {
            panic!("singleEvent stack overflow at {event_id}");
        }
        let has_relay = relay.is_some();
        let relay = relay.unwrap_or(RV::True);

        // status-changed guard
        if let (EffectHandle::Cond(c), EvTarget::Poke(p)) = (effect, target) {
            if dex.cond_effect_type(c) == EffectType::Status
                && self.poke(p).status.as_str() != dex.conds_key(c)
            {
                return relay;
            }
        }

        let has_cb = match effect {
            EffectHandle::Cond(c) => self.cond_has_callback(dex, c, &format!("on{event_id}")),
            EffectHandle::MoveEff(m) => {
                super::moveexec::move_has_callback(dex, m, &format!("on{event_id}"))
            }
            EffectHandle::Item(i) => {
                dex.items.get(i).callbacks.iter().any(|c| c == &format!("on{event_id}"))
            }
            _ => false,
        };
        if !has_cb {
            return relay;
        }

        self.effect_stack.push(EffectFrame { effect, state });
        self.event_stack.push(EventFrame {
            id: event_id.to_string(),
            target: target.poke(),
            source,
            effect: source_effect,
            modifier: 1.0,
        });
        self.event_depth += 1;

        let result = super::conditions::dispatch(
            self,
            dex,
            effect,
            &format!("on{event_id}"),
            state,
            target,
            source,
            source_effect,
            relay.clone(),
            has_relay,
        );

        self.event_depth -= 1;
        self.event_stack.pop();
        self.effect_stack.pop();

        match result {
            RV::Undef => relay,
            rv => rv,
        }
    }

    // ---------------------------------------------------------- runEvent

    #[allow(clippy::too_many_arguments)]
    pub fn run_event(
        &mut self,
        dex: &Dex,
        event_id: &str,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
        on_effect: bool,
        fast_exit: bool,
    ) -> RV {
        if self.event_depth >= 8 {
            panic!("runEvent stack overflow at {event_id}");
        }
        let mut handlers = self.find_event_handlers(dex, target, event_id, source);
        if on_effect {
            // sourceEffect's own callback runs first (used by Damage event)
            let has_cb = match source_effect {
                EffectHandle::Cond(c) => self.cond_has_callback(dex, c, &format!("on{event_id}")),
                EffectHandle::MoveEff(m) => {
                    super::moveexec::move_has_callback(dex, m, &format!("on{event_id}"))
                }
                _ => false,
            };
            if has_cb {
                let l = self.resolve_priority(
                    dex,
                    Listener {
                        effect: source_effect,
                        has_callback: true,
                        state: StateLoc::None,
                        holder: match target {
                            EvTarget::Poke(p) => Holder::Poke(p),
                            EvTarget::Side(s) => Holder::Side(s),
                            EvTarget::Battle => Holder::Battle,
                        },
                        end_kind: EndKind::None,
                        order: None,
                        priority: 0.0,
                        sub_order: 0.0,
                        speed: 0.0,
                        effect_order: 0.0,
                        state_token: 0,
                    },
                    &format!("on{event_id}"),
                );
                handlers.insert(0, l);
            }
        }

        if matches!(event_id, "Invulnerability" | "TryHit" | "DamagingHit" | "EntryHazard") {
            // stable sort by compareLeftToRightOrder
            handlers.sort_by(|a, b| {
                compare_left_to_right(a, b).partial_cmp(&0.0).unwrap()
            });
        } else if fast_exit {
            handlers.sort_by(|a, b| compare_redirect_order(a, b).partial_cmp(&0.0).unwrap());
        } else {
            self.speed_sort(&mut handlers, compare_priority);
        }

        let has_relay = relay.is_some();
        let mut relay = relay.unwrap_or(RV::True);

        self.event_stack.push(EventFrame {
            id: event_id.to_string(),
            target: target.poke(),
            source,
            effect: source_effect,
            modifier: 1.0,
        });
        self.event_depth += 1;

        for handler in &handlers {
            // status-changed guard on holder
            if let (EffectHandle::Cond(c), Holder::Poke(p)) = (handler.effect, handler.holder) {
                if dex.cond_effect_type(c) == EffectType::Status
                    && self.poke(p).status.as_str() != dex.conds_key(c)
                {
                    continue;
                }
            }
            if !handler.has_callback {
                continue;
            }
            self.effect_stack.push(EffectFrame { effect: handler.effect, state: handler.state });
            let holder_target = match handler.holder {
                Holder::Poke(p) => EvTarget::Poke(p),
                Holder::Side(s) => EvTarget::Side(s),
                _ => EvTarget::Battle,
            };
            // PS passes the EVENT's args (target/source/sourceEffect), not
            // the holder, to the callback; effectState.target is the holder.
            let _ = holder_target;
            let result = super::conditions::dispatch(
                self,
                dex,
                handler.effect,
                &format!("on{event_id}"),
                handler.state,
                target,
                source,
                source_effect,
                relay.clone(),
                has_relay,
            );
            self.effect_stack.pop();

            if result != RV::Undef {
                relay = result;
                if !relay.truthy() || fast_exit {
                    break;
                }
            }
        }

        self.event_depth -= 1;
        let frame = self.event_stack.pop().unwrap();

        // final modifier application on non-negative integer relay
        if let RV::Num(n) = relay {
            if n >= 0.0 && n.fract() == 0.0 {
                relay = RV::Num(self.modify(n, frame.modifier, 1.0));
            }
        }
        relay
    }

    pub fn priority_event(
        &mut self,
        dex: &Dex,
        event_id: &str,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
    ) -> RV {
        self.run_event(dex, event_id, target, source, source_effect, relay, false, true)
    }

    // --------------------------------------------------------- eachEvent

    pub fn each_event(&mut self, dex: &Dex, event_id: &str, effect: Option<EffectHandle>) {
        let mut actives = self.get_all_active(false);
        if std::env::var_os("NC_TRACE").is_some() {
            let speeds: Vec<i32> = actives.iter().map(|&p| self.poke(p).speed).collect();
            eprintln!("[trace] eachEvent {event_id} actives={} speeds={:?} seed={}", actives.len(), speeds, self.prng.seed_str());
        }
        let effect = effect.unwrap_or_else(|| self.current_effect());
        // speedSort by speed desc
        let speeds: Vec<(PokeId, f64)> =
            actives.iter().map(|&p| (p, self.poke(p).speed as f64)).collect();
        let mut list = speeds;
        self.speed_sort(&mut list, |a, b| b.1 - a.1);
        actives = list.into_iter().map(|(p, _)| p).collect();
        for pokemon in actives {
            self.run_event(dex, event_id, EvTarget::Poke(pokemon), None, effect, None, false, false);
        }
    }

    // -------------------------------------------------------- fieldEvent

    /// fieldEvent('Residual') / fieldEvent('SwitchIn', targets).
    pub fn field_event(&mut self, dex: &Dex, event_id: &str, targets: Option<&[PokeId]>) {
        let callback_name = format!("on{event_id}");
        let get_key = event_id == "Residual";
        let mut handlers =
            self.find_field_event_handlers(dex, &format!("onField{event_id}"), get_key, None);
        for side_n in 0..2u8 {
            handlers.extend(self.find_side_event_handlers(
                dex,
                side_n,
                &format!("onSide{event_id}"),
                get_key,
                None,
            ));
            if let Some(active) = self.active_id(side_n as usize) {
                if event_id == "SwitchIn" {
                    handlers.extend(self.find_pokemon_event_handlers(
                        dex,
                        active,
                        &format!("onAny{event_id}"),
                        false,
                    ));
                }
                if let Some(ts) = targets {
                    if !ts.contains(&active) {
                        continue;
                    }
                }
                handlers.extend(self.find_pokemon_event_handlers(dex, active, &callback_name, get_key));
                handlers.extend(self.find_side_event_handlers(
                    dex,
                    side_n,
                    &callback_name,
                    false,
                    Some(active),
                ));
                handlers.extend(self.find_field_event_handlers(dex, &callback_name, false, Some(active)));
                // battle handlers: none.
            }
        }
        self.speed_sort(&mut handlers, compare_priority);
        while !handlers.is_empty() {
            let handler = handlers.remove(0);
            // fainted holder (unless slot condition)
            if let Holder::Poke(p) = handler.holder {
                if self.poke(p).fainted && !matches!(handler.state, StateLoc::SlotCond(..)) {
                    continue;
                }
            }
            if event_id == "Residual" && handler.end_kind != EndKind::None {
                // PS decrements the captured state object; only do it if the
                // state at this location is still that object.
                let fresh = self
                    .state_at(handler.state)
                    .map(|s| s.effect_order == handler.state_token)
                    .unwrap_or(false);
                let dur = if fresh { self.state_at(handler.state).and_then(|s| s.duration) } else { None };
                if let Some(d) = dur {
                    if d > 0 {
                        let state = self.state_at_mut(handler.state).unwrap();
                        state.duration = Some(d - 1);
                        if d - 1 == 0 {
                            self.run_end_kind(dex, handler.end_kind, handler.state, handler.holder);
                            if self.ended {
                                return;
                            }
                            continue;
                        }
                    }
                }
            }

            // staleness: the state at this location must still be the same
            // object (identity via effect_order token).
            match handler.state {
                StateLoc::None | StateLoc::Format => {}
                loc => match self.state_at(loc) {
                    Some(s) if s.effect_order == handler.state_token => {}
                    _ => continue,
                },
            }

            let handler_event = match handler.holder {
                Holder::Side(_) => format!("Side{event_id}"),
                Holder::Field => format!("Field{event_id}"),
                _ => event_id.to_string(),
            };
            if handler.has_callback {
                let target = match handler.holder {
                    Holder::Poke(p) => EvTarget::Poke(p),
                    Holder::Side(s) => EvTarget::Side(s),
                    _ => EvTarget::Battle,
                };
                self.single_event(
                    dex,
                    &handler_event,
                    handler.effect,
                    handler.state,
                    target,
                    None,
                    EffectHandle::None,
                    None,
                );
            }
            self.faint_messages(dex, false);
            if self.ended {
                return;
            }
        }
    }

    fn run_end_kind(&mut self, dex: &Dex, kind: EndKind, state: StateLoc, holder: Holder) {
        match (kind, state, holder) {
            (EndKind::RemoveVolatile, StateLoc::Volatile(p, c), _) => {
                self.remove_volatile_id(dex, p, c);
            }
            (EndKind::ClearStatus, StateLoc::Status(p), _) => {
                self.pokemon_clear_status(dex, p);
            }
            (EndKind::RemoveSideCond, StateLoc::SideCond(s, c), _) => {
                self.remove_side_condition(dex, s, c);
            }
            (EndKind::ClearWeather, _, _) => {
                self.clear_weather(dex);
            }
            (EndKind::RemovePseudoWeather, StateLoc::PseudoWeather(c), _) => {
                self.remove_pseudo_weather(dex, c);
            }
            _ => {}
        }
    }

    // ------------------------------------------------------ modify/chain

    /// PS `modify(value, numerator, denominator)` with 4096 fixed point.
    pub fn modify(&self, value: f64, numerator: f64, denominator: f64) -> f64 {
        let modifier = tr(numerator * 4096.0 / denominator);
        tr((tr(value * modifier) + 2048.0 - 1.0) / 4096.0)
    }

    /// PS `chainModify` on the current event frame.
    pub fn chain_modify(&mut self, numerator: f64, denominator: f64) {
        let frame = self.event_stack.last_mut().expect("chainModify outside event");
        let previous = tr(frame.modifier * 4096.0);
        let next = tr(numerator * 4096.0 / denominator);
        frame.modifier = (((previous * next) as i64 + 2048) >> 12) as f64 / 4096.0;
    }
}

use super::tr;
