//! The event system: a faithful port of PS `runEvent`/`singleEvent`/
//! `fieldEvent`/`eachEvent` + `findEventHandlers`/`resolvePriority`/
//! `speedSort`. Handler ordering (including PRNG-consuming tie shuffles) is
//! the load-bearing part — treat every deviation as a conformance bug.
//!
//! Identity is integer end-to-end (M4): every event is a `&'static Ev` whose
//! per-prefix callback ids (`Cb`) resolve once per process against the loaded
//! dex; handler membership is a `CbMask` bit test. PS's string composition
//! (`'on' + event`) happens only in the one-time `EvCbs` build.

use crate::dex::{Cb, Dex, EffectType};
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

/// An event kind. Declared once as a static; the per-dex callback ids are
/// resolved lazily on first use (one dex per process — the same assumption
/// the string-interning design documented).
pub struct Ev {
    pub name: &'static str,
    cell: std::sync::OnceLock<EvCbs>,
}

/// Precomputed per-event callback ids + dispatch flags.
#[derive(Clone, Copy, Debug)]
pub struct EvCbs {
    pub on: Cb,
    pub on_ally: Cb,
    pub on_any: Cb,
    pub on_foe: Cb,
    pub on_source: Cb,
    pub on_field: Cb,
    pub on_side: Cb,
    /// PS-composed singleEvent ids for Side/Field holders in fieldEvent.
    pub side_name: &'static str,
    pub field_name: &'static str,
    /// Can any condition/item in this format handle this event at all?
    /// (A runEvent with zero handlers is side-effect-free in PS too — it
    /// consumes no PRNG and returns the relay unchanged.)
    pub possible: bool,
    /// findEventHandlers runs the ally/foe/source prefix dance.
    pub prefixed: bool,
    /// runEvent sorts by compareLeftToRightOrder for this event.
    pub ltr: bool,
}

impl Ev {
    pub const fn new(name: &'static str) -> Ev {
        Ev { name, cell: std::sync::OnceLock::new() }
    }

    #[inline]
    pub fn cbs(&self, dex: &Dex) -> EvCbs {
        *self.cell.get_or_init(|| self.build(dex))
    }

    fn build(&self, dex: &Dex) -> EvCbs {
        let name = self.name;
        let get = |p: &str| dex.cb(&format!("{p}{name}"));
        let on = get("on");
        let on_ally = get("onAlly");
        let on_any = get("onAny");
        let on_foe = get("onFoe");
        let on_source = get("onSource");
        EvCbs {
            on,
            on_ally,
            on_any,
            on_foe,
            on_source,
            on_field: get("onField"),
            on_side: get("onSide"),
            side_name: Box::leak(format!("Side{name}").into_boxed_str()),
            field_name: Box::leak(format!("Field{name}").into_boxed_str()),
            possible: [on, on_ally, on_any, on_foe, on_source]
                .iter()
                .any(|c| dex.possible_mask.has(*c)),
            prefixed: !matches!(
                name,
                "BeforeTurn" | "Update" | "Weather" | "WeatherChange" | "TerrainChange"
            ),
            ltr: matches!(name, "Invulnerability" | "TryHit" | "DamagingHit" | "EntryHazard"),
        }
    }
}

macro_rules! declare_events {
    ($($n:ident),* $(,)?) => {
        /// One static per event kind reachable in this port. Call sites pass
        /// `&ev::Name`.
        #[allow(non_upper_case_globals)]
        pub mod ev {
            use super::Ev;
            $(pub static $n: Ev = Ev::new(stringify!($n));)*
        }
    };
}

declare_events!(
    Accuracy,
    AfterBoost,
    AfterEachBoost,
    AfterFaint,
    AfterHit,
    AfterMove,
    AfterMoveSecondary,
    AfterMoveSecondarySelf,
    AfterMoveSelf,
    AfterSetStatus,
    AfterSubDamage,
    AfterSwitchInSelf,
    AfterTakeItem,
    AfterUseItem,
    Attract,
    BasePower,
    BeforeFaint,
    BeforeMove,
    BeforeSwitchIn,
    BeforeSwitchOut,
    BeforeTurn,
    ChargeMove,
    CriticalHit,
    Damage,
    DisableMove,
    DragOut,
    Eat,
    EatItem,
    Effectiveness,
    End,
    EntryHazard,
    Faint,
    FieldEnd,
    FieldRestart,
    FieldStart,
    Flinch,
    FractionalPriority,
    Heal,
    Hit,
    HitField,
    HitProtect,
    HitSide,
    Immunity,
    Invulnerability,
    LockMove,
    MaybeTrapPokemon,
    ModifyAccuracy,
    ModifyCritRatio,
    ModifyDamage,
    ModifyMove,
    ModifyPriority,
    MoveAborted,
    MoveFail,
    NegateImmunity,
    OverrideAction,
    PrepareHit,
    Residual,
    Restart,
    SemiLockMove,
    SetStatus,
    SetWeather,
    SideConditionStart,
    SideEnd,
    SideRestart,
    SideStart,
    StallMove,
    Start,
    SwitchIn,
    SwitchOut,
    TakeItem,
    TrapPokemon,
    Try,
    TryAddVolatile,
    TryBoost,
    TryEatItem,
    TryHeal,
    TryHit,
    TryHitField,
    TryHitSide,
    TryImmunity,
    TryMove,
    TryPrimaryHit,
    TrySecondaryHit,
    Update,
    Use,
    UseItem,
    UseMoveMessage,
    Weather,
    WeatherChange,
);

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
    /// The (possibly prefixed) callback id this listener was collected under
    /// — onSourceAccuracy, onFoeBeforeSwitchOut, ... Dispatch uses it.
    pub cb: Cb,
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

/// `NC_TRACE` env flag, read once (the lookup is too slow for hot paths).
pub(crate) fn trace_enabled() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var_os("NC_TRACE").is_some())
}

impl Battle {
    /// PS speedSort: selection sort; full ties shuffled with the battle PRNG.
    pub fn speed_sort<T>(&mut self, list: &mut [T], cmp: impl Fn(&T, &T) -> f64) {
        if list.len() < 2 {
            return;
        }
        let mut sorted = 0;
        let mut next_indexes: smallvec::SmallVec<[usize; 16]> = smallvec::SmallVec::new();
        while sorted + 1 < list.len() {
            next_indexes.clear();
            next_indexes.push(sorted);
            for i in (sorted + 1)..list.len() {
                let delta = cmp(&list[next_indexes[0]], &list[i]);
                if delta < 0.0 {
                    continue;
                }
                if delta > 0.0 {
                    next_indexes.clear();
                    next_indexes.push(i);
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
                if trace_enabled() {
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

    /// Union of status/volatile/item masks for one pokemon (slot conditions
    /// excluded — they are walked separately).
    pub fn recompute_poke_mask(&self, dex: &Dex, id: PokeId) -> crate::dex::CbMask {
        let p = self.poke(id);
        let mut m = crate::dex::CbMask::EMPTY;
        if p.status != Status::None && p.status != Status::Fnt {
            if let Some(c) = dex.status_conds[p.status as usize] {
                m.or_with(&dex.cond(c).mask);
            }
        }
        for (c, _) in &p.volatiles {
            m.or_with(&dex.cond(*c).mask);
        }
        if let Some(item) = p.item {
            m.or_with(&dex.items.get(item).mask);
        }
        m
    }

    pub fn refresh_poke_mask(&mut self, dex: &Dex, id: PokeId) {
        self.poke_mut(id).handler_mask = self.recompute_poke_mask(dex, id);
    }

    pub fn recompute_side_mask(&self, dex: &Dex, side_n: u8) -> crate::dex::CbMask {
        let mut m = crate::dex::CbMask::EMPTY;
        for (c, _) in &self.sides[side_n as usize].side_conditions {
            m.or_with(&dex.cond(*c).mask);
        }
        m
    }

    pub fn refresh_side_mask(&mut self, dex: &Dex, side_n: u8) {
        self.sides[side_n as usize].handler_mask = self.recompute_side_mask(dex, side_n);
    }

    /// resolvePriority: fill sort metadata for a listener.
    fn resolve_priority(&self, dex: &Dex, mut h: Listener, cb: Cb) -> Listener {
        h.cb = cb;
        if let EffectHandle::Cond(c) = h.effect {
            let n = dex.cond(c).cb_num(cb);
            h.order = n.order.map(|v| v as i64);
            h.priority = n.priority.unwrap_or(0) as f64;
            h.sub_order = n.sub_order.unwrap_or(0) as f64;
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
        } else if let EffectHandle::Item(i) = h.effect {
            let n = dex.items.get(i).cb_num(cb);
            h.order = n.order.map(|v| v as i64);
            h.priority = n.priority.unwrap_or(0) as f64;
            h.sub_order = n.sub_order.unwrap_or(0) as f64;
            if h.sub_order == 0.0 {
                h.sub_order = 8.0;
            }
        }
        let ends_switch_in = dex.cb_ends_switch_in(cb);
        if ends_switch_in || dex.cb_ends_redirect_target(cb) {
            h.effect_order = self.state_at(h.state).map(|s| s.effect_order as f64).unwrap_or(0.0);
        }
        if let Holder::Poke(p) = h.holder {
            h.speed = self.poke(p).speed as f64;
            if ends_switch_in {
                let fpv = self.field_position_value(p);
                let idx = self.speed_order.iter().position(|&v| v == fpv).map(|i| i as f64).unwrap_or(-1.0);
                h.speed -= idx / 2.0;
            }
        }
        h
    }

    /// findPokemonEventHandlers: status → volatiles → (ability) → item →
    /// (species) → slot conditions.
    pub fn find_pokemon_event_handlers(
        &self,
        dex: &Dex,
        pokemon: PokeId,
        cb: Cb,
        get_key_duration: bool,
        handlers: &mut Vec<Listener>,
    ) {
        let p = self.poke(pokemon);
        debug_assert_eq!(
            p.handler_mask,
            self.recompute_poke_mask(dex, pokemon),
            "stale handler_mask on {pokemon:?}"
        );
        if !get_key_duration
            && !p.handler_mask.has(cb)
            && (p.position != 0 || self.sides[pokemon.side as usize].slot_conditions.is_empty())
        {
            return;
        }

        // status
        if p.status != Status::None && p.status != Status::Fnt {
            if let Some(c) = dex.status_conds[p.status as usize] {
                let has_cb = dex.cond(c).mask.has(cb);
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
                            cb: Cb::NONE,
                        },
                        cb,
                    ));
                }
            }
        }
        // volatiles (insertion order)
        for (c, state) in &p.volatiles {
            let has_cb = dex.cond(*c).mask.has(cb);
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
                        cb: Cb::NONE,
                    },
                    cb,
                ));
            }
        }
        // ability: 'No Ability' has no handlers — skip.
        // item (milestone 2: item callbacks)
        if let Some(item) = p.item {
            let has_cb = dex.items.get(item).mask.has(cb);
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
                        cb: Cb::NONE,
                    },
                    cb,
                ));
            }
        }
        // species: no runtime species conditions in gen2.
        // slot conditions (futuremove — milestone 2)
        let side = &self.sides[pokemon.side as usize];
        if p.position == 0 {
            for (c, state) in &side.slot_conditions {
                let has_cb = dex.cond(*c).mask.has(cb);
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
                            cb: Cb::NONE,
                        },
                        cb,
                    ));
                }
            }
        }
    }

    pub fn find_side_event_handlers(
        &self,
        dex: &Dex,
        side_n: u8,
        cb: Cb,
        get_key_duration: bool,
        custom_holder: Option<PokeId>,
        handlers: &mut Vec<Listener>,
    ) {
        debug_assert_eq!(
            self.sides[side_n as usize].handler_mask,
            self.recompute_side_mask(dex, side_n),
            "stale handler_mask on side {side_n}"
        );
        if !get_key_duration && !self.sides[side_n as usize].handler_mask.has(cb) {
            return;
        }
        for (c, state) in &self.sides[side_n as usize].side_conditions {
            let has_cb = dex.cond(*c).mask.has(cb);
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
                        cb: Cb::NONE,
                    },
                    cb,
                ));
            }
        }
    }

    pub fn find_field_event_handlers(
        &self,
        dex: &Dex,
        cb: Cb,
        get_key_duration: bool,
        custom_holder: Option<PokeId>,
        handlers: &mut Vec<Listener>,
    ) {
        for (c, state) in &self.field.pseudo_weather {
            let has_cb = dex.cond(*c).mask.has(cb);
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
                        cb: Cb::NONE,
                    },
                    cb,
                ));
            }
        }
        if let Some(w) = self.field.weather {
            let has_cb = dex.cond(w).mask.has(cb);
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
                        cb: Cb::NONE,
                    },
                    cb,
                ));
            }
        }
        // terrain: none in gen2.
    }

    /// findEventHandlers (the full prefix dance).
    pub fn find_event_handlers(
        &self,
        dex: &Dex,
        target: EvTarget,
        cbs: EvCbs,
        source: Option<PokeId>,
        handlers: &mut Vec<Listener>,
    ) {
        let prefixed = cbs.prefixed;
        let should_bubble_down = matches!(target, EvTarget::Side(_));
        let mut side_target: Option<u8> = match target {
            EvTarget::Side(s) => Some(s),
            _ => None,
        };

        if let EvTarget::Poke(p) = target {
            let target_active = self.poke(p).is_active;
            let source_active = source.map(|s| self.poke(s).is_active).unwrap_or(false);
            if target_active || source_active {
                self.find_pokemon_event_handlers(dex, p, cbs.on, false, handlers);
                if prefixed {
                    // allies incl. self (singles: self only, if hp)
                    if let Some(ally) = self.ally_of(p) {
                        self.find_pokemon_event_handlers(dex, ally, cbs.on_ally, false, handlers);
                        self.find_pokemon_event_handlers(dex, ally, cbs.on_any, false, handlers);
                    }
                    if let Some(foe) = self.foe_of(p, false) {
                        self.find_pokemon_event_handlers(dex, foe, cbs.on_foe, false, handlers);
                        self.find_pokemon_event_handlers(dex, foe, cbs.on_any, false, handlers);
                    }
                }
                side_target = Some(p.side);
            }
        }
        if let Some(src) = source {
            if prefixed {
                self.find_pokemon_event_handlers(dex, src, cbs.on_source, false, handlers);
            }
        }
        if let Some(t_side) = side_target {
            for side_n in 0..2u8 {
                if should_bubble_down {
                    if let Some(active) = self.active_id(side_n as usize) {
                        if side_n == t_side {
                            self.find_pokemon_event_handlers(dex, active, cbs.on, false, handlers);
                        } else if prefixed {
                            self.find_pokemon_event_handlers(dex, active, cbs.on_foe, false, handlers);
                        }
                        if prefixed {
                            self.find_pokemon_event_handlers(dex, active, cbs.on_any, false, handlers);
                        }
                    }
                }
                if side_n == t_side {
                    self.find_side_event_handlers(dex, side_n, cbs.on, false, None, handlers);
                } else if prefixed {
                    self.find_side_event_handlers(dex, side_n, cbs.on_foe, false, None, handlers);
                }
                if prefixed {
                    self.find_side_event_handlers(dex, side_n, cbs.on_any, false, None, handlers);
                }
            }
        }
        self.find_field_event_handlers(dex, cbs.on, false, None, handlers);
        // findBattleEventHandlers: the format itself has no runtime handlers
        // in NC2000 (rules act via pseudo-weathers).
    }

    // ----------------------------------------------------------- helpers

    /// side.allies() incl. self (hp only): singles → self if hp.
    fn ally_of(&self, p: PokeId) -> Option<PokeId> {
        match self.active_id(p.side as usize) {
            Some(a) if self.poke(a).hp > 0 => Some(a),
            _ => None,
        }
    }

    /// pokemon.foes(): foe active with hp (all=false).
    fn foe_of(&self, p: PokeId, all: bool) -> Option<PokeId> {
        match self.active_id(1 - p.side as usize) {
            Some(a) if all || self.poke(a).hp > 0 => Some(a),
            _ => None,
        }
    }

    /// side.allies() as a list (kept for non-hot callers).
    pub fn allies_and_self(&self, p: PokeId) -> Vec<PokeId> {
        self.ally_of(p).into_iter().collect()
    }

    /// pokemon.foes() as a list (kept for non-hot callers).
    pub fn foes_of(&self, p: PokeId, all: bool) -> Vec<PokeId> {
        self.foe_of(p, all).into_iter().collect()
    }

    /// Grab a scratch listener buffer (returned by `put_scratch`). The pool
    /// lives on the battle; clones inherit empty vectors (capacity is not
    /// cloned), so this never affects state comparison or clone cost.
    fn take_scratch(&mut self) -> Vec<Listener> {
        self.listener_pool.0.pop().unwrap_or_default()
    }

    fn put_scratch(&mut self, mut v: Vec<Listener>) {
        v.clear();
        self.listener_pool.0.push(v);
    }

    // ------------------------------------------------------- singleEvent

    #[allow(clippy::too_many_arguments)]
    pub fn single_event(
        &mut self,
        dex: &Dex,
        ev: &'static Ev,
        effect: EffectHandle,
        state: StateLoc,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
    ) -> RV {
        let cb = ev.cbs(dex).on;
        self.single_event_cb(dex, ev.name, cb, effect, state, target, source, source_effect, relay)
    }

    #[allow(clippy::too_many_arguments)]
    fn single_event_cb(
        &mut self,
        dex: &Dex,
        event_name: &'static str,
        cb: Cb,
        effect: EffectHandle,
        state: StateLoc,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
    ) -> RV {
        if self.event_depth >= 8 {
            panic!("singleEvent stack overflow at {event_name}");
        }
        let has_relay = relay.is_some();
        let relay = relay.unwrap_or(RV::True);

        // status-changed guard
        if let (EffectHandle::Cond(c), EvTarget::Poke(p)) = (effect, target) {
            if dex.cond_effect_type(c) == EffectType::Status
                && dex.status_conds[self.poke(p).status as usize] != Some(c)
            {
                return relay;
            }
        }

        let has_cb = match effect {
            EffectHandle::Cond(c) => dex.cond(c).mask.has(cb),
            EffectHandle::MoveEff(m) => {
                super::moveexec::active_move_has_callback(self, dex, m, cb)
            }
            EffectHandle::Item(i) => dex.items.get(i).mask.has(cb),
            _ => false,
        };
        if !has_cb {
            return relay;
        }

        self.effect_stack.push(EffectFrame { effect, state });
        self.event_stack.push(EventFrame {
            id: event_name,
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
            cb,
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
        ev: &'static Ev,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
        on_effect: bool,
        fast_exit: bool,
    ) -> RV {
        if self.event_depth >= 8 {
            panic!("runEvent stack overflow at {}", ev.name);
        }
        let cbs = ev.cbs(dex);
        let mut handlers = self.take_scratch();
        if cbs.possible {
            self.find_event_handlers(dex, target, cbs, source, &mut handlers);
        }
        if on_effect {
            // sourceEffect's own callback runs first (used by Damage event)
            let has_cb = match source_effect {
                EffectHandle::Cond(c) => dex.cond(c).mask.has(cbs.on),
                EffectHandle::MoveEff(m) => {
                    super::moveexec::active_move_has_callback(self, dex, m, cbs.on)
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
                        cb: Cb::NONE,
                    },
                    cbs.on,
                );
                handlers.insert(0, l);
            }
        }

        if cbs.ltr {
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
            id: ev.name,
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
                    && dex.status_conds[self.poke(p).status as usize] != Some(c)
                {
                    continue;
                }
            }
            if !handler.has_callback {
                continue;
            }
            self.effect_stack.push(EffectFrame { effect: handler.effect, state: handler.state });
            let result = super::conditions::dispatch(
                self,
                dex,
                handler.effect,
                handler.cb,
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
        self.put_scratch(handlers);

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
        ev: &'static Ev,
        target: EvTarget,
        source: Option<PokeId>,
        source_effect: EffectHandle,
        relay: Option<RV>,
    ) -> RV {
        self.run_event(dex, ev, target, source, source_effect, relay, false, true)
    }

    // --------------------------------------------------------- eachEvent

    pub fn each_event(&mut self, dex: &Dex, ev: &'static Ev, effect: Option<EffectHandle>) {
        // getAllActive (non-fainted), then speedSort by speed desc — at most
        // one active per side in singles, collected on the stack.
        let mut buf = [(PokeId { side: 0, slot: 0 }, 0.0f64); 2];
        let mut n = 0;
        for side in 0..2 {
            if let Some(id) = self.active_id(side) {
                if !self.poke(id).fainted {
                    buf[n] = (id, self.poke(id).speed as f64);
                    n += 1;
                }
            }
        }
        if trace_enabled() {
            let speeds: Vec<f64> = buf[..n].iter().map(|&(_, s)| s).collect();
            eprintln!("[trace] eachEvent {} actives={n} speeds={:?} seed={}", ev.name, speeds, self.prng.seed_str());
        }
        let effect = effect.unwrap_or_else(|| self.current_effect());
        self.speed_sort(&mut buf[..n], |a, b| b.1 - a.1);
        for i in 0..n {
            let pokemon = buf[i].0;
            self.run_event(dex, ev, EvTarget::Poke(pokemon), None, effect, None, false, false);
        }
    }

    // -------------------------------------------------------- fieldEvent

    /// fieldEvent('Residual') / fieldEvent('SwitchIn', targets).
    pub fn field_event(&mut self, dex: &Dex, ev: &'static Ev, targets: Option<&[PokeId]>) {
        let cbs = ev.cbs(dex);
        let get_key = ev.name == "Residual";
        let mut handlers = self.take_scratch();
        self.find_field_event_handlers(dex, cbs.on_field, get_key, None, &mut handlers);
        for side_n in 0..2u8 {
            self.find_side_event_handlers(dex, side_n, cbs.on_side, get_key, None, &mut handlers);
            if let Some(active) = self.active_id(side_n as usize) {
                if ev.name == "SwitchIn" {
                    self.find_pokemon_event_handlers(dex, active, cbs.on_any, false, &mut handlers);
                }
                if let Some(ts) = targets {
                    if !ts.contains(&active) {
                        continue;
                    }
                }
                self.find_pokemon_event_handlers(dex, active, cbs.on, get_key, &mut handlers);
                self.find_side_event_handlers(dex, side_n, cbs.on, false, Some(active), &mut handlers);
                self.find_field_event_handlers(dex, cbs.on, false, Some(active), &mut handlers);
                // battle handlers: none.
            }
        }
        self.speed_sort(&mut handlers, compare_priority);
        let mut idx = 0;
        while idx < handlers.len() {
            let handler = handlers[idx].clone();
            idx += 1;
            // fainted holder (unless slot condition)
            if let Holder::Poke(p) = handler.holder {
                if self.poke(p).fainted && !matches!(handler.state, StateLoc::SlotCond(..)) {
                    continue;
                }
            }
            if get_key && handler.end_kind != EndKind::None {
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
                                self.put_scratch(handlers);
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

            if handler.has_callback {
                let (frame_name, target) = match handler.holder {
                    Holder::Poke(p) => (ev.name, EvTarget::Poke(p)),
                    Holder::Side(s) => (cbs.side_name, EvTarget::Side(s)),
                    _ => (cbs.field_name, EvTarget::Battle),
                };
                self.single_event_cb(
                    dex,
                    frame_name,
                    handler.cb,
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
                self.put_scratch(handlers);
                return;
            }
        }
        self.put_scratch(handlers);
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

/// A pool of listener buffers reused across nested run/fieldEvents. Lives on
/// the battle so search stepping allocates nothing; clones deliberately start
/// with an empty pool (capacity is scratch, not state).
#[derive(Debug, Default)]
pub struct ScratchPool(pub Vec<Vec<Listener>>);

impl Clone for ScratchPool {
    fn clone(&self) -> Self {
        ScratchPool(Vec::new())
    }
}
