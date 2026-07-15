//! Battle state — a flat, cheaply clonable object graph (no Rc, no back
//! references). Cloning a `Battle` is a plain deep copy suitable for search.
//!
//! Layout mirrors what PS mutates (sim/pokemon.ts, side.ts, field.ts) but only
//! the parts reachable in gen2stadium2 NC2000: singles, 6 registered / 3
//! picked, no abilities, no mega/z/dynamax/tera.
//!
//! Pokemon identity: `PokeId { side, slot }` where `slot` indexes the side's
//! `roster` (construction order, never reordered). PS's mutable
//! `side.pokemon` array is mirrored by `Side::party` (display order); PS's
//! `pokemon.position` is kept in sync exactly like PS does.

use crate::dex::{Accuracy, Category, CondId, FixedDamage, HitEffect, ItemId, MoveId, Multihit, SparseBoosts, SpeciesId};
use crate::prng::Prng;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PokeId {
    pub side: u8,
    pub slot: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Status {
    None,
    Brn,
    Par,
    Slp,
    Frz,
    Psn,
    Tox,
    /// PS sets `status = 'fnt'` on fainted actives at forced-switch time.
    Fnt,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::None => "",
            Status::Brn => "brn",
            Status::Par => "par",
            Status::Slp => "slp",
            Status::Frz => "frz",
            Status::Psn => "psn",
            Status::Tox => "tox",
            Status::Fnt => "fnt",
        }
    }

    pub fn from_str(s: &str) -> Status {
        match s {
            "brn" => Status::Brn,
            "par" => Status::Par,
            "slp" => Status::Slp,
            "frz" => Status::Frz,
            "psn" => Status::Psn,
            "tox" => Status::Tox,
            "fnt" => Status::Fnt,
            _ => Status::None,
        }
    }
}

/// Scalar value inside an effect-state bag (PS `EffectState` holds arbitrary
/// scalars; these are exactly what the fixture essence serializes).
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Bool(bool),
    Str(String),
}

impl Scalar {
    pub fn as_int(&self) -> i64 {
        match self {
            Scalar::Int(v) => *v,
            Scalar::Bool(b) => *b as i64,
            Scalar::Str(_) => 0,
        }
    }
}

/// What PS stores in `EffectState`: scalar keys (serialized into the fixture
/// essence) plus non-scalar references (dropped from the essence but needed
/// at runtime: source Pokemon, sourceEffect).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EffectState {
    pub id: String,
    /// Volatiles carry `name` (addVolatile sets it) — a scalar, in essence.
    pub name: Option<String>,
    pub duration: Option<i32>,
    /// `sourceSlot` — a scalar string ("p1a"), in essence.
    pub source_slot: Option<String>,
    /// `source` — a Pokemon reference, not in essence.
    pub source: Option<PokeId>,
    /// `sourceEffect` — an Effect reference, not in essence. Holds the
    /// effect's id (move id for partiallytrapped, etc.).
    pub source_effect: Option<String>,
    /// Everything else scalar: time, startTime, counter, boundDivisor, move...
    /// Insertion-ordered like a JS object (affects nothing in essence compare,
    /// which is key-based, but keep Vec for faithful iteration anyway).
    pub data: Vec<(String, Scalar)>,
    /// PS initEffectState effectOrder (handler ordering for SwitchIn events).
    pub effect_order: u32,
}

impl EffectState {
    pub fn get(&self, key: &str) -> Option<&Scalar> {
        self.data.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn get_int(&self, key: &str) -> i64 {
        self.get(key).map(|v| v.as_int()).unwrap_or(0)
    }

    pub fn set(&mut self, key: &str, value: Scalar) {
        if let Some(entry) = self.data.iter_mut().find(|(k, _)| k == key) {
            entry.1 = value;
        } else {
            self.data.push((key.to_string(), value));
        }
    }

    pub fn set_int(&mut self, key: &str, value: i64) {
        self.set(key, Scalar::Int(value));
    }

    pub fn remove(&mut self, key: &str) {
        self.data.retain(|(k, _)| k != key);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MoveSlot {
    pub id: MoveId,
    pub pp: i32,
    pub maxpp: i32,
    pub disabled: bool,
    pub used: bool,
    /// PS shares MoveSlot OBJECTS between `moveSlots` and `baseMoveSlots`
    /// (`moveSlots = baseMoveSlots.slice()`), so pp/disabled/used mutations
    /// persist through clearVolatile. `shared` marks slots that mirror writes
    /// into `base_move_slots` (false only for transform/mimic slots — M2).
    pub shared: bool,
}

/// Boost table indices: atk, def, spa, spd, spe, accuracy, evasion.
pub type Boosts = [i8; 7];

/// PS `pokemon.switchFlag`: false | true | move id (selfSwitch).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SwitchFlag {
    No,
    Yes,
    Move(MoveId),
}

impl SwitchFlag {
    pub fn is_set(&self) -> bool {
        !matches!(self, SwitchFlag::No)
    }
}

/// PS `moveThisTurnResult`: undefined | null | false | true.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoveResult {
    #[default]
    Undef,
    Null,
    False,
    True,
}

/// Entry in `pokemon.attackedBy` (Counter/Mirror Coat bookkeeping).
#[derive(Clone, Debug)]
pub struct Attacker {
    pub source: PokeId,
    pub damage: i64,
    pub move_id: MoveId,
    pub this_turn: bool,
    pub slot: String,
    /// PS damageValue: number | false | undefined.
    pub damage_value: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Pokemon {
    // ----- set-derived, fixed for the battle
    pub species: SpeciesId,
    pub name: String,
    pub level: u8,
    /// "", "M", "F"
    pub gender: String,
    pub happiness: u8,
    pub set_ivs: [i32; 6],
    pub set_evs: [i32; 6],
    pub base_move_slots: Vec<MoveSlot>,

    // ----- computed stats
    /// Stored stats (atk, def, spa, spd, spe) after level/DV/stat-exp math.
    pub stored_stats: [i32; 5],
    pub base_maxhp: i32,
    pub maxhp: i32,

    // ----- mutable battle state
    pub hp: i32,
    pub status: Status,
    pub status_state: EffectState,
    pub boosts: Boosts,
    pub move_slots: Vec<MoveSlot>,
    pub item: Option<ItemId>,
    pub last_item: Option<ItemId>,
    pub item_state: EffectState,
    pub types: Vec<String>,
    /// Insertion-ordered (PS object key order drives handler collection).
    pub volatiles: Vec<(CondId, EffectState)>,
    pub transformed: bool,
    pub fainted: bool,
    pub faint_queued: bool,

    pub is_active: bool,
    pub is_started: bool,
    pub position: u8,
    pub active_turns: i32,
    pub active_move_actions: i32,
    pub newly_switched: bool,
    pub being_called_back: bool,
    pub dragged_in: Option<u16>,
    pub previously_switched_in: i32,

    pub switch_flag: SwitchFlag,
    pub force_switch_flag: bool,
    pub skip_before_switch_out: bool,

    pub trapped: bool,
    pub maybe_trapped: bool,

    pub last_move: Option<MoveId>,
    pub last_move_encore: Option<MoveId>,
    pub last_move_used: Option<MoveId>,
    pub last_move_target_loc: Option<i8>,
    pub move_this_turn: Option<MoveId>,
    pub move_this_turn_result: MoveResult,
    pub move_last_turn_result: MoveResult,
    pub hurt_this_turn: Option<i32>,
    pub stats_raised_this_turn: bool,
    pub stats_lowered_this_turn: bool,
    pub used_item_this_turn: bool,
    pub last_damage: i64,
    pub attacked_by: Vec<Attacker>,
    pub times_attacked: i32,

    pub speed: i32,
}

impl Pokemon {
    pub fn volatile(&self, id: CondId) -> Option<&EffectState> {
        self.volatiles.iter().find(|(k, _)| *k == id).map(|(_, v)| v)
    }

    pub fn volatile_mut(&mut self, id: CondId) -> Option<&mut EffectState> {
        self.volatiles.iter_mut().find(|(k, _)| *k == id).map(|(_, v)| v)
    }

    pub fn has_volatile(&self, id: CondId) -> bool {
        self.volatiles.iter().any(|(k, _)| *k == id)
    }

    pub fn has_type(&self, ty: &str) -> bool {
        self.types.iter().any(|t| t == ty)
    }

    pub fn get_move_slot(&self, id: MoveId) -> Option<&MoveSlot> {
        self.move_slots.iter().find(|m| m.id == id)
    }

    pub fn get_move_slot_mut(&mut self, id: MoveId) -> Option<&mut MoveSlot> {
        self.move_slots.iter_mut().find(|m| m.id == id)
    }
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub cant_undo: bool,
    pub error: bool,
    pub actions: Vec<ChosenAction>,
    pub forced_switches_left: u32,
    pub forced_passes_left: u32,
    pub switch_ins: Vec<u8>, // display positions already chosen to switch in
}

impl Default for Choice {
    fn default() -> Self {
        Choice {
            cant_undo: false,
            error: false,
            actions: Vec::new(),
            forced_switches_left: 0,
            forced_passes_left: 0,
            switch_ins: Vec::new(),
        }
    }
}

/// side.ts ChosenAction (gen2 singles slice).
#[derive(Clone, Debug)]
pub enum ChosenAction {
    Move {
        pokemon: PokeId,
        target_loc: i8,
        move_id: MoveId,
        move_slot: Option<usize>,
    },
    Switch {
        insta: bool,
        pokemon: PokeId,
        target: PokeId,
    },
    Team {
        pokemon: PokeId,
        index: u8,
        priority: i32,
    },
    Pass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestKind {
    TeamPreview,
    Move,
    Switch,
    Wait,
}

#[derive(Clone, Debug)]
pub struct Side {
    pub name: String,
    /// All constructed pokemon in construction order — never reordered.
    pub roster: Vec<Pokemon>,
    /// PS `side.pokemon`: display order, mutated by team choice + switches.
    /// Values are roster slots.
    pub party: Vec<u8>,
    /// Roster slot of the active pokemon (singles: one slot).
    pub active: Option<u8>,
    pub pokemon_left: i32,
    pub total_fainted: i32,
    /// Insertion-ordered.
    pub side_conditions: Vec<(CondId, EffectState)>,
    pub slot_conditions: Vec<(CondId, EffectState)>,
    /// Stadium 2 self-KO clause bookkeeping (side.lastMove).
    pub last_move: Option<MoveId>,
    pub fainted_this_turn: Option<u8>,
    pub fainted_last_turn: Option<u8>,
    pub request: Option<RequestKind>,
    pub choice: Choice,
}

impl Side {
    pub fn pokemon_at(&self, position: usize) -> Option<u8> {
        self.party.get(position).copied()
    }

    pub fn side_condition(&self, id: CondId) -> Option<&EffectState> {
        self.side_conditions.iter().find(|(k, _)| *k == id).map(|(_, v)| v)
    }

    pub fn has_side_condition(&self, id: CondId) -> bool {
        self.side_conditions.iter().any(|(k, _)| *k == id)
    }

    /// PS side.requestState (derived from activeRequest).
    pub fn request_state(&self) -> Option<RequestKind> {
        match self.request {
            Some(RequestKind::Wait) | None => None,
            other => other,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Field {
    pub weather: Option<CondId>,
    pub weather_state: EffectState,
    /// Insertion-ordered. Keys are interned runtime cond ids (rule
    /// pseudo-weathers are interned too).
    pub pseudo_weather: Vec<(CondId, EffectState)>,
}

impl Field {
    pub fn has_pseudo_weather(&self, id: CondId) -> bool {
        self.pseudo_weather.iter().any(|(k, _)| *k == id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestState {
    None,
    TeamPreview,
    Move,
    Switch,
}

impl RequestState {
    pub fn as_str(self) -> &'static str {
        match self {
            RequestState::None => "",
            RequestState::TeamPreview => "teampreview",
            RequestState::Move => "move",
            RequestState::Switch => "switch",
        }
    }
}

/// battle-queue.ts Action, resolved (gen2 singles slice).
#[derive(Clone, Debug)]
pub struct Action {
    pub choice: ActionKind,
    pub order: i64,
    pub priority: f64,
    pub fractional_priority: f64,
    pub speed: f64,
    pub pokemon: Option<PokeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ActionKind {
    Start,
    BeforeTurn,
    Residual,
    Team { index: u8 },
    Move { move_id: MoveId, target_loc: i8, original_target: Option<PokeId>, source_effect: Option<MoveId> },
    Switch { insta: bool, target: PokeId, source_effect: Option<MoveId> },
    RunSwitch,
}

#[derive(Clone, Debug)]
pub struct FaintEntry {
    pub target: PokeId,
    pub source: Option<PokeId>,
    pub effect: Option<crate::battle::EffectHandle>,
}

/// Where an `EffectState` lives (PS holds object references; we hold paths).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateLoc {
    Status(PokeId),
    Volatile(PokeId, CondId),
    SideCond(u8, CondId),
    SlotCond(u8, u8, CondId),
    Weather,
    PseudoWeather(CondId),
    Format,
    None,
}

/// Who holds a handler (PS effectHolder).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holder {
    Poke(PokeId),
    Side(u8),
    Field,
    Battle,
}

/// PS `this.event` frame (drives default args of damage/boost/heal).
#[derive(Clone, Debug)]
pub struct EventFrame {
    pub id: String,
    pub target: Option<PokeId>,
    pub source: Option<PokeId>,
    pub effect: crate::battle::EffectHandle,
    pub modifier: f64,
}

/// PS `this.effect`/`this.effectState` frame.
#[derive(Clone, Debug)]
pub struct EffectFrame {
    pub effect: crate::battle::EffectHandle,
    pub state: StateLoc,
}

/// PS ActiveMove: static move data snapshot + per-use mutable state.
#[derive(Clone, Debug)]
pub struct ActiveMove {
    /// None for synthetic moves (confusion self-hit).
    pub id: Option<MoveId>,
    pub name: String,
    pub move_type: String,
    pub base_move_type: String,
    pub category: Category,
    pub base_power: i32,
    pub accuracy: Accuracy,
    pub priority: i8,
    pub target: String,
    pub crit_ratio: i32,
    pub will_crit: Option<bool>,
    pub status: Option<String>,
    pub volatile_status: Option<String>,
    pub side_condition: Option<String>,
    pub weather: Option<String>,
    pub pseudo_weather: Option<String>,
    pub boosts: SparseBoosts,
    pub has_boosts: bool,
    pub heal: Option<(i32, i32)>,
    pub drain: Option<(i32, i32)>,
    pub recoil: Option<(i32, i32)>,
    pub struggle_recoil: bool,
    pub multihit: Option<Multihit>,
    pub secondaries: Vec<HitEffect>,
    pub self_effect: Option<HitEffect>,
    pub damage: Option<FixedDamage>,
    pub ohko: bool,
    pub selfdestruct: bool,
    pub self_switch: Option<String>,
    pub force_switch: bool,
    pub ignore_immunity: bool,
    pub ignore_accuracy: bool,
    pub ignore_evasion: bool,
    pub ignore_positive_evasion: bool,
    pub ignore_offensive: bool,
    pub ignore_defensive: bool,
    pub sleep_usable: bool,
    pub no_damage_variance: bool,
    pub always_hit: bool,
    pub thaws_target: bool,
    pub flags: Vec<String>,
    pub has_callbacks: Vec<String>,
    // ---- per-use mutable
    pub hit: i32,
    pub last_hit: bool,
    pub total_damage: Option<i64>,
    pub source_effect: Option<MoveId>,
    pub is_confusion_self_hit: bool,
    pub spread_hit: bool,
    /// Per-target-slot hit data ("p1a" → (crit, typeMod)).
    pub move_hit_data: Vec<(String, (bool, i32))>,
}

impl ActiveMove {
    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }

    pub fn hit_data_mut(&mut self, slot: String) -> &mut (bool, i32) {
        if !self.move_hit_data.iter().any(|(s, _)| *s == slot) {
            self.move_hit_data.push((slot.clone(), (false, 0)));
        }
        &mut self.move_hit_data.iter_mut().find(|(s, _)| *s == slot).unwrap().1
    }

    pub fn hit_data(&self, slot: &str) -> (bool, i32) {
        self.move_hit_data
            .iter()
            .find(|(s, _)| s == slot)
            .map(|(_, d)| *d)
            .unwrap_or((false, 0))
    }
}

#[derive(Clone, Debug)]
pub struct Battle {
    pub prng: Prng,
    pub turn: u16,
    pub request_state: RequestState,
    pub mid_turn: bool,
    pub started: bool,
    pub ended: bool,
    /// Winner side name ("P1"/"P2"), "" = tie, None = undecided.
    pub winner: Option<String>,
    pub field: Field,
    pub sides: [Side; 2],
    pub queue: Vec<Action>,
    pub faint_queue: Vec<FaintEntry>,
    pub log: Vec<String>,
    pub effect_order: u32,
    pub event_depth: u32,
    pub last_move_line: i64,
    pub last_successful_move_this_turn: Option<MoveId>,
    pub last_damage: i64,
    pub quick_claw_roll: bool,
    /// Field-position values sorted by speed at last runSwitch (resolvePriority).
    pub speed_order: Vec<usize>,
    /// Format data effect state (unused scalars, kept for parity).
    pub format_data: EffectState,
    /// Log length threshold bookkeeping (PS sentLogPos, used only by the
    /// singleEvent infinite-loop guard; we track it for parity of behavior).
    pub sent_log_pos: usize,
    // ---- event machinery (empty between turns; PS this.event/this.effect)
    pub event_stack: Vec<EventFrame>,
    pub effect_stack: Vec<EffectFrame>,
    pub active_move: Option<ActiveMove>,
    pub active_pokemon: Option<PokeId>,
    pub active_target: Option<PokeId>,
    pub last_move_id: Option<MoveId>,
    /// Cached dex key of `field.weather` ('' if none) — lets weather checks
    /// avoid threading `dex` everywhere.
    pub field_weather_key: String,
}

/// Sparse boosts as an ordered list (PS object iteration order).
pub type SparseBoostsOwned = Vec<(usize, i8)>;

/// Convenience: full 7-slot boost names.
pub const BOOST_NAMES: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

/// Map BTreeMap-based scalar bags (unused placeholder to keep serde imports out).
pub type ScalarMap = BTreeMap<String, Scalar>;
