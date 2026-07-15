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

use crate::dex::{Accuracy, Category, CondId, FixedDamage, HitEffect, ItemId, MoveId, Multihit, SparseBoosts, SpeciesId, TypeId, TypeList};
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
/// scalars; these are exactly what the fixture essence serializes). String
/// values are stored as interned ids and rendered back at essence time
/// (`MoveK` → move key, `CondK` → condition key, `Slot` → "p1a").
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
    Bool(bool),
    MoveK(MoveId),
    CondK(CondId),
    Slot(u8, u8),
}

impl Scalar {
    pub fn as_int(&self) -> i64 {
        match self {
            Scalar::Int(v) => *v,
            Scalar::Float(v) => *v as i64,
            Scalar::Bool(b) => *b as i64,
            _ => 0,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Scalar::Int(v) => *v as f64,
            Scalar::Float(v) => *v,
            Scalar::Bool(b) => *b as i64 as f64,
            _ => 0.0,
        }
    }

    pub fn as_move(&self) -> Option<MoveId> {
        match self {
            Scalar::MoveK(m) => Some(*m),
            _ => None,
        }
    }
}

/// The identity an `EffectState` belongs to (PS stores the effect's string
/// id; we store the interned handle and render the string at essence time).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EffId {
    #[default]
    None,
    Cond(CondId),
    Item(ItemId),
    Format,
}

impl EffId {
    pub fn is_empty(self) -> bool {
        self == EffId::None
    }

    pub fn cond(self) -> Option<CondId> {
        match self {
            EffId::Cond(c) => Some(c),
            _ => None,
        }
    }
}

/// Keys of scalar data PS effects store on their state (fixed universe for
/// this format; `as_str` must render the exact PS key for the essence).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DK {
    BoundDivisor,
    ContactHitCount,
    Counter,
    EndingTurn,
    HitCount,
    Hp,
    Layers,
    LinkedStatus,
    Move,
    Multiplier,
    StartTime,
    TargetLoc,
    TargetSlot,
    Time,
    TotalDamage,
}

impl DK {
    pub fn as_str(self) -> &'static str {
        match self {
            DK::BoundDivisor => "boundDivisor",
            DK::ContactHitCount => "contactHitCount",
            DK::Counter => "counter",
            DK::EndingTurn => "endingTurn",
            DK::HitCount => "hitCount",
            DK::Hp => "hp",
            DK::Layers => "layers",
            DK::LinkedStatus => "linkedStatus",
            DK::Move => "move",
            DK::Multiplier => "multiplier",
            DK::StartTime => "startTime",
            DK::TargetLoc => "targetLoc",
            DK::TargetSlot => "targetSlot",
            DK::Time => "time",
            DK::TotalDamage => "totalDamage",
        }
    }
}

/// What PS stores in `EffectState`: scalar keys (serialized into the fixture
/// essence) plus non-scalar references (dropped from the essence but needed
/// at runtime: source Pokemon, sourceEffect).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EffectState {
    pub id: EffId,
    /// Volatiles carry `name` (addVolatile sets the condition's display name)
    /// — rendered from `id` at essence time when set.
    pub has_name: bool,
    pub duration: Option<i32>,
    /// `sourceSlot` — (side, position), rendered "p1a" in essence.
    pub source_slot: Option<(u8, u8)>,
    /// `source` — a Pokemon reference, not in essence.
    pub source: Option<PokeId>,
    /// `sourceEffect` — an Effect reference, not in essence.
    pub source_effect: Option<crate::battle::EffectHandle>,
    /// `linkedPokemon` — Pokemon references, not in essence (trapped/trapper).
    pub linked_pokemon: smallvec::SmallVec<[PokeId; 2]>,
    /// The paired condition id for linked volatiles. On the *target* side PS
    /// stores the string form (also mirrored into `data` for essence); on the
    /// *source* side it stores the Condition object (essence-invisible).
    pub linked_status: Option<CondId>,
    /// bide `lastDamageSource` (object in PS — essence-invisible).
    pub last_damage_source: Option<PokeId>,
    /// leppaberry `moveSlot` (object reference in PS): move slot index.
    pub slot_ref: Option<usize>,
    /// futuremove `moveData.damage` (nested object — essence-invisible).
    pub future_damage: Option<f64>,
    /// Everything else scalar: time, startTime, counter, boundDivisor, move...
    /// Insertion-ordered like a JS object (affects nothing in essence compare,
    /// which is key-based, but keep order for faithful iteration anyway).
    pub data: smallvec::SmallVec<[(DK, Scalar); 3]>,
    /// PS initEffectState effectOrder (handler ordering for SwitchIn events).
    pub effect_order: u32,
}

impl EffectState {
    pub fn get(&self, key: DK) -> Option<&Scalar> {
        self.data.iter().find(|(k, _)| *k == key).map(|(_, v)| v)
    }

    pub fn get_int(&self, key: DK) -> i64 {
        self.get(key).map(|v| v.as_int()).unwrap_or(0)
    }

    pub fn set(&mut self, key: DK, value: Scalar) {
        if let Some(entry) = self.data.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = value;
        } else {
            self.data.push((key, value));
        }
    }

    pub fn set_int(&mut self, key: DK, value: i64) {
        self.set(key, Scalar::Int(value));
    }

    /// The stored move id under `DK::Move` (lockedmove/encore/futuremove).
    pub fn get_move(&self) -> Option<MoveId> {
        self.get(DK::Move).and_then(|v| v.as_move())
    }

    pub fn remove(&mut self, key: DK) {
        self.data.retain(|(k, _)| *k != key);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
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

/// Inline move-slot list (gen 2: at most 4 moves) — Copy, so cloning a
/// pokemon never touches the heap for moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MoveSlots {
    slots: [MoveSlot; 4],
    n: u8,
}

impl MoveSlots {
    pub fn len(&self) -> usize {
        self.n as usize
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn push(&mut self, slot: MoveSlot) {
        assert!(self.n < 4, "more than 4 move slots");
        self.slots[self.n as usize] = slot;
        self.n += 1;
    }

    pub fn iter(&self) -> std::slice::Iter<'_, MoveSlot> {
        self.slots[..self.n as usize].iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, MoveSlot> {
        self.slots[..self.n as usize].iter_mut()
    }
}

impl std::ops::Index<usize> for MoveSlots {
    type Output = MoveSlot;
    fn index(&self, i: usize) -> &MoveSlot {
        &self.slots[..self.n as usize][i]
    }
}

impl std::ops::IndexMut<usize> for MoveSlots {
    fn index_mut(&mut self, i: usize) -> &mut MoveSlot {
        &mut self.slots[..self.n as usize][i]
    }
}

impl IntoIterator for MoveSlots {
    type Item = MoveSlot;
    type IntoIter = std::iter::Take<std::array::IntoIter<MoveSlot, 4>>;
    fn into_iter(self) -> Self::IntoIter {
        self.slots.into_iter().take(self.n as usize)
    }
}

impl<'a> IntoIterator for &'a MoveSlots {
    type Item = &'a MoveSlot;
    type IntoIter = std::slice::Iter<'a, MoveSlot>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &'a mut MoveSlots {
    type Item = &'a mut MoveSlot;
    type IntoIter = std::slice::IterMut<'a, MoveSlot>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Inline pokemon nickname (PS truncates to 20 chars at construction).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PokeName {
    buf: [u8; 24],
    len: u8,
}

impl PokeName {
    pub fn new(name: &str) -> PokeName {
        let bytes = name.as_bytes();
        assert!(bytes.len() <= 24, "pokemon name too long: {name}");
        let mut buf = [0u8; 24];
        buf[..bytes.len()].copy_from_slice(bytes);
        PokeName { buf, len: bytes.len() as u8 }
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.buf[..self.len as usize]).unwrap()
    }
}

impl std::fmt::Display for PokeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Debug for PokeName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

/// PS gender: "M" / "F" / "" (genderless).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gender {
    M,
    F,
    N,
}

impl Gender {
    pub fn as_str(self) -> &'static str {
        match self {
            Gender::M => "M",
            Gender::F => "F",
            Gender::N => "",
        }
    }
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
#[derive(Clone, Copy, Debug)]
pub struct Attacker {
    pub source: PokeId,
    pub damage: i64,
    pub move_id: MoveId,
    pub this_turn: bool,
    /// PS damageValue: number | false | undefined.
    pub damage_value: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Pokemon {
    // ----- set-derived, fixed for the battle
    pub species: SpeciesId,
    pub base_species: SpeciesId,
    pub name: PokeName,
    pub level: u8,
    pub gender: Gender,
    pub happiness: u8,
    pub set_ivs: [i32; 6],
    pub set_evs: [i32; 6],
    pub base_move_slots: MoveSlots,
    /// Gen 2 hidden power (from DVs).
    pub hp_type: TypeId,
    pub hp_power: i32,
    pub base_hp_type: TypeId,
    pub base_hp_power: i32,
    /// Stats before transform (transform copies stored stats).
    pub base_stored_stats: [i32; 5],

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
    pub move_slots: MoveSlots,
    pub item: Option<ItemId>,
    pub last_item: Option<ItemId>,
    pub item_state: EffectState,
    pub types: TypeList,
    /// Insertion-ordered (PS object key order drives handler collection).
    pub volatiles: Vec<(CondId, EffectState)>,
    /// Union of status/volatile/item callback masks (slot conditions
    /// excluded) — collection fast path. Maintained by refresh_poke_mask at
    /// every status/volatile/item mutation; debug builds assert freshness.
    pub handler_mask: crate::dex::CbMask,
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

    pub fn has_type(&self, ty: TypeId) -> bool {
        self.types.has(ty)
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
    pub name: &'static str,
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
    /// Union of side-condition callback masks (collection fast path).
    pub handler_mask: crate::dex::CbMask,
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
    /// beforeTurnCallback carrier (pursuit).
    BeforeTurnMove { move_id: MoveId, target_loc: i8 },
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
    pub id: &'static str,
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
    pub move_type: TypeId,
    pub base_move_type: TypeId,
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
    pub stalling_move: bool,
    pub non_ghost_target: Option<String>,
    pub flags: Vec<String>,
    pub cb_mask: crate::dex::CbMask,
    // ---- per-use mutable
    pub hit: i32,
    pub last_hit: bool,
    pub total_damage: Option<i64>,
    pub source_effect: Option<MoveId>,
    pub is_confusion_self_hit: bool,
    pub spread_hit: bool,
    /// magnitude's rolled value (onUseMoveMessage).
    pub magnitude: Option<i64>,
    /// beatup ally roster (mutated by getDamage).
    pub allies: Option<Vec<PokeId>>,
    /// triattack move.statusRoll.
    pub status_roll: Option<String>,
    /// curse's `delete move.onHit`.
    pub on_hit_suppressed: bool,
    /// present's `move.infiltrates = true`.
    pub infiltrates: bool,
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
    /// Protocol-log recording. Disabling it (search mode) skips all log
    /// pushes; battle STATE and PRNG consumption are unaffected — the log is
    /// write-only except for `hint(once)`'s dedup scan and
    /// `attr_last_move`/`retarget_last_move`, which only rewrite log lines.
    pub log_enabled: bool,
    pub effect_order: u32,
    pub event_depth: u32,
    pub last_move_line: i64,
    pub last_successful_move_this_turn: Option<MoveId>,
    pub last_damage: i64,
    pub quick_claw_roll: bool,
    /// Field-position values sorted by speed at last runSwitch (resolvePriority).
    pub speed_order: [usize; 2],
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
    /// The boost table flowing through a TryBoost event (mist mutates it).
    pub pending_boosts: Option<Vec<(usize, i8)>>,
    /// Cached dex key of `field.weather` ('' if none) — lets weather checks
    /// avoid threading `dex` everywhere.
    pub field_weather_key: String,
    /// Reusable listener buffers (never state; clones start empty).
    pub listener_pool: crate::battle::events::ScratchPool,
    /// Union of every handler mask in the battle (all roster pokemon, side +
    /// slot conditions, weather, pseudo-weathers). runEvent skips handler
    /// collection when the event's callbacks miss this mask entirely.
    pub battle_mask: crate::dex::CbMask,
}

/// Sparse boosts as an ordered list (PS object iteration order).
pub type SparseBoostsOwned = Vec<(usize, i8)>;

/// Convenience: full 7-slot boost names.
pub const BOOST_NAMES: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

/// Map BTreeMap-based scalar bags (unused placeholder to keep serde imports out).
pub type ScalarMap = BTreeMap<String, Scalar>;
