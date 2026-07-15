//! Battle state — a flat, cheaply clonable object graph (no Rc, no back
//! references). Cloning a `Battle` is a plain deep copy suitable for search.
//!
//! Layout mirrors what PS mutates (sim/pokemon.ts, side.ts, field.ts) but only
//! the parts reachable in gen2stadium2 NC2000: singles, 6 registered / 3
//! picked, no abilities, no mega/z/dynamax/tera.

use crate::dex::{CondId, ItemId, MoveId, SpeciesId};
use crate::prng::Prng;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    None,
    Brn,
    Par,
    Slp,
    Frz,
    Psn,
    Tox,
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
        }
    }
}

/// Scalar value inside an effect-state bag (PS `EffectState` holds arbitrary
/// scalars; we keep them keyed until porting hardens each into a typed field).
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Bool(bool),
    Str(String),
}

/// Mirror of PS `EffectState` minus back references. `data` carries the
/// per-effect counters (sleep turns, substitute HP, rollout count, ...).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EffectState {
    pub duration: Option<i32>,
    pub data: BTreeMap<String, Scalar>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoveSlot {
    pub id: MoveId,
    pub pp: u8,
    pub maxpp: u8,
    pub disabled: bool,
}

/// Boost table indices: atk, def, spa, spd, spe, accuracy, evasion.
pub type Boosts = [i8; 7];

#[derive(Clone, Debug)]
pub struct Pokemon {
    pub species: SpeciesId,
    pub level: u8,
    /// Stored stats (atk, def, spa, spd, spe) after level/DV/stat-exp math.
    pub stats: [u16; 5],
    pub maxhp: u16,
    pub hp: u16,
    pub status: Status,
    pub status_state: EffectState,
    pub boosts: Boosts,
    pub moves: Vec<MoveSlot>,
    pub item: Option<ItemId>,
    pub last_item: Option<ItemId>,
    pub types: Vec<String>,
    pub volatiles: BTreeMap<CondId, EffectState>,
    pub transformed: bool,
    pub fainted: bool,
    /// Gen 2 gender is DV-derived; kept for Attract/Rage interactions.
    pub gender: Option<char>,
    pub happiness: u8,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct Side {
    /// Party in current order (PS reorders this array on switch).
    pub pokemon: Vec<Pokemon>,
    /// Index into `pokemon` of the active slot (singles: one).
    pub active: Option<u8>,
    pub pokemon_left: u8,
    pub side_conditions: BTreeMap<CondId, EffectState>,
}

#[derive(Clone, Debug, Default)]
pub struct Field {
    pub weather: Option<CondId>,
    pub weather_state: EffectState,
    pub pseudo_weather: BTreeMap<CondId, EffectState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestState {
    TeamPreview,
    Move,
    Switch,
    None,
}

#[derive(Clone, Debug)]
pub struct Battle {
    pub prng: Prng,
    pub turn: u16,
    pub request_state: RequestState,
    pub field: Field,
    pub sides: [Side; 2],
    pub log: Vec<String>,
    pub ended: bool,
    /// 0 = p1, 1 = p2, None = tie/undecided.
    pub winner: Option<u8>,
}
