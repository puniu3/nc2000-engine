//! Loader for `data/gen2stadium2.json` — the flattened reference dex exported
//! from PS (functions stripped, callback names listed).
//!
//! Design: typed fields for the hot data; everything else stays reachable via
//! `extra` so no information is lost while porting. String keys are interned
//! to dense u16 ids at load; battle state stores only ids.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MoveId(pub u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpeciesId(pub u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ItemId(pub u16);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CondId(pub u16);

#[derive(Debug, Deserialize)]
pub struct StatsTable {
    pub hp: u16,
    pub atk: u16,
    pub def: u16,
    pub spa: u16,
    pub spd: u16,
    pub spe: u16,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeciesData {
    pub num: i32,
    pub name: String,
    pub types: Vec<String>,
    pub base_stats: StatsTable,
    #[serde(default)]
    pub gender: Option<String>,
    pub weightkg: f64,
    #[serde(default)]
    pub tier: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveData {
    pub num: i32,
    pub name: String,
    #[serde(default)]
    pub base_power: u16,
    /// `true` (never misses) or a percentage.
    pub accuracy: Value,
    pub pp: u8,
    pub priority: i8,
    pub category: String, // Physical | Special | Status
    #[serde(rename = "type")]
    pub move_type: String,
    pub target: String,
    #[serde(default)]
    pub flags: BTreeMap<String, Value>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub volatile_status: Option<String>,
    #[serde(default)]
    pub side_condition: Option<String>,
    #[serde(default)]
    pub weather: Option<String>,
    #[serde(default)]
    pub secondary: Option<Value>,
    #[serde(default)]
    pub secondaries: Option<Value>,
    #[serde(default)]
    pub drain: Option<(u8, u8)>,
    #[serde(default)]
    pub recoil: Option<(u8, u8)>,
    #[serde(default)]
    pub multihit: Option<Value>,
    #[serde(default)]
    pub crit_ratio: Option<u8>,
    #[serde(default)]
    pub condition: Option<Value>,
    /// Names of the PS callbacks this entry carries (empty = pure data).
    #[serde(default)]
    pub callbacks: Vec<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ItemData {
    pub name: String,
    #[serde(default)]
    pub is_berry: bool,
    #[serde(default)]
    pub condition: Option<Value>,
    #[serde(default)]
    pub callbacks: Vec<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionData {
    #[serde(default)]
    pub effect_type: Option<String>, // Status | Weather | Condition | ...
    #[serde(default)]
    pub duration: Option<u8>,
    #[serde(default)]
    pub counter_max: Option<u16>,
    #[serde(default)]
    pub callbacks: Vec<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
pub struct TypeData {
    pub name: String,
    /// 0 = neutral, 1 = weak to, 2 = resists, 3 = immune (PS integer coding).
    #[serde(rename = "damageTaken")]
    pub damage_taken: BTreeMap<String, u8>,
}

#[derive(Debug, Deserialize)]
pub struct DexFile {
    pub meta: Value,
    pub typechart: BTreeMap<String, TypeData>,
    pub species: BTreeMap<String, SpeciesData>,
    pub moves: BTreeMap<String, MoveData>,
    pub items: BTreeMap<String, ItemData>,
    pub conditions: BTreeMap<String, ConditionData>,
}

/// Interned table: dense ids in sorted-key order (stable across loads).
pub struct Table<I, T> {
    pub keys: Vec<String>,
    pub values: Vec<T>,
    index: BTreeMap<String, u16>,
    _id: std::marker::PhantomData<I>,
}

impl<I: From<u16> + Into<u16> + Copy, T> Table<I, T> {
    fn build(map: BTreeMap<String, T>) -> Self {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        let mut index = BTreeMap::new();
        for (i, (k, v)) in map.into_iter().enumerate() {
            index.insert(k.clone(), i as u16);
            keys.push(k);
            values.push(v);
        }
        Table { keys, values, index, _id: std::marker::PhantomData }
    }

    pub fn id(&self, key: &str) -> Option<I> {
        self.index.get(key).map(|&i| I::from(i))
    }

    pub fn get(&self, id: I) -> &T {
        &self.values[id.into() as usize]
    }

    pub fn key(&self, id: I) -> &str {
        &self.keys[id.into() as usize]
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

macro_rules! impl_id {
    ($t:ty) => {
        impl From<u16> for $t {
            fn from(v: u16) -> Self {
                Self(v)
            }
        }
        impl From<$t> for u16 {
            fn from(v: $t) -> u16 {
                v.0
            }
        }
    };
}
impl_id!(MoveId);
impl_id!(SpeciesId);
impl_id!(ItemId);
impl_id!(CondId);

pub struct Dex {
    pub species: Table<SpeciesId, SpeciesData>,
    pub moves: Table<MoveId, MoveData>,
    pub items: Table<ItemId, ItemData>,
    pub conditions: Table<CondId, ConditionData>,
    pub typechart: BTreeMap<String, TypeData>,
}

impl Dex {
    pub fn from_json(json: &str) -> Result<Dex, serde_json::Error> {
        let file: DexFile = serde_json::from_str(json)?;
        Ok(Dex {
            species: Table::build(file.species),
            moves: Table::build(file.moves),
            items: Table::build(file.items),
            conditions: Table::build(file.conditions),
            typechart: file.typechart,
        })
    }
}
