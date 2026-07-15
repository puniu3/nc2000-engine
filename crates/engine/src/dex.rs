//! Loader for `data/gen2stadium2.json` — the flattened reference dex exported
//! from PS (functions stripped, callback names listed).
//!
//! Design: typed fields for the hot data; everything else stays reachable via
//! `extra` so no information is lost while porting. String keys are interned
//! to dense u16 ids at load; battle state stores only ids.
//!
//! `MoveStatic` is the fully-parsed per-move record (PS `Move` after its
//! constructor normalization: secondaries array, self block, critRatio, ...).
//! `ActiveMove` (see `battle::active_move`) is a cheap clone of it plus the
//! mutable per-use fields.

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
/// Id into the RAW `conditions` data table (export shape); distinct from
/// `CondId`, which indexes the interned runtime `conds` table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RawCondId(pub u16);

/// PS `toID`: lowercase, strip non-alphanumerics.
pub fn toid(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

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

impl ItemData {
    pub fn has_callback(&self, name: &str) -> bool {
        self.callbacks.iter().any(|c| c == name)
    }

    /// `on{Event}Order`/`Priority`/`SubOrder` numbers (resolvePriority).
    pub fn num(&self, key: &str) -> Option<i32> {
        self.extra.get(key).and_then(|v| v.as_i64()).map(|v| v as i32)
    }

    /// Item `boosts` table (berserkgene).
    pub fn boosts(&self) -> SparseBoosts {
        self.extra.get("boosts").map(parse_sparse_boosts).unwrap_or_default()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionData {
    #[serde(default)]
    pub name: Option<String>,
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

impl ConditionData {
    /// `on{Event}Order` / `Priority` / `SubOrder` data fields (resolvePriority).
    pub fn handler_num(&self, key: &str) -> Option<i32> {
        self.extra.get(key).and_then(|v| v.as_i64()).map(|v| v as i32)
    }
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
impl_id!(RawCondId);

// ------------------------------------------------------------- MoveStatic

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    Physical,
    Special,
    Status,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Accuracy {
    AlwaysHits,
    Pct(i32),
}

/// Sparse boost table in PS's key iteration order (data object insertion
/// order: PS objects iterate in insertion order; move data writes boosts in
/// atk/def/spa/spd/spe/accuracy/evasion order in practice — we normalize to
/// that fixed order, which matches `for (boostName in boost)` in PS because
/// the JSON preserves the original insertion order and PS data files list
/// stats in canonical order).
pub type SparseBoosts = Vec<(usize, i8)>; // (boost index, delta)

pub const BOOST_KEYS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

pub fn boost_index(key: &str) -> Option<usize> {
    BOOST_KEYS.iter().position(|&k| k == key)
}

fn parse_sparse_boosts(v: &Value) -> SparseBoosts {
    // serde_json is built with preserve_order, so iteration order here is the
    // JSON object's own key order == PS's `for (boostName in boost)` order.
    let mut out = Vec::new();
    if let Value::Object(map) = v {
        for (key, val) in map {
            if let (Some(idx), Some(n)) = (boost_index(key), val.as_i64()) {
                out.push((idx, n as i8));
            }
        }
    }
    out
}

/// A `secondary` or `self` block on a move (PS SecondaryEffect / HitEffect).
#[derive(Clone, Debug, Default)]
pub struct HitEffect {
    pub chance: Option<i32>,
    pub boosts: SparseBoosts,
    pub status: Option<String>,
    pub volatile_status: Option<String>,
    pub self_effect: Option<Box<HitEffect>>,
    pub kingsrock: bool,
    /// Callback names inside this block (e.g. thief secondary.onHit).
    pub has_on_hit: bool,
}

fn parse_hit_effect(v: &Value) -> HitEffect {
    let mut h = HitEffect::default();
    if let Value::Object(map) = v {
        h.chance = map.get("chance").and_then(|x| x.as_i64()).map(|x| x as i32);
        if let Some(b) = map.get("boosts") {
            h.boosts = parse_sparse_boosts(b);
        }
        h.status = map.get("status").and_then(|x| x.as_str()).map(String::from);
        h.volatile_status = map.get("volatileStatus").and_then(|x| x.as_str()).map(String::from);
        if let Some(s) = map.get("self") {
            h.self_effect = Some(Box::new(parse_hit_effect(s)));
        }
        h.kingsrock = map.get("kingsrock").map(|x| !x.is_null()).unwrap_or(false);
        h.has_on_hit = false; // set from the move's callback-name list at parse
    }
    h
}

#[derive(Clone, Debug, PartialEq)]
pub enum FixedDamage {
    Level,
    Amount(i32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Multihit {
    Fixed(i32),
    Range(i32, i32),
}

/// Fully-parsed static move record.
#[derive(Clone, Debug)]
pub struct MoveStatic {
    pub name: String,
    pub move_type: String,
    pub category: Category,
    pub base_power: i32,
    pub accuracy: Accuracy,
    pub pp: i32,
    pub no_pp_boosts: bool,
    pub priority: i8,
    pub target: String,
    pub crit_ratio: i32,
    pub will_crit: Option<bool>,
    pub flags: Vec<String>,
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
    /// PS callback names on the move (non-empty = milestone 2 move).
    pub callbacks: Vec<String>,
    /// Condition block data (for moves that define one), kept raw.
    pub condition: Option<Value>,
}

impl MoveStatic {
    fn parse(d: &MoveData) -> MoveStatic {
        let x = |k: &str| d.extra.get(k);
        let xb = |k: &str| x(k).map(|v| v.as_bool() == Some(true) || v.is_string()).unwrap_or(false);
        let category = match d.category.as_str() {
            "Physical" => Category::Physical,
            "Special" => Category::Special,
            _ => Category::Status,
        };
        let accuracy = match &d.accuracy {
            Value::Bool(true) => Accuracy::AlwaysHits,
            Value::Number(n) => Accuracy::Pct(n.as_i64().unwrap_or(100) as i32),
            _ => Accuracy::Pct(100),
        };
        let multihit = d.multihit.as_ref().and_then(|v| match v {
            Value::Number(n) => Some(Multihit::Fixed(n.as_i64().unwrap() as i32)),
            Value::Array(a) if a.len() == 2 => Some(Multihit::Range(
                a[0].as_i64().unwrap() as i32,
                a[1].as_i64().unwrap() as i32,
            )),
            _ => None,
        });
        let damage = x("damage").and_then(|v| match v {
            Value::String(s) if s == "level" => Some(FixedDamage::Level),
            Value::Number(n) => Some(FixedDamage::Amount(n.as_i64().unwrap() as i32)),
            _ => None,
        });
        let mut secondaries: Vec<HitEffect> = d
            .secondaries
            .as_ref()
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(parse_hit_effect).collect())
            .unwrap_or_default();
        // The export records sub-block callbacks as "secondary.onHit" /
        // "self.onHit" on the move's callback list.
        if d.callbacks.iter().any(|c| c == "secondary.onHit") {
            for s in &mut secondaries {
                s.has_on_hit = true;
            }
        }
        let mut self_effect = d.extra.get("self").map(parse_hit_effect);
        if d.callbacks.iter().any(|c| c == "self.onHit") {
            if let Some(se) = &mut self_effect {
                se.has_on_hit = true;
            }
        }
        MoveStatic {
            name: d.name.clone(),
            move_type: d.move_type.clone(),
            category,
            base_power: d.base_power as i32,
            accuracy,
            pp: d.pp as i32,
            no_pp_boosts: xb("noPPBoosts"),
            priority: d.priority,
            target: d.target.clone(),
            crit_ratio: d.crit_ratio.unwrap_or(1) as i32,
            will_crit: x("willCrit").and_then(|v| v.as_bool()),
            flags: d.flags.keys().cloned().collect(),
            status: d.status.clone(),
            volatile_status: d.volatile_status.clone(),
            side_condition: d.side_condition.clone(),
            weather: d.weather.clone(),
            pseudo_weather: x("pseudoWeather").and_then(|v| v.as_str()).map(String::from),
            boosts: x("boosts").map(parse_sparse_boosts).unwrap_or_default(),
            has_boosts: x("boosts").is_some(),
            heal: x("heal").and_then(|v| v.as_array()).map(|a| {
                (a[0].as_i64().unwrap() as i32, a[1].as_i64().unwrap() as i32)
            }),
            drain: d.drain.map(|(a, b)| (a as i32, b as i32)),
            recoil: d.recoil.map(|(a, b)| (a as i32, b as i32)),
            struggle_recoil: xb("struggleRecoil"),
            multihit,
            secondaries,
            self_effect,
            damage,
            ohko: xb("ohko"),
            selfdestruct: xb("selfdestruct"),
            self_switch: x("selfSwitch").and_then(|v| {
                if v.as_bool() == Some(true) {
                    Some("true".to_string())
                } else {
                    v.as_str().map(String::from)
                }
            }),
            force_switch: xb("forceSwitch"),
            ignore_immunity: match x("ignoreImmunity") {
                Some(Value::Bool(b)) => *b,
                Some(Value::Object(_)) => true, // e.g. {Ground: true} — treat per-type later if needed
                _ => category == Category::Status,
            },
            ignore_accuracy: xb("ignoreAccuracy"),
            ignore_evasion: xb("ignoreEvasion"),
            ignore_positive_evasion: xb("ignorePositiveEvasion"),
            ignore_offensive: xb("ignoreOffensive"),
            ignore_defensive: xb("ignoreDefensive"),
            sleep_usable: xb("sleepUsable"),
            no_damage_variance: xb("noDamageVariance"),
            always_hit: xb("alwaysHit"),
            thaws_target: xb("thawsTarget"),
            stalling_move: xb("stallingMove"),
            non_ghost_target: x("nonGhostTarget").and_then(|v| v.as_str()).map(String::from),
            callbacks: d.callbacks.clone(),
            condition: d.condition.clone(),
        }
    }

    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
}

/// PS effect types reachable in this format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectType {
    Condition,
    Status,
    Weather,
    Move,
    Item,
    Rule,
    Format,
}

/// Interned runtime condition entry. Mirrors what PS
/// `dex.conditions.getByID(id)` resolves for every id the battle can touch:
/// dex conditions, move `condition` blocks, format rules acting as
/// pseudo-weathers, and marker volatiles that exist only as ids.
#[derive(Clone, Debug)]
pub struct CondEntry {
    pub name: String,
    pub effect_type: EffectType,
    pub duration: Option<i32>,
    pub counter_max: Option<i32>,
    /// Not transferred by Baton Pass.
    pub no_copy: bool,
    pub callbacks: Vec<String>,
    /// `on{Event}Order` / `Priority` / `SubOrder` numbers from the data.
    pub nums: BTreeMap<String, i32>,
}

impl CondEntry {
    pub fn has_callback(&self, name: &str) -> bool {
        self.callbacks.iter().any(|c| c == name)
    }

    pub fn num(&self, key: &str) -> Option<i32> {
        self.nums.get(key).copied()
    }
}

fn handler_nums(obj: &BTreeMap<String, Value>) -> BTreeMap<String, i32> {
    let mut nums = BTreeMap::new();
    for (k, v) in obj {
        if k.starts_with("on")
            && (k.ends_with("Order") || k.ends_with("Priority") || k.ends_with("SubOrder"))
        {
            if let Some(n) = v.as_i64() {
                nums.insert(k.clone(), n as i32);
            }
        }
    }
    nums
}

pub struct Dex {
    pub species: Table<SpeciesId, SpeciesData>,
    pub moves: Table<MoveId, MoveData>,
    pub items: Table<ItemId, ItemData>,
    pub conditions: Table<RawCondId, ConditionData>,
    pub typechart: BTreeMap<String, TypeData>,
    /// Parsed static move records, indexed by MoveId.
    pub moves_static: Vec<MoveStatic>,
    /// Interned runtime conditions (superset of `conditions`).
    pub conds: Table<CondId, CondEntry>,
    /// Every callback name any condition or item in this format can handle
    /// (plus the code-only builtins). Battle event dispatch uses this to skip
    /// handler collection for events that can never have handlers. Move
    /// callbacks are NOT included — they never enter `findEventHandlers`
    /// (single_event / the runEvent onEffect branch check the move directly).
    pub possible_callbacks: std::collections::HashSet<String>,
}

impl Dex {
    pub fn from_json(json: &str) -> Result<Dex, serde_json::Error> {
        let mut file: DexFile = serde_json::from_str(json)?;
        // Synthetic 'recharge' pseudo-move (PS resolves it as a nonexistent
        // move; it only ever reaches BeforeMove, where mustrecharge aborts it).
        if !file.moves.contains_key("recharge") {
            let recharge: MoveData = serde_json::from_value(serde_json::json!({
                "num": 0,
                "name": "Recharge",
                "basePower": 0,
                "accuracy": true,
                "pp": 1,
                "priority": 0,
                "category": "Physical",
                "type": "???",
                "target": "normal",
                "flags": {},
                "callbacks": [],
            }))?;
            file.moves.insert("recharge".to_string(), recharge);
        }
        // Non-function constant callbacks are invisible to the exporter:
        // gen2 teleport carries `onTry: false` (fails silently, no message).
        if let Some(teleport) = file.moves.get_mut("teleport") {
            if !teleport.callbacks.iter().any(|c| c == "onTry") {
                teleport.callbacks.push("onTry".to_string());
            }
        }
        let moves = Table::build(file.moves);
        let moves_static: Vec<MoveStatic> = moves.values.iter().map(MoveStatic::parse).collect();

        // ---- build the interned runtime condition table
        let mut entries: BTreeMap<String, CondEntry> = BTreeMap::new();
        for (id, c) in &file.conditions {
            let effect_type = match c.effect_type.as_deref() {
                Some("Status") => EffectType::Status,
                Some("Weather") => EffectType::Weather,
                _ => EffectType::Condition,
            };
            entries.insert(
                id.clone(),
                CondEntry {
                    name: c.name.clone().unwrap_or_else(|| id.clone()),
                    effect_type,
                    duration: c.duration.map(|d| d as i32),
                    counter_max: c.counter_max.map(|d| d as i32),
                    no_copy: c.extra.get("noCopy").and_then(|v| v.as_bool()).unwrap_or(false),
                    callbacks: c.callbacks.clone(),
                    nums: handler_nums(&c.extra),
                },
            );
        }
        // move condition blocks (volatiles keyed by move id, e.g. lightscreen).
        // Moves WITHOUT a condition block still intern as empty conditions:
        // PS conditions.get(moveid) resolves a wrapper (twoturnmove adds e.g.
        // a 'solarbeam' volatile).
        for (i, ms) in moves_static.iter().enumerate() {
            let id = moves.keys[i].clone();
            if entries.contains_key(&id) {
                continue;
            }
            let has_block = matches!(&ms.condition, Some(Value::Object(_)));
            let cond_map: BTreeMap<String, Value> = match &ms.condition {
                Some(Value::Object(cond)) => {
                    cond.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                }
                _ => BTreeMap::new(),
            };
            let callbacks = ms
                .callbacks
                .iter()
                .filter_map(|c| c.strip_prefix("condition."))
                .map(String::from)
                .collect();
            entries.insert(
                id.clone(),
                CondEntry {
                    // PS: a move without a condition block resolves to a
                    // nonexistent Condition whose name is the raw id.
                    name: if has_block { ms.name.clone() } else { id },
                    effect_type: EffectType::Condition,
                    duration: cond_map.get("duration").and_then(|v| v.as_i64()).map(|v| v as i32),
                    counter_max: cond_map
                        .get("counterMax")
                        .and_then(|v| v.as_i64())
                        .map(|v| v as i32),
                    no_copy: cond_map.get("noCopy").and_then(|v| v.as_bool()).unwrap_or(false),
                    callbacks,
                    nums: handler_nums(&cond_map),
                },
            );
        }
        // rules that act as runtime effects (pseudo-weathers / onSetStatus)
        for (id, name, callbacks) in [
            ("maxtotallevel", "Max Total Level", vec![]),
            ("stadiumsleepclause", "Stadium Sleep Clause", vec!["onSetStatus".to_string()]),
            ("freezeclausemod", "Freeze Clause Mod", vec!["onSetStatus".to_string()]),
        ] {
            entries.insert(
                id.to_string(),
                CondEntry {
                    name: name.to_string(),
                    effect_type: EffectType::Rule,
                    duration: None,
                    counter_max: None,
                    no_copy: false,
                    callbacks,
                    nums: BTreeMap::new(),
                },
            );
        }
        // marker/synthetic conditions PS resolves as nonexistent-or-special
        for (id, name) in [
            ("brnattackdrop", "brnattackdrop"),
            ("parspeeddrop", "parspeeddrop"),
            ("recoil", "Recoil"),
            ("drain", "Drain"),
            ("confused", "confused"),
            // mysteryberry adds this ephemeral marker (PS resolves the id via
            // the out-of-dex leppaberry item entry; no callbacks fire).
            ("leppaberry", "Leppa Berry"),
        ] {
            entries.entry(id.to_string()).or_insert_with(|| CondEntry {
                name: name.to_string(),
                effect_type: EffectType::Condition,
                duration: None,
                counter_max: None,
                no_copy: false,
                callbacks: Vec::new(),
                nums: BTreeMap::new(),
            });
        }

        let items = Table::build(file.items);
        let mut possible_callbacks: std::collections::HashSet<String> =
            entries.values().flat_map(|e| e.callbacks.iter().cloned()).collect();
        for item in &items.values {
            possible_callbacks.extend(item.callbacks.iter().cloned());
        }
        // crate::battle::conditions::has_builtin constants
        possible_callbacks.insert("onLockMove".to_string());
        possible_callbacks.insert("onSemiLockMove".to_string());

        Ok(Dex {
            species: Table::build(file.species),
            moves,
            items,
            conditions: Table::build(file.conditions),
            typechart: file.typechart,
            moves_static,
            conds: Table::build(entries),
            possible_callbacks,
        })
    }

    /// Can ANY condition/item in this format handle `callback_name`?
    pub fn callback_possible(&self, callback_name: &str) -> bool {
        self.possible_callbacks.contains(callback_name)
    }

    pub fn move_static(&self, id: MoveId) -> &MoveStatic {
        &self.moves_static[id.0 as usize]
    }

    pub fn conds_id(&self, key: &str) -> Option<CondId> {
        self.conds.id(key)
    }

    pub fn conds_key(&self, id: CondId) -> &str {
        self.conds.key(id)
    }

    pub fn cond(&self, id: CondId) -> &CondEntry {
        self.conds.get(id)
    }

    pub fn cond_display_name(&self, id: CondId) -> &str {
        &self.conds.get(id).name
    }

    pub fn cond_effect_type(&self, id: CondId) -> EffectType {
        self.conds.get(id).effect_type
    }

    /// PS `dex.getEffectiveness`: +1 super effective, -1 resisted, 0 neutral/immune.
    pub fn get_effectiveness(&self, attack_type: &str, defend_type: &str) -> i32 {
        let Some(t) = self.typechart.get(&toid(defend_type)) else { return 0 };
        match t.damage_taken.get(attack_type).copied().unwrap_or(0) {
            1 => 1,
            2 => -1,
            _ => 0,
        }
    }

    /// PS `dex.getImmunity`: false = immune. `source_type` may be a move type
    /// or a status id ('psn', 'trapped', 'sandstorm', ...).
    pub fn get_immunity(&self, source_type: &str, defend_types: &[String]) -> bool {
        for ty in defend_types {
            if let Some(t) = self.typechart.get(&toid(ty)) {
                if t.damage_taken.get(source_type).copied().unwrap_or(0) == 3 {
                    return false;
                }
            }
        }
        true
    }
}
