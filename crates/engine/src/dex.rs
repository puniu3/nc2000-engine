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

/// Dense id of a registered callback name ("onTryHit", "basePowerCallback",
/// ...). Assigned at dex load in deterministic (BTreeMap) order; a name that
/// was never registered maps to `Cb::NONE`, which no `CbMask` contains.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cb(pub u16);

impl Cb {
    pub const NONE: Cb = Cb(u16::MAX);

    #[inline]
    pub fn exists(self) -> bool {
        self.0 != u16::MAX
    }
}

/// 256-bit set over registered callback ids (the gen2stadium2 dex registers
/// ~90 distinct names; load asserts the bound).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CbMask(pub(crate) [u64; 4]);

impl CbMask {
    pub const EMPTY: CbMask = CbMask([0; 4]);

    #[inline(always)]
    pub fn has(&self, cb: Cb) -> bool {
        let i = cb.0 as usize;
        i < 256 && self.0[i >> 6] & (1 << (i & 63)) != 0
    }

    #[inline(always)]
    pub fn set(&mut self, cb: Cb) {
        let i = cb.0 as usize;
        assert!(i < 256, "CbMask overflow");
        self.0[i >> 6] |= 1 << (i & 63);
    }

    #[inline(always)]
    pub fn clear(&mut self, cb: Cb) {
        let i = cb.0 as usize;
        if i < 256 {
            self.0[i >> 6] &= !(1 << (i & 63));
        }
    }

    #[inline(always)]
    pub fn or_with(&mut self, other: &CbMask) {
        for k in 0..4 {
            self.0[k] |= other.0[k];
        }
    }

    #[inline(always)]
    pub const fn intersects(&self, other: &CbMask) -> bool {
        ((self.0[0] & other.0[0])
            | (self.0[1] & other.0[1])
            | (self.0[2] & other.0[2])
            | (self.0[3] & other.0[3]))
            != 0
    }
}

/// `on{Event}Order` / `Priority` / `SubOrder` numbers a condition or item
/// declares for one callback (PS resolvePriority data lookups, precomputed).
#[derive(Clone, Copy, Debug, Default)]
pub struct CbNums {
    pub order: Option<i32>,
    pub priority: Option<i32>,
    pub sub_order: Option<i32>,
}

/// Per-call-site cached `dex.conds_id(<literal>)`. Each expansion owns a
/// `OnceLock`, so the BTreeMap lookup happens once per site per process (one
/// format's dex per process — the same assumption the event system makes).
#[macro_export]
macro_rules! cond_id {
    ($dex:expr, $name:literal) => {{
        static C: std::sync::OnceLock<Option<$crate::dex::CondId>> = std::sync::OnceLock::new();
        *C.get_or_init(|| $dex.conds_id($name))
    }};
}

/// Dense type id ("Water", "Flying", ..., plus the "???" pseudo-type).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeId(pub u8);

/// A pokemon's type list (1–2 entries in gen 2; conversion can set 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TypeList {
    pub t: [TypeId; 2],
    pub n: u8,
}

impl TypeList {
    pub fn one(a: TypeId) -> TypeList {
        TypeList { t: [a, a], n: 1 }
    }

    pub fn two(a: TypeId, b: TypeId) -> TypeList {
        TypeList { t: [a, b], n: 2 }
    }

    #[inline]
    pub fn has(&self, ty: TypeId) -> bool {
        self.t[0] == ty || (self.n == 2 && self.t[1] == ty)
    }

    pub fn iter(&self) -> impl Iterator<Item = TypeId> + '_ {
        self.t[..self.n as usize].iter().copied()
    }
}

/// Type ids referenced by literal in the battle code.
#[derive(Clone, Copy, Debug)]
pub struct KnownTypes {
    pub unknown: TypeId, // "???"
    pub normal: TypeId,
    pub water: TypeId,
    pub fire: TypeId,
    pub flying: TypeId,
    pub ghost: TypeId,
    pub grass: TypeId,
    pub fighting: TypeId,
    pub ground: TypeId,
}

/// Item ids referenced by the stat/speed hot path (gen2stadium2 getStat).
#[derive(Clone, Copy, Debug)]
pub struct KnownItems {
    pub thickclub: Option<ItemId>,
    pub lightball: Option<ItemId>,
    pub metalpowder: Option<ItemId>,
    pub quickclaw: Option<ItemId>,
}

/// Species ids referenced by the stat / item hot path.
#[derive(Clone, Copy, Debug)]
pub struct KnownSpecies {
    pub cubone: Option<SpeciesId>,
    pub marowak: Option<SpeciesId>,
    pub pikachu: Option<SpeciesId>,
    pub ditto: Option<SpeciesId>,
    pub farfetchd: Option<SpeciesId>,
    pub chansey: Option<SpeciesId>,
}

/// Precomputed `durationCallback` behavior for a condition entry.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DurationCbKind {
    #[default]
    None = 0,
    Rand3To6,
    Rand2To4,
    Rand3To5,
    Rand2To6,
    Rand3To7,
    Const5,
}

/// Precomputed `damageCallback` behavior for a move entry.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DamageCbKind {
    #[default]
    None = 0,
    Counter,
    MirrorCoat,
    Psywave,
    SuperFang,
}

/// Precomputed `basePowerCallback` behavior for a move entry.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasePowerCbKind {
    #[default]
    None = 0,
    FlailReversal,
    Frustration,
    Return,
    TripleKick,
    Pursuit,
    BeatUp,
    HiddenPower,
    FuryCutter,
    Rollout,
}

/// Precomputed `beforeMoveCallback` behavior for a move entry.
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BeforeMoveCbKind {
    #[default]
    None = 0,
    Bide,
}

/// Dense integer opcode for condition callback dispatch (`Cb` <= 255).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CondCbOp {
    #[default]
    Unported = 0,
    Noop,
    BrnStart,
    BrnAfterMoveSelf,
    BrnSwitchIn,
    ParStart,
    ParBeforeMove,
    ParSwitchIn,
    SlpStart,
    SlpBeforeMove,
    FrzStart,
    FrzBeforeMove,
    FrzAfterMoveSecondary,
    FrzAfterMoveSecondarySelf,
    FrzResidual,
    PsnStart,
    PsnAfterMoveSelf,
    ToxStart,
    ToxAfterMoveSelf,
    ToxSwitchIn,
    ToxAfterSwitchInSelf,
    ConfusionStart,
    ConfusionEnd,
    ConfusionBeforeMove,
    FlinchBeforeMove,
    PartiallyTrappedStart,
    PartiallyTrappedResidual,
    PartiallyTrappedEnd,
    PartiallyTrappedTrapPokemon,
    ResidualDmgStart,
    ResidualDmgAfterMoveSelf,
    MustRechargeBeforeMove,
    MustRechargeStart,
    MustRechargeLockMove,
    TrappedTrapPokemon,
    TrappedStart,
    RainDanceFieldStart,
    RainDanceFieldResidual,
    RainDanceFieldEnd,
    SunnyDayFieldStart,
    SunnyDayFieldResidual,
    SunnyDayFieldEnd,
    SunnyDayImmunity,
    SandstormFieldStart,
    SandstormFieldResidual,
    SandstormWeather,
    SandstormFieldEnd,
    SleepClauseModSetStatus,
    FreezeClauseModSetStatus,
    LightScreenSideStart,
    ReflectSideStart,
    LightScreenSideEnd,
    ReflectSideEnd,
    SafeguardSideStart,
    SafeguardSideEnd,
    SafeguardSetStatus,
    SafeguardTryAddVolatile,
    SpikesSideStart,
    SpikesEntryHazard,
    LeechSeedStart,
    LeechSeedAfterMoveSelf,
    NightmareStart,
    NightmareAfterMoveSelf,
    CurseStart,
    CurseAfterMoveSelf,
    ForesightStart,
    ForesightNegateImmunity,
    LockOnSourceAccuracy,
    DestinyBondStart,
    DestinyBondFaint,
    DestinyBondBeforeMove,
    DestinyBondMoveAborted,
    DestinyBondFoeAfterMoveSelf,
    DestinyBondFoeSwitchOut,
    PerishSongResidual,
    PerishSongEnd,
    EncoreStart,
    EncoreOverrideAction,
    EncoreResidual,
    EncoreEnd,
    EncoreDisableMove,
    DisableStart,
    DisableEnd,
    DisableBeforeMove,
    DisableDisableMove,
    MistStart,
    MistTryBoost,
    SubstituteStart,
    SubstituteEnd,
    SubstituteTryPrimaryHit,
    BideStart,
    BideDamage,
    BideBeforeMove,
    BideMoveAborted,
    BideEnd,
    BideSemiLockMove,
    RageStart,
    RageHit,
    RageBeforeMove,
    RolloutStart,
    RolloutResidual,
    RolloutLockMove,
    FuryCutterStart,
    FuryCutterRestart,
    DefenseCurlMinimizeRestart,
    MinimizeSourceModifyDamage,
    AttractStart,
    AttractUpdate,
    AttractBeforeMove,
    AttractEnd,
    FocusEnergyStart,
    FocusEnergyModifyCritRatio,
    ProtectStart,
    ProtectTryHit,
    EndureStart,
    EndureDamage,
    StallStart,
    StallMove,
    StallRestart,
    TwoTurnMoveStart,
    TwoTurnMoveEnd,
    TwoTurnMoveLockMove,
    TwoTurnMoveAborted,
    DigImmunity,
    DigInvulnerability,
    FlyInvulnerability,
    DigSourceBasePower,
    FlySourceBasePower,
    LockedMoveStart,
    LockedMoveResidual,
    LockedMoveEnd,
    LockedMoveLockMove,
    LockedMoveAborted,
    PursuitFoeBeforeSwitchOut,
    FutureMoveStart,
    FutureMoveResidual,
    FutureMoveEnd,
}

/// Dense integer opcode for item callback dispatch (`Cb` <= 255).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ItemCbOp {
    #[default]
    Unported = 0,
    TypeBoostModifyDamage,
    ScopeLensCrit,
    StickCrit,
    LuckyPunchCrit,
    BrightPowderAccuracy,
    FocusBandDamage,
    KingsRockModifyMove,
    LeftoversResidual,
    BerryJuiceResidual,
    BerryResidual,
    BerryTryEat,
    GoldBerryTryEat,
    BerryEat,
    GoldBerryEat,
    BitterBerryUpdate,
    BitterBerryEat,
    BurntBerryUpdate,
    IceBerryUpdate,
    PrzCureBerryUpdate,
    PsnCureBerryUpdate,
    MintBerryCheck,
    BurntBerryEat,
    IceBerryEat,
    MintBerryEat,
    PrzCureBerryEat,
    PsnCureBerryEat,
    MiracleBerryCheck,
    MiracleBerryEat,
    MysteryBerryUpdate,
    MysteryBerryEat,
    MailTakeItem,
    BerserkGeneUpdate,
}

/// Dense integer opcode for move callback dispatch (`Cb` <= 255).
#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoveCbOp {
    #[default]
    Unported = 0,
    StruggleModifyMove,
    ThunderModifyMove,
    HiddenPowerModifyMove,
    CurseModifyMove,
    MagnitudeModifyMove,
    PresentModifyMove,
    BeatUpModifyMove,
    RolloutModifyMove,
    MagnitudeUseMoveMessage,
    ChargeTryMove,
    SkullBashTryMove,
    SolarBeamTryMove,
    ChargePrepareHit,
    ProtectDetectEndurePrepareHit,
    DestinyPerishPrepareHit,
    RestTry,
    SleepTalkSnoreTry,
    NoopTry,
    FalseTry,
    FutureSightTry,
    SubstituteTryHit,
    CurseTryHit,
    DisableTryHit,
    ForesightTryHit,
    LockOnTryHit,
    RoarTryHit,
    SplashTryHit,
    SwaggerTryHit,
    SleepTalkTryHit,
    AttractTryImmunity,
    DreamEaterTryImmunity,
    LeechSeedTryImmunity,
    HighJumpKickMoveFail,
    OutrageMoveFail,
    FalseSwipeDamage,
    HazeHitField,
    PerishSongHitField,
    SubstituteHit,
    BellyDrumHit,
    ConversionHit,
    Conversion2Hit,
    ProtectDetectEndureHit,
    HealBellHit,
    LockOnHit,
    MeanLookHit,
    MetronomeHit,
    MimicHit,
    MirrorMoveHit,
    MoonlightHit,
    PainSplitHit,
    PayDayHit,
    PsychUpHit,
    RestHit,
    SketchHit,
    SleepTalkHit,
    SpiteHit,
    TransformHit,
    TriAttackHit,
    BatonPassHit,
    FuryCutterHit,
    CurseHit,
    ThiefSecondaryOnHit,
    TriAttackSecondaryOnHit,
    RapidSpinSelfOnHit,
    BatonPassSelfOnHit,
    RapidSpinAfterHit,
    RolloutAfterMove,
}

fn default_item_cb_ops() -> [ItemCbOp; 256] {
    [ItemCbOp::Unported; 256]
}

const _: () = {
    assert!(std::mem::size_of::<CondCbOp>() == 1);
    assert!(std::mem::size_of::<ItemCbOp>() == 1);
    assert!(std::mem::size_of::<MoveCbOp>() == 1);
};

/// Hot callback ids resolved once at load (literal-name checks in the battle
/// code go through these instead of hashing the string every time).
#[derive(Clone, Copy, Debug)]
pub struct KnownCbs {
    pub on_hit: Cb,
    pub on_after_hit: Cb,
    pub on_start: Cb,
    pub on_set_status: Cb,
    pub damage_callback: Cb,
    pub base_power_callback: Cb,
    pub before_move_callback: Cb,
    pub before_turn_callback: Cb,
    pub on_hit_field: Cb,
    pub on_hit_side: Cb,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MoveId(pub u16);
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
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
    /// Registered-callback bitset (filled at load, after registry build).
    #[serde(skip)]
    pub mask: CbMask,
    /// Per-callback Order/Priority/SubOrder numbers (filled at load).
    #[serde(skip)]
    pub cb_nums: Vec<(Cb, CbNums)>,
    /// Precomputed 1.1x type boost target (if this is a Gen 2 type-boost item).
    #[serde(skip)]
    pub type_boost: Option<TypeId>,
    /// Direct jump-table op indexed by `Cb.0`.
    #[serde(skip, default = "default_item_cb_ops")]
    pub cb_ops: [ItemCbOp; 256],
}

impl ItemData {
    #[inline]
    pub fn cb_num(&self, cb: Cb) -> CbNums {
        self.cb_nums.iter().find(|(c, _)| *c == cb).map(|(_, n)| *n).unwrap_or_default()
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
    interned_keys: Vec<&'static str>,
    pub values: Vec<T>,
    index: crate::fxhash::FxHashMap<&'static str, u16>,
    _id: std::marker::PhantomData<I>,
}

impl<I: From<u16> + Into<u16> + Copy, T> Table<I, T> {
    fn build(map: BTreeMap<String, T>) -> Self {
        let mut keys = Vec::with_capacity(map.len());
        let mut interned_keys = Vec::with_capacity(map.len());
        let mut values = Vec::with_capacity(map.len());
        let mut index = crate::fxhash::FxHashMap::default();
        for (i, (k, v)) in map.into_iter().enumerate() {
            let ik = leak_intern(&k);
            index.insert(ik, i as u16);
            keys.push(k);
            interned_keys.push(ik);
            values.push(v);
        }
        Table { keys, interned_keys, values, index, _id: std::marker::PhantomData }
    }

    #[inline]
    pub fn id(&self, key: &str) -> Option<I> {
        self.index.get(key).map(|&i| I::from(i))
    }

    #[inline]
    pub fn get(&self, id: I) -> &T {
        &self.values[id.into() as usize]
    }

    #[inline]
    pub fn key(&self, id: I) -> &'static str {
        self.interned_keys[id.into() as usize]
    }

    pub fn len(&self) -> usize {
        self.values.len()
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
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct SparseBoosts {
    entries: [(usize, i8); 7],
    len: u8,
}

impl SparseBoosts {
    pub const EMPTY: Self = Self {
        entries: [(0, 0); 7],
        len: 0,
    };

    #[inline]
    pub const fn new() -> Self {
        Self::EMPTY
    }

    #[inline]
    pub fn from_slice(slice: &[(usize, i8)]) -> Self {
        assert!(slice.len() <= 7, "SparseBoosts overflow");
        let mut entries = [(0, 0); 7];
        let mut i = 0;
        while i < slice.len() {
            entries[i] = slice[i];
            i += 1;
        }
        Self {
            entries,
            len: slice.len() as u8,
        }
    }

    #[inline]
    pub fn push(&mut self, item: (usize, i8)) {
        assert!((self.len as usize) < 7, "SparseBoosts overflow");
        self.entries[self.len as usize] = item;
        self.len += 1;
    }

    #[inline]
    pub fn retain(&mut self, mut keep: impl FnMut(&(usize, i8)) -> bool) {
        let mut n = 0usize;
        for i in 0..self.len as usize {
            let entry = self.entries[i];
            if keep(&entry) {
                self.entries[n] = entry;
                n += 1;
            }
        }
        self.len = n as u8;
    }

    #[inline]
    pub fn as_slice(&self) -> &[(usize, i8)] {
        &self.entries[..self.len as usize]
    }
}

impl std::ops::Deref for SparseBoosts {
    type Target = [(usize, i8)];
    #[inline]
    fn deref(&self) -> &[(usize, i8)] {
        self.as_slice()
    }
}

impl std::fmt::Debug for SparseBoosts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl std::hash::Hash for SparseBoosts {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl IntoIterator for SparseBoosts {
    type Item = (usize, i8);
    type IntoIter = std::iter::Take<std::array::IntoIter<(usize, i8), 7>>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter().take(self.len as usize)
    }
}

impl<'a> IntoIterator for &'a SparseBoosts {
    type Item = &'a (usize, i8);
    type IntoIter = std::slice::Iter<'a, (usize, i8)>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

pub const BOOST_KEYS: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"];

pub fn boost_index(key: &str) -> Option<usize> {
    BOOST_KEYS.iter().position(|&k| k == key)
}

fn parse_sparse_boosts(v: &Value) -> SparseBoosts {
    // serde_json is built with preserve_order, so iteration order here is the
    // JSON object's own key order == PS's `for (boostName in boost)` order.
    let mut out = SparseBoosts::new();
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
    pub status: Option<&'static str>,
    pub volatile_status: Option<&'static str>,
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
        h.status = map.get("status").and_then(|x| x.as_str()).map(leak_intern);
        h.volatile_status = map.get("volatileStatus").and_then(|x| x.as_str()).map(leak_intern);
        if let Some(s) = map.get("self") {
            h.self_effect = Some(Box::new(parse_hit_effect(s)));
        }
        h.kingsrock = map.get("kingsrock").map(|x| !x.is_null()).unwrap_or(false);
        h.has_on_hit = false; // set from the move's callback-name list at parse
    }
    h
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FixedDamage {
    Level,
    Amount(i32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Multihit {
    Fixed(i32),
    Range(i32, i32),
}

/// Fully-parsed static move record.
#[derive(Clone, Debug)]
pub struct MoveStatic {
    pub name: String,
    pub move_type: TypeId,
    // (target/flags interned below — see `target` and `flags` fields)
    pub category: Category,
    pub base_power: i32,
    pub accuracy: Accuracy,
    pub pp: i32,
    pub no_pp_boosts: bool,
    pub priority: i8,
    pub target: &'static str,
    pub crit_ratio: i32,
    pub will_crit: Option<bool>,
    /// Bitmask over `Dex::flag_bit` (move flags: contact/charge/protect/...).
    pub flags: u64,
    pub status: Option<&'static str>,
    pub volatile_status: Option<&'static str>,
    pub side_condition: Option<&'static str>,
    pub weather: Option<&'static str>,
    pub pseudo_weather: Option<&'static str>,
    pub boosts: SparseBoosts,
    pub has_boosts: bool,
    pub heal: Option<(i32, i32)>,
    pub drain: Option<(i32, i32)>,
    pub recoil: Option<(i32, i32)>,
    pub struggle_recoil: bool,
    pub multihit: Option<Multihit>,
    pub secondaries: smallvec::SmallVec<[HitEffect; 2]>,
    pub self_effect: Option<HitEffect>,
    pub damage: Option<FixedDamage>,
    pub ohko: bool,
    pub selfdestruct: bool,
    pub self_switch: Option<&'static str>,
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
    pub non_ghost_target: Option<&'static str>,
    /// PS callback names on the move (non-empty = milestone 2 move).
    pub callbacks: Vec<String>,
    /// Registered-callback bitset over `callbacks` (dotted sub-block names
    /// like "secondary.onHit" excluded — they never reach mask checks).
    pub cb_mask: CbMask,
    /// Condition block data (for moves that define one), kept raw.
    pub condition: Option<Value>,
    /// Direct jump-table ops indexed by `Cb.0`.
    pub cb_ops: [MoveCbOp; 256],
    pub secondary_on_hit_op: MoveCbOp,
    pub self_on_hit_op: MoveCbOp,
    pub damage_cb: DamageCbKind,
    pub base_power_cb: BasePowerCbKind,
    pub before_move_cb: BeforeMoveCbKind,
    pub kingsrock: bool,
    pub calls_other_move: bool,
    pub is_pursuit: bool,
    pub is_solarbeam: bool,
}

/// Intern a short string once, globally (move targets, flag names).
pub(crate) fn leak_intern(s: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};
    static TABLE: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut t = TABLE.get_or_init(|| Mutex::new(HashSet::new())).lock().unwrap();
    match t.get(s) {
        Some(&v) => v,
        None => {
            let v: &'static str = Box::leak(s.to_owned().into_boxed_str());
            t.insert(v);
            v
        }
    }
}

impl MoveStatic {
    fn parse(
        d: &MoveData,
        type_index: &crate::fxhash::FxHashMap<String, u8>,
        flag_bits: &crate::fxhash::FxHashMap<String, u64>,
    ) -> MoveStatic {
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
        let mut secondaries: smallvec::SmallVec<[HitEffect; 2]> = d
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
            move_type: TypeId(*type_index.get(&d.move_type).expect("move type interned")),
            category,
            base_power: d.base_power as i32,
            accuracy,
            pp: d.pp as i32,
            no_pp_boosts: xb("noPPBoosts"),
            priority: d.priority,
            target: leak_intern(&d.target),
            crit_ratio: d.crit_ratio.unwrap_or(1) as i32,
            will_crit: x("willCrit").and_then(|v| v.as_bool()),
            flags: d.flags.keys().map(|k| flag_bits.get(k).copied().unwrap_or(0)).fold(0, |a, b| a | b),
            status: d.status.as_deref().map(leak_intern),
            volatile_status: d.volatile_status.as_deref().map(leak_intern),
            side_condition: d.side_condition.as_deref().map(leak_intern),
            weather: d.weather.as_deref().map(leak_intern),
            pseudo_weather: x("pseudoWeather").and_then(|v| v.as_str()).map(leak_intern),
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
                    Some("true")
                } else {
                    v.as_str().map(leak_intern)
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
            non_ghost_target: x("nonGhostTarget").and_then(|v| v.as_str()).map(leak_intern),
            callbacks: d.callbacks.clone(),
            cb_mask: CbMask::EMPTY,
            condition: d.condition.clone(),
            cb_ops: [MoveCbOp::Unported; 256],
            secondary_on_hit_op: MoveCbOp::Unported,
            self_on_hit_op: MoveCbOp::Unported,
            damage_cb: DamageCbKind::None,
            base_power_cb: BasePowerCbKind::None,
            before_move_cb: BeforeMoveCbKind::None,
            kingsrock: false,
            calls_other_move: false,
            is_pursuit: false,
            is_solarbeam: false,
        }
    }

    /// Flag test through the dex registry (unknown flags are never set).
    pub fn has_flag(&self, dex: &Dex, flag: &str) -> bool {
        self.flags & dex.flag_bit(flag) != 0
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
    /// Registered-callback bitset over `callbacks` + code-only builtins
    /// (`conditions::has_builtin`). Filled at load.
    pub mask: CbMask,
    /// Precomputed per-callback Order/Priority/SubOrder (from `nums`).
    pub cb_nums: Vec<(Cb, CbNums)>,
    /// Direct jump-table ops indexed by `Cb.0`.
    pub cb_ops: [CondCbOp; 256],
    pub duration_cb: DurationCbKind,
    pub has_on_restart: bool,
    pub has_on_field_restart: bool,
    pub has_on_side_restart: bool,
    pub is_permanent_rule: bool,
}

impl CondEntry {
    pub fn has_callback(&self, name: &str) -> bool {
        self.callbacks.iter().any(|c| c == name)
    }

    #[inline]
    pub fn cb_num(&self, cb: Cb) -> CbNums {
        self.cb_nums.iter().find(|(c, _)| *c == cb).map(|(_, n)| *n).unwrap_or_default()
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
    pub replay_fingerprint: [u8; 16],
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
    /// Registered callback names, dense-id order (`Cb` indexes this).
    pub cb_names: Vec<String>,
    cb_index: crate::fxhash::FxHashMap<String, u16>,
    /// Union of all condition + item masks (same universe as
    /// `possible_callbacks`; move callbacks excluded).
    pub possible_mask: CbMask,
    /// Per-Cb flag bits: 1 = name ends with "SwitchIn", 2 = ends with
    /// "RedirectTarget" (resolvePriority special cases).
    pub cb_flags: Vec<u8>,
    pub known: KnownCbs,
    pub known_items: KnownItems,
    pub known_species: KnownSpecies,
    pub known_types: KnownTypes,
    /// Bitmask over `TypeId.0` for Gen 2 special-split types.
    pub special_type_mask: u32,
    /// Display names of interned types ("Fire", ..., "???"), TypeId order.
    pub type_names: Vec<String>,
    /// Keys: both display ("Fire") and toid ("fire") forms.
    type_index: crate::fxhash::FxHashMap<String, u8>,
    /// attacker-major NxN: +1 super effective, -1 resisted, 0 neutral/immune.
    eff_matrix: Vec<i8>,
    /// attacker-major NxN: damage_taken == 3 (immune).
    imm_matrix: Vec<bool>,
    /// Non-type damage_taken keys (psn/brn/trapped/sandstorm/...) → per-type
    /// immunity row.
    status_imm: crate::fxhash::FxHashMap<String, Box<[bool]>>,
    /// Precomputed TypeList per species (SpeciesId order).
    species_types: Vec<TypeList>,
    /// Move-flag name → bitmask bit.
    flag_bits: crate::fxhash::FxHashMap<String, u64>,
    /// CondId of each `Status` discriminant ("", brn, par, slp, frz, psn,
    /// tox, fnt) — collection must not hit the string index per event.
    pub status_conds: [Option<CondId>; 8],
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
        // ---- type interning (typechart order is BTreeMap-sorted, then "???")
        let mut type_names: Vec<String> = Vec::new();
        let mut type_index: crate::fxhash::FxHashMap<String, u8> = Default::default();
        for (key, td) in &file.typechart {
            let i = type_names.len() as u8;
            type_names.push(td.name.clone());
            type_index.insert(td.name.clone(), i);
            type_index.insert(key.clone(), i);
        }
        {
            let i = type_names.len() as u8;
            type_names.push("???".to_string());
            type_index.insert("???".to_string(), i);
        }
        let nt = type_names.len();
        let tid = |name: &str| -> TypeId {
            TypeId(*type_index.get(name).unwrap_or_else(|| panic!("unknown type {name}")))
        };
        // attacker-major matrices; the "???" row/col stays neutral/non-immune.
        let mut eff_matrix = vec![0i8; nt * nt];
        let mut imm_matrix = vec![false; nt * nt];
        let mut status_imm: crate::fxhash::FxHashMap<String, Vec<bool>> = Default::default();
        for (key, td) in &file.typechart {
            let def = tid(key).0 as usize;
            for (att_key, &code) in &td.damage_taken {
                if let Some(&att) = type_index.get(att_key) {
                    eff_matrix[att as usize * nt + def] = match code {
                        1 => 1,
                        2 => -1,
                        _ => 0,
                    };
                    imm_matrix[att as usize * nt + def] = code == 3;
                } else {
                    let row = status_imm.entry(att_key.clone()).or_insert_with(|| vec![false; nt]);
                    row[def] = code == 3;
                }
            }
        }
        let status_imm: crate::fxhash::FxHashMap<String, Box<[bool]>> =
            status_imm.into_iter().map(|(k, v)| (k, v.into_boxed_slice())).collect();
        let mut special_type_mask = 0u32;
        for name in ["Fire", "Water", "Grass", "Ice", "Electric", "Dark", "Psychic", "Dragon"] {
            if let Some(&i) = type_index.get(name) {
                if (i as u32) < 32 {
                    special_type_mask |= 1u32 << i;
                }
            }
        }
        let known_types = KnownTypes {
            unknown: tid("???"),
            normal: tid("Normal"),
            water: tid("Water"),
            fire: tid("Fire"),
            flying: tid("Flying"),
            ghost: tid("Ghost"),
            grass: tid("Grass"),
            fighting: tid("Fighting"),
            ground: tid("Ground"),
        };

        let moves = Table::build(file.moves);
        // move-flag registry (gen2: ~20 distinct flags; bitmask capacity 64)
        let mut flag_bits: crate::fxhash::FxHashMap<String, u64> = Default::default();
        for d in &moves.values {
            for k in d.flags.keys() {
                if !flag_bits.contains_key(k) {
                    let bit = 1u64 << flag_bits.len();
                    assert!(flag_bits.len() < 64, "flag bitmask overflow");
                    flag_bits.insert(k.clone(), bit);
                }
            }
        }
        let moves_static: Vec<MoveStatic> = moves
            .values
            .iter()
            .map(|d| MoveStatic::parse(d, &type_index, &flag_bits))
            .collect();

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
                    mask: CbMask::EMPTY,
                    cb_nums: Vec::new(),
                    cb_ops: [CondCbOp::Unported; 256],
                    duration_cb: DurationCbKind::None,
                    has_on_restart: false,
                    has_on_field_restart: false,
                    has_on_side_restart: false,
                    is_permanent_rule: false,
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
                    mask: CbMask::EMPTY,
                    cb_nums: Vec::new(),
                    cb_ops: [CondCbOp::Unported; 256],
                    duration_cb: DurationCbKind::None,
                    has_on_restart: false,
                    has_on_field_restart: false,
                    has_on_side_restart: false,
                    is_permanent_rule: false,
                },
            );
        }
        // rules that act as runtime effects (pseudo-weathers / onSetStatus)
        for (id, name, callbacks) in [
            ("maxtotallevel", "Max Total Level", vec![]),
            ("stadiumsleepclause", "Stadium Sleep Clause", vec!["onSetStatus".to_string()]),
            ("sleepclausemod", "Sleep Clause Mod", vec!["onSetStatus".to_string()]),
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
                    mask: CbMask::EMPTY,
                    cb_nums: Vec::new(),
                    cb_ops: [CondCbOp::Unported; 256],
                    duration_cb: DurationCbKind::None,
                    has_on_restart: false,
                    has_on_field_restart: false,
                    has_on_side_restart: false,
                    is_permanent_rule: true,
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
                mask: CbMask::EMPTY,
                cb_nums: Vec::new(),
                cb_ops: [CondCbOp::Unported; 256],
                duration_cb: DurationCbKind::None,
                has_on_restart: false,
                has_on_field_restart: false,
                has_on_side_restart: false,
                is_permanent_rule: false,
            });
        }

        let mut items = Table::build(file.items);
        let mut possible_callbacks: std::collections::HashSet<String> =
            entries.values().flat_map(|e| e.callbacks.iter().cloned()).collect();
        for item in &items.values {
            possible_callbacks.extend(item.callbacks.iter().cloned());
        }
        // crate::battle::conditions::has_builtin constants
        possible_callbacks.insert("onLockMove".to_string());
        possible_callbacks.insert("onSemiLockMove".to_string());

        // ---- callback registry: dense ids for every name any entity carries.
        // Registration order is deterministic (BTreeMap iteration + fixed
        // extras), so ids are stable across loads of the same data.
        let mut cb_names: Vec<String> = Vec::new();
        let mut cb_index: crate::fxhash::FxHashMap<String, u16> = Default::default();
        let register = |name: &str, cb_names: &mut Vec<String>, cb_index: &mut crate::fxhash::FxHashMap<String, u16>| -> Cb {
            if let Some(&i) = cb_index.get(name) {
                return Cb(i);
            }
            let i = cb_names.len() as u16;
            cb_names.push(name.to_string());
            cb_index.insert(name.to_string(), i);
            Cb(i)
        };
        // strip the Order/Priority/SubOrder suffix off a nums key
        fn nums_base(key: &str) -> (&str, u8) {
            if let Some(b) = key.strip_suffix("SubOrder") {
                (b, 2)
            } else if let Some(b) = key.strip_suffix("Priority") {
                (b, 1)
            } else if let Some(b) = key.strip_suffix("Order") {
                (b, 0)
            } else {
                (key, 3)
            }
        }
        for e in entries.values() {
            for c in &e.callbacks {
                register(c, &mut cb_names, &mut cb_index);
            }
            for k in e.nums.keys() {
                register(nums_base(k).0, &mut cb_names, &mut cb_index);
            }
        }
        for item in &items.values {
            for c in &item.callbacks {
                register(c, &mut cb_names, &mut cb_index);
            }
            for k in item.extra.keys() {
                if k.starts_with("on")
                    && (k.ends_with("Order") || k.ends_with("Priority") || k.ends_with("SubOrder"))
                {
                    register(nums_base(k).0, &mut cb_names, &mut cb_index);
                }
            }
        }
        for ms in &moves_static {
            for c in &ms.callbacks {
                if !c.contains('.') {
                    register(c, &mut cb_names, &mut cb_index);
                }
            }
        }
        for builtin in [
            "onLockMove",
            "onSemiLockMove",
            "onHit",
            "onAfterHit",
            "onStart",
            "damageCallback",
            "basePowerCallback",
            "beforeMoveCallback",
            "onHitField",
            "onHitSide",
            "onSetStatus",
            "beforeTurnCallback",
        ] {
            register(builtin, &mut cb_names, &mut cb_index);
        }
        assert!(cb_names.len() <= 256, "CbMask capacity exceeded: {} callbacks", cb_names.len());

        let lookup = |name: &str| cb_index.get(name).map(|&i| Cb(i)).unwrap_or(Cb::NONE);

        // fill per-entry masks + precomputed nums
        let mut moves_static = moves_static;
        for (key, e) in entries.iter_mut() {
            for c in &e.callbacks {
                e.mask.set(lookup(c));
            }
            for extra in ["onLockMove", "onSemiLockMove"] {
                if crate::battle::conditions::has_builtin(key, extra) {
                    e.mask.set(lookup(extra));
                }
            }
            for (k, &n) in &e.nums {
                let (base, kind) = nums_base(k);
                let cb = lookup(base);
                let slot = match e.cb_nums.iter_mut().find(|(c, _)| *c == cb) {
                    Some((_, s)) => s,
                    None => {
                        e.cb_nums.push((cb, CbNums::default()));
                        &mut e.cb_nums.last_mut().unwrap().1
                    }
                };
                match kind {
                    0 => slot.order = Some(n),
                    1 => slot.priority = Some(n),
                    _ => slot.sub_order = Some(n),
                }
            }
            e.has_on_restart = e.callbacks.iter().any(|c| c == "onRestart");
            e.has_on_field_restart = e.callbacks.iter().any(|c| c == "onFieldRestart");
            e.has_on_side_restart = e.callbacks.iter().any(|c| c == "onSideRestart");
            e.duration_cb = classify_duration_cb(key);
            for (idx, cb_name) in cb_names.iter().enumerate() {
                e.cb_ops[idx] = classify_cond_cb(key, cb_name);
            }
        }
        for (idx_item, item) in items.values.iter_mut().enumerate() {
            let item_key = &items.keys[idx_item];
            for c in &item.callbacks {
                item.mask.set(lookup(c));
            }
            let mut cb_nums: Vec<(Cb, CbNums)> = Vec::new();
            for (k, v) in &item.extra {
                if !(k.starts_with("on")
                    && (k.ends_with("Order") || k.ends_with("Priority") || k.ends_with("SubOrder")))
                {
                    continue;
                }
                let Some(n) = v.as_i64() else { continue };
                let (base, kind) = nums_base(k);
                let cb = lookup(base);
                let slot = match cb_nums.iter_mut().find(|(c, _)| *c == cb) {
                    Some((_, s)) => s,
                    None => {
                        cb_nums.push((cb, CbNums::default()));
                        &mut cb_nums.last_mut().unwrap().1
                    }
                };
                match kind {
                    0 => slot.order = Some(n as i32),
                    1 => slot.priority = Some(n as i32),
                    _ => slot.sub_order = Some(n as i32),
                }
            }
            item.cb_nums = cb_nums;
            item.type_boost = item_type_boost_name(item_key).and_then(|tn| type_index.get(tn).map(|&i| TypeId(i)));
            for (idx, cb_name) in cb_names.iter().enumerate() {
                item.cb_ops[idx] = classify_item_cb(item_key, cb_name, item.type_boost);
            }
        }
        for (idx_mv, ms) in moves_static.iter_mut().enumerate() {
            let move_key = &moves.keys[idx_mv];
            for c in &ms.callbacks {
                if !c.contains('.') {
                    ms.cb_mask.set(lookup(c));
                }
            }
            for (idx, cb_name) in cb_names.iter().enumerate() {
                ms.cb_ops[idx] = classify_move_cb(move_key, cb_name);
            }
            ms.secondary_on_hit_op = classify_move_cb(move_key, "secondary.onHit");
            ms.self_on_hit_op = classify_move_cb(move_key, "self.onHit");
            ms.damage_cb = classify_damage_cb(move_key);
            ms.base_power_cb = classify_base_power_cb(move_key);
            ms.before_move_cb = classify_before_move_cb(move_key);
            ms.kingsrock = KINGSROCK_MOVES_TABLE.contains(&move_key.as_str());
            ms.calls_other_move = matches!(
                move_key.as_str(),
                "metronome" | "mimic" | "mirrormove" | "sketch" | "sleeptalk" | "transform"
            );
            ms.is_pursuit = move_key == "pursuit";
            ms.is_solarbeam = move_key == "solarbeam";
        }

        let mut possible_mask = CbMask::EMPTY;
        for e in entries.values() {
            possible_mask.or_with(&e.mask);
        }
        for item in &items.values {
            possible_mask.or_with(&item.mask);
        }

        let cb_flags: Vec<u8> = cb_names
            .iter()
            .map(|n| {
                (n.ends_with("SwitchIn") as u8) | ((n.ends_with("RedirectTarget") as u8) << 1)
            })
            .collect();

        let known = KnownCbs {
            on_hit: lookup("onHit"),
            on_after_hit: lookup("onAfterHit"),
            on_start: lookup("onStart"),
            on_set_status: lookup("onSetStatus"),
            damage_callback: lookup("damageCallback"),
            base_power_callback: lookup("basePowerCallback"),
            before_move_callback: lookup("beforeMoveCallback"),
            before_turn_callback: lookup("beforeTurnCallback"),
            on_hit_field: lookup("onHitField"),
            on_hit_side: lookup("onHitSide"),
        };

        let conds = Table::build(entries);
        let status_conds = ["", "brn", "par", "slp", "frz", "psn", "tox", "fnt"]
            .map(|s| if s.is_empty() { None } else { conds.id(s) });
        let known_items = KnownItems {
            thickclub: items.id("thickclub"),
            lightball: items.id("lightball"),
            metalpowder: items.id("metalpowder"),
            quickclaw: items.id("quickclaw"),
        };
        let species = Table::build(file.species);
        let species_types: Vec<TypeList> = species
            .values
            .iter()
            .map(|sp| {
                let a = TypeId(*type_index.get(&sp.types[0]).expect("species type interned"));
                match sp.types.get(1) {
                    Some(b) => TypeList::two(a, TypeId(*type_index.get(b).unwrap())),
                    None => TypeList::one(a),
                }
            })
            .collect();
        let known_species = KnownSpecies {
            cubone: species.id("cubone"),
            marowak: species.id("marowak"),
            pikachu: species.id("pikachu"),
            ditto: species.id("ditto"),
            farfetchd: species.id("farfetchd"),
            chansey: species.id("chansey"),
        };

        Ok(Dex {
            replay_fingerprint: crate::replay::fingerprint(json),
            species,
            moves,
            items,
            conditions: Table::build(file.conditions),
            typechart: file.typechart,
            moves_static,
            conds,
            possible_callbacks,
            cb_names,
            cb_index,
            possible_mask,
            cb_flags,
            known,
            known_items,
            known_species,
            known_types,
            special_type_mask,
            type_names,
            type_index,
            eff_matrix,
            imm_matrix,
            status_imm,
            species_types,
            flag_bits,
            status_conds,
        })
    }

    /// Type id by display ("Fire") or toid ("fire") name.
    #[inline]
    pub fn type_id(&self, name: &str) -> Option<TypeId> {
        self.type_index.get(name).map(|&i| TypeId(i))
    }

    #[inline]
    pub fn type_name(&self, t: TypeId) -> &str {
        &self.type_names[t.0 as usize]
    }

    /// +1 super effective, -1 resisted, 0 neutral/immune (PS getEffectiveness).
    #[inline]
    pub fn eff(&self, att: TypeId, def: TypeId) -> i32 {
        self.eff_matrix[att.0 as usize * self.type_names.len() + def.0 as usize] as i32
    }

    /// damage_taken == 3 for this matchup (PS getImmunity == false).
    #[inline]
    pub fn type_immune(&self, att: TypeId, def: TypeId) -> bool {
        self.imm_matrix[att.0 as usize * self.type_names.len() + def.0 as usize]
    }

    /// PS getImmunity for a non-type key ('psn', 'trapped', 'sandstorm', ...):
    /// false = immune. Unknown keys are never immune (PS: no damageTaken row).
    #[inline]
    pub fn status_key_immune(&self, key: &str, def: TypeId) -> bool {
        self.status_imm.get(key).map(|row| row[def.0 as usize]).unwrap_or(false)
    }

    #[inline]
    pub fn species_types(&self, id: SpeciesId) -> TypeList {
        self.species_types[id.0 as usize]
    }

    /// Bit for a move-flag name (0 if the flag doesn't occur in this dex).
    #[inline]
    pub fn flag_bit(&self, flag: &str) -> u64 {
        self.flag_bits.get(flag).copied().unwrap_or(0)
    }

    /// Dense id of a callback name (`Cb::NONE` if never registered — such a
    /// name can have no handlers in this format).
    #[inline]
    pub fn cb(&self, name: &str) -> Cb {
        self.cb_index.get(name).map(|&i| Cb(i)).unwrap_or(Cb::NONE)
    }

    #[inline]
    pub fn cb_key(&self, cb: Cb) -> &str {
        &self.cb_names[cb.0 as usize]
    }

    /// resolvePriority's `callbackName.endsWith("SwitchIn")`.
    #[inline]
    pub fn cb_ends_switch_in(&self, cb: Cb) -> bool {
        cb.exists() && self.cb_flags[cb.0 as usize] & 1 != 0
    }

    #[inline]
    pub fn cb_ends_redirect_target(&self, cb: Cb) -> bool {
        cb.exists() && self.cb_flags[cb.0 as usize] & 2 != 0
    }

    pub fn move_static(&self, id: MoveId) -> &MoveStatic {
        &self.moves_static[id.0 as usize]
    }

    pub fn canonical_move_id(&self, id: MoveId) -> MoveId {
        self.moves.get(id).extra.get("placeholderFor").and_then(Value::as_str)
            .map(|name| self.moves.id(&toid(name)).expect("move placeholder must resolve"))
            .unwrap_or(id)
    }

    pub fn conds_id(&self, key: &str) -> Option<CondId> {
        self.conds.id(key)
    }

    #[inline]
    pub fn conds_key(&self, id: CondId) -> &'static str {
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
}

fn classify_duration_cb(cond: &str) -> DurationCbKind {
    match cond {
        "partiallytrapped" => DurationCbKind::Rand3To6,
        "lockedmove" => DurationCbKind::Rand2To4,
        "bide" => DurationCbKind::Rand3To5,
        "disable" => DurationCbKind::Rand2To6,
        "encore" => DurationCbKind::Rand3To7,
        "safeguard" | "raindance" | "sunnyday" | "sandstorm" => DurationCbKind::Const5,
        _ => DurationCbKind::None,
    }
}

fn classify_damage_cb(mv: &str) -> DamageCbKind {
    match mv {
        "counter" => DamageCbKind::Counter,
        "mirrorcoat" => DamageCbKind::MirrorCoat,
        "psywave" => DamageCbKind::Psywave,
        "superfang" => DamageCbKind::SuperFang,
        _ => DamageCbKind::None,
    }
}

fn classify_base_power_cb(mv: &str) -> BasePowerCbKind {
    match mv {
        "flail" | "reversal" => BasePowerCbKind::FlailReversal,
        "frustration" => BasePowerCbKind::Frustration,
        "return" => BasePowerCbKind::Return,
        "triplekick" => BasePowerCbKind::TripleKick,
        "pursuit" => BasePowerCbKind::Pursuit,
        "beatup" => BasePowerCbKind::BeatUp,
        "hiddenpower" => BasePowerCbKind::HiddenPower,
        "furycutter" => BasePowerCbKind::FuryCutter,
        "rollout" => BasePowerCbKind::Rollout,
        _ => BasePowerCbKind::None,
    }
}

fn classify_before_move_cb(mv: &str) -> BeforeMoveCbKind {
    match mv {
        "bide" => BeforeMoveCbKind::Bide,
        _ => BeforeMoveCbKind::None,
    }
}

fn item_type_boost_name(key: &str) -> Option<&'static str> {
    Some(match key {
        "blackbelt" => "Fighting",
        "blackglasses" => "Dark",
        "charcoal" => "Fire",
        "dragonfang" => "Dragon",
        "hardstone" => "Rock",
        "magnet" => "Electric",
        "metalcoat" => "Steel",
        "miracleseed" => "Grass",
        "mysticwater" => "Water",
        "nevermeltice" => "Ice",
        "pinkbow" => "Normal",
        "poisonbarb" => "Poison",
        "polkadotbow" => "Normal",
        "sharpbeak" => "Flying",
        "silverpowder" => "Bug",
        "softsand" => "Ground",
        "spelltag" => "Ghost",
        "twistedspoon" => "Psychic",
        _ => return None,
    })
}

const KINGSROCK_MOVES_TABLE: [&str; 91] = [
    "absorb", "aeroblast", "barrage", "beatup", "bide", "bonerush", "bonemerang", "cometpunch",
    "counter", "crabhammer", "crosschop", "cut", "dig", "doublekick", "doubleslap", "doubleedge",
    "dragonrage", "drillpeck", "eggbomb", "explosion", "extremespeed", "falseswipe", "feintattack",
    "flail", "fly", "frustration", "furyattack", "furycutter", "furyswipes", "gigadrain",
    "hiddenpower", "highjumpkick", "hornattack", "hydropump", "jumpkick", "karatechop",
    "leechlife", "machpunch", "magnitude", "megadrain", "megakick", "megapunch", "megahorn",
    "mirrorcoat", "nightshade", "outrage", "payday", "peck", "petaldance", "pinmissile", "pound",
    "present", "pursuit", "psywave", "quickattack", "rage", "rapidspin", "razorleaf", "razorwind",
    "return", "reversal", "rockthrow", "rollout", "scratch", "seismictoss", "selfdestruct",
    "skullbash", "skyattack", "slam", "slash", "snore", "solarbeam", "sonicboom", "spikecannon",
    "strength", "struggle", "submission", "superfang", "surf", "swift", "tackle", "takedown",
    "thief", "thrash", "triplekick", "twineedle", "visegrip", "vinewhip", "watergun", "waterfall",
    "wingattack",
];

fn classify_cond_cb(cond: &str, cb: &str) -> CondCbOp {
    match (cond, cb) {
        ("trapper" | "brnattackdrop" | "parspeeddrop" | "leppaberry", _) => CondCbOp::Noop,
        ("stadiumsleepclause", "onSetStatus") => CondCbOp::Noop,
        ("brn", "onStart") => CondCbOp::BrnStart,
        ("brn", "onAfterMoveSelf") | ("brn", "onAfterSwitchInSelf") => CondCbOp::BrnAfterMoveSelf,
        ("brn", "onSwitchIn") => CondCbOp::BrnSwitchIn,
        ("par", "onStart") => CondCbOp::ParStart,
        ("par", "onBeforeMove") => CondCbOp::ParBeforeMove,
        ("par", "onSwitchIn") => CondCbOp::ParSwitchIn,
        ("slp", "onStart") => CondCbOp::SlpStart,
        ("slp", "onBeforeMove") => CondCbOp::SlpBeforeMove,
        ("frz", "onStart") => CondCbOp::FrzStart,
        ("frz", "onBeforeMove") => CondCbOp::FrzBeforeMove,
        ("frz", "onAfterMoveSecondary") => CondCbOp::FrzAfterMoveSecondary,
        ("frz", "onAfterMoveSecondarySelf") => CondCbOp::FrzAfterMoveSecondarySelf,
        ("frz", "onResidual") => CondCbOp::FrzResidual,
        ("psn", "onStart") => CondCbOp::PsnStart,
        ("psn", "onAfterMoveSelf") | ("psn", "onAfterSwitchInSelf") => CondCbOp::PsnAfterMoveSelf,
        ("tox", "onStart") => CondCbOp::ToxStart,
        ("tox", "onAfterMoveSelf") => CondCbOp::ToxAfterMoveSelf,
        ("tox", "onSwitchIn") => CondCbOp::ToxSwitchIn,
        ("tox", "onAfterSwitchInSelf") => CondCbOp::ToxAfterSwitchInSelf,
        ("confusion", "onStart") => CondCbOp::ConfusionStart,
        ("confusion", "onEnd") => CondCbOp::ConfusionEnd,
        ("confusion", "onBeforeMove") => CondCbOp::ConfusionBeforeMove,
        ("flinch", "onBeforeMove") => CondCbOp::FlinchBeforeMove,
        ("partiallytrapped", "onStart") => CondCbOp::PartiallyTrappedStart,
        ("partiallytrapped", "onResidual") => CondCbOp::PartiallyTrappedResidual,
        ("partiallytrapped", "onEnd") => CondCbOp::PartiallyTrappedEnd,
        ("partiallytrapped", "onTrapPokemon") => CondCbOp::PartiallyTrappedTrapPokemon,
        ("residualdmg", "onStart") => CondCbOp::ResidualDmgStart,
        ("residualdmg", "onAfterMoveSelf") | ("residualdmg", "onAfterSwitchInSelf") => {
            CondCbOp::ResidualDmgAfterMoveSelf
        }
        ("mustrecharge", "onBeforeMove") => CondCbOp::MustRechargeBeforeMove,
        ("mustrecharge", "onStart") => CondCbOp::MustRechargeStart,
        ("mustrecharge", "onLockMove") => CondCbOp::MustRechargeLockMove,
        ("trapped", "onTrapPokemon") => CondCbOp::TrappedTrapPokemon,
        ("trapped", "onStart") => CondCbOp::TrappedStart,
        ("raindance", "onFieldStart") => CondCbOp::RainDanceFieldStart,
        ("raindance", "onFieldResidual") => CondCbOp::RainDanceFieldResidual,
        ("raindance", "onFieldEnd") => CondCbOp::RainDanceFieldEnd,
        ("sunnyday", "onFieldStart") => CondCbOp::SunnyDayFieldStart,
        ("sunnyday", "onFieldResidual") => CondCbOp::SunnyDayFieldResidual,
        ("sunnyday", "onFieldEnd") => CondCbOp::SunnyDayFieldEnd,
        ("sunnyday", "onImmunity") => CondCbOp::SunnyDayImmunity,
        ("sandstorm", "onFieldStart") => CondCbOp::SandstormFieldStart,
        ("sandstorm", "onFieldResidual") => CondCbOp::SandstormFieldResidual,
        ("sandstorm", "onWeather") => CondCbOp::SandstormWeather,
        ("sandstorm", "onFieldEnd") => CondCbOp::SandstormFieldEnd,
        ("sleepclausemod", "onSetStatus") => CondCbOp::SleepClauseModSetStatus,
        ("freezeclausemod", "onSetStatus") => CondCbOp::FreezeClauseModSetStatus,
        ("lightscreen", "onSideStart") => CondCbOp::LightScreenSideStart,
        ("reflect", "onSideStart") => CondCbOp::ReflectSideStart,
        ("lightscreen", "onSideEnd") => CondCbOp::LightScreenSideEnd,
        ("reflect", "onSideEnd") => CondCbOp::ReflectSideEnd,
        ("safeguard", "onSideStart") => CondCbOp::SafeguardSideStart,
        ("safeguard", "onSideEnd") => CondCbOp::SafeguardSideEnd,
        ("safeguard", "onSetStatus") => CondCbOp::SafeguardSetStatus,
        ("safeguard", "onTryAddVolatile") => CondCbOp::SafeguardTryAddVolatile,
        ("spikes", "onSideStart") => CondCbOp::SpikesSideStart,
        ("spikes", "onEntryHazard") => CondCbOp::SpikesEntryHazard,
        ("leechseed", "onStart") => CondCbOp::LeechSeedStart,
        ("leechseed", "onAfterMoveSelf") => CondCbOp::LeechSeedAfterMoveSelf,
        ("nightmare", "onStart") => CondCbOp::NightmareStart,
        ("nightmare", "onAfterMoveSelf") => CondCbOp::NightmareAfterMoveSelf,
        ("curse", "onStart") => CondCbOp::CurseStart,
        ("curse", "onAfterMoveSelf") => CondCbOp::CurseAfterMoveSelf,
        ("foresight", "onStart") => CondCbOp::ForesightStart,
        ("foresight", "onNegateImmunity") => CondCbOp::ForesightNegateImmunity,
        ("lockon", "onSourceAccuracy") => CondCbOp::LockOnSourceAccuracy,
        ("destinybond", "onStart") => CondCbOp::DestinyBondStart,
        ("destinybond", "onFaint") => CondCbOp::DestinyBondFaint,
        ("destinybond", "onBeforeMove") => CondCbOp::DestinyBondBeforeMove,
        ("destinybond", "onMoveAborted") => CondCbOp::DestinyBondMoveAborted,
        ("destinybond", "onFoeAfterMoveSelf") => CondCbOp::DestinyBondFoeAfterMoveSelf,
        ("destinybond", "onFoeSwitchOut") => CondCbOp::DestinyBondFoeSwitchOut,
        ("perishsong", "onResidual") => CondCbOp::PerishSongResidual,
        ("perishsong", "onEnd") => CondCbOp::PerishSongEnd,
        ("encore", "onStart") => CondCbOp::EncoreStart,
        ("encore", "onOverrideAction") => CondCbOp::EncoreOverrideAction,
        ("encore", "onResidual") => CondCbOp::EncoreResidual,
        ("encore", "onEnd") => CondCbOp::EncoreEnd,
        ("encore", "onDisableMove") => CondCbOp::EncoreDisableMove,
        ("disable", "onStart") => CondCbOp::DisableStart,
        ("disable", "onEnd") => CondCbOp::DisableEnd,
        ("disable", "onBeforeMove") => CondCbOp::DisableBeforeMove,
        ("disable", "onDisableMove") => CondCbOp::DisableDisableMove,
        ("mist", "onStart") => CondCbOp::MistStart,
        ("mist", "onTryBoost") => CondCbOp::MistTryBoost,
        ("substitute", "onStart") => CondCbOp::SubstituteStart,
        ("substitute", "onEnd") => CondCbOp::SubstituteEnd,
        ("substitute", "onTryPrimaryHit") => CondCbOp::SubstituteTryPrimaryHit,
        ("bide", "onStart") => CondCbOp::BideStart,
        ("bide", "onDamage") => CondCbOp::BideDamage,
        ("bide", "onBeforeMove") => CondCbOp::BideBeforeMove,
        ("bide", "onMoveAborted") => CondCbOp::BideMoveAborted,
        ("bide", "onEnd") => CondCbOp::BideEnd,
        ("bide", "onSemiLockMove") => CondCbOp::BideSemiLockMove,
        ("rage", "onStart") => CondCbOp::RageStart,
        ("rage", "onHit") => CondCbOp::RageHit,
        ("rage", "onBeforeMove") => CondCbOp::RageBeforeMove,
        ("rollout", "onStart") => CondCbOp::RolloutStart,
        ("rollout", "onResidual") => CondCbOp::RolloutResidual,
        ("rollout", "onLockMove") => CondCbOp::RolloutLockMove,
        ("furycutter", "onStart") => CondCbOp::FuryCutterStart,
        ("furycutter", "onRestart") => CondCbOp::FuryCutterRestart,
        ("defensecurl", "onRestart") | ("minimize", "onRestart") => {
            CondCbOp::DefenseCurlMinimizeRestart
        }
        ("minimize", "onSourceModifyDamage") => CondCbOp::MinimizeSourceModifyDamage,
        ("attract", "onStart") => CondCbOp::AttractStart,
        ("attract", "onUpdate") => CondCbOp::AttractUpdate,
        ("attract", "onBeforeMove") => CondCbOp::AttractBeforeMove,
        ("attract", "onEnd") => CondCbOp::AttractEnd,
        ("focusenergy", "onStart") => CondCbOp::FocusEnergyStart,
        ("focusenergy", "onModifyCritRatio") => CondCbOp::FocusEnergyModifyCritRatio,
        ("protect", "onStart") => CondCbOp::ProtectStart,
        ("protect", "onTryHit") => CondCbOp::ProtectTryHit,
        ("endure", "onStart") => CondCbOp::EndureStart,
        ("endure", "onDamage") => CondCbOp::EndureDamage,
        ("stall", "onStart") => CondCbOp::StallStart,
        ("stall", "onStallMove") => CondCbOp::StallMove,
        ("stall", "onRestart") => CondCbOp::StallRestart,
        ("twoturnmove", "onStart") => CondCbOp::TwoTurnMoveStart,
        ("twoturnmove", "onEnd") => CondCbOp::TwoTurnMoveEnd,
        ("twoturnmove", "onLockMove") => CondCbOp::TwoTurnMoveLockMove,
        ("twoturnmove", "onMoveAborted") => CondCbOp::TwoTurnMoveAborted,
        ("dig", "onImmunity") => CondCbOp::DigImmunity,
        ("dig", "onInvulnerability") => CondCbOp::DigInvulnerability,
        ("fly", "onInvulnerability") => CondCbOp::FlyInvulnerability,
        ("dig", "onSourceBasePower") => CondCbOp::DigSourceBasePower,
        ("fly", "onSourceBasePower") => CondCbOp::FlySourceBasePower,
        ("lockedmove", "onStart") => CondCbOp::LockedMoveStart,
        ("lockedmove", "onResidual") => CondCbOp::LockedMoveResidual,
        ("lockedmove", "onEnd") => CondCbOp::LockedMoveEnd,
        ("lockedmove", "onLockMove") => CondCbOp::LockedMoveLockMove,
        ("lockedmove", "onMoveAborted") => CondCbOp::LockedMoveAborted,
        ("pursuit", "onFoeBeforeSwitchOut") => CondCbOp::PursuitFoeBeforeSwitchOut,
        ("futuremove", "onStart") => CondCbOp::FutureMoveStart,
        ("futuremove", "onResidual") => CondCbOp::FutureMoveResidual,
        ("futuremove", "onEnd") => CondCbOp::FutureMoveEnd,
        _ => CondCbOp::Unported,
    }
}

fn classify_item_cb(key: &str, cb: &str, type_boost: Option<TypeId>) -> ItemCbOp {
    if type_boost.is_some() && cb == "onModifyDamage" {
        return ItemCbOp::TypeBoostModifyDamage;
    }
    match (key, cb) {
        ("scopelens", "onModifyCritRatio") => ItemCbOp::ScopeLensCrit,
        ("stick", "onModifyCritRatio") => ItemCbOp::StickCrit,
        ("luckypunch", "onModifyCritRatio") => ItemCbOp::LuckyPunchCrit,
        ("brightpowder", "onModifyAccuracy") => ItemCbOp::BrightPowderAccuracy,
        ("focusband", "onDamage") => ItemCbOp::FocusBandDamage,
        ("kingsrock", "onModifyMove") => ItemCbOp::KingsRockModifyMove,
        ("leftovers", "onResidual") => ItemCbOp::LeftoversResidual,
        ("berryjuice", "onResidual") => ItemCbOp::BerryJuiceResidual,
        ("berry", "onResidual") | ("goldberry", "onResidual") => ItemCbOp::BerryResidual,
        ("berry", "onTryEatItem") => ItemCbOp::BerryTryEat,
        ("goldberry", "onTryEatItem") => ItemCbOp::GoldBerryTryEat,
        ("berry", "onEat") => ItemCbOp::BerryEat,
        ("goldberry", "onEat") => ItemCbOp::GoldBerryEat,
        ("bitterberry", "onUpdate") => ItemCbOp::BitterBerryUpdate,
        ("bitterberry", "onEat") => ItemCbOp::BitterBerryEat,
        ("burntberry", "onUpdate") => ItemCbOp::BurntBerryUpdate,
        ("iceberry", "onUpdate") => ItemCbOp::IceBerryUpdate,
        ("przcureberry", "onUpdate") => ItemCbOp::PrzCureBerryUpdate,
        ("psncureberry", "onUpdate") => ItemCbOp::PsnCureBerryUpdate,
        ("mintberry", "onBeforeMove") | ("mintberry", "onResidual") => ItemCbOp::MintBerryCheck,
        ("burntberry", "onEat") => ItemCbOp::BurntBerryEat,
        ("iceberry", "onEat") => ItemCbOp::IceBerryEat,
        ("mintberry", "onEat") => ItemCbOp::MintBerryEat,
        ("przcureberry", "onEat") => ItemCbOp::PrzCureBerryEat,
        ("psncureberry", "onEat") => ItemCbOp::PsnCureBerryEat,
        ("miracleberry", "onBeforeMove") | ("miracleberry", "onResidual") => ItemCbOp::MiracleBerryCheck,
        ("miracleberry", "onEat") => ItemCbOp::MiracleBerryEat,
        ("mysteryberry", "onUpdate") => ItemCbOp::MysteryBerryUpdate,
        ("mysteryberry", "onEat") => ItemCbOp::MysteryBerryEat,
        ("mail", "onTakeItem") => ItemCbOp::MailTakeItem,
        ("berserkgene", "onUpdate") => ItemCbOp::BerserkGeneUpdate,
        _ => ItemCbOp::Unported,
    }
}

fn classify_move_cb(key: &str, cb: &str) -> MoveCbOp {
    match (key, cb) {
        ("struggle", "onModifyMove") => MoveCbOp::StruggleModifyMove,
        ("thunder", "onModifyMove") => MoveCbOp::ThunderModifyMove,
        ("hiddenpower", "onModifyMove") => MoveCbOp::HiddenPowerModifyMove,
        ("curse", "onModifyMove") => MoveCbOp::CurseModifyMove,
        ("magnitude", "onModifyMove") => MoveCbOp::MagnitudeModifyMove,
        ("present", "onModifyMove") => MoveCbOp::PresentModifyMove,
        ("beatup", "onModifyMove") => MoveCbOp::BeatUpModifyMove,
        ("rollout", "onModifyMove") => MoveCbOp::RolloutModifyMove,
        ("magnitude", "onUseMoveMessage") => MoveCbOp::MagnitudeUseMoveMessage,
        ("dig", "onTryMove") | ("fly", "onTryMove") | ("razorwind", "onTryMove")
        | ("skyattack", "onTryMove") => MoveCbOp::ChargeTryMove,
        ("skullbash", "onTryMove") => MoveCbOp::SkullBashTryMove,
        ("solarbeam", "onTryMove") => MoveCbOp::SolarBeamTryMove,
        ("dig", "onPrepareHit") | ("fly", "onPrepareHit") | ("razorwind", "onPrepareHit")
        | ("skullbash", "onPrepareHit") | ("skyattack", "onPrepareHit")
        | ("solarbeam", "onPrepareHit") => MoveCbOp::ChargePrepareHit,
        ("protect", "onPrepareHit") | ("detect", "onPrepareHit") | ("endure", "onPrepareHit") => {
            MoveCbOp::ProtectDetectEndurePrepareHit
        }
        ("destinybond", "onPrepareHit") | ("perishsong", "onPrepareHit") => {
            MoveCbOp::DestinyPerishPrepareHit
        }
        ("rest", "onTry") => MoveCbOp::RestTry,
        ("sleeptalk", "onTry") | ("snore", "onTry") => MoveCbOp::SleepTalkSnoreTry,
        ("splash", "onTry") => MoveCbOp::NoopTry,
        ("teleport", "onTry") => MoveCbOp::FalseTry,
        ("futuresight", "onTry") => MoveCbOp::FutureSightTry,
        ("substitute", "onTryHit") => MoveCbOp::SubstituteTryHit,
        ("curse", "onTryHit") => MoveCbOp::CurseTryHit,
        ("disable", "onTryHit") => MoveCbOp::DisableTryHit,
        ("foresight", "onTryHit") => MoveCbOp::ForesightTryHit,
        ("lockon", "onTryHit") | ("mindreader", "onTryHit") => MoveCbOp::LockOnTryHit,
        ("roar", "onTryHit") | ("whirlwind", "onTryHit") => MoveCbOp::RoarTryHit,
        ("splash", "onTryHit") => MoveCbOp::SplashTryHit,
        ("swagger", "onTryHit") => MoveCbOp::SwaggerTryHit,
        ("sleeptalk", "onTryHit") => MoveCbOp::SleepTalkTryHit,
        ("attract", "onTryImmunity") => MoveCbOp::AttractTryImmunity,
        ("dreameater", "onTryImmunity") => MoveCbOp::DreamEaterTryImmunity,
        ("leechseed", "onTryImmunity") => MoveCbOp::LeechSeedTryImmunity,
        ("highjumpkick", "onMoveFail") | ("jumpkick", "onMoveFail") => {
            MoveCbOp::HighJumpKickMoveFail
        }
        ("outrage", "onMoveFail") | ("petaldance", "onMoveFail") | ("thrash", "onMoveFail") => {
            MoveCbOp::OutrageMoveFail
        }
        ("falseswipe", "onDamage") => MoveCbOp::FalseSwipeDamage,
        ("haze", "onHitField") => MoveCbOp::HazeHitField,
        ("perishsong", "onHitField") => MoveCbOp::PerishSongHitField,
        ("substitute", "onHit") => MoveCbOp::SubstituteHit,
        ("bellydrum", "onHit") => MoveCbOp::BellyDrumHit,
        ("conversion", "onHit") => MoveCbOp::ConversionHit,
        ("conversion2", "onHit") => MoveCbOp::Conversion2Hit,
        ("detect", "onHit") | ("protect", "onHit") | ("endure", "onHit") => {
            MoveCbOp::ProtectDetectEndureHit
        }
        ("healbell", "onHit") => MoveCbOp::HealBellHit,
        ("lockon", "onHit") | ("mindreader", "onHit") => MoveCbOp::LockOnHit,
        ("meanlook", "onHit") | ("spiderweb", "onHit") => MoveCbOp::MeanLookHit,
        ("metronome", "onHit") => MoveCbOp::MetronomeHit,
        ("mimic", "onHit") => MoveCbOp::MimicHit,
        ("mirrormove", "onHit") => MoveCbOp::MirrorMoveHit,
        ("moonlight", "onHit") | ("morningsun", "onHit") | ("synthesis", "onHit") => {
            MoveCbOp::MoonlightHit
        }
        ("painsplit", "onHit") => MoveCbOp::PainSplitHit,
        ("payday", "onHit") => MoveCbOp::PayDayHit,
        ("psychup", "onHit") => MoveCbOp::PsychUpHit,
        ("rest", "onHit") => MoveCbOp::RestHit,
        ("sketch", "onHit") => MoveCbOp::SketchHit,
        ("sleeptalk", "onHit") => MoveCbOp::SleepTalkHit,
        ("spite", "onHit") => MoveCbOp::SpiteHit,
        ("transform", "onHit") => MoveCbOp::TransformHit,
        ("triattack", "onHit") => MoveCbOp::TriAttackHit,
        ("batonpass", "onHit") => MoveCbOp::BatonPassHit,
        ("furycutter", "onHit") => MoveCbOp::FuryCutterHit,
        ("curse", "onHit") => MoveCbOp::CurseHit,
        ("thief", "secondary.onHit") => MoveCbOp::ThiefSecondaryOnHit,
        ("triattack", "secondary.onHit") => MoveCbOp::TriAttackSecondaryOnHit,
        ("rapidspin", "self.onHit") => MoveCbOp::RapidSpinSelfOnHit,
        ("batonpass", "self.onHit") => MoveCbOp::BatonPassSelfOnHit,
        ("rapidspin", "onAfterHit") => MoveCbOp::RapidSpinAfterHit,
        ("rollout", "onAfterMove") => MoveCbOp::RolloutAfterMove,
        _ => MoveCbOp::Unported,
    }
}
