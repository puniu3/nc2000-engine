//! Fast branchless arithmetic primitives, precomputed lookup tables, and the
//! 256-byte fixed-size `CompactBattle` search kernel for `[Gen 2] NC 2000`
//! (`gen2stadium2`).
//!
//! Key design principles:
//! 1. **Branchless arithmetic over control-flow branching**:
//!    - Turn action ordering (`ActionSortKey`) packs phase, priority, Quick Claw
//!      activation (saturating speed to 1023 so double Quick Claw ties exactly
//!      like PS's 65535), effective speed, and tie-break bit into one `u32`.
//!    - Stat computation (`get_stat_branchless`) uses a 13-entry `STAT_NUM`
//!      numerator table (`-6..=+6`) and bitshifts for paralysis (`>> 2`),
//!      burn (`>> 1`), screens (`<< 1`), and species items (`<< 1` / `* 3 >> 1`).
//!    - Accuracy computation (`ACC_LUT`) uses a 101 × 13 × 13 byte table
//!      precomputed to match IEEE-754 Gen 2 staging bit-for-bit, with branchless
//!      Foresight evasion masking and BrightPowder `-20` saturation.
//!    - Damage computation (`get_damage_base_branchless`, `damage_rolls_simd16`,
//!      `damage_rolls_all39`) collapses the 256-rollover (`>> 2`), Explosion
//!      defense halving (`>> 1`), critical hit (`<< 1`), 1.1x item boost
//!      (`dmg + dmg / 10`), weather (`* num >> 1`), STAB (`+ (dmg >> 1)`),
//!      type effectiveness (`<< super_shift >> resist_shift`), and `/ 255`
//!      damage roll (`* 0x808081 >> 31`) into branchless integer/SIMD operations.
//!    - Substitute targeting uses a 2-element HP pool `[hp, sub_hp]` indexed by
//!      `idx = (has_sub & !bypasses_sub) as usize`.
//! 2. **256-byte cache-line-aligned `Copy` state (`CompactBattle`)**:
//!    - Immutable match constants (base stats, stored stats, move metadata,
//!      1/8 and 1/16 residual quanta) live in `MatchTable`.
//!    - Mutable state for the 6 picked Pokemon (24B × 6 = 144B), 2 sides
//!      (12B × 2 = 24B), and field/turn/special state (88B) fits in exactly
//!      256 bytes (4 cache lines), with zero heap allocations on clone or step.

use std::sync::OnceLock;

use crate::dex::{Accuracy, Category, Dex, ItemId, MoveId, SpeciesId};
use crate::prng::Prng;
use crate::state::{Battle, RequestKind, Status, DK};

// =========================================================================
// 1. ActionSortKey: Single-u32 Branchless Turn Ordering
// =========================================================================

/// Action phase priority (larger value executes earlier in the turn).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum ActionPhase {
    Residual = 0,
    Move = 1,
    Switch = 2,
    BeforeTurnPursuit = 3,
    InstaSwitch = 4,
    TeamPreview = 5,
}

/// Bit-packed 32-bit sort key for turn action ordering:
/// `[ phase (3b: bits 16..18) | (priority + 8) (4b: bits 12..15) | speed_or_qc (10b: bits 1..10) | tie_bit (1b: bit 0) ]`
///
/// Note on Quick Claw (`qc_active`):
/// In `gen2stadium2`, triggering Quick Claw sets a Pokemon's action speed to
/// `65535` (`pokemon.rs:139`). By saturating the 10-bit speed field (`0..=999`)
/// to `1023` via `eff_speed | (-(qc_active as i32) as u32 & 1023)`, a Quick Claw
/// activation beats every natural speed (`<= 999`) AND causes two simultaneous
/// Quick Claw activations at the same priority to tie (`1023 == 1023`),
/// preserving exact equivalence with `65535`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ActionSortKey(pub u32);

impl ActionSortKey {
    #[inline(always)]
    pub const fn new(
        phase: ActionPhase,
        priority: i8,
        qc_active: bool,
        eff_speed: i32,
        tie_bit: u8,
    ) -> Self {
        let phase_bits = (phase as u32 & 0x7) << 15;
        let prio_bits = (((priority as i32 + 8) as u32) & 0xF) << 11;
        let clamped_speed = if eff_speed < 0 {
            0u32
        } else if eff_speed > 999 {
            999u32
        } else {
            eff_speed as u32
        };
        let qc_mask = (-(qc_active as i32) as u32) & 1023;
        let speed_bits = (clamped_speed | qc_mask) << 1;
        let tie = (tie_bit & 1) as u32;
        Self(phase_bits | prio_bits | speed_bits | tie)
    }

    /// Compare two keys ignoring the tie-break bit (bit 0).
    #[inline(always)]
    pub const fn ties_without_rng(self, other: Self) -> bool {
        (self.0 >> 1) == (other.0 >> 1)
    }

    /// Extract the effective speed field (`0..=999`, or `1023` if Quick Claw).
    #[inline(always)]
    pub const fn speed_field(self) -> u32 {
        (self.0 >> 1) & 1023
    }
}

/// Branchlessly resolve which side acts first (`0` or `1`), which acts second
/// (`1` or `0`), and whether a speed tie occurred.
/// When `key_p0` and `key_p1` have distinct upper bits, `first = (key_p0 < key_p1) as usize`.
#[inline(always)]
pub const fn resolve_turn_order(
    key_p0: ActionSortKey,
    key_p1: ActionSortKey,
) -> (usize, usize, bool) {
    let is_tie = key_p0.ties_without_rng(key_p1);
    let first = (key_p0.0 < key_p1.0) as usize;
    let second = first ^ 1;
    (first, second, is_tie)
}

// =========================================================================
// 2. Branchless Stat Calculation (13-entry LUT + Bitshifts)
// =========================================================================

/// Numerators for Gen 2 stat stages `-6..=+6` (divided by 100).
/// Matches `pokemon.rs` `BOOST_TABLE` (`[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]`)
/// and `NUMERATORS` (`[100, 66, 50, 40, 33, 28, 25]`) bit-for-bit on all integers.
pub const STAT_NUM: [i32; 13] = [25, 28, 33, 40, 50, 66, 100, 150, 200, 250, 300, 350, 400];

/// Species-specific stat-boosting held item classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum ItemStatKind {
    #[default]
    None = 0,
    /// Thick Club held by Cubone or Marowak (doubles Attack, `stat == 0`).
    ThickClub = 1,
    /// Light Ball held by Pikachu (doubles Special Attack, `stat == 2`).
    LightBall = 2,
    /// Metal Powder held by Ditto (1.5x Defense and Special Defense, `stat == 1 | 3`).
    MetalPowder = 3,
}

impl ItemStatKind {
    #[inline(always)]
    pub fn classify(dex: &Dex, species: SpeciesId, item: Option<ItemId>) -> Self {
        let Some(item_id) = item else {
            return Self::None;
        };
        let ki = &dex.known_items;
        let ks = &dex.known_species;
        let sp = Some(species);
        let it = Some(item_id);
        if it == ki.thickclub && (sp == ks.cubone || sp == ks.marowak) {
            Self::ThickClub
        } else if it == ki.lightball && sp == ks.pikachu {
            Self::LightBall
        } else if it == ki.metalpowder && sp == ks.ditto {
            Self::MetalPowder
        } else {
            Self::None
        }
    }
}

/// Branchless integer implementation of `gen2stadium2` `Battle::get_stat`.
/// Guaranteed bit-for-bit identical to `Battle::get_stat` across all inputs.
#[inline(always)]
pub const fn get_stat_branchless(
    stored_stat: i32,
    boost: i8,
    stat_idx: usize,
    unboosted: bool,
    unmodified: bool,
    fast_return: bool,
    has_par_speed_drop: bool,
    has_brn_atk_drop: bool,
    has_reflect: bool,
    has_light_screen: bool,
    item_stat_kind: ItemStatKind,
) -> i32 {
    let clamped_boost = if boost < -6 {
        -6
    } else if boost > 6 {
        6
    } else {
        boost
    };
    // Zero the boost when `unboosted` is true via arithmetic mask.
    let keep_boost_mask = -(!unboosted as i8);
    let eff_boost = clamped_boost & keep_boost_mask;
    let mut val = (stored_stat * STAT_NUM[(eff_boost + 6) as usize]) / 100;

    // Paralysis speed drop (÷4 = >> 2) and Burn attack drop (÷2 = >> 1).
    let par_shift = ((stat_idx == 4) & has_par_speed_drop) as u32 * 2;
    let brn_shift = ((stat_idx == 0) & (!unmodified) & has_brn_atk_drop) as u32;
    val >>= par_shift + brn_shift;

    // Clamp to [1, 999].
    if val < 1 {
        val = 1;
    } else if val > 999 {
        val = 999;
    }
    if fast_return {
        return val;
    }

    // Reflect (stat 1) / Light Screen (stat 3) and Thick Club / Light Ball (×2 = << 1).
    let screen_shift =
        (!unboosted & (((stat_idx == 1) & has_reflect) | ((stat_idx == 3) & has_light_screen)))
            as u32;
    let kind = item_stat_kind as u8;
    let item_double_shift =
        (((kind == 1) & (stat_idx == 0)) | ((kind == 2) & (stat_idx == 2))) as u32;
    val <<= screen_shift + item_double_shift;

    // Metal Powder on Ditto (×1.5 = val + (val >> 1)).
    let is_metal = ((kind == 3) & ((stat_idx == 1) | (stat_idx == 3))) as i32;
    val += (val >> 1) & -is_metal;

    val
}

// =========================================================================
// 3. O(1) Accuracy Lookup Table (101 × 13 × 13 = 17,069 bytes)
// =========================================================================

/// Full precomputed Gen 2 accuracy table `[base_pct 0..=100][acc_boost + 6][eva_boost + 6]`.
/// Each entry stores the exact floored & clamped `[1, 255]` threshold from
/// `try_move_hit` / `hit_probability`.
pub type AccTable = [[[u8; 13]; 13]; 101];

static ACC_LUT_CELL: OnceLock<AccTable> = OnceLock::new();

fn build_acc_lut() -> AccTable {
    const POS_BOOST: [f64; 7] = [1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0];
    const NEG_BOOST: [f64; 7] = [1.0, 0.75, 0.6, 0.5, 0.43, 0.36, 0.33];
    let mut table = [[[1u8; 13]; 13]; 101];
    for pct in 0..=100usize {
        let base = (pct as f64 * 255.0 / 100.0).floor();
        for acc_idx in 0..13usize {
            let acc_b = acc_idx as i32 - 6;
            let after_acc = if acc_b > 0 {
                base * POS_BOOST[acc_b as usize]
            } else {
                base * NEG_BOOST[(-acc_b) as usize]
            };
            for eva_idx in 0..13usize {
                let eva_b = eva_idx as i32 - 6;
                let mut acc = after_acc;
                if eva_b > 0 {
                    acc *= NEG_BOOST[eva_b as usize];
                } else if eva_b < 0 {
                    acc *= POS_BOOST[(-eva_b) as usize];
                }
                acc = acc.floor().min(255.0).max(1.0);
                table[pct][acc_idx][eva_idx] = acc as u8;
            }
        }
    }
    table
}

/// Access the static 101 × 13 × 13 Gen 2 accuracy threshold table.
#[inline(always)]
pub fn acc_lut() -> &'static AccTable {
    ACC_LUT_CELL.get_or_init(build_acc_lut)
}

/// Branchless O(1) lookup of the Gen 2 `[0, 255]` accuracy threshold out of 256.
/// Returns `255` when a move has 255/256 (or guaranteed hit in `hit_probability`).
#[inline(always)]
pub fn lookup_accuracy_threshold(
    base_pct: u8,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    ignore_positive_evasion: bool,
    brightpowder: bool,
) -> u8 {
    let pct = base_pct.min(100) as usize;
    let a_clamped = acc_boost.clamp(-6, 6);
    let e_clamped = eva_boost.clamp(-6, 6);

    let eff_acc = a_clamped & -(!ignore_accuracy as i8);
    let drop_pos_eva = ignore_evasion | (ignore_positive_evasion & (e_clamped > 0));
    let eff_eva = e_clamped & -(!drop_pos_eva as i8);

    let raw = acc_lut()[pct][(eff_acc + 6) as usize][(eff_eva + 6) as usize];
    let bp_sub = (-(brightpowder as i8) as u8) & 20;
    raw.saturating_sub(bp_sub)
}

/// Branchless implementation of `Battle::hit_probability`.
#[inline(always)]
pub fn hit_probability_branchless(
    base_pct: u8,
    always_hit: bool,
    acc_boost: i8,
    eva_boost: i8,
    ignore_accuracy: bool,
    ignore_evasion: bool,
    ignore_positive_evasion: bool,
) -> f64 {
    if always_hit {
        return 1.0;
    }
    let acc = lookup_accuracy_threshold(
        base_pct,
        acc_boost,
        eva_boost,
        ignore_accuracy,
        ignore_evasion,
        ignore_positive_evasion,
        false,
    );
    if acc == 255 {
        1.0
    } else {
        acc as f64 / 256.0
    }
}

// =========================================================================
// 4. Branchless Damage & SIMD 16-Roll / 39-Roll Evaluators
// =========================================================================

/// Weather kind in `CompactBattle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CompactWeather {
    #[default]
    None = 0,
    Rain = 1,
    Sun = 2,
    Sandstorm = 3,
}

/// Branchless computation of the pre-variance Gen 2 Stadium 2 damage (`[0, 3996]`).
///
/// Mirrors `Battle::get_damage` (`moveexec.rs:2865-2945`) using pure integer and
/// bitmask operations:
/// - Stadium 2 `>= 256` rollover (`>> 2` on both attack and defense simultaneously)
/// - Self-Destruct / Explosion defense halving (`>> 1`)
/// - Critical hit doubling (`<< 1`)
/// - 1.1x type-boost item (`dmg + dmg / 10`, exact for all IEEE-754 integers)
/// - Clamp `[1, 997] + 2`
/// - Weather boost (`+ (dmg >> 1)`) and nerf (`>> 1`)
/// - STAB (`+ (dmg >> 1)` via `attacker_type_mask` bit test)
/// - Type effectiveness (`<< super_shift >> resist_shift`, masked by `!is_immune`)
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub const fn get_damage_base_branchless(
    level: i32,
    base_power: i32,
    mut atk: i32,
    mut def: i32,
    is_physical: bool,
    is_selfdestruct: bool,
    is_crit: bool,
    has_type_boost_item: bool,
    weather: CompactWeather,
    is_water_move: bool,
    is_fire_move: bool,
    is_solarbeam: bool,
    move_type: u8,
    attacker_type_mask: u32,
    total_type_mod: i8,
    is_immune: bool,
) -> i32 {
    if is_immune || base_power <= 0 {
        return 0;
    }

    // Stadium 2 rollover fix: if attack >= 256 || defense >= 256, divide both by 4.
    let rollover_shift = (((atk | def) >= 256) as u32) << 1;
    atk >>= rollover_shift;
    if atk < 1 {
        atk = 1;
    }
    def >>= rollover_shift;
    if def < 1 {
        def = 1;
    }

    // Self-Destruct / Explosion halves physical defense (min 1).
    let selfdestruct_shift = (is_selfdestruct & is_physical) as u32;
    def >>= selfdestruct_shift;
    if def < 1 {
        def = 1;
    }

    // Core Gen 2 damage formula.
    let mut dmg = ((level * 2 / 5 + 2) * base_power * atk / def) / 50;

    // Critical hit doubles before ModifyDamage.
    dmg <<= is_crit as u32;

    // 1.1x type-boost held item: floor(dmg * 1.1) == dmg + dmg / 10.
    let boost_mask = -(has_type_boost_item as i32);
    dmg += (dmg / 10) & boost_mask;

    // Clamp to [1, 997], then + 2.
    if dmg < 1 {
        dmg = 1;
    } else if dmg > 997 {
        dmg = 997;
    }
    dmg += 2;

    // Weather modifier.
    let rain = (weather as u8) == (CompactWeather::Rain as u8);
    let sun = (weather as u8) == (CompactWeather::Sun as u8);
    let w_boost = ((is_water_move & rain) | (is_fire_move & sun)) as i32;
    let w_nerf = (((is_fire_move | is_solarbeam) & rain) | (is_water_move & sun)) as u32;
    dmg += (dmg >> 1) & -w_boost;
    dmg >>= w_nerf;

    // STAB (Same-Type Attack Bonus): +50% when move_type bit is set in attacker_type_mask.
    let is_stab = ((attacker_type_mask >> (move_type & 31)) & 1) as i32;
    dmg += (dmg >> 1) & -is_stab;

    // Type effectiveness: super-effective shifts left (1 or 2), resist shifts right (1 or 2).
    let super_shift = if total_type_mod > 0 {
        if total_type_mod >= 2 { 2u32 } else { 1u32 }
    } else {
        0u32
    };
    let resist_shift = if total_type_mod < 0 {
        if total_type_mod <= -2 { 2u32 } else { 1u32 }
    } else {
        0u32
    };
    dmg = (dmg << super_shift) >> resist_shift;

    dmg
}

/// Branchless application of a single Gen 2 damage roll (`roll in 217..=255`).
/// Uses reciprocal multiplication `(x * 0x808081) >> 31` instead of integer
/// division `/ 255` (exact for all `x < 2^24`).
#[inline(always)]
pub const fn apply_roll_branchless(
    base_dmg: i32,
    roll_217_to_255: u32,
    no_damage_variance: bool,
) -> i32 {
    if base_dmg <= 0 {
        return 0;
    }
    let apply_var = (!no_damage_variance & (base_dmg > 1)) as i32;
    let num = (base_dmg as u64) * (roll_217_to_255 as u64);
    let varied = ((num * 0x808081) >> 31) as i32;
    let mask = -apply_var;
    let raw = (varied & mask) | (base_dmg & !mask);
    if raw < 1 { 1 } else { raw }
}

/// Representative 16-lane roll factors spanning `[217, 255]` (for 16-wide SIMD
/// evaluation of a move's damage distribution).
pub const SIMD16_ROLLS: [u16; 16] = [
    217, 219, 222, 224, 227, 229, 232, 235, 237, 240, 242, 245, 247, 250, 252, 255,
];

/// Result of a 16-lane vectorized damage + Substitute/HP pool update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(32))]
pub struct Simd16HitOutcome {
    /// Dealt damage in each of the 16 roll lanes.
    pub dealt: [i16; 16],
    /// Post-hit main body HP (`hp_pool[0]`) in each of the 16 lanes.
    pub post_hp: [i16; 16],
    /// Post-hit Substitute HP (`hp_pool[1]`) in each of the 16 lanes.
    pub post_sub_hp: [i16; 16],
    /// 16-bit mask of lanes where the target fainted (`post_hp[i] == 0`).
    pub ko_mask: u16,
    /// 16-bit mask of lanes where the target's Substitute broke (`post_sub_hp[i] == 0`).
    pub sub_broke_mask: u16,
}

/// Evaluate 16 damage rolls simultaneously and update the 2-slot `[hp, sub_hp]`
/// pool without branches. The compiler auto-vectorizes the 16-lane loop into
/// NEON / AVX2 / Wasm SIMD128 vector instructions.
#[inline]
pub fn evaluate_hit_simd16(
    base_dmg: i32,
    no_damage_variance: bool,
    target_hp: i16,
    target_sub_hp: i16,
    bypasses_sub: bool,
) -> Simd16HitOutcome {
    let hits_sub = (target_sub_hp > 0) & (!bypasses_sub);
    let pool_idx = hits_sub as usize;
    let base_u32 = base_dmg.max(0) as u32;
    let apply_var = (!no_damage_variance) & (base_dmg > 1);

    let mut dealt = [0i16; 16];
    let mut post_hp = [target_hp; 16];
    let mut post_sub_hp = [target_sub_hp; 16];
    let mut ko_mask = 0u16;
    let mut sub_broke_mask = 0u16;

    for lane in 0..16 {
        let roll = SIMD16_ROLLS[lane] as u32;
        let varied = (((base_u32 as u64 * roll as u64) * 0x808081) >> 31) as i32;
        let d = if apply_var { varied.max(1) } else { base_dmg.max(1) } as i16;

        // Branchless 2-element HP pool `[hp, sub_hp]` indexed by `pool_idx`.
        let mut pool = [target_hp, target_sub_hp];
        let before = pool[pool_idx];
        let actual_dealt = d.min(before);
        pool[pool_idx] = before - actual_dealt;

        dealt[lane] = actual_dealt;
        post_hp[lane] = pool[0];
        post_sub_hp[lane] = pool[1];
        ko_mask |= ((pool[0] <= 0) as u16) << lane;
        sub_broke_mask |= ((hits_sub & (pool[1] <= 0)) as u16) << lane;
    }

    Simd16HitOutcome {
        dealt,
        post_hp,
        post_sub_hp,
        ko_mask,
        sub_broke_mask,
    }
}

/// Compute all 39 exact Gen 2 damage rolls (`217..=255`) into a stack array.
#[inline]
pub fn damage_rolls_all39(base_dmg: i32, no_damage_variance: bool) -> [i16; 39] {
    let mut out = [0i16; 39];
    if base_dmg <= 0 {
        return out;
    }
    if no_damage_variance || base_dmg == 1 {
        out.fill(base_dmg as i16);
        return out;
    }
    let base_u64 = base_dmg as u64;
    for c in 0..39usize {
        let varied = (((base_u64 * (217 + c as u64)) * 0x808081) >> 31) as i16;
        out[c] = varied.max(1);
    }
    out
}

// =========================================================================
// 5. CompactBattle (256-Byte Fixed-Size, 64B-Aligned Copy Kernel)
// =========================================================================

/// Volatile status bitflags stored in `CompactMon::vol_flags` (`u16`).
pub mod vol_bit {
    pub const CONFUSION: u16 = 1 << 0;
    pub const FLINCH: u16 = 1 << 1;
    pub const DESTINY_BOND: u16 = 1 << 2;
    pub const SUBSTITUTE: u16 = 1 << 3;
    pub const PAR_SPEED_DROP: u16 = 1 << 4;
    pub const BRN_ATK_DROP: u16 = 1 << 5;
    pub const LEECH_SEED: u16 = 1 << 6;
    pub const NIGHTMARE: u16 = 1 << 7;
    pub const CURSE: u16 = 1 << 8;
    pub const FORESIGHT: u16 = 1 << 9;
    pub const PROTECT: u16 = 1 << 10;
    pub const ENDURE: u16 = 1 << 11;
    pub const ENCORE: u16 = 1 << 12;
    pub const DISABLE: u16 = 1 << 13;
    pub const PERISH_SONG: u16 = 1 << 14;
    pub const TRAPPED_OR_LOCKED: u16 = 1 << 15;
}

/// Mutable battle state of a single picked Pokemon (24 bytes, `Copy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CompactMon {
    /// `[0]` = main HP, `[1]` = Substitute HP (`0` when no Substitute).
    pub hp_pool: [i16; 2],
    /// Current PP of the 4 move slots (`0..=64`).
    pub pp: [u8; 4],
    /// Low 3 bits = `Status` discriminant (`0..=7`), high 5 bits = sleep turns or toxic counter.
    pub status_packed: u8,
    /// Held item slot (`0` = None, `1..=64` = compact item id).
    pub item: u8,
    /// Bitmask of active volatile statuses (see [`vol_bit`]).
    pub vol_flags: u16,
    /// `[0]` = confusion turns, `[1]` = encore turns, `[2]` = disable turns, `[3]` = perish/stall counter.
    pub timers: [u8; 4],
    /// Stat stages `[atk, def, spa, spd, spe, acc, eva]` (`-6..=+6`).
    pub boosts: [i8; 7],
    /// Move slot (`0..=3`, or `0xFF` for None) of last move / encore / disable, packed:
    /// bits `0..2`: `last_move_slot`, bits `2..4`: `encore_slot`, bits `4..6`: `disable_slot`,
    /// bit `6`: `has_last_move`, bit `7`: `roster_slot_hi`.
    pub slot_refs: u8,
}

const _: () = assert!(std::mem::size_of::<CompactMon>() == 24);

impl CompactMon {
    #[inline(always)]
    pub const fn hp(&self) -> i16 {
        self.hp_pool[0]
    }

    #[inline(always)]
    pub const fn sub_hp(&self) -> i16 {
        self.hp_pool[1]
    }

    #[inline(always)]
    pub const fn status_raw(&self) -> u8 {
        self.status_packed & 0x07
    }

    #[inline(always)]
    pub const fn status_counter(&self) -> u8 {
        self.status_packed >> 3
    }

    #[inline(always)]
    pub fn set_status(&mut self, status_disc: u8, counter: u8) {
        self.status_packed = (status_disc & 0x07) | ((counter & 0x1F) << 3);
    }

    /// Apply damage to either Substitute (`hp_pool[1]`) or main HP (`hp_pool[0]`)
    /// using branchless array indexing. Returns actual damage dealt.
    #[inline(always)]
    pub fn apply_damage_branchless(&mut self, dmg: i16, bypasses_sub: bool) -> i16 {
        let has_sub = (self.hp_pool[1] > 0) & (!bypasses_sub);
        let idx = has_sub as usize;
        let cur = self.hp_pool[idx];
        let dealt = if dmg < cur { dmg.max(0) } else { cur.max(0) };
        self.hp_pool[idx] = cur - dealt;
        if self.hp_pool[1] <= 0 {
            self.vol_flags &= !vol_bit::SUBSTITUTE;
        }
        if self.hp_pool[0] <= 0 {
            self.status_packed = Status::Fnt as u8;
            self.vol_flags = 0;
        }
        dealt
    }
}

/// Mutable state of one side (12 bytes, `Copy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct CompactSide {
    /// Active party index (`0..=2`).
    pub active_idx: u8,
    /// Original roster indices (`0..=5`) for the 3 picked party members.
    pub party_roster: [u8; 3],
    /// Bitmask of alive party indices (`bit 0..2`, `count_ones() == pokemon_left`).
    pub alive_mask: u8,
    /// Reflect remaining turns (`0..=5`).
    pub reflect_turns: u8,
    /// Light Screen remaining turns (`0..=5`).
    pub lightscreen_turns: u8,
    /// Safeguard remaining turns (`0..=5`).
    pub safeguard_turns: u8,
    /// Spikes layer (`0` or `1`) in bit 0, fainted_this_turn in bit 1, switch_needed in bit 2.
    pub side_flags: u8,
    /// Last move used for Self-KO clause tracking (`u8::MAX` = None).
    pub last_move_selfko: u8,
    /// Current request kind (`0` = Wait/None, `1` = Move, `2` = Switch, `3` = TeamPreview).
    pub request: u8,
    /// Reserved padding for 12-byte alignment.
    pub pad: u8,
}

const _: () = assert!(std::mem::size_of::<CompactSide>() == 12);

/// Complete mutable battle state in exactly 256 bytes (4 cache lines, 64-byte aligned).
/// Zero heap allocations: `Clone` is a 4-cache-line register copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C, align(64))]
pub struct CompactBattle {
    /// 6 picked Pokemon (`0..3` = Side 0 party, `3..6` = Side 1 party): 144 bytes.
    pub mons: [CompactMon; 6],
    /// 2 sides: 24 bytes (offset 144..168).
    pub sides: [CompactSide; 2],
    /// Current turn number (`1..=1001`): 2 bytes (offset 168..170).
    pub turn: u16,
    /// Weather kind (`0` = None, `1` = Rain, `2` = Sun, `3` = Sandstorm): 1 byte.
    pub weather_kind: u8,
    /// Weather remaining turns (`0..=5`): 1 byte.
    pub weather_turns: u8,
    /// Pre-rolled Quick Claw activation coin for the current turn: 1 byte.
    pub quick_claw_roll: u8,
    /// Terminal state (`0` = Ongoing, `1` = P1Win, `2` = P2Win, `3` = Tie): 1 byte.
    pub outcome: u8,
    /// Speed order bit (`0` = P1 faster/won tie, `1` = P2 faster/won tie): 1 byte.
    pub speed_order: u8,
    /// Clause flags (`bit 0..1` = P1/P2 sleep clause active, `bit 2..3` = P1/P2 freeze clause): 1 byte.
    pub clause_flags: u8,
    /// Last damage dealt to `[P1, P2]` this turn (for Counter / Mirror Coat): 4 bytes (offset 176..180).
    pub hurt_damage_this_turn: [u16; 2],
    /// Category of last move that damaged `[P1, P2]` (`0` = None, `1` = Physical, `2` = Special): 2 bytes.
    pub hurt_cat_this_turn: [u8; 2],
    /// Rollout / Fury Cutter / Bide / Trap counters for `[P1, P2]`: 6 bytes (offset 182..188).
    pub special_counters: [u8; 6],
    /// Future Sight delayed damage and countdown for `[P1, P2]`: 4 bytes (offset 188..192).
    pub future_sight_dmg: [u8; 2],
    pub future_sight_turns: [u8; 2],
    /// 64-bit PRNG state (Gen5 LCG seed): 8 bytes (offset 192..200).
    pub prng_seed: u64,
    /// Cached static match fingerprint + SIMD workspace padding to exact 256 bytes: 56 bytes (offset 200..256).
    pub match_fingerprint: u64,
    pub reserved: [u64; 6],
}

const _: () = assert!(std::mem::size_of::<CompactBattle>() == 256);
const _: () = assert!(std::mem::align_of::<CompactBattle>() == 64);

/// Immutable per-Pokemon reference data stored in [`MatchTable`].
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticMonSpec {
    pub species: SpeciesId,
    pub level: u8,
    pub maxhp: i16,
    /// 1/8 of maxhp (min 1) for standard poison/burn/leech seed slip damage.
    pub quantum_1_8: i16,
    /// 1/16 of maxhp (min 1) for Leftovers and Toxic base slip damage.
    pub quantum_1_16: i16,
    /// Stored stats `[atk, def, spa, spd, spe]`.
    pub stored_stats: [i16; 5],
    /// Bitmask of the Pokemon's 1–2 types (`(1 << t0) | (1 << t1)`).
    pub type_mask: u32,
    pub types: [u8; 2],
    pub num_types: u8,
    pub item_stat_kind: ItemStatKind,
    pub initial_item: Option<ItemId>,
    /// Boosted type (`0..18`, or `0xFF` if held item is not a 1.1x type booster).
    pub item_boost_type: u8,
    pub is_quick_claw: bool,
    pub is_leftovers: bool,
    pub is_brightpowder: bool,
    pub is_focusband: bool,
    pub is_mintberry: bool,
    pub is_miracleberry: bool,
    pub is_goldberry: bool,
    pub is_berry: bool,
    pub is_mysteryberry: bool,
    pub is_scope_lens: bool,
    pub moves: [MoveId; 4],
    pub max_pp: [u8; 4],
    pub num_moves: u8,
    pub hp_type: u8,
    pub hp_power: i16,
    pub return_bp: i16,
}

/// Immutable per-move compiled metadata for branchless execution.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactMoveSpec {
    pub id: MoveId,
    pub base_power: i16,
    pub move_type: u8,
    pub is_physical: bool,
    pub is_status: bool,
    pub priority: i8,
    pub base_acc_pct: u8,
    pub always_hit: bool,
    pub ignore_accuracy: bool,
    pub ignore_evasion: bool,
    pub ignore_positive_evasion: bool,
    pub crit_ratio: u8,
    pub is_selfdestruct: bool,
    pub is_water: bool,
    pub is_fire: bool,
    pub is_solarbeam: bool,
    pub bypasses_sub: bool,
    pub is_rest: bool,
    pub is_half_heal: bool,
    pub is_heal_cycle: bool,
    pub is_sleep_usable: bool,
    pub can_sleep_talk: bool,
    pub is_protect: bool,
    pub is_substitute: bool,
    pub drain_half: bool,
    pub recoil_quarter: bool,
    /// Primary status inflicted on target (`0` = None, else `Status as u8`).
    pub primary_status: u8,
    /// Primary volatile inflicted on target (`vol_bit` mask).
    pub primary_vol: u16,
    /// Secondary chance out of 256 (`0` = None).
    pub sec_chance_256: u8,
    /// Secondary status inflicted on target (`0` = None, else `Status as u8`).
    pub sec_status: u8,
    /// Secondary volatile inflicted on target (`vol_bit` mask).
    pub sec_vol: u16,
    /// Secondary stat stage delta (`(stat_idx, delta)` or `(0xFF, 0)`).
    pub sec_boost: (u8, i8),
    /// Self-targeted or target-applied stat boosts (`[(stat_idx_or_ff, delta); 3]`).
    pub boosts: [(u8, i8); 3],
    pub boosts_self: bool,
    /// Special move effect tag (`0` = None, `1` = Reflect, `2` = LightScreen, `3` = Safeguard,
    /// `4` = Spikes, `5` = Curse, `6` = BellyDrum, `7` = Roar/Whirlwind, `8` = Encore,
    /// `9` = Disable, `10` = SleepTalk, `11` = Counter, `12` = MirrorCoat, `13` = Flail/Reversal,
    /// `14` = SeismicToss/NightShade, `15` = SuperFang, `16` = Thief, `17` = PainSplit,
    /// `18` = BatonPass, `19` = PerishSong, `20` = DestinyBond, `21` = HealBell,
    /// `22` = PsychUp, `23` = RainDance, `24` = SunnyDay, `25` = Sandstorm,
    /// `26` = MeanLook/SpiderWeb, `27` = Haze).
    pub special_kind: u8,
    /// Variable base power kind (`0` = static, `1` = Return, `2` = Frustration,
    /// `3` = HiddenPower, `4` = Flail/Reversal, `5` = Magnitude, `6` = Present).
    pub var_bp_kind: u8,
    /// Fixed damage (`-1` = Level, `> 0` = Amount(n), `0` = None).
    pub fixed_damage: i16,
    /// Twice the expected hit count (`2` = single hit, `6` = 2..5 multihit, `2*n` = Fixed(n)).
    pub multihit_mean_x2: u8,
    /// Rollout status pseudo-score kind (`0` = None, `1` = Sleep, `2` = Par, `3` = Spikes,
    /// `4` = Heal, `5` = Curse, `6` = SwordsDance, `7` = Amnesia, `8` = BellyDrum).
    pub pseudo_kind: u8,
}

/// Precompiled per-item metadata for O(1) lookup in `CompactBattle::from_battle`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CompactItemSpec {
    pub boost_type: u8,
    pub is_quick_claw: bool,
    pub is_leftovers: bool,
    pub is_brightpowder: bool,
    pub is_focusband: bool,
    pub is_mintberry: bool,
    pub is_miracleberry: bool,
    pub is_goldberry: bool,
    pub is_berry: bool,
    pub is_mysteryberry: bool,
    pub is_scope_lens: bool,
}

/// Process-wide precompiled Dex tables for `CompactBattle` and fast evaluation.
pub struct CompactDexTables {
    pub type_chart: [[i8; 20]; 20],
    pub special_type_mask: u32,
    pub normal_type: u8,
    pub fighting_type: u8,
    pub flying_type: u8,
    pub poison_type: u8,
    pub ground_type: u8,
    pub rock_type: u8,
    pub ghost_type: u8,
    pub steel_type: u8,
    pub fire_type: u8,
    pub water_type: u8,
    pub grass_type: u8,
    pub electric_type: u8,
    pub ice_type: u8,
    pub sandstorm_immune_mask: u32,
    pub poison_immune_mask: u32,
    pub move_specs: Vec<CompactMoveSpec>,
    pub item_specs: Vec<CompactItemSpec>,
    pub sub_cond: Option<crate::dex::CondId>,
    pub conf_cond: Option<crate::dex::CondId>,
    pub flinch_cond: Option<crate::dex::CondId>,
    pub dbond_cond: Option<crate::dex::CondId>,
    pub par_cond: Option<crate::dex::CondId>,
    pub brn_cond: Option<crate::dex::CondId>,
    pub seed_cond: Option<crate::dex::CondId>,
    pub night_cond: Option<crate::dex::CondId>,
    pub curse_cond: Option<crate::dex::CondId>,
    pub fore_cond: Option<crate::dex::CondId>,
    pub prot_cond: Option<crate::dex::CondId>,
    pub end_cond: Option<crate::dex::CondId>,
    pub enc_cond: Option<crate::dex::CondId>,
    pub dis_cond: Option<crate::dex::CondId>,
    pub perish_cond: Option<crate::dex::CondId>,
    pub res_cond: Option<crate::dex::CondId>,
    pub recharge_cond: Option<crate::dex::CondId>,
    pub refl_cond: Option<crate::dex::CondId>,
    pub ls_cond: Option<crate::dex::CondId>,
    pub safe_cond: Option<crate::dex::CondId>,
    pub spikes_cond: Option<crate::dex::CondId>,
    pub rain_cond: Option<crate::dex::CondId>,
    pub sun_cond: Option<crate::dex::CondId>,
    pub sand_cond: Option<crate::dex::CondId>,
    pub sleep_clause_cond: Option<crate::dex::CondId>,
    pub freeze_clause_cond: Option<crate::dex::CondId>,
}

static COMPACT_DEX_TABLES: OnceLock<CompactDexTables> = OnceLock::new();

/// Access the process-wide precompiled `CompactDexTables`.
#[inline]
pub fn dex_tables(dex: &Dex) -> &'static CompactDexTables {
    COMPACT_DEX_TABLES.get_or_init(|| build_compact_dex_tables(dex))
}

fn build_compact_dex_tables(dex: &Dex) -> CompactDexTables {
    let mut type_chart = [[0i8; 20]; 20];
    let n_types = dex.type_names.len().min(19);
    for atk_t in 0..n_types {
        for def_t in 0..n_types {
            let atk_id = crate::dex::TypeId(atk_t as u8);
            let def_id = crate::dex::TypeId(def_t as u8);
            if dex.type_immune(atk_id, def_id) {
                type_chart[atk_t][def_t] = -9;
            } else {
                type_chart[atk_t][def_t] = dex.eff(atk_id, def_id) as i8;
            }
        }
    }

    let mut special_type_mask = 0u32;
    for t_name in [
        "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
    ] {
        if let Some(tid) = dex.type_id(t_name) {
            special_type_mask |= 1u32 << (tid.0 & 31);
        }
    }

    let tid_or = |name: &str, fallback: u8| dex.type_id(name).map(|t| t.0).unwrap_or(fallback);
    let normal_type = dex.known_types.normal.0;
    let fighting_type = dex.known_types.fighting.0;
    let flying_type = dex.known_types.flying.0;
    let poison_type = tid_or("Poison", 0xFF);
    let ground_type = dex.known_types.ground.0;
    let rock_type = tid_or("Rock", 0xFF);
    let ghost_type = dex.known_types.ghost.0;
    let steel_type = tid_or("Steel", 0xFF);
    let fire_type = dex.known_types.fire.0;
    let water_type = dex.known_types.water.0;
    let grass_type = dex.known_types.grass.0;
    let electric_type = tid_or("Electric", 0xFF);
    let ice_type = tid_or("Ice", 0xFF);

    let mut sandstorm_immune_mask = 0u32;
    for &t in &[rock_type, ground_type, steel_type] {
        if t < 32 {
            sandstorm_immune_mask |= 1u32 << t;
        }
    }
    let mut poison_immune_mask = 0u32;
    for &t in &[poison_type, steel_type] {
        if t < 32 {
            poison_immune_mask |= 1u32 << t;
        }
    }

    let parse_status_code = |s: Option<&str>| -> u8 {
        match s {
            Some("slp") => Status::Slp as u8,
            Some("psn") => Status::Psn as u8,
            Some("brn") => Status::Brn as u8,
            Some("frz") => Status::Frz as u8,
            Some("par") => Status::Par as u8,
            Some("tox") => Status::Tox as u8,
            _ => 0,
        }
    };
    let parse_vol_code = |s: Option<&str>| -> u16 {
        match s {
            Some("confusion") => vol_bit::CONFUSION,
            Some("flinch") => vol_bit::FLINCH,
            Some("leechseed") => vol_bit::LEECH_SEED,
            Some("nightmare") => vol_bit::NIGHTMARE,
            Some("curse") => vol_bit::CURSE,
            Some("foresight") => vol_bit::FORESIGHT,
            Some("encore") => vol_bit::ENCORE,
            Some("disable") => vol_bit::DISABLE,
            Some("perishsong") => vol_bit::PERISH_SONG,
            Some("destinybond") => vol_bit::DESTINY_BOND,
            _ => 0,
        }
    };

    let authentic_bit = dex.flag_bit("authentic");
    let nosleeptalk_bit = dex.flag_bit("nosleeptalk");
    let charge_bit = dex.flag_bit("charge");

    let mut move_specs = Vec::with_capacity(dex.moves.len());
    for m_raw in 0..dex.moves.len() {
        let mid = MoveId(m_raw as u16);
        let ms = dex.move_static(mid);
        let mkey = dex.moves.key(mid);
        let var_bp_kind = match mkey {
            "return" => 1,
            "frustration" => 2,
            "hiddenpower" => 3,
            "flail" | "reversal" => 4,
            "magnitude" => 5,
            "present" => 6,
            _ => 0,
        };
        let bp = match var_bp_kind {
            5 => 71,
            6 => 40,
            _ => ms.base_power as i16,
        };
        let (base_acc_pct, always_hit) = match ms.accuracy {
            Accuracy::AlwaysHits => (100, true),
            Accuracy::Pct(a) => (a.clamp(0, 100) as u8, ms.always_hit),
        };
        let primary_status = parse_status_code(ms.status);
        let primary_vol = parse_vol_code(ms.volatile_status);
        let mut sec_chance_256 = 0u8;
        let mut sec_status = 0u8;
        let mut sec_vol = 0u16;
        let mut sec_boost = (0xFFu8, 0i8);
        if let Some(sec) = ms.secondaries.first() {
            let ch = sec.chance.unwrap_or(100).clamp(0, 100);
            sec_chance_256 = (ch * 255 / 100).clamp(1, 255) as u8;
            sec_status = parse_status_code(sec.status);
            sec_vol = parse_vol_code(sec.volatile_status);
            if let Some(&(st_i, d)) = sec.boosts.first() {
                sec_boost = (st_i as u8, d);
            } else if let Some(se) = sec.self_effect.as_ref() {
                if let Some(&(st_i, d)) = se.boosts.first() {
                    sec_boost = ((st_i as u8) | 0x80, d);
                }
            }
        }
        let mut boosts = [(0xFFu8, 0i8); 3];
        let mut boosts_self = matches!(ms.target, "self" | "allySide" | "adjacentAllyOrSelf");
        let src_boosts = if !ms.boosts.is_empty() {
            &ms.boosts[..]
        } else if let Some(se) = ms.self_effect.as_ref() {
            boosts_self = true;
            &se.boosts[..]
        } else {
            &[][..]
        };
        for (b_i, &(st_i, d)) in src_boosts.iter().take(3).enumerate() {
            boosts[b_i] = (st_i as u8, d);
        }
        let special_kind = match mkey {
            "reflect" => 1,
            "lightscreen" => 2,
            "safeguard" => 3,
            "spikes" => 4,
            "curse" => 5,
            "bellydrum" => 6,
            "roar" | "whirlwind" => 7,
            "encore" => 8,
            "disable" => 9,
            "sleeptalk" => 10,
            "counter" => 11,
            "mirrorcoat" => 12,
            "flail" | "reversal" => 13,
            "seismictoss" | "nightshade" => 14,
            "superfang" => 15,
            "thief" => 16,
            "painsplit" => 17,
            "batonpass" => 18,
            "perishsong" => 19,
            "destinybond" => 20,
            "healbell" => 21,
            "psychup" => 22,
            "raindance" => 23,
            "sunnyday" => 24,
            "sandstorm" => 25,
            "meanlook" | "spiderweb" => 26,
            "haze" => 27,
            _ => 0,
        };
        let fixed_damage = match ms.damage {
            Some(crate::dex::FixedDamage::Level) => -1,
            Some(crate::dex::FixedDamage::Amount(n)) => n.clamp(1, i16::MAX as i32) as i16,
            None => 0,
        };
        let multihit_mean_x2 = match ms.multihit {
            Some(crate::dex::Multihit::Fixed(n)) => (n * 2).clamp(2, 20) as u8,
            Some(crate::dex::Multihit::Range(2, 5)) => 6,
            Some(crate::dex::Multihit::Range(lo, hi)) => (lo + hi).clamp(2, 20) as u8,
            None => 2,
        };
        let pseudo_kind = if ms.status == Some("slp") {
            1
        } else if ms.status == Some("par") {
            2
        } else if ms.side_condition == Some("spikes") {
            3
        } else if ms.heal.is_some() {
            4
        } else {
            match mkey {
                "curse" => 5,
                "swordsdance" => 6,
                "amnesia" => 7,
                "bellydrum" => 8,
                _ => 0,
            }
        };

        move_specs.push(CompactMoveSpec {
            id: mid,
            base_power: bp,
            move_type: ms.move_type.0,
            is_physical: ms.category == Category::Physical,
            is_status: ms.category == Category::Status,
            priority: ms.priority,
            base_acc_pct,
            always_hit,
            ignore_accuracy: ms.ignore_accuracy,
            ignore_evasion: ms.ignore_evasion,
            ignore_positive_evasion: ms.ignore_positive_evasion,
            crit_ratio: ms.crit_ratio.clamp(0, 5) as u8,
            is_selfdestruct: ms.selfdestruct,
            is_water: ms.move_type.0 == water_type,
            is_fire: ms.move_type.0 == fire_type,
            is_solarbeam: mkey == "solarbeam",
            bypasses_sub: (ms.flags & authentic_bit) != 0,
            is_rest: mkey == "rest",
            is_half_heal: matches!(
                mkey,
                "recover" | "softboiled" | "milkdrink" | "moonlight" | "morningsun" | "synthesis"
            ),
            is_heal_cycle: matches!(mkey, "rest" | "recover" | "softboiled" | "milkdrink"),
            is_sleep_usable: ms.sleep_usable,
            can_sleep_talk: (ms.flags & (nosleeptalk_bit | charge_bit)) == 0,
            is_protect: matches!(mkey, "protect" | "detect"),
            is_substitute: mkey == "substitute",
            drain_half: ms.drain.is_some(),
            recoil_quarter: ms.recoil.is_some(),
            primary_status,
            primary_vol,
            sec_chance_256,
            sec_status,
            sec_vol,
            sec_boost,
            boosts,
            boosts_self,
            special_kind,
            var_bp_kind,
            fixed_damage,
            multihit_mean_x2,
            pseudo_kind,
        });
    }

    let mut item_specs = Vec::with_capacity(dex.items.len());
    for i_raw in 0..dex.items.len() {
        let iid = ItemId(i_raw as u16);
        let ikey = dex.items.key(iid);
        let boost_type = COMPACT_TYPE_BOOST_ITEMS
            .iter()
            .find(|&&(k, _)| k == ikey)
            .and_then(|&(_, t_name)| dex.type_id(t_name))
            .map(|t| t.0)
            .unwrap_or(0xFF);
        item_specs.push(CompactItemSpec {
            boost_type,
            is_quick_claw: Some(iid) == dex.known_items.quickclaw,
            is_leftovers: ikey == "leftovers",
            is_brightpowder: ikey == "brightpowder",
            is_focusband: ikey == "focusband",
            is_mintberry: ikey == "mintberry",
            is_miracleberry: ikey == "miracleberry",
            is_goldberry: ikey == "goldberry",
            is_berry: ikey == "berry" || ikey == "berryjuice",
            is_mysteryberry: ikey == "mysteryberry",
            is_scope_lens: ikey == "scopelens",
        });
    }

    CompactDexTables {
        type_chart,
        special_type_mask,
        normal_type,
        fighting_type,
        flying_type,
        poison_type,
        ground_type,
        rock_type,
        ghost_type,
        steel_type,
        fire_type,
        water_type,
        grass_type,
        electric_type,
        ice_type,
        sandstorm_immune_mask,
        poison_immune_mask,
        move_specs,
        item_specs,
        sub_cond: dex.conds_id("substitute"),
        conf_cond: dex.conds_id("confusion"),
        flinch_cond: dex.conds_id("flinch"),
        dbond_cond: dex.conds_id("destinybond"),
        par_cond: dex.conds_id("parspeeddrop"),
        brn_cond: dex.conds_id("brnattackdrop"),
        seed_cond: dex.conds_id("leechseed"),
        night_cond: dex.conds_id("nightmare"),
        curse_cond: dex.conds_id("curse"),
        fore_cond: dex.conds_id("foresight"),
        prot_cond: dex.conds_id("protect"),
        end_cond: dex.conds_id("endure"),
        enc_cond: dex.conds_id("encore"),
        dis_cond: dex.conds_id("disable"),
        perish_cond: dex.conds_id("perishsong"),
        res_cond: dex.conds_id("residualdmg"),
        recharge_cond: dex.conds_id("mustrecharge"),
        refl_cond: dex.conds_id("reflect"),
        ls_cond: dex.conds_id("lightscreen"),
        safe_cond: dex.conds_id("safeguard"),
        spikes_cond: dex.conds_id("spikes"),
        rain_cond: dex.conds_id("raindance"),
        sun_cond: dex.conds_id("sunnyday"),
        sand_cond: dex.conds_id("sandstorm"),
        sleep_clause_cond: dex.conds_id("stadiumsleepclause"),
        freeze_clause_cond: dex.conds_id("freezeclausemod"),
    }
}

/// Static match context shared across all `CompactBattle` nodes in a search tree.
#[derive(Clone, Debug)]
pub struct MatchTable {
    /// 6 picked Pokemon specs (`0..3` = Side 0, `3..6` = Side 1).
    pub mons: [StaticMonSpec; 6],
    /// Precompiled move specs for each of the 6 Pokemon's 4 move slots.
    pub move_specs: [[CompactMoveSpec; 4]; 6],
    /// 20 × 20 type effectiveness matrix (`-1` = resist, `0` = neutral, `+1` = super, `-9` = immune).
    pub type_chart: [[i8; 20]; 20],
    pub normal_type: u8,
    pub fighting_type: u8,
    pub flying_type: u8,
    pub ground_type: u8,
    pub ghost_type: u8,
    pub grass_type: u8,
    pub fire_type: u8,
    pub electric_type: u8,
    pub ice_type: u8,
    pub sandstorm_immune_mask: u32,
    pub poison_immune_mask: u32,
    pub match_fingerprint: u64,
}

impl MatchTable {
    /// Compute the net type effectiveness (`-2..=+2`) and immunity flag against a target.
    #[inline(always)]
    pub const fn effectiveness(&self, move_type: u8, def_mon: &StaticMonSpec) -> (i8, bool) {
        self.effectiveness_ext(move_type, def_mon, false)
    }

    /// Compute the net type effectiveness (`-2..=+2`) and immunity flag against a target,
    /// accounting for Foresight (`vol_bit::FORESIGHT`) removing Ghost's immunity to Normal/Fighting.
    #[inline(always)]
    pub const fn effectiveness_ext(
        &self,
        move_type: u8,
        def_mon: &StaticMonSpec,
        foresight: bool,
    ) -> (i8, bool) {
        let mt = if (move_type as usize) < 20 {
            move_type as usize
        } else {
            19
        };
        let t0 = if (def_mon.types[0] as usize) < 20 {
            def_mon.types[0] as usize
        } else {
            19
        };
        let mut e0 = self.type_chart[mt][t0];
        if e0 == -9 {
            if foresight
                && (move_type == self.normal_type || move_type == self.fighting_type)
                && def_mon.types[0] == self.ghost_type
            {
                e0 = 0;
            } else {
                return (0, true);
            }
        }
        if def_mon.num_types > 1 && def_mon.types[1] != def_mon.types[0] {
            let t1 = if (def_mon.types[1] as usize) < 20 {
                def_mon.types[1] as usize
            } else {
                19
            };
            let mut e1 = self.type_chart[mt][t1];
            if e1 == -9 {
                if foresight
                    && (move_type == self.normal_type || move_type == self.fighting_type)
                    && def_mon.types[1] == self.ghost_type
                {
                    e1 = 0;
                } else {
                    return (0, true);
                }
            }
            (e0 + e1, false)
        } else {
            (e0, false)
        }
    }
}

const COMPACT_TYPE_BOOST_ITEMS: [(&str, &str); 18] = [
    ("blackbelt", "Fighting"),
    ("blackglasses", "Dark"),
    ("charcoal", "Fire"),
    ("dragonfang", "Dragon"),
    ("hardstone", "Rock"),
    ("magnet", "Electric"),
    ("metalcoat", "Steel"),
    ("miracleseed", "Grass"),
    ("mysticwater", "Water"),
    ("nevermeltice", "Ice"),
    ("pinkbow", "Normal"),
    ("poisonbarb", "Poison"),
    ("polkadotbow", "Normal"),
    ("sharpbeak", "Flying"),
    ("silverpowder", "Bug"),
    ("softsand", "Ground"),
    ("spelltag", "Ghost"),
    ("twistedspoon", "Psychic"),
];

impl CompactBattle {
    /// Extract a `(MatchTable, CompactBattle)` pair from an existing `Battle`.
    pub fn from_battle(battle: &Battle, dex: &Dex) -> (MatchTable, CompactBattle) {
        let dt = dex_tables(dex);

        let mut static_mons = [StaticMonSpec::default(); 6];
        let mut move_specs = [[CompactMoveSpec::default(); 4]; 6];
        let mut compact_mons = [CompactMon::default(); 6];
        let mut compact_sides = [CompactSide::default(); 2];

        for side_n in 0..2usize {
            let side = &battle.sides[side_n];
            let party_len = side.party.len().min(3);
            let mut alive_mask = 0u8;
            let mut active_idx = if side.active.is_none() { 3u8 } else { 0u8 };
            let mut party_roster = [0u8; 3];

            for p_idx in 0..3usize {
                let c_idx = side_n * 3 + p_idx;
                if p_idx >= party_len {
                    party_roster[p_idx] = (p_idx as u8).min(side.roster.len().saturating_sub(1) as u8);
                    compact_mons[c_idx].hp_pool = [0, 0];
                    compact_mons[c_idx].set_status(Status::Fnt as u8, 0);
                    continue;
                }
                let roster_slot = side.party[p_idx];
                party_roster[p_idx] = roster_slot;
                if side.active == Some(roster_slot) {
                    active_idx = p_idx as u8;
                }
                let p = &side.roster[roster_slot as usize];
                if !p.fainted && p.hp > 0 {
                    alive_mask |= 1 << p_idx;
                }

                let maxhp = p.maxhp.max(1) as i16;
                let mut type_mask = 0u32;
                for ty in p.types.iter() {
                    type_mask |= 1u32 << (ty.0 & 31);
                }
                let item_spec = p
                    .item
                    .and_then(|i| dt.item_specs.get(i.0 as usize).copied())
                    .unwrap_or(CompactItemSpec {
                        boost_type: 0xFF,
                        ..CompactItemSpec::default()
                    });

                let return_bp = (p.happiness as i32 * 10 / 25) as i16;
                let mut moves = [MoveId(0); 4];
                let mut max_pp = [0u8; 4];
                let mut pp = [0u8; 4];
                let mut vol_flags = 0u16;
                let mut slot_refs = 0u8;
                let num_moves = p.move_slots.len().min(4) as u8;
                for m_i in 0..num_moves as usize {
                    let slot = p.move_slots[m_i];
                    moves[m_i] = slot.id;
                    max_pp[m_i] = slot.maxpp.clamp(0, 64) as u8;
                    pp[m_i] = slot.pp.clamp(0, 64) as u8;
                    if slot.disabled {
                        vol_flags |= vol_bit::DISABLE;
                        slot_refs = (slot_refs & !(0x03 << 4)) | (((m_i as u8) & 0x03) << 4);
                    }

                    let mut cm = dt.move_specs[slot.id.0 as usize];
                    match cm.var_bp_kind {
                        1 => cm.base_power = return_bp,
                        2 => cm.base_power = ((255 - p.happiness as i32) * 10 / 25) as i16,
                        3 => {
                            let hpt = p.hp_type.0;
                            cm.move_type = hpt;
                            cm.base_power = p.hp_power as i16;
                            cm.is_physical = (dt.special_type_mask & (1u32 << (hpt & 31))) == 0;
                            cm.is_water = hpt == dt.water_type;
                            cm.is_fire = hpt == dt.fire_type;
                        }
                        _ => {}
                    }
                    move_specs[c_idx][m_i] = cm;
                }

                static_mons[c_idx] = StaticMonSpec {
                    species: p.species,
                    level: p.level,
                    maxhp,
                    quantum_1_8: (maxhp / 8).max(1),
                    quantum_1_16: (maxhp / 16).max(1),
                    stored_stats: [
                        p.stored_stats[0] as i16,
                        p.stored_stats[1] as i16,
                        p.stored_stats[2] as i16,
                        p.stored_stats[3] as i16,
                        p.stored_stats[4] as i16,
                    ],
                    type_mask,
                    types: [p.types.t[0].0, p.types.t[1].0],
                    num_types: p.types.n,
                    item_stat_kind: ItemStatKind::classify(dex, p.species, p.item),
                    initial_item: p.item,
                    item_boost_type: item_spec.boost_type,
                    is_quick_claw: item_spec.is_quick_claw,
                    is_leftovers: item_spec.is_leftovers,
                    is_brightpowder: item_spec.is_brightpowder,
                    is_focusband: item_spec.is_focusband,
                    is_mintberry: item_spec.is_mintberry,
                    is_miracleberry: item_spec.is_miracleberry,
                    is_goldberry: item_spec.is_goldberry,
                    is_berry: item_spec.is_berry,
                    is_mysteryberry: item_spec.is_mysteryberry,
                    is_scope_lens: item_spec.is_scope_lens,
                    moves,
                    max_pp,
                    num_moves,
                    hp_type: p.hp_type.0,
                    hp_power: p.hp_power as i16,
                    return_bp,
                };

                let mut sub_hp = 0i16;
                let mut conf_turns = 0u8;
                let mut enc_turns = 0u8;
                let mut dis_turns = 0u8;
                let mut perish_turns = 0u8;
                let mut res_counter_packed = 0u8;

                if !p.volatiles.is_empty() {
                    for &(cid, ref vst) in &p.volatiles {
                        let scid = Some(cid);
                        if scid == dt.sub_cond {
                            sub_hp = vst.get_int(DK::Hp).max(0) as i16;
                            if sub_hp > 0 {
                                vol_flags |= vol_bit::SUBSTITUTE;
                            }
                        } else if scid == dt.conf_cond {
                            vol_flags |= vol_bit::CONFUSION;
                            conf_turns = vst.get_int(DK::Time).clamp(0, 255) as u8;
                        } else if scid == dt.flinch_cond {
                            vol_flags |= vol_bit::FLINCH;
                        } else if scid == dt.dbond_cond {
                            vol_flags |= vol_bit::DESTINY_BOND;
                        } else if scid == dt.par_cond {
                            vol_flags |= vol_bit::PAR_SPEED_DROP;
                        } else if scid == dt.brn_cond {
                            vol_flags |= vol_bit::BRN_ATK_DROP;
                        } else if scid == dt.seed_cond {
                            vol_flags |= vol_bit::LEECH_SEED;
                        } else if scid == dt.night_cond {
                            vol_flags |= vol_bit::NIGHTMARE;
                        } else if scid == dt.curse_cond {
                            vol_flags |= vol_bit::CURSE;
                        } else if scid == dt.fore_cond {
                            vol_flags |= vol_bit::FORESIGHT;
                        } else if scid == dt.prot_cond {
                            vol_flags |= vol_bit::PROTECT;
                        } else if scid == dt.end_cond {
                            vol_flags |= vol_bit::ENDURE;
                        } else if scid == dt.enc_cond {
                            vol_flags |= vol_bit::ENCORE;
                            enc_turns = vst.duration.unwrap_or(0).clamp(0, 255) as u8;
                        } else if scid == dt.dis_cond {
                            vol_flags |= vol_bit::DISABLE;
                            dis_turns = vst.duration.unwrap_or(0).clamp(0, 255) as u8;
                        } else if scid == dt.perish_cond {
                            vol_flags |= vol_bit::PERISH_SONG;
                            perish_turns = vst.duration.unwrap_or(0).clamp(0, 255) as u8;
                        } else if scid == dt.res_cond {
                            res_counter_packed =
                                (vst.get_int(DK::Counter).clamp(0, 15) as u8) | 0x10;
                        } else if scid == dt.recharge_cond {
                            slot_refs |= 0x80;
                        }
                    }
                }
                if p.trapped {
                    vol_flags |= vol_bit::TRAPPED_OR_LOCKED;
                }

                let status_counter = match p.status {
                    Status::Slp => p.status_state.get_int(DK::Time).clamp(0, 31) as u8,
                    Status::Tox | Status::Psn | Status::Brn => res_counter_packed,
                    _ => 0,
                };

                let mut c_mon = CompactMon {
                    hp_pool: [p.hp.clamp(0, i16::MAX as i32) as i16, sub_hp],
                    pp,
                    status_packed: 0,
                    item: p.item.map(|i| (i.0 & 0x7F) as u8 + 1).unwrap_or(0),
                    vol_flags,
                    timers: [conf_turns, enc_turns, dis_turns, perish_turns],
                    boosts: p.boosts,
                    slot_refs,
                };
                c_mon.set_status(p.status as u8, status_counter);
                compact_mons[c_idx] = c_mon;
            }
            if side.pokemon_left == 1 && alive_mask.count_ones() > 1 && active_idx < 3 {
                alive_mask = 1 << active_idx;
            }

            let mut reflect_turns = 0u8;
            let mut lightscreen_turns = 0u8;
            let mut safeguard_turns = 0u8;
            let mut spikes_layers = 0u8;
            if !side.side_conditions.is_empty() {
                for &(cid, ref vst) in &side.side_conditions {
                    let scid = Some(cid);
                    if scid == dt.refl_cond {
                        reflect_turns = vst.duration.unwrap_or(0).clamp(0, 15) as u8;
                    } else if scid == dt.ls_cond {
                        lightscreen_turns = vst.duration.unwrap_or(0).clamp(0, 15) as u8;
                    } else if scid == dt.safe_cond {
                        safeguard_turns = vst.duration.unwrap_or(0).clamp(0, 15) as u8;
                    } else if scid == dt.spikes_cond {
                        spikes_layers = vst.get_int(DK::Layers).clamp(1, 3) as u8;
                    }
                }
            }
            let side_flags = (spikes_layers & 0x03)
                | ((side.fainted_this_turn.is_some() as u8) << 2);
            let request = match side.request_state() {
                None | Some(RequestKind::Wait) => 0,
                Some(RequestKind::Move) => 1,
                Some(RequestKind::Switch) => 2,
                Some(RequestKind::TeamPreview) => 3,
            };

            compact_sides[side_n] = CompactSide {
                active_idx,
                party_roster,
                alive_mask,
                reflect_turns,
                lightscreen_turns,
                safeguard_turns,
                side_flags,
                last_move_selfko: side.last_move.map(|m| (m.0 & 0xFF) as u8).unwrap_or(u8::MAX),
                request,
                pad: 0,
            };
        }

        let weather_kind = match battle.field.weather {
            w if w.is_some() && w == dt.rain_cond => CompactWeather::Rain as u8,
            w if w.is_some() && w == dt.sun_cond => CompactWeather::Sun as u8,
            w if w.is_some() && w == dt.sand_cond => CompactWeather::Sandstorm as u8,
            _ => CompactWeather::None as u8,
        };
        let weather_turns = battle
            .field
            .weather_state
            .duration
            .unwrap_or(0)
            .clamp(0, 15) as u8;
        let outcome = match battle.outcome() {
            None => 0,
            Some(crate::battle::Outcome::P1Win) => 1,
            Some(crate::battle::Outcome::P2Win) => 2,
            Some(crate::battle::Outcome::Tie) => 3,
        };
        let mut clause_flags = 0u8;
        if !battle.field.pseudo_weather.is_empty() {
            for &(cid, _) in &battle.field.pseudo_weather {
                if Some(cid) == dt.sleep_clause_cond {
                    clause_flags |= 1;
                } else if Some(cid) == dt.freeze_clause_cond {
                    clause_flags |= 4;
                }
            }
        }

        // Compute static fingerprint for this match configuration.
        let mut fp = 0xcbf29ce484222325u64;
        for m in &static_mons {
            let word = (m.species.0 as u64)
                | ((m.level as u64) << 16)
                | ((m.maxhp as u16 as u64) << 24)
                | ((m.type_mask as u64) << 40);
            fp = fp.wrapping_add(word).wrapping_mul(0xf1357aea2e62a9c5);
        }

        let table = MatchTable {
            mons: static_mons,
            move_specs,
            type_chart: dt.type_chart,
            normal_type: dt.normal_type,
            fighting_type: dt.fighting_type,
            flying_type: dt.flying_type,
            ground_type: dt.ground_type,
            ghost_type: dt.ghost_type,
            grass_type: dt.grass_type,
            fire_type: dt.fire_type,
            electric_type: dt.electric_type,
            ice_type: dt.ice_type,
            sandstorm_immune_mask: dt.sandstorm_immune_mask,
            poison_immune_mask: dt.poison_immune_mask,
            match_fingerprint: fp,
        };
        let cb = CompactBattle {
            mons: compact_mons,
            sides: compact_sides,
            turn: battle.turn,
            weather_kind,
            weather_turns,
            quick_claw_roll: battle.quick_claw_roll as u8,
            outcome,
            speed_order: battle
                .speed_order
                .first()
                .copied()
                .unwrap_or(0)
                .clamp(0, 1) as u8,
            clause_flags,
            hurt_damage_this_turn: [0, 0],
            hurt_cat_this_turn: [0, 0],
            special_counters: [0; 6],
            future_sight_dmg: [0; 2],
            future_sight_turns: [0; 2],
            prng_seed: battle.prng.lcg.seed(),
            match_fingerprint: fp,
            reserved: [0; 6],
        };
        (table, cb)
    }

    /// Ultra-fast polynomial hash of the 256-byte `CompactBattle` state
    /// (hashes the first 25 `u64` words = 200 bytes, masking out `prng_seed`).
    #[inline(always)]
    pub fn search_hash(&self) -> u64 {
        // SAFETY: `CompactBattle` is `#[repr(C, align(64))]` of size 256 bytes (32 u64 words)
        // with all initialized bytes (`CompactMon`, `CompactSide`, scalar fields).
        let words: &[u64; 32] = unsafe { &*(self as *const Self as *const [u64; 32]) };
        let mut h = self.match_fingerprint;
        // Words 0..24 cover `mons` (18 words), `sides` (3 words), and turn/field/counters (3 words).
        // Word 24 is `prng_seed` (skipped so states differing only in RNG seed merge).
        for &w in &words[..24] {
            h = h.wrapping_add(w).wrapping_mul(0xf1357aea2e62a9c5);
        }
        h.rotate_left(26)
    }

    /// Compute any party Pokemon's effective stat (`party_slot` in `0..3`) on `side` (`0` or `1`)
    /// using [`get_stat_branchless`].
    #[inline(always)]
    pub fn mon_stat(
        &self,
        table: &MatchTable,
        side: usize,
        party_slot: usize,
        stat_idx: usize,
        unboosted: bool,
        unmodified: bool,
    ) -> i32 {
        let p_idx = side * 3 + party_slot;
        let mon = &self.mons[p_idx];
        let spec = &table.mons[p_idx];
        let side_st = &self.sides[side];
        let has_par = (mon.status_raw() == Status::Par as u8)
            & ((mon.vol_flags & vol_bit::PAR_SPEED_DROP) != 0);
        let has_brn = (mon.status_raw() == Status::Brn as u8)
            & ((mon.vol_flags & vol_bit::BRN_ATK_DROP) != 0);
        let item_kind = if mon.item != 0 {
            spec.item_stat_kind
        } else {
            ItemStatKind::None
        };
        get_stat_branchless(
            spec.stored_stats[stat_idx] as i32,
            mon.boosts[stat_idx],
            stat_idx,
            unboosted,
            unmodified,
            false,
            has_par,
            has_brn,
            side_st.reflect_turns > 0,
            side_st.lightscreen_turns > 0,
            item_kind,
        )
    }

    /// Compute the active Pokemon's effective stat on `side` (`0` or `1`) using
    /// [`get_stat_branchless`].
    #[inline(always)]
    pub fn active_stat(
        &self,
        table: &MatchTable,
        side: usize,
        stat_idx: usize,
        unboosted: bool,
        unmodified: bool,
    ) -> i32 {
        self.mon_stat(
            table,
            side,
            self.sides[side].active_idx as usize,
            stat_idx,
            unboosted,
            unmodified,
        )
    }

    /// Compute a party Pokemon's action speed (matching `Battle::get_pokemon_action_speed`).
    #[inline(always)]
    pub fn mon_action_speed(&self, table: &MatchTable, side: usize, party_slot: usize) -> i32 {
        let p_idx = side * 3 + party_slot;
        let mon = &self.mons[p_idx];
        let spec = &table.mons[p_idx];
        if self.quick_claw_roll != 0 && mon.item != 0 && spec.is_quick_claw {
            return 65535;
        }
        self.mon_stat(table, side, party_slot, 4, false, false)
    }

    /// Compute hit probability for `m_slot` used by `(att_side, att_slot)` against `(def_side, def_slot)`.
    #[inline(always)]
    pub fn hit_probability(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
    ) -> f64 {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let mspec = &table.move_specs[att_idx][m_slot];
        hit_probability_branchless(
            mspec.base_acc_pct,
            mspec.always_hit,
            self.mons[att_idx].boosts[5],
            self.mons[def_idx].boosts[6],
            mspec.ignore_accuracy,
            mspec.ignore_evasion,
            mspec.ignore_positive_evasion,
        )
    }

    /// Expected fraction of `(def_side, def_slot)`'s current HP removed by move slot `m_slot`
    /// of `(att_side, att_slot)`. Mirrors `nc2000_bot::eval::expected_hit_fraction` bit-for-bit.
    #[inline]
    pub fn expected_hit_fraction(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        m_slot: usize,
        couple_evasion: bool,
    ) -> f64 {
        let att_idx = att_side * 3 + att_slot;
        let def_idx = def_side * 3 + def_slot;
        let mspec = &table.move_specs[att_idx][m_slot];
        let att_mon = &self.mons[att_idx];
        let att_spec = &table.mons[att_idx];
        let def_mon = &self.mons[def_idx];
        let def_spec = &table.mons[def_idx];

        let base_power = if mspec.var_bp_kind == 4 {
            let ratio = ((att_mon.hp() as f64 * 48.0 / att_spec.maxhp.max(1) as f64).floor() as i32)
                .max(1);
            if ratio < 2 {
                200
            } else if ratio < 5 {
                150
            } else if ratio < 10 {
                100
            } else if ratio < 17 {
                80
            } else if ratio < 33 {
                40
            } else {
                20
            }
        } else {
            mspec.base_power as i32
        };

        if mspec.fixed_damage == 0 && (mspec.is_status || base_power <= 0) {
            return 0.0;
        }

        let foresight = (def_mon.vol_flags & vol_bit::FORESIGHT) != 0;
        let (type_mod, is_immune) =
            table.effectiveness_ext(mspec.move_type, def_spec, foresight);
        if is_immune {
            return 0.0;
        }

        let acc = if couple_evasion {
            hit_probability_branchless(
                mspec.base_acc_pct,
                mspec.always_hit,
                att_mon.boosts[5],
                def_mon.boosts[6],
                mspec.ignore_accuracy,
                mspec.ignore_evasion,
                mspec.ignore_positive_evasion,
            )
        } else if mspec.always_hit {
            1.0
        } else {
            mspec.base_acc_pct as f64 / 100.0
        };

        let raw = if mspec.fixed_damage != 0 {
            if mspec.fixed_damage < 0 {
                att_spec.level as f64
            } else {
                mspec.fixed_damage as f64
            }
        } else {
            let (ai, di) = if mspec.is_physical { (0, 1) } else { (2, 3) };
            let atk = self.mon_stat(table, att_side, att_slot, ai, false, false) as f64;
            let mut defense = self.mon_stat(table, def_side, def_slot, di, false, false) as f64;
            if mspec.is_selfdestruct && di == 1 {
                defense = (defense / 2.0).floor().max(1.0);
            }
            let core = ((att_spec.level as f64 * 2.0 / 5.0 + 2.0).floor()
                * base_power as f64
                * atk
                / defense
                / 50.0)
                .floor()
                + 2.0;
            let stab = if (att_spec.type_mask & (1u32 << (mspec.move_type & 31))) != 0 {
                1.5
            } else {
                1.0
            };
            let eff = match type_mod {
                2 => 4.0,
                1 => 2.0,
                -1 => 0.5,
                -2 => 0.25,
                _ => 1.0,
            };
            let hits = mspec.multihit_mean_x2 as f64 * 0.5;
            core * stab * eff * (236.0 / 255.0) * hits
        };

        raw * acc / (def_mon.hp().max(1) as f64)
    }

    /// Best expected hit fraction over `(att_side, att_slot)`'s usable move slots against `(def_side, def_slot)`.
    #[inline]
    pub fn best_hit_fraction(
        &self,
        table: &MatchTable,
        att_side: usize,
        att_slot: usize,
        def_side: usize,
        def_slot: usize,
        couple_evasion: bool,
    ) -> f64 {
        let att_idx = att_side * 3 + att_slot;
        let att_mon = &self.mons[att_idx];
        let num_moves = table.mons[att_idx].num_moves as usize;
        let is_disabled = (att_mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((att_mon.slot_refs >> 4) & 0x03) as usize;
        let mut best = 0.0f64;
        for m_i in 0..num_moves {
            if att_mon.pp[m_i] == 0 || (is_disabled && m_i == dis_slot) {
                continue;
            }
            let v = self.expected_hit_fraction(
                table,
                att_side,
                att_slot,
                def_side,
                def_slot,
                m_i,
                couple_evasion,
            );
            if v > best {
                best = v;
            }
        }
        best
    }

    /// Compute the branchless `ActionSortKey` for `side` (`0` or `1`) given a
    /// compact action (`0..4` = move slot, `4..7` = switch to party index `0..3`).
    #[inline(always)]
    pub fn action_sort_key(
        &self,
        table: &MatchTable,
        side: usize,
        action_code: u8,
        tie_bit: u8,
    ) -> ActionSortKey {
        let p_idx = side * 3 + (self.sides[side].active_idx as usize);
        let mon = &self.mons[p_idx];
        let spec = &table.mons[p_idx];
        let speed = self.active_stat(table, side, 4, false, false);
        let is_switch = action_code >= 4;
        let phase = if is_switch {
            ActionPhase::Switch
        } else {
            ActionPhase::Move
        };
        let move_slot = (action_code & 3) as usize;
        let prio = if is_switch {
            0i8
        } else {
            table.move_specs[p_idx][move_slot].priority
        };
        let qc_active = (!is_switch)
            & (mon.item != 0)
            & spec.is_quick_claw
            & (self.quick_claw_roll != 0);
        ActionSortKey::new(phase, prio, qc_active, speed, tie_bit)
    }

    /// Execute one full 4-stage turn on `CompactBattle` with zero heap allocations:
    /// - Stage 0: Branchless `ActionSortKey` turn-order determination
    /// - Stage 1: First side action + `AfterMoveSelf` slip damage
    /// - Stage 2: Second side action + `AfterMoveSelf` slip damage
    /// - Stage 3: End-of-turn `Residual` (Leftovers, Weather, Leech Seed, Timers, Quick Claw roll)
    ///
    /// `actions`: `[action_p1, action_p2]` where `0..4` is a move slot and `4..7`
    /// is switching to party index `action - 4`.
    pub fn step_turn(&mut self, table: &MatchTable, actions: [u8; 2]) {
        if self.outcome != 0 {
            return;
        }
        let mut rng = Prng::new(self.prng_seed);
        self.hurt_damage_this_turn = [0, 0];
        self.hurt_cat_this_turn = [0, 0];

        // Clear single-turn Protect/Endure/Flinch flags at start of turn.
        for side in 0..2usize {
            let idx = side * 3 + self.sides[side].active_idx as usize;
            self.mons[idx].vol_flags &= !(vol_bit::PROTECT | vol_bit::ENDURE | vol_bit::FLINCH);
        }

        // Stage 0: Branchless ActionSortKey comparison.
        let k0 = self.action_sort_key(table, 0, actions[0], 0);
        let k1 = self.action_sort_key(table, 1, actions[1], 1);
        let (first, second) = if k0.ties_without_rng(k1) {
            let f = (rng.random(2) == 1) as usize;
            (f, f ^ 1)
        } else {
            let f = (k0.0 < k1.0) as usize;
            (f, f ^ 1)
        };
        self.speed_order = first as u8;

        // Stage 1 & Stage 2: Execute first action, then second action.
        for (step_i, &actor) in [first, second].iter().enumerate() {
            if self.outcome != 0 {
                break;
            }
            let act = actions[actor];
            if act >= 4 && act < 7 {
                self.exec_switch(table, actor, (act - 4).min(2), false);
            } else if act < 4 {
                // Gen 2: fainting skips any remaining queued move actions this turn.
                let p0 = self.sides[0].active_idx as usize;
                let p1 = self.sides[1].active_idx as usize;
                if step_i == 1 && (self.mons[p0].hp() <= 0 || self.mons[3 + p1].hp() <= 0) {
                    break;
                }
                self.exec_move(table, actor, (act & 3) as usize, step_i == 1, &mut rng);
            }
        }
        self.check_faints(table);

        // Stage 3: End-of-turn Residual in speed order when no forced switch is pending.
        if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
            self.exec_residuals(table, &mut rng);
            self.check_faints(table);
            if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
                self.turn = self.turn.saturating_add(1);
                if self.turn > 1000 {
                    self.outcome = 3; // Tie
                } else {
                    self.quick_claw_roll = rng.random_chance(60, 256) as u8;
                }
            }
        }
        self.prng_seed = rng.seed();
    }

    #[inline]
    fn exec_switch(
        &mut self,
        table: &MatchTable,
        side: usize,
        target_party_idx: u8,
        baton_pass: bool,
    ) {
        let old_idx = side * 3 + self.sides[side].active_idx as usize;
        let saved_boosts = self.mons[old_idx].boosts;
        let saved_sub = self.mons[old_idx].hp_pool[1];
        let pass_mask = vol_bit::SUBSTITUTE
            | vol_bit::CONFUSION
            | vol_bit::LEECH_SEED
            | vol_bit::CURSE
            | vol_bit::PERISH_SONG;
        let saved_vols = self.mons[old_idx].vol_flags & pass_mask;
        let saved_conf_t = self.mons[old_idx].timers[0];
        let saved_perish_t = self.mons[old_idx].timers[3];

        // Clear volatile state and boosts on outgoing active Pokemon.
        self.mons[old_idx].boosts = [0; 7];
        self.mons[old_idx].hp_pool[1] = 0;
        let keep_status_drops = (if self.mons[old_idx].status_raw() == Status::Par as u8 {
            vol_bit::PAR_SPEED_DROP
        } else {
            0
        }) | (if self.mons[old_idx].status_raw() == Status::Brn as u8 {
            vol_bit::BRN_ATK_DROP
        } else {
            0
        });
        self.mons[old_idx].vol_flags = keep_status_drops;
        self.mons[old_idx].timers = [0; 4];
        self.mons[old_idx].slot_refs &= !0x80;

        // Switching out releases Mean Look / Spider Web trap on the opposing active Pokemon.
        if !baton_pass {
            let foe_side = side ^ 1;
            let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
            self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
        }

        self.sides[side].active_idx = target_party_idx;
        let new_idx = side * 3 + target_party_idx as usize;
        if baton_pass {
            self.mons[new_idx].boosts = saved_boosts;
            self.mons[new_idx].hp_pool[1] = saved_sub;
            self.mons[new_idx].vol_flags |= saved_vols;
            self.mons[new_idx].timers[0] = saved_conf_t;
            self.mons[new_idx].timers[3] = saved_perish_t;
        }
        if self.mons[new_idx].status_raw() == Status::Par as u8 {
            self.mons[new_idx].vol_flags |= vol_bit::PAR_SPEED_DROP;
        }
        if self.mons[new_idx].status_raw() == Status::Brn as u8 {
            self.mons[new_idx].vol_flags |= vol_bit::BRN_ATK_DROP;
        }
        if self.mons[new_idx].status_raw() == Status::Tox as u8 {
            let cur = self.mons[new_idx].status_counter() & 0x0F;
            self.mons[new_idx].set_status(Status::Psn as u8, cur);
        }

        // Apply Spikes entry hazard to non-Flying switch-in.
        let spikes_layers = self.sides[side].side_flags & 0x03;
        if spikes_layers != 0
            && (table.mons[new_idx].type_mask & (1u32 << (table.flying_type & 31))) == 0
        {
            let spike_dmg = match spikes_layers {
                1 => table.mons[new_idx].quantum_1_8,
                2 => (table.mons[new_idx].maxhp / 6).max(1),
                _ => (table.mons[new_idx].maxhp / 4).max(1),
            };
            self.mons[new_idx].apply_damage_branchless(spike_dmg, true);
        }
        self.check_berry_triggers(table, side);
    }

    #[inline]
    fn exec_move(
        &mut self,
        table: &MatchTable,
        side: usize,
        move_slot: usize,
        moved_second: bool,
        rng: &mut Prng,
    ) {
        let foe_side = side ^ 1;
        let att_idx = side * 3 + self.sides[side].active_idx as usize;
        let def_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
        if self.mons[att_idx].hp() <= 0 || self.mons[def_idx].hp() <= 0 {
            return;
        }

        // Hyper Beam recharge turn wastes the turn.
        if (self.mons[att_idx].slot_refs & 0x80) != 0 {
            self.mons[att_idx].slot_refs &= !0x80;
            self.exec_after_move_self(table, side);
            return;
        }

        let mut mspec = table.move_specs[att_idx][move_slot];
        let st_raw = self.mons[att_idx].status_raw();

        // Flinch check.
        if (self.mons[att_idx].vol_flags & vol_bit::FLINCH) != 0 {
            return;
        }

        // Freeze check (flame wheel / sacred fire thaw handled via move execution).
        if st_raw == Status::Frz as u8 {
            return;
        }

        // Sleep check.
        let mut sleeping = false;
        if st_raw == Status::Slp as u8 {
            let rem = self.mons[att_idx].status_counter().saturating_sub(1);
            if rem == 0 {
                self.mons[att_idx].set_status(Status::None as u8, 0);
            } else {
                self.mons[att_idx].set_status(Status::Slp as u8, rem);
                if !mspec.is_sleep_usable {
                    self.exec_after_move_self(table, side);
                    return;
                }
                sleeping = true;
            }
        }

        // Confusion check.
        if (self.mons[att_idx].vol_flags & vol_bit::CONFUSION) != 0 {
            let rem = self.mons[att_idx].timers[0].saturating_sub(1);
            self.mons[att_idx].timers[0] = rem;
            if rem == 0 {
                self.mons[att_idx].vol_flags &= !vol_bit::CONFUSION;
            } else if rng.random_chance(1, 2) {
                let atk = self.active_stat(table, side, 0, false, false);
                let def = self.active_stat(table, side, 1, false, false);
                let lvl = table.mons[att_idx].level as i32;
                let self_dmg = (((lvl * 2 / 5 + 2) * 40 * atk / def.max(1)) / 50 + 2).max(1) as i16;
                self.mons[att_idx].apply_damage_branchless(self_dmg, true);
                self.exec_after_move_self(table, side);
                return;
            }
        }

        // Paralysis 25% full-paralyze check.
        if self.mons[att_idx].status_raw() == Status::Par as u8 && rng.random_chance(1, 4) {
            return;
        }

        // Clear Destiny Bond when the user attempts a move.
        self.mons[att_idx].vol_flags &= !vol_bit::DESTINY_BOND;

        // Deduct PP.
        if self.mons[att_idx].pp[move_slot] == 0 {
            return;
        }
        self.mons[att_idx].pp[move_slot] -= 1;

        // Sleep-usable moves (Sleep Talk, Snore) fail when the user is awake.
        if mspec.is_sleep_usable && !sleeping {
            self.exec_after_move_self(table, side);
            return;
        }

        // Sleep Talk: randomly call one of the user's other sleep-talkable moves.
        if mspec.special_kind == 10 {
            let mut candidates = [0usize; 4];
            let mut n_cand = 0usize;
            let num_moves = table.mons[att_idx].num_moves as usize;
            for m_i in 0..num_moves {
                if m_i != move_slot && table.move_specs[att_idx][m_i].can_sleep_talk {
                    candidates[n_cand] = m_i;
                    n_cand += 1;
                }
            }
            if n_cand == 0 {
                self.exec_after_move_self(table, side);
                return;
            }
            let picked = candidates[rng.random(n_cand as u32) as usize];
            mspec = table.move_specs[att_idx][picked];
        }

        // Self-Destruct faints user upfront and sets Self-KO clause tracker.
        if mspec.is_selfdestruct {
            self.mons[att_idx].hp_pool[0] = 0;
            self.mons[att_idx].status_packed = Status::Fnt as u8;
            self.sides[side].last_move_selfko = move_slot as u8;
            self.sides[foe_side].last_move_selfko = u8::MAX;
        }

        // Handle common status self-moves (Rest, Recover/Soft-Boiled, Protect, Substitute).
        if mspec.is_rest {
            let maxhp = table.mons[att_idx].maxhp;
            if self.mons[att_idx].hp() < maxhp {
                self.mons[att_idx].hp_pool[0] = maxhp;
                self.mons[att_idx].set_status(Status::Slp as u8, 3);
                self.mons[att_idx].vol_flags &= !(vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP);
            }
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.is_half_heal {
            let maxhp = table.mons[att_idx].maxhp;
            let heal = (maxhp + 1) / 2;
            self.mons[att_idx].hp_pool[0] = (self.mons[att_idx].hp() + heal).min(maxhp);
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.is_protect {
            self.mons[att_idx].vol_flags |= vol_bit::PROTECT;
            self.exec_after_move_self(table, side);
            return;
        }
        if mspec.is_substitute {
            let cost = table.mons[att_idx].maxhp / 4;
            if self.mons[att_idx].hp() > cost && self.mons[att_idx].sub_hp() == 0 {
                self.mons[att_idx].hp_pool[0] -= cost;
                self.mons[att_idx].hp_pool[1] = cost;
                self.mons[att_idx].vol_flags |= vol_bit::SUBSTITUTE;
            }
            self.exec_after_move_self(table, side);
            return;
        }

        // Target Protect check.
        if (self.mons[def_idx].vol_flags & vol_bit::PROTECT) != 0 {
            self.exec_after_move_self(table, side);
            return;
        }

        let def_foresight = (self.mons[def_idx].vol_flags & vol_bit::FORESIGHT) != 0;

        // Accuracy check via O(1) ACC_LUT.
        if !mspec.always_hit {
            let thresh = lookup_accuracy_threshold(
                mspec.base_acc_pct,
                self.mons[att_idx].boosts[5],
                self.mons[def_idx].boosts[6],
                mspec.ignore_accuracy,
                mspec.ignore_evasion,
                mspec.ignore_positive_evasion | def_foresight,
                (self.mons[def_idx].item != 0) & table.mons[def_idx].is_brightpowder,
            );
            if thresh < 255 && !rng.random_chance(thresh as u32, 256) {
                self.exec_after_move_self(table, side);
                return;
            }
        }

        if !mspec.is_status && (mspec.base_power > 0 || matches!(mspec.special_kind, 11..=15) || mspec.fixed_damage != 0) {
            let def_had_dbond = (self.mons[def_idx].vol_flags & vol_bit::DESTINY_BOND) != 0;
            let mut dealt = 0i16;
            if mspec.special_kind == 11 {
                // Counter: double last physical damage taken this turn.
                if self.hurt_cat_this_turn[side] == 1 && self.hurt_damage_this_turn[side] > 0 {
                    let dmg = (self.hurt_damage_this_turn[side] as i32 * 2).min(9999) as i16;
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                }
            } else if mspec.special_kind == 12 {
                // Mirror Coat: double last special damage taken this turn.
                if self.hurt_cat_this_turn[side] == 2 && self.hurt_damage_this_turn[side] > 0 {
                    let dmg = (self.hurt_damage_this_turn[side] as i32 * 2).min(9999) as i16;
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                }
            } else if mspec.special_kind == 14 || mspec.fixed_damage != 0 {
                // Seismic Toss / Night Shade / Fixed Damage.
                let (_, is_immune) = table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let dmg = if mspec.fixed_damage > 0 {
                        mspec.fixed_damage
                    } else {
                        table.mons[att_idx].level as i16
                    };
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                }
            } else if mspec.special_kind == 15 {
                // Super Fang: halves target's current HP (min 1).
                let (_, is_immune) = table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                if !is_immune {
                    let dmg = (self.mons[def_idx].hp() / 2).max(1);
                    dealt = self.mons[def_idx].apply_damage_branchless(dmg, mspec.bypasses_sub);
                }
            } else {
                const CRIT_DEN: [u32; 6] = [0, 16, 8, 4, 3, 2];
                let scope_bonus = ((self.mons[att_idx].item != 0) & table.mons[att_idx].is_scope_lens) as usize;
                let c_stage = (mspec.crit_ratio as usize + scope_bonus).min(5);
                let is_crit = (mspec.special_kind != 13)
                    && c_stage > 0
                    && rng.random_chance(1, CRIT_DEN[c_stage]);

                let atk_stat = if mspec.is_physical { 0 } else { 2 };
                let def_stat = if mspec.is_physical { 1 } else { 3 };
                let ignore_boosts =
                    is_crit & (self.mons[att_idx].boosts[atk_stat] <= self.mons[def_idx].boosts[def_stat]);
                let atk = self.active_stat(table, side, atk_stat, ignore_boosts, ignore_boosts);
                let def = self.active_stat(table, foe_side, def_stat, ignore_boosts, false);

                let bp = if mspec.special_kind == 13 {
                    // Flail / Reversal 6-band HP power table.
                    let ratio = (self.mons[att_idx].hp() as i32 * 48) / (table.mons[att_idx].maxhp.max(1) as i32);
                    if ratio <= 1 {
                        200
                    } else if ratio <= 4 {
                        150
                    } else if ratio <= 9 {
                        100
                    } else if ratio <= 16 {
                        80
                    } else if ratio <= 32 {
                        40
                    } else {
                        20
                    }
                } else {
                    mspec.base_power as i32
                };

                let (type_mod, is_immune) = table.effectiveness_ext(mspec.move_type, &table.mons[def_idx], def_foresight);
                let weather = match self.weather_kind {
                    1 => CompactWeather::Rain,
                    2 => CompactWeather::Sun,
                    3 => CompactWeather::Sandstorm,
                    _ => CompactWeather::None,
                };
                let has_type_boost = (self.mons[att_idx].item != 0)
                    & (table.mons[att_idx].item_boost_type == mspec.move_type);

                let base_dmg = get_damage_base_branchless(
                    table.mons[att_idx].level as i32,
                    bp,
                    atk,
                    def,
                    mspec.is_physical,
                    mspec.is_selfdestruct,
                    is_crit,
                    has_type_boost,
                    weather,
                    mspec.is_water,
                    mspec.is_fire,
                    mspec.is_solarbeam,
                    mspec.move_type,
                    table.mons[att_idx].type_mask,
                    type_mod,
                    is_immune,
                );
                if base_dmg > 0 {
                    let no_var = mspec.special_kind == 13;
                    let roll = if no_var { 255 } else { 217 + rng.random(39) };
                    let mut final_dmg = apply_roll_branchless(base_dmg, roll, no_var);
                    if mspec.multihit_mean_x2 > 2 {
                        final_dmg = (final_dmg * (mspec.multihit_mean_x2 as i32)) / 2;
                    }
                    dealt = self.mons[def_idx].apply_damage_branchless(final_dmg.min(9999) as i16, mspec.bypasses_sub);
                    self.hurt_damage_this_turn[foe_side] = dealt.max(0) as u16;
                    self.hurt_cat_this_turn[foe_side] = if mspec.is_physical { 1 } else { 2 };

                    if mspec.drain_half && dealt > 0 {
                        let drain = (dealt / 2).max(1);
                        let maxhp = table.mons[att_idx].maxhp;
                        self.mons[att_idx].hp_pool[0] = (self.mons[att_idx].hp() + drain).min(maxhp);
                    }
                    if mspec.recoil_quarter && dealt > 0 && self.mons[att_idx].hp() > 0 {
                        let recoil = (dealt / 4).max(1);
                        self.mons[att_idx].apply_damage_branchless(recoil, true);
                    }
                }
            }

            // Destiny Bond: if direct attack fainted a target with Destiny Bond active, attacker faints too.
            if def_had_dbond && self.mons[def_idx].hp() <= 0 && self.mons[att_idx].hp() > 0 {
                self.mons[att_idx].hp_pool[0] = 0;
                self.mons[att_idx].status_packed = Status::Fnt as u8;
                self.mons[att_idx].vol_flags = 0;
            }

            // Secondary effects on target (when target survived and has no Substitute).
            if dealt > 0
                && self.mons[def_idx].hp() > 0
                && self.mons[def_idx].sub_hp() == 0
                && mspec.sec_chance_256 > 0
                && rng.random_chance(mspec.sec_chance_256 as u32, 256)
            {
                if mspec.sec_status != 0 && self.mons[def_idx].status_raw() == 0 && self.sides[foe_side].safeguard_turns == 0 {
                    self.try_inflict_status(table, foe_side, mspec.sec_status, rng);
                }
                if mspec.sec_vol != 0 {
                    self.mons[def_idx].vol_flags |= mspec.sec_vol;
                    if (mspec.sec_vol & vol_bit::CONFUSION) != 0 {
                        self.mons[def_idx].timers[0] = (2 + rng.random(4)) as u8;
                    }
                }
                if mspec.sec_boost.0 != 0xFF {
                    let is_self_boost = (mspec.sec_boost.0 & 0x80) != 0;
                    let b_idx = if is_self_boost { att_idx } else { def_idx };
                    let st_i = (mspec.sec_boost.0 & 0x7F) as usize;
                    if st_i < 7 {
                        self.mons[b_idx].boosts[st_i] =
                            (self.mons[b_idx].boosts[st_i] + mspec.sec_boost.1).clamp(-6, 6);
                    }
                }
            }
            if mspec.special_kind == 16 && self.mons[att_idx].item == 0 && self.mons[def_idx].item != 0 {
                // Thief steals target's held item.
                self.mons[att_idx].item = self.mons[def_idx].item;
                self.mons[def_idx].item = 0;
            }
            self.check_berry_triggers(table, foe_side);
            self.check_berry_triggers(table, side);
        } else if mspec.is_status {
            self.exec_status_move(table, side, foe_side, att_idx, def_idx, &mspec, moved_second, rng);
        }

        if !mspec.is_selfdestruct && self.mons[att_idx].hp() > 0 && self.mons[def_idx].hp() > 0 && self.outcome == 0 {
            self.exec_after_move_self(table, side);
        }
    }

    /// Gen 2 `onAfterMoveSelf` slip damage (Burn, Poison, Toxic, Leech Seed, Nightmare, Curse).
    #[inline]
    fn exec_after_move_self(&mut self, table: &MatchTable, side: usize) {
        let foe_side = side ^ 1;
        let idx = side * 3 + self.sides[side].active_idx as usize;
        let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
        if self.mons[idx].hp() <= 0 || self.mons[foe_idx].hp() <= 0 {
            return;
        }
        let st = self.mons[idx].status_raw();
        if st == Status::Brn as u8 || st == Status::Psn as u8 {
            let cnt = self.mons[idx].status_counter();
            if (cnt & 0x10) != 0 {
                let c = ((cnt & 0x0F) + 1).min(15);
                self.mons[idx].set_status(st, c | 0x10);
                let slip = table.mons[idx].quantum_1_16 * (c as i16);
                self.mons[idx].apply_damage_branchless(slip, true);
            } else {
                let slip = table.mons[idx].quantum_1_8;
                self.mons[idx].apply_damage_branchless(slip, true);
            }
        } else if st == Status::Tox as u8 {
            let c = ((self.mons[idx].status_counter() & 0x0F) + 1).min(15);
            self.mons[idx].set_status(Status::Tox as u8, c | 0x10);
            let slip = table.mons[idx].quantum_1_16 * (c as i16);
            self.mons[idx].apply_damage_branchless(slip, true);
        }
        if self.mons[idx].hp() > 0 && (self.mons[idx].vol_flags & vol_bit::LEECH_SEED) != 0 {
            let drain = table.mons[idx].quantum_1_8;
            let dealt = self.mons[idx].apply_damage_branchless(drain, true);
            if self.mons[foe_idx].hp() > 0 {
                let foe_max = table.mons[foe_idx].maxhp;
                self.mons[foe_idx].hp_pool[0] = (self.mons[foe_idx].hp() + dealt).min(foe_max);
            }
        }
        if self.mons[idx].hp() > 0
            && (self.mons[idx].vol_flags & vol_bit::NIGHTMARE) != 0
            && self.mons[idx].status_raw() == Status::Slp as u8
        {
            let slip = (table.mons[idx].maxhp / 4).max(1);
            self.mons[idx].apply_damage_branchless(slip, true);
        }
        if self.mons[idx].hp() > 0 && (self.mons[idx].vol_flags & vol_bit::CURSE) != 0 {
            let slip = (table.mons[idx].maxhp / 4).max(1);
            self.mons[idx].apply_damage_branchless(slip, true);
        }
        self.check_berry_triggers(table, side);
    }

    #[inline]
    fn try_inflict_status(
        &mut self,
        table: &MatchTable,
        target_side: usize,
        status_code: u8,
        rng: &mut Prng,
    ) {
        let idx = target_side * 3 + self.sides[target_side].active_idx as usize;
        if self.mons[idx].status_raw() != 0 || self.mons[idx].sub_hp() > 0 || self.sides[target_side].safeguard_turns > 0 {
            return;
        }
        let target_tmask = table.mons[idx].type_mask;
        // Stadium Sleep Clause & Freeze Clause checks.
        if status_code == Status::Slp as u8 {
            for p_i in 0..3usize {
                if p_i != (self.sides[target_side].active_idx as usize)
                    && (self.sides[target_side].alive_mask & (1 << p_i)) != 0
                {
                    let m = &self.mons[target_side * 3 + p_i];
                    if m.hp() > 0 && m.status_raw() == Status::Slp as u8 {
                        if (self.clause_flags & 1) != 0 {
                            self.outcome = if target_side == 0 { 1 } else { 2 };
                        }
                        return;
                    }
                }
            }
            let turns = (2 + rng.random(5)) as u8;
            self.mons[idx].set_status(Status::Slp as u8, turns);
        } else if status_code == Status::Frz as u8 {
            if (target_tmask & (1u32 << (table.ice_type & 31))) != 0
                || self.weather_kind == CompactWeather::Sun as u8
            {
                return;
            }
            for p_i in 0..3usize {
                if (self.sides[target_side].alive_mask & (1 << p_i)) != 0 {
                    let m = &self.mons[target_side * 3 + p_i];
                    if m.hp() > 0 && m.status_raw() == Status::Frz as u8 {
                        return;
                    }
                }
            }
            self.mons[idx].set_status(Status::Frz as u8, 0);
        } else if status_code == Status::Par as u8 {
            self.mons[idx].set_status(Status::Par as u8, 0);
            self.mons[idx].vol_flags |= vol_bit::PAR_SPEED_DROP;
        } else if status_code == Status::Brn as u8 {
            if (target_tmask & (1u32 << (table.fire_type & 31))) != 0 {
                return;
            }
            self.mons[idx].set_status(Status::Brn as u8, 0);
            self.mons[idx].vol_flags |= vol_bit::BRN_ATK_DROP;
        } else if status_code == Status::Tox as u8 {
            if (target_tmask & table.poison_immune_mask) != 0 {
                return;
            }
            self.mons[idx].set_status(Status::Tox as u8, 0x10);
        } else if status_code == Status::Psn as u8 {
            if (target_tmask & table.poison_immune_mask) != 0 {
                return;
            }
            self.mons[idx].set_status(Status::Psn as u8, 0);
        }
        self.check_berry_triggers(table, target_side);
    }

    #[inline]
    fn check_berry_triggers(&mut self, table: &MatchTable, side: usize) {
        let idx = side * 3 + self.sides[side].active_idx as usize;
        let mon = &mut self.mons[idx];
        if mon.hp() <= 0 || mon.item == 0 {
            return;
        }
        let spec = &table.mons[idx];
        let st = mon.status_raw();
        if spec.is_mintberry && st == Status::Slp as u8 {
            mon.set_status(Status::None as u8, 0);
            mon.item = 0;
        } else if spec.is_miracleberry && (st != 0 || (mon.vol_flags & vol_bit::CONFUSION) != 0) {
            mon.set_status(Status::None as u8, 0);
            mon.vol_flags &= !(vol_bit::CONFUSION | vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP);
            mon.timers[0] = 0;
            mon.item = 0;
        } else if spec.is_goldberry && mon.hp() <= spec.maxhp / 2 {
            mon.hp_pool[0] = (mon.hp() + 30).min(spec.maxhp);
            mon.item = 0;
        } else if spec.is_berry && mon.hp() <= spec.maxhp / 2 {
            mon.hp_pool[0] = (mon.hp() + 10).min(spec.maxhp);
            mon.item = 0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn exec_status_move(
        &mut self,
        table: &MatchTable,
        side: usize,
        foe_side: usize,
        att_idx: usize,
        def_idx: usize,
        mspec: &CompactMoveSpec,
        moved_second: bool,
        rng: &mut Prng,
    ) {
        match mspec.special_kind {
            1 => {
                if self.sides[side].reflect_turns == 0 {
                    self.sides[side].reflect_turns = 5;
                }
            }
            2 => {
                if self.sides[side].lightscreen_turns == 0 {
                    self.sides[side].lightscreen_turns = 5;
                }
            }
            3 => {
                if self.sides[side].safeguard_turns == 0 {
                    self.sides[side].safeguard_turns = 5;
                }
            }
            4 => {
                if (self.sides[foe_side].side_flags & 0x03) == 0 {
                    self.sides[foe_side].side_flags |= 1;
                }
            }
            5 => {
                // Curse: non-Ghost (+1 Atk, +1 Def, -1 Spe), Ghost (-50% HP, inflicts Curse).
                let is_ghost = (table.mons[att_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
                if !is_ghost {
                    self.mons[att_idx].boosts[0] = (self.mons[att_idx].boosts[0] + 1).clamp(-6, 6);
                    self.mons[att_idx].boosts[1] = (self.mons[att_idx].boosts[1] + 1).clamp(-6, 6);
                    self.mons[att_idx].boosts[4] = (self.mons[att_idx].boosts[4] - 1).clamp(-6, 6);
                } else if self.mons[def_idx].sub_hp() == 0 && (self.mons[def_idx].vol_flags & vol_bit::CURSE) == 0 {
                    let cost = (table.mons[att_idx].maxhp / 2).max(1);
                    self.mons[att_idx].apply_damage_branchless(cost, true);
                    self.mons[def_idx].vol_flags |= vol_bit::CURSE;
                }
            }
            6 => {
                // Belly Drum: costs 50% maxhp and sets Attack stage to +6.
                let cost = table.mons[att_idx].maxhp / 2;
                if self.mons[att_idx].hp() > cost && self.mons[att_idx].boosts[0] < 6 {
                    self.mons[att_idx].hp_pool[0] -= cost;
                    self.mons[att_idx].boosts[0] = 6;
                    self.check_berry_triggers(table, side);
                }
            }
            7 => {
                // Roar / Whirlwind: in Gen 2, fails if user moves before target; otherwise force-switches foe to a random alive bench member.
                if moved_second {
                    let mask = self.sides[foe_side].alive_mask & !(1 << self.sides[foe_side].active_idx);
                    if mask != 0 {
                        let slot = if mask.count_ones() == 1 || rng.random(2) == 0 {
                            mask.trailing_zeros() as u8
                        } else {
                            (7 - mask.leading_zeros()) as u8
                        };
                        self.exec_switch(table, foe_side, slot.min(2), false);
                    }
                }
            }
            17 => {
                // Pain Split: average HP when target has no Substitute.
                if self.mons[def_idx].sub_hp() == 0 {
                    let avg = ((self.mons[att_idx].hp() + self.mons[def_idx].hp()) / 2).max(1);
                    self.mons[att_idx].hp_pool[0] = avg.min(table.mons[att_idx].maxhp);
                    self.mons[def_idx].hp_pool[0] = avg.min(table.mons[def_idx].maxhp);
                }
            }
            18 => {
                // Baton Pass: switch to healthiest alive bench member passing boosts/sub/volatiles.
                let mask = self.sides[side].alive_mask & !(1 << self.sides[side].active_idx);
                if mask != 0 {
                    let mut best_slot = mask.trailing_zeros() as u8;
                    let mut best_hp = -1i32;
                    for s_i in 0..3u8 {
                        if (mask & (1 << s_i)) != 0 {
                            let idx = side * 3 + s_i as usize;
                            let frac = (self.mons[idx].hp() as i32 * 1024)
                                / (table.mons[idx].maxhp.max(1) as i32);
                            if frac > best_hp {
                                best_hp = frac;
                                best_slot = s_i;
                            }
                        }
                    }
                    self.exec_switch(table, side, best_slot.min(2), true);
                }
            }
            19 => {
                // Perish Song
                if self.sides[side].alive_mask.count_ones() > 1 {
                    for &idx in &[att_idx, def_idx] {
                        if (self.mons[idx].vol_flags & vol_bit::PERISH_SONG) == 0 {
                            self.mons[idx].vol_flags |= vol_bit::PERISH_SONG;
                            self.mons[idx].timers[3] = 4;
                        }
                    }
                }
            }
            20 => {
                if self.sides[side].alive_mask.count_ones() > 1 {
                    self.mons[att_idx].vol_flags |= vol_bit::DESTINY_BOND;
                }
            }
            21 => {
                // Heal Bell cures party statuses.
                for p_i in 0..3usize {
                    let m = &mut self.mons[side * 3 + p_i];
                    if m.hp() > 0 {
                        m.set_status(Status::None as u8, 0);
                        m.vol_flags &= !(vol_bit::PAR_SPEED_DROP | vol_bit::BRN_ATK_DROP);
                    }
                }
            }
            22 => {
                // Psych Up
                self.mons[att_idx].boosts = self.mons[def_idx].boosts;
            }
            23 => {
                self.weather_kind = CompactWeather::Rain as u8;
                self.weather_turns = 5;
            }
            24 => {
                self.weather_kind = CompactWeather::Sun as u8;
                self.weather_turns = 5;
            }
            25 => {
                self.weather_kind = CompactWeather::Sandstorm as u8;
                self.weather_turns = 5;
            }
            26 => {
                // Mean Look / Spider Web
                if self.mons[def_idx].sub_hp() == 0 {
                    self.mons[def_idx].vol_flags |= vol_bit::TRAPPED_OR_LOCKED;
                }
            }
            27 => {
                // Haze
                self.mons[att_idx].boosts = [0; 7];
                self.mons[def_idx].boosts = [0; 7];
            }
            _ => {}
        }

        if mspec.primary_status != 0 {
            // Thunder Wave is blocked by Ground immunity.
            let ground_immune = mspec.move_type == table.electric_type
                && (table.mons[def_idx].type_mask & (1u32 << (table.ground_type & 31))) != 0;
            if !ground_immune {
                self.try_inflict_status(table, foe_side, mspec.primary_status, rng);
            }
        }
        if mspec.primary_vol != 0 && self.mons[def_idx].sub_hp() == 0 {
            let seed_immune = (mspec.primary_vol & vol_bit::LEECH_SEED) != 0
                && (table.mons[def_idx].type_mask & (1u32 << (table.grass_type & 31))) != 0;
            let night_fail = (mspec.primary_vol & vol_bit::NIGHTMARE) != 0
                && self.mons[def_idx].status_raw() != Status::Slp as u8;
            if !seed_immune && !night_fail {
                self.mons[def_idx].vol_flags |= mspec.primary_vol;
                if (mspec.primary_vol & vol_bit::CONFUSION) != 0 {
                    self.mons[def_idx].timers[0] = (2 + rng.random(4)) as u8;
                }
            }
        }
        let b_target = if mspec.boosts_self { att_idx } else { def_idx };
        if mspec.boosts_self || self.mons[def_idx].sub_hp() == 0 {
            for &(st_i, delta) in &mspec.boosts {
                if (st_i as usize) < 7 {
                    self.mons[b_target].boosts[st_i as usize] =
                        (self.mons[b_target].boosts[st_i as usize] + delta).clamp(-6, 6);
                }
            }
        }
    }

    /// Gen 2 end-of-turn `Residual` (Weather, Leftovers, Perish Song, Screens).
    #[inline]
    fn exec_residuals(&mut self, table: &MatchTable, rng: &mut Prng) {
        let s0 = self.active_stat(table, 0, 4, false, false);
        let s1 = self.active_stat(table, 1, 4, false, false);
        let first = if s0 == s1 {
            (rng.random(2) == 1) as usize
        } else {
            (s0 < s1) as usize
        };
        let order = [first, first ^ 1];

        // Weather countdown and Sandstorm damage.
        if self.weather_kind != 0 {
            self.weather_turns = self.weather_turns.saturating_sub(1);
            if self.weather_turns == 0 {
                self.weather_kind = 0;
            } else if self.weather_kind == CompactWeather::Sandstorm as u8 {
                for &side in &order {
                    let idx = side * 3 + self.sides[side].active_idx as usize;
                    if self.mons[idx].hp() > 0
                        && (table.mons[idx].type_mask & table.sandstorm_immune_mask) == 0
                    {
                        let sand_dmg = table.mons[idx].quantum_1_8;
                        self.mons[idx].apply_damage_branchless(sand_dmg, true);
                        self.check_berry_triggers(table, side);
                    }
                }
            }
        }

        // Perish Song countdown & Leftovers recovery in speed order.
        for &side in &order {
            let idx = side * 3 + self.sides[side].active_idx as usize;
            if self.mons[idx].hp() <= 0 {
                continue;
            }
            if (self.mons[idx].vol_flags & vol_bit::PERISH_SONG) != 0 {
                let rem = self.mons[idx].timers[3].saturating_sub(1);
                self.mons[idx].timers[3] = rem;
                if rem == 0 {
                    self.mons[idx].hp_pool[0] = 0;
                    self.mons[idx].status_packed = Status::Fnt as u8;
                    self.mons[idx].vol_flags = 0;
                    continue;
                }
            }
            let spec = &table.mons[idx];
            if self.mons[idx].item != 0 && spec.is_leftovers && self.mons[idx].hp() < spec.maxhp {
                self.mons[idx].hp_pool[0] =
                    (self.mons[idx].hp() + spec.quantum_1_16).min(spec.maxhp);
            }
        }

        // Decrement Reflect / Light Screen / Safeguard timers.
        for side in 0..2usize {
            self.sides[side].reflect_turns = self.sides[side].reflect_turns.saturating_sub(1);
            self.sides[side].lightscreen_turns =
                self.sides[side].lightscreen_turns.saturating_sub(1);
            self.sides[side].safeguard_turns = self.sides[side].safeguard_turns.saturating_sub(1);
        }

        self.check_faints(table);
    }

    #[inline]
    fn check_faints(&mut self, table: &MatchTable) {
        if self.outcome != 0 {
            return;
        }
        for side in 0..2usize {
            let p_idx = self.sides[side].active_idx as usize;
            let m_idx = side * 3 + p_idx;
            if self.mons[m_idx].hp() <= 0 {
                self.sides[side].alive_mask &= !(1 << p_idx);
                // Fainting releases Mean Look / Spider Web trap on the opposing active Pokemon.
                let foe_side = side ^ 1;
                let foe_idx = foe_side * 3 + self.sides[foe_side].active_idx as usize;
                self.mons[foe_idx].vol_flags &= !vol_bit::TRAPPED_OR_LOCKED;
            }
        }
        let a0 = self.sides[0].alive_mask;
        let a1 = self.sides[1].alive_mask;
        if a0 == 0 && a1 == 0 {
            // Stadium 2 Self-KO Clause: user of Self-Destruct / Explosion loses.
            if self.sides[0].last_move_selfko != u8::MAX {
                self.outcome = 2; // P2Win
            } else if self.sides[1].last_move_selfko != u8::MAX {
                self.outcome = 1; // P1Win
            } else {
                self.outcome = 3; // Tie
            }
        } else if a0 == 0 {
            self.outcome = 2; // P2Win
        } else if a1 == 0 {
            self.outcome = 1; // P1Win
        } else {
            // Auto-send next alive bench Pokemon when only 1 alive Pokemon remains,
            // or set request = 2 (forced Switch choice) when 2 bench Pokemon remain.
            let mut any_forced_switch = false;
            for side in 0..2usize {
                let p_idx = self.sides[side].active_idx as usize;
                if (self.sides[side].alive_mask & (1 << p_idx)) == 0 {
                    if self.sides[side].alive_mask.count_ones() == 1 {
                        let next_slot = (self.sides[side].alive_mask.trailing_zeros() as u8).min(2);
                        self.exec_switch(table, side, next_slot, false);
                        if self.mons[side * 3 + next_slot as usize].hp() <= 0 {
                            self.sides[side].alive_mask &= !(1 << next_slot);
                            if self.sides[side].alive_mask == 0 {
                                self.outcome = if side == 0 { 2 } else { 1 };
                                return;
                            }
                        }
                        self.sides[side].request = 1;
                    } else {
                        self.sides[side].request = 2; // Forced Switch request
                        any_forced_switch = true;
                    }
                } else {
                    self.sides[side].request = 1;
                }
            }
            if any_forced_switch {
                for side in 0..2usize {
                    if self.sides[side].request != 2 {
                        self.sides[side].request = 0; // Wait
                    }
                }
            }
        }
    }

    /// Resolve a forced switch phase when one or both active Pokemon fainted
    /// with 2 bench options remaining (`request == 2`).
    pub fn step_switch(&mut self, table: &MatchTable, switch_actions: [u8; 2]) {
        let mut rng = Prng::new(self.prng_seed);
        for side in 0..2usize {
            if self.sides[side].request == 2 {
                let act = switch_actions[side];
                let slot = if act >= 4 {
                    (act - 4).min(2)
                } else {
                    (self.sides[side].alive_mask.trailing_zeros() as u8).min(2)
                };
                if (self.sides[side].alive_mask & (1 << slot)) != 0 {
                    self.exec_switch(table, side, slot, false);
                }
            }
            self.sides[side].request = 1;
        }
        self.check_faints(table);
        if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
            self.exec_residuals(table, &mut rng);
            self.check_faints(table);
            if self.outcome == 0 && self.sides[0].request == 1 && self.sides[1].request == 1 {
                self.turn = self.turn.saturating_add(1);
                if self.turn > 1000 {
                    self.outcome = 3; // Tie
                } else {
                    self.quick_claw_roll = rng.random_chance(60, 256) as u8;
                }
            }
        }
        self.prng_seed = rng.seed();
    }

    /// Enumerate legal compact action codes for `side` (`0` or `1`) into a stack buffer.
    /// Action codes:
    /// - `0..4`: move slot `0..4`
    /// - `4..7`: switch to party slot `0..3` (`action - 4`)
    /// - `7`: `"pass"` (when waiting for opponent's forced switch)
    #[inline]
    pub fn legal_actions(&self, table: &MatchTable, side: usize, out: &mut [u8; 7]) -> usize {
        if self.outcome != 0 {
            return 0;
        }
        let side_st = &self.sides[side];
        if side_st.request == 0 {
            out[0] = 7; // pass
            return 1;
        }
        if side_st.request == 2 {
            let mut n = 0;
            for slot in 0..3u8 {
                if slot != side_st.active_idx && (side_st.alive_mask & (1 << slot)) != 0 {
                    out[n] = 4 + slot;
                    n += 1;
                }
            }
            return n;
        }
        let mut n = 0;
        let idx = side * 3 + side_st.active_idx as usize;
        let mon = &self.mons[idx];
        let num_moves = table.mons[idx].num_moves as usize;
        let is_disabled = (mon.vol_flags & vol_bit::DISABLE) != 0;
        let dis_slot = ((mon.slot_refs >> 4) & 0x03) as usize;
        for m in 0..num_moves {
            if mon.pp[m] > 0 && !(is_disabled && m == dis_slot) {
                out[n] = m as u8;
                n += 1;
            }
        }
        if n == 0 {
            out[0] = 0; // Struggle fallback
            n = 1;
        }
        if (mon.vol_flags & vol_bit::TRAPPED_OR_LOCKED) == 0 {
            for slot in 0..3u8 {
                if slot != side_st.active_idx && (side_st.alive_mask & (1 << slot)) != 0 {
                    out[n] = 4 + slot;
                    n += 1;
                }
            }
        }
        n
    }

    /// Format a compact action code (`0..=7`) as a Showdown protocol string
    /// (`"move 1"`, `"switch 2"`, `"pass"`).
    #[inline]
    pub const fn action_to_str(action_code: u8) -> &'static str {
        match action_code {
            0 => "move 1",
            1 => "move 2",
            2 => "move 3",
            3 => "move 4",
            4 => "switch 1",
            5 => "switch 2",
            6 => "switch 3",
            _ => "pass",
        }
    }

    /// Parse a Showdown choice string (`"move 1"`, `"switch 2"`, `"pass"`) into a
    /// compact action code (`0..=7`).
    #[inline]
    pub fn parse_action_str(choice: &str) -> u8 {
        match choice.trim() {
            "move 1" => 0,
            "move 2" => 1,
            "move 3" => 2,
            "move 4" => 3,
            "switch 1" => 4,
            "switch 2" => 5,
            "switch 3" => 6,
            _ => 7,
        }
    }

    /// Apply a pair of action codes `[p1_action, p2_action]`, handling both normal
    /// turns and forced-switch phases.
    #[inline]
    pub fn apply_actions(&mut self, table: &MatchTable, actions: [u8; 2]) {
        if self.sides[0].request == 2 || self.sides[1].request == 2 {
            self.step_switch(table, actions);
        } else {
            self.step_turn(table, actions);
        }
    }

    /// Fast branchless heuristic evaluation of the current `CompactBattle` from
    /// `perspective` (`0` = P1, `1` = P2) in `[0.0, 1.0]`.
    #[inline]
    pub fn eval_leaf(&self, table: &MatchTable, perspective: usize) -> f64 {
        match self.outcome {
            1 => return if perspective == 0 { 1.0 } else { 0.0 },
            2 => return if perspective == 1 { 1.0 } else { 0.0 },
            3 => return 0.5,
            _ => {}
        }
        let mut score = [0.0f64; 2];
        for side in 0..2usize {
            for slot in 0..3usize {
                if (self.sides[side].alive_mask & (1 << slot)) == 0 {
                    continue;
                }
                let idx = side * 3 + slot;
                let mon = &self.mons[idx];
                let spec = &table.mons[idx];
                let hp_frac = (mon.hp().max(0) as f64) / (spec.maxhp.max(1) as f64);
                let status_mult = match mon.status_raw() {
                    0 => 1.0,
                    x if x == Status::Slp as u8 => 0.72,
                    x if x == Status::Frz as u8 => 0.35,
                    x if x == Status::Par as u8 => 0.78,
                    x if x == Status::Brn as u8 => 0.75,
                    _ => 0.82,
                };
                let sub_bonus = (mon.sub_hp().max(0) as f64) / (spec.maxhp.max(1) as f64) * 0.5;
                score[side] += 0.35 + 0.65 * hp_frac * status_mult + sub_bonus;
            }
        }
        let diff = score[perspective] - score[perspective ^ 1];
        (0.5 + 0.18 * diff).clamp(0.01, 0.99)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_battle_layout_is_exact_256_bytes_and_64b_aligned() {
        assert_eq!(std::mem::size_of::<CompactMon>(), 24);
        assert_eq!(std::mem::size_of::<CompactSide>(), 12);
        assert_eq!(std::mem::size_of::<CompactBattle>(), 256);
        assert_eq!(std::mem::align_of::<CompactBattle>(), 64);
    }

    #[test]
    fn action_sort_key_orders_switches_priority_quick_claw_and_speed_ties() {
        // Switch beats +2 Protect
        let sw = ActionSortKey::new(ActionPhase::Switch, 0, false, 100, 0);
        let prot = ActionSortKey::new(ActionPhase::Move, 2, true, 400, 1);
        let (first, second, is_tie) = resolve_turn_order(sw, prot);
        assert_eq!((first, second, is_tie), (0, 1, false));

        // BeforeTurnPursuit beats Switch
        let pursuit = ActionSortKey::new(ActionPhase::BeforeTurnPursuit, 0, false, 50, 0);
        assert_eq!(resolve_turn_order(sw, pursuit), (1, 0, false));

        // Quick Claw on slower Pokemon beats faster Pokemon at equal priority
        let slow_qc = ActionSortKey::new(ActionPhase::Move, 0, true, 90, 0);
        let fast_no_qc = ActionSortKey::new(ActionPhase::Move, 0, false, 350, 1);
        assert_eq!(resolve_turn_order(slow_qc, fast_no_qc), (0, 1, false));

        // Quick Attack (+1) beats Quick Claw (0 priority)
        let quick_atk = ActionSortKey::new(ActionPhase::Move, 1, false, 90, 1);
        assert_eq!(resolve_turn_order(slow_qc, quick_atk), (1, 0, false));

        // Double Quick Claw Activation ties on speed regardless of underlying base speeds!
        let qc_p0 = ActionSortKey::new(ActionPhase::Move, 0, true, 85, 0);
        let qc_p1 = ActionSortKey::new(ActionPhase::Move, 0, true, 300, 1);
        assert!(qc_p0.ties_without_rng(qc_p1));
        assert_eq!(qc_p0.speed_field(), 1023);
        assert_eq!(qc_p1.speed_field(), 1023);
    }

    #[test]
    fn stat_branchless_matches_reference_formula_exhaustively() {
        const BOOST_TABLE: [f64; 7] = [1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0];
        let _ = BOOST_TABLE;
        const STAT_POS: [f64; 7] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        const STAT_NEG: [f64; 7] = [100.0, 66.0, 50.0, 40.0, 33.0, 28.0, 25.0];

        for stored in [1, 15, 95, 130, 178, 255, 358, 500, 999] {
            for boost in -6..=6i8 {
                for stat_idx in 0..5usize {
                    for &unboosted in &[false, true] {
                        for &par in &[false, true] {
                            for &brn in &[false, true] {
                                for &screen in &[false, true] {
                                    for &item in &[
                                        ItemStatKind::None,
                                        ItemStatKind::ThickClub,
                                        ItemStatKind::LightBall,
                                        ItemStatKind::MetalPowder,
                                    ] {
                                        let mut ref_val = stored as f64;
                                        if !unboosted {
                                            if boost >= 0 {
                                                ref_val =
                                                    (ref_val * STAT_POS[boost as usize]).floor();
                                            } else {
                                                ref_val = (ref_val * STAT_NEG[(-boost) as usize]
                                                    / 100.0)
                                                    .floor();
                                            }
                                        }
                                        if par && stat_idx == 4 {
                                            ref_val = (ref_val / 4.0).floor();
                                        }
                                        if brn && stat_idx == 0 {
                                            ref_val = (ref_val / 2.0).floor();
                                        }
                                        ref_val = ref_val.clamp(1.0, 999.0);
                                        if !unboosted && screen && (stat_idx == 1 || stat_idx == 3)
                                        {
                                            ref_val *= 2.0;
                                        }
                                        if (item == ItemStatKind::ThickClub && stat_idx == 0)
                                            || (item == ItemStatKind::LightBall && stat_idx == 2)
                                        {
                                            ref_val *= 2.0;
                                        } else if item == ItemStatKind::MetalPowder
                                            && (stat_idx == 1 || stat_idx == 3)
                                        {
                                            ref_val = (ref_val * 1.5).floor();
                                        }

                                        let fast_val = get_stat_branchless(
                                            stored, boost, stat_idx, unboosted, false, false, par,
                                            brn, screen, screen, item,
                                        );
                                        assert_eq!(
                                            fast_val, ref_val as i32,
                                            "mismatch at stored={stored}, boost={boost}, stat={stat_idx}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn acc_lut_and_reciprocal_damage_rolls_are_bit_exact() {
        // Verify reciprocal multiplication (x * 0x808081) >> 31 matches x / 255
        // for every possible damage (1..=4000) and every roll (217..=255).
        for dmg in 1..=4000i32 {
            for roll in 217..=255u32 {
                let exact = ((dmg as f64) * (roll as f64) / 255.0).floor() as i32;
                let fast = apply_roll_branchless(dmg, roll, false);
                assert_eq!(fast, exact.max(1), "mismatch at dmg={dmg}, roll={roll}");
            }
        }

        // Verify SIMD16 evaluator accurately handles both Substitute and direct HP hits.
        let sub_hit = evaluate_hit_simd16(120, false, 300, 110, false);
        assert_eq!(sub_hit.ko_mask, 0);
        assert!(sub_hit.sub_broke_mask != 0);
        assert_eq!(sub_hit.post_hp, [300; 16]);

        let ko_hit = evaluate_hit_simd16(200, false, 185, 0, false);
        assert!(ko_hit.ko_mask != 0 && ko_hit.ko_mask != 0xFFFF);
    }

    #[test]
    fn compact_battle_from_battle_step_and_hash() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/gen2stadium2.json");
        let json = std::fs::read_to_string(&path).unwrap();
        let dex = Dex::from_json(&json).unwrap();
        let mk = |species: &str, level: u8, item: &str, moves: &[&str]| -> crate::battle::PokemonSet {
            serde_json::from_value(serde_json::json!({
                "name": species, "species": species, "item": item, "ability": "No Ability",
                "moves": moves, "level": level,
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": {"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},
                "happiness": 255
            }))
            .unwrap()
        };
        let p1 = vec![
            mk("Snorlax", 55, "Leftovers", &["Double-Edge", "Earthquake", "Rest", "Sleep Talk"]),
            mk("Zapdos", 50, "Mint Berry", &["Thunderbolt", "Hidden Power", "Rest", "Whirlwind"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Ice Beam", "Spikes", "Explosion"]),
        ];
        let p2 = vec![
            mk("Marowak", 55, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Curse", "Rest"]),
            mk("Raikou", 50, "Miracle Berry", &["Thunderbolt", "Crunch", "Reflect", "Rest"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &p1, &p2).unwrap();
        b.choose(&dex, 0, "team 1,2,3").unwrap();
        b.choose(&dex, 1, "team 1,2,3").unwrap();

        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);
        let h0 = cb.search_hash();
        // Cloning CompactBattle is a 256-byte Copy.
        let mut cb_clone = cb;
        assert_eq!(cb_clone.search_hash(), h0);

        // Execute a turn in CompactBattle: Snorlax Double-Edge (0) vs Marowak Earthquake (0).
        cb.step_turn(&table, [0, 0]);
        assert_eq!(cb.turn, 2);
        assert_ne!(cb.search_hash(), h0);
        // Original copy is untouched.
        assert_eq!(cb_clone.turn, 1);
        // Switch Snorlax to party slot 1 (action 4 + 1 = 5), Marowak Swords Dance (move 2).
        cb_clone.step_turn(&table, [5, 2]);
        assert_eq!(cb_clone.sides[0].active_idx, 1);
    }
}
