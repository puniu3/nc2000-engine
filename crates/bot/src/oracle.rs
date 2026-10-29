//! External game-theoretic board equity oracle (`E*(s) in [0, 1]`) for
//! ceiling / headroom experiments on `nc2000-bot`.
//!
//! Design invariants:
//! 1. **Zero hidden-information or future-RNG leakage**:
//!    - Operates exclusively on `(cb: &CompactBattle, table: &MatchTable)` —
//!      the exact same determinized board representation passed to
//!      [`crate::eval::compact_eval_leaf`]. In `Blind` play, hidden opponent
//!      sets come only from `Belief`'s determinized `Battle`.
//!    - Ignores `cb.prng_seed` and `cb.quick_claw_roll` completely, reseeding
//!      both on every internal simulation from a canonical state-derived seed
//!      (`quick_claw_roll` drawn with exact `60 / 256` Gen 2 Stadium 2 odds).
//! 2. **Simultaneous-move RM+ stage-game equilibrium extraction**:
//!    - Rather than averaging exploration-polluted UCB visits at the root,
//!      `CompactOracleSearch` accumulates linear-iteration-weighted joint root
//!      payoffs over the top-visited actions and solves the stage game via
//!      Regret Matching+ (`RM+`) with dust purification. This eliminates UCB
//!      exploration drag on forced wins/losses AND extracts mixed equilibrium
//!      rollout policies for [`OracleMode::Coupled`] from a **single** cached
//!      oracle evaluation per state (preventing deterministic switch ping-pong).
//! 3. **Bit-exact cache invariance (`0.0 ULP`) & exact side-swap antisymmetry**:
//!    - Canonical 128-bit state digest `(k_lo, k_hi)` canonicalizes `(side 0,
//!      side 1)` order via 128-bit per-side digests `d0` vs `d1` and stores
//!      `(val_f32, top2_actions, top_prob)` in a fixed `262,144`-slot
//!      thread-local direct-mapped cache (~8 MB/thread).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use nc2000_engine::compact::{CompactBattle, MatchTable};

use crate::eval::{self, EvalWeights};
use crate::mcts::{self, Playout, RolloutRules};
use crate::rng::SplitMix64;

/// How (if at all) the external equity oracle participates in search.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OracleMode {
    /// Disabled (default): 100% bit-identical to baseline `nc2000-bot`.
    #[default]
    Off,
    /// Null-control seam test: computes the canonical 128-bit oracle state key
    /// and exercises the cache path, then returns the exact baseline
    /// `eval::compact_eval_leaf` value. Must tie `Off` at `0.500` under CRN.
    EchoBase,
    /// `Oracle-Cutoff8`: keeps the shipped rollout policy unchanged and
    /// evaluates truncated rollout cutoff leaves (`cb.turn > cutoff`) with
    /// `oracle_equity(cb, table, leaf_iters)`.
    Cutoff { leaf_iters: u32 },
    /// `Oracle-TreeLeaf0`: evaluates `oracle_equity(cb, table, leaf_iters)`
    /// directly at the post-preview MCTS tree leaf (`0` rollout turns).
    TreeLeaf { leaf_iters: u32 },
    /// `Oracle-Coupled8`: `mtg_ai`-style conservation-coupled rollout + cutoff.
    /// Samples rollout actions in the greedy branch from the oracle's RM+
    /// stage-game equilibrium at `(cb, table)` (`rollout_iters`) and evaluates
    /// rollout cutoff leaves with `oracle_equity(..., leaf_iters)`.
    Coupled { rollout_iters: u32, leaf_iters: u32 },
}

/// Process-wide telemetry counters for oracle evaluations and rollout picks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OracleStats {
    pub eval_calls: u64,
    pub eval_hits: u64,
    pub eval_misses: u64,
    pub pick_calls: u64,
    pub pick_hits: u64,
    pub pick_misses: u64,
}

static STAT_EVAL_CALLS: AtomicU64 = AtomicU64::new(0);
static STAT_EVAL_HITS: AtomicU64 = AtomicU64::new(0);
static STAT_EVAL_MISSES: AtomicU64 = AtomicU64::new(0);
static STAT_PICK_CALLS: AtomicU64 = AtomicU64::new(0);
static STAT_PICK_HITS: AtomicU64 = AtomicU64::new(0);
static STAT_PICK_MISSES: AtomicU64 = AtomicU64::new(0);

pub fn reset_oracle_stats() {
    STAT_EVAL_CALLS.store(0, Ordering::Relaxed);
    STAT_EVAL_HITS.store(0, Ordering::Relaxed);
    STAT_EVAL_MISSES.store(0, Ordering::Relaxed);
    STAT_PICK_CALLS.store(0, Ordering::Relaxed);
    STAT_PICK_HITS.store(0, Ordering::Relaxed);
    STAT_PICK_MISSES.store(0, Ordering::Relaxed);
}

pub fn oracle_stats() -> OracleStats {
    OracleStats {
        eval_calls: STAT_EVAL_CALLS.load(Ordering::Relaxed),
        eval_hits: STAT_EVAL_HITS.load(Ordering::Relaxed),
        eval_misses: STAT_EVAL_MISSES.load(Ordering::Relaxed),
        pick_calls: STAT_PICK_CALLS.load(Ordering::Relaxed),
        pick_hits: STAT_PICK_HITS.load(Ordering::Relaxed),
        pick_misses: STAT_PICK_MISSES.load(Ordering::Relaxed),
    }
}

const EVAL_CACHE_SLOTS: usize = 1 << 18; // 262,144 slots (8 MB / thread)
const SUBGAME_TABLE_SLOTS: usize = 1 << 11; // 2,048 open-addressed slots

/// Cached canonical stage-game solution (32 bytes, 8-byte aligned).
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct EvalCacheEntry {
    k_lo: u64,
    k_hi: u64,
    val: f32,
    /// Top-2 purified RM+ equilibrium actions `[a0, a1]` per canonical side.
    acts: [[u8; 2]; 2],
    /// Probability of `acts[side][0]` quantized to `0..=255` (`255` = pure `acts[side][0]`).
    prob0: [u8; 2],
    _pad: [u8; 6],
}

const _: () = assert!(std::mem::size_of::<EvalCacheEntry>() == 32);

#[derive(Clone, Copy, Default)]
struct SubTableSlot {
    key: u64,
    epoch: u32,
    node_idx: u32,
}

#[derive(Clone, Copy)]
struct OracleNode {
    acts: [[u8; 7]; 2],
    len: [u8; 2],
    n: [[u32; 7]; 2],
    w: [[f64; 7]; 2],
}

struct OracleThreadCtx {
    eval_cache: Box<[EvalCacheEntry]>,
    sub_table: Box<[SubTableSlot]>,
    nodes: Vec<OracleNode>,
    epoch: u32,
}

impl OracleThreadCtx {
    fn new() -> Self {
        Self {
            eval_cache: vec![EvalCacheEntry::default(); EVAL_CACHE_SLOTS].into_boxed_slice(),
            sub_table: vec![SubTableSlot::default(); SUBGAME_TABLE_SLOTS].into_boxed_slice(),
            nodes: Vec::with_capacity(512),
            epoch: 1,
        }
    }

    fn clear_caches(&mut self) {
        self.eval_cache.fill(EvalCacheEntry::default());
    }
}

thread_local! {
    static ORACLE_CTX: RefCell<OracleThreadCtx> = RefCell::new(OracleThreadCtx::new());
}

/// Clear the calling thread's direct-mapped oracle cache (used by cold-vs-warm
/// bit-exactness tests).
pub fn clear_oracle_thread_caches() {
    ORACLE_CTX.with(|cell| cell.borrow_mut().clear_caches());
}

#[inline(always)]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline(always)]
fn absorb128(a: &mut u64, b: &mut u64, word: u64) {
    *a = mix64(a.wrapping_add(word).wrapping_add(0x9E37_79B9_7F4A_7C15));
    *b = mix64(b.wrapping_add(word.rotate_left(29) ^ *a).wrapping_add(0xD1B5_4A32_D192_ED03));
}

/// Compute a 128-bit digest of one side's game-relevant state on `(cb, table)`.
#[inline]
fn side_digest(cb: &CompactBattle, table: &MatchTable, side: usize) -> (u64, u64) {
    let mut a = 0x243F_6A88_85A3_08D3u64;
    let mut b = 0x1319_8A2E_0370_7344u64;
    let s = &cb.sides[side];

    let side_word = (s.active_idx as u64)
        | ((s.alive_mask as u64) << 8)
        | ((s.reflect_turns as u64) << 16)
        | ((s.lightscreen_turns as u64) << 24)
        | ((s.safeguard_turns as u64) << 32)
        | (((s.side_flags & 0x7F) as u64) << 40)
        | ((s.last_move_selfko as u64) << 48)
        | ((s.request as u64) << 56);
    absorb128(&mut a, &mut b, side_word);

    let hurt_word =
        (cb.hurt_damage_this_turn[side] as u64) | ((cb.hurt_cat_this_turn[side] as u64) << 16);
    absorb128(&mut a, &mut b, hurt_word);

    for slot in 0..3usize {
        let idx = side * 3 + slot;
        let mon = &cb.mons[idx];
        if (s.alive_mask & (1 << slot)) == 0 || mon.hp() <= 0 {
            absorb128(&mut a, &mut b, 0xDEAD_BEEF_0000_0000u64 | (slot as u64));
            continue;
        }
        let spec = &table.mons[idx];
        absorb128(&mut a, &mut b, spec.static_fp);

        let hp_pp = (mon.hp() as u16 as u64)
            | ((mon.sub_hp().max(0) as u16 as u64) << 16)
            | ((u32::from_le_bytes(mon.pp) as u64) << 32);
        absorb128(&mut a, &mut b, hp_pp);

        let st_tm = (mon.status_packed as u64)
            | ((mon.item as u64) << 8)
            | ((mon.vol_flags as u64) << 16)
            | ((u32::from_le_bytes(mon.timers) as u64) << 32);
        absorb128(&mut a, &mut b, st_tm);

        let mut boost_bytes = [0u8; 8];
        for i in 0..7usize {
            boost_bytes[i] = mon.boosts[i] as u8;
        }
        boost_bytes[7] = mon.slot_refs;
        absorb128(&mut a, &mut b, u64::from_le_bytes(boost_bytes));
    }

    (a, b)
}

#[inline]
fn global_field_word(cb: &CompactBattle, iters: u32) -> u64 {
    let bp_disabled = ((cb.sides[0].side_flags & 0x80) != 0) as u64;
    let near_cap_turn = if cb.turn >= 950 { cb.turn as u64 } else { 0u64 };
    (cb.weather_kind as u64)
        | ((cb.weather_turns as u64) << 8)
        | ((cb.clause_flags as u64) << 16)
        | (bp_disabled << 24)
        | (near_cap_turn << 25)
        | ((iters as u64) << 40)
}

#[inline]
fn finalize_non_zero_key(mut lo: u64, mut hi: u64) -> (u64, u64) {
    if lo == 0 && hi == 0 {
        lo = 1;
        hi = 1;
    }
    (lo, hi)
}

/// Swap side 0 and side 1 on a `(CompactBattle, MatchTable)` pair while
/// preserving global flags (`sides[0].side_flags & 0x80`, `clause_flags`,
/// `weather_*`).
pub fn swap_compact_sides(cb: &CompactBattle, table: &MatchTable) -> (CompactBattle, MatchTable) {
    let mut scb = *cb;
    scb.mons = [
        cb.mons[3], cb.mons[4], cb.mons[5], cb.mons[0], cb.mons[1], cb.mons[2],
    ];
    let bp_bit = cb.sides[0].side_flags & 0x80;
    let mut s0 = cb.sides[1];
    let mut s1 = cb.sides[0];
    s0.side_flags = (s0.side_flags & 0x7F) | bp_bit;
    s1.side_flags &= 0x7F;
    scb.sides = [s0, s1];
    scb.outcome = match cb.outcome {
        1 => 2,
        2 => 1,
        other => other,
    };
    scb.speed_order = if cb.speed_order <= 1 {
        cb.speed_order ^ 1
    } else {
        cb.speed_order
    };
    scb.hurt_damage_this_turn = [cb.hurt_damage_this_turn[1], cb.hurt_damage_this_turn[0]];
    scb.hurt_cat_this_turn = [cb.hurt_cat_this_turn[1], cb.hurt_cat_this_turn[0]];

    let mut st = table.clone();
    st.mons = [
        table.mons[3],
        table.mons[4],
        table.mons[5],
        table.mons[0],
        table.mons[1],
        table.mons[2],
    ];
    st.move_specs = [
        table.move_specs[3],
        table.move_specs[4],
        table.move_specs[5],
        table.move_specs[0],
        table.move_specs[1],
        table.move_specs[2],
    ];
    let mut fp = 0xcbf2_9ce4_8422_2325u64;
    for m in &st.mons {
        fp = fp.wrapping_add(m.static_fp).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }
    st.match_fingerprint = fp;
    scb.match_fp_lo = fp as u16;
    (scb, st)
}

/// Exercise the canonical-key + cache lookup seam for [`OracleMode::EchoBase`]
/// and return the exact baseline leaf value.
pub fn echo_base_leaf(cb: &CompactBattle, table: &MatchTable, weights: &EvalWeights) -> f64 {
    STAT_EVAL_CALLS.fetch_add(1, Ordering::Relaxed);
    let d0 = side_digest(cb, table, 0);
    let d1 = side_digest(cb, table, 1);
    let (c0, c1) = if d0 <= d1 { (d0, d1) } else { (d1, d0) };
    let mut lo = c0.0;
    let mut hi = c0.1;
    absorb128(&mut lo, &mut hi, c1.0);
    absorb128(&mut lo, &mut hi, c1.1);
    absorb128(&mut lo, &mut hi, global_field_word(cb, 0));
    let (k_lo, k_hi) = finalize_non_zero_key(lo, hi);
    ORACLE_CTX.with(|cell| {
        let ctx = cell.borrow();
        let slot = &ctx.eval_cache[(k_lo as usize) & (EVAL_CACHE_SLOTS - 1)];
        if slot.k_lo == k_lo && slot.k_hi == k_hi {
            STAT_EVAL_HITS.fetch_add(1, Ordering::Relaxed);
        } else {
            STAT_EVAL_MISSES.fetch_add(1, Ordering::Relaxed);
        }
    });
    eval::compact_eval_leaf(cb, table, weights)
}

/// Solve or fetch the cached canonical stage-game solution for `(cb, table, iters)`.
/// Returns `(swapped, EvalCacheEntry)` where `swapped` indicates whether side 0
/// and side 1 were transposed into canonical order (`d0 > d1`).
#[inline]
fn query_oracle_entry(
    cb: &CompactBattle,
    table: &MatchTable,
    iters: u32,
    is_pick: bool,
) -> (bool, EvalCacheEntry) {
    if is_pick {
        STAT_PICK_CALLS.fetch_add(1, Ordering::Relaxed);
    } else {
        STAT_EVAL_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    let d0 = side_digest(cb, table, 0);
    let d1 = side_digest(cb, table, 1);
    let swapped = d0 > d1;
    let (c0, c1) = if !swapped { (d0, d1) } else { (d1, d0) };

    let mut lo = c0.0;
    let mut hi = c0.1;
    absorb128(&mut lo, &mut hi, c1.0);
    absorb128(&mut lo, &mut hi, c1.1);
    absorb128(&mut lo, &mut hi, global_field_word(cb, iters));
    let (k_lo, k_hi) = finalize_non_zero_key(lo, hi);
    let slot_idx = (k_lo as usize) & (EVAL_CACHE_SLOTS - 1);

    let cached: Option<EvalCacheEntry> = ORACLE_CTX.with(|cell| {
        let ctx = cell.borrow();
        let entry = &ctx.eval_cache[slot_idx];
        if entry.k_lo == k_lo && entry.k_hi == k_hi {
            Some(*entry)
        } else {
            None
        }
    });

    if let Some(mut entry) = cached {
        if d0 == d1 {
            entry.val = 0.5;
        }
        if is_pick {
            STAT_PICK_HITS.fetch_add(1, Ordering::Relaxed);
        } else {
            STAT_EVAL_HITS.fetch_add(1, Ordering::Relaxed);
        }
        return (swapped, entry);
    }

    if is_pick {
        STAT_PICK_MISSES.fetch_add(1, Ordering::Relaxed);
    } else {
        STAT_EVAL_MISSES.fetch_add(1, Ordering::Relaxed);
    }

    let sym_seed = mix64(k_lo ^ k_hi.rotate_left(23));
    let (raw_val, acts, prob0) = if swapped {
        let (scb, st) = swap_compact_sides(cb, table);
        run_oracle_uncached(&scb, &st, sym_seed, iters)
    } else {
        run_oracle_uncached(cb, table, sym_seed, iters)
    };
    let val_f32 = if d0 == d1 {
        0.5f32
    } else {
        (raw_val as f32).clamp(0.0, 1.0)
    };
    let entry = EvalCacheEntry {
        k_lo,
        k_hi,
        val: val_f32,
        acts,
        prob0,
        _pad: [0; 6],
    };
    ORACLE_CTX.with(|cell| {
        let mut ctx = cell.borrow_mut();
        ctx.eval_cache[slot_idx] = entry;
    });
    (swapped, entry)
}

/// Evaluate the game-theoretic board equity `E*(cb) in [0.0, 1.0]` from side 0's
/// perspective using `iters` iterations of state-keyed simultaneous-move UCB +
/// root RM+ stage-game equilibrium on the determinized `CompactBattle` kernel.
pub fn oracle_equity(cb: &CompactBattle, table: &MatchTable, iters: u32) -> f64 {
    match cb.outcome {
        1 => return 1.0,
        2 => return 0.0,
        3 => return 0.5,
        _ => {}
    }
    let a0 = cb.sides[0].alive_mask;
    let a1 = cb.sides[1].alive_mask;
    if a0 == 0 && a1 == 0 {
        return 0.5;
    }
    if a0 == 0 {
        return 0.0;
    }
    if a1 == 0 {
        return 1.0;
    }
    if cb.sides[0].request == 3 || cb.sides[1].request == 3 || iters == 0 {
        return eval::compact_eval_leaf(cb, table, &EvalWeights::default());
    }

    let (swapped, entry) = query_oracle_entry(cb, table, iters, false);
    let canon_f64 = entry.val as f64;
    if swapped {
        1.0 - canon_f64
    } else {
        canon_f64
    }
}

/// Conservation-coupled rollout action picker (`OracleMode::Coupled`):
/// queries the cached simultaneous-move RM+ equilibrium at `(cb, table, rollout_iters)`
/// and samples from the purified top-2 equilibrium support for `side`.
/// Note: when both sides pick on the same turn of a rollout, the second side's
/// query is a guaranteed 100% L1 cache hit!
pub fn oracle_rollout_pick(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    cs: &[u8],
    rollout_iters: u32,
    rng: &mut SplitMix64,
) -> u8 {
    if cs.is_empty() {
        return 7;
    }
    if cs.len() == 1 {
        return cs[0];
    }

    let last_mon = cb.sides[side].alive_mask.count_ones() == 1;
    let att_idx = side * 3 + (cb.sides[side].active_idx.min(2) as usize);
    let mut safe = [0u8; 7];
    let mut n_safe = 0usize;
    for &c in cs {
        let suicidal = c < 4 && last_mon && table.move_specs[att_idx][c as usize].is_selfdestruct;
        if !suicidal && n_safe < 7 {
            safe[n_safe] = c;
            n_safe += 1;
        }
    }
    if n_safe == 0 {
        return cs[0];
    }
    if n_safe == 1 {
        return safe[0];
    }

    let (swapped, entry) = query_oracle_entry(cb, table, rollout_iters, true);
    let canon_side = if swapped { side ^ 1 } else { side };
    let a0 = entry.acts[canon_side][0];
    let a1 = entry.acts[canon_side][1];
    let p0 = entry.prob0[canon_side];

    let chosen = if p0 == 255 || a0 == a1 {
        a0
    } else if ((rng.next() & 0xFF) as u8) < p0 {
        a0
    } else {
        a1
    };

    if safe[..n_safe].contains(&chosen) {
        return chosen;
    }
    if safe[..n_safe].contains(&a0) {
        return a0;
    }
    if safe[..n_safe].contains(&a1) {
        return a1;
    }
    safe[0]
}

/// Bucketed internal node hash (`hp_b16` + discrete state + `turn`) for the
/// oracle's local state-keyed UCB search tree.
#[inline]
fn oracle_subgame_node_key(cb: &CompactBattle, table: &MatchTable) -> u64 {
    let mut h = (cb.turn as u64)
        .wrapping_add((cb.weather_kind as u64) << 16)
        .wrapping_add((cb.weather_turns as u64) << 24)
        .wrapping_add((cb.clause_flags as u64) << 32)
        .wrapping_mul(0x9E37_79B9_7F4A_7C15);

    for side in 0..2usize {
        let s = &cb.sides[side];
        let sw = (s.active_idx as u64)
            | ((s.alive_mask as u64) << 8)
            | ((s.reflect_turns as u64) << 16)
            | ((s.lightscreen_turns as u64) << 24)
            | ((s.safeguard_turns as u64) << 32)
            | (((s.side_flags & 0x7F) as u64) << 40)
            | ((s.request as u64) << 48)
            | ((cb.hurt_cat_this_turn[side] as u64) << 56);
        h = mix64(h.wrapping_add(sw));

        for slot in 0..3usize {
            let idx = side * 3 + slot;
            let mon = &cb.mons[idx];
            if (s.alive_mask & (1 << slot)) == 0 || mon.hp() <= 0 {
                h = mix64(h.wrapping_add(0xDEAD_0000 + idx as u64));
                continue;
            }
            let maxhp = table.mons[idx].maxhp.max(1) as i32;
            let hp = mon.hp().max(1) as i32;
            let hp_b16 = ((hp * 16 + maxhp - 1) / maxhp).clamp(1, 16) as u64;
            let thr_bits = (((hp * 2 > maxhp) as u64) << 5) | (((hp * 4 > maxhp) as u64) << 6);
            let sub_b8 = if mon.sub_hp() > 0 {
                (((mon.sub_hp() as i32 * 8 + maxhp - 1) / maxhp).clamp(1, 8)) as u64
            } else {
                0u64
            };
            let pp_b = (mon.pp[0].min(8) as u64)
                | ((mon.pp[1].min(8) as u64) << 4)
                | ((mon.pp[2].min(8) as u64) << 8)
                | ((mon.pp[3].min(8) as u64) << 12);
            let w0 = hp_b16
                | thr_bits
                | (sub_b8 << 8)
                | ((mon.status_packed as u64) << 16)
                | ((mon.item as u64) << 24)
                | ((mon.vol_flags as u64) << 32)
                | (pp_b << 48);
            h = mix64(h.wrapping_add(w0));

            let mut bb = [0u8; 8];
            for i in 0..7usize {
                bb[i] = mon.boosts[i] as u8;
            }
            bb[7] = mon.slot_refs ^ mon.timers[3].wrapping_shl(2);
            h = mix64(h.wrapping_add(u64::from_le_bytes(bb)));
        }
    }
    if h == 0 { 1 } else { h }
}

impl OracleNode {
    #[inline]
    fn from_cb(cb: &CompactBattle, table: &MatchTable) -> Self {
        let mut acts = [[7u8; 7]; 2];
        let mut len = [0u8; 2];

        for side in 0..2usize {
            if cb.sides[side].request == 0 {
                continue;
            }
            let mut raw = [0u8; 7];
            let n_raw = cb.legal_actions(table, side, &mut raw);
            if n_raw == 0 {
                continue;
            }
            let last_mon = cb.sides[side].alive_mask.count_ones() == 1;
            let att_idx = side * 3 + (cb.sides[side].active_idx.min(2) as usize);
            let mut n_keep = 0usize;
            if last_mon {
                for &c in &raw[..n_raw] {
                    let is_self_ko =
                        c < 4 && table.move_specs[att_idx][c as usize].is_selfdestruct;
                    if !is_self_ko {
                        acts[side][n_keep] = c;
                        n_keep += 1;
                    }
                }
            }
            if n_keep == 0 {
                acts[side][..n_raw].copy_from_slice(&raw[..n_raw]);
                n_keep = n_raw;
            }
            len[side] = n_keep as u8;
        }

        Self {
            acts,
            len,
            n: [[0; 7]; 2],
            w: [[0.0; 7]; 2],
        }
    }
}

#[inline]
fn select_oracle_ucb(node: &OracleNode, side: usize, c_ucb: f64, rng: &mut SplitMix64) -> usize {
    let k = node.len[side] as usize;
    if k <= 1 {
        return 0;
    }
    let mut total_n = 0u32;
    let mut unvisited = [0usize; 7];
    let mut n_unvisited = 0usize;
    for i in 0..k {
        let ni = node.n[side][i];
        if ni == 0 {
            unvisited[n_unvisited] = i;
            n_unvisited += 1;
        }
        total_n += ni;
    }
    if n_unvisited > 0 {
        return unvisited[rng.below(n_unvisited)];
    }
    let ln_total = (total_n as f64).ln();
    let mut best_i = 0usize;
    let mut best_ucb = f64::NEG_INFINITY;
    for i in 0..k {
        let ni = node.n[side][i] as f64;
        let mean = node.w[side][i] / ni;
        let ucb = mean + c_ucb * (ln_total / ni).sqrt();
        if ucb > best_ucb {
            best_ucb = ucb;
            best_i = i;
        }
    }
    best_i
}

#[inline]
fn sub_table_lookup_or_insert(
    table: &mut [SubTableSlot],
    epoch: u32,
    key: u64,
    nodes: &mut Vec<OracleNode>,
    cb: &CompactBattle,
    match_table: &MatchTable,
) -> (usize, bool) {
    let mask = SUBGAME_TABLE_SLOTS - 1;
    let mut slot_i = (key as usize) & mask;
    for _ in 0..8 {
        let s = &mut table[slot_i];
        if s.epoch != epoch {
            let idx = nodes.len();
            nodes.push(OracleNode::from_cb(cb, match_table));
            *s = SubTableSlot {
                key,
                epoch,
                node_idx: idx as u32,
            };
            return (idx, true);
        }
        if s.key == key {
            return (s.node_idx as usize, false);
        }
        slot_i = (slot_i + 1) & mask;
    }
    let idx = nodes.len();
    nodes.push(OracleNode::from_cb(cb, match_table));
    (idx, true)
}

/// Solve a small (`<= 3 x 3`) zero-sum stage-game payoff matrix `mat` (from
/// row player's perspective) via 60 sweeps of linear-averaged Regret Matching+
/// (`RM+`), purify dust (`< 0.25 * max_prob`), and return `(nash_value, row_p, col_q)`.
#[inline]
fn solve_stage_matrix_rm_plus(
    mat: &[[f64; 3]; 3],
    r_len: usize,
    c_len: usize,
) -> (f64, [f64; 3], [f64; 3]) {
    if r_len == 1 && c_len == 1 {
        return (mat[0][0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    }
    if c_len == 1 {
        let mut best_r = 0usize;
        for r in 1..r_len {
            if mat[r][0] > mat[best_r][0] {
                best_r = r;
            }
        }
        let mut p = [0.0; 3];
        p[best_r] = 1.0;
        return (mat[best_r][0], p, [1.0, 0.0, 0.0]);
    }
    if r_len == 1 {
        let mut best_c = 0usize;
        for c in 1..c_len {
            if mat[0][c] < mat[0][best_c] {
                best_c = c;
            }
        }
        let mut q = [0.0; 3];
        q[best_c] = 1.0;
        return (mat[0][best_c], [1.0, 0.0, 0.0], q);
    }

    let mut reg_r = [0.0f64; 3];
    let mut reg_c = [0.0f64; 3];
    let mut sum_p = [0.0f64; 3];
    let mut sum_q = [0.0f64; 3];
    let unif_r = 1.0 / (r_len as f64);
    let unif_c = 1.0 / (c_len as f64);

    for t in 1..=60usize {
        let mut p = [0.0f64; 3];
        let mut s_r = 0.0f64;
        for r in 0..r_len {
            s_r += reg_r[r];
        }
        if s_r > 1e-12 {
            for r in 0..r_len {
                p[r] = reg_r[r] / s_r;
            }
        } else {
            for r in 0..r_len {
                p[r] = unif_r;
            }
        }

        let mut q = [0.0f64; 3];
        let mut s_c = 0.0f64;
        for c in 0..c_len {
            s_c += reg_c[c];
        }
        if s_c > 1e-12 {
            for c in 0..c_len {
                q[c] = reg_c[c] / s_c;
            }
        } else {
            for c in 0..c_len {
                q[c] = unif_c;
            }
        }

        let mut row_u = [0.0f64; 3];
        let mut col_u = [0.0f64; 3];
        let mut game_u = 0.0f64;
        for r in 0..r_len {
            let mut ur = 0.0f64;
            for c in 0..c_len {
                ur += mat[r][c] * q[c];
                col_u[c] += (1.0 - mat[r][c]) * p[r];
            }
            row_u[r] = ur;
            game_u += p[r] * ur;
        }
        let game_u_c = 1.0 - game_u;
        for r in 0..r_len {
            reg_r[r] = (reg_r[r] + row_u[r] - game_u).max(0.0);
        }
        for c in 0..c_len {
            reg_c[c] = (reg_c[c] + col_u[c] - game_u_c).max(0.0);
        }

        let w = t as f64;
        for r in 0..r_len {
            sum_p[r] += w * p[r];
        }
        for c in 0..c_len {
            sum_q[c] += w * q[c];
        }
    }

    let purify = |raw: &[f64; 3], len: usize| -> [f64; 3] {
        let mut max_v = 0.0f64;
        for &x in &raw[..len] {
            if x > max_v {
                max_v = x;
            }
        }
        let cutoff = 0.25 * max_v;
        let mut out = [0.0f64; 3];
        let mut tot = 0.0f64;
        for i in 0..len {
            if raw[i] >= cutoff {
                out[i] = raw[i];
                tot += raw[i];
            }
        }
        if tot > 1e-12 {
            for x in &mut out[..len] {
                *x /= tot;
            }
        } else {
            out[0] = 1.0;
        }
        out
    };

    let p = purify(&sum_p, r_len);
    let q = purify(&sum_q, c_len);
    let mut val = 0.0f64;
    for r in 0..r_len {
        for c in 0..c_len {
            val += p[r] * q[c] * mat[r][c];
        }
    }
    (val.clamp(0.0, 1.0), p, q)
}

#[inline]
fn pack_top2_actions(acts_codes: &[u8], support_idx: &[usize; 3], support_len: usize, probs: &[f64; 3]) -> ([u8; 2], u8) {
    let mut best0 = 0usize;
    for i in 1..support_len {
        if probs[i] > probs[best0] {
            best0 = i;
        }
    }
    let a0 = acts_codes[support_idx[best0]];
    if support_len <= 1 || probs[best0] >= 0.85 {
        return ([a0, a0], 255);
    }
    let mut best1 = None;
    for i in 0..support_len {
        if i == best0 {
            continue;
        }
        if best1.is_none_or(|b| probs[i] > probs[b]) {
            best1 = Some(i);
        }
    }
    let Some(b1) = best1 else {
        return ([a0, a0], 255);
    };
    if probs[b1] <= 0.05 {
        return ([a0, a0], 255);
    }
    let a1 = acts_codes[support_idx[b1]];
    let p_frac = (probs[best0] / (probs[best0] + probs[b1])).clamp(0.0, 1.0);
    let p_u8 = (p_frac * 255.0).round().clamp(1.0, 254.0) as u8;
    ([a0, a1], p_u8)
}

const ORACLE_UCB_C: f64 = 0.5;
const ORACLE_MAX_TREE_DEPTH: usize = 36;
const ORACLE_SUBGAME_TURN_SPAN: u16 = 40;
const ORACLE_LEAF_ROLLOUT_TURNS: u16 = 12;

fn run_oracle_uncached(
    root_cb: &CompactBattle,
    table: &MatchTable,
    sym_seed: u64,
    iters: u32,
) -> (f64, [[u8; 2]; 2], [u8; 2]) {
    let base_weights = EvalWeights {
        residual_scale: 0.0,
        ..EvalWeights::default()
    };
    let leaf_playout = Playout::Heavy {
        eps: 0.2,
        turns: ORACLE_LEAF_ROLLOUT_TURNS,
        weights: base_weights.clone(),
    };
    let leaf_rules = RolloutRules {
        perish_escape: true,
        perish_combo: true,
        baton_pass: (root_cb.sides[0].side_flags & 0x80) == 0,
        oracle: OracleMode::Off,
        ..RolloutRules::default()
    };
    let tree_turn_cap = 1000u16.min(root_cb.turn.saturating_add(ORACLE_SUBGAME_TURN_SPAN));

    ORACLE_CTX.with(|cell| {
        let mut ctx = cell.borrow_mut();
        ctx.epoch = ctx.epoch.wrapping_add(1);
        if ctx.epoch == 0 {
            ctx.sub_table.fill(SubTableSlot::default());
            ctx.epoch = 1;
        }
        let epoch = ctx.epoch;
        ctx.nodes.clear();
        ctx.nodes.push(OracleNode::from_cb(root_cb, table));

        let root_len0 = ctx.nodes[0].len[0] as usize;
        let root_len1 = ctx.nodes[0].len[1] as usize;
        if root_len0 == 0 && root_len1 == 0 {
            let v = eval::compact_eval_leaf(root_cb, table, &base_weights);
            return (v, [[7, 7], [7, 7]], [255, 255]);
        }

        let mut rw_wt = [[0.0f64; 7]; 2];
        let mut rw_val = [[0.0f64; 7]; 2];
        let mut cell_wt = [[0.0f64; 7]; 7];
        let mut cell_sum0 = [[0.0f64; 7]; 7];
        let mut all_wt = 0.0f64;
        let mut all_sum0 = 0.0f64;

        let mut path = [(0u16, 0u8, 0u8); ORACLE_MAX_TREE_DEPTH * 2];

        for iter_idx in 0..iters {
            let mut rng = SplitMix64::new(
                sym_seed ^ (iter_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            let mut sim = *root_cb;
            sim.prng_seed = rng.next() | 1;
            sim.quick_claw_roll = (((rng.next() & 0xFF) as u16) < 60) as u8;

            let mut path_len = 0usize;
            let mut root_choice = [7u8; 2];
            let mut depth = 0usize;

            let reward0 = loop {
                match sim.outcome {
                    1 => break 1.0,
                    2 => break 0.0,
                    3 => break 0.5,
                    _ => {}
                }
                if sim.turn > tree_turn_cap || depth >= ORACLE_MAX_TREE_DEPTH {
                    break eval::compact_eval_leaf(&sim, table, &base_weights);
                }

                let (node_idx, is_new) = if depth == 0 {
                    (0usize, false)
                } else {
                    let key = oracle_subgame_node_key(&sim, table);
                    let OracleThreadCtx {
                        ref mut sub_table,
                        ref mut nodes,
                        ..
                    } = *ctx;
                    sub_table_lookup_or_insert(sub_table, epoch, key, nodes, &sim, table)
                };

                if is_new {
                    let cutoff =
                        1000u16.min(sim.turn.saturating_add(ORACLE_LEAF_ROLLOUT_TURNS));
                    break mcts::compact_playout_value(
                        &mut sim,
                        table,
                        &leaf_playout,
                        cutoff,
                        &mut rng,
                        leaf_rules,
                    );
                }

                let node = &ctx.nodes[node_idx];
                if node.len[0] == 0 && node.len[1] == 0 {
                    break eval::compact_eval_leaf(&sim, table, &base_weights);
                }

                let mut joint = [7u8; 2];
                for s in 0..2usize {
                    if node.len[s] > 0 {
                        let ai = select_oracle_ucb(node, s, ORACLE_UCB_C, &mut rng);
                        joint[s] = node.acts[s][ai];
                        if depth == 0 {
                            root_choice[s] = ai as u8;
                        }
                        if path_len < path.len() {
                            path[path_len] = (node_idx as u16, s as u8, ai as u8);
                            path_len += 1;
                        }
                    }
                }

                sim.apply_actions(table, joint);
                depth += 1;
            };

            for &(ni, s, ai) in &path[..path_len] {
                let node = &mut ctx.nodes[ni as usize];
                let s_idx = s as usize;
                let a_idx = ai as usize;
                node.n[s_idx][a_idx] += 1;
                node.w[s_idx][a_idx] += if s_idx == 0 { reward0 } else { 1.0 - reward0 };
            }

            let iter_w = (iter_idx + 1) as f64;
            all_wt += iter_w;
            all_sum0 += iter_w * reward0;
            for s in 0..2usize {
                let ai = root_choice[s] as usize;
                if ai < 7 {
                    rw_wt[s][ai] += iter_w;
                    rw_val[s][ai] += iter_w * (if s == 0 { reward0 } else { 1.0 - reward0 });
                }
            }
            let r0 = root_choice[0] as usize;
            let r1 = root_choice[1] as usize;
            if r0 < 7 && r1 < 7 {
                cell_wt[r0][r1] += iter_w;
                cell_sum0[r0][r1] += iter_w * reward0;
            }
        }

        let root_node = &ctx.nodes[0];

        if root_len0 > 0 && root_len1 == 0 {
            let max_n = (0..root_len0).map(|i| root_node.n[0][i]).max().unwrap_or(0);
            let min_n = (max_n / 6).max(1);
            let best = (0..root_len0)
                .filter(|&i| root_node.n[0][i] >= min_n && rw_wt[0][i] > 0.0)
                .max_by(|&a, &b| {
                    (rw_val[0][a] / rw_wt[0][a]).total_cmp(&(rw_val[0][b] / rw_wt[0][b]))
                })
                .unwrap_or(0);
            let v = if rw_wt[0][best] > 0.0 {
                rw_val[0][best] / rw_wt[0][best]
            } else {
                0.5
            };
            let act = root_node.acts[0][best];
            return (v, [[act, act], [7, 7]], [255, 255]);
        }

        if root_len0 == 0 && root_len1 > 0 {
            let max_n = (0..root_len1).map(|i| root_node.n[1][i]).max().unwrap_or(0);
            let min_n = (max_n / 6).max(1);
            let best = (0..root_len1)
                .filter(|&i| root_node.n[1][i] >= min_n && rw_wt[1][i] > 0.0)
                .max_by(|&a, &b| {
                    (rw_val[1][a] / rw_wt[1][a]).total_cmp(&(rw_val[1][b] / rw_wt[1][b]))
                })
                .unwrap_or(0);
            let v1 = if rw_wt[1][best] > 0.0 {
                rw_val[1][best] / rw_wt[1][best]
            } else {
                0.5
            };
            let act = root_node.acts[1][best];
            return (1.0 - v1, [[7, 7], [act, act]], [255, 255]);
        }

        // Simultaneous stage game (`root_len0 > 0 && root_len1 > 0`):
        // Select top-3 actions per side by UCB visits (excluding barely visited blunders).
        let pick_support = |side: usize, k: usize| -> ([usize; 3], usize) {
            let mut order = [0usize, 1, 2, 3, 4, 5, 6];
            order[..k].sort_by(|&a, &b| {
                root_node.n[side][b]
                    .cmp(&root_node.n[side][a])
                    .then_with(|| {
                        let ma = rw_val[side][a] / rw_wt[side][a].max(1e-9);
                        let mb = rw_val[side][b] / rw_wt[side][b].max(1e-9);
                        mb.total_cmp(&ma)
                    })
            });
            let max_n = root_node.n[side][order[0]];
            let min_n = (max_n / 6).max(1);
            let mut supp = [order[0]; 3];
            let mut slen = 1usize;
            for &idx in order[1..k].iter().take(2) {
                if root_node.n[side][idx] >= min_n && rw_wt[side][idx] > 0.0 {
                    supp[slen] = idx;
                    slen += 1;
                }
            }
            (supp, slen)
        };

        let (rows, r_len) = pick_support(0, root_len0);
        let (cols, c_len) = pick_support(1, root_len1);

        let all_mean0 = if all_wt > 0.0 { all_sum0 / all_wt } else { 0.5 };
        // Mild empirical-Bayes shrinkage weight (~1.5 early iterations) for sparse off-diagonal cells.
        let prior_wt = (iters as f64) * 1.5;
        let mut mat = [[0.5f64; 3]; 3];
        for r in 0..r_len {
            let ri = rows[r];
            let r_mean0 = if rw_wt[0][ri] > 0.0 {
                rw_val[0][ri] / rw_wt[0][ri]
            } else {
                all_mean0
            };
            for c in 0..c_len {
                let ci = cols[c];
                let c_mean0 = if rw_wt[1][ci] > 0.0 {
                    1.0 - (rw_val[1][ci] / rw_wt[1][ci])
                } else {
                    all_mean0
                };
                let prior = (r_mean0 + c_mean0 - all_mean0).clamp(0.0, 1.0);
                let cw = cell_wt[ri][ci];
                mat[r][c] = (cell_sum0[ri][ci] + prior_wt * prior) / (cw + prior_wt);
            }
        }

        let (nash_val, p_row, q_col) = solve_stage_matrix_rm_plus(&mat, r_len, c_len);
        let (acts0, prob0) = pack_top2_actions(&root_node.acts[0], &rows, r_len, &p_row);
        let (acts1, prob1) = pack_top2_actions(&root_node.acts[1], &cols, c_len, &q_col);

        (nash_val, [acts0, acts1], [prob0, prob1])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use nc2000_engine::dex::Dex;
    use nc2000_engine::state::Battle;

    fn test_dex() -> Dex {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let text = std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap();
        Dex::from_json(&text).unwrap()
    }

    fn mon(species: &str, item: &str, level: u8, moves: &[&str]) -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255
        }))
        .unwrap()
    }

    fn advance_preview(b: &mut Battle, dex: &Dex) {
        let p0 = b.legal_choices(dex, 0)[0];
        let p1 = b.legal_choices(dex, 1)[0];
        b.apply_choices(dex, [Some(p0), Some(p1)]).unwrap();
    }

    #[test]
    fn cache_hit_matches_cache_miss_to_zero_ulp_and_ignores_prng() {
        let dex = test_dex();
        let t0 = vec![
            mon("Zapdos", "Leftovers", 50, &["Thunderbolt", "Hidden Power", "Rest", "Sleep Talk"]),
            mon("Snorlax", "Leftovers", 50, &["Double-Edge", "Curse", "Earthquake", "Rest"]),
            mon("Cloyster", "Leftovers", 50, &["Surf", "Ice Beam", "Spikes", "Explosion"]),
        ];
        let t1 = vec![
            mon("Raikou", "Leftovers", 50, &["Thunderbolt", "Crunch", "Rest", "Roar"]),
            mon("Marowak", "Thick Club", 50, &["Earthquake", "Rock Slide", "Swords Dance", "Body Slam"]),
            mon("Starmie", "MiracleBerry", 50, &["Surf", "Psychic", "Thunder Wave", "Recover"]),
        ];
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t0, &t1).unwrap();
        advance_preview(&mut b, &dex);
        let (table, cb) = CompactBattle::from_battle(&b, &dex);

        clear_oracle_thread_caches();
        let cold = oracle_equity(&cb, &table, 96);
        let warm = oracle_equity(&cb, &table, 96);
        assert_eq!(cold.to_bits(), warm.to_bits(), "cache hit must be 0.0-ULP identical to miss");

        let mut cb_diff_rng = cb;
        cb_diff_rng.prng_seed ^= 0xDEAD_BEEF_CAFE_BABEu64;
        cb_diff_rng.quick_claw_roll ^= 1;
        clear_oracle_thread_caches();
        let diff_rng_cold = oracle_equity(&cb_diff_rng, &table, 96);
        assert_eq!(
            cold.to_bits(),
            diff_rng_cold.to_bits(),
            "oracle_equity must have zero future PRNG / Quick Claw leakage"
        );
    }

    #[test]
    fn side_swap_antisymmetry_holds_exactly() {
        let dex = test_dex();
        let t0 = vec![
            mon("Jolteon", "Leftovers", 50, &["Thunderbolt", "Agility", "Baton Pass", "Hidden Power"]),
            mon("Marowak", "Thick Club", 55, &["Earthquake", "Rock Slide", "Body Slam", "Swords Dance"]),
            mon("Suicune", "Leftovers", 50, &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
        ];
        let t1 = vec![
            mon("Snorlax", "Leftovers", 55, &["Double-Edge", "Curse", "Rest", "Earthquake"]),
            mon("Skarmory", "Leftovers", 50, &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mon("Tyranitar", "Leftovers", 50, &["Rock Slide", "Crunch", "Earthquake", "Roar"]),
        ];
        let mut b = Battle::from_fixture(&dex, "9,8,7,6", &t0, &t1).unwrap();
        advance_preview(&mut b, &dex);
        let (table, mut cb) = CompactBattle::from_battle(&b, &dex);
        cb.mons[0].boosts[4] = 2;
        cb.mons[3].hp_pool[0] = (table.mons[3].maxhp * 3) / 5;

        clear_oracle_thread_caches();
        let e0 = oracle_equity(&cb, &table, 96);
        clear_oracle_thread_caches();
        let (scb, st) = swap_compact_sides(&cb, &table);
        let e1 = oracle_equity(&scb, &st, 96);
        assert!(
            (e0 + e1 - 1.0).abs() < 1e-6,
            "expected E(s) + E(swap(s)) == 1.0, got e0={e0}, e1={e1}, sum={}",
            e0 + e1
        );
    }

    #[test]
    fn oracle_recognizes_baton_pass_synergy_and_picks_pass() {
        let dex = test_dex();
        let t0 = vec![
            mon("Jolteon", "Leftovers", 50, &["Thunderbolt", "Charm", "Baton Pass", "Growth"]),
            mon("Marowak", "Thick Club", 50, &["Earthquake", "Rock Slide", "Body Slam", "Rest"]),
            mon("Cloyster", "Leftovers", 50, &["Surf", "Ice Beam", "Rest", "Spikes"]),
        ];
        let t1 = vec![
            mon("Snorlax", "Leftovers", 55, &["Body Slam", "Earthquake", "Rest", "Sleep Talk"]),
            mon("Raikou", "Leftovers", 50, &["Thunderbolt", "Crunch", "Rest", "Sleep Talk"]),
            mon("Tyranitar", "Leftovers", 50, &["Rock Slide", "Crunch", "Fire Blast", "Rest"]),
        ];
        let mut b = Battle::from_fixture(&dex, "5,6,7,8", &t0, &t1).unwrap();
        advance_preview(&mut b, &dex);
        let (table, mut cb_unboosted) = CompactBattle::from_battle(&b, &dex);
        cb_unboosted.mons[0].hp_pool[0] = (table.mons[0].maxhp * 2) / 5;

        let mut cb_boosted = cb_unboosted;
        cb_boosted.mons[0].boosts[0] = 2;
        cb_boosted.mons[0].boosts[4] = 2;

        clear_oracle_thread_caches();
        let eq_unboosted = oracle_equity(&cb_unboosted, &table, 160);
        let eq_boosted = oracle_equity(&cb_boosted, &table, 160);
        assert!(
            eq_boosted > eq_unboosted + 0.15,
            "oracle must value +2 Atk/+2 Spe Baton Pass Jolteon + Marowak far above unboosted: boosted={eq_boosted:.3} vs unboosted={eq_unboosted:.3}"
        );

        let mut cs = [0u8; 7];
        let n = cb_boosted.legal_actions(&table, 0, &mut cs);
        clear_oracle_thread_caches();
        let mut rng_cold = SplitMix64::new(42);
        let mut rng_warm = SplitMix64::new(42);
        let pick_cold = oracle_rollout_pick(&cb_boosted, &table, 0, &cs[..n], 128, &mut rng_cold);
        let pick_warm = oracle_rollout_pick(&cb_boosted, &table, 0, &cs[..n], 128, &mut rng_warm);
        assert_eq!(pick_cold, pick_warm, "pick cache must be bit-exact");
        assert_eq!(
            pick_cold, 2,
            "oracle_rollout_pick should choose Baton Pass (slot 2), got {pick_cold}"
        );
    }
}
