//! Compact-AlphaZero-DUCT: Universal Species-Free Interior Policy Prior (`P_net`),
//! Sound Interior No-Op Mask (`compact_interior_noop_mask`), and 0-ULP Antisymmetric
//! Siamese Value Network (`V_net`) natively on 256-byte `CompactBattle`.
//!
//! Zero heap allocations on inference, zero species/move/item/team IDs (100% general
//! across unseen Gen 2 rosters), WASM-friendly pure Rust `f32`, preserving exact `1v1`
//! `linear.rs` Schur-Bellman valuation and outer `relay.rs` Baton-Pass Bellman subgame.

use nc2000_engine::compact::{
    ext_vol_bit, vol_bit, CompactBattle, CompactWeather, MatchTable, SpecialMoveKind,
};
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::state::Status;

use crate::compact_az_weights::{
    POLICY_ACT_DIM, POLICY_CTX_DIM, POLICY_H1, POLICY_H2, POLICY_WEIGHTS_LOADED, P_B1, P_B2,
    P_TAU, P_W2_MEAN, P_W2_SELF, P_W_ACT, P_W_CTX, P_W_OUT, VALUE_H1, VALUE_H2, VALUE_SIDE_DIM,
    VALUE_WEIGHTS_LOADED, V_B1, V_B2, V_SKIP_BS, V_SKIP_FG, V_W1_OPP, V_W1_SELF, V_W2_OPP,
    V_W2_SELF, V_W_OUT,
};
use crate::eval::{compact_eval01_blindspot_norelay, compact_turn_limit_outcome, EvalWeights};
use crate::rng::SplitMix64;
use crate::smmcts::{run_compact_iteration, CompactNode, RmConfig, SelRule};
use crate::tempo::{
    compute_fast_golf_core, effective_speed, entry_hp, evaluate_hybrid_golf_temp,
    expected_sleep_turns, fast_ko_prob, scale_cached_move, spikes_layers, stage3_logit, CacheView,
    FastGolfEval, FastPair, TempoMatchCache,
};

pub use crate::compact_az_weights::{
    POLICY_ACT_DIM as AZ_POLICY_ACT_DIM, POLICY_CTX_DIM as AZ_POLICY_CTX_DIM,
    VALUE_SIDE_DIM as AZ_VALUE_SIDE_DIM,
};

#[inline(always)]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x.clamp(-20.0, 20.0)).exp())
}

// =========================================================================
// 1. Sound Interior No-Op Mask (`depth >= 1`)
// =========================================================================

#[inline]
fn compact_foe_has_residual(cb: &CompactBattle, table: &MatchTable, side: usize) -> bool {
    let opp = 1 - side;
    let d_slot = cb.sides[opp].active_idx as usize;
    if d_slot >= 3 || (cb.sides[opp].alive_mask & (1 << d_slot)) == 0 {
        return false;
    }
    let idx = opp * 3 + d_slot;
    let foe = &cb.mons[idx];
    if foe.hp() <= 0 {
        return false;
    }
    let st = foe.status_raw();
    if st == Status::Psn as u8 || st == Status::Tox as u8 || st == Status::Brn as u8 {
        return true;
    }
    if (foe.vol_flags & (vol_bit::CURSE | vol_bit::LEECH_SEED)) != 0 || foe.partially_trapped() {
        return true;
    }
    if (foe.vol_flags & vol_bit::NIGHTMARE) != 0
        && (st == Status::Slp as u8
            || (table.ruleset.is_castle()
                && (foe.ext_flags & ext_vol_bit::MIRACLE_NIGHTMARE) != 0))
    {
        return true;
    }
    cb.weather_turns > 0
        && cb.weather_kind == CompactWeather::Sandstorm
        && (table.mons[idx].type_mask & table.sandstorm_immune_mask) == 0
}

#[inline]
fn compact_wait_exempt(cb: &CompactBattle, table: &MatchTable, side: usize) -> bool {
    let opp = 1 - side;
    let a_slot = cb.sides[side].active_idx as usize;
    let d_slot = cb.sides[opp].active_idx as usize;
    if a_slot >= 3 || d_slot >= 3 {
        return true;
    }
    let a_idx = side * 3 + a_slot;
    let d_idx = opp * 3 + d_slot;
    let me = &cb.mons[a_idx];
    let foe = &cb.mons[d_idx];
    let foe_alive = (cb.sides[opp].alive_mask & (1 << d_slot)) != 0 && foe.hp() > 0;

    if (me.vol_flags & vol_bit::ENCORE) != 0
        && ((me.vol_flags & vol_bit::SUBSTITUTE) != 0
            || me.boosts.iter().take(5).any(|&b| b > 0)
            || (foe_alive && foe.is_trapped()))
    {
        return true;
    }
    if foe_alive && (foe.vol_flags & vol_bit::DESTINY_BOND) != 0 {
        return true;
    }
    if foe_alive && (foe.vol_flags & vol_bit::PERISH_SONG) != 0 && foe.perish_turns() == 1 {
        return true;
    }
    if compact_foe_has_residual(cb, table, side) {
        return true;
    }
    if me.hp() > 0
        && me.hp() < table.mons[a_idx].maxhp
        && table.item_spec(me.item).is_leftovers
        && foe_alive
    {
        let fst = foe.status_raw();
        if fst == Status::Slp as u8 || fst == Status::Frz as u8 {
            return true;
        }
        let n_fm = table.mons[d_idx].num_moves as usize;
        for m in 0..n_fm {
            if foe.pp[m] > 0
                && matches!(
                    table.move_specs[d_idx][m].special_kind,
                    SpecialMoveKind::Counter | SpecialMoveKind::MirrorCoat
                )
            {
                return true;
            }
        }
    }
    false
}

#[inline]
fn mon_immune_to_type(table: &MatchTable, mon_idx: usize, move_type: u8) -> bool {
    if move_type >= 20 {
        return false;
    }
    let mt = move_type as usize;
    let spec = &table.mons[mon_idx];
    let t0 = spec.types[0] as usize;
    if t0 < 20 && table.type_chart[mt][t0] == -9 {
        return true;
    }
    if spec.num_types > 1 {
        let t1 = spec.types[1] as usize;
        if t1 < 20 && table.type_chart[mt][t1] == -9 {
            return true;
        }
    }
    false
}

/// Strictly sound turn-order-independent interior no-op bitmask for `CompactBattle` (`depth >= 1`).
/// Bit `i` is `1` iff `acts[i]` is proven useless/self-forfeiting in this determinized state.
/// Never masks all legal actions (returns `0` if every action would be masked).
#[inline]
pub fn compact_interior_noop_mask(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    acts: &[u8],
    n_acts: usize,
) -> u8 {
    if n_acts <= 1 || cb.sides[side].request != 1 {
        return 0;
    }
    let opp = 1 - side;
    let a_slot = cb.sides[side].active_idx as usize;
    let d_slot = cb.sides[opp].active_idx as usize;
    if a_slot >= 3 || d_slot >= 3 {
        return 0;
    }
    let a_idx = side * 3 + a_slot;
    let d_idx = opp * 3 + d_slot;
    let me = &cb.mons[a_idx];
    let foe = &cb.mons[d_idx];
    let maxhp = table.mons[a_idx].maxhp.max(1);
    let my_alive = cb.sides[side].alive_mask.count_ones();
    let opp_alive = cb.sides[opp].alive_mask.count_ones();
    let is_castle = table.ruleset.is_castle();
    let wait_ex = compact_wait_exempt(cb, table, side);

    let foe_alive = (cb.sides[opp].alive_mask & (1 << d_slot)) != 0 && foe.hp() > 0;
    let mut foe_has_sleep_move = false;
    let mut foe_has_rapid_spin = false;
    let mut foe_has_heal_bell = false;
    let mut foe_can_uncap_boost = false;
    if foe_alive {
        let n_fm = table.mons[d_idx].num_moves as usize;
        for m in 0..n_fm {
            if foe.pp[m] == 0 {
                continue;
            }
            let fms = &table.move_specs[d_idx][m];
            if fms.primary_status == Status::Slp as u8 {
                foe_has_sleep_move = true;
            }
            if fms.special_kind == SpecialMoveKind::RapidSpin {
                foe_has_rapid_spin = true;
            }
            if fms.special_kind == SpecialMoveKind::HealBell {
                foe_has_heal_bell = true;
            }
            if fms.special_kind == SpecialMoveKind::Haze
                || (!fms.boosts_self && fms.boosts.iter().any(|&(st_i, d)| st_i < 7 && d < 0))
                || ((fms.sec_boost.0 & 0x80) == 0 && (fms.sec_boost.0 & 0x7F) < 7 && fms.sec_boost.1 < 0)
            {
                foe_can_uncap_boost = true;
            }
        }
    }
    let me_item = table.item_spec(me.item);
    let me_has_heal_berry = me_item.is_goldberry || me_item.is_berry_juice || me_item.is_berry;
    // Check if any living foe bench mon is asleep (or carries Heal Bell).
    let mut foe_bench_asleep = false;
    for slot in 0..3usize {
        if (cb.sides[opp].alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let idx = opp * 3 + slot;
        let m = &cb.mons[idx];
        if m.hp() <= 0 {
            continue;
        }
        if slot != d_slot && m.status_raw() == Status::Slp as u8 {
            foe_bench_asleep = true;
        }
        if !foe_has_heal_bell {
            for mi in 0..table.mons[idx].num_moves as usize {
                if m.pp[mi] > 0 && table.move_specs[idx][mi].special_kind == SpecialMoveKind::HealBell
                {
                    foe_has_heal_bell = true;
                    break;
                }
            }
        }
    }
    let sleep_clause_on = (cb.clause_flags & 1) != 0 || is_castle;
    let me_is_ghost = (table.mons[a_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
    let me_has_sub = (me.vol_flags & vol_bit::SUBSTITUTE) != 0;
    let me_st = me.status_raw();
    let me_waking_or_awake = me_st != Status::Slp as u8 || me.status_counter() <= 1;
    let castle_spin_util = is_castle
        && ((cb.sides[side].side_flags & 0x03) != 0
            || (me.vol_flags & vol_bit::LEECH_SEED) != 0
            || me.partially_trapped());
    let foe_foresight = foe_alive && (foe.vol_flags & vol_bit::FORESIGHT) != 0;

    let mut mask = 0u8;
    for i in 0..n_acts.min(7) {
        let act = acts[i];
        if act >= 4 {
            continue;
        }
        let m_slot = act as usize;
        if m_slot >= table.mons[a_idx].num_moves as usize {
            continue;
        }
        let ms = &table.move_specs[a_idx][m_slot];

        // 1. Unconditional last-mon Self-KO forfeit (even under wait_exempt).
        if my_alive <= 1 && ms.is_selfdestruct {
            mask |= 1u8 << i;
            continue;
        }
        // 2. Sleep Clause forfeit/fail against living sleeping bench mon (runs BEFORE wait_exempt
        //    on PS since inflicting sleep forfeits the game; final all-masked guard preserves safety).
        if sleep_clause_on
            && ms.primary_status == Status::Slp as u8
            && foe_bench_asleep
            && !foe_has_heal_bell
            && (is_castle || cb.sides[opp].safeguard_turns == 0)
            && (!is_castle || !wait_ex)
        {
            mask |= 1u8 << i;
            continue;
        }
        if wait_ex {
            continue;
        }

        // 3. Frozen or safely-awake Sleep Talk / Snore (never mask awake when foe has a sleep move).
        if ms.is_sleep_usable {
            if me_st == Status::Frz as u8
                || (me_waking_or_awake
                    && ((me_st != Status::None as u8 && me_st != Status::Slp as u8)
                        || me_has_sub
                        || cb.sides[side].safeguard_turns > 0
                        || !foe_has_sleep_move))
            {
                mask |= 1u8 << i;
                continue;
            }
        }

        // 4. Side conditions already active / Baton Pass on last mon / Phaze vs last foe / DBond on last mon.
        match ms.special_kind {
            SpecialMoveKind::Reflect if cb.sides[side].reflect_turns > 0 => {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::LightScreen if cb.sides[side].lightscreen_turns > 0 => {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::Safeguard if cb.sides[side].safeguard_turns > 0 => {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::Spikes
                if (cb.sides[opp].side_flags & 0x03) >= 1 && !foe_has_rapid_spin =>
            {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::BatonPass if my_alive <= 1 => {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::Phaze if opp_alive <= 1 => {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::DestinyBond if my_alive <= 1 => {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::BellyDrum
                if (me.boosts[0] >= 6 && !foe_can_uncap_boost)
                    || (!me_has_heal_berry && (me.hp() as i32) * 16 <= (maxhp as i32) * 7) =>
            {
                mask |= 1u8 << i;
                continue;
            }
            SpecialMoveKind::Curse
                if !me_is_ghost
                    && !foe_can_uncap_boost
                    && me.boosts[0] >= 6
                    && me.boosts[1] >= 6
                    && me.boosts[4] <= -6 =>
            {
                mask |= 1u8 << i;
                continue;
            }
            _ => {}
        }

        // 5. Substitute already up or safely below HP-bucket-4 boundary (sound under 16-bucket TT).
        if ms.is_substitute
            && (me_has_sub
                || maxhp <= 1
                || (!me_has_heal_berry && (me.hp() as i32) * 16 <= (maxhp as i32) * 3))
        {
            mask |= 1u8 << i;
            continue;
        }

        // 6. Pure self-boost already capped at +6 (only when active foe cannot Haze/drop stats).
        if ms.is_status
            && ms.boosts_self
            && !foe_can_uncap_boost
            && !matches!(
                ms.special_kind,
                SpecialMoveKind::DefenseCurl | SpecialMoveKind::Minimize | SpecialMoveKind::Curse
            )
        {
            let mut any_b = false;
            let mut all_capped = true;
            for &(st_i, delta) in &ms.boosts {
                if st_i < 7 && delta != 0 {
                    any_b = true;
                    if delta <= 0 || me.boosts[st_i as usize] < 6 {
                        all_capped = false;
                        break;
                    }
                }
            }
            if any_b && all_capped {
                mask |= 1u8 << i;
                continue;
            }
        }

        // 7. Universally type-immune damaging move (EVERY living foe mon is immune).
        if !ms.is_status
            && !ms.ignore_immunity
            && !ms.is_selfdestruct
            && ms.base_power > 0
            && !foe_foresight
            && !(ms.special_kind == SpecialMoveKind::RapidSpin && castle_spin_util)
        {
            let mut any_foe = false;
            let mut all_immune = true;
            for slot in 0..3usize {
                if (cb.sides[opp].alive_mask & (1 << slot)) == 0 {
                    continue;
                }
                let f_idx = opp * 3 + slot;
                if cb.mons[f_idx].hp() <= 0 {
                    continue;
                }
                any_foe = true;
                if !mon_immune_to_type(table, f_idx, ms.move_type) {
                    all_immune = false;
                    break;
                }
            }
            if any_foe && all_immune {
                mask |= 1u8 << i;
                continue;
            }
        }
    }

    let all_bits = ((1u16 << n_acts.min(7)) - 1) as u8;
    if (mask & all_bits) == all_bits {
        0
    } else {
        mask
    }
}

// =========================================================================
// 2. Fast O(1) Move & Pair Damage Queries via `TempoMatchCache`
// =========================================================================

/// Return `(mean_dmg_vs_maxhp, ohko_prob_vs_cur_hp, p_hit)` for a specific attacking move slot.
#[inline]
fn move_dmg_stats(
    cb: &CompactBattle,
    table: &MatchTable,
    cview: Option<&CacheView<'_>>,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    m_slot: usize,
) -> (f32, f32, f32) {
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let ms = &table.move_specs[att_idx][m_slot];
    if ms.is_status {
        return (0.0, 0.0, (ms.base_acc_pct as f32 / 100.0).clamp(0.0, 1.0));
    }
    let def_hp = cb.mons[def_idx].hp().max(0) as f32;
    let def_maxhp = table.mons[def_idx].maxhp.max(1) as f32;
    if def_hp <= 0.0 {
        return (0.0, 0.0, 1.0);
    }
    if let Some(cv) = cview {
        let ce = cv.pair(att_side, att_slot, def_slot);
        let def_mon = &cb.mons[def_idx];
        let att_mon = &cb.mons[att_idx];
        let weather_active = cb.weather_turns > 0
            && matches!(cb.weather_kind, CompactWeather::Rain | CompactWeather::Sun);
        let def_has_sub = cb.sides[def_side].active_idx as usize == def_slot && def_mon.sub_hp() > 0;
        let cache_ok = !ce.any_dynamic_bp
            && !(ce.has_weather_acc && weather_active)
            && !(ce.has_drain && def_has_sub)
            && (def_mon.vol_flags & vol_bit::FORESIGHT) == 0
            && (att_mon.ext_flags
                & (ext_vol_bit::FOCUS_ENERGY
                    | ext_vol_bit::LOCK_ON
                    | ext_vol_bit::DEFENSE_CURL
                    | ext_vol_bit::TRANSFORMED
                    | ext_vol_bit::RAGE))
                == 0
            && (def_mon.ext_flags
                & (ext_vol_bit::MINIMIZE
                    | ext_vol_bit::TRANSFORMED
                    | ext_vol_bit::DIGGING
                    | ext_vol_bit::FLYING))
                == 0;
        if cache_ok {
            let mut md = ce.moves[m_slot];
            if md.max_norm <= 0.0 && md.mean_dmg <= 0.0 {
                return (0.0, 0.0, 0.0);
            }
            let att_is_active = cb.sides[att_side].active_idx as usize == att_slot;
            let att_burned = att_mon.status_raw() == Status::Brn as u8
                && (table.ruleset.is_castle()
                    || !att_is_active
                    || (att_mon.vol_flags & vol_bit::BRN_ATK_DROP) != 0);
            let def_ref = cb.sides[def_side].reflect_turns;
            let def_ls = cb.sides[def_side].lightscreen_turns;
            if att_mon.boosts != [0; 7]
                || def_mon.boosts != [0; 7]
                || att_burned
                || def_ref > 0
                || def_ls > 0
                || weather_active
            {
                md = scale_cached_move(
                    md,
                    &att_mon.boosts,
                    &def_mon.boosts,
                    att_burned,
                    def_ref,
                    def_ls,
                    cb.weather_kind,
                    weather_active,
                );
            }
            let p_ko = fast_ko_prob(&md, def_hp) as f32;
            let mean_frac = (md.mean_dmg / def_maxhp).clamp(0.0, 2.0);
            return (mean_frac, p_ko, md.p_hit);
        }
    }
    let (frac, acc) =
        cb.expected_hit_fraction_with_acc(table, att_side, att_slot, def_side, def_slot, m_slot);
    let hp_frac = (def_hp / def_maxhp).max(0.01);
    let p_ko = ((frac as f32) / hp_frac).clamp(0.0, 1.0);
    ((frac as f32).clamp(0.0, 2.0), p_ko, (acc as f32).clamp(0.0, 1.0))
}

/// Return `(best_non_boom_dmg_vs_maxhp, best_ohko_prob)` for `(att_side, att_slot) -> (def_side, def_slot)`.
#[inline]
fn best_pair_dmg_stats(
    cb: &CompactBattle,
    table: &MatchTable,
    cview: Option<&CacheView<'_>>,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
) -> (f32, f32) {
    let att_idx = att_side * 3 + att_slot;
    let att_mon = &cb.mons[att_idx];
    let n_moves = table.mons[att_idx].num_moves as usize;
    let mut best_d = 0.0f32;
    let mut best_ko = 0.0f32;
    for m in 0..n_moves {
        if att_mon.pp[m] == 0 || att_mon.is_slot_disabled(m) {
            continue;
        }
        let ms = &table.move_specs[att_idx][m];
        if ms.is_status || ms.is_selfdestruct {
            continue;
        }
        let (d, ko, _) = move_dmg_stats(cb, table, cview, att_side, att_slot, def_side, def_slot, m);
        let eff_d = if ms.is_recharge && ko < 0.85 { d * 0.55 } else { d };
        if eff_d > best_d {
            best_d = eff_d;
        }
        if ko > best_ko {
            best_ko = ko;
        }
    }
    (best_d, best_ko)
}

// =========================================================================
// 3. Policy Feature Extractor & `P_net` Inference (Species-ID-Free)
// =========================================================================

/// Extract `(ctx[24], act_feats[n_acts][28])` for `side` at `cb` (pure mechanics, zero species/team IDs).
#[inline]
pub fn extract_policy_features(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
    side: usize,
    acts: &[u8],
    n_acts: usize,
) -> ([f32; POLICY_CTX_DIM], [[f32; POLICY_ACT_DIM]; 7]) {
    let mut ctx = [0.0f32; POLICY_CTX_DIM];
    let mut af = [[0.0f32; POLICY_ACT_DIM]; 7];
    let opp = 1 - side;
    let a_slot = (cb.sides[side].active_idx as usize).min(2);
    let d_slot = (cb.sides[opp].active_idx as usize).min(2);
    let a_idx = side * 3 + a_slot;
    let d_idx = opp * 3 + d_slot;
    let me = &cb.mons[a_idx];
    let foe = &cb.mons[d_idx];
    let my_maxhp = table.mons[a_idx].maxhp.max(1) as f32;
    let foe_maxhp = table.mons[d_idx].maxhp.max(1) as f32;
    let my_alive = cb.sides[side].alive_mask.count_ones() as f32;
    let opp_alive = cb.sides[opp].alive_mask.count_ones() as f32;
    let me_alive = (cb.sides[side].alive_mask & (1 << a_slot)) != 0 && me.hp() > 0;
    let foe_alive = (cb.sides[opp].alive_mask & (1 << d_slot)) != 0 && foe.hp() > 0;
    let is_forced_sw = cb.sides[side].request == 2;
    let view_opt = cache.and_then(|c| c.view_for(cb));
    let cview = view_opt.as_ref();

    let my_hp_f = if me_alive { (me.hp() as f32 / my_maxhp).clamp(0.0, 1.0) } else { 0.0 };
    let foe_hp_f = if foe_alive { (foe.hp() as f32 / foe_maxhp).clamp(0.0, 1.0) } else { 0.0 };
    let my_st = me.status_raw();
    let foe_st = foe.status_raw();

    let my_spd = effective_speed(cb, table, side, a_slot);
    let foe_spd = effective_speed(cb, table, opp, d_slot);
    let spd_cmp = if my_spd > foe_spd {
        1.0
    } else if my_spd < foe_spd {
        -1.0
    } else {
        0.0
    };
    let my_pos_b: f32 = me.boosts.iter().take(5).map(|&b| b.max(0) as f32).sum::<f32>() / 4.0;
    let foe_pos_b: f32 = foe.boosts.iter().take(5).map(|&b| b.max(0) as f32).sum::<f32>() / 4.0;

    let (my_best_d, my_best_ko) = if me_alive && foe_alive {
        best_pair_dmg_stats(cb, table, cview, side, a_slot, opp, d_slot)
    } else {
        (0.0, 0.0)
    };
    let (foe_best_d, foe_best_ko) = if me_alive && foe_alive {
        best_pair_dmg_stats(cb, table, cview, opp, d_slot, side, a_slot)
    } else {
        (0.0, 0.0)
    };
    let my_best_hp = if foe_hp_f > 0.01 { (my_best_d / foe_hp_f).min(1.5) } else { 0.0 };
    let foe_best_hp = if my_hp_f > 0.01 { (foe_best_d / my_hp_f).min(1.5) } else { 0.0 };

    let perish_urg = |m: &nc2000_engine::compact::CompactMon| -> f32 {
        if (m.vol_flags & vol_bit::PERISH_SONG) == 0 {
            return 0.0;
        }
        match m.perish_turns() {
            1 => 1.0,
            2 => 0.5,
            3 => 0.25,
            _ => 0.0,
        }
    };
    let passing_alive = is_forced_sw
        && me_alive
        && (my_pos_b > 0.0 || (me.vol_flags & vol_bit::SUBSTITUTE) != 0);

    ctx[0] = my_hp_f;
    ctx[1] = foe_hp_f;
    ctx[2] = (expected_sleep_turns(cb, side, a_slot) as f32 / 4.0).clamp(0.0, 1.0);
    ctx[3] = if my_st == Status::Par as u8 { 1.0 } else { 0.0 };
    ctx[4] = if matches!(my_st, 2 | 3 | 4 | 5) { 1.0 } else { 0.0 }; // brn/frz/psn/tox
    ctx[5] = (expected_sleep_turns(cb, opp, d_slot) as f32 / 4.0).clamp(0.0, 1.0);
    ctx[6] = if foe_st != 0 && foe_st != Status::Slp as u8 { 1.0 } else { 0.0 };
    ctx[7] = ((me.sub_hp() as f32 / my_maxhp) * 4.0).clamp(0.0, 1.0);
    ctx[8] = ((foe.sub_hp() as f32 / foe_maxhp) * 4.0).clamp(0.0, 1.0);
    ctx[9] = spd_cmp;
    ctx[10] = my_pos_b.min(1.5);
    ctx[11] = foe_pos_b.min(1.5);
    ctx[12] = my_best_hp;
    ctx[13] = my_best_ko;
    ctx[14] = foe_best_hp;
    ctx[15] = foe_best_ko;
    ctx[16] = my_alive / 3.0;
    ctx[17] = opp_alive / 3.0;
    ctx[18] = perish_urg(me);
    ctx[19] = perish_urg(foe);
    ctx[20] = if me.is_trapped() || (me.vol_flags & vol_bit::ENCORE) != 0 { 1.0 } else { 0.0 };
    let my_team_statused = (0..3usize).any(|sl| {
        (cb.sides[side].alive_mask & (1 << sl)) != 0
            && cb.mons[side * 3 + sl].hp() > 0
            && cb.mons[side * 3 + sl].status_raw() != 0
    });
    let me_ispec = table.item_spec(me.item);
    ctx[21] = if spikes_layers(cb, opp) > 0 { 1.0 } else { 0.0 };
    ctx[22] = if spikes_layers(cb, side) > 0 { 1.0 } else { 0.0 };
    ctx[23] = if passing_alive { 1.5 } else if is_forced_sw { 1.0 } else { 0.0 };
    ctx[24] = if foe_alive && (foe.vol_flags & vol_bit::DESTINY_BOND) != 0 { 1.0 } else { 0.0 };
    ctx[25] = if foe_alive && foe.is_trapped() { 1.0 } else { 0.0 };
    ctx[26] = if my_team_statused { 1.0 } else { 0.0 };
    ctx[27] = if me_ispec.is_mintberry || me_ispec.is_miracleberry {
        1.0
    } else if me_ispec.is_leftovers {
        0.5
    } else {
        0.0
    };

    let opp_has_asleep = (0..3usize).any(|sl| {
        (cb.sides[opp].alive_mask & (1 << sl)) != 0
            && cb.mons[opp * 3 + sl].hp() > 0
            && cb.mons[opp * 3 + sl].status_raw() == Status::Slp as u8
    });
    let sleep_open = ((cb.clause_flags & 1) == 0 && !table.ruleset.is_castle()) || !opp_has_asleep;
    let me_is_ghost = (table.mons[a_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
    let flying_bit = 1u32 << (table.flying_type & 31);

    // Precompute best non-boom move dmg and best switch metrics for relative within-set flags.
    let mut max_legal_move_d = 0.0f32;
    let mut best_move_i = 0xFFusize;
    let mut move_dmgs = [(0.0f32, 0.0f32, 0.0f32); 7];
    let mut best_sw_def_i = 0xFFusize;
    let mut best_sw_def_v = f32::INFINITY;
    let mut best_sw_off_i = 0xFFusize;
    let mut best_sw_off_v = -1.0f32;
    let mut best_sw_hp_i = 0xFFusize;
    let mut best_sw_hp_v = -1.0f32;

    let mut sw_cache = [(0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32); 7]; // (hp_f, t2f_d, t2f_ko, f2t_d, f2t_ko)

    for i in 0..n_acts.min(7) {
        let act = acts[i];
        if act < 4 {
            let m_slot = act as usize;
            if m_slot < table.mons[a_idx].num_moves as usize && me_alive && foe_alive {
                let (d, ko, acc) = move_dmg_stats(cb, table, cview, side, a_slot, opp, d_slot, m_slot);
                move_dmgs[i] = (d, ko, acc);
                if !table.move_specs[a_idx][m_slot].is_selfdestruct && d > max_legal_move_d {
                    max_legal_move_d = d;
                    best_move_i = i;
                }
            }
        } else if (4..7).contains(&act) {
            let t_slot = (act - 4) as usize;
            let t_idx = side * 3 + t_slot;
            let t_maxhp = table.mons[t_idx].maxhp.max(1) as f32;
            let t_hp_f = (cb.mons[t_idx].hp().max(0) as f32 / t_maxhp).clamp(0.0, 1.0);
            let (t2f_d, t2f_ko) = if foe_alive {
                best_pair_dmg_stats(cb, table, cview, side, t_slot, opp, d_slot)
            } else {
                (0.0, 0.0)
            };
            let (f2t_d, f2t_ko) = if foe_alive {
                best_pair_dmg_stats(cb, table, cview, opp, d_slot, side, t_slot)
            } else {
                (0.0, 0.0)
            };
            sw_cache[i] = (t_hp_f, t2f_d, t2f_ko, f2t_d, f2t_ko);
            if f2t_d < best_sw_def_v {
                best_sw_def_v = f2t_d;
                best_sw_def_i = i;
            }
            if t2f_d > best_sw_off_v {
                best_sw_off_v = t2f_d;
                best_sw_off_i = i;
            }
            if t_hp_f > best_sw_hp_v {
                best_sw_hp_v = t_hp_f;
                best_sw_hp_i = i;
            }
        }
    }

    for i in 0..n_acts.min(7) {
        let act = acts[i];
        let row = &mut af[i];
        if act < 4 {
            let m_slot = act as usize;
            if m_slot >= table.mons[a_idx].num_moves as usize {
                continue;
            }
            let ms = &table.move_specs[a_idx][m_slot];
            let (d_max, ko, acc) = move_dmgs[i];
            let d_hp = if foe_hp_f > 0.01 { (d_max / foe_hp_f).min(1.5) } else { 0.0 };
            row[0] = 0.0; // move
            row[1] = d_hp;
            row[2] = d_max.min(1.5);
            row[3] = ko;
            row[4] = (2.0 * d_hp).clamp(0.0, 1.0);
            row[5] = if i == best_move_i && max_legal_move_d > 0.01 { 1.0 } else { 0.0 };
            row[6] = if max_legal_move_d > 0.01 && !ms.is_selfdestruct {
                (d_max / max_legal_move_d).clamp(0.0, 1.0)
            } else {
                0.0
            };
            row[7] = acc;
            row[8] = if ms.priority > 0 { 1.0 } else if ms.priority < 0 { -1.0 } else { 0.0 };
            row[9] = if ms.is_selfdestruct { 1.0 } else { 0.0 };
            row[10] = if ms.is_rest { 1.0 } else { 0.0 };
            row[11] = if ms.is_half_heal || ms.special_kind == SpecialMoveKind::WeatherHeal {
                1.0
            } else {
                0.0
            };
            row[12] = if (ms.special_kind == SpecialMoveKind::Curse && !me_is_ghost)
                || ms.special_kind == SpecialMoveKind::BellyDrum
                || (ms.is_status
                    && ms.boosts_self
                    && ms.boosts.iter().any(|&(st_i, d)| st_i < 5 && d > 0))
            {
                1.0
            } else {
                0.0
            };
            row[13] = if ms.is_substitute { 1.0 } else { 0.0 };
            row[14] = if ms.primary_status == Status::Slp as u8
                && sleep_open
                && foe_st == 0
                && foe.sub_hp() == 0
                && cb.sides[opp].safeguard_turns == 0
            {
                1.0
            } else {
                0.0
            };
            let foe_status_open = foe_alive
                && foe_st == 0
                && foe.sub_hp() == 0
                && cb.sides[opp].safeguard_turns == 0;
            let primary_st_live = if ms.primary_status == Status::Tox as u8
                || ms.primary_status == Status::Psn as u8
            {
                foe_status_open && (table.mons[d_idx].type_mask & table.poison_immune_mask) == 0
            } else if ms.primary_status == Status::Par as u8 {
                foe_status_open && (ms.ignore_immunity || !mon_immune_to_type(table, d_idx, ms.move_type))
            } else if ms.primary_status != 0 && ms.primary_status != Status::Slp as u8 {
                foe_status_open
            } else {
                false
            };
            let grass_bit = 1u32 << (table.grass_type & 31);
            let vol_st_live = foe_alive
                && foe.sub_hp() == 0
                && (((ms.primary_vol & vol_bit::CONFUSION) != 0 && cb.sides[opp].safeguard_turns == 0)
                    || ((ms.primary_vol & vol_bit::LEECH_SEED) != 0
                        && (table.mons[d_idx].type_mask & grass_bit) == 0
                        && (foe.vol_flags & vol_bit::LEECH_SEED) == 0)
                    || ((ms.primary_vol & vol_bit::NIGHTMARE) != 0 && foe_st == Status::Slp as u8)
                    || (ms.special_kind == SpecialMoveKind::Curse
                        && me_is_ghost
                        && (foe.vol_flags & vol_bit::CURSE) == 0));
            row[15] = if primary_st_live || vol_st_live || ms.special_kind == SpecialMoveKind::Swagger {
                1.0
            } else {
                0.0
            };
            row[16] = if matches!(ms.special_kind, SpecialMoveKind::Phaze | SpecialMoveKind::Haze) {
                1.0
            } else {
                0.0
            };
            row[17] = if ms.special_kind == SpecialMoveKind::BatonPass { 1.0 } else { 0.0 };
            row[18] = if ms.special_kind == SpecialMoveKind::Spikes { 1.0 } else { 0.0 };
            row[19] = if ms.is_protect || ms.is_endure { 1.0 } else { 0.0 };
            row[20] = if matches!(
                ms.special_kind,
                SpecialMoveKind::PerishSong
                    | SpecialMoveKind::MeanLook
                    | SpecialMoveKind::DestinyBond
            ) {
                1.0
            } else {
                0.0
            };
            row[21] = if ms.is_sleep_usable { 1.0 } else { 0.0 };
            row[22] = if matches!(
                ms.special_kind,
                SpecialMoveKind::Pursuit
                    | SpecialMoveKind::Counter
                    | SpecialMoveKind::MirrorCoat
                    | SpecialMoveKind::Thief
                    | SpecialMoveKind::PainSplit
            ) {
                1.0
            } else {
                0.0
            };
            row[23] = if ms.sec_chance_256 > 0 {
                (ms.sec_chance_256 as f32 / 256.0).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let pp = me.pp[m_slot];
            row[24] = if pp <= 2 { 1.0 } else if pp <= 4 { 0.5 } else { 0.0 };
            // Average damage against foe's living bench (switch-read signal).
            if !ms.is_status && me_alive {
                let mut b_sum = 0.0f32;
                let mut b_cnt = 0.0f32;
                for bs in 0..3usize {
                    if bs == d_slot || (cb.sides[opp].alive_mask & (1 << bs)) == 0 {
                        continue;
                    }
                    if cb.mons[opp * 3 + bs].hp() <= 0 {
                        continue;
                    }
                    let (bd, _, _) = move_dmg_stats(cb, table, cview, side, a_slot, opp, bs, m_slot);
                    b_sum += bd.min(1.5);
                    b_cnt += 1.0;
                }
                row[25] = if b_cnt > 0.0 { b_sum / b_cnt } else { 0.0 };
            }
            row[26] = if foe_alive
                && !ms.is_status
                && !ms.ignore_immunity
                && ms.base_power > 0
                && (foe.vol_flags & vol_bit::FORESIGHT) == 0
                && mon_immune_to_type(table, d_idx, ms.move_type)
            {
                1.0
            } else {
                0.0
            };
            row[27] = if ms.is_recharge || ms.recoil_quarter { 1.0 } else { 0.0 };
            row[28] = if matches!(
                ms.special_kind,
                SpecialMoveKind::Reflect | SpecialMoveKind::LightScreen | SpecialMoveKind::Safeguard
            ) {
                1.0
            } else {
                0.0
            };
            row[29] = if ms.special_kind == SpecialMoveKind::RapidSpin {
                if spikes_layers(cb, side) > 0
                    || (me.vol_flags & vol_bit::LEECH_SEED) != 0
                    || me.partially_trapped()
                {
                    1.0
                } else {
                    0.25
                }
            } else {
                0.0
            };
            row[30] = if ms.special_kind == SpecialMoveKind::HealBell { 1.0 } else { 0.0 };
            row[31] = if matches!(ms.special_kind, SpecialMoveKind::Encore | SpecialMoveKind::Disable)
            {
                1.0
            } else {
                0.0
            };
        } else if (4..7).contains(&act) {
            let t_slot = (act - 4) as usize;
            let t_idx = side * 3 + t_slot;
            let t_mon = &cb.mons[t_idx];
            let t_maxhp = table.mons[t_idx].maxhp.max(1) as f32;
            let (t_hp_f, t2f_d, t2f_ko, f2t_d, f2t_ko) = sw_cache[i];
            let t_ehp_f = (entry_hp(cb, table, side, t_slot).max(0) as f32 / t_maxhp).clamp(0.0, 1.0);
            let t_st = t_mon.status_raw();
            let t_spd = effective_speed(cb, table, side, t_slot);
            let t_spd_cmp = if t_spd > foe_spd { 1.0 } else if t_spd < foe_spd { -1.0 } else { 0.0 };
            let t2f_hp = if foe_hp_f > 0.01 { (t2f_d / foe_hp_f).min(1.5) } else { 0.0 };
            let f2t_hp = if t_ehp_f > 0.01 { (f2t_d / t_ehp_f).min(1.5) } else { 1.5 };

            let (has_rec, rec_pp_f, has_phaze, has_boom, has_slp, has_setup) = if let Some(cv) = cview {
                let cm = cv.mon(side, t_slot);
                let mut rpp = 0.0f32;
                if (cm.half_heal_slot as usize) < 4 {
                    rpp += t_mon.pp[cm.half_heal_slot as usize] as f32;
                }
                if (cm.rest_slot as usize) < 4 {
                    rpp += 2.0 * (t_mon.pp[cm.rest_slot as usize] as f32);
                }
                let ph = ((cm.phaze_slot as usize) < 4 && t_mon.pp[cm.phaze_slot as usize] > 0)
                    || ((cm.haze_slot as usize) < 4 && t_mon.pp[cm.haze_slot as usize] > 0);
                let n_tm = table.mons[t_idx].num_moves as usize;
                let bm = (0..n_tm).any(|m| table.move_specs[t_idx][m].is_selfdestruct && t_mon.pp[m] > 0);
                let sl = cm.sleep_mask != 0
                    && (0..n_tm).any(|m| ((cm.sleep_mask >> m) & 1) != 0 && t_mon.pp[m] > 0);
                let su = cm.setup_mask != 0
                    && (0..n_tm).any(|m| ((cm.setup_mask >> m) & 1) != 0 && t_mon.pp[m] > 0);
                (rpp > 0.0, (rpp / 20.0).clamp(0.0, 1.0), ph, bm, sl, su)
            } else {
                let mut rpp = 0.0f32;
                let mut ph = false;
                let mut bm = false;
                let mut sl = false;
                let mut su = false;
                let t_ghost = (table.mons[t_idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
                for m in 0..table.mons[t_idx].num_moves as usize {
                    let pp = t_mon.pp[m];
                    if pp == 0 {
                        continue;
                    }
                    let tms = &table.move_specs[t_idx][m];
                    if tms.is_half_heal || tms.special_kind == SpecialMoveKind::WeatherHeal {
                        rpp += pp as f32;
                    }
                    if tms.is_rest {
                        rpp += 2.0 * (pp as f32);
                    }
                    if matches!(tms.special_kind, SpecialMoveKind::Phaze | SpecialMoveKind::Haze) {
                        ph = true;
                    }
                    if tms.is_selfdestruct {
                        bm = true;
                    }
                    if tms.primary_status == Status::Slp as u8 {
                        sl = true;
                    }
                    if (tms.special_kind == SpecialMoveKind::Curse && !t_ghost)
                        || tms.special_kind == SpecialMoveKind::BellyDrum
                        || (tms.is_status
                            && tms.boosts_self
                            && tms.boosts.iter().any(|&(st_i, d)| st_i < 5 && d > 0))
                    {
                        su = true;
                    }
                }
                (rpp > 0.0, (rpp / 20.0).clamp(0.0, 1.0), ph, bm, sl, su)
            };

            let mut bp_dmg = 0.0f32;
            if passing_alive && foe_alive {
                let mut tmp = *cb;
                tmp.mons[t_idx].boosts = me.boosts;
                tmp.mons[t_idx].vol_flags |= me.vol_flags & vol_bit::SUBSTITUTE;
                bp_dmg = (tmp.best_hit_fraction(table, side, t_slot, opp, d_slot) as f32).min(1.5);
            }
            let mut team_d = 0.0f32;
            let mut team_n = 0.0f32;
            for es in 0..3usize {
                if (cb.sides[opp].alive_mask & (1 << es)) != 0 && cb.mons[opp * 3 + es].hp() > 0 {
                    let (ed, _) = best_pair_dmg_stats(cb, table, cview, side, t_slot, opp, es);
                    team_d += ed.min(1.5);
                    team_n += 1.0;
                }
            }

            row[0] = 1.0; // switch
            row[1] = t_hp_f;
            row[2] = t_ehp_f;
            row[3] = (expected_sleep_turns(cb, side, t_slot) as f32 / 4.0).clamp(0.0, 1.0);
            row[4] = if t_st != 0 && t_st != Status::Slp as u8 { 1.0 } else { 0.0 };
            row[5] = t2f_hp;
            row[6] = t2f_d.min(1.5);
            row[7] = t2f_ko;
            row[8] = f2t_hp;
            row[9] = f2t_d.min(1.5);
            row[10] = f2t_ko;
            row[11] = t_spd_cmp;
            row[12] = if has_rec && f2t_d < 0.45 && t_st != Status::Frz as u8 { 1.0 } else { 0.0 };
            row[13] = (t2f_d - my_best_d).clamp(-1.5, 1.5);
            row[14] = (foe_best_d - f2t_d).clamp(-1.5, 1.5);
            row[15] = if has_phaze { 1.0 } else { 0.0 };
            row[16] = if has_boom { 1.0 } else { 0.0 };
            row[17] = if has_slp && sleep_open { 1.0 } else { 0.0 };
            row[18] = if has_setup { 1.0 } else { 0.0 };
            row[19] = if table.item_spec(t_mon.item).is_leftovers { 1.0 } else { 0.0 };
            row[20] = if spikes_layers(cb, side) > 0 && (table.mons[t_idx].type_mask & flying_bit) == 0
            {
                1.0
            } else {
                0.0
            };
            row[21] = if t_ehp_f > f2t_d * 1.1 && t_ehp_f > 0.05 { 1.0 } else { 0.0 };
            row[22] = bp_dmg;
            row[23] = if team_n > 0.0 { team_d / team_n } else { 0.0 };
            row[24] = rec_pp_f;
            row[25] = if i == best_sw_def_i { 1.0 } else { 0.0 };
            row[26] = if i == best_sw_off_i { 1.0 } else { 0.0 };
            row[27] = if i == best_sw_hp_i { 1.0 } else { 0.0 };
        }
    }
    (ctx, af)
}

/// Evaluate embedded `P_net` + `compact_interior_noop_mask` + floor `eps_p` over safe actions.
/// Returns `(priors[7], noop_mask)`. Zero heap allocations (`~200-300 ns` per side).
#[inline]
pub fn policy_priors(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
    side: usize,
    acts: &[u8],
    n_acts: usize,
    eps_p: f32,
) -> ([f32; 7], u8) {
    let mut priors = [0.0f32; 7];
    let n = n_acts.min(7);
    if n == 0 {
        return (priors, 0);
    }
    if n == 1 {
        priors[0] = 1.0;
        return (priors, 0);
    }
    let noop_mask = compact_interior_noop_mask(cb, table, side, acts, n);
    let mut safe_cnt = 0usize;
    let mut last_safe = 0usize;
    for i in 0..n {
        if ((noop_mask >> i) & 1) == 0 {
            safe_cnt += 1;
            last_safe = i;
        }
    }
    if safe_cnt == 1 {
        priors[last_safe] = 1.0;
        return (priors, noop_mask);
    }
    let safe_cnt_f = safe_cnt.max(1) as f32;
    if !POLICY_WEIGHTS_LOADED {
        let u = 1.0 / safe_cnt_f;
        for i in 0..n {
            priors[i] = if ((noop_mask >> i) & 1) == 0 { u } else { 0.0 };
        }
        return (priors, noop_mask);
    }

    let (ctx, af) = extract_policy_features(cb, table, cache, side, acts, n);

    // Layer 1 shared context projection (computed ONCE per side).
    let mut ctx_proj = [0.0f32; POLICY_H1];
    for h in 0..POLICY_H1 {
        let row = &P_W_CTX[h];
        let mut s = P_B1[h];
        for k in 0..POLICY_CTX_DIM {
            s += row[k] * ctx[k];
        }
        ctx_proj[h] = s;
    }

    // Layer 1 per-action activation + mean pooling across safe actions.
    let mut h1 = [[0.0f32; POLICY_H1]; 7];
    let mut h1_mean = [0.0f32; POLICY_H1];
    for i in 0..n {
        if ((noop_mask >> i) & 1) != 0 {
            continue;
        }
        let act_v = &af[i];
        let dst = &mut h1[i];
        for h in 0..POLICY_H1 {
            let row = &P_W_ACT[h];
            let mut s = ctx_proj[h];
            for k in 0..POLICY_ACT_DIM {
                s += row[k] * act_v[k];
            }
            let v = silu(s);
            dst[h] = v;
            h1_mean[h] += v;
        }
    }
    let inv_safe = 1.0 / safe_cnt_f;
    for m in &mut h1_mean {
        *m *= inv_safe;
    }

    // Layer 2 mean-context projection (computed ONCE per side).
    let mut mean_proj = [0.0f32; POLICY_H2];
    for j in 0..POLICY_H2 {
        let row = &P_W2_MEAN[j];
        let mut s = P_B2[j];
        for h in 0..POLICY_H1 {
            s += row[h] * h1_mean[h];
        }
        mean_proj[j] = s;
    }

    // Layer 2 per-action logit + stable softmax over unmasked actions.
    let inv_tau = 1.0 / P_TAU.max(0.1);
    let mut logits = [f32::NEG_INFINITY; 7];
    let mut max_l = f32::NEG_INFINITY;
    for i in 0..n {
        if ((noop_mask >> i) & 1) != 0 {
            continue;
        }
        let h1_i = &h1[i];
        let mut z = 0.0f32;
        for j in 0..POLICY_H2 {
            let row = &P_W2_SELF[j];
            let mut s = mean_proj[j];
            for h in 0..POLICY_H1 {
                s += row[h] * h1_i[h];
            }
            z += P_W_OUT[j] * silu(s);
        }
        let l = z * inv_tau;
        logits[i] = l;
        if l > max_l {
            max_l = l;
        }
    }

    let mut exp_sum = 0.0f32;
    for i in 0..n {
        if ((noop_mask >> i) & 1) == 0 {
            let e = (logits[i] - max_l).clamp(-40.0, 0.0).exp();
            priors[i] = e;
            exp_sum += e;
        }
    }
    let eps = eps_p.clamp(0.0, 0.25);
    let unif = eps * inv_safe;
    let scale = if exp_sum > 1e-12 { (1.0 - eps) / exp_sum } else { 0.0 };
    for i in 0..n {
        if ((noop_mask >> i) & 1) == 0 {
            priors[i] = if scale > 0.0 {
                priors[i] * scale + unif
            } else {
                inv_safe
            };
        } else {
            priors[i] = 0.0;
        }
    }
    (priors, noop_mask)
}

// =========================================================================
// 4. 0-ULP Antisymmetric Siamese Value Feature Extractor & `V_net`
// =========================================================================

#[inline]
fn canonical_side_order(
    cb: &CompactBattle,
    table: &MatchTable,
    ttk: &[[[FastPair; 3]; 3]; 2],
    side: usize,
) -> [usize; 3] {
    let act = (cb.sides[side].active_idx as usize).min(2);
    let opp = 1 - side;
    let mut bench = [0usize; 2];
    let mut nb = 0usize;
    for sl in 0..3usize {
        if sl != act {
            bench[nb] = sl;
            nb += 1;
        }
    }
    let bench_key = |sl: usize| -> (u8, i32, i32, [i16; 5], i16, u8, u8) {
        let alive = u8::from((cb.sides[side].alive_mask & (1 << sl)) != 0 && cb.mons[side * 3 + sl].hp() > 0);
        let idx = side * 3 + sl;
        let maxhp = table.mons[idx].maxhp.max(1);
        let hp_q = if alive != 0 {
            ((cb.mons[idx].hp() as i32) * 10_000) / (maxhp as i32)
        } else {
            -1
        };
        let mut best_d = 0.0f64;
        for e in 0..3usize {
            if (cb.sides[opp].alive_mask & (1 << e)) != 0 && ttk[side][sl][e].raw_dmg > best_d {
                best_d = ttk[side][sl][e].raw_dmg;
            }
        }
        let dmg_q = (best_d * 10_000.0) as i32;
        (
            alive,
            hp_q,
            dmg_q,
            table.mons[idx].stored_stats,
            maxhp,
            cb.mons[idx].status_raw(),
            cb.mons[idx].item,
        )
    };
    if bench_key(bench[1]) > bench_key(bench[0]) {
        bench.swap(0, 1);
    }
    [act, bench[0], bench[1]]
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn encode_value_side(
    cb: &CompactBattle,
    table: &MatchTable,
    fg: &FastGolfEval,
    ttk: &[[[FastPair; 3]; 3]; 2],
    entry_hps: &[i16; 6],
    g_stag: f32,
    side: usize,
    ord: &[[usize; 3]; 2],
    cview: Option<&CacheView<'_>>,
) -> [f32; VALUE_SIDE_DIM] {
    let mut x = [0.0f32; VALUE_SIDE_DIM];
    let opp = 1 - side;
    let act = ord[side][0];
    let a_idx = side * 3 + act;
    let a_mon = &cb.mons[a_idx];
    let a_alive = (cb.sides[side].alive_mask & (1 << act)) != 0 && a_mon.hp() > 0;
    let a_maxhp = table.mons[a_idx].maxhp.max(1) as f32;

    let mon_rec_pp = |slot: usize| -> (f32, f32, f32, f32, f32) {
        let idx = side * 3 + slot;
        let m = &cb.mons[idx];
        let mut h_pp = 0.0f32;
        let mut r_pp = 0.0f32;
        let mut has_boom = 0.0f32;
        let mut has_setup = 0.0f32;
        let mut has_phaze = 0.0f32;
        let n_m = table.mons[idx].num_moves as usize;
        if let Some(cv) = cview {
            let cm = cv.mon(side, slot);
            if (cm.half_heal_slot as usize) < 4 {
                h_pp = m.pp[cm.half_heal_slot as usize] as f32;
            }
            if (cm.rest_slot as usize) < 4 {
                r_pp = m.pp[cm.rest_slot as usize] as f32;
            }
            if (0..n_m).any(|mi| table.move_specs[idx][mi].is_selfdestruct && m.pp[mi] > 0) {
                has_boom = 1.0;
            }
            if cm.setup_mask != 0 && (0..n_m).any(|mi| ((cm.setup_mask >> mi) & 1) != 0 && m.pp[mi] > 0)
            {
                has_setup = 1.0;
            }
            if ((cm.phaze_slot as usize) < 4 && m.pp[cm.phaze_slot as usize] > 0)
                || ((cm.haze_slot as usize) < 4 && m.pp[cm.haze_slot as usize] > 0)
            {
                has_phaze = 1.0;
            }
        } else {
            let is_g = (table.mons[idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
            for mi in 0..n_m {
                let pp = m.pp[mi];
                if pp == 0 {
                    continue;
                }
                let ms = &table.move_specs[idx][mi];
                if ms.is_half_heal || ms.special_kind == SpecialMoveKind::WeatherHeal {
                    h_pp = h_pp.max(pp as f32);
                }
                if ms.is_rest {
                    r_pp = r_pp.max(pp as f32);
                }
                if ms.is_selfdestruct {
                    has_boom = 1.0;
                }
                if (ms.special_kind == SpecialMoveKind::Curse && !is_g)
                    || ms.special_kind == SpecialMoveKind::BellyDrum
                    || (ms.is_status
                        && ms.boosts_self
                        && ms.boosts.iter().any(|&(st_i, d)| st_i < 5 && d > 0))
                {
                    has_setup = 1.0;
                }
                if matches!(ms.special_kind, SpecialMoveKind::Phaze | SpecialMoveKind::Haze) {
                    has_phaze = 1.0;
                }
            }
        }
        (
            (h_pp / 16.0).clamp(0.0, 1.0),
            (r_pp / 10.0).clamp(0.0, 1.0),
            has_boom,
            has_setup,
            has_phaze,
        )
    };

    // 1. Active mon (18 floats: 0..18)
    let (a_hpp, a_rpp, a_boom, a_setup, a_phaze) = mon_rec_pp(act);
    if a_alive {
        let st = a_mon.status_raw();
        let cnt = a_mon.status_counter();
        x[0] = 1.0;
        x[1] = (a_mon.hp() as f32 / a_maxhp).clamp(0.0, 1.0);
        x[2] = (entry_hps[a_idx].max(0) as f32 / a_maxhp).clamp(0.0, 1.0);
        x[3] = (expected_sleep_turns(cb, side, act) as f32 / 4.0).clamp(0.0, 1.0);
        x[4] = if st == Status::Par as u8 { 1.0 } else { 0.0 };
        x[5] = if st == Status::Brn as u8 { 1.0 } else { 0.0 };
        x[6] = if st == Status::Psn as u8 { 1.0 } else { 0.0 };
        x[7] = if st == Status::Tox as u8 {
            ((cnt & 0x0F) as f32 + 1.0).min(8.0) / 8.0
        } else {
            0.0
        };
        x[8] = if st == Status::Frz as u8 { 1.0 } else { 0.0 };
        x[9] = (a_mon.boosts[0] as f32 / 4.0).clamp(-1.5, 1.5);
        x[10] = (a_mon.boosts[1] as f32 / 4.0).clamp(-1.5, 1.5);
        x[11] = (a_mon.boosts[2] as f32 / 4.0).clamp(-1.5, 1.5);
        x[12] = (a_mon.boosts[3] as f32 / 4.0).clamp(-1.5, 1.5);
        x[13] = (a_mon.boosts[4] as f32 / 4.0).clamp(-1.5, 1.5);
        x[14] = ((a_mon.sub_hp() as f32 / a_maxhp) * 4.0).clamp(0.0, 1.0);
        x[15] = if table.item_spec(a_mon.item).is_leftovers { 1.0 } else { 0.0 };
        x[16] = a_hpp;
        x[17] = a_rpp;
    }

    // 2. Canonical bench 1 & 2 (2 * 14 = 28 floats: 18..46)
    let mut rec_res_sum = if a_alive {
        (1.0 + (a_hpp * 16.0 + 2.2 * a_rpp * 10.0) / 8.0).ln()
    } else {
        0.0
    };
    for bi in 0..2usize {
        let base = 18 + bi * 14;
        let sl = ord[side][bi + 1];
        let idx = side * 3 + sl;
        let m = &cb.mons[idx];
        let alive = (cb.sides[side].alive_mask & (1 << sl)) != 0 && m.hp() > 0;
        if !alive {
            continue;
        }
        let maxhp = table.mons[idx].maxhp.max(1) as f32;
        let st = m.status_raw();
        let ispec = table.item_spec(m.item);
        let (hpp, rpp, boom, setup, phaze) = mon_rec_pp(sl);
        rec_res_sum += (1.0 + (hpp * 16.0 + 2.2 * rpp * 10.0) / 8.0).ln();
        x[base] = 1.0;
        x[base + 1] = (m.hp() as f32 / maxhp).clamp(0.0, 1.0);
        x[base + 2] = (entry_hps[idx].max(0) as f32 / maxhp).clamp(0.0, 1.0);
        x[base + 3] = (expected_sleep_turns(cb, side, sl) as f32 / 4.0).clamp(0.0, 1.0);
        x[base + 4] = if st == Status::Par as u8 { 1.0 } else { 0.0 };
        x[base + 5] = if st == Status::Brn as u8 || st == Status::Psn as u8 || st == Status::Tox as u8
        {
            1.0
        } else {
            0.0
        };
        x[base + 6] = if st == Status::Frz as u8 { 1.0 } else { 0.0 };
        x[base + 7] = if ispec.is_leftovers { 1.0 } else { 0.0 };
        x[base + 8] = if ispec.is_miracleberry || ispec.is_mintberry || ispec.is_goldberry {
            1.0
        } else {
            0.0
        };
        x[base + 9] = hpp;
        x[base + 10] = rpp;
        x[base + 11] = boom;
        x[base + 12] = setup;
        x[base + 13] = phaze;
    }

    // 3. Ordered 3x3 Matchup Tensor: ord[side][i] -> ord[opp][j] (9 * 4 = 36 floats: 46..82)
    let mut off_i = 46usize;
    let mut max_boom_dmg = 0.0f32;
    let my_alive_cnt = cb.sides[side].alive_mask.count_ones();
    for i in 0..3usize {
        let a = ord[side][i];
        let a_ok = (cb.sides[side].alive_mask & (1 << a)) != 0 && cb.mons[side * 3 + a].hp() > 0;
        for j in 0..3usize {
            let e = ord[opp][j];
            let e_ok = (cb.sides[opp].alive_mask & (1 << e)) != 0 && cb.mons[opp * 3 + e].hp() > 0;
            if a_ok && e_ok {
                let p = ttk[side][a][e];
                x[off_i] = (1.0 / (1.0 + p.eff_ttk as f32)).clamp(0.0, 1.0);
                x[off_i + 1] = (p.raw_dmg as f32).clamp(0.0, 1.5);
                x[off_i + 2] = (p.ohko_prob as f32).clamp(0.0, 1.0);
                x[off_i + 3] = if p.def_sustain { 1.0 } else { 0.0 };
                if my_alive_cnt >= 2 && (p.boom_dmg_frac as f32) > max_boom_dmg {
                    max_boom_dmg = p.boom_dmg_frac as f32;
                }
            }
            off_i += 4;
        }
    }

    // 4. Global side summary (12 floats: 82..94)
    let opp_alive_cnt = cb.sides[opp].alive_mask.count_ones();
    let opp_asleep = (0..3usize).any(|sl| {
        (cb.sides[opp].alive_mask & (1 << sl)) != 0
            && cb.mons[opp * 3 + sl].hp() > 0
            && cb.mons[opp * 3 + sl].status_raw() == Status::Slp as u8
    });
    let opp_act = ord[opp][0];
    let act_spd_cmp = if a_alive
        && (cb.sides[opp].alive_mask & (1 << opp_act)) != 0
        && cb.mons[opp * 3 + opp_act].hp() > 0
    {
        ttk[side][act][opp_act].speed_cmp as f32
    } else {
        0.0
    };
    let my_perish = if a_alive && (a_mon.vol_flags & vol_bit::PERISH_SONG) != 0 {
        match a_mon.perish_turns() {
            1 => 1.0,
            2 => 0.5,
            3 => 0.25,
            _ => 0.0,
        }
    } else {
        0.0
    };

    x[82] = (my_alive_cnt as f32) / 3.0;
    x[83] = if opp_alive_cnt >= 2 && spikes_layers(cb, opp) > 0 { 1.0 } else { 0.0 };
    x[84] = (cb.sides[side].reflect_turns as f32 / 5.0).clamp(0.0, 1.0);
    x[85] = (cb.sides[side].lightscreen_turns as f32 / 5.0).clamp(0.0, 1.0);
    x[86] = (cb.sides[side].safeguard_turns as f32 / 5.0).clamp(0.0, 1.0);
    x[87] = if opp_asleep { 1.0 } else { 0.0 };
    x[88] = (1.0 / (1.0 + (fg.g_turns[side] as f32) / 6.0)).clamp(0.0, 1.0);
    x[89] = act_spd_cmp;
    x[90] = my_perish;
    x[91] = g_stag;
    x[92] = (rec_res_sum / 4.0).clamp(0.0, 1.5);
    x[93] = max_boom_dmg.min(1.5);
    if a_alive {
        let a_ispec = table.item_spec(a_mon.item);
        x[94] = if a_ispec.is_miracleberry
            || a_ispec.is_mintberry
            || a_ispec.is_goldberry
            || a_ispec.is_berry_juice
        {
            1.0
        } else {
            0.0
        };
        x[95] = a_boom;
        x[96] = a_setup;
        x[97] = a_phaze;
        x[98] = if a_mon.is_trapped() || (a_mon.vol_flags & vol_bit::ENCORE) != 0 {
            1.0
        } else {
            0.0
        };
        x[99] = if (a_mon.vol_flags & vol_bit::DESTINY_BOND) != 0 { 1.0 } else { 0.0 };
    }
    x
}

/// Extract strictly 0-ULP side-swap symmetric `(x0, x1, logit_fg, logit_bs_nr)`.
#[inline]
pub fn extract_value_features(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
) -> ([f32; VALUE_SIDE_DIM], [f32; VALUE_SIDE_DIM], f32, f32) {
    let w = EvalWeights::default();
    let bs_nr = compact_eval01_blindspot_norelay(cb, table, &w);
    let (fg, ttk, entry_hps) = compute_fast_golf_core(cb, table, cache);
    let logit_fg = stage3_logit(fg.win_prob) as f32;
    let logit_bs = stage3_logit(bs_nr) as f32;

    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    let mut wall_hard = [0.0f64; 2];
    for s in 0..2usize {
        let opp = 1 - s;
        let mut hard = 0.0f64;
        for e in 0..3usize {
            if (cb.sides[opp].alive_mask & (1 << e)) == 0 {
                continue;
            }
            let mut best_b = 7.5f64;
            for a in 0..3usize {
                if (cb.sides[s].alive_mask & (1 << a)) == 0 {
                    continue;
                }
                let p = ttk[s][a][e];
                let mut t = p.eff_ttk;
                if cb.sides[s].alive_mask.count_ones() >= 2 && p.boom_dmg_frac >= 0.72 {
                    t = t.min(1.6 / p.boom_dmg_frac.max(0.5));
                }
                if t < best_b {
                    best_b = t;
                }
            }
            if best_b > hard {
                hard = best_b;
            }
        }
        wall_hard[s] = hard;
    }
    let g_stag = if n0 + n1 >= 4 {
        ((wall_hard[0].min(wall_hard[1]) - 4.2) / 2.8).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };

    let ord = [
        canonical_side_order(cb, table, &ttk, 0),
        canonical_side_order(cb, table, &ttk, 1),
    ];
    let view_opt = cache.and_then(|c| c.view_for(cb));
    let cview = view_opt.as_ref();
    let x0 = encode_value_side(cb, table, &fg, &ttk, &entry_hps, g_stag, 0, &ord, cview);
    let x1 = encode_value_side(cb, table, &fg, &ttk, &entry_hps, g_stag, 1, &ord, cview);
    (x0, x1, logit_fg, logit_bs)
}

/// Evaluate embedded 0-ULP antisymmetric Siamese `V_net` on `CompactBattle`.
/// Preserves exact `1v1` `linear.rs` valuation and wraps outer `relay::apply_relay_leaf_lift`.
#[inline]
pub fn evaluate_az_value(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
    leaf_temp: f64,
) -> f64 {
    if let Some(v) = compact_turn_limit_outcome(cb, table) {
        return v;
    }
    let w = EvalWeights::default();
    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    if n0 <= 1 && n1 <= 1 {
        return compact_eval01_blindspot_norelay(cb, table, &w);
    }
    if !VALUE_WEIGHTS_LOADED {
        return evaluate_hybrid_golf_temp(cb, table, cache, leaf_temp);
    }

    let (x0, x1, logit_fg, logit_bs) = extract_value_features(cb, table, cache);

    let mut h0_1 = [0.0f32; VALUE_H1];
    let mut h1_1 = [0.0f32; VALUE_H1];
    for h in 0..VALUE_H1 {
        let ws = &V_W1_SELF[h];
        let wo = &V_W1_OPP[h];
        let mut s0 = V_B1[h];
        let mut s1 = V_B1[h];
        for k in 0..VALUE_SIDE_DIM {
            s0 += ws[k] * x0[k] + wo[k] * x1[k];
            s1 += ws[k] * x1[k] + wo[k] * x0[k];
        }
        h0_1[h] = silu(s0);
        h1_1[h] = silu(s1);
    }

    let mut z = V_SKIP_FG * logit_fg + V_SKIP_BS * logit_bs;
    for j in 0..VALUE_H2 {
        let ws = &V_W2_SELF[j];
        let wo = &V_W2_OPP[j];
        let mut s0 = V_B2[j];
        let mut s1 = V_B2[j];
        for h in 0..VALUE_H1 {
            s0 += ws[h] * h0_1[h] + wo[h] * h1_1[h];
            s1 += ws[h] * h1_1[h] + wo[h] * h0_1[h];
        }
        let d = silu(s0) - silu(s1);
        z += V_W_OUT[j] * d;
    }

    let z64 = (f64::from(z) / leaf_temp.max(0.1)).clamp(-20.0, 20.0);
    let v_base = 1.0 / (1.0 + (-z64).exp());
    crate::relay::apply_relay_leaf_lift(cb, table, &w, v_base).clamp(0.000_1, 0.999_9)
}

// =========================================================================
// 5. Fast Full-Info `CompactBattle` Teacher Search (for Gen-1 & Gen-2 Data)
// =========================================================================

#[derive(Clone, Debug)]
pub struct CompactTeacherResult {
    pub acts: [[u8; 7]; 2],
    pub n_acts: [u8; 2],
    pub visits: [[u32; 7]; 2],
    pub q_side: [[f64; 7]; 2],
    pub v_root0: f64,
    pub best_acts: [u8; 2],
}

/// Fast `CompactBattle`-native full-information teacher using shipped `smmcts::run_compact_iteration`
/// (`c = 0.4`, `EvalBlindspot + 8r + perish_combo + baton_pass + relay`, `hp_buckets = 16`).
pub fn search_compact_teacher(
    base_cb: &CompactBattle,
    table: &MatchTable,
    iterations: u32,
    seed: u64,
) -> CompactTeacherResult {
    let cfg = RmConfig {
        iterations,
        rule: SelRule::Ucb,
        c: 0.4,
        ..RmConfig::default()
    };
    let mut turn_cap = base_cb.turn.saturating_add(cfg.horizon);
    if let Some(lim) = table.ruleset.turn_limit() {
        turn_cap = turn_cap.min(lim);
    }
    let mut root_cb = *base_cb;
    let root_idle = root_cb.sides[0].request == 1 && root_cb.sides[1].request == 1;
    if root_idle {
        root_cb.quick_claw_roll = 0;
    }
    let root_node = CompactNode::at(&root_cb, table);
    let omit_qc = root_idle || !root_cb.quick_claw_observable(table);
    let root_key = root_cb.search_key(table, cfg.hp_buckets, omit_qc);

    let mut nodes = Vec::with_capacity((iterations as usize / 2).clamp(64, 65_536));
    let mut tt: FxHashMap<u64, usize> = FxHashMap::default();
    nodes.push(root_node);
    tt.insert(root_key, 0);

    let mut rng = SplitMix64::new(seed ^ 0xA24B_AED4_963E_E407);
    let mut rj = [0usize; 2];
    let mut depth_out = 0u32;

    for _ in 0..iterations {
        let mut cb = root_cb;
        let mut prng = nc2000_engine::prng::Prng::new(rng.next());
        let qc = nc2000_engine::state::Battle::roll_quick_claw_with(table.ruleset, &mut prng);
        cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
        cb.prng_seed = prng.seed();
        run_compact_iteration(
            &cfg,
            &mut rng,
            &mut nodes,
            &mut tt,
            &mut cb,
            table,
            turn_cap,
            0,
            [None, None],
            &mut rj,
            &mut depth_out,
        );
    }

    let rn = &nodes[0];
    let mut acts = [[0u8; 7]; 2];
    let mut n_acts = [0u8; 2];
    let mut visits = [[0u32; 7]; 2];
    let mut q_side = [[0.5f64; 7]; 2];
    let mut best_acts = [7u8; 2];

    for s in 0..2usize {
        let k = (rn.len[s] as usize).min(6);
        n_acts[s] = k as u8;
        acts[s][..k].copy_from_slice(&rn.acts[s][..k]);
        visits[s][..k].copy_from_slice(&rn.n[s][..k]);
        for i in 0..k {
            if visits[s][i] > 0 {
                q_side[s][i] = rn.w[s][i] / f64::from(visits[s][i]);
            }
        }
        if k > 0 {
            let noop = compact_interior_noop_mask(&root_cb, table, s, &acts[s], k);
            let mut best_i = 0usize;
            let mut best_v = 0u32;
            let mut best_q = f64::NEG_INFINITY;
            let mut found = false;
            for i in 0..k {
                if ((noop >> i) & 1) != 0 {
                    continue;
                }
                let vi = visits[s][i];
                let qi = q_side[s][i];
                if !found || vi > best_v || (vi == best_v && qi > best_q) {
                    found = true;
                    best_i = i;
                    best_v = vi;
                    best_q = qi;
                }
            }
            best_acts[s] = acts[s][best_i];
        }
    }

    let n0_tot: u32 = visits[0][..n_acts[0] as usize].iter().sum();
    let n1_tot: u32 = visits[1][..n_acts[1] as usize].iter().sum();
    let v_raw0 = if n0_tot > 0 {
        let w0: f64 = rn.w[0][..n_acts[0] as usize].iter().sum();
        (w0 / f64::from(n0_tot)).clamp(0.0, 1.0)
    } else if n1_tot > 0 {
        let w1: f64 = rn.w[1][..n_acts[1] as usize].iter().sum();
        (1.0 - w1 / f64::from(n1_tot)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let v_root0 = if v_raw0 > 0.000_001 && v_raw0 < 0.999_999 {
        rn.backed_up_value(v_raw0)
    } else {
        v_raw0
    };

    CompactTeacherResult {
        acts,
        n_acts,
        visits,
        q_side,
        v_root0,
        best_acts,
    }
}

/// Fast full-information Compact-AlphaZero-DUCT teacher (`PUCT + P_net + V_net + 0r/2r`) for Gen-2 Expert Iteration.
#[allow(clippy::too_many_arguments)]
pub fn search_compact_az_teacher(
    base_cb: &CompactBattle,
    table: &MatchTable,
    cache: &TempoMatchCache,
    iterations: u32,
    rollout_turns: u16,
    c_puct: f64,
    fpu_red: f64,
    puct_eps: f32,
    backup_lambda: f64,
    backup_tau: f64,
    seed: u64,
) -> CompactTeacherResult {
    #[derive(Clone, Copy)]
    struct AzNode {
        n_acts: [u8; 2],
        acts: [[u8; 7]; 2],
        visits: [[u32; 7]; 2],
        sums: [[f64; 7]; 2],
        priors: [[f32; 7]; 2],
        noop_mask: [u8; 2],
        priors_ready: bool,
        total: u32,
    }
    impl AzNode {
        fn new(cb: &CompactBattle, table: &MatchTable) -> Self {
            let mut acts = [[0u8; 7]; 2];
            let n0 = cb.legal_actions(table, 0, &mut acts[0]).min(7) as u8;
            let n1 = cb.legal_actions(table, 1, &mut acts[1]).min(7) as u8;
            Self {
                n_acts: [n0, n1],
                acts,
                visits: [[0; 7]; 2],
                sums: [[0.0; 7]; 2],
                priors: [[0.0; 7]; 2],
                noop_mask: [0; 2],
                priors_ready: false,
                total: 0,
            }
        }
        #[inline]
        fn ensure_priors(
            &mut self,
            cb: &CompactBattle,
            table: &MatchTable,
            cache: &TempoMatchCache,
            eps_p: f32,
        ) {
            if self.priors_ready {
                return;
            }
            for s in 0..2usize {
                let k = self.n_acts[s] as usize;
                if k > 1 {
                    let (p, m) = policy_priors(cb, table, Some(cache), s, &self.acts[s], k, eps_p);
                    self.priors[s] = p;
                    self.noop_mask[s] = m;
                } else if k == 1 {
                    self.priors[s][0] = 1.0;
                }
            }
            self.priors_ready = true;
        }
        #[inline]
        fn select_puct(&self, side: usize, c_puct: f64, fpu_red: f64) -> usize {
            let n = self.n_acts[side] as usize;
            if n <= 1 {
                return 0;
            }
            let mask = self.noop_mask[side];
            if self.total == 0 {
                let mut best_i = 0usize;
                let mut best_p = f32::NEG_INFINITY;
                for i in 0..n {
                    if ((mask >> i) & 1) == 0 && self.priors[side][i] > best_p {
                        best_p = self.priors[side][i];
                        best_i = i;
                    }
                }
                return best_i;
            }
            let mut n_tot = 0u32;
            let mut w_tot = 0.0f64;
            for i in 0..n {
                n_tot += self.visits[side][i];
                w_tot += self.sums[side][i];
            }
            let r_bar0 = if n_tot > 0 { w_tot / f64::from(n_tot) } else { 0.5 };
            let r_bar_s = if side == 0 { r_bar0 } else { 1.0 - r_bar0 };
            let fpu = (r_bar_s - fpu_red).clamp(0.01, 0.99);
            let sqrt_n = f64::from(self.total.max(1)).sqrt();
            let mut best_i = 0usize;
            let mut best_u = f64::NEG_INFINITY;
            for i in 0..n {
                if ((mask >> i) & 1) != 0 {
                    continue;
                }
                let vi = self.visits[side][i];
                let q = if vi > 0 {
                    let raw = self.sums[side][i] / f64::from(vi);
                    if side == 0 { raw } else { 1.0 - raw }
                } else {
                    fpu
                };
                let u = q + c_puct * f64::from(self.priors[side][i]) * sqrt_n / f64::from(vi + 1);
                if u > best_u {
                    best_u = u;
                    best_i = i;
                }
            }
            best_i
        }
        #[inline]
        fn backed_up_value(&self, raw_val0: f64, lambda: f64, tau: f64) -> f64 {
            if lambda <= 0.0 {
                return raw_val0;
            }
            let k0 = self.n_acts[0] as usize;
            let k1 = self.n_acts[1] as usize;
            let mut n0 = 0u32;
            let mut w0 = 0.0f64;
            for i in 0..k0 {
                n0 += self.visits[0][i];
                w0 += self.sums[0][i];
            }
            let mut n1 = 0u32;
            let mut w1_s0 = 0.0f64;
            for j in 0..k1 {
                n1 += self.visits[1][j];
                w1_s0 += self.sums[1][j];
            }
            let n = n0.max(n1);
            if n < 2 {
                return raw_val0;
            }
            let r_bar0 = if n0 > 0 {
                w0 / f64::from(n0)
            } else if n1 > 0 {
                w1_s0 / f64::from(n1)
            } else {
                return raw_val0;
            };
            let gain0 = if k0 > 1 && n0 >= 2 {
                let mut mu0 = r_bar0;
                for i in 0..k0 {
                    let na = self.visits[0][i];
                    if na > 0 {
                        let est = (self.sums[0][i] + tau * r_bar0) / (f64::from(na) + tau);
                        if est > mu0 {
                            mu0 = est;
                        }
                    }
                }
                (mu0 - r_bar0).max(0.0)
            } else {
                0.0
            };
            let blunder1 = if k1 > 1 && n1 >= 2 {
                let r_bar1 = 1.0 - (w1_s0 / f64::from(n1));
                let mut mu1 = r_bar1;
                for j in 0..k1 {
                    let nb = self.visits[1][j];
                    if nb > 0 {
                        let wb1 = f64::from(nb) - self.sums[1][j];
                        let est = (wb1 + tau * r_bar1) / (f64::from(nb) + tau);
                        if est > mu1 {
                            mu1 = est;
                        }
                    }
                }
                (mu1 - r_bar1).max(0.0)
            } else {
                0.0
            };
            let v_star = (r_bar0 + gain0 - blunder1).clamp(0.000_002, 0.999_998);
            let lam = lambda * (f64::from(n) / (f64::from(n) + 6.0));
            ((1.0 - lam) * raw_val0 + lam * v_star).clamp(0.000_002, 0.999_998)
        }
    }

    let mut turn_cap = base_cb.turn.saturating_add(100);
    if let Some(lim) = table.ruleset.turn_limit() {
        turn_cap = turn_cap.min(lim);
    }
    let mut root_cb = *base_cb;
    let root_idle = root_cb.sides[0].request == 1 && root_cb.sides[1].request == 1;
    if root_idle {
        root_cb.quick_claw_roll = 0;
    }
    let mut rn0 = AzNode::new(&root_cb, table);
    rn0.ensure_priors(&root_cb, table, cache, puct_eps);
    let omit_qc = root_idle || !root_cb.quick_claw_observable(table);
    let root_key = root_cb.search_key(table, 16, omit_qc);

    let mut nodes = Vec::with_capacity((iterations as usize / 2).clamp(64, 65_536));
    let mut tt: FxHashMap<u64, usize> = FxHashMap::default();
    nodes.push(rn0);
    tt.insert(root_key, 0);

    let mut rng = SplitMix64::new(seed ^ 0xA24B_AED4_963E_E407);
    let rollout_playout = crate::mcts::Playout::Heavy {
        turns: rollout_turns,
        eps: 0.2,
        weights: EvalWeights::default(),
    };
    let rollout_rules = RmConfig::default().rollout_rules();
    let mut path: Vec<(usize, usize, usize)> = Vec::with_capacity(64);

    for _ in 0..iterations {
        let mut cb = root_cb;
        let mut prng = nc2000_engine::prng::Prng::new(rng.next());
        let qc = nc2000_engine::state::Battle::roll_quick_claw_with(table.ruleset, &mut prng);
        cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
        cb.prng_seed = prng.seed();

        let mut cur_id = 0usize;
        let mut depth = 0u8;
        path.clear();

        let leaf_val = loop {
            nodes[cur_id].ensure_priors(&cb, table, cache, puct_eps);
            let node = &nodes[cur_id];
            let n0_cur = node.n_acts[0] as usize;
            let n1_cur = node.n_acts[1] as usize;
            let i0 = node.select_puct(0, c_puct, fpu_red);
            let i1 = node.select_puct(1, c_puct, fpu_red);
            let a0 = if n0_cur > 0 { node.acts[0][i0] } else { 7 };
            let a1 = if n1_cur > 0 { node.acts[1][i1] } else { 7 };
            if depth == 0 || n0_cur > 1 || n1_cur > 1 {
                path.push((cur_id, i0, i1));
            }
            cb.apply_actions(table, [a0, a1]);
            depth = depth.saturating_add(1);
            if cb.outcome != 0 {
                break match cb.outcome {
                    1 => 1.0,
                    2 => 0.0,
                    _ => 0.5,
                };
            }
            if let Some(v) = compact_turn_limit_outcome(&cb, table) {
                break v;
            }
            if depth >= 250 || cb.turn > turn_cap {
                break evaluate_az_value(&cb, table, Some(cache), 1.0);
            }
            let oq = !cb.quick_claw_observable(table);
            let nk = cb.search_key(table, 16, oq);
            if let Some(&nid) = tt.get(&nk) {
                cur_id = nid;
            } else {
                let nid = nodes.len();
                nodes.push(AzNode::new(&cb, table));
                tt.insert(nk, nid);
                if rollout_turns == 0 {
                    if cb.sides[0].request == 2 || cb.sides[1].request == 2 {
                        crate::tempo::resolve_compact_forced_switches(
                            &mut cb,
                            table,
                            &EvalWeights::default(),
                        );
                    }
                    break evaluate_az_value(&cb, table, Some(cache), 1.0);
                } else {
                    let cutoff = turn_cap.min(cb.turn.saturating_add(rollout_turns.saturating_sub(1)));
                    break crate::mcts::compact_playout_with_leaf(
                        &mut cb,
                        table,
                        &rollout_playout,
                        cutoff,
                        &mut rng,
                        rollout_rules,
                        |cb_l, t_l| evaluate_az_value(cb_l, t_l, Some(cache), 1.0),
                    );
                }
            }
        };

        let mut val = leaf_val;
        for &(nid, i0, i1) in path.iter().rev() {
            let n = &mut nodes[nid];
            n.total += 1;
            if n.n_acts[0] > 0 {
                n.visits[0][i0] += 1;
                n.sums[0][i0] += val;
            }
            if n.n_acts[1] > 0 {
                n.visits[1][i1] += 1;
                n.sums[1][i1] += val;
            }
            if nid != 0 && val > 0.000_001 && val < 0.999_999 {
                val = n.backed_up_value(val, backup_lambda, backup_tau);
            }
        }
    }

    let rn = &nodes[0];
    let mut q_side = [[0.5f64; 7]; 2];
    let mut best_acts = [7u8; 2];
    for s in 0..2usize {
        let k = rn.n_acts[s] as usize;
        for i in 0..k {
            if rn.visits[s][i] > 0 {
                let raw = rn.sums[s][i] / f64::from(rn.visits[s][i]);
                q_side[s][i] = if s == 0 { raw } else { 1.0 - raw };
            }
        }
        if k > 0 {
            let mut best_i = 0usize;
            let mut best_v = 0u32;
            let mut best_q = f64::NEG_INFINITY;
            let mut found = false;
            for i in 0..k {
                if ((rn.noop_mask[s] >> i) & 1) != 0 {
                    continue;
                }
                let vi = rn.visits[s][i];
                let qi = q_side[s][i];
                if !found || vi > best_v || (vi == best_v && qi > best_q) {
                    found = true;
                    best_i = i;
                    best_v = vi;
                    best_q = qi;
                }
            }
            best_acts[s] = rn.acts[s][best_i];
        }
    }
    let n0_tot: u32 = rn.visits[0][..rn.n_acts[0] as usize].iter().sum();
    let n1_tot: u32 = rn.visits[1][..rn.n_acts[1] as usize].iter().sum();
    let v_root0 = if n0_tot > 0 {
        let w0: f64 = rn.sums[0][..rn.n_acts[0] as usize].iter().sum();
        (w0 / f64::from(n0_tot)).clamp(0.0, 1.0)
    } else if n1_tot > 0 {
        let w0: f64 = rn.sums[1][..rn.n_acts[1] as usize].iter().sum();
        (w0 / f64::from(n1_tot)).clamp(0.0, 1.0)
    } else {
        0.5
    };
    CompactTeacherResult {
        acts: rn.acts,
        n_acts: rn.n_acts,
        visits: rn.visits,
        q_side,
        v_root0,
        best_acts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use nc2000_engine::dex::Dex;
    use nc2000_engine::state::Battle;
    use serde_json::json;

    fn mk(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
        serde_json::from_value(json!({
            "name": species,
            "species": species,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "level": level,
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
            "gender": "M"
        }))
        .unwrap()
    }

    fn start_b(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Battle {
        let mut b = Battle::from_fixture(dex, "1,2,3,4", p1, p2).unwrap();
        let pk = |n| match n {
            1 => "team 1",
            2 => "team 1,2",
            _ => "team 1,2,3",
        };
        b.choose(dex, 0, pk(p1.len())).unwrap();
        b.choose(dex, 1, pk(p2.len())).unwrap();
        b
    }

    #[test]
    fn compact_az_symmetry_bench_invariance_and_noop_soundness() {
        let dex = conformance::load_dex();
        let t1 = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
        ];
        let t2 = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
        ];

        // 1. Exact 0-ULP side-swap symmetry of value features, value eval, and policy features.
        let mut b_ab = start_b(&dex, &t1, &t2);
        b_ab.choose(&dex, 0, "move 3").unwrap(); // Curse
        b_ab.choose(&dex, 1, "move 1").unwrap(); // Thunderbolt
        let mut b_ba = start_b(&dex, &t2, &t1);
        b_ba.choose(&dex, 0, "move 1").unwrap();
        b_ba.choose(&dex, 1, "move 3").unwrap();

        let (tbl_ab, cb_ab) = CompactBattle::from_battle(&b_ab, &dex);
        let (tbl_ba, cb_ba) = CompactBattle::from_battle(&b_ba, &dex);
        let c_ab = TempoMatchCache::build(&cb_ab, &tbl_ab);
        let c_ba = TempoMatchCache::build(&cb_ba, &tbl_ba);

        let (x0_ab, x1_ab, fg_ab, bs_ab) = extract_value_features(&cb_ab, &tbl_ab, Some(&c_ab));
        let (x0_ba, x1_ba, fg_ba, bs_ba) = extract_value_features(&cb_ba, &tbl_ba, Some(&c_ba));
        assert_eq!(x0_ab, x1_ba, "x0(A,B) must equal x1(B,A) bit-for-bit");
        assert_eq!(x1_ab, x0_ba, "x1(A,B) must equal x0(B,A) bit-for-bit");
        assert_eq!(fg_ab, -fg_ba, "logit_fg must negate bit-for-bit");
        assert_eq!(bs_ab, -bs_ba, "logit_bs must negate bit-for-bit");

        let v_ab = evaluate_az_value(&cb_ab, &tbl_ab, Some(&c_ab), 1.0);
        let v_ba = evaluate_az_value(&cb_ba, &tbl_ba, Some(&c_ba), 1.0);
        assert!(
            (v_ab + v_ba - 1.0).abs() < 1e-12,
            "evaluate_az_value must sum to 1.0: {v_ab} + {v_ba}"
        );

        let mut acts0 = [0u8; 7];
        let n0 = cb_ab.legal_actions(&tbl_ab, 0, &mut acts0);
        let mut acts1 = [0u8; 7];
        let n1 = cb_ba.legal_actions(&tbl_ba, 1, &mut acts1);
        assert_eq!(n0, n1);
        let (p0, m0) = policy_priors(&cb_ab, &tbl_ab, Some(&c_ab), 0, &acts0, n0, 0.02);
        let (p1, m1) = policy_priors(&cb_ba, &tbl_ba, Some(&c_ba), 1, &acts1, n1, 0.02);
        assert_eq!(m0, m1);
        for i in 0..n0 {
            assert!((p0[i] - p1[i]).abs() < 1e-6, "policy_priors side-swap mismatch at {i}");
        }

        // 2. Bench-slot permutation invariance of value features.
        let t1_perm = vec![t1[0].clone(), t1[2].clone(), t1[1].clone()];
        let b_perm = start_b(&dex, &t1_perm, &t2);
        let b_orig = start_b(&dex, &t1, &t2);
        let (tp, cbp) = CompactBattle::from_battle(&b_perm, &dex);
        let (to, cbo) = CompactBattle::from_battle(&b_orig, &dex);
        let (x0_o, x1_o, _, _) = extract_value_features(&cbo, &to, None);
        let (x0_p, x1_p, _, _) = extract_value_features(&cbp, &tp, None);
        assert_eq!(x0_o, x0_p, "bench swap must leave x0 bit-identical");
        assert_eq!(&x1_o[..88], &x1_p[..88], "bench swap must leave x1[..88] bit-identical");
        assert_eq!(&x1_o[89..], &x1_p[89..], "bench swap must leave x1[89..] bit-identical");
        assert!((x1_o[88] - x1_p[88]).abs() < 0.02, "bench swap fg g_turns drift: {} vs {}", x1_o[88], x1_p[88]);

        // 3. No-op mask soundness checks.
        // 3a. Awake Sleep Talk masked vs non-sleep foe, NOT masked vs Hypnosis Gengar!
        let zap = vec![mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Rest", "Sleep Talk"])];
        let lax = vec![mk("Snorlax", 50, "Leftovers", &["Body Slam", "Rest"])];
        let gar = vec![mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Hypnosis"])];
        let (tzl, cbzl) = CompactBattle::from_battle(&start_b(&dex, &zap, &lax), &dex);
        let mut azl = [0u8; 7];
        let nzl = cbzl.legal_actions(&tzl, 0, &mut azl);
        let mzl = compact_interior_noop_mask(&cbzl, &tzl, 0, &azl, nzl);
        assert_ne!(mzl & (1 << 2), 0, "awake Sleep Talk vs non-sleep Snorlax must be masked");

        let (tzg, cbzg) = CompactBattle::from_battle(&start_b(&dex, &zap, &gar), &dex);
        let mut azg = [0u8; 7];
        let nzg = cbzg.legal_actions(&tzg, 0, &mut azg);
        let mzg = compact_interior_noop_mask(&cbzg, &tzg, 0, &azg, nzg);
        assert_eq!(mzg & (1 << 2), 0, "awake Sleep Talk vs Hypnosis Gengar must NOT be masked");

        // 3b. Last-mon Explosion ALWAYS masked even under Destiny Bond wait_exempt.
        let boom_m = vec![mk("Gengar", 50, "", &["Thunderbolt", "Explosion"])];
        let db_m = vec![mk("Misdreavus", 50, "", &["Thunder", "Destiny Bond"])];
        let (tbm, mut cbbm) = CompactBattle::from_battle(&start_b(&dex, &boom_m, &db_m), &dex);
        cbbm.mons[3].vol_flags |= vol_bit::DESTINY_BOND;
        let mut abm = [0u8; 7];
        let nbm = cbbm.legal_actions(&tbm, 0, &mut abm);
        let mbm = compact_interior_noop_mask(&cbbm, &tbm, 0, &abm, nbm);
        assert_ne!(mbm & (1 << 1), 0, "last-mon Explosion must be masked even vs Destiny Bond");

        // 3c. Spikes already up: masked vs non-Spinner, NOT masked vs Rapid Spin foe.
        let cloy = vec![mk("Cloyster", 50, "", &["Surf", "Spikes"])];
        let spin_f = vec![mk("Forretress", 50, "", &["Rapid Spin", "Spikes"])];
        let (tcl, mut cbcl) = CompactBattle::from_battle(&start_b(&dex, &cloy, &lax), &dex);
        cbcl.sides[1].side_flags |= 1;
        let mut acl = [0u8; 7];
        let ncl = cbcl.legal_actions(&tcl, 0, &mut acl);
        assert_ne!(
            compact_interior_noop_mask(&cbcl, &tcl, 0, &acl, ncl) & (1 << 1),
            0,
            "duplicate Spikes vs non-Spinner must be masked"
        );
        let (tcs, mut cbcs) = CompactBattle::from_battle(&start_b(&dex, &cloy, &spin_f), &dex);
        cbcs.sides[1].side_flags |= 1;
        let mut acs = [0u8; 7];
        let ncs = cbcs.legal_actions(&tcs, 0, &mut acs);
        assert_eq!(
            compact_interior_noop_mask(&cbcs, &tcs, 0, &acs, ncs) & (1 << 1),
            0,
            "duplicate Spikes vs Rapid Spin foe must NOT be masked"
        );

        // 3d. Universal Ground immunity vs Zapdos-only team masked, vs Zapdos+Snorlax NOT masked.
        let eq_m = vec![mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide"])];
        let (tez, cbez) = CompactBattle::from_battle(&start_b(&dex, &eq_m, &zap), &dex);
        let mut aez = [0u8; 7];
        let nez = cbez.legal_actions(&tez, 0, &mut aez);
        assert_ne!(
            compact_interior_noop_mask(&cbez, &tez, 0, &aez, nez) & (1 << 0),
            0,
            "Earthquake vs Zapdos-only team must be masked"
        );
        let zap_lax = vec![zap[0].clone(), lax[0].clone()];
        let (tezl, cbezl) = CompactBattle::from_battle(&start_b(&dex, &eq_m, &zap_lax), &dex);
        let mut aezl = [0u8; 7];
        let nezl = cbezl.legal_actions(&tezl, 0, &mut aezl);
        assert_eq!(
            compact_interior_noop_mask(&cbezl, &tezl, 0, &aezl, nezl) & (1 << 0),
            0,
            "Earthquake vs Zapdos+Snorlax must NOT be masked (switch read!)"
        );
    }
}
