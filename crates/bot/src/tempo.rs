//! Turn-Economy ("Golf") Evaluator & Standalone Agent PoC.
//!
//! Converts heterogeneous Gen 2 / NC2000 resources (HP, typing, stats, boosts,
//! recovery, status, Substitute, Spikes, Perish Song, Speed, switches, roles)
//! into a single unified currency: **expected turns to wipe the opposing team**
//! (`G_0` vs `G_1`), then maps the turn race to win probability and solves the
//! 1-ply simultaneous-move matrix game via exact LP Nash equilibrium.
//!
//! Kept completely independent from the shipped SM-MCTS (`BlindAgent` / `eval.rs`)
//! stack so experiments cannot perturb production search or cache epochs.

use nc2000_engine::battle::{Outcome, SearchChoice};
use nc2000_engine::compact::{
    ext_vol_bit, get_castle_speed_x3360, role_bit, vol_bit, CompactBattle, CompactMoveSpec,
    CompactWeather, MatchTable, RosterCompactCache, SpecialMoveKind,
};
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::{stat_idx, Battle, Status};

use std::sync::Arc;

use crate::agent::Agent;
use crate::belief::{Belief, FallbackPolicy};
use crate::blind::BlindSearch;
use crate::endgame_root::{
    reply_risk_and_mistake_ev, select_blind_root_choice, EndgameRootStats,
};
use crate::eval::{
    compact_eval01, compact_eval_leaf_blindspot, compact_turn_limit_outcome, EvalWeights,
};
use crate::exact::solve_matrix_full;
use crate::linear::DmgKernel1D;
use crate::mask::{dominated_reason, public_noop_reason};
use crate::mcts::{set_last_sim_end, SimEndInfo};
use crate::observe::Observer;
use crate::player_state::{submitted_team, HiddenTimerKind, HiddenTimerSpec, LocalPlayer};
use crate::preview::MetaPool;
use crate::rng::SplitMix64;
use crate::root_policy::{
    solve_bayesian_equilibrium_stage2, solve_bayesian_equilibrium_with_priors,
    solve_initial_bayesian_prior, RootCell, RootType, Stage2BayesianEquilibrium, MIN_MIX_MARGIN,
};
use crate::smmcts::{RmConfig, SelRule, SkuctSearch};

/// Ablation switch between the smoothed 1-ply-safe Golf model (`Full`) and the
/// unsmoothed minimal Golf model (`Raw`, without HP/Rest/overflow/phaze-cap workarounds).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TempoVariant {
    Full,
    Raw,
}

/// Leaf evaluator mode for `TempoMctsAgent` (enables controlled leaf A/B inside the exact same DUCT tree).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TempoLeafMode {
    TempoFull,
    TempoRaw,
    FastGolf,
    HybridGolf,
    Stage3Log,
    Stage3Mlp,
    AzValue,
    CfValue,
    EvalStatic,
    EvalBlindspot,
}

/// Maximum cap on effective turns-to-KO for walled / stalemated matchups.
pub const MAX_TTK: f64 = 24.0;

/// Maximum turns simulated in the exact 1-HP absorption convolution.
const CONV_MAX_TURNS: usize = 20;

/// Per-turn race variance constant `k` in Brownian race win probability
/// `Phi((G_1 - G_0) / sqrt(k * (G_0 + G_1)))`.
const RACE_VAR_PER_TURN: f64 = 0.65;

/// Detailed pairwise matchup turn breakdown for `(att_side, att_slot) -> (def_side, def_slot)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PairTtk {
    /// Expected turns for attacker to KO defender *without* defender healing.
    pub raw_ttk: f64,
    /// Effective turns for attacker to KO defender *accounting for* defender healing / stall.
    pub eff_ttk: f64,
    /// Mean net damage fraction per turn dealt to defender (attack minus Leftovers + status slip).
    pub net_dmg_frac: f64,
    /// Mean raw attack damage fraction per turn (before residuals).
    pub raw_dmg_frac: f64,
    /// Probability that attacker KOs defender on turn 1.
    pub ohko_prob: f64,
    /// Probability that Explosion / Self-Destruct KOs defender in 1 turn (0 if none).
    pub boom_ko_prob: f64,
    /// Fraction of defender's turns consumed by recovery to sustain against this attacker (`0..=1`).
    pub def_heal_duty: f64,
    /// Whether defender can sustainably wall this attacker via recovery.
    pub def_can_sustain: bool,
    /// Speed comparison: `+1` attacker faster, `0` tie, `-1` defender faster.
    pub speed_cmp: i8,
    /// Move slot chosen for `raw_ttk` (`0..4`, or `0xFF` if none).
    pub best_move_slot: u8,
    /// Whether Perish Song trapped countdown capped `eff_ttk`.
    pub perish_capped: bool,
}

/// One enemy-assignment step in a side's optimal 3v3 team plan.
#[derive(Clone, Copy, Debug, Default)]
pub struct AssignmentStep {
    pub enemy_slot: usize,
    pub killer_slot: usize,
    /// Turns spent by primary killer (plus any spill-over turns if primary dies first).
    pub turns: f64,
    /// Fraction of primary killer's max HP consumed during this engagement.
    pub hp_cost: f64,
    /// Whether a teammate had to finish the enemy due to primary killer running out of HP.
    pub spilled: bool,
}

/// Optimal 3v3 wipe plan ("Golf scorecard") for one side.
#[derive(Clone, Debug, Default)]
pub struct TeamPlan {
    /// Total expected turns to wipe all living opposing Pokemon.
    pub total_turns: f64,
    /// Portion of `total_turns` spent on switching/pivoting overhead.
    pub switch_turns: f64,
    /// Portion added by capacity overflow (assigned killer runs out of HP).
    pub overflow_turns: f64,
    /// Adjustment from opponent having a healthy living Phazer/Hazer against active boosts.
    pub phaze_cap_turns: f64,
    /// Speed-order adjustment on the active matchup (`-0.25` faster, `+0.25` slower, `0.0` tie).
    pub active_speed_adj: f64,
    /// Per living enemy assignment.
    pub steps: Vec<AssignmentStep>,
}

/// Full symmetric turn-economy evaluation of a battle state.
#[derive(Clone, Debug)]
pub struct TempoEval {
    /// Total expected turns `[G_0, G_1]` for side 0 and side 1 to wipe the opponent.
    pub g_turns: [f64; 2],
    /// Turn advantage from side 0's perspective: `G_1 - G_0` (positive = side 0 ahead).
    pub turn_advantage: f64,
    /// Brownian-race win probability from side 0's perspective in `[0, 1]`.
    pub win_prob: f64,
    /// Detailed wipe plans for `[side 0, side 1]`.
    pub plans: [TeamPlan; 2],
    /// Pairwise TTK matrices: `ttk[att_side][att_slot][def_slot]`.
    pub ttk: [[[PairTtk; 3]; 3]; 2],
}

impl TempoEval {
    /// Perspective-adjusted turn advantage (`> 0` favors `side`).
    #[inline]
    pub fn advantage_for(&self, side: usize) -> f64 {
        if side == 0 {
            self.turn_advantage
        } else {
            -self.turn_advantage
        }
    }

    /// Perspective-adjusted win probability in `[0, 1]`.
    #[inline]
    pub fn win_prob_for(&self, side: usize) -> f64 {
        if side == 0 {
            self.win_prob
        } else {
            1.0 - self.win_prob
        }
    }
}

/// Result of 1-ply simultaneous-move turn-economy analysis at a decision point.
#[derive(Clone, Debug)]
pub struct TurnMatrixAnalysis {
    pub own_choices: Vec<SearchChoice>,
    pub opp_choices: Vec<SearchChoice>,
    /// Row-major `n_own * n_opp` expected win probability for `side`.
    pub win_matrix: Vec<f64>,
    /// Row-major `n_own * n_opp` expected turn advantage `ΔG` for `side`.
    pub turn_matrix: Vec<f64>,
    /// Nash equilibrium mixed strategy for `side`.
    pub own_policy: Vec<f64>,
    /// Nash equilibrium mixed strategy for opponent.
    pub opp_policy: Vec<f64>,
    /// Equilibrium win probability for `side`.
    pub eq_win_prob: f64,
    /// Expected turn advantage under equilibrium `(own_policy, opp_policy)`.
    pub eq_turn_adv: f64,
}

// =========================================================================
// Helper Mechanics
// =========================================================================

#[inline]
pub(crate) fn effective_speed(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> i32 {
    if table.ruleset.is_castle() {
        return cb.castle_speed_x3360(table, side, slot) as i32;
    }
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let is_active = cb.sides[side].active_idx as usize == slot;
    // In PS, `exec_switch` clears `vol_flags = 0` on bench and re-attaches `PAR_SPEED_DROP` on switch-in.
    if !is_active && mon.status_raw() == Status::Par as u8 {
        let base = (table.mons[idx].stored_stats[stat_idx::SPE] as i32).max(1);
        return (base / 4).max(1);
    }
    cb.mon_stat(table, side, slot, stat_idx::SPE, false, false)
}

#[inline]
fn effective_speed_unboosted(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    slot: usize,
) -> i32 {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    if table.ruleset.is_castle() {
        get_castle_speed_x3360(
            table.mons[idx].stored_stats[stat_idx::SPE] as i32,
            0,
            mon.status_raw() == Status::Par as u8,
        ) as i32
    } else {
        let is_active = cb.sides[side].active_idx as usize == slot;
        if !is_active && mon.status_raw() == Status::Par as u8 {
            let base = (table.mons[idx].stored_stats[stat_idx::SPE] as i32).max(1);
            return (base / 4).max(1);
        }
        cb.mon_stat(table, side, slot, stat_idx::SPE, true, false)
    }
}

/// True when `status_raw` + packed `status_counter` (`0x10` residualdmg flag) deals escalating
/// Toxic-style `((cnt & 0x0F) + 1) / 16` slip damage (matches `compact.rs:6414` and `linear.rs:106`).
#[inline]
fn is_toxic_residual(ruleset: Ruleset, status_raw: u8, cnt_packed: u8) -> bool {
    status_raw == Status::Tox as u8
        || ((cnt_packed & 0x10) != 0
            && (status_raw == Status::Psn as u8
                || (!ruleset.is_castle() && status_raw == Status::Brn as u8)))
}

#[inline]
fn is_grounded(table: &MatchTable, idx: usize) -> bool {
    (table.mons[idx].type_mask & (1u32 << (table.flying_type & 31))) == 0
}

#[inline]
pub(crate) fn spikes_layers(cb: &CompactBattle, side: usize) -> u8 {
    cb.sides[side].side_flags & 0x03
}

/// Effective starting HP for a mon if it fights now (deducting Spikes if benched & grounded).
#[inline]
pub(crate) fn entry_hp(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> i16 {
    let idx = side * 3 + slot;
    let mut hp = cb.mons[idx].hp();
    if hp <= 0 {
        return 0;
    }
    let is_active = cb.sides[side].active_idx as usize == slot;
    let layers = spikes_layers(cb, side);
    if !is_active && layers > 0 && is_grounded(table, idx) {
        let spike_dmg = match layers {
            1 => table.mons[idx].quantum_1_8,
            2 => (table.mons[idx].maxhp / 6).max(1),
            _ => (table.mons[idx].maxhp / 4).max(1),
        };
        hp = (hp - spike_dmg).max(0);
    }
    hp
}

#[inline]
fn instant_heal_frac(cb: &CompactBattle, ms: &CompactMoveSpec) -> Option<f64> {
    if ms.is_half_heal {
        Some(0.5)
    } else if ms.special_kind == SpecialMoveKind::WeatherHeal {
        let weather = if cb.weather_turns > 0 {
            cb.weather_kind
        } else {
            CompactWeather::None
        };
        let frac = match weather {
            CompactWeather::Sun => 1.0,
            CompactWeather::Rain | CompactWeather::Sandstorm => 0.25,
            CompactWeather::None => 0.5,
        };
        Some(frac)
    } else {
        None
    }
}

/// Expected remaining asleep turns before waking (0 if not asleep).
#[inline]
pub(crate) fn expected_sleep_turns(cb: &CompactBattle, side: usize, slot: usize) -> f64 {
    let mon = &cb.mons[side * 3 + slot];
    if mon.status_raw() != Status::Slp as u8 {
        return 0.0;
    }
    // In `CompactBattle`, `status_counter()` is the exact remaining sleep turns:
    // counter == 1 means wakes on this turn before acting (0 wasted turns).
    // counter >= 2 means sleeps `counter - 1` more turns.
    let c = mon.status_counter();
    if c <= 1 {
        0.0
    } else {
        (c - 1) as f64
    }
}

/// Expected fraction of turns `Sleep Talk` executes a damaging move when asleep.
fn sleep_talk_attack_rate(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> f64 {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];
    let n_moves = spec.num_moves as usize;
    if n_moves <= 1 {
        return 0.0;
    }
    let mut has_st = false;
    let mut usable_attacks = 0usize;
    for m_i in 0..n_moves {
        let ms = &table.move_specs[idx][m_i];
        if ms.special_kind == SpecialMoveKind::SleepTalk
            && mon.pp[m_i] > 0
            && !mon.is_slot_disabled(m_i)
        {
            has_st = true;
        } else if !ms.is_status && !ms.is_selfdestruct && ms.can_sleep_talk {
            usable_attacks += 1;
        }
    }
    if !has_st {
        return 0.0;
    }
    usable_attacks as f64 / (n_moves - 1) as f64
}

/// Normal CDF approximation (Abramowitz-Stegun 7.1.26, max error < 1.5e-7).
#[inline]
pub fn normal_cdf(z: f64) -> f64 {
    if z == 0.0 {
        return 0.5;
    }
    if z <= -8.0 {
        return 0.0;
    }
    if z >= 8.0 {
        return 1.0;
    }
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let erfc_x = poly * (-x * x).exp();
    if z >= 0.0 {
        1.0 - 0.5 * erfc_x
    } else {
        0.5 * erfc_x
    }
}

/// Convert team turn counts `(g0, g1)` into Brownian race win probability for side 0.
#[inline]
pub fn turns_to_win_prob(g0: f64, g1: f64) -> f64 {
    if g0 <= 0.0 && g1 <= 0.0 {
        return 0.5;
    }
    if g0 <= 0.0 {
        return 1.0;
    }
    if g1 <= 0.0 {
        return 0.0;
    }
    let sigma = (RACE_VAR_PER_TURN * (g0 + g1).max(1.0)).sqrt();
    normal_cdf((g1 - g0) / sigma)
}

// =========================================================================
// Exact 1-HP Forward Absorption Convolution for Raw TTK
// =========================================================================

/// Compute exact expected turns to KO `def` from `start_hp` using damage kernel `kern`
/// (with Gen 2 residuals, Substitute, Sleep/Freeze/Paralysis/Confusion, Recharge, and Berry).
#[allow(clippy::too_many_arguments)]
fn convolve_raw_ttk(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    start_hp: i16,
    kern: &DmgKernel1D,
    st_kern: Option<&DmgKernel1D>,
    is_recharge: bool,
    move_pp: u8,
) -> (f64, f64) {
    if start_hp <= 0 {
        return (0.0, 1.0);
    }
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let att_mon = &cb.mons[att_idx];
    let def_mon = &cb.mons[def_idx];
    let def_spec = &table.mons[def_idx];
    let maxhp = def_spec.maxhp.max(1);

    // Effective HP resolution: exact 1-HP for all NC2000 mons (`maxhp <= 399`).
    const MAX_BINS: usize = 400;
    let scale = ((maxhp as usize) / (MAX_BINS - 1)) + 1;
    let max_bin = ((maxhp as usize) + scale - 1) / scale;
    let start_bin = ((start_hp as usize) + scale - 1) / scale;

    // One-time healing berry bonus on defender (added cleanly when dipping below 50%).
    let def_item = table.item_spec(def_mon.item);
    let berry_heal: i32 = if def_item.is_goldberry {
        30
    } else if def_item.is_berry_juice {
        20
    } else if def_item.is_berry {
        10
    } else {
        0
    };

    // Constant per-turn HP residual on defender (exact Gen 2 quanta).
    let mut base_residual: i32 = 0;
    if def_item.is_leftovers {
        base_residual += def_spec.quantum_1_16 as i32;
    }
    let def_status = def_mon.status_raw();
    let def_cnt = def_mon.status_counter();
    let is_toxic = is_toxic_residual(table.ruleset, def_status, def_cnt);
    if !is_toxic && (def_status == Status::Brn as u8 || def_status == Status::Psn as u8) {
        base_residual -= def_spec.quantum_1_8 as i32;
    }
    if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        base_residual -= def_spec.quantum_1_8 as i32;
    }
    if (def_mon.vol_flags & vol_bit::CURSE) != 0 {
        base_residual -= (def_spec.maxhp / 4).max(1) as i32;
    }
    if (def_mon.vol_flags & vol_bit::NIGHTMARE) != 0
        && (def_status == Status::Slp as u8
            || (table.ruleset.is_castle()
                && (def_mon.ext_flags & ext_vol_bit::MIRACLE_NIGHTMARE) != 0))
    {
        base_residual -= (def_spec.maxhp / 4).max(1) as i32;
    }
    if def_mon.partially_trapped() {
        base_residual -= def_spec.quantum_1_16 as i32;
    }
    if cb.weather_turns > 0
        && cb.weather_kind == CompactWeather::Sandstorm
        && (def_spec.type_mask & table.sandstorm_immune_mask) == 0
    {
        base_residual -= def_spec.quantum_1_8 as i32;
    }
    let tox_base = if is_toxic {
        (def_cnt & 0x0F) as i32
    } else {
        0
    };

    // Attacker status / volatility constraints.
    let att_status = att_mon.status_raw();
    let sleep_turns = if att_status == Status::Slp as u8 {
        expected_sleep_turns(cb, att_side, att_slot).round() as usize
    } else {
        0
    };
    let is_frozen = att_status == Status::Frz as u8;
    let is_par = att_status == Status::Par as u8;
    let conf_turns = if (att_mon.vol_flags & vol_bit::CONFUSION) != 0 {
        att_mon.timers[0].max(1) as usize
    } else {
        0
    };
    let must_recharge_t1 = att_mon.must_recharge();

    // Substitute shield on active defender: compute extra turns to break Sub first.
    let sub_hp = if cb.sides[def_side].active_idx as usize == def_slot {
        def_mon.sub_hp()
    } else {
        0
    };
    let sub_extra_turns = if sub_hp > 0 && kern.mean_dmg > 0.0 {
        let p_break_1 = kern.ko_prob(sub_hp) as f64;
        if p_break_1 >= 0.99 {
            1.0
        } else if p_break_1 > 0.05 {
            (1.0 / p_break_1).min(4.0)
        } else {
            ((sub_hp as f64) / (kern.mean_dmg as f64)).ceil().min(4.0)
        }
    } else {
        0.0
    };

    // Track exact probability distribution over HP bins `1..=max_bin`.
    // Include one-time berry via effective starting HP if berry would trigger before KO.
    let init_bin = if berry_heal > 0 && (kern.max_dmg as i32) < (start_hp as i32) {
        let boosted = (start_hp as i32 + berry_heal).min(maxhp as i32) as usize;
        ((boosted + scale - 1) / scale).min(max_bin)
    } else {
        start_bin
    };

    let mut cur = [0.0f32; MAX_BINS];
    let mut nxt = [0.0f32; MAX_BINS];
    cur[init_bin] = 1.0;
    let mut lo = init_bin;
    let mut hi = init_bin;

    let mut sum_s = 1.0f64; // S(0) = 1.0
    let mut prev_s = 1.0f64;
    let mut s1 = 1.0f64;
    let mut p_thawed = if is_frozen { 0.0f32 } else { 1.0f32 };
    let max_t = CONV_MAX_TURNS.min(if move_pp == 0 { 1 } else { CONV_MAX_TURNS });

    for turn in 1..=max_t {
        // Choose active attack kernel for this turn.
        let (active_kern, mut p_act): (&DmgKernel1D, f32) = if turn == 1 && must_recharge_t1 {
            (kern, 0.0)
        } else if turn <= sleep_turns {
            if let Some(sk) = st_kern {
                (sk, 1.0)
            } else {
                (kern, 0.0)
            }
        } else if is_recharge && turn % 2 == 0 {
            // Non-KO Hyper Beam recharges every second turn.
            (kern, 0.0)
        } else if (turn as u8) > move_pp && move_pp < 20 {
            // Ran out of PP for this move.
            (kern, 0.0)
        } else {
            (kern, 1.0)
        };

        if is_frozen && p_act > 0.0 {
            // Gen 2 thaw chance: 25/256 ≈ 0.09765625 per turn.
            p_thawed = p_thawed + (1.0 - p_thawed) * (25.0 / 256.0);
            p_act *= p_thawed;
        }
        if is_par && p_act > 0.0 {
            p_act *= 0.75;
        }
        if turn <= conf_turns && p_act > 0.0 {
            p_act *= 0.5;
        }

        // Residual for this turn (Toxic escalates each turn: `(cnt & 0x0F) + 1` on Turn 1).
        let res_hp = if is_toxic {
            let step = (tox_base + turn as i32).min(15);
            base_residual - step * (def_spec.quantum_1_16 as i32)
        } else {
            base_residual
        };

        for v in &mut nxt[0..=max_bin] {
            *v = 0.0;
        }
        let mut new_lo = max_bin + 1;
        let mut new_hi = 0usize;

        for b in lo..=hi {
            let p = cur[b];
            if p <= 1e-7 {
                continue;
            }
            let hp_val = (b * scale).min(maxhp as usize) as i32;

            // Branch 1: attacker fails to act (0 attack damage, residuals still apply).
            let p_idle = p * (1.0 - p_act);
            if p_idle > 1e-8 {
                let h_after = (hp_val + res_hp).min(maxhp as i32);
                if h_after > 0 {
                    let nb = (((h_after as usize) + scale - 1) / scale).min(max_bin);
                    nxt[nb] += p_idle;
                    if nb < new_lo {
                        new_lo = nb;
                    }
                    if nb > new_hi {
                        new_hi = nb;
                    }
                }
            }

            // Branch 2: attacker executes `active_kern`.
            if p_act > 0.0 {
                let p_base = p * p_act;
                for i in 0..active_kern.len {
                    let (dmg, prob) = active_kern.pts[i];
                    let pm = p_base * prob;
                    if pm <= 1e-8 {
                        continue;
                    }
                    let h_hit = hp_val - dmg as i32;
                    if h_hit <= 0 {
                        // Absorbed (KO'd before end-of-turn Leftovers/residuals).
                        continue;
                    }
                    let h_after = (h_hit + res_hp).min(maxhp as i32);
                    if h_after > 0 {
                        let nb = (((h_after as usize) + scale - 1) / scale).min(max_bin);
                        nxt[nb] += pm;
                        if nb < new_lo {
                            new_lo = nb;
                        }
                        if nb > new_hi {
                            new_hi = nb;
                        }
                    }
                }
            }
        }

        let mut s_k = 0.0f64;
        if new_lo <= new_hi {
            for b in new_lo..=new_hi {
                let v = nxt[b];
                cur[b] = v;
                s_k += v as f64;
            }
            // Clear stale mass outside new range.
            for b in lo..new_lo {
                cur[b] = 0.0;
            }
            for b in (new_hi + 1)..=hi {
                cur[b] = 0.0;
            }
            lo = new_lo;
            hi = new_hi;
        } else {
            s_k = 0.0;
            lo = 1;
            hi = 0;
        }

        if turn == 1 {
            s1 = s_k;
        }
        if turn < max_t {
            sum_s += s_k;
        }
        if s_k < 1e-5 {
            prev_s = 0.0;
            break;
        }
        prev_s = s_k;
    }

    // Tail geometric extrapolation if still alive at `max_t`.
    let tail = if prev_s > 1e-4 {
        // Approximate remaining geometric tail smoothly capped at MAX_TTK.
        let s_prev_turn = (sum_s - 1.0).max(prev_s);
        let ratio = if s_prev_turn > 0.0 {
            (prev_s / s_prev_turn).clamp(0.0, 0.96)
        } else {
            0.96
        };
        (prev_s / (1.0 - ratio).max(0.04)).min(MAX_TTK)
    } else {
        0.0
    };

    let total_ttk = (sum_s + tail + sub_extra_turns).min(MAX_TTK);
    let ohko_prob = if sub_hp > 0 { 0.0 } else { (1.0 - s1).clamp(0.0, 1.0) };
    (total_ttk, ohko_prob)
}

// =========================================================================
// Pairwise Matchup Analysis (including Recovery Duty Cycle & Perish Song)
// =========================================================================

struct RawPairInfo {
    ttk: PairTtk,
    max_norm_dmg_frac: f64,
    max_crit_dmg_frac: f64,
    best_move_pp: u8,
}

fn compute_raw_pair(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
) -> RawPairInfo {
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let att_mon = &cb.mons[att_idx];
    let def_mon = &cb.mons[def_idx];
    let att_spec = &table.mons[att_idx];
    let def_spec = &table.mons[def_idx];

    let s_att = effective_speed(cb, table, att_side, att_slot);
    let s_def = effective_speed(cb, table, def_side, def_slot);
    let speed_cmp = if s_att > s_def {
        1
    } else if s_att < s_def {
        -1
    } else {
        0
    };

    let start_hp = entry_hp(cb, table, def_side, def_slot);
    if start_hp <= 0 || att_mon.hp() <= 0 {
        return RawPairInfo {
            ttk: PairTtk {
                raw_ttk: if start_hp <= 0 { 0.0 } else { MAX_TTK },
                eff_ttk: if start_hp <= 0 { 0.0 } else { MAX_TTK },
                speed_cmp,
                best_move_slot: 0xFF,
                ..PairTtk::default()
            },
            max_norm_dmg_frac: 0.0,
            max_crit_dmg_frac: 0.0,
            best_move_pp: 0,
        };
    }

    let maxhp_f = def_spec.maxhp.max(1) as f64;
    let def_item = table.item_spec(def_mon.item);

    // Defender's passive residual per turn as a fraction of maxhp (positive = heals defender).
    let mut residual_frac = 0.0f64;
    if def_item.is_leftovers {
        residual_frac += (def_spec.quantum_1_16 as f64) / maxhp_f;
    }
    let def_status = def_mon.status_raw();
    let def_cnt = def_mon.status_counter();
    if is_toxic_residual(table.ruleset, def_status, def_cnt) {
        let step = ((def_cnt & 0x0F) as f64 + 1.5).min(15.0);
        residual_frac -= step * (def_spec.quantum_1_16 as f64) / maxhp_f;
    } else if def_status == Status::Brn as u8 || def_status == Status::Psn as u8 {
        residual_frac -= (def_spec.quantum_1_8 as f64) / maxhp_f;
    }
    if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        residual_frac -= (def_spec.quantum_1_8 as f64) / maxhp_f;
    }
    if (def_mon.vol_flags & vol_bit::CURSE) != 0 {
        residual_frac -= ((def_spec.maxhp / 4).max(1) as f64) / maxhp_f;
    }
    if (def_mon.vol_flags & vol_bit::NIGHTMARE) != 0
        && (def_status == Status::Slp as u8
            || (table.ruleset.is_castle()
                && (def_mon.ext_flags & ext_vol_bit::MIRACLE_NIGHTMARE) != 0))
    {
        residual_frac -= ((def_spec.maxhp / 4).max(1) as f64) / maxhp_f;
    }
    if def_mon.partially_trapped() {
        residual_frac -= (def_spec.quantum_1_16 as f64) / maxhp_f;
    }
    if cb.weather_turns > 0
        && cb.weather_kind == CompactWeather::Sandstorm
        && (def_spec.type_mask & table.sandstorm_immune_mask) == 0
    {
        residual_frac -= (def_spec.quantum_1_8 as f64) / maxhp_f;
    }

    let st_kern_val = if att_mon.status_raw() == Status::Slp as u8
        && (att_spec.role_flags & role_bit::SLEEP_USABLE) != 0
    {
        let sk = DmgKernel1D::sleep_talk(cb, table, att_side, att_slot, def_side, def_slot);
        if sk.max_dmg > 0 { Some(sk) } else { None }
    } else {
        None
    };

    let mut best_ttk = MAX_TTK;
    let mut best_ohko = 0.0f64;
    let mut best_raw_dmg = 0.0f64;
    let mut best_max_norm = 0.0f64;
    let mut best_max_crit = 0.0f64;
    let mut best_slot = 0xFFu8;
    let mut best_pp = 0u8;
    let mut boom_ko_prob = 0.0f64;

    let n_moves = att_spec.num_moves as usize;
    for m_i in 0..n_moves {
        if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
            continue;
        }
        let ms = &table.move_specs[att_idx][m_i];
        if ms.is_status {
            continue;
        }
        let kern = DmgKernel1D::for_move(cb, table, att_side, att_slot, def_side, def_slot, m_i);
        if kern.max_dmg <= 0 {
            continue;
        }
        if ms.is_selfdestruct {
            let p_ko = kern.ko_prob(start_hp) as f64;
            if p_ko > boom_ko_prob {
                boom_ko_prob = p_ko;
            }
            continue;
        }

        let (ttk, ohko) = convolve_raw_ttk(
            cb,
            table,
            att_side,
            att_slot,
            def_side,
            def_slot,
            start_hp,
            &kern,
            st_kern_val.as_ref(),
            ms.is_recharge,
            att_mon.pp[m_i],
        );
        let raw_dmg = (kern.mean_dmg as f64) / maxhp_f;
        // Estimate non-crit max roll (~max_dmg when no crit, else ~half of crit max_dmg or 75th-ish support).
        let mut max_crit = (kern.max_dmg as f64) / maxhp_f;
        let mut max_norm = raw_dmg * 1.08;
        if kern.len > 0 {
            // Find highest damage point with probability >= 0.08 (normal roll quartile, not 1/64 crit).
            let mut norm_hi = 0i16;
            for k in 0..kern.len {
                if kern.pts[k].1 >= 0.06 && kern.pts[k].0 > norm_hi {
                    norm_hi = kern.pts[k].0;
                }
            }
            if norm_hi > 0 {
                max_norm = (norm_hi as f64) / maxhp_f;
            }
            max_crit = max_crit.max(max_norm);
        }

        let better = if (ttk - best_ttk).abs() > 1e-4 {
            ttk < best_ttk
        } else {
            raw_dmg > best_raw_dmg
        };
        if better {
            best_ttk = ttk;
            best_ohko = ohko;
            best_raw_dmg = raw_dmg;
            best_max_norm = max_norm;
            best_max_crit = max_crit;
            best_slot = m_i as u8;
            best_pp = att_mon.pp[m_i];
        }
    }

    // If no damaging moves (or all immune), still let negative residuals (Burn/Poison/Toxic) KO defender.
    if best_slot == 0xFF && residual_frac < -1e-4 {
        let zero_k = DmgKernel1D::zero();
        let (ttk, ohko) = convolve_raw_ttk(
            cb,
            table,
            att_side,
            att_slot,
            def_side,
            def_slot,
            start_hp,
            &zero_k,
            None,
            false,
            20,
        );
        best_ttk = ttk;
        best_ohko = ohko;
    }

    // Perish Song hard countdown override when active defender is trapped (or last mon).
    let mut perish_capped = false;
    let def_is_active = cb.sides[def_side].active_idx as usize == def_slot;
    if def_is_active {
        let pt = def_mon.perish_turns();
        let trapped = def_mon.is_trapped() || cb.sides[def_side].alive_mask.count_ones() <= 1;
        if pt > 0 && trapped && (pt as f64) < best_ttk {
            best_ttk = pt as f64;
            perish_capped = true;
        }
    }

    // Net damage fraction per turn (positive = defender loses HP per turn).
    let att_act_mult = if att_mon.status_raw() == Status::Par as u8 {
        0.75
    } else if att_mon.status_raw() == Status::Frz as u8 {
        0.15
    } else {
        1.0
    };
    let net_dmg_frac = best_raw_dmg * att_act_mult - residual_frac;

    RawPairInfo {
        ttk: PairTtk {
            raw_ttk: best_ttk,
            eff_ttk: best_ttk,
            net_dmg_frac,
            raw_dmg_frac: best_raw_dmg,
            ohko_prob: best_ohko,
            boom_ko_prob,
            def_heal_duty: 0.0,
            def_can_sustain: false,
            speed_cmp,
            best_move_slot: best_slot,
            perish_capped,
        },
        max_norm_dmg_frac: best_max_norm,
        max_crit_dmg_frac: best_max_crit,
        best_move_pp: best_pp,
    }
}

/// Apply Gen 2 recovery dynamics (50% Recover/Soft-Boiled/WeatherHeal + Rest/Sleep Talk/Mint Berry)
/// to a coupled pair `(A -> D, D -> A)`.
fn apply_recovery_for_defender(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    att_to_def: &mut RawPairInfo,
    def_to_att: &mut RawPairInfo,
    variant: TempoVariant,
) {
    if att_to_def.ttk.perish_capped {
        return;
    }
    let def_idx = def_side * 3 + def_slot;
    let def_mon = &cb.mons[def_idx];
    let def_spec = &table.mons[def_idx];
    let def_hp = entry_hp(cb, table, def_side, def_slot);
    if def_hp <= 0 || cb.mons[att_side * 3 + att_slot].hp() <= 0 {
        return;
    }

    let def_maxhp = def_spec.maxhp.max(1) as f64;
    let def_hp_frac = (def_hp as f64) / def_maxhp;
    let def_faster = def_to_att.ttk.speed_cmp > 0;
    let def_asleep = def_mon.status_raw() == Status::Slp as u8
        && expected_sleep_turns(cb, def_side, def_slot) >= 1.0;
    let def_frozen = def_mon.status_raw() == Status::Frz as u8;
    if def_frozen {
        return;
    }

    // Inspect defender's recovery moves.
    let mut best_half_heal = 0.0f64;
    let mut half_heal_pp = 0u8;
    let mut rest_pp = 0u8;
    for m_i in 0..def_spec.num_moves as usize {
        if def_mon.pp[m_i] == 0 || def_mon.is_slot_disabled(m_i) {
            continue;
        }
        let ms = &table.move_specs[def_idx][m_i];
        if let Some(hf) = instant_heal_frac(cb, ms) {
            if hf > best_half_heal {
                best_half_heal = hf;
                half_heal_pp = def_mon.pp[m_i];
            }
        }
        if ms.is_rest {
            rest_pp = rest_pp.max(def_mon.pp[m_i]);
        }
    }

    let net_dmg = att_to_def.ttk.net_dmg_frac;
    let raw_dmg = att_to_def.ttk.raw_dmg_frac;
    let max_norm = att_to_def.max_norm_dmg_frac;
    let max_crit = att_to_def.max_crit_dmg_frac;
    let att_pp = att_to_def.best_move_pp.max(1) as f64;

    // Probability defender survives to execute its first recovery move.
    let p_survive_t1 = if def_faster {
        // Faster defender heals before taking a hit, unless asleep and already in KO range.
        if def_asleep {
            1.0 - att_to_def.ttk.ohko_prob
        } else {
            1.0
        }
    } else {
        (1.0 - att_to_def.ttk.ohko_prob).clamp(0.0, 1.0)
    };
    if p_survive_t1 < 0.05 {
        return;
    }

    // Evaluate sustainable recovery options: (sustain_ttk_att, stall_ttk_def, duty).
    let mut best_option: Option<(f64, f64, f64)> = None;

    // Option 1: 50% Heal (Recover / Soft-Boiled / Milk Drink / Weather Heal).
    if best_half_heal > 0.0 && half_heal_pp > 0 && !def_asleep {
        // Sustainable if attacker's net damage per turn does not exceed heal amount,
        // and a normal non-crit hit does not OHKO from the post-heal HP band.
        let safe_hp_after_heal = if def_faster {
            (def_hp_frac + best_half_heal).min(1.0)
        } else {
            (1.0 - max_norm + best_half_heal).min(1.0)
        };
        let can_survive_loop = max_norm < safe_hp_after_heal - 0.02 && max_norm < 0.98;
        if can_survive_loop && net_dmg <= best_half_heal + 0.015 {
            let duty = if net_dmg <= 0.0 {
                0.0
            } else {
                (net_dmg / best_half_heal).clamp(0.0, 1.0)
            };

            // How many turns until attacker breaks through via PP exhaustion or crit?
            let heal_pp_turns = if duty > 0.05 {
                (half_heal_pp as f64) / duty + att_to_def.ttk.raw_ttk
            } else {
                MAX_TTK
            };
            // Crit break hazard: when defender hovers at `[1 - best_half_heal, 1]`,
            // a crit breaks through if `max_crit` can KO from the low end of the heal cycle.
            let floor_hp = (1.0 - max_norm).clamp(0.25, 0.75);
            let crit_break_turns = if max_crit >= floor_hp && raw_dmg > 0.05 {
                // Gen 2 base crit rate = 1/16; only lethal when defender is near floor (~duty fraction of turns).
                let p_crit_break = (1.0 / 16.0) * duty.max(0.25);
                att_to_def.ttk.raw_ttk + (1.0 / p_crit_break)
            } else {
                MAX_TTK
            };
            let t_att_sustain = p_survive_t1 * heal_pp_turns.min(crit_break_turns).min(MAX_TTK)
                + (1.0 - p_survive_t1) * att_to_def.ttk.raw_ttk;

            // Defender's attacking TTK while healing:
            // - `Full`: exact excess-HP heal cost beyond starting HP.
            // - `Raw`: simple duty-cycle scaling `raw_ttk / (1 - duty)`.
            let t_def_stall = match variant {
                TempoVariant::Full => {
                    let dmg_over_kill = def_to_att.ttk.raw_ttk * net_dmg.max(0.0);
                    let excess_hp = (dmg_over_kill - (def_hp_frac - 0.15).max(0.0)).max(0.0);
                    let net_heal_per_turn = best_half_heal - net_dmg.max(0.0);
                    if excess_hp <= 1e-4 {
                        def_to_att.ttk.raw_ttk
                    } else if net_heal_per_turn > 0.02 {
                        (def_to_att.ttk.raw_ttk + excess_hp / net_heal_per_turn).min(MAX_TTK)
                    } else {
                        att_pp.min(MAX_TTK)
                    }
                }
                TempoVariant::Raw => {
                    let free_rate = (1.0 - duty).max(0.0);
                    if free_rate > 0.04 {
                        (def_to_att.ttk.raw_ttk / free_rate).min(MAX_TTK)
                    } else {
                        att_pp.min(MAX_TTK)
                    }
                }
            };

            best_option = Some((t_att_sustain, t_def_stall, duty));
        }
    }

    // Option 2: Rest (100% full heal + status cure, 3-turn cycle; Mint/Miracle Berry = 1 turn once).
    if rest_pp > 0 && !def_asleep {
        let def_item = table.item_spec(def_mon.item);
        let has_wake_berry = def_item.is_mintberry || def_item.is_miracleberry;
        // Rest clears Burn/Poison/Toxic, so recompute damage without status slip!
        let mut rest_res = 0.0;
        if def_item.is_leftovers {
            rest_res += (def_spec.quantum_1_16 as f64) / def_maxhp;
        }
        if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
            rest_res -= (def_spec.quantum_1_8 as f64) / def_maxhp;
        }
        if (def_mon.vol_flags & vol_bit::CURSE) != 0 {
            rest_res -= ((def_spec.maxhp / 4).max(1) as f64) / def_maxhp;
        }
        if def_mon.partially_trapped() {
            rest_res -= (def_spec.quantum_1_16 as f64) / def_maxhp;
        }
        if cb.weather_turns > 0
            && cb.weather_kind == CompactWeather::Sandstorm
            && (def_spec.type_mask & table.sandstorm_immune_mask) == 0
        {
            rest_res -= (def_spec.quantum_1_8 as f64) / def_maxhp;
        }
        let att_act_mult = if cb.mons[att_side * 3 + att_slot].status_raw() == Status::Par as u8 {
            0.75
        } else {
            1.0
        };
        let rest_net_dmg = raw_dmg * att_act_mult - rest_res;

        // Sleep hits taken during Rest + 2 asleep turns: ~2.0 if faster, ~2.8 if slower.
        let sleep_hits = if def_faster { 2.0 } else { 2.8 };
        let hp_after_wake = 1.0 - sleep_hits * rest_net_dmg.max(0.0);
        let h_rest_threshold = max_norm.clamp(0.18, 0.50);
        let rest_margin = hp_after_wake - h_rest_threshold;

        let rest_ok = match variant {
            TempoVariant::Full => rest_margin > -0.06 && max_norm < 0.50,
            TempoVariant::Raw => rest_margin > 0.02 && max_norm < 0.50,
        };
        if rest_ok {
            let sustain_w = match variant {
                TempoVariant::Full => ((rest_margin + 0.06) / 0.12).clamp(0.0, 1.0),
                TempoVariant::Raw => 1.0,
            };
            let k_awake = if rest_net_dmg > 0.01 {
                ((hp_after_wake - h_rest_threshold).max(0.0) / rest_net_dmg).max(0.0)
            } else {
                20.0
            };
            let st_rate = sleep_talk_attack_rate(cb, table, def_side, def_slot);
            let cycle_len = 3.0 + k_awake;
            let non_act_turns = (3.0 - 2.0 * st_rate).max(1.0);
            let duty = if rest_net_dmg <= 0.0 {
                0.0
            } else {
                (non_act_turns / cycle_len).clamp(0.0, 1.0)
            };

            let pp_turns = ((rest_pp as f64) * cycle_len + att_to_def.ttk.raw_ttk).min(MAX_TTK);
            let crit_break_turns = if max_crit + max_norm >= 0.95 && raw_dmg > 0.15 {
                att_to_def.ttk.raw_ttk + 14.0
            } else {
                MAX_TTK
            };
            let full_sustain = p_survive_t1 * pp_turns.min(crit_break_turns).min(MAX_TTK)
                + (1.0 - p_survive_t1) * att_to_def.ttk.raw_ttk;
            let t_att_sustain =
                att_to_def.ttk.raw_ttk * (1.0 - sustain_w) + full_sustain * sustain_w;
            let full_stall_def = match variant {
                TempoVariant::Full => {
                    let dmg_over_kill = def_to_att.ttk.raw_ttk * rest_net_dmg.max(0.0);
                    let excess_hp = (dmg_over_kill - (def_hp_frac - 0.15).max(0.0)).max(0.0);
                    let net_heal_per_cycle = (hp_after_wake - h_rest_threshold).max(0.0);
                    if excess_hp <= 1e-4 {
                        def_to_att.ttk.raw_ttk
                    } else if net_heal_per_cycle > 0.04 {
                        (def_to_att.ttk.raw_ttk + non_act_turns * (excess_hp / net_heal_per_cycle))
                            .min(MAX_TTK)
                    } else {
                        att_pp.min(MAX_TTK)
                    }
                }
                TempoVariant::Raw => {
                    let free_rate = (1.0 - duty).max(0.0);
                    if free_rate > 0.05 {
                        (def_to_att.ttk.raw_ttk / free_rate).min(MAX_TTK)
                    } else {
                        att_pp.min(MAX_TTK)
                    }
                }
            };
            let t_def_stall =
                def_to_att.ttk.raw_ttk * (1.0 - sustain_w) + full_stall_def * sustain_w;

            let better = match best_option {
                None => true,
                Some((prev_a, prev_d, _)) => (t_att_sustain - t_def_stall) > (prev_a - prev_d),
            };
            if better {
                best_option = Some((t_att_sustain, t_def_stall, duty * sustain_w));
            }
        } else if has_wake_berry && max_norm < 0.90 && def_hp_frac < 0.85 {
            // Mint/Miracle Berry: one-time 1-turn full heal adds extra turns to attacker's TTK.
            let extra_att = if rest_net_dmg > 0.05 {
                ((1.0 - def_hp_frac.min(0.6)) / rest_net_dmg).min(4.0)
            } else {
                3.0
            };
            let t_att_berry =
                (att_to_def.ttk.raw_ttk + p_survive_t1 * extra_att).min(MAX_TTK);
            let t_def_berry = (def_to_att.ttk.raw_ttk + 1.0).min(MAX_TTK);
            let better = match best_option {
                None => true,
                Some((prev_a, prev_d, _)) => (t_att_berry - t_def_berry) > (prev_a - prev_d),
            };
            if better {
                best_option = Some((t_att_berry, t_def_berry, 0.25));
            }
        }
    }

    if let Some((t_att_sustain, t_def_stall, duty)) = best_option {
        match variant {
            TempoVariant::Full => {
                if t_att_sustain > att_to_def.ttk.raw_ttk + 0.1 {
                    att_to_def.ttk.eff_ttk = att_to_def.ttk.eff_ttk.max(t_att_sustain);
                    att_to_def.ttk.def_heal_duty = duty;
                    att_to_def.ttk.def_can_sustain = true;

                    let safe_attack_turns =
                        att_to_def.ttk.raw_ttk - if def_faster { 0.75 } else { 1.5 };
                    if def_to_att.ttk.raw_ttk > safe_attack_turns {
                        def_to_att.ttk.eff_ttk = def_to_att.ttk.eff_ttk.max(t_def_stall);
                    }
                }
            }
            TempoVariant::Raw => {
                let race_margin_def = att_to_def.ttk.raw_ttk - def_to_att.ttk.raw_ttk;
                let stall_margin_def = t_att_sustain - t_def_stall;
                if stall_margin_def > race_margin_def + 0.05 {
                    att_to_def.ttk.eff_ttk = att_to_def.ttk.eff_ttk.max(t_att_sustain);
                    att_to_def.ttk.def_heal_duty = duty;
                    att_to_def.ttk.def_can_sustain = true;
                    def_to_att.ttk.eff_ttk = def_to_att.ttk.eff_ttk.max(t_def_stall);
                }
            }
        }
    }
}

// =========================================================================
// 3v3 Team Golf Score Solver (Capacitated Optimal Assignment + Phaze Cap)
// =========================================================================

/// Check whether `side` has a living healthy Phazer/Hazer that can neutralize boosts on `foe_active`.
/// Returns `Some(reset_turns)` (how many turns opponent needs to wipe boosts) if so.
fn opponent_phaze_reset_turns(
    cb: &CompactBattle,
    table: &MatchTable,
    boosted_side: usize,
    ttk: &[[[PairTtk; 3]; 3]; 2],
) -> Option<f64> {
    let act_slot = cb.sides[boosted_side].active_idx as usize;
    let act_mon = &cb.mons[boosted_side * 3 + act_slot];
    if act_mon.hp() <= 0 {
        return None;
    }
    let has_pos_boost = act_mon.boosts[0] > 0
        || act_mon.boosts[1] > 0
        || act_mon.boosts[2] > 0
        || act_mon.boosts[3] > 0
        || act_mon.boosts[4] > 0
        || act_mon.boosts[6] > 0;
    if !has_pos_boost {
        return None;
    }

    let opp_side = 1 - boosted_side;
    let opp_act = cb.sides[opp_side].active_idx as usize;
    let mut best_reset: Option<f64> = None;

    for slot in 0..3usize {
        if (cb.sides[opp_side].alive_mask & (1 << slot)) == 0 {
            continue;
        }
        let idx = opp_side * 3 + slot;
        let mon = &cb.mons[idx];
        if mon.hp() <= 0 || mon.status_raw() == Status::Frz as u8 {
            continue;
        }
        if mon.status_raw() == Status::Slp as u8
            && expected_sleep_turns(cb, opp_side, slot) >= 1.0
        {
            continue;
        }
        let spec = &table.mons[idx];
        let mut has_phaze = false;
        let mut has_haze = false;
        for m_i in 0..spec.num_moves as usize {
            if mon.pp[m_i] == 0 || mon.is_slot_disabled(m_i) {
                continue;
            }
            match table.move_specs[idx][m_i].special_kind {
                SpecialMoveKind::Phaze => {
                    // Phaze fails when boosted side has only 1 mon alive.
                    if cb.sides[boosted_side].alive_mask.count_ones() >= 2 {
                        has_phaze = true;
                    }
                }
                SpecialMoveKind::Haze => {
                    has_haze = true;
                }
                _ => {}
            }
        }
        if !has_phaze && !has_haze {
            continue;
        }

        // How many hits does the phazer/hazer need to survive?
        // - Active Haze (faster): 0 hits! Active Haze (slower) or Active Phaze (-1 prio): 1 hit!
        // - Bench Phazer/Hazer: 1 entry hit + (0 if faster Haze else 1 hit) = 1..2 hits!
        let hits_needed = if slot == opp_act {
            if has_haze && ttk[opp_side][slot][act_slot].speed_cmp > 0 {
                0.5
            } else {
                1.0
            }
        } else if has_haze && ttk[opp_side][slot][act_slot].speed_cmp > 0 {
            1.2
        } else {
            2.0
        };

        let surv_turns = ttk[boosted_side][act_slot][slot].raw_ttk;
        if surv_turns > hits_needed + 0.15 {
            // Cost to reset = 1 turn if active (+ hp taken), 2 turns if bench (+ hp taken).
            let reset_cost = if slot == opp_act { 1.0 } else { 2.0 };
            best_reset = Some(match best_reset {
                None => reset_cost,
                Some(prev) => prev.min(reset_cost),
            });
        }
    }
    best_reset
}

/// Compute `side`'s optimal 3v3 Golf Plan `G_side(s)` given the full pairwise TTK table.
fn solve_team_plan(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    ttk: &[[[PairTtk; 3]; 3]; 2],
    unboosted_active_ttk: Option<&[PairTtk; 3]>,
    variant: TempoVariant,
) -> TeamPlan {
    let opp = 1 - side;
    let mut allies = [0usize; 3];
    let mut n_allies = 0usize;
    let mut enemies = [0usize; 3];
    let mut n_enemies = 0usize;

    for slot in 0..3 {
        if (cb.sides[side].alive_mask & (1 << slot)) != 0 && cb.mons[side * 3 + slot].hp() > 0 {
            allies[n_allies] = slot;
            n_allies += 1;
        }
        if (cb.sides[opp].alive_mask & (1 << slot)) != 0 && cb.mons[opp * 3 + slot].hp() > 0 {
            enemies[n_enemies] = slot;
            n_enemies += 1;
        }
    }

    if n_enemies == 0 {
        return TeamPlan { total_turns: 0.0, ..TeamPlan::default() };
    }
    if n_allies == 0 {
        return TeamPlan { total_turns: MAX_TTK * (n_enemies as f64), ..TeamPlan::default() };
    }

    let act_ally = cb.sides[side].active_idx as usize;
    let act_enemy = cb.sides[opp].active_idx as usize;
    let ally_active_alive =
        cb.mons[side * 3 + act_ally].hp() > 0 && cb.sides[side].request != 2;
    let enemy_active_alive =
        cb.mons[opp * 3 + act_enemy].hp() > 0 && cb.sides[opp].request != 2;

    // Precompute each ally's available HP budget (entry HP fraction).
    let mut base_hp_budget = [0.0f64; 3];
    for &a in &allies[..n_allies] {
        let hp = entry_hp(cb, table, side, a) as f64;
        let maxhp = table.mons[side * 3 + a].maxhp.max(1) as f64;
        base_hp_budget[a] = (hp / maxhp).clamp(0.0, 1.0);
    }

    // Active matchup speed tie-break (`-0.25` if active ally faster than active enemy, `+0.25` if slower).
    let active_speed_adj = if ally_active_alive && enemy_active_alive {
        match ttk[side][act_ally][act_enemy].speed_cmp {
            1 => -0.25,
            -1 => 0.25,
            _ => 0.0,
        }
    } else {
        0.0
    };

    // Fast path: 1v1 exact race.
    if n_allies == 1 && n_enemies == 1 {
        let a = allies[0];
        let e = enemies[0];
        let p = ttk[side][a][e];
        let total = (p.eff_ttk + active_speed_adj).max(0.25);
        return TeamPlan {
            total_turns: total,
            switch_turns: 0.0,
            overflow_turns: 0.0,
            phaze_cap_turns: 0.0,
            active_speed_adj,
            steps: vec![AssignmentStep {
                enemy_slot: e,
                killer_slot: a,
                turns: p.eff_ttk,
                hp_cost: 0.0,
                spilled: false,
            }],
        };
    }

    // Enumerate all `n_allies ^ n_enemies <= 27` primary-killer assignments.
    let total_combos = n_allies.pow(n_enemies as u32);
    let mut best_plan = TeamPlan {
        total_turns: f64::INFINITY,
        ..TeamPlan::default()
    };

    // Order enemies so the currently active enemy is evaluated FIRST (active-turn tempo).
    let mut ordered_enemies = [0usize; 3];
    {
        let mut pos = 0;
        for &e in &enemies[..n_enemies] {
            if e == act_enemy {
                ordered_enemies[pos] = e;
                pos += 1;
            }
        }
        for &e in &enemies[..n_enemies] {
            if e != act_enemy {
                ordered_enemies[pos] = e;
                pos += 1;
            }
        }
    }

    for combo in 0..total_combos {
        let mut assign = [0usize; 3];
        let mut x = combo;
        for idx_e in 0..n_enemies {
            assign[idx_e] = allies[x % n_allies];
            x /= n_allies;
        }

        let mut hp_rem = base_hp_budget;
        let mut used_mask = 0u8;
        let mut kill_turns = 0.0f64;
        let mut switch_turns = 0.0f64;
        let mut overflow_turns = 0.0f64;
        let mut steps = Vec::with_capacity(n_enemies);

        let mut infeasible = false;
        let mut active_switched_out = false;
        for idx_e in 0..n_enemies {
            let e = ordered_enemies[idx_e];
            let a = assign[idx_e];
            if a != act_ally {
                active_switched_out = true;
            }
            // In Pokemon, switching `act_ally` out wipes all its stat boosts!
            // `Full`: evaluate reused active unboosted; `Raw`: skip boost-wipe check.
            let pair_ae = if variant == TempoVariant::Full && a == act_ally && active_switched_out {
                unboosted_active_ttk.map_or(ttk[side][a][e], |row| row[e])
            } else {
                ttk[side][a][e]
            };
            let pair_ea = ttk[opp][e][a];

            // Do not allow starting a fresh enemy with an already-exhausted ally while another ally still has HP.
            if variant == TempoVariant::Full
                && hp_rem[a] <= 0.02
                && allies[..n_allies].iter().any(|&other| other != a && hp_rem[other] > 0.02)
            {
                infeasible = true;
                break;
            }

            // Switch cost:
            // - Active enemy handled by benched ally (only if `act_ally` is alive; forced switch is free!):
            //   +1.0 immediate switch turn (+ 1 free hit on entry).
            // - Subsequent enemy handled by a NEW distinct ally beyond the first: +0.5 pivot friction.
            let mut entry_hit_hp = 0.0f64;
            if ally_active_alive && enemy_active_alive && e == act_enemy && a != act_ally {
                switch_turns += 1.0;
                // Incoming free hit from active enemy when switching in.
                entry_hit_hp = pair_ea.net_dmg_frac.max(0.0);
            } else if used_mask != 0 && (used_mask & (1 << a)) == 0 {
                switch_turns += 0.5;
            }
            used_mask |= 1 << a;

            // HP consumed by ally `a` to kill enemy `e`.
            let t_need = pair_ae.eff_ttk;
            let hits_taken = if pair_ea.def_can_sustain {
                // Ally `a` can sustainably heal against `e`: net HP drain is minimal.
                0.15 * pair_ea.def_heal_duty
            } else {
                let raw_hits = if pair_ae.speed_cmp > 0 {
                    (t_need - 1.0).max(0.0)
                } else if pair_ae.speed_cmp < 0 {
                    t_need
                } else {
                    (t_need - 0.5).max(0.0)
                };
                // Discount hits taken if enemy is currently asleep without Sleep Talk.
                let enemy_sleep = expected_sleep_turns(cb, opp, e);
                let enemy_st = sleep_talk_attack_rate(cb, table, opp, e);
                let eff_hits = (raw_hits - enemy_sleep * (1.0 - enemy_st)).max(0.0);
                eff_hits * pair_ea.net_dmg_frac.max(0.0)
            };
            let hp_cost = (entry_hit_hp + hits_taken).min(10.0);

            let avail = hp_rem[a];
            if hp_cost <= avail + 1e-6 || hp_cost <= 1e-6 {
                // Ally `a` completes this KO within its HP budget!
                hp_rem[a] = (avail - hp_cost).max(0.0);
                kill_turns += t_need;
                steps.push(AssignmentStep {
                    enemy_slot: e,
                    killer_slot: a,
                    turns: t_need,
                    hp_cost,
                    spilled: false,
                });
            } else {
                let frac_done = if entry_hit_hp >= avail {
                    0.0
                } else {
                    let net_avail = (avail - entry_hit_hp).max(0.0);
                    let base_frac = if hits_taken > 1e-6 {
                        (net_avail / hits_taken).clamp(0.0, 0.999)
                    } else {
                        0.0
                    };
                    if variant == TempoVariant::Full && pair_ae.speed_cmp < 0 {
                        base_frac * (1.0 - pair_ea.ohko_prob).clamp(0.0, 1.0)
                    } else {
                        base_frac
                    }
                };
                let rem_frac = (1.0 - frac_done).max(0.0);
                let turns_by_a = if frac_done <= 0.02 && pair_ae.speed_cmp <= 0 {
                    1.0
                } else {
                    frac_done * t_need
                };
                hp_rem[a] = 0.0;

                let mut best_spill = match variant {
                    TempoVariant::Full => {
                        let deficit_hp = (hp_cost - avail).clamp(0.0, 2.0);
                        (rem_frac * t_need + 2.0 * deficit_hp)
                            .min(MAX_TTK - turns_by_a.min(MAX_TTK - 0.5))
                    }
                    TempoVariant::Raw => rem_frac * MAX_TTK + 2.0,
                };
                let mut best_helper: Option<(usize, f64)> = None;
                for &b in &allies[..n_allies] {
                    if b == a || hp_rem[b] <= 0.02 {
                        continue;
                    }
                    let pair_be = ttk[side][b][e];
                    let t_b = match variant {
                        TempoVariant::Full => {
                            let raw_fin = rem_frac * pair_be.raw_ttk;
                            let eff_fin = rem_frac * pair_be.eff_ttk;
                            let stall_w = ((raw_fin - 1.0) / 1.5).clamp(0.0, 1.0);
                            raw_fin * (1.0 - stall_w) + eff_fin * stall_w
                        }
                        TempoVariant::Raw => (rem_frac * pair_be.eff_ttk).max(0.5),
                    };
                    let cost_b = if ttk[opp][e][b].def_can_sustain {
                        0.1 * rem_frac
                    } else {
                        let hits_b = if pair_be.speed_cmp > 0 {
                            (t_b - 1.0).max(0.0)
                        } else {
                            t_b
                        };
                        hits_b * ttk[opp][e][b].net_dmg_frac.max(0.0)
                    };
                    let extra_pen = if cost_b > hp_rem[b] {
                        2.5 * (cost_b - hp_rem[b])
                    } else {
                        0.0
                    };
                    let pivot_pen = if (used_mask & (1 << b)) == 0 {
                        match variant {
                            TempoVariant::Full => 0.5 * (rem_frac * 2.0).min(1.0),
                            TempoVariant::Raw => 0.5,
                        }
                    } else {
                        0.0
                    };
                    let cand = t_b + extra_pen + pivot_pen;
                    if cand < best_spill {
                        best_spill = cand;
                        best_helper = Some((b, cost_b));
                    }
                }
                if let Some((b, cost_b)) = best_helper {
                    hp_rem[b] = (hp_rem[b] - cost_b).max(0.0);
                    used_mask |= 1 << b;
                }
                let total_step_turns = turns_by_a + best_spill;
                kill_turns += turns_by_a;
                overflow_turns += best_spill;
                steps.push(AssignmentStep {
                    enemy_slot: e,
                    killer_slot: a,
                    turns: total_step_turns,
                    hp_cost: avail,
                    spilled: true,
                });
            }
        }
        if infeasible {
            continue;
        }

        // Speed bonus applies when the active ally actually fights the active enemy.
        let speed_adj = if assign[0] == act_ally {
            active_speed_adj
        } else {
            0.0
        };
        let total = (kill_turns + switch_turns + overflow_turns + speed_adj).max(0.25);
        if total < best_plan.total_turns {
            best_plan = TeamPlan {
                total_turns: total,
                switch_turns,
                overflow_turns,
                phaze_cap_turns: 0.0,
                active_speed_adj: speed_adj,
                steps,
            };
        }
    }

    // Phazer / Hazer cap (`Full` only): if active ally has boosts AND opponent has a healthy living
    // Phazer/Hazer that survives resetting them, boosts cannot save more than `reset_turns`.
    if variant == TempoVariant::Full {
        if let Some(unb) = unboosted_active_ttk {
            if let Some(reset_turns) = opponent_phaze_reset_turns(cb, table, side, ttk) {
                let mut unb_ttk = *ttk;
                unb_ttk[side][act_ally] = *unb;
                let unb_plan = solve_team_plan(cb, table, side, &unb_ttk, None, variant);
                let boost_savings = unb_plan.total_turns - best_plan.total_turns;
                if boost_savings > reset_turns {
                    let excess = boost_savings - reset_turns;
                    best_plan.phaze_cap_turns = excess;
                    best_plan.total_turns += excess;
                }
            }
        }
    }

    best_plan
}

// =========================================================================
// Main Public Evaluation API
// =========================================================================

/// Evaluate a `CompactBattle` in turn-economy ("Golf") units (`TempoVariant::Full`).
pub fn evaluate_compact_tempo(cb: &CompactBattle, table: &MatchTable) -> TempoEval {
    evaluate_compact_tempo_variant(cb, table, TempoVariant::Full)
}

/// Evaluate a `CompactBattle` in turn-economy ("Golf") units under `variant` (`Full` or `Raw`).
pub fn evaluate_compact_tempo_variant(
    cb: &CompactBattle,
    table: &MatchTable,
    variant: TempoVariant,
) -> TempoEval {
    if let Some(wp) = compact_turn_limit_outcome(cb, table) {
        let (g0, g1) = if wp > 0.75 {
            (0.0, 6.0)
        } else if wp < 0.25 {
            (6.0, 0.0)
        } else {
            (0.0, 0.0)
        };
        return TempoEval {
            g_turns: [g0, g1],
            turn_advantage: g1 - g0,
            win_prob: wp,
            plans: [TeamPlan::default(), TeamPlan::default()],
            ttk: [[[PairTtk::default(); 3]; 3]; 2],
        };
    }

    let mut raw = [
        [
            [
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
            ],
            [
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
            ],
            [
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
            ],
        ],
        [
            [
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
            ],
            [
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
            ],
            [
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
                RawPairInfo {
                    ttk: PairTtk::default(),
                    max_norm_dmg_frac: 0.0,
                    max_crit_dmg_frac: 0.0,
                    best_move_pp: 0,
                },
            ],
        ],
    ];

    // Step 1: Raw exact-HP TTK for all living pairs.
    for s0 in 0..3 {
        if (cb.sides[0].alive_mask & (1 << s0)) == 0 {
            continue;
        }
        for s1 in 0..3 {
            if (cb.sides[1].alive_mask & (1 << s1)) == 0 {
                continue;
            }
            raw[0][s0][s1] = compute_raw_pair(cb, table, 0, s0, 1, s1);
            raw[1][s1][s0] = compute_raw_pair(cb, table, 1, s1, 0, s0);
        }
    }

    // Step 2: Coupled recovery dynamics (50% heal / Rest / Sleep Talk / Mint Berry) symmetrically.
    let mut ttk = [[[PairTtk::default(); 3]; 3]; 2];
    for s0 in 0..3 {
        if (cb.sides[0].alive_mask & (1 << s0)) == 0 {
            continue;
        }
        for s1 in 0..3 {
            if (cb.sides[1].alive_mask & (1 << s1)) == 0 {
                continue;
            }
            let mut r01 = RawPairInfo {
                ttk: raw[0][s0][s1].ttk,
                max_norm_dmg_frac: raw[0][s0][s1].max_norm_dmg_frac,
                max_crit_dmg_frac: raw[0][s0][s1].max_crit_dmg_frac,
                best_move_pp: raw[0][s0][s1].best_move_pp,
            };
            let mut r10 = RawPairInfo {
                ttk: raw[1][s1][s0].ttk,
                max_norm_dmg_frac: raw[1][s1][s0].max_norm_dmg_frac,
                max_crit_dmg_frac: raw[1][s1][s0].max_crit_dmg_frac,
                best_move_pp: raw[1][s1][s0].best_move_pp,
            };
            // Evaluate both defenders independently against raw offense, then merge cleanly.
            let mut r01_for_d1 = RawPairInfo {
                ttk: r01.ttk,
                max_norm_dmg_frac: r01.max_norm_dmg_frac,
                max_crit_dmg_frac: r01.max_crit_dmg_frac,
                best_move_pp: r01.best_move_pp,
            };
            let mut r10_for_d1 = RawPairInfo {
                ttk: r10.ttk,
                max_norm_dmg_frac: r10.max_norm_dmg_frac,
                max_crit_dmg_frac: r10.max_crit_dmg_frac,
                best_move_pp: r10.best_move_pp,
            };
            apply_recovery_for_defender(
                cb,
                table,
                0,
                s0,
                1,
                s1,
                &mut r01_for_d1,
                &mut r10_for_d1,
                variant,
            );

            let mut r10_for_d0 = RawPairInfo {
                ttk: r10.ttk,
                max_norm_dmg_frac: r10.max_norm_dmg_frac,
                max_crit_dmg_frac: r10.max_crit_dmg_frac,
                best_move_pp: r10.best_move_pp,
            };
            let mut r01_for_d0 = RawPairInfo {
                ttk: r01.ttk,
                max_norm_dmg_frac: r01.max_norm_dmg_frac,
                max_crit_dmg_frac: r01.max_crit_dmg_frac,
                best_move_pp: r01.best_move_pp,
            };
            apply_recovery_for_defender(
                cb,
                table,
                1,
                s1,
                0,
                s0,
                &mut r10_for_d0,
                &mut r01_for_d0,
                variant,
            );

            r01.ttk = r01_for_d1.ttk;
            r01.ttk.eff_ttk = r01.ttk.eff_ttk.max(r01_for_d0.ttk.eff_ttk);
            r10.ttk = r10_for_d0.ttk;
            r10.ttk.eff_ttk = r10.ttk.eff_ttk.max(r10_for_d1.ttk.eff_ttk);

            ttk[0][s0][s1] = r01.ttk;
            ttk[1][s1][s0] = r10.ttk;
        }
    }

    // Step 3: Whenever an active mon has non-zero stat boosts (`Full`), also precompute unboosted TTKs
    // so `solve_team_plan` wipes boosts on switch-out and applies phaze caps.
    let unb_row = |s: usize| -> Option<[PairTtk; 3]> {
        if variant == TempoVariant::Raw {
            return None;
        }
        let act = cb.sides[s].active_idx as usize;
        let act_mon = &cb.mons[s * 3 + act];
        if act_mon.hp() <= 0 || act_mon.boosts == [0; 7] {
            return None;
        }
        let mut cb_unb = *cb;
        cb_unb.mons[s * 3 + act].boosts = [0; 7];
        let opp = 1 - s;
        let mut row = [PairTtk::default(); 3];
        for e in 0..3 {
            if (cb.sides[opp].alive_mask & (1 << e)) == 0 {
                continue;
            }
            let mut r_ae = compute_raw_pair(&cb_unb, table, s, act, opp, e);
            let mut r_ea = compute_raw_pair(&cb_unb, table, opp, e, s, act);
            apply_recovery_for_defender(
                &cb_unb,
                table,
                s,
                act,
                opp,
                e,
                &mut r_ae,
                &mut r_ea,
                variant,
            );
            row[e] = r_ae.ttk;
        }
        Some(row)
    };
    let unb0 = unb_row(0);
    let unb1 = unb_row(1);

    let plan0 = solve_team_plan(cb, table, 0, &ttk, unb0.as_ref(), variant);
    let plan1 = solve_team_plan(cb, table, 1, &ttk, unb1.as_ref(), variant);
    let g0 = plan0.total_turns;
    let g1 = plan1.total_turns;
    let turn_advantage = g1 - g0;
    let win_prob = turns_to_win_prob(g0, g1);

    TempoEval {
        g_turns: [g0, g1],
        turn_advantage,
        win_prob,
        plans: [plan0, plan1],
        ttk,
    }
}

/// Evaluate a full `Battle` state in turn-economy ("Golf") units (`TempoVariant::Full`).
pub fn evaluate_tempo(battle: &Battle, dex: &Dex) -> TempoEval {
    evaluate_tempo_variant(battle, dex, TempoVariant::Full)
}

/// Evaluate a full `Battle` state in turn-economy ("Golf") units under `variant`.
pub fn evaluate_tempo_variant(battle: &Battle, dex: &Dex, variant: TempoVariant) -> TempoEval {
    if let Some(o) = battle.outcome() {
        let (g0, g1, wp) = match o {
            Outcome::P1Win => (0.0, 6.0, 1.0),
            Outcome::P2Win => (6.0, 0.0, 0.0),
            Outcome::Tie => (0.0, 0.0, 0.5),
        };
        return TempoEval {
            g_turns: [g0, g1],
            turn_advantage: g1 - g0,
            win_prob: wp,
            plans: [TeamPlan::default(), TeamPlan::default()],
            ttk: [[[PairTtk::default(); 3]; 3]; 2],
        };
    }
    let (table, cb) = CompactBattle::from_battle(battle, dex);
    evaluate_compact_tempo_variant(&cb, &table, variant)
}

/// Format a concise human-readable turn-economy breakdown of `battle`.
pub fn format_tempo_explanation(battle: &Battle, dex: &Dex, eval: &TempoEval) -> String {
    let mut out = String::new();
    let mon_name = |side: usize, slot: usize| -> String {
        if let Some(&roster_idx) = battle.sides[side].party.get(slot) {
            let p = &battle.sides[side].roster[roster_idx as usize];
            format!("{}({}/{})", p.name.as_str(), p.hp, p.maxhp)
        } else {
            format!("P{}#{}", side + 1, slot)
        }
    };

    out.push_str(&format!(
        "Golf Turns: P1={:.2}t vs P2={:.2}t | ΔG(P1)={:+.2}t | P1 WinProb={:.1}%\n",
        eval.g_turns[0],
        eval.g_turns[1],
        eval.turn_advantage,
        eval.win_prob * 100.0
    ));
    for s in 0..2 {
        let plan = &eval.plans[s];
        let opp = 1 - s;
        out.push_str(&format!(
            "  P{} Plan ({:.2}t = kills + sw {:.2}t + ovf {:.2}t + phaze_cap {:.2}t + spd {:+.2}t):\n",
            s + 1,
            plan.total_turns,
            plan.switch_turns,
            plan.overflow_turns,
            plan.phaze_cap_turns,
            plan.active_speed_adj
        ));
        for step in &plan.steps {
            let pair = eval.ttk[s][step.killer_slot][step.enemy_slot];
            let wall_tag = if pair.def_can_sustain {
                format!(" [foe heals duty={:.0}%]", pair.def_heal_duty * 100.0)
            } else {
                String::new()
            };
            let spill_tag = if step.spilled { " [spilled]" } else { "" };
            out.push_str(&format!(
                "    {} -> {} : {:.2}t (raw {:.2}t, dmg {:.0}%/t, hp_cost {:.0}%){}{}\n",
                mon_name(s, step.killer_slot),
                mon_name(opp, step.enemy_slot),
                step.turns,
                pair.raw_ttk,
                pair.net_dmg_frac * 100.0,
                step.hp_cost * 100.0,
                wall_tag,
                spill_tag
            ));
        }
    }
    out
}

// =========================================================================
// 1-Ply Simultaneous-Move Matrix Game & Standalone TempoAgent
// =========================================================================

fn encode_choice(
    table: &MatchTable,
    cb: &CompactBattle,
    dex: &Dex,
    side: usize,
    choice: SearchChoice,
) -> Option<u8> {
    let act_slot = cb.sides[side].active_idx.min(2) as usize;
    let act_idx = side * 3 + act_slot;
    let num_moves = table.mons[act_idx].num_moves as usize;
    match choice {
        SearchChoice::Pass => Some(7),
        SearchChoice::Move(id) => {
            if let Some(m_i) = (0..num_moves).find(|&m_i| table.mons[act_idx].moves[m_i] == id) {
                return Some(m_i as u8);
            }
            let mon = &cb.mons[act_idx];
            if mon.must_recharge() || Some(id) == dex.known_moves.recharge {
                return Some(0);
            }
            if Some(id) == dex.known_moves.struggle {
                return Some(0);
            }
            if mon.is_multi_turn_locked() {
                return Some(mon.locked_move_slot());
            }
            None
        }
        SearchChoice::Switch(pos) if (1..=3).contains(&pos) => Some(pos + 3),
        _ => None,
    }
}

/// Resolve any post-turn forced switch (`request == 2`) in `cb` cleanly using `evaluate_compact_tempo_variant`.
fn resolve_forced_switches_variant(cb: &mut CompactBattle, table: &MatchTable, variant: TempoVariant) {
    for _ in 0..3 {
        if cb.outcome != 0 {
            return;
        }
        let r0 = cb.sides[0].request == 2;
        let r1 = cb.sides[1].request == 2;
        if !r0 && !r1 {
            return;
        }
        let bench_opts = |s: usize| -> Vec<u8> {
            let act = cb.sides[s].active_idx as usize;
            let mut v = Vec::with_capacity(2);
            for slot in 0..3 {
                if slot != act
                    && (cb.sides[s].alive_mask & (1 << slot)) != 0
                    && cb.mons[s * 3 + slot].hp() > 0
                {
                    v.push((slot + 4) as u8);
                }
            }
            if v.is_empty() {
                v.push(4);
            }
            v
        };
        let opts0 = if r0 { bench_opts(0) } else { vec![7] };
        let opts1 = if r1 { bench_opts(1) } else { vec![7] };

        let (pick0, pick1) = if r0 && !r1 {
            let best = opts0
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let mut ca = *cb;
                    ca.step_switch(table, [a, 7]);
                    let mut cb_b = *cb;
                    cb_b.step_switch(table, [b, 7]);
                    evaluate_compact_tempo_variant(&ca, table, variant)
                        .win_prob
                        .total_cmp(&evaluate_compact_tempo_variant(&cb_b, table, variant).win_prob)
                })
                .unwrap();
            (best, 7)
        } else if !r0 && r1 {
            let best = opts1
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let mut ca = *cb;
                    ca.step_switch(table, [7, a]);
                    let mut cb_b = *cb;
                    cb_b.step_switch(table, [7, b]);
                    evaluate_compact_tempo_variant(&ca, table, variant)
                        .win_prob
                        .total_cmp(&evaluate_compact_tempo_variant(&cb_b, table, variant).win_prob)
                })
                .unwrap();
            (7, best)
        } else {
            let mut best_p0 = opts0[0];
            let mut best_val0 = f64::NEG_INFINITY;
            for &a in &opts0 {
                let mut worst = f64::INFINITY;
                for &b in &opts1 {
                    let mut c = *cb;
                    c.step_switch(table, [a, b]);
                    let v = evaluate_compact_tempo_variant(&c, table, variant).win_prob;
                    if v < worst {
                        worst = v;
                    }
                }
                if worst > best_val0 {
                    best_val0 = worst;
                    best_p0 = a;
                }
            }
            let best_p1 = opts1
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let mut ca = *cb;
                    ca.step_switch(table, [best_p0, a]);
                    let mut cb_b = *cb;
                    cb_b.step_switch(table, [best_p0, b]);
                    evaluate_compact_tempo_variant(&ca, table, variant)
                        .win_prob
                        .total_cmp(&evaluate_compact_tempo_variant(&cb_b, table, variant).win_prob)
                })
                .unwrap();
            (best_p0, best_p1)
        };
        cb.step_switch(table, [pick0, pick1]);
    }
}

/// Analyze a simultaneous-move decision point with 1-ply engine lookahead + Nash LP (`Full`).
pub fn analyze_turn_matrix(
    battle: &Battle,
    dex: &Dex,
    side: usize,
    own_choices: &[SearchChoice],
    samples_per_cell: usize,
    seed: u64,
) -> TurnMatrixAnalysis {
    analyze_turn_matrix_variant(
        battle,
        dex,
        side,
        own_choices,
        samples_per_cell,
        seed,
        TempoVariant::Full,
    )
}

/// Analyze a simultaneous-move decision point with 1-ply engine lookahead + Nash LP under `variant`.
pub fn analyze_turn_matrix_variant(
    battle: &Battle,
    dex: &Dex,
    side: usize,
    own_choices: &[SearchChoice],
    samples_per_cell: usize,
    seed: u64,
    variant: TempoVariant,
) -> TurnMatrixAnalysis {
    let opp = 1 - side;
    let mut b_tmp = battle.clone();
    b_tmp.log_enabled = false;
    let opp_choices = b_tmp.legal_choices(dex, opp);
    let (table, base_cb) = CompactBattle::from_battle(battle, dex);

    let n_own = own_choices.len();
    let n_opp = opp_choices.len().max(1);
    let mut win_matrix = vec![0.5; n_own * n_opp];
    let mut turn_matrix = vec![0.0; n_own * n_opp];

    let k = samples_per_cell.max(1);
    let mut rng = SplitMix64::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let cell_seeds: Vec<u64> = (0..k).map(|_| rng.next()).collect();

    for (i, &own_c) in own_choices.iter().enumerate() {
        let Some(a_own) = encode_choice(&table, &base_cb, dex, side, own_c) else {
            continue;
        };
        for (j, &opp_c) in opp_choices.iter().enumerate() {
            let Some(a_opp) = encode_choice(&table, &base_cb, dex, opp, opp_c) else {
                continue;
            };
            let actions = if side == 0 {
                [a_own, a_opp]
            } else {
                [a_opp, a_own]
            };

            let mut sum_wp = 0.0;
            let mut sum_tg = 0.0;
            for &s in &cell_seeds {
                let mut cb = base_cb;
                cb.prng_seed = s;
                cb.step_turn(&table, actions);
                resolve_forced_switches_variant(&mut cb, &table, variant);
                let ev = evaluate_compact_tempo_variant(&cb, &table, variant);
                sum_wp += ev.win_prob_for(side);
                sum_tg += ev.advantage_for(side);
            }
            win_matrix[i * n_opp + j] = sum_wp / (k as f64);
            turn_matrix[i * n_opp + j] = sum_tg / (k as f64);
        }
    }

    let sol = solve_matrix_full(&win_matrix, n_own, n_opp);
    let mut eq_turn_adv = 0.0;
    for i in 0..n_own {
        for j in 0..n_opp {
            eq_turn_adv += sol.x[i] * sol.y[j] * turn_matrix[i * n_opp + j];
        }
    }

    TurnMatrixAnalysis {
        own_choices: own_choices.to_vec(),
        opp_choices,
        win_matrix,
        turn_matrix,
        own_policy: sol.x,
        opp_policy: sol.y,
        eq_win_prob: sol.value,
        eq_turn_adv,
    }
}

/// Fast 2-stage Team Preview selector shared across `TempoAgent` and `TempoMctsAgent`:
/// Stage 1 scores unique 3-mon subsets (up to 20) against 2 opponent samples;
/// Stage 2 picks the best lead among the top 3 subsets against 3 opponent samples.
fn fast_tempo_team_preview(
    battle: &Battle,
    dex: &Dex,
    side: usize,
    choices: &[SearchChoice],
    variant: TempoVariant,
    use_fast_golf: bool,
) -> SearchChoice {
    if choices.len() <= 1 {
        return choices[0];
    }
    let opp = 1 - side;
    let mut b_tmp = battle.clone();
    b_tmp.log_enabled = false;
    let opp_choices = b_tmp.legal_choices(dex, opp);
    let step2 = (opp_choices.len() / 2).max(1);
    let opp_s2: Vec<SearchChoice> = opp_choices.iter().step_by(step2).take(2).copied().collect();
    let step3 = (opp_choices.len() / 3).max(1);
    let opp_s3: Vec<SearchChoice> = opp_choices.iter().step_by(step3).take(3).copied().collect();

    // Group choices by sorted 3-mon subset.
    let mut subsets: Vec<([u8; 3], Vec<SearchChoice>)> = Vec::new();
    for &c in choices {
        if let SearchChoice::Team(t) = c {
            let mut key = t;
            key.sort_unstable();
            if let Some((_, list)) = subsets.iter_mut().find(|(k, _)| *k == key) {
                list.push(c);
            } else {
                subsets.push((key, vec![c]));
            }
        }
    }
    if subsets.is_empty() {
        return choices[0];
    }

    let score_pick = |my_c: SearchChoice, opp_list: &[SearchChoice]| -> f64 {
        let mut sum = 0.0;
        let mut cnt = 0usize;
        for &opp_c in opp_list {
            let mut sim = battle.clone();
            sim.log_enabled = false;
            let picks = if side == 0 {
                [Some(my_c), Some(opp_c)]
            } else {
                [Some(opp_c), Some(my_c)]
            };
            if sim.apply_choices(dex, picks).is_ok() {
                let wp = if use_fast_golf {
                    let (t, cb) = CompactBattle::from_battle(&sim, dex);
                    let fg = evaluate_fast_golf(&cb, &t, None);
                    if side == 0 { fg.win_prob } else { 1.0 - fg.win_prob }
                } else {
                    evaluate_tempo_variant(&sim, dex, variant).win_prob_for(side)
                };
                sum += wp;
                cnt += 1;
            }
        }
        if cnt > 0 { sum / (cnt as f64) } else { 0.5 }
    };

    let mut ranked: Vec<(f64, usize)> = subsets
        .iter()
        .enumerate()
        .map(|(idx, (_, list))| (score_pick(list[0], &opp_s2), idx))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut best_choice = choices[0];
    let mut best_score = f64::NEG_INFINITY;
    for &(_, idx) in ranked.iter().take(3) {
        for &my_c in &subsets[idx].1 {
            let sc = score_pick(my_c, &opp_s3);
            if sc > best_score {
                best_score = sc;
                best_choice = my_c;
            }
        }
    }
    best_choice
}

/// Standalone Turn-Economy ("Golf") Agent (1-ply simultaneous Nash equilibrium over `TempoEval`).
pub struct TempoAgent {
    pub samples_per_cell: usize,
    pub deterministic: bool,
    pub variant: TempoVariant,
    pub fixed_preview: bool,
    rng: SplitMix64,
}

impl TempoAgent {
    pub fn new(seed: u64) -> Self {
        Self {
            samples_per_cell: 8,
            deterministic: false,
            variant: TempoVariant::Full,
            fixed_preview: false,
            rng: SplitMix64::new(seed),
        }
    }

    pub fn with_samples(seed: u64, samples_per_cell: usize, deterministic: bool) -> Self {
        Self {
            samples_per_cell: samples_per_cell.max(1),
            deterministic,
            variant: TempoVariant::Full,
            fixed_preview: false,
            rng: SplitMix64::new(seed),
        }
    }

    pub fn with_variant(
        seed: u64,
        samples_per_cell: usize,
        deterministic: bool,
        variant: TempoVariant,
    ) -> Self {
        Self {
            samples_per_cell: samples_per_cell.max(1),
            deterministic,
            variant,
            fixed_preview: false,
            rng: SplitMix64::new(seed),
        }
    }
}

impl Agent for TempoAgent {
    fn name(&self) -> String {
        let tag = match self.variant {
            TempoVariant::Full => "full",
            TempoVariant::Raw => "raw",
        };
        format!("tempo-1ply-{tag}:{}", self.samples_per_cell)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        if matches!(choices[0], SearchChoice::Team(_)) {
            if self.fixed_preview {
                let d = SearchChoice::Team([1, 2, 3]);
                return if choices.contains(&d) { d } else { choices[0] };
            }
            return fast_tempo_team_preview(battle, dex, side, choices, self.variant, false);
        }

        // Forced switch (no Move actions): pick the bench switch maximizing TempoEval win prob.
        let has_moves = choices.iter().any(|c| matches!(c, SearchChoice::Move(_)));
        if !has_moves {
            let (table, base_cb) = CompactBattle::from_battle(battle, dex);
            let variant = self.variant;
            return choices
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let score = |c: SearchChoice| -> f64 {
                        let Some(act) = encode_choice(&table, &base_cb, dex, side, c) else {
                            return -1.0;
                        };
                        let mut cb = base_cb;
                        let sw = if side == 0 { [act, 7] } else { [7, act] };
                        cb.step_switch(&table, sw);
                        evaluate_compact_tempo_variant(&cb, &table, variant).win_prob_for(side)
                    };
                    score(a).total_cmp(&score(b))
                })
                .unwrap();
        }

        let seed = self.rng.next();
        let analysis = analyze_turn_matrix_variant(
            battle,
            dex,
            side,
            choices,
            self.samples_per_cell,
            seed,
            self.variant,
        );

        if self.deterministic {
            let best_idx = analysis
                .own_policy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            choices[best_idx]
        } else {
            let mut r = self.rng.next_f64();
            for (i, &p) in analysis.own_policy.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return choices[i];
                }
            }
            choices[choices.len() - 1]
        }
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        if choices.len() == 1 || !choices.iter().any(|c| matches!(c, SearchChoice::Move(_))) {
            let pick = self.choose(battle, dex, side, choices);
            return choices.iter().map(|&c| if c == pick { 1.0 } else { 0.0 }).collect();
        }
        let seed = self.rng.next();
        analyze_turn_matrix_variant(
            battle,
            dex,
            side,
            choices,
            self.samples_per_cell,
            seed,
            self.variant,
        )
        .own_policy
    }
}

// =========================================================================
// CompactBattle-Native Zero-Rollout DUCT SM-MCTS (`TempoMctsAgent`)
// =========================================================================

#[derive(Clone, Copy, Debug)]
pub struct TempoMctsConfig {
    pub iterations: u32,
    pub c: f64,
    pub max_depth: u8,
    pub leaf: TempoLeafMode,
    pub deterministic: bool,
    pub fixed_preview: bool,
    /// Use empirical-Bayes minimax backup (`backed_up_value`, matching `smmcts.rs`) instead of raw average backup.
    pub minimax_backup: bool,
    /// Use `argmax visits` (with `dominated_reason` guard, matching `skuct`) instead of whole-history Nash LP at root.
    pub argmax_root: bool,
    /// Optional `CompactBattle` heavy rollout turns (`0` = pure zero-rollout leaf eval).
    pub rollout_turns: u16,
}

impl Default for TempoMctsConfig {
    fn default() -> Self {
        Self {
            iterations: 300,
            c: 0.4,
            max_depth: 16,
            leaf: TempoLeafMode::TempoFull,
            deterministic: false,
            fixed_preview: false,
            minimax_backup: false,
            argmax_root: false,
            rollout_turns: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TempoMctsResult {
    pub own_choices: Vec<SearchChoice>,
    pub own_policy: Vec<f64>,
    pub visits: Vec<u32>,
    pub means: Vec<f64>,
    pub eq_win_prob: f64,
    pub mean_depth: f64,
    pub nodes: usize,
}

#[derive(Clone, Copy)]
struct DuctNode {
    n_acts: [u8; 2],
    acts: [[u8; 6]; 2],
    noop_mask: [u8; 2],
    priors_ready: bool,
    total: u32,
    priors_q: [[u16; 6]; 2],
    visits: [[u32; 6]; 2],
    sums: [[f64; 6]; 2],
}

impl DuctNode {
    fn new(cb: &CompactBattle, table: &MatchTable) -> Self {
        let mut tmp0 = [0u8; 7];
        let mut tmp1 = [0u8; 7];
        let n0 = cb.legal_actions(table, 0, &mut tmp0).min(6) as u8;
        let n1 = cb.legal_actions(table, 1, &mut tmp1).min(6) as u8;
        let mut acts = [[0u8; 6]; 2];
        acts[0].copy_from_slice(&tmp0[..6]);
        acts[1].copy_from_slice(&tmp1[..6]);
        Self {
            n_acts: [n0, n1],
            acts,
            noop_mask: [0; 2],
            priors_ready: false,
            total: 0,
            priors_q: [[0; 6]; 2],
            visits: [[0; 6]; 2],
            sums: [[0.0; 6]; 2],
        }
    }

    #[inline]
    fn ensure_puct_priors(
        &mut self,
        cb: &CompactBattle,
        table: &MatchTable,
        cache: Option<&TempoMatchCache>,
        eps_p: f32,
        use_cf_policy: bool,
    ) {
        if self.priors_ready {
            return;
        }
        let ((p0, m0), (p1, m1)) = crate::compact_az::policy_priors_both_mode(
            cb,
            table,
            cache,
            &self.acts[0],
            self.n_acts[0] as usize,
            &self.acts[1],
            self.n_acts[1] as usize,
            eps_p,
            use_cf_policy,
        );
        for i in 0..6 {
            self.priors_q[0][i] = (p0[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            self.priors_q[1][i] = (p1[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
        }
        self.noop_mask[0] = m0;
        self.noop_mask[1] = m1;
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
            let mut best_p = 0u16;
            let mut found = false;
            for i in 0..n {
                if ((mask >> i) & 1) == 0 && (!found || self.priors_q[side][i] > best_p) {
                    found = true;
                    best_p = self.priors_q[side][i];
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
        let scale = c_puct * (1.0 / 65535.0) * f64::from(self.total.max(1)).sqrt();
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
            let u = q + scale * f64::from(self.priors_q[side][i]) / f64::from(vi + 1);
            if u > best_u {
                best_u = u;
                best_i = i;
            }
        }
        best_i
    }

    #[inline]
    fn select_ucb(&self, side: usize, c: f64, rng: &mut SplitMix64, rand_untried: bool) -> usize {
        let n = self.n_acts[side] as usize;
        if n <= 1 {
            return 0;
        }
        if rand_untried {
            let mut n_untried = 0usize;
            for i in 0..n {
                if self.visits[side][i] == 0 {
                    n_untried += 1;
                }
            }
            if n_untried > 0 {
                let mut target = rng.below(n_untried);
                for i in 0..n {
                    if self.visits[side][i] == 0 {
                        if target == 0 {
                            return i;
                        }
                        target -= 1;
                    }
                }
            }
        } else {
            for i in 0..n {
                if self.visits[side][i] == 0 {
                    return i;
                }
            }
        }
        let log_n = (self.total.max(1) as f64).ln();
        let mut best_i = 0;
        let mut best_u = f64::NEG_INFINITY;
        for i in 0..n {
            let v = self.visits[side][i] as f64;
            let raw_mean = self.sums[side][i] / v;
            let mean = if side == 0 { raw_mean } else { 1.0 - raw_mean };
            let u = mean + c * (log_n / v).sqrt();
            if u > best_u {
                best_u = u;
                best_i = i;
            }
        }
        best_i
    }

    /// Empirical-Bayes minimax value contraction matching `smmcts::Node::backed_up_value`.
    #[inline]
    fn backed_up_value(&self, raw_val0: f64) -> f64 {
        self.backed_up_value_cfg(raw_val0, 0.85, 1.5)
    }

    #[inline]
    fn backed_up_value_cfg(&self, raw_val0: f64, lambda_max: f64, tau: f64) -> f64 {
        if lambda_max <= 0.0 {
            return raw_val0;
        }
        let n0_len = self.n_acts[0] as usize;
        let n1_len = self.n_acts[1] as usize;
        let mut n0 = 0u32;
        let mut w0 = 0.0f64;
        for i in 0..n0_len {
            n0 += self.visits[0][i];
            w0 += self.sums[0][i];
        }
        let mut n1 = 0u32;
        let mut w1_s0 = 0.0f64;
        for j in 0..n1_len {
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
        let gain0 = if n0_len > 1 && n0 >= 2 {
            let mut mu0 = r_bar0;
            for i in 0..n0_len {
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
        let blunder1 = if n1_len > 1 && n1 >= 2 {
            let r_bar1 = 1.0 - (w1_s0 / f64::from(n1));
            let mut mu1 = r_bar1;
            for j in 0..n1_len {
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
        let lambda = lambda_max * (f64::from(n) / (f64::from(n) + 6.0));
        ((1.0 - lambda) * raw_val0 + lambda * v_star).clamp(0.000_002, 0.999_998)
    }
}

#[inline]
fn mix64(mut h: u64, v: u64) -> u64 {
    h ^= v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = h.rotate_left(27).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^ (h >> 31)
}

#[inline]
fn compact_world_has_qc(cb: &CompactBattle, table: &MatchTable) -> bool {
    for i in 0..6usize {
        if table.item_spec(cb.mons[i].item).is_quick_claw {
            return true;
        }
    }
    false
}

#[inline]
fn compact_observable_qc_bits(cb: &CompactBattle, table: &MatchTable, any_qc: bool) -> u8 {
    if !any_qc || cb.quick_claw_roll == 0 {
        return 0;
    }
    let non_idle = cb.sides[0].request != 1 || cb.sides[1].request != 1;
    let mut live_qc = 0u8;
    for s in 0..2usize {
        let mask = cb.sides[s].alive_mask;
        for slot in 0..3usize {
            let idx = s * 3 + slot;
            if table.item_spec(cb.mons[idx].item).is_quick_claw
                && (non_idle || ((mask & (1 << slot)) != 0 && cb.mons[idx].hp() > 0))
            {
                live_qc |= 1u8 << s;
                break;
            }
        }
    }
    if live_qc == 0 {
        return 0;
    }
    if table.ruleset.is_castle() {
        cb.quick_claw_roll & live_qc
    } else {
        cb.quick_claw_roll & 1
    }
}

/// Compact 16-HP-bucket state hash for zero-alloc DUCT tree indexing.
/// Includes every action-legality bit, turn counter, items, timers, Protect stall counter,
/// weather/screen/Safeguard durations, Future Sight, Castle overlay flags, and observable Quick Claw.
#[inline]
fn compact_duct_key(cb: &CompactBattle, table: &MatchTable, _depth: u8) -> u64 {
    let any_qc = compact_world_has_qc(cb, table);
    let qc = if _depth == 0 && cb.sides[0].request == 1 && cb.sides[1].request == 1 {
        0
    } else {
        compact_observable_qc_bits(cb, table, any_qc)
    };
    compact_duct_key_seeded_qc(cb, table, 0x517C_C1B7_2722_0A95, qc)
}

#[inline]
fn compact_duct_key_seeded_qc(
    cb: &CompactBattle,
    table: &MatchTable,
    seed: u64,
    qc_bits: u8,
) -> u64 {
    let hdr = (cb.turn as u64)
        | ((cb.weather_turns as u64) << 16)
        | ((cb.turn_extra_flags as u64) << 24)
        | ((qc_bits as u64) << 32)
        | ((cb.future_sight_dmg[0] as u64) << 40)
        | (((cb.future_sight_dmg[1] as u64) & 0xFF) << 56);
    let mut h = mix64(seed, hdr);
    let s0 = &cb.sides[0];
    let s1 = &cb.sides[1];
    let side_word = ((s0.active_idx as u64) << 0)
        | ((s0.alive_mask as u64) << 3)
        | ((s0.request as u64) << 6)
        | (((s0.side_flags & 0x7F) as u64) << 8)
        | (((s0.reflect_turns.min(7)) as u64) << 15)
        | (((s0.lightscreen_turns.min(7)) as u64) << 18)
        | (((s0.safeguard_turns.min(7)) as u64) << 21)
        | ((s1.active_idx as u64) << 24)
        | ((s1.alive_mask as u64) << 27)
        | ((s1.request as u64) << 30)
        | (((s1.side_flags & 0x7F) as u64) << 32)
        | (((s1.reflect_turns.min(7)) as u64) << 39)
        | (((s1.lightscreen_turns.min(7)) as u64) << 42)
        | (((s1.safeguard_turns.min(7)) as u64) << 45)
        | (if cb.weather_turns > 0 {
            (cb.weather_kind as u64) << 48
        } else {
            0
        })
        | ((cb.bp_switch_flags as u64) << 50)
        | (((cb.future_sight_turns[0].min(3)) as u64) << 58)
        | (((cb.future_sight_turns[1].min(3)) as u64) << 60);
    h = mix64(h, side_word);

    for side in 0..2usize {
        let act = cb.sides[side].active_idx as usize;
        let mask = cb.sides[side].alive_mask;
        for slot in 0..3usize {
            if (mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = side * 3 + slot;
            let mon = &cb.mons[idx];
            let maxhp = table.mons[idx].maxhp.max(1) as i32;
            let hp = mon.hp().max(0) as i32;
            // Floor HP bucketing (parity with Battle::search_key / CompactBattle::search_key):
            // isolates `hp == maxhp` (100% HP where Rest fails!) in bucket 16 vs 94-99.9% in bucket 15.
            let hp_b = if hp == 0 {
                0u64
            } else {
                ((hp * 16) / maxhp) as u64
            };
            let st = mon.status_raw() as u64;
            let st_c = if st == Status::Slp as u64 {
                mon.status_counter().min(5) as u64
            } else if is_toxic_residual(table.ruleset, mon.status_raw(), mon.status_counter()) {
                (mon.status_counter() & 0x1F) as u64
            } else {
                0
            };
            let pp_exact = u32::from_le_bytes(mon.pp) as u64;
            let mut mword = ((idx as u64) << 0) | (hp_b << 4) | (st << 9) | (st_c << 12);
            if slot == act {
                let sub_b = if mon.sub_hp() <= 0 {
                    0u64
                } else {
                    (((mon.sub_hp() as i32 * 16) / maxhp).max(1).min(15)) as u64
                };
                let perish = mon.perish_turns() as u64;
                let ptrap = mon.partially_trapped() as u64;
                let vol = (mon.vol_flags
                    & (vol_bit::SUBSTITUTE
                        | vol_bit::CONFUSION
                        | vol_bit::ENCORE
                        | vol_bit::DISABLE
                        | vol_bit::TRAPPED_OR_LOCKED
                        | vol_bit::LEECH_SEED
                        | vol_bit::PAR_SPEED_DROP
                        | vol_bit::BRN_ATK_DROP
                        | vol_bit::CURSE
                        | vol_bit::NIGHTMARE
                        | vol_bit::FORESIGHT
                        | vol_bit::DESTINY_BOND)) as u64;
                // Keep FULL `slot_refs` (including `last_move_slot` for Encore/Disable!).
                let slot_ctrl = mon.slot_refs as u64;
                let lock_ctrl = mon.rollout_rage as u64;
                mword |= (sub_b << 17)
                    | (perish << 21)
                    | (ptrap << 24)
                    | (vol << 26)
                    | (slot_ctrl << 42)
                    | (lock_ctrl << 50);
                h = mix64(h, mword);
                let mut bword = ((mon.ext_flags as u64) << 32) | ((mon.item as u64) << 48);
                for (bi, &bv) in mon.boosts.iter().enumerate() {
                    bword |= (((bv + 6).clamp(0, 12) as u64) & 0xF) << (bi * 4);
                }
                let tword = (mon.timers[0] as u64)
                    | ((mon.timers[1] as u64) << 8)
                    | ((mon.timers[2] as u64) << 16)
                    | ((mon.timers[3] as u64) << 24)
                    | ((mon.ext_timers[0] as u64) << 32)
                    | ((mon.ext_timers[1] as u64) << 40)
                    | ((mon.ext_timers[2] as u64) << 48)
                    | ((mon.ext_timers[3] as u64) << 56);
                h = mix64(h, bword);
                h = mix64(h, tword);
                h = mix64(
                    h,
                    pp_exact
                        | ((cb.speed_order as u64) << 32)
                        | ((cb.hurt_cat_this_turn[side] as u64) << 40)
                        | (((cb.hurt_damage_this_turn[side] / 16) as u64) << 48),
                );
            } else {
                mword |= ((mon.item as u64) << 17) | (pp_exact << 25);
                h = mix64(h, mword);
            }
        }
    }
    h
}

#[inline]
fn eval_duct_leaf(
    cb: &CompactBattle,
    table: &MatchTable,
    mode: TempoLeafMode,
    cache: Option<&TempoMatchCache>,
) -> f64 {
    eval_duct_leaf_temp(cb, table, mode, cache, 1.0)
}

#[inline]
fn eval_duct_leaf_temp(
    cb: &CompactBattle,
    table: &MatchTable,
    mode: TempoLeafMode,
    cache: Option<&TempoMatchCache>,
    leaf_temp: f64,
) -> f64 {
    if cb.outcome != 0 {
        return match cb.outcome {
            1 => 1.0,
            2 => 0.0,
            _ => 0.5,
        };
    }
    if let Some(v) = compact_turn_limit_outcome(cb, table) {
        return v;
    }
    let raw = match mode {
        TempoLeafMode::TempoFull => {
            evaluate_compact_tempo_variant(cb, table, TempoVariant::Full).win_prob
        }
        TempoLeafMode::TempoRaw => {
            evaluate_compact_tempo_variant(cb, table, TempoVariant::Raw).win_prob
        }
        TempoLeafMode::FastGolf => evaluate_fast_golf(cb, table, cache).win_prob,
        TempoLeafMode::HybridGolf => {
            return if (leaf_temp - 1.0).abs() <= 1e-9 {
                evaluate_hybrid_golf(cb, table, cache)
            } else {
                evaluate_hybrid_golf_temp(cb, table, cache, leaf_temp)
            };
        }
        TempoLeafMode::Stage3Log => {
            return evaluate_stage3_log(cb, table, cache, leaf_temp);
        }
        TempoLeafMode::Stage3Mlp => {
            return evaluate_stage3_mlp(cb, table, cache, leaf_temp);
        }
        TempoLeafMode::AzValue => {
            return crate::compact_az::evaluate_az_value(cb, table, cache, leaf_temp);
        }
        TempoLeafMode::CfValue => {
            return crate::compact_az::evaluate_cf_value(cb, table, cache, leaf_temp);
        }
        TempoLeafMode::EvalStatic => {
            return compact_eval01(cb, table, &EvalWeights::default());
        }
        TempoLeafMode::EvalBlindspot => {
            return compact_eval_leaf_blindspot(cb, table, &EvalWeights::default());
        }
    };
    // Keep non-terminal heuristic Golf leaves strictly away from exact 0/1 terminal thresholds
    // so `backed_up_value` and `EndgameRootStats::is_win/is_loss` never misclassify heuristic leaves.
    raw.clamp(0.000_1, 0.999_9)
}

/// Fast `CompactBattle`-native ε-greedy (`eps=0.20`) heavy rollout action picker for controlled
/// rollout-vs-zero-rollout ablations (`rollout_turns = 0, 2, 4, 8`), aligned with `mcts.rs`
/// (`linear::compact_rollout_linear_pick`, `relay::rollout_relay_pick`, `best_midturn_bp_switch`,
/// Perish trap/escape).
#[inline]
#[allow(clippy::too_many_arguments)]
fn compact_rollout_pick_fast(
    cb: &CompactBattle,
    table: &MatchTable,
    weights: &EvalWeights,
    side: usize,
    cs: &[u8],
    turn_offset: u8,
    pivot_token: &mut [u8; 2],
    rng: &mut SplitMix64,
) -> u8 {
    if cs.len() == 1 {
        return cs[0];
    }
    let att_slot = cb.sides[side].active_idx as usize;
    let att_idx = side * 3 + att_slot;
    let def_side = 1 - side;
    let def_slot = cb.sides[def_side].active_idx as usize;
    let last_mon = cb.sides[side].alive_mask.count_ones() == 1;
    let suicidal = |act: u8| -> bool {
        act < 4 && last_mon && table.move_specs[att_idx][act as usize].is_selfdestruct
    };
    let rand_safe = |rng: &mut SplitMix64| -> u8 {
        if last_mon {
            let mut safe = [0u8; 7];
            let mut ns = 0usize;
            for &c in cs {
                if !suicidal(c) {
                    safe[ns] = c;
                    ns += 1;
                }
            }
            if ns > 0 {
                return safe[rng.below(ns)];
            }
        }
        cs[rng.below(cs.len())]
    };

    // Perish Song last-turn escape (matching mcts.rs: `hp_frac * (1.0 - inc)`).
    if cb.sides[side].request == 1
        && (cb.mons[att_idx].vol_flags & vol_bit::PERISH_SONG) != 0
        && cb.mons[att_idx].perish_turns() == 1
    {
        let foe_alive = (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
            && cb.mons[def_side * 3 + def_slot].hp() > 0;
        let mut best_sw = None;
        let mut best_score = f64::NEG_INFINITY;
        for &c in cs {
            if (4..7).contains(&c) {
                let sl = (c - 4) as usize;
                let idx = side * 3 + sl;
                let inc = if foe_alive {
                    cb.best_hit_fraction(table, def_side, def_slot, side, sl)
                } else {
                    0.0
                };
                let sc = (cb.mons[idx].hp() as f64)
                    / (table.mons[idx].maxhp.max(1) as f64)
                    * (1.0 - inc).max(0.0);
                if best_sw.is_none() || sc >= best_score {
                    best_score = sc;
                    best_sw = Some(c);
                }
            }
        }
        if let Some(sw) = best_sw {
            return sw;
        }
    }

    if rng.next_f64() < 0.20 {
        return rand_safe(rng);
    }

    if turn_offset < 5
        && (cb.sides[0].side_flags & nc2000_engine::compact::SIDE0_DISABLE_BP_FIX) == 0
    {
        if let crate::relay::RelayRolloutDecision::Chosen(act) = crate::relay::rollout_relay_pick(
            cb,
            table,
            weights,
            side,
            cs,
            turn_offset,
            pivot_token,
            false,
            rng,
        ) {
            return act;
        }
    }

    if cb.sides[side].request == 1
        && (cb.sides[side].alive_mask & (1 << att_slot)) != 0
        && (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
    {
        if let Some(la_pick) = crate::linear::compact_rollout_linear_pick(cb, table, side, cs) {
            return la_pick;
        }
        let mut best_c = None;
        let mut best_s = 0.0f64;
        for &c in cs {
            if c < 4 && !suicidal(c) {
                let (score, _) = cb.expected_hit_fraction_with_acc(
                    table, side, att_slot, def_side, def_slot, c as usize,
                );
                if score > best_s {
                    best_s = score;
                    best_c = Some(c);
                }
            }
        }
        if let Some(c) = best_c {
            return c;
        }
        return rand_safe(rng);
    }

    // Forced or mid-turn Baton Pass switch: reuse `relay::best_midturn_bp_switch` when passing alive.
    let foe_alive = (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
        && cb.mons[def_side * 3 + def_slot].hp() > 0;
    let passing_alive = cb.sides[side].request == 2
        && att_slot < 3
        && cb.mons[att_idx].hp() > 0
        && foe_alive
        && (cb.mons[att_idx].boosts[0] > 0
            || cb.mons[att_idx].boosts[1] > 0
            || cb.mons[att_idx].boosts[2] > 0
            || cb.mons[att_idx].boosts[3] > 0
            || cb.mons[att_idx].boosts[4] > 0
            || cb.mons[att_idx].boosts[6] > 0
            || (cb.mons[att_idx].vol_flags & vol_bit::SUBSTITUTE) != 0);
    if passing_alive {
        if let Some(sw) = crate::relay::best_midturn_bp_switch(cb, table, weights, side, cs) {
            return sw;
        }
    }
    let mut best_sw = cs[0];
    let mut best_s = f64::NEG_INFINITY;
    for &c in cs {
        if (4..7).contains(&c) {
            let sl = (c - 4) as usize;
            let sc = if foe_alive {
                cb.best_hit_fraction(table, side, sl, def_side, def_slot).min(1.5)
                    - cb.best_hit_fraction(table, def_side, def_slot, side, sl).min(1.5)
            } else {
                (cb.mons[side * 3 + sl].hp() as f64) / (table.mons[side * 3 + sl].maxhp.max(1) as f64)
            };
            if sc > best_s {
                best_s = sc;
                best_sw = c;
            }
        }
    }
    best_sw
}

/// Optionally roll out `cb` for up to `rollout_turns` turns on `CompactBattle`, then score with `mode`.
#[inline]
fn eval_duct_leaf_with_rollout(
    cb: &mut CompactBattle,
    table: &MatchTable,
    mode: TempoLeafMode,
    rollout_turns: u16,
    cache: Option<&TempoMatchCache>,
    rng: &mut SplitMix64,
) -> f64 {
    if rollout_turns == 0 || cb.outcome != 0 {
        return eval_duct_leaf(cb, table, mode, cache);
    }
    let start_turn = cb.turn;
    let raw_cutoff = cb.turn.saturating_add(rollout_turns.saturating_sub(1));
    let cutoff = table
        .ruleset
        .turn_limit()
        .map_or(raw_cutoff, |lim| raw_cutoff.min(lim));
    let weights = EvalWeights::default();
    let mut pivot_token = [1u8; 2];
    let mut safety = 0u8;
    while cb.outcome == 0 && cb.turn <= cutoff && safety < 32 {
        safety += 1;
        if let Some(v) = compact_turn_limit_outcome(cb, table) {
            return v;
        }
        let turn_offset = cb.turn.saturating_sub(start_turn).min(255) as u8;
        let mut picks = [7u8; 2];
        for s in 0..2usize {
            if cb.sides[s].request == 0 {
                continue;
            }
            let mut cs = [0u8; 7];
            let n = cb.legal_actions(table, s, &mut cs);
            if n > 0 {
                picks[s] = compact_rollout_pick_fast(
                    cb,
                    table,
                    &weights,
                    s,
                    &cs[..n],
                    turn_offset,
                    &mut pivot_token,
                    rng,
                );
            }
        }
        cb.prng_seed = rng.next();
        crate::mcts::apply_compact_actions_with_bp_target(cb, table, picks);
    }
    eval_duct_leaf(cb, table, mode, cache)
}

// =========================================================================
// FastGolf (v2): Cached Closed-Form Turn-Economy Evaluator (~1-2 us/leaf)
// =========================================================================

/// Wall-cap for `FastGolf` in multi-mon states (1v1 uses `FAST_WALL_CAP_1V1`).
const FAST_WALL_CAP: f64 = 7.5;
const FAST_WALL_CAP_1V1: f64 = 12.0;

/// Gen 2 exact stat-stage multipliers for stages `-6..=+6` (`idx = stage + 6`).
const GEN2_STAGE_MULT: [f32; 13] = [
    0.25, 0.28, 0.33, 0.40, 0.50, 0.66, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
];

/// Gen 2 exact accuracy / evasion stage multipliers for stages `-6..=+6` (`3/(3-s)` / `(3+s)/3`).
const GEN2_ACC_STAGE_MULT: [f32; 13] = [
    0.33, 0.36, 0.43, 0.50, 0.60, 0.75, 1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0,
];

const MOVE_FLAG_USABLE: u16 = 1 << 0;
const MOVE_FLAG_PHYSICAL: u16 = 1 << 1;
const MOVE_FLAG_FIXED: u16 = 1 << 2;
const MOVE_FLAG_WATER: u16 = 1 << 3;
const MOVE_FLAG_FIRE: u16 = 1 << 4;
const MOVE_FLAG_SOLARBEAM: u16 = 1 << 5;
const MOVE_FLAG_SELFDESTRUCT: u16 = 1 << 6;
pub(crate) const MOVE_FLAG_RECHARGE: u16 = 1 << 7;
const MOVE_FLAG_ALWAYS_HIT: u16 = 1 << 8;

#[derive(Clone, Copy, Default)]
pub(crate) struct CachedMoveDmg {
    pub(crate) mean_dmg: f32,
    pub(crate) min_norm: f32,
    pub(crate) max_norm: f32,
    pub(crate) max_crit: f32,
    pub(crate) p_hit: f32,
    pub(crate) p_crit: f32,
    pub(crate) flags: u16,
}

#[derive(Clone, Copy)]
pub(crate) struct StaticMonCache {
    pub(crate) half_heal_slot: u8,
    pub(crate) half_heal_is_weather: bool,
    pub(crate) rest_slot: u8,
    pub(crate) sleep_talk_slot: u8,
    pub(crate) phaze_slot: u8,
    pub(crate) haze_slot: u8,
    pub(crate) bp_slot: u8,
    pub(crate) perish_slot: u8,
    pub(crate) dbond_slot: u8,
    pub(crate) encore_slot: u8,
    pub(crate) meanlook_slot: u8,
    pub(crate) pursuit_slot: u8,
    pub(crate) healbell_slot: u8,
    pub(crate) setup_mask: u8,
    pub(crate) sleep_mask: u8,
    pub(crate) toxic_mask: u8,
    pub(crate) twave_mask: u8,
    pub(crate) st_attack_rate: f64,
    pub(crate) inv_maxhp: f64,
    pub(crate) q16_frac: f64,
    pub(crate) q8_frac: f64,
    pub(crate) sand_immune: bool,
    pub(crate) init_boost_type: u8,
    pub(crate) init_castle_boost_type: u8,
    pub(crate) init_stat_kind: nc2000_engine::compact::ItemStatKind,
    pub(crate) init_brightpowder: bool,
    pub(crate) init_scope_lens: bool,
}

impl Default for StaticMonCache {
    fn default() -> Self {
        Self {
            half_heal_slot: 0xFF,
            half_heal_is_weather: false,
            rest_slot: 0xFF,
            sleep_talk_slot: 0xFF,
            phaze_slot: 0xFF,
            haze_slot: 0xFF,
            bp_slot: 0xFF,
            perish_slot: 0xFF,
            dbond_slot: 0xFF,
            encore_slot: 0xFF,
            meanlook_slot: 0xFF,
            pursuit_slot: 0xFF,
            healbell_slot: 0xFF,
            setup_mask: 0,
            sleep_mask: 0,
            toxic_mask: 0,
            twave_mask: 0,
            st_attack_rate: 0.0,
            inv_maxhp: 1.0,
            q16_frac: 0.0625,
            q8_frac: 0.125,
            sand_immune: false,
            init_boost_type: 0xFF,
            init_castle_boost_type: 0xFF,
            init_stat_kind: nc2000_engine::compact::ItemStatKind::None,
            init_brightpowder: false,
            init_scope_lens: false,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct CachedPairEntry {
    pub(crate) moves: [CachedMoveDmg; 4],
    pub(crate) best_norm_slot: u8,
    pub(crate) boom_slot: u8,
    pub(crate) any_dynamic_bp: bool,
    pub(crate) has_weather_acc: bool,
    pub(crate) has_drain: bool,
    pub(crate) has_recharge: bool,
}

impl Default for CachedPairEntry {
    fn default() -> Self {
        Self {
            moves: [CachedMoveDmg::default(); 4],
            best_norm_slot: 0xFF,
            boom_slot: 0xFF,
            any_dynamic_bp: false,
            has_weather_acc: false,
            has_drain: false,
            has_recharge: false,
        }
    }
}

/// Per-search precomputed static damage & trait table across all 6 mons and 18 directed pairs.
/// Built once at search root (~2 us) and reused across thousands of MCTS leaves with O(1) lookup,
/// including O(1) analytical Gen 2 boost/burn/screen/weather scaling (zero `DmgKernel1D` on leaves).
/// Tracks `party_roster` so switched `Battle` clones (which permute `side.party`) remap O(1) safely.
#[derive(Clone)]
pub struct TempoMatchCache {
    party_roster: [[u8; 3]; 2],
    mons: [StaticMonCache; 6],
    entries: [[[CachedPairEntry; 3]; 3]; 2],
}

#[derive(Clone, Copy)]
pub(crate) struct CacheView<'a> {
    cache: &'a TempoMatchCache,
    map: [[usize; 3]; 2],
}

impl<'a> CacheView<'a> {
    #[inline(always)]
    pub(crate) fn mon(&self, side: usize, slot: usize) -> &'a StaticMonCache {
        &self.cache.mons[side * 3 + self.map[side][slot]]
    }

    #[inline(always)]
    pub(crate) fn pair(&self, att_side: usize, att_slot: usize, def_slot: usize) -> &'a CachedPairEntry {
        let ca = self.map[att_side][att_slot];
        let cd = self.map[1 - att_side][def_slot];
        &self.cache.entries[att_side][ca][cd]
    }
}

impl TempoMatchCache {
    #[inline]
    pub(crate) fn view_for(&self, cb: &CompactBattle) -> Option<CacheView<'_>> {
        if cb.sides[0].party_roster == self.party_roster[0]
            && cb.sides[1].party_roster == self.party_roster[1]
        {
            return Some(CacheView { cache: self, map: [[0, 1, 2], [0, 1, 2]] });
        }
        let mut map = [[0usize; 3]; 2];
        for s in 0..2 {
            for slot in 0..3 {
                let r = cb.sides[s].party_roster[slot];
                let pos = (0..3).find(|&k| self.party_roster[s][k] == r)?;
                map[s][slot] = pos;
            }
        }
        Some(CacheView { cache: self, map })
    }

    pub fn build(base_cb: &CompactBattle, table: &MatchTable) -> Self {
        let mut clean_cb = *base_cb;
        clean_cb.weather_turns = 0;
        for s in 0..2 {
            clean_cb.sides[s].reflect_turns = 0;
            clean_cb.sides[s].lightscreen_turns = 0;
        }
        let mut mons = [StaticMonCache::default(); 6];
        for i in 0..6 {
            clean_cb.mons[i].boosts = [0; 7];
            clean_cb.mons[i].vol_flags = 0;
            clean_cb.mons[i].ext_flags = 0;
            clean_cb.mons[i].ext_timers = [0; 4];
            clean_cb.mons[i].rollout_rage = 0;
            clean_cb.mons[i].status_packed = 0;
            let maxhp = table.mons[i].maxhp.max(1);
            clean_cb.mons[i].hp_pool[0] = maxhp;
            clean_cb.mons[i].hp_pool[1] = 0;
            let n_moves = table.mons[i].num_moves as usize;
            let mut half_heal_slot = 0xFFu8;
            let mut half_heal_is_weather = false;
            let mut rest_slot = 0xFFu8;
            let mut sleep_talk_slot = 0xFFu8;
            let mut phaze_slot = 0xFFu8;
            let mut haze_slot = 0xFFu8;
            let mut bp_slot = 0xFFu8;
            let mut perish_slot = 0xFFu8;
            let mut dbond_slot = 0xFFu8;
            let mut encore_slot = 0xFFu8;
            let mut meanlook_slot = 0xFFu8;
            let mut pursuit_slot = 0xFFu8;
            let mut healbell_slot = 0xFFu8;
            let mut setup_mask = 0u8;
            let mut sleep_mask = 0u8;
            let mut toxic_mask = 0u8;
            let mut twave_mask = 0u8;
            let mut usable_st_attacks = 0usize;
            for m in 0..n_moves {
                clean_cb.mons[i].pp[m] = table.mons[i].max_pp[m].max(1);
                let ms = &table.move_specs[i][m];
                if ms.is_half_heal {
                    half_heal_slot = m as u8;
                    half_heal_is_weather = false;
                } else if ms.special_kind == SpecialMoveKind::WeatherHeal && half_heal_slot == 0xFF
                {
                    half_heal_slot = m as u8;
                    half_heal_is_weather = true;
                }
                if ms.is_rest {
                    rest_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::Phaze && phaze_slot == 0xFF {
                    phaze_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::Haze && haze_slot == 0xFF {
                    haze_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::BatonPass && bp_slot == 0xFF {
                    bp_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::PerishSong && perish_slot == 0xFF {
                    perish_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::DestinyBond && dbond_slot == 0xFF {
                    dbond_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::Encore && encore_slot == 0xFF {
                    encore_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::MeanLook && meanlook_slot == 0xFF {
                    meanlook_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::Pursuit && pursuit_slot == 0xFF {
                    pursuit_slot = m as u8;
                }
                if ms.special_kind == SpecialMoveKind::HealBell && healbell_slot == 0xFF {
                    healbell_slot = m as u8;
                }
                let is_ghost = (table.mons[i].type_mask & (1u32 << (table.ghost_type & 31))) != 0;
                if (ms.special_kind == SpecialMoveKind::Curse && !is_ghost)
                    || ms.special_kind == SpecialMoveKind::BellyDrum
                    || (ms.is_status
                        && ms.boosts_self
                        && ms.boosts.iter().any(|&(st, d)| st < 5 && d > 0))
                {
                    setup_mask |= 1u8 << m;
                }
                if ms.primary_status == Status::Slp as u8 {
                    sleep_mask |= 1u8 << m;
                } else if ms.primary_status == Status::Tox as u8 {
                    toxic_mask |= 1u8 << m;
                } else if ms.primary_status == Status::Par as u8 {
                    twave_mask |= 1u8 << m;
                }
                if ms.special_kind == SpecialMoveKind::SleepTalk {
                    sleep_talk_slot = m as u8;
                } else if !ms.is_status && !ms.is_selfdestruct && ms.can_sleep_talk {
                    usable_st_attacks += 1;
                }
            }
            let inv_maxhp = 1.0 / (maxhp as f64);
            let ispec = table.item_spec(base_cb.mons[i].item);
            mons[i] = StaticMonCache {
                half_heal_slot,
                half_heal_is_weather,
                rest_slot,
                sleep_talk_slot,
                phaze_slot,
                haze_slot,
                bp_slot,
                perish_slot,
                dbond_slot,
                encore_slot,
                meanlook_slot,
                pursuit_slot,
                healbell_slot,
                setup_mask,
                sleep_mask,
                toxic_mask,
                twave_mask,
                st_attack_rate: if sleep_talk_slot != 0xFF && n_moves > 1 {
                    usable_st_attacks as f64 / (n_moves - 1) as f64
                } else {
                    0.0
                },
                inv_maxhp,
                q16_frac: (table.mons[i].quantum_1_16 as f64) * inv_maxhp,
                q8_frac: (table.mons[i].quantum_1_8 as f64) * inv_maxhp,
                sand_immune: (table.mons[i].type_mask & table.sandstorm_immune_mask) != 0,
                init_boost_type: ispec.boost_type,
                init_castle_boost_type: ispec.castle_boost_type,
                init_stat_kind: ispec.stat_kind,
                init_brightpowder: ispec.is_brightpowder,
                init_scope_lens: ispec.is_scope_lens,
            };
        }

        let mut entries = [[[CachedPairEntry::default(); 3]; 3]; 2];
        for att_side in 0..2 {
            let def_side = 1 - att_side;
            for att_slot in 0..3 {
                let att_idx = att_side * 3 + att_slot;
                let n_moves = table.mons[att_idx].num_moves as usize;
                for def_slot in 0..3 {
                    let def_idx = def_side * 3 + def_slot;
                    let mut entry = CachedPairEntry::default();
                    let mut best_norm_dmg = 0.0f32;
                    let mut best_boom_dmg = 0.0f32;
                    for m_i in 0..n_moves {
                        let ms = &table.move_specs[att_idx][m_i];
                        if ms.is_status {
                            continue;
                        }
                        if ms.drain_half {
                            entry.has_drain = true;
                        }
                        if ms.is_recharge {
                            entry.has_recharge = true;
                        }
                        if matches!(
                            ms.special_kind,
                            SpecialMoveKind::FlailReversal
                                | SpecialMoveKind::SuperFang
                                | SpecialMoveKind::PainSplit
                                | SpecialMoveKind::Counter
                                | SpecialMoveKind::MirrorCoat
                                | SpecialMoveKind::Rollout
                                | SpecialMoveKind::FuryCutter
                                | SpecialMoveKind::Rage
                                | SpecialMoveKind::DreamEater
                                | SpecialMoveKind::BeatUp
                                | SpecialMoveKind::Ohko
                        ) {
                            entry.any_dynamic_bp = true;
                        }
                        if ms.special_kind == SpecialMoveKind::Thunder {
                            entry.has_weather_acc = true;
                        }
                        let kern = DmgKernel1D::for_move(
                            &clean_cb,
                            table,
                            att_side,
                            att_slot,
                            def_side,
                            def_slot,
                            m_i,
                        );
                        if kern.max_dmg <= 0 {
                            continue;
                        }
                        let sm = summarize_kernel(&kern, ms);
                        entry.moves[m_i] = sm;
                        if (sm.flags & MOVE_FLAG_SELFDESTRUCT) != 0 {
                            if sm.mean_dmg > best_boom_dmg {
                                best_boom_dmg = sm.mean_dmg;
                                entry.boom_slot = m_i as u8;
                            }
                        } else {
                            let eff = if (sm.flags & MOVE_FLAG_RECHARGE) != 0 {
                                sm.mean_dmg * 0.55
                            } else {
                                sm.mean_dmg
                            };
                            if eff > best_norm_dmg {
                                best_norm_dmg = eff;
                                entry.best_norm_slot = m_i as u8;
                            }
                        }
                    }
                    entries[att_side][att_slot][def_slot] = entry;
                }
            }
        }
        Self {
            party_roster: [base_cb.sides[0].party_roster, base_cb.sides[1].party_roster],
            mons,
            entries,
        }
    }
}

/// Per-world 6×6 precomputed static damage & trait table for Team Preview.
/// Built once per opponent candidate world (~8 μs via <= 4 synthetic 3v3 blocks) and
/// sliced O(1) (~40 ns) into a 3×3 `TempoMatchCache` for every `(pick0, pick1)` pair
/// across both Stage-1 screening (up to 3,600 cells) and Stage-2 persistent PUCT trees.
#[derive(Clone)]
pub struct PreviewTempoCache {
    mons: [[StaticMonCache; 6]; 2],
    entries: [[[CachedPairEntry; 6]; 6]; 2],
}

impl PreviewTempoCache {
    pub fn build_for_preview_roster(
        roster: &nc2000_engine::compact::CompactPreviewRoster,
        party_lens: [usize; 2],
    ) -> Self {
        let mut mons = [[StaticMonCache::default(); 6]; 2];
        let mut entries = [[[CachedPairEntry::default(); 6]; 6]; 2];

        let make_blocks = |plen: usize| -> ([[u8; 3]; 2], usize) {
            let p = plen.clamp(1, 6) as u8;
            let b0 = [1, if p >= 2 { 2 } else { 0 }, if p >= 3 { 3 } else { 0 }];
            if p <= 3 {
                ([b0, b0], 1)
            } else {
                let b1 = [
                    4,
                    if p >= 5 { 5 } else { 1 },
                    if p >= 6 { 6 } else if p == 5 { 1 } else { 2 },
                ];
                ([b0, b1], 2)
            }
        };

        let (blk0, n0) = make_blocks(party_lens[0]);
        let (blk1, n1) = make_blocks(party_lens[1]);
        let picked0 = party_lens[0].min(3);
        let picked1 = party_lens[1].min(3);

        for b0 in 0..n0 {
            for b1 in 0..n1 {
                let (table, cb) = roster.build_turn1([blk0[b0], blk1[b1]], 0);
                let sub = TempoMatchCache::build(&cb, &table);
                let r0 = cb.sides[0].party_roster;
                let r1 = cb.sides[1].party_roster;
                for a in 0..picked0 {
                    let ra = (r0[a] as usize).min(5);
                    mons[0][ra] = sub.mons[a];
                    for d in 0..picked1 {
                        let rd = (r1[d] as usize).min(5);
                        entries[0][ra][rd] = sub.entries[0][a][d];
                    }
                }
                for a in 0..picked1 {
                    let ra = (r1[a] as usize).min(5);
                    mons[1][ra] = sub.mons[3 + a];
                    for d in 0..picked0 {
                        let rd = (r0[d] as usize).min(5);
                        entries[1][ra][rd] = sub.entries[1][a][d];
                    }
                }
            }
        }
        Self { mons, entries }
    }

    /// O(1) slice into a 3×3 `TempoMatchCache` matching `cb.sides[*].party_roster`
    /// (bit-for-bit identical to `TempoMatchCache::build(&cb, &table)`).
    #[inline]
    pub fn slice_for(&self, cb: &CompactBattle) -> TempoMatchCache {
        let mut mons = [StaticMonCache::default(); 6];
        let mut entries = [[[CachedPairEntry::default(); 3]; 3]; 2];
        let r0 = cb.sides[0].party_roster;
        let r1 = cb.sides[1].party_roster;
        for a in 0..3usize {
            let ra = (r0[a] as usize).min(5);
            mons[a] = self.mons[0][ra];
            for d in 0..3usize {
                let rd = (r1[d] as usize).min(5);
                entries[0][a][d] = self.entries[0][ra][rd];
            }
        }
        for a in 0..3usize {
            let ra = (r1[a] as usize).min(5);
            mons[3 + a] = self.mons[1][ra];
            for d in 0..3usize {
                let rd = (r0[d] as usize).min(5);
                entries[1][a][d] = self.entries[1][ra][rd];
            }
        }
        TempoMatchCache {
            party_roster: [r0, r1],
            mons,
            entries,
        }
    }
}

#[inline]
fn summarize_kernel(kern: &DmgKernel1D, ms: &CompactMoveSpec) -> CachedMoveDmg {
    let mut p_hit = 0.0f32;
    let mut min_norm = f32::INFINITY;
    let mut max_norm = 0.0f32;
    let mut p_crit = 0.0f32;
    for k in 0..kern.len {
        let (dmg, prob) = kern.pts[k];
        if dmg > 0 {
            p_hit += prob;
            if prob >= 0.06 {
                let df = dmg as f32;
                if df < min_norm {
                    min_norm = df;
                }
                if df > max_norm {
                    max_norm = df;
                }
            } else {
                p_crit += prob;
            }
        }
    }
    if !min_norm.is_finite() {
        min_norm = kern.mean_dmg;
        max_norm = kern.max_dmg as f32;
    }
    let crit_frac = if p_hit > 1e-4 { (p_crit / p_hit).clamp(0.0, 1.0) } else { 0.0 };
    let mut flags = MOVE_FLAG_USABLE;
    if ms.is_physical {
        flags |= MOVE_FLAG_PHYSICAL;
    }
    if ms.fixed_damage != 0 {
        flags |= MOVE_FLAG_FIXED;
    }
    if ms.is_water {
        flags |= MOVE_FLAG_WATER;
    }
    if ms.is_fire {
        flags |= MOVE_FLAG_FIRE;
    }
    if ms.is_solarbeam {
        flags |= MOVE_FLAG_SOLARBEAM;
    }
    if ms.is_selfdestruct {
        flags |= MOVE_FLAG_SELFDESTRUCT;
    }
    if ms.is_recharge {
        flags |= MOVE_FLAG_RECHARGE;
    }
    if ms.always_hit {
        flags |= MOVE_FLAG_ALWAYS_HIT;
    }
    CachedMoveDmg {
        mean_dmg: kern.mean_dmg,
        min_norm,
        max_norm,
        max_crit: (kern.max_dmg as f32).max(max_norm),
        p_hit: p_hit.clamp(0.0, 1.0),
        p_crit: crit_frac,
        flags,
    }
}

/// Apply Gen 2 boosts, Burn, duration-weighted Screens, Weather, and Acc/Eva to a cached clean `CachedMoveDmg` in O(1).
#[inline(always)]
pub(crate) fn scale_cached_move(
    md: CachedMoveDmg,
    att_boosts: &[i8; 7],
    def_boosts: &[i8; 7],
    att_burned: bool,
    def_reflect_turns: u8,
    def_lightscreen_turns: u8,
    weather_kind: CompactWeather,
    weather_active: bool,
) -> CachedMoveDmg {
    let mut out = md;
    let acc_b = att_boosts[5];
    let eva_b = def_boosts[6];
    if (md.flags & MOVE_FLAG_ALWAYS_HIT) == 0 && (acc_b != 0 || eva_b != 0) {
        let ratio = GEN2_ACC_STAGE_MULT[(acc_b + 6).clamp(0, 12) as usize]
            / GEN2_ACC_STAGE_MULT[(eva_b + 6).clamp(0, 12) as usize];
        let old_h = out.p_hit.max(1e-4);
        let new_h = (out.p_hit * ratio).clamp(0.05, 1.0);
        out.mean_dmg *= new_h / old_h;
        out.p_hit = new_h;
    }
    if (md.flags & MOVE_FLAG_FIXED) != 0 {
        return out;
    }
    let is_phys = (md.flags & MOVE_FLAG_PHYSICAL) != 0;
    let (atk_b, def_b) = if is_phys {
        (att_boosts[0], def_boosts[1])
    } else {
        (att_boosts[2], def_boosts[3])
    };
    let stat_mult = if atk_b != 0 || def_b != 0 {
        GEN2_STAGE_MULT[(atk_b + 6).clamp(0, 12) as usize]
            / GEN2_STAGE_MULT[(def_b + 6).clamp(0, 12) as usize]
    } else {
        1.0
    };
    let burn_mult = if is_phys && att_burned { 0.5 } else { 1.0 };
    // Duration-weighted Reflect/Light Screen (never treat a 1-turn-remaining screen as permanent 0.5x!).
    let rem_screen = if is_phys { def_reflect_turns } else { def_lightscreen_turns };
    let screen_mult = if rem_screen > 0 {
        let w = ((rem_screen.min(5) as f32) / 4.5).clamp(0.25, 0.85);
        1.0 - 0.5 * w
    } else {
        1.0
    };
    let weather_mult = if weather_active {
        match weather_kind {
            CompactWeather::Rain => {
                if (md.flags & MOVE_FLAG_WATER) != 0 {
                    1.5
                } else if (md.flags & (MOVE_FLAG_FIRE | MOVE_FLAG_SOLARBEAM)) != 0 {
                    0.5
                } else {
                    1.0
                }
            }
            CompactWeather::Sun => {
                if (md.flags & MOVE_FLAG_FIRE) != 0 {
                    1.5
                } else if (md.flags & MOVE_FLAG_WATER) != 0 {
                    0.5
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    } else {
        1.0
    };

    let norm_mult = stat_mult * burn_mult * screen_mult * weather_mult;
    // Gen 2 crit: ignores boosts/burn/screens iff atk_boost <= def_boost.
    let crit_mult = if atk_b <= def_b { weather_mult } else { stat_mult * weather_mult };
    if (norm_mult - 1.0).abs() > 1e-4 || (crit_mult - 1.0).abs() > 1e-4 {
        let eff_mean_mult = (1.0 - out.p_crit) * norm_mult + out.p_crit * crit_mult;
        out.mean_dmg *= eff_mean_mult;
        out.min_norm *= norm_mult;
        out.max_norm *= norm_mult;
        out.max_crit = (out.max_crit * crit_mult).max(out.max_norm);
    }
    out
}

#[inline(always)]
pub(crate) fn fast_ko_prob(m: &CachedMoveDmg, hp: f32) -> f64 {
    if hp <= 0.0 {
        return 1.0;
    }
    if hp > m.max_crit {
        return 0.0;
    }
    let p_norm = if hp <= m.min_norm {
        1.0f32
    } else if hp >= m.max_norm {
        0.0f32
    } else {
        (m.max_norm - hp) / (m.max_norm - m.min_norm).max(1.0)
    };
    let min_crit = m.max_crit * 0.85;
    let p_c = if hp <= min_crit {
        1.0f32
    } else {
        ((m.max_crit - hp) / (m.max_crit - min_crit).max(1.0)).clamp(0.0, 1.0)
    };
    (m.p_hit * ((1.0 - m.p_crit) * p_norm + m.p_crit * p_c)).clamp(0.0, 1.0) as f64
}

#[derive(Clone, Copy, Default)]
pub(crate) struct FastPair {
    pub(crate) raw_ttk: f64,
    pub(crate) eff_ttk: f64,
    pub(crate) net_dmg: f64,
    pub(crate) raw_dmg: f64,
    pub(crate) max_norm_frac: f64,
    pub(crate) ohko_prob: f64,
    pub(crate) boom_ko_prob: f64,
    pub(crate) boom_dmg_frac: f64,
    pub(crate) heal_duty: f64,
    pub(crate) def_sustain: bool,
    pub(crate) speed_cmp: i8,
    pub(crate) perish_capped: bool,
    pub(crate) best_pp: u8,
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn compute_fast_pair(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    entry_hps: &[i16; 6],
    speeds: &[i32; 6],
    sleep_eff_delays: &[f64; 6],
    ignore_att_boosts: bool,
    ignore_def_boosts: bool,
    is_1v1: bool,
    cache: Option<&CacheView<'_>>,
) -> FastPair {
    let wall_cap = if is_1v1 { FAST_WALL_CAP_1V1 } else { FAST_WALL_CAP };
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let att_mon = &cb.mons[att_idx];
    let def_mon = &cb.mons[def_idx];
    let att_spec = &table.mons[att_idx];
    let def_spec = &table.mons[def_idx];

    let s_att = if ignore_att_boosts && att_mon.boosts[4] != 0 {
        effective_speed_unboosted(cb, table, att_side, att_slot)
    } else {
        speeds[att_idx]
    };
    let s_def = if ignore_def_boosts && def_mon.boosts[4] != 0 {
        effective_speed_unboosted(cb, table, def_side, def_slot)
    } else {
        speeds[def_idx]
    };
    let speed_cmp = if s_att > s_def { 1 } else if s_att < s_def { -1 } else { 0 };

    let start_hp = entry_hps[def_idx];
    if start_hp <= 0 || att_mon.hp() <= 0 {
        return FastPair {
            raw_ttk: if start_hp <= 0 { 0.0 } else { wall_cap },
            eff_ttk: if start_hp <= 0 { 0.0 } else { wall_cap },
            speed_cmp,
            ..FastPair::default()
        };
    }

    let (inv_maxhp, q16_frac, q8_frac, sand_immune) = if let Some(c) = cache {
        let m = c.mon(def_side, def_slot);
        (m.inv_maxhp, m.q16_frac, m.q8_frac, m.sand_immune)
    } else {
        let inv = 1.0 / (def_spec.maxhp.max(1) as f64);
        (
            inv,
            (def_spec.quantum_1_16 as f64) * inv,
            (def_spec.quantum_1_8 as f64) * inv,
            (def_spec.type_mask & table.sandstorm_immune_mask) != 0,
        )
    };
    let def_item = table.item_spec(def_mon.item);

    // Passive residuals on defender (positive = heals defender).
    let mut res_frac = 0.0f64;
    if def_item.is_leftovers {
        res_frac += q16_frac;
    }
    let def_status = def_mon.status_raw();
    let def_cnt = def_mon.status_counter();
    if is_toxic_residual(table.ruleset, def_status, def_cnt) {
        let step = ((def_cnt & 0x0F) as f64 + 2.0).min(15.0);
        res_frac -= step * q16_frac;
    } else if def_status == Status::Brn as u8 || def_status == Status::Psn as u8 {
        res_frac -= q8_frac;
    }
    if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        res_frac -= q8_frac;
    }
    if (def_mon.vol_flags & vol_bit::CURSE) != 0 {
        res_frac -= 0.25;
    }
    if (def_mon.vol_flags & vol_bit::NIGHTMARE) != 0
        && (def_status == Status::Slp as u8
            || (table.ruleset.is_castle()
                && (def_mon.ext_flags & ext_vol_bit::MIRACLE_NIGHTMARE) != 0))
    {
        res_frac -= 0.25;
    }
    if def_mon.partially_trapped() {
        res_frac -= q16_frac;
    }
    if cb.weather_turns > 0 && cb.weather_kind == CompactWeather::Sandstorm && !sand_immune {
        res_frac -= q8_frac;
    }

    let zero_boosts = [0i8; 7];
    let eff_att_boosts = if ignore_att_boosts { &zero_boosts } else { &att_mon.boosts };
    let eff_def_boosts = if ignore_def_boosts { &zero_boosts } else { &def_mon.boosts };
    let att_status = att_mon.status_raw();
    let att_is_active = cb.sides[att_side].active_idx as usize == att_slot;
    // In PS, benched burned mons have `vol_flags == 0` and re-attach `BRN_ATK_DROP` upon switching in.
    let att_burned = att_status == Status::Brn as u8
        && (table.ruleset.is_castle()
            || !att_is_active
            || (att_mon.vol_flags & vol_bit::BRN_ATK_DROP) != 0);
    let def_reflect_turns = cb.sides[def_side].reflect_turns;
    let def_lightscreen_turns = cb.sides[def_side].lightscreen_turns;
    let weather_active = cb.weather_turns > 0
        && matches!(cb.weather_kind, CompactWeather::Rain | CompactWeather::Sun);

    let needs_scale = *eff_att_boosts != [0; 7]
        || *eff_def_boosts != [0; 7]
        || att_burned
        || def_reflect_turns > 0
        || def_lightscreen_turns > 0
        || weather_active;

    // Cache stays valid even across boosts, burn, screens, weather, and consumed healing berries!
    // Falls back to exact `DmgKernel1D` on dynamic-BP moves, Foresight, Thunder-in-weather,
    // extended volatile damage modifiers (Focus Energy / Lock-On / Minimize / Transform / Dig / Fly),
    // and Castle `ovfl=1` boosted stat wrapping.
    const ATT_CACHE_BREAK_EXT: u16 = ext_vol_bit::FOCUS_ENERGY
        | ext_vol_bit::LOCK_ON
        | ext_vol_bit::DEFENSE_CURL
        | ext_vol_bit::TRANSFORMED
        | ext_vol_bit::RAGE;
    const DEF_CACHE_BREAK_EXT: u16 =
        ext_vol_bit::MINIMIZE | ext_vol_bit::TRANSFORMED | ext_vol_bit::DIGGING | ext_vol_bit::FLYING;
    let cached_entry = cache.and_then(|c| {
        let e = c.pair(att_side, att_slot, def_slot);
        let def_has_sub = cb.sides[def_side].active_idx as usize == def_slot && def_mon.sub_hp() > 0;
        if e.any_dynamic_bp
            || (e.has_weather_acc && weather_active)
            || (e.has_drain && def_has_sub)
            || (def_mon.vol_flags & vol_bit::FORESIGHT) != 0
            || (att_mon.ext_flags & ATT_CACHE_BREAK_EXT) != 0
            || (def_mon.ext_flags & DEF_CACHE_BREAK_EXT) != 0
            || (table.ruleset.is_castle()
                && table.ruleset.overflow_mode()
                && (*eff_att_boosts != [0; 7] || *eff_def_boosts != [0; 7]))
        {
            return None;
        }
        let att_is = table.item_spec(att_mon.item);
        let cm_a = c.mon(att_side, att_slot);
        let cm_d = c.mon(def_side, def_slot);
        let item_ok = att_is.boost_type == cm_a.init_boost_type
            && att_is.castle_boost_type == cm_a.init_castle_boost_type
            && att_is.stat_kind == cm_a.init_stat_kind
            && att_is.is_scope_lens == cm_a.init_scope_lens
            && def_item.stat_kind == cm_d.init_stat_kind
            && def_item.is_brightpowder == cm_d.init_brightpowder;
        if item_ok { Some(e) } else { None }
    });

    let start_hp_f = start_hp as f32;
    // Effective Turn-1 KO HP threshold accounting for net end-of-turn chip (Poison/Burn/Toxic/Leech/Sand).
    let ko_hp_f = if res_frac < -1e-4 {
        (start_hp_f + (res_frac * (def_spec.maxhp.max(1) as f64)) as f32).max(1.0)
    } else {
        start_hp_f
    };
    let mut best_mean = 0.0f64;
    let mut best_max_norm = 0.0f64;
    let mut best_ohko = 0.0f64;
    let mut best_pp = 0u8;
    let mut boom_ko = 0.0f64;
    let mut boom_dmg = 0.0f64;

    // Fast path: O(1) precomputed best-move lookup for clean unboosted states; scaled states scan
    // all <=4 cached moves (O(1) arithmetic, zero DmgKernel1D) so weather/fixed-dmg/Hyper-Beam OHKOs
    // are never missed.
    let mut used_fast_slot = false;
    if let Some(ce) = cached_entry {
        if !needs_scale && !ce.has_recharge {
            let bs = ce.best_norm_slot as usize;
            let b_ok = bs == 0xFF || (att_mon.pp[bs] > 0 && !att_mon.is_slot_disabled(bs));
            if b_ok {
                used_fast_slot = true;
                if bs < 4 {
                    let md = ce.moves[bs];
                    let p_ko = fast_ko_prob(&md, ko_hp_f);
                    let mut d_mean = (md.mean_dmg as f64) * inv_maxhp;
                    if (md.flags & MOVE_FLAG_RECHARGE) != 0 && p_ko < 0.85 {
                        d_mean *= 0.55;
                    }
                    best_mean = d_mean;
                    best_max_norm = (md.max_norm as f64) * inv_maxhp;
                    best_ohko = p_ko;
                    best_pp = att_mon.pp[bs];
                }
                let bms = ce.boom_slot as usize;
                if bms < 4 && att_mon.pp[bms] > 0 && !att_mon.is_slot_disabled(bms) {
                    let md = ce.moves[bms];
                    boom_ko = fast_ko_prob(&md, ko_hp_f);
                    boom_dmg = (md.mean_dmg as f64) * inv_maxhp;
                }
            }
        }
    }

    if !used_fast_slot {
        let mut best_score = 0.0f64;
        let n_moves = att_spec.num_moves as usize;
        for m_i in 0..n_moves {
            if att_mon.pp[m_i] == 0 || att_mon.is_slot_disabled(m_i) {
                continue;
            }
            let ms = &table.move_specs[att_idx][m_i];
            if ms.is_status {
                continue;
            }
            let md = if let Some(ce) = cached_entry {
                let raw = ce.moves[m_i];
                if (raw.flags & MOVE_FLAG_USABLE) == 0 {
                    continue;
                }
                if needs_scale {
                    scale_cached_move(
                        raw,
                        eff_att_boosts,
                        eff_def_boosts,
                        att_burned,
                        def_reflect_turns,
                        def_lightscreen_turns,
                        cb.weather_kind,
                        weather_active,
                    )
                } else {
                    raw
                }
            } else {
                let mut tmp_cb;
                let use_cb = if ignore_att_boosts
                    || ignore_def_boosts
                    || def_reflect_turns > 0
                    || def_lightscreen_turns > 0
                {
                    tmp_cb = *cb;
                    if ignore_att_boosts {
                        tmp_cb.mons[att_idx].boosts = [0; 7];
                    }
                    if ignore_def_boosts {
                        tmp_cb.mons[def_idx].boosts = [0; 7];
                    }
                    // Compute unscreened kernel, then apply duration-weighted screen multiplier consistently.
                    tmp_cb.sides[def_side].reflect_turns = 0;
                    tmp_cb.sides[def_side].lightscreen_turns = 0;
                    &tmp_cb
                } else {
                    cb
                };
                let kern = DmgKernel1D::for_move(
                    use_cb, table, att_side, att_slot, def_side, def_slot, m_i,
                );
                if kern.max_dmg <= 0 {
                    continue;
                }
                let mut sm = summarize_kernel(&kern, ms);
                if (ms.is_physical && def_reflect_turns > 0)
                    || (!ms.is_physical && def_lightscreen_turns > 0)
                {
                    let rem_screen =
                        if ms.is_physical { def_reflect_turns } else { def_lightscreen_turns };
                    let w = ((rem_screen.min(5) as f32) / 4.5).clamp(0.25, 0.85);
                    let m_sc = 1.0 - 0.5 * w;
                    let eff_m = (1.0 - sm.p_crit) * m_sc + sm.p_crit;
                    sm.mean_dmg *= eff_m;
                    sm.min_norm *= m_sc;
                    sm.max_norm *= m_sc;
                }
                sm
            };
            if (md.flags & MOVE_FLAG_USABLE) == 0 {
                continue;
            }
            let p_ko = fast_ko_prob(&md, ko_hp_f);
            if (md.flags & MOVE_FLAG_SELFDESTRUCT) != 0 {
                if p_ko > boom_ko {
                    boom_ko = p_ko;
                }
                let bd = (md.mean_dmg as f64) * inv_maxhp;
                if bd > boom_dmg {
                    boom_dmg = bd;
                }
                continue;
            }
            let mut d_mean = (md.mean_dmg as f64) * inv_maxhp;
            if (md.flags & MOVE_FLAG_RECHARGE) != 0 && p_ko < 0.85 {
                d_mean *= 0.55;
            }
            let score = d_mean + 0.15 * p_ko;
            if score > best_score {
                best_score = score;
                best_mean = d_mean;
                best_max_norm = (md.max_norm as f64) * inv_maxhp;
                best_ohko = p_ko;
                best_pp = att_mon.pp[m_i];
            }
        }
    }

    // One-time berry effective HP bonus if not OHKO'd.
    let berry_hp: f64 = if def_item.is_goldberry {
        30.0
    } else if def_item.is_berry_juice {
        20.0
    } else if def_item.is_berry {
        10.0
    } else {
        0.0
    };
    let hp_frac = if berry_hp > 0.0 && best_ohko < 0.5 {
        (((start_hp as f64) + berry_hp) * inv_maxhp).min(1.0)
    } else {
        ((start_hp as f64) * inv_maxhp).clamp(0.0, 1.0)
    };

    // Attacker action efficiency & initial turn delays.
    let mut act_rate = 1.0f64;
    if att_status == Status::Par as u8 {
        act_rate *= 0.75;
    } else if att_status == Status::Frz as u8 {
        act_rate *= 0.15;
    }
    if (att_mon.vol_flags & vol_bit::CONFUSION) != 0 {
        act_rate *= 0.75;
    }
    let mut delay = sleep_eff_delays[att_idx];
    if att_mon.must_recharge() {
        delay += 1.0;
    }

    // Substitute shield on active defender.
    let sub_hp = if cb.sides[def_side].active_idx as usize == def_slot {
        def_mon.sub_hp()
    } else {
        0
    };
    let sub_turns = if sub_hp > 0 {
        let sub_frac = (sub_hp as f64) * inv_maxhp;
        (sub_frac / (best_mean * act_rate).max(0.05)).clamp(1.0, 3.0)
    } else {
        0.0
    };
    let ohko_prob = if sub_hp > 0 || delay >= 1.0 {
        0.0
    } else {
        (best_ohko * act_rate).clamp(0.0, 1.0)
    };

    let d_net = best_mean * act_rate - res_frac;

    // Closed-form blended continuous + discrete TTK:
    // Every living Pokemon costs >= ~0.85..1.0 turns while keeping smooth 1%-HP gradient.
    let mut raw_ttk = if d_net <= 0.015 {
        (wall_cap - 1.5 * (1.0 - hp_frac)).max(1.0)
    } else {
        let t_cont = hp_frac / d_net;
        let mut t_hit = if t_cont < 1.0 {
            0.85 + 0.15 * t_cont
        } else {
            0.55 * t_cont + 0.45 * t_cont.ceil()
        };
        if ohko_prob > 0.01 && t_cont <= 1.35 {
            t_hit = ohko_prob * (0.85 + 0.15 * t_cont.min(1.0))
                + (1.0 - ohko_prob) * t_hit.max(1.65);
        }
        (t_hit + delay + sub_turns).min(wall_cap)
    };

    // Perish Song trapped hard countdown override.
    let mut perish_capped = false;
    if cb.sides[def_side].active_idx as usize == def_slot {
        let pt = def_mon.perish_turns();
        let trapped = def_mon.is_trapped() || cb.sides[def_side].alive_mask.count_ones() <= 1;
        if pt > 0 && trapped && (pt as f64) < raw_ttk {
            raw_ttk = pt as f64;
            perish_capped = true;
        }
    }

    let (eff_boom_ko, eff_boom_dmg) = if delay >= 1.0 {
        (0.0, boom_dmg * act_rate * 0.25)
    } else {
        ((boom_ko * act_rate).clamp(0.0, 1.0), boom_dmg * act_rate)
    };

    FastPair {
        raw_ttk,
        eff_ttk: raw_ttk,
        net_dmg: d_net,
        raw_dmg: best_mean,
        max_norm_frac: best_max_norm,
        ohko_prob,
        boom_ko_prob: eff_boom_ko,
        boom_dmg_frac: eff_boom_dmg,
        heal_duty: 0.0,
        def_sustain: false,
        speed_cmp,
        perish_capped,
        best_pp,
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn apply_fast_recovery(
    cb: &CompactBattle,
    table: &MatchTable,
    att_side: usize,
    att_slot: usize,
    def_side: usize,
    def_slot: usize,
    a2d: &mut FastPair,
    d2a: &mut FastPair,
    entry_hps: &[i16; 6],
    sleep_turns: &[f64; 6],
    is_1v1: bool,
    cache: Option<&CacheView<'_>>,
) {
    if a2d.perish_capped {
        return;
    }
    let def_idx = def_side * 3 + def_slot;
    let def_mon = &cb.mons[def_idx];
    let def_spec = &table.mons[def_idx];
    // Fast exit: if defender has neither Recover nor Rest in its moveset, skip immediately!
    let (half_heal, half_pp, rest_pp, st_rate_cached, inv_maxhp, q16_frac, q8_frac, sand_immune) =
        if let Some(c) = cache {
            let m = c.mon(def_side, def_slot);
            if m.half_heal_slot == 0xFF && m.rest_slot == 0xFF {
                return;
            }
            let (hh, hpp) = if m.half_heal_slot < 4 {
                let s = m.half_heal_slot as usize;
                if def_mon.pp[s] > 0 && !def_mon.is_slot_disabled(s) {
                    let frac = if m.half_heal_is_weather && cb.weather_turns > 0 {
                        match cb.weather_kind {
                            CompactWeather::Sun => 1.0,
                            CompactWeather::Rain | CompactWeather::Sandstorm => 0.25,
                            CompactWeather::None => 0.5,
                        }
                    } else {
                        0.5
                    };
                    (frac, def_mon.pp[s])
                } else {
                    (0.0, 0)
                }
            } else {
                (0.0, 0)
            };
            let rpp = if m.rest_slot < 4 {
                let s = m.rest_slot as usize;
                if def_mon.pp[s] > 0 && !def_mon.is_slot_disabled(s) {
                    def_mon.pp[s]
                } else {
                    0
                }
            } else {
                0
            };
            let str_c = if m.sleep_talk_slot < 4 {
                let s = m.sleep_talk_slot as usize;
                if def_mon.pp[s] > 0 && !def_mon.is_slot_disabled(s) {
                    m.st_attack_rate
                } else {
                    0.0
                }
            } else {
                0.0
            };
            (hh, hpp, rpp, Some(str_c), m.inv_maxhp, m.q16_frac, m.q8_frac, m.sand_immune)
        } else {
            let mut hh = 0.0f64;
            let mut hpp = 0u8;
            let mut rpp = 0u8;
            for m_i in 0..def_spec.num_moves as usize {
                if def_mon.pp[m_i] == 0 || def_mon.is_slot_disabled(m_i) {
                    continue;
                }
                let ms = &table.move_specs[def_idx][m_i];
                if let Some(hf) = instant_heal_frac(cb, ms) {
                    if hf > hh {
                        hh = hf;
                        hpp = def_mon.pp[m_i];
                    }
                }
                if ms.is_rest {
                    rpp = rpp.max(def_mon.pp[m_i]);
                }
            }
            let inv = 1.0 / (def_spec.maxhp.max(1) as f64);
            (
                hh,
                hpp,
                rpp,
                None,
                inv,
                (def_spec.quantum_1_16 as f64) * inv,
                (def_spec.quantum_1_8 as f64) * inv,
                (def_spec.type_mask & table.sandstorm_immune_mask) != 0,
            )
        };
    if half_pp == 0 && rest_pp == 0 {
        return;
    }

    let def_hp = entry_hps[def_idx];
    if def_hp <= 0 || cb.mons[att_side * 3 + att_slot].hp() <= 0 {
        return;
    }
    let def_status = def_mon.status_raw();
    if def_status == Status::Frz as u8 {
        return;
    }

    let raw_hp_frac = (def_hp as f64) * inv_maxhp;
    let slp_t = sleep_turns[def_idx];
    let def_faster = d2a.speed_cmp > 0;

    // Sleeping wall: does NOT lose Rest/Recover permanently; it takes `slp_t` free hits before waking!
    // If attacker KOs it before wake-up, recovery fails; otherwise evaluate recovery from `wake_hp`.
    let (hp_frac, p_surv) = if slp_t > 0.0 {
        let wake_hp = raw_hp_frac - slp_t * a2d.net_dmg.max(0.0);
        if wake_hp <= 0.04 || a2d.raw_ttk <= slp_t + 0.35 {
            return;
        }
        let ps = if def_faster {
            1.0
        } else if a2d.max_norm_frac > 0.01 && wake_hp < a2d.max_norm_frac {
            (wake_hp / a2d.max_norm_frac).clamp(0.0, 1.0)
        } else {
            (1.0 - a2d.ohko_prob).clamp(0.0, 1.0)
        };
        (wake_hp.clamp(0.04, 1.0), ps)
    } else {
        let ps = if def_faster { 1.0 } else { (1.0 - a2d.ohko_prob).clamp(0.0, 1.0) };
        (raw_hp_frac, ps)
    };
    if p_surv < 0.05 {
        return;
    }

    // Paralyzed healer: 25% full-para reliably breaks Recover/Rest loops.
    let is_par = def_status == Status::Par as u8;
    let par_heal_mult = if is_par { 0.75 } else { 1.0 };
    let par_wall_disc = if is_par { 0.58 } else { 1.0 };

    let wall_cap = if is_1v1 { FAST_WALL_CAP_1V1 } else { FAST_WALL_CAP };
    // Lone-wall pressure: when defender is the LAST alive mon facing 2+ attackers, it cannot pivot
    // out of dual-attacker pressure, status, crits, or 2x PP pool.
    let lone_wall_disc = if !is_1v1
        && cb.sides[def_side].alive_mask.count_ones() == 1
        && cb.sides[att_side].alive_mask.count_ones() >= 2
    {
        0.50
    } else {
        1.0
    };
    // Discount wall bonus if attacker carries a massive Explosion (>65% dmg) that breaks the wall.
    let boom_disc = if !is_1v1 && a2d.boom_dmg_frac > 0.65 {
        (1.0 - 0.55 * ((a2d.boom_dmg_frac - 0.65) / 0.35).clamp(0.0, 1.0)).clamp(0.45, 1.0)
    } else {
        1.0
    };

    let mut best_opt: Option<(f64, f64, f64)> = None;

    // 50% Recover / Soft-Boiled / Milk Drink / Weather Heal
    if half_heal > 0.0 && half_pp > 0 {
        let eff_heal = half_heal * par_heal_mult;
        let safe_after = if def_faster {
            (hp_frac + half_heal).min(1.0)
        } else {
            (1.0 - a2d.max_norm_frac + half_heal).min(1.0)
        };
        if a2d.max_norm_frac < safe_after - 0.02 && a2d.net_dmg <= eff_heal + 0.015 {
            let duty = (a2d.net_dmg.max(0.0) / eff_heal).clamp(0.0, 1.0);
            let pp_scale = ((half_pp as f64) / 16.0).clamp(0.35, 1.0);
            let wall_full = if is_1v1 {
                (a2d.raw_ttk + (half_pp as f64) * par_wall_disc / duty.max(0.20)).min(wall_cap)
            } else {
                let extra = 3.5
                    * (1.0 - 0.45 * duty)
                    * pp_scale
                    * boom_disc
                    * par_wall_disc
                    * lone_wall_disc;
                (a2d.raw_ttk + extra).min(wall_cap)
            };
            let heal_debt = (1.0 - hp_frac) / eff_heal;
            let t_att = p_surv * (wall_full - 0.9 * heal_debt).max(a2d.raw_ttk)
                + (1.0 - p_surv) * a2d.raw_ttk;

            let eff_att_turns =
                (d2a.raw_ttk - if def_faster { 1.0 } else { 0.0 } - slp_t).max(0.0);
            let dmg_over_kill = eff_att_turns * a2d.net_dmg.max(0.0);
            let excess_hp = (dmg_over_kill - (hp_frac - 0.15).max(0.0)).max(0.0);
            let net_heal = eff_heal - a2d.net_dmg.max(0.0);
            let t_def = if excess_hp <= 1e-4 {
                d2a.raw_ttk
            } else if is_1v1 {
                if net_heal > 0.025 {
                    (d2a.raw_ttk + excess_hp / net_heal).min(wall_cap)
                } else {
                    (a2d.best_pp.max(1) as f64).min(wall_cap)
                }
            } else {
                // Multi-mon: healing turns spent to offset HP saved by `def_sustain` (never blows up to 7.5t!).
                let heal_turns = (excess_hp / eff_heal.max(0.25)).min(1.6);
                (d2a.raw_ttk + heal_turns).min(wall_cap)
            };
            best_opt = Some((t_att, t_def, duty));
        }
    }

    // Rest (+ Sleep Talk / Mint Berry)
    if rest_pp > 0 {
        let def_item = table.item_spec(def_mon.item);
        let has_wake_berry = def_item.is_mintberry || def_item.is_miracleberry;
        let mut rest_res = 0.0;
        if def_item.is_leftovers {
            rest_res += q16_frac;
        }
        if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
            rest_res -= q8_frac;
        }
        if (def_mon.vol_flags & vol_bit::CURSE) != 0 {
            rest_res -= 0.25;
        }
        if def_mon.partially_trapped() {
            rest_res -= q16_frac;
        }
        if cb.weather_turns > 0 && cb.weather_kind == CompactWeather::Sandstorm && !sand_immune {
            rest_res -= q8_frac;
        }
        let att_mult = if cb.mons[att_side * 3 + att_slot].status_raw() == Status::Par as u8 {
            0.75
        } else {
            1.0
        };
        let rest_net = a2d.raw_dmg * att_mult - rest_res;
        let sleep_hits = if def_faster { 2.0 } else { 2.8 };
        let hp_after_wake = 1.0 - sleep_hits * rest_net.max(0.0);
        let h_rest = a2d.max_norm_frac.clamp(0.18, 0.48);

        if hp_after_wake > h_rest && a2d.max_norm_frac < 0.48 {
            let st_rate =
                st_rate_cached.unwrap_or_else(|| sleep_talk_attack_rate(cb, table, def_side, def_slot));
            let non_act = (3.0 - 2.0 * st_rate).max(1.0);
            let k_awake = if rest_net > 0.01 {
                ((hp_after_wake - h_rest) / rest_net).max(0.0)
            } else {
                12.0
            };
            let duty = (non_act / (3.0 + k_awake)).clamp(0.0, 1.0);
            let pp_scale = ((rest_pp as f64) / 10.0).clamp(0.4, 1.0);
            let wall_full = if is_1v1 {
                (a2d.raw_ttk + (rest_pp as f64) * 3.0 * par_wall_disc).min(wall_cap)
            } else {
                let extra = 3.2
                    * (1.0 - 0.40 * duty)
                    * pp_scale
                    * boom_disc
                    * par_wall_disc
                    * lone_wall_disc;
                (a2d.raw_ttk + extra).min(wall_cap)
            };
            let heal_debt = (1.0 - hp_frac) * non_act;
            let t_att = p_surv * (wall_full - 0.75 * heal_debt).max(a2d.raw_ttk)
                + (1.0 - p_surv) * a2d.raw_ttk;

            let eff_att_turns =
                (d2a.raw_ttk - if def_faster { 1.0 } else { 0.0 } - slp_t).max(0.0);
            let dmg_over_kill = eff_att_turns * rest_net.max(0.0);
            let excess_hp = (dmg_over_kill - (hp_frac - 0.15).max(0.0)).max(0.0);
            let net_cycle = (hp_after_wake - h_rest).max(0.0);
            let t_def = if excess_hp <= 1e-4 {
                d2a.raw_ttk
            } else if is_1v1 {
                if net_cycle > 0.03 {
                    (d2a.raw_ttk + non_act * (excess_hp / net_cycle)).min(wall_cap)
                } else {
                    (a2d.best_pp.max(1) as f64).min(wall_cap)
                }
            } else {
                let heal_turns = (non_act * (excess_hp / 0.65)).min(2.0);
                (d2a.raw_ttk + heal_turns).min(wall_cap)
            };
            let better = best_opt.map_or(true, |(pa, pd, _)| (t_att - t_def) > (pa - pd));
            if better {
                best_opt = Some((t_att, t_def, duty));
            }
        } else if has_wake_berry && a2d.max_norm_frac < 0.88 && hp_frac < 0.85 {
            let extra = ((1.0 - hp_frac.min(0.6)) / rest_net.max(0.15)).min(2.5);
            let t_att = (a2d.raw_ttk + p_surv * extra).min(wall_cap);
            let t_def = (d2a.raw_ttk + 1.0).min(wall_cap);
            let better = best_opt.map_or(true, |(pa, pd, _)| (t_att - t_def) > (pa - pd));
            if better {
                best_opt = Some((t_att, t_def, 0.25));
            }
        }
    }

    if let Some((t_att, t_def, duty)) = best_opt {
        if t_att > a2d.raw_ttk + 0.05 {
            a2d.eff_ttk = a2d.eff_ttk.max(t_att);
            a2d.heal_duty = duty;
            a2d.def_sustain = true;
            let safe_turns = a2d.raw_ttk - if def_faster { 0.75 } else { 1.5 };
            if d2a.raw_ttk > safe_turns {
                d2a.eff_ttk = d2a.eff_ttk.max(t_def);
            }
        }
    }
}

#[inline]
fn blend_fast_pair(a: FastPair, b: FastPair, w_b: f64) -> FastPair {
    let w_a = 1.0 - w_b;
    FastPair {
        raw_ttk: w_a * a.raw_ttk + w_b * b.raw_ttk,
        eff_ttk: w_a * a.eff_ttk + w_b * b.eff_ttk,
        net_dmg: w_a * a.net_dmg + w_b * b.net_dmg,
        raw_dmg: w_a * a.raw_dmg + w_b * b.raw_dmg,
        max_norm_frac: w_a * a.max_norm_frac + w_b * b.max_norm_frac,
        ohko_prob: w_a * a.ohko_prob + w_b * b.ohko_prob,
        boom_ko_prob: w_a * a.boom_ko_prob + w_b * b.boom_ko_prob,
        boom_dmg_frac: w_a * a.boom_dmg_frac + w_b * b.boom_dmg_frac,
        heal_duty: w_a * a.heal_duty + w_b * b.heal_duty,
        def_sustain: a.def_sustain || (w_b >= 0.5 && b.def_sustain),
        speed_cmp: if w_b >= 0.5 { b.speed_cmp } else { a.speed_cmp },
        perish_capped: a.perish_capped,
        best_pp: a.best_pp,
    }
}

/// Zero-alloc stack 3v3 Golf score solver with factored branch-and-bound pruning, strictly-monotonic
/// exhausted-team spill accounting, last-enemy Explosion (+ 2-step teammate cleanup), and
/// strictly-monotonic log-sum-exp team-depth backup turns (multi-mon only; 1v1 unchanged).
#[inline]
#[allow(clippy::too_many_arguments)]
fn solve_fast_team_turns(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    ttk: &[[[FastPair; 3]; 3]; 2],
    unb_active: Option<&[FastPair; 3]>,
    entry_hps: &[i16; 6],
    sleep_eff_delays: &[f64; 6],
    cache: Option<&CacheView<'_>>,
) -> f64 {
    let opp = 1 - side;
    let mut allies = [0usize; 3];
    let mut n_allies = 0usize;
    let mut enemies = [0usize; 3];
    let mut n_enemies = 0usize;

    for slot in 0..3 {
        if (cb.sides[side].alive_mask & (1 << slot)) != 0 && cb.mons[side * 3 + slot].hp() > 0 {
            allies[n_allies] = slot;
            n_allies += 1;
        }
        if (cb.sides[opp].alive_mask & (1 << slot)) != 0 && cb.mons[opp * 3 + slot].hp() > 0 {
            enemies[n_enemies] = slot;
            n_enemies += 1;
        }
    }
    if n_enemies == 0 {
        return 0.0;
    }
    if n_allies == 0 {
        return FAST_WALL_CAP * (n_enemies as f64);
    }

    let act_ally = cb.sides[side].active_idx as usize;
    let act_enemy = cb.sides[opp].active_idx as usize;
    let ally_active_alive =
        cb.mons[side * 3 + act_ally].hp() > 0 && cb.sides[side].request != 2;
    let enemy_active_alive =
        cb.mons[opp * 3 + act_enemy].hp() > 0 && cb.sides[opp].request != 2;

    let active_speed_adj = if ally_active_alive && enemy_active_alive {
        match ttk[side][act_ally][act_enemy].speed_cmp {
            1 => -0.25,
            -1 => 0.25,
            _ => 0.0,
        }
    } else {
        0.0
    };

    if n_allies == 1 && n_enemies == 1 {
        let p = ttk[side][allies[0]][enemies[0]];
        let p_opp = ttk[opp][enemies[0]][allies[0]];
        if p.eff_ttk >= FAST_WALL_CAP_1V1 - 0.5 && p_opp.eff_ttk >= FAST_WALL_CAP_1V1 - 0.5 {
            // Bounded pure turn-currency PP-stall tiebreaker ONLY in mutual 1v1 wall wars.
            let e_idx = opp * 3 + enemies[0];
            let mut opp_pp = 0u32;
            for m in 0..table.mons[e_idx].num_moves as usize {
                opp_pp += cb.mons[e_idx].pp[m] as u32;
            }
            return FAST_WALL_CAP_1V1 + 0.02 * (opp_pp as f64).min(80.0) + active_speed_adj;
        }
        return (p.eff_ttk + active_speed_adj).max(0.25);
    }

    let mut base_hp = [0.0f64; 3];
    let mut opp_kill_ally = [2.5f64; 3];
    let mut ally_perish = [false; 3];
    let mut ally_dbond = [false; 3];
    let mut enemy_dbond = [false; 3];
    let has_usable_special = |s: usize, slot: usize, kind: SpecialMoveKind| -> bool {
        let idx = s * 3 + slot;
        let mon = &cb.mons[idx];
        if mon.hp() <= 0
            || mon.status_raw() == Status::Frz as u8
            || sleep_eff_delays[idx] >= 1.0
        {
            return false;
        }
        if let Some(c) = cache {
            let cm = c.mon(s, slot);
            let ms = match kind {
                SpecialMoveKind::PerishSong => cm.perish_slot as usize,
                SpecialMoveKind::DestinyBond => cm.dbond_slot as usize,
                _ => 0xFF,
            };
            ms < 4 && mon.pp[ms] > 0 && !mon.is_slot_disabled(ms)
        } else {
            (0..table.mons[idx].num_moves as usize).any(|m| {
                mon.pp[m] > 0
                    && !mon.is_slot_disabled(m)
                    && table.move_specs[idx][m].special_kind == kind
            })
        }
    };
    for &a in &allies[..n_allies] {
        let idx = side * 3 + a;
        let inv = cache.map_or_else(
            || 1.0 / (table.mons[idx].maxhp.max(1) as f64),
            |c| c.mon(side, a).inv_maxhp,
        );
        base_hp[a] = ((entry_hps[idx] as f64) * inv).clamp(0.0, 1.0);
        let mut min_opp_t = FAST_WALL_CAP;
        for &e in &enemies[..n_enemies] {
            if ttk[opp][e][a].eff_ttk < min_opp_t {
                min_opp_t = ttk[opp][e][a].eff_ttk;
            }
        }
        opp_kill_ally[a] = min_opp_t.clamp(1.0, 5.5);
        if n_allies >= 2 {
            ally_perish[a] = has_usable_special(side, a, SpecialMoveKind::PerishSong);
            ally_dbond[a] = has_usable_special(side, a, SpecialMoveKind::DestinyBond);
        }
    }
    if n_enemies >= 2 {
        for &e in &enemies[..n_enemies] {
            enemy_dbond[e] = has_usable_special(opp, e, SpecialMoveKind::DestinyBond);
        }
    }

    let mut ordered_enemies = [0usize; 3];
    {
        let mut pos = 0;
        for &e in &enemies[..n_enemies] {
            if enemy_active_alive && e == act_enemy {
                ordered_enemies[pos] = e;
                pos += 1;
            }
        }
        for &e in &enemies[..n_enemies] {
            if !enemy_active_alive || e != act_enemy {
                ordered_enemies[pos] = e;
                pos += 1;
            }
        }
    }

    let opp_boom_threat_w = if n_enemies > n_allies { 0.85 } else { 0.30 };

    // Single-step transition helper: ally `a` handles `ordered_enemies[idx_e]`, mutating `hp_rem`,
    // `used_mask`, `switched_out_active`, and returning incremental turns added.
    let step_enemy = |idx_e: usize,
                      a: usize,
                      hp_rem: &mut [f64; 3],
                      used_mask: &mut u8,
                      switched_out_active: &mut bool|
     -> f64 {
        let e = ordered_enemies[idx_e];
        if a != act_ally {
            *switched_out_active = true;
        }
        let pair_ae = if a == act_ally && *switched_out_active {
            unb_active.map_or(ttk[side][a][e], |row| row[e])
        } else {
            ttk[side][a][e]
        };
        let pair_ea = ttk[opp][e][a];

        let mut add_t = 0.0f64;
        let mut entry_hit_hp = 0.0f64;
        if ally_active_alive && enemy_active_alive && e == act_enemy && a != act_ally {
            add_t += 0.65;
            entry_hit_hp = (0.5 * pair_ea.net_dmg.max(0.0)).min(0.35);
        } else if *used_mask != 0 && (*used_mask & (1 << a)) == 0 {
            add_t += 0.45;
        }
        *used_mask |= 1 << a;

        let t_need = pair_ae.eff_ttk;
        let avail = hp_rem[a];
        let is_last_enemy = idx_e + 1 == n_enemies;
        let sw_risk = if a == act_ally && ally_active_alive { 0.0 } else { 0.35 };
        let min_boom_hp = if a == act_ally && ally_active_alive {
            0.02
        } else {
            (entry_hit_hp + 0.12).max(0.15)
        };
        let has_other_alive =
            n_allies >= 2 && allies[..n_allies].iter().any(|&b| b != a && hp_rem[b] > 0.02);

        // 1. Explosion / Self-Destruct option (and last-enemy 2-step Explosion + Teammate Cleanup):
        //    Requires another living ally so user never self-loses.
        let eff_boom_ko = if pair_ae.speed_cmp < 0 {
            pair_ae.boom_ko_prob * (1.0 - pair_ea.ohko_prob).clamp(0.0, 1.0)
        } else {
            pair_ae.boom_ko_prob
        };
        if has_other_alive
            && avail > min_boom_hp
            && (eff_boom_ko > 0.5 || (is_last_enemy && pair_ae.boom_dmg_frac > 0.50))
        {
            if eff_boom_ko > 0.5 && (is_last_enemy || t_need > 2.0) {
                let sac_pen = if is_last_enemy {
                    0.0
                } else {
                    let hp_ratio = (avail / base_hp[a].max(0.05)).clamp(0.15, 1.0);
                    let opp_saved = opp_kill_ally[a] * hp_ratio;
                    let has_absorber = enemies[..n_enemies]
                        .iter()
                        .any(|&e2| e2 != e && ttk[side][a][e2].boom_dmg_frac < 0.45);
                    opp_saved + if has_absorber { 0.85 } else { 0.0 }
                };
                let boom_cost = 1.0 / eff_boom_ko + sac_pen + sw_risk;
                if boom_cost < t_need {
                    hp_rem[a] = 0.0;
                    return add_t + boom_cost;
                }
            } else if is_last_enemy && pair_ae.boom_dmg_frac > 0.50 {
                // Expected turns for Last-Enemy Explosion + Teammate Cleanup:
                // `eff_boom_ko` wins on Turn 1; `(1 - eff_boom_ko)` leaves `rem_hp` for surviving teammate `b`.
                let p_exec = if pair_ae.speed_cmp < 0 {
                    (1.0 - pair_ea.ohko_prob).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if p_exec > 0.45 {
                    let rem_hp = (1.0 - pair_ae.boom_dmg_frac * 0.94).clamp(0.04, 0.50);
                    let mut best_clean = f64::INFINITY;
                    for &b in &allies[..n_allies] {
                        if b == a || hp_rem[b] <= 0.02 {
                            continue;
                        }
                        let pair_be = ttk[side][b][e];
                        let surv_b = if pair_be.speed_cmp < 0 {
                            (1.0 - ttk[opp][e][b].ohko_prob).clamp(0.0, 1.0)
                        } else {
                            1.0
                        };
                        if surv_b < 0.25 {
                            continue;
                        }
                        let t_fin = (rem_hp * pair_be.raw_ttk).max(0.85) / surv_b;
                        if t_fin < best_clean {
                            best_clean = t_fin;
                        }
                    }
                    if best_clean.is_finite() {
                        let p_ko = eff_boom_ko.clamp(0.0, 1.0);
                        let exp_after = (1.0 - p_ko) * best_clean;
                        let combo_cost = (1.0 + exp_after) / p_exec + sw_risk;
                        if combo_cost < t_need {
                            hp_rem[a] = 0.0;
                            return add_t + combo_cost;
                        }
                    }
                }
            }
        }

        // 2. Last-enemy Perish Song / Destiny Bond checkmate (cannot switch out when `is_last_enemy`!).
        let mut best_checkmate = t_need;
        let mut checkmate_sac = false;
        if is_last_enemy && has_other_alive && avail > min_boom_hp {
            let p_act = if pair_ae.speed_cmp > 0 {
                1.0
            } else {
                (1.0 - pair_ea.ohko_prob).clamp(0.0, 1.0)
            };
            if ally_perish[a] && p_act > 0.45 {
                let perish_t = 3.5 / p_act + sw_risk;
                if perish_t < best_checkmate {
                    best_checkmate = perish_t;
                    checkmate_sac = false;
                }
            }
            if ally_dbond[a] && p_act > 0.55 {
                // Faster Destiny Bond forces 1-for-1 trade checkmate vs last enemy; slower requires surviving 1 hit.
                let db_base = if pair_ae.speed_cmp > 0 { 2.0 } else { 2.6 };
                let db_t = db_base / p_act + sw_risk;
                if db_t < best_checkmate {
                    best_checkmate = db_t;
                    checkmate_sac = true;
                }
            }
            if best_checkmate < t_need - 0.05 {
                if checkmate_sac {
                    hp_rem[a] = 0.0;
                }
                return add_t + best_checkmate;
            }
        }

        let e_slp = sleep_eff_delays[opp * 3 + e];
        let raw_hits = if pair_ae.speed_cmp > 0 {
            let base_h = (t_need - 1.0).max(0.0);
            let miss_h = (1.0 - pair_ae.ohko_prob).clamp(0.0, 1.0);
            base_h.max(miss_h)
        } else if pair_ae.speed_cmp < 0 {
            t_need.max(1.0)
        } else {
            (t_need - 0.5).max(0.5)
        };
        // Conservation of actions per turn: a sustaining enemy spends `heal_duty` of its turns
        // clicking Recover/Rest and ONLY `(1 - heal_duty)` of its turns attacking!
        let e_att_frac = if pair_ae.def_sustain {
            (1.0 - pair_ae.heal_duty).clamp(0.25, 1.0)
        } else {
            1.0
        };
        let eff_hits = ((raw_hits - e_slp).max(0.0)) * e_att_frac;
        let raw_hp_drain = eff_hits * pair_ea.net_dmg.max(0.0);
        let mut hits_taken = if pair_ea.def_sustain {
            (0.35 * raw_hp_drain).max(0.15 * pair_ea.heal_duty)
        } else {
            raw_hp_drain
        };
        // Defensive Explosion & Destiny Bond threat when enemy `e` has backup (`n_enemies >= 2`).
        if n_enemies >= 2 && e_slp < 0.5 {
            let p_e_acts = if pair_ae.speed_cmp > 0 {
                (1.0 - pair_ae.ohko_prob).clamp(0.0, 1.0)
            } else {
                1.0
            };
            if pair_ea.boom_dmg_frac > 0.55 {
                let boom_drain = opp_boom_threat_w * p_e_acts * pair_ea.boom_dmg_frac;
                if boom_drain > hits_taken {
                    hits_taken = boom_drain;
                }
            }
            if enemy_dbond[e] && (pair_ae.speed_cmp < 0 || pair_ae.ohko_prob < 0.85) {
                let db_drain = opp_boom_threat_w * p_e_acts * 0.90;
                if db_drain > hits_taken {
                    hits_taken = db_drain;
                }
            }
        }
        let fragile_bump = if !pair_ea.def_sustain
            && e_slp < 0.5
            && avail < pair_ea.raw_dmg
            && pair_ea.raw_dmg > 0.05
        {
            let p_die_t1 = if pair_ae.speed_cmp > 0 {
                1.0 - pair_ae.ohko_prob
            } else {
                1.0
            };
            p_die_t1 * (pair_ea.raw_dmg - avail) * 1.5
        } else {
            0.0
        };
        let hp_cost = (entry_hit_hp + hits_taken + fragile_bump).min(4.0);

        if avail > 0.02 && (hp_cost <= avail + 1e-6 || hp_cost <= 1e-6) {
            hp_rem[a] = (avail - hp_cost).max(0.0);
            add_t + t_need
        } else {
            let frac_done = if avail <= 0.02 || entry_hit_hp >= avail {
                0.0
            } else {
                let net_avail = (avail - entry_hit_hp).max(0.0);
                let base = if hits_taken > 1e-6 {
                    (net_avail / hits_taken).clamp(0.0, 0.999)
                } else {
                    0.0
                };
                if pair_ae.speed_cmp < 0 {
                    base * (1.0 - pair_ea.ohko_prob).clamp(0.0, 1.0)
                } else {
                    base
                }
            };
            let rem_frac = (1.0 - frac_done).max(0.0);
            let deficit_hp = (hp_cost - avail).clamp(0.0, 2.0);
            let turns_by_a = if avail <= 0.02 {
                0.0
            } else if frac_done <= 0.02 && pair_ae.speed_cmp <= 0 {
                1.0
            } else {
                frac_done * t_need
            };
            hp_rem[a] = 0.0;

            let mut best_spill = if avail <= 0.02 {
                FAST_WALL_CAP
            } else {
                (rem_frac * (t_need.max(2.0) + 4.0) + 2.8 * deficit_hp).min(FAST_WALL_CAP)
            };
            let mut best_helper: Option<(usize, f64)> = None;
            for &b in &allies[..n_allies] {
                if b == a || hp_rem[b] <= 0.02 {
                    continue;
                }
                let pair_be = ttk[side][b][e];
                let pair_eb = ttk[opp][e][b];
                // Helper `b` cannot finish `e` if `b` is slower and gets OHKO'd before moving!
                let surv_b = if pair_be.speed_cmp < 0 {
                    (1.0 - pair_eb.ohko_prob).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if surv_b <= 0.05 {
                    continue;
                }
                let raw_fin = rem_frac * pair_be.raw_ttk;
                let eff_fin = rem_frac * pair_be.eff_ttk;
                let stall_w = ((raw_fin - 1.0) / 1.5).clamp(0.0, 1.0);
                let t_b = raw_fin * (1.0 - stall_w) + eff_fin * stall_w;
                let b_e_att = if pair_be.def_sustain {
                    (1.0 - pair_be.heal_duty).clamp(0.25, 1.0)
                } else {
                    1.0
                };
                let cost_b = if pair_eb.def_sustain {
                    0.20 * rem_frac
                } else {
                    let hits_b = if pair_be.speed_cmp > 0 { (t_b - 1.0).max(0.0) } else { t_b };
                    hits_b * b_e_att * pair_eb.net_dmg.max(0.0)
                };
                let extra_pen = if cost_b > hp_rem[b] {
                    let any_third =
                        allies[..n_allies].iter().any(|&c| c != a && c != b && hp_rem[c] > 0.02);
                    if any_third {
                        (1.8 * (cost_b - hp_rem[b])).min(FAST_WALL_CAP - t_b).max(0.0)
                    } else {
                        // Last surviving helper dies partway: unkillable fraction incurs dead-team wall cost!
                        let done_b = (hp_rem[b] / cost_b.max(1e-6)).clamp(0.0, 0.999) * surv_b;
                        let unkillable = (1.0 - done_b).max(0.0);
                        (unkillable * FAST_WALL_CAP).min(FAST_WALL_CAP - t_b).max(0.0)
                    }
                } else if surv_b < 0.99 {
                    (1.0 - surv_b) * (FAST_WALL_CAP - t_b).max(0.0)
                } else {
                    0.0
                };
                let pivot_pen = if (*used_mask & (1 << b)) == 0 {
                    0.4 * (rem_frac * 2.0).min(1.0)
                } else {
                    0.0
                };
                let cand = (t_b + extra_pen + pivot_pen).min(FAST_WALL_CAP);
                if cand < best_spill {
                    best_spill = cand;
                    best_helper = Some((b, cost_b));
                }
            }
            if let Some((b, cost_b)) = best_helper {
                hp_rem[b] = (hp_rem[b] - cost_b).max(0.0);
                *used_mask |= 1 << b;
            }
            add_t + turns_by_a + best_spill
        }
    };

    // Factored nested loops over `n_enemies in 1..=3` (eliminates integer `%`/`/` and shares
    // Enemy-0 and Enemy-1 prefix states across all subtrees, cutting work ~3x!).
    let mut best_total = f64::INFINITY;
    for &a0 in &allies[..n_allies] {
        let mut hp0 = base_hp;
        let mut um0 = 0u8;
        let mut sw0 = false;
        let t0 = step_enemy(0, a0, &mut hp0, &mut um0, &mut sw0);
        let speed_adj = if ally_active_alive && enemy_active_alive && a0 == act_ally {
            active_speed_adj
        } else {
            0.0
        };
        if n_enemies == 1 {
            let cand = (t0 + speed_adj).max(0.25);
            if cand < best_total {
                best_total = cand;
            }
            continue;
        }
        let min_rem1 = 0.85 * ((n_enemies - 1) as f64) + speed_adj;
        if t0 + min_rem1 >= best_total {
            continue;
        }
        for &a1 in &allies[..n_allies] {
            let mut hp1 = hp0;
            let mut um1 = um0;
            let mut sw1 = sw0;
            let t1 = t0 + step_enemy(1, a1, &mut hp1, &mut um1, &mut sw1);
            if n_enemies == 2 {
                let cand = (t1 + speed_adj).max(0.25);
                if cand < best_total {
                    best_total = cand;
                }
                continue;
            }
            if t1 + 0.85 + speed_adj >= best_total {
                continue;
            }
            for &a2 in &allies[..n_allies] {
                let mut hp2 = hp1;
                let mut um2 = um1;
                let mut sw2 = sw1;
                let t2 = t1 + step_enemy(2, a2, &mut hp2, &mut um2, &mut sw2);
                let cand = (t2 + speed_adj).max(0.25);
                if cand < best_total {
                    best_total = cand;
                }
            }
        }
    }

    // Strictly-monotonic Log-Sum-Exp Free-Energy Team-Depth term (pure turn units, `n_allies >= 2` only):
    // When `n_allies == 1`, there is no backup ally and `best_total` already accounts for exact sequential
    // HP exhaustion across all enemies (blending `g_depth` when `n_allies == 1` would dilute dead-team spill!).
    if n_allies >= 2 {
        const DEPTH_W: f64 = 0.26;
        const DEPTH_TAU: f64 = 2.2;
        let inv_tau = 1.0 / DEPTH_TAU;
        let mut g_depth = 0.0f64;
        for &e in &enemies[..n_enemies] {
            let mut z = 0.0f64;
            for &a in &allies[..n_allies] {
                let p = if a == act_ally && (!ally_active_alive || e != act_enemy) {
                    unb_active.map_or(ttk[side][a][e], |row| row[e])
                } else {
                    ttk[side][a][e]
                };
                let mut t_a = p.eff_ttk;
                // In 2v1/3v1 (`n_enemies == 1`), credit last-enemy Explosion / Perish / DBond in `g_depth` too!
                if n_enemies == 1 && base_hp[a] > 0.08 {
                    let p_act = if p.speed_cmp >= 0 {
                        1.0
                    } else {
                        (1.0 - ttk[opp][e][a].ohko_prob).clamp(0.0, 1.0)
                    };
                    if p_act > 0.45 {
                        if p.boom_dmg_frac > 0.55 {
                            let bt = (1.0 + (1.0 - p.boom_ko_prob).max(0.0) * 1.1) / p_act;
                            if bt < t_a {
                                t_a = bt;
                            }
                        }
                        if ally_perish[a] && (3.5 / p_act) < t_a {
                            t_a = 3.5 / p_act;
                        }
                        if ally_dbond[a] && p_act > 0.55 {
                            let dt = (if p.speed_cmp > 0 { 2.0 } else { 2.6 }) / p_act;
                            if dt < t_a {
                                t_a = dt;
                            }
                        }
                    }
                }
                let w_a = 0.20 + 0.80 * base_hp[a];
                z += w_a * (-t_a * inv_tau).exp();
            }
            let td = if z > 1e-12 {
                (-DEPTH_TAU * z.ln()).clamp(0.4, FAST_WALL_CAP)
            } else {
                FAST_WALL_CAP
            };
            g_depth += td;
        }
        ((1.0 - DEPTH_W) * best_total + DEPTH_W * g_depth).max(0.25)
    } else {
        best_total
    }
}

/// Output of `FastGolf` (v2) evaluator.
#[derive(Clone, Copy, Debug)]
pub struct FastGolfEval {
    pub g_turns: [f64; 2],
    pub turn_advantage: f64,
    pub win_prob: f64,
}

/// Ultra-fast zero-alloc Turn-Economy ("FastGolf") evaluator (~0.4-0.6 us/leaf with cache).
pub fn evaluate_fast_golf(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
) -> FastGolfEval {
    compute_fast_golf_core(cb, table, cache).0
}

#[inline]
pub(crate) fn compute_fast_golf_core(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
) -> (FastGolfEval, [[[FastPair; 3]; 3]; 2], [i16; 6]) {
    if let Some(wp) = compact_turn_limit_outcome(cb, table) {
        let (g0, g1) = if wp > 0.75 {
            (0.0, 6.0)
        } else if wp < 0.25 {
            (6.0, 0.0)
        } else {
            (0.0, 0.0)
        };
        return (
            FastGolfEval { g_turns: [g0, g1], turn_advantage: g1 - g0, win_prob: wp },
            [[[FastPair::default(); 3]; 3]; 2],
            [0i16; 6],
        );
    }

    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    let is_1v1 = n0 <= 1 && n1 <= 1;
    let view_opt = cache.and_then(|c| c.view_for(cb));
    let cache = view_opt.as_ref();

    // Precompute entry HP, effective speed, and sleep delays ONCE across all 6 mons.
    let mut entry_hps = [0i16; 6];
    let mut speeds = [1i32; 6];
    let mut sleep_turns = [0.0f64; 6];
    let mut sleep_eff_delays = [0.0f64; 6];
    for s in 0..2 {
        let mask = cb.sides[s].alive_mask;
        for slot in 0..3 {
            if (mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = s * 3 + slot;
            entry_hps[idx] = entry_hp(cb, table, s, slot);
            speeds[idx] = effective_speed(cb, table, s, slot);
            if cb.mons[idx].status_raw() == Status::Slp as u8 {
                let slp = expected_sleep_turns(cb, s, slot);
                sleep_turns[idx] = slp;
                let st_rate = if let Some(c) = cache {
                    let cm = c.mon(s, slot);
                    let st_s = cm.sleep_talk_slot as usize;
                    if st_s < 4
                        && cb.mons[idx].pp[st_s] > 0
                        && !cb.mons[idx].is_slot_disabled(st_s)
                    {
                        cm.st_attack_rate
                    } else {
                        0.0
                    }
                } else {
                    sleep_talk_attack_rate(cb, table, s, slot)
                };
                sleep_eff_delays[idx] = slp * (1.0 - st_rate);
            }
        }
    }

    let mut ttk = [[[FastPair::default(); 3]; 3]; 2];
    for s0 in 0..3 {
        if (cb.sides[0].alive_mask & (1 << s0)) == 0 {
            continue;
        }
        for s1 in 0..3 {
            if (cb.sides[1].alive_mask & (1 << s1)) == 0 {
                continue;
            }
            let r01 = compute_fast_pair(
                cb, table, 0, s0, 1, s1, &entry_hps, &speeds, &sleep_eff_delays, false, false,
                is_1v1, cache,
            );
            let r10 = compute_fast_pair(
                cb, table, 1, s1, 0, s0, &entry_hps, &speeds, &sleep_eff_delays, false, false,
                is_1v1, cache,
            );

            let mut a01 = r01;
            let mut b10 = r10;
            apply_fast_recovery(
                cb, table, 0, s0, 1, s1, &mut a01, &mut b10, &entry_hps, &sleep_turns, is_1v1,
                cache,
            );

            let mut a10 = r10;
            let mut b01 = r01;
            apply_fast_recovery(
                cb, table, 1, s1, 0, s0, &mut a10, &mut b01, &entry_hps, &sleep_turns, is_1v1,
                cache,
            );

            a01.eff_ttk = a01.eff_ttk.max(b01.eff_ttk);
            a10.eff_ttk = a10.eff_ttk.max(b10.eff_ttk);
            ttk[0][s0][s1] = a01;
            ttk[1][s1][s0] = a10;
        }
    }

    // Zero-copy unboosted active rows (both offensive & defensive) + Phazer / Haze / Encore discount.
    let unb_fast = |s: usize| -> Option<([FastPair; 3], [FastPair; 3])> {
        let act = cb.sides[s].active_idx as usize;
        let act_mon = &cb.mons[s * 3 + act];
        if act_mon.hp() <= 0 || act_mon.boosts == [0; 7] {
            return None;
        }
        let opp = 1 - s;
        let mut att_row = [FastPair::default(); 3];
        let mut def_row = [FastPair::default(); 3];
        for e in 0..3 {
            if (cb.sides[opp].alive_mask & (1 << e)) == 0 {
                continue;
            }
            let r_ae = compute_fast_pair(
                cb, table, s, act, opp, e, &entry_hps, &speeds, &sleep_eff_delays, true, false,
                is_1v1, cache,
            );
            let r_ea = compute_fast_pair(
                cb, table, opp, e, s, act, &entry_hps, &speeds, &sleep_eff_delays, false, true,
                is_1v1, cache,
            );
            let mut a_ae = r_ae;
            let mut b_ea = r_ea;
            apply_fast_recovery(
                cb, table, s, act, opp, e, &mut a_ae, &mut b_ea, &entry_hps, &sleep_turns, is_1v1,
                cache,
            );
            let mut a_ea = r_ea;
            let mut b_ae = r_ae;
            apply_fast_recovery(
                cb, table, opp, e, s, act, &mut a_ea, &mut b_ae, &entry_hps, &sleep_turns, is_1v1,
                cache,
            );
            a_ae.eff_ttk = a_ae.eff_ttk.max(b_ae.eff_ttk);
            a_ea.eff_ttk = a_ea.eff_ttk.max(b_ea.eff_ttk);
            att_row[e] = a_ae;
            def_row[e] = a_ea;
        }
        Some((att_row, def_row))
    };
    let unb0 = unb_fast(0);
    let unb1 = unb_fast(1);

    // Discount active stat boosts when opponent has a living healthy Phazer/Hazer or active is Encored!
    // Compute weights for both sides from unmodified `ttk` first, then blend symmetrically so
    // dual-boosted dual-phazer states preserve exact side-swap antisymmetry (`v(A,B) + v(B,A) == 1`).
    let mut blend_active_s = [0.0f64; 2];
    let mut blend_bench_s = [0.0f64; 2];
    for s in 0..2usize {
        let unb_opt = if s == 0 { &unb0 } else { &unb1 };
        if unb_opt.is_none() {
            continue;
        }
        let act = cb.sides[s].active_idx as usize;
        let act_idx = s * 3 + act;
        let act_mon = &cb.mons[act_idx];
        let pos_boosts = act_mon.boosts[0].max(0)
            + act_mon.boosts[1].max(0)
            + act_mon.boosts[2].max(0)
            + act_mon.boosts[3].max(0);
        if pos_boosts <= 0 {
            continue;
        }
        let opp = 1 - s;
        let my_alive = cb.sides[s].alive_mask.count_ones();
        let opp_act = cb.sides[opp].active_idx as usize;

        let mut blend_active = 0.0f64;
        let mut blend_bench = 0.0f64;

        // 1. Encore lock on a boosted setup mon without Baton Pass.
        if (act_mon.vol_flags & vol_bit::ENCORE) != 0 {
            let has_bp = if let Some(c) = cache {
                let bs = c.mon(s, act).bp_slot as usize;
                bs < 4 && act_mon.pp[bs] > 0 && !act_mon.is_slot_disabled(bs)
            } else {
                (0..table.mons[act_idx].num_moves as usize).any(|m| {
                    act_mon.pp[m] > 0
                        && !act_mon.is_slot_disabled(m)
                        && table.move_specs[act_idx][m].special_kind == SpecialMoveKind::BatonPass
                })
            };
            if !has_bp {
                blend_active = blend_active.max(0.65);
                blend_bench = blend_bench.max(0.65);
            }
        }

        // 2. Living healthy opponent Phazer (Roar/Whirlwind) or Hazer.
        for e in 0..3usize {
            if (cb.sides[opp].alive_mask & (1 << e)) == 0 {
                continue;
            }
            let e_idx = opp * 3 + e;
            let e_mon = &cb.mons[e_idx];
            if e_mon.hp() <= 0 || e_mon.status_raw() == Status::Frz as u8 || sleep_turns[e_idx] >= 1.5
            {
                continue;
            }
            let (can_phaze, can_haze) = if let Some(c) = cache {
                let cm_e = c.mon(opp, e);
                let ps = cm_e.phaze_slot as usize;
                let hs = cm_e.haze_slot as usize;
                (
                    my_alive >= 2 && ps < 4 && e_mon.pp[ps] > 0 && !e_mon.is_slot_disabled(ps),
                    hs < 4 && e_mon.pp[hs] > 0 && !e_mon.is_slot_disabled(hs),
                )
            } else {
                let mut cp = false;
                let mut ch = false;
                for m in 0..table.mons[e_idx].num_moves as usize {
                    if e_mon.pp[m] == 0 || e_mon.is_slot_disabled(m) {
                        continue;
                    }
                    match table.move_specs[e_idx][m].special_kind {
                        SpecialMoveKind::Phaze if my_alive >= 2 => cp = true,
                        SpecialMoveKind::Haze => ch = true,
                        _ => {}
                    }
                }
                (cp, ch)
            };
            if !can_phaze && !can_haze {
                continue;
            }
            let p_ae = ttk[s][act][e];
            let surv = (1.0 - p_ae.ohko_prob).clamp(0.0, 1.0);
            if e == opp_act && p_ae.raw_ttk >= 1.35 {
                let w = 0.65 * surv;
                blend_active = blend_active.max(w);
                blend_bench = blend_bench.max(w);
            } else if e != opp_act && p_ae.raw_ttk >= 2.0 {
                // Bench phazer can switch in and force out after at most 1 boosted hit on active enemy.
                blend_bench = blend_bench.max(0.45 * surv);
            }
        }
        blend_active_s[s] = blend_active;
        blend_bench_s[s] = blend_bench;
    }

    let act0 = cb.sides[0].active_idx as usize;
    let act1 = cb.sides[1].active_idx as usize;
    // Active-vs-active cell: apply commutative symmetric blend if both sides are discounted.
    let w0_act = if unb0.is_some() { blend_active_s[0] } else { 0.0 };
    let w1_act = if unb1.is_some() { blend_active_s[1] } else { 0.0 };
    if w0_act > 0.0 && w1_act > 0.0 {
        let (ref u0_att, ref u0_def) = unb0.as_ref().unwrap();
        let (ref u1_att, ref u1_def) = unb1.as_ref().unwrap();
        let sum_w = w0_act + w1_act;
        let tot = sum_w.min(1.0);
        let wa0 = tot * (w0_act / sum_w);
        let wa1 = tot * (w1_act / sum_w);
        let blend3 = |base: FastPair, a: FastPair, wa: f64, b: FastPair, wb: f64| -> FastPair {
            let w_base = (1.0 - wa - wb).max(0.0);
            FastPair {
                raw_ttk: w_base * base.raw_ttk + wa * a.raw_ttk + wb * b.raw_ttk,
                eff_ttk: w_base * base.eff_ttk + wa * a.eff_ttk + wb * b.eff_ttk,
                net_dmg: w_base * base.net_dmg + wa * a.net_dmg + wb * b.net_dmg,
                raw_dmg: w_base * base.raw_dmg + wa * a.raw_dmg + wb * b.raw_dmg,
                max_norm_frac: w_base * base.max_norm_frac + wa * a.max_norm_frac + wb * b.max_norm_frac,
                ohko_prob: w_base * base.ohko_prob + wa * a.ohko_prob + wb * b.ohko_prob,
                boom_ko_prob: w_base * base.boom_ko_prob + wa * a.boom_ko_prob + wb * b.boom_ko_prob,
                boom_dmg_frac: w_base * base.boom_dmg_frac + wa * a.boom_dmg_frac + wb * b.boom_dmg_frac,
                heal_duty: w_base * base.heal_duty + wa * a.heal_duty + wb * b.heal_duty,
                def_sustain: base.def_sustain || (wa >= 0.5 && a.def_sustain) || (wb >= 0.5 && b.def_sustain),
                speed_cmp: if wa > wb && wa >= 0.5 {
                    a.speed_cmp
                } else if wb > wa && wb >= 0.5 {
                    b.speed_cmp
                } else {
                    base.speed_cmp
                },
                perish_capped: base.perish_capped,
                best_pp: base.best_pp,
            }
        };
        let orig01 = ttk[0][act0][act1];
        let orig10 = ttk[1][act1][act0];
        ttk[0][act0][act1] = blend3(orig01, u0_att[act1], wa0, u1_def[act0], wa1);
        ttk[1][act1][act0] = blend3(orig10, u1_att[act0], wa1, u0_def[act1], wa0);
    } else if w0_act > 0.0 {
        let (ref u0_att, ref u0_def) = unb0.as_ref().unwrap();
        ttk[0][act0][act1] = blend_fast_pair(ttk[0][act0][act1], u0_att[act1], w0_act);
        ttk[1][act1][act0] = blend_fast_pair(ttk[1][act1][act0], u0_def[act1], w0_act);
    } else if w1_act > 0.0 {
        let (ref u1_att, ref u1_def) = unb1.as_ref().unwrap();
        ttk[1][act1][act0] = blend_fast_pair(ttk[1][act1][act0], u1_att[act0], w1_act);
        ttk[0][act0][act1] = blend_fast_pair(ttk[0][act0][act1], u1_def[act0], w1_act);
    }
    // Active-vs-bench cells: strictly disjoint across `s = 0` and `s = 1`, so symmetric regardless of order.
    for s in 0..2usize {
        let wb = blend_bench_s[s];
        if wb <= 0.0 {
            continue;
        }
        let unb_opt = if s == 0 { &unb0 } else { &unb1 };
        let Some((ref unb_att, ref unb_def)) = *unb_opt else {
            continue;
        };
        let opp = 1 - s;
        let act = cb.sides[s].active_idx as usize;
        let opp_act = cb.sides[opp].active_idx as usize;
        for e in 0..3usize {
            if e == opp_act || (cb.sides[opp].alive_mask & (1 << e)) == 0 {
                continue;
            }
            ttk[s][act][e] = blend_fast_pair(ttk[s][act][e], unb_att[e], wb);
            ttk[opp][e][act] = blend_fast_pair(ttk[opp][e][act], unb_def[e], wb);
        }
    }

    let g0 = solve_fast_team_turns(
        cb,
        table,
        0,
        &ttk,
        unb0.as_ref().map(|x| &x.0),
        &entry_hps,
        &sleep_eff_delays,
        cache,
    );
    let g1 = solve_fast_team_turns(
        cb,
        table,
        1,
        &ttk,
        unb1.as_ref().map(|x| &x.0),
        &entry_hps,
        &sleep_eff_delays,
        cache,
    );
    let adv = g1 - g0;
    let wp = if g0 <= 0.0 && g1 <= 0.0 {
        0.5
    } else if g0 <= 0.0 {
        1.0
    } else if g1 <= 0.0 {
        0.0
    } else {
        // Empirically calibrated Brownian race variance per turn across real NC2000 states:
        // sharp in pure 1v1 races (`1.15/turn`), calmly increasing across multi-mon switch games
        // (`2.85..4.80/turn`) so DUCT MCTS exploration (`c = 0.4`) and Nash LP averaging are well calibrated.
        let var_per_turn = if is_1v1 {
            1.15
        } else {
            let extra_mons = (n0 + n1).saturating_sub(2) as f64;
            2.2 + 0.65 * extra_mons
        };
        let sigma = (var_per_turn * (g0 + g1).max(1.0)).sqrt();
        normal_cdf(adv / sigma)
    };
    (FastGolfEval { g_turns: [g0, g1], turn_advantage: adv, win_prob: wp }, ttk, entry_hps)
}

/// Strictly-antisymmetric Hybrid leaf evaluator combining:
/// 1. `FastGolf` turn-economy (3×3 matchup matrix, walls, recovery duty cycle, team-depth spill),
/// 2. `eval::compact_eval01_blindspot_norelay` (`linear.rs` exact 1v1 Schur-Bellman / Toeplitz
///    stopping-time race, 2v1 `V_mat` saddle point, Perish trap, Swagger/Encore/Sub refund, and
///    material/status balance sheet anchoring passive utility-mon leads), and
/// 3. `relay::apply_relay_leaf_lift` (192-state Setup-Relay simultaneous-move Bellman subgame).
#[inline]
pub fn evaluate_hybrid_golf(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
) -> f64 {
    if let Some(v) = compact_turn_limit_outcome(cb, table) {
        return v;
    }
    let w = EvalWeights::default();
    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    let bs_nr = crate::eval::compact_eval01_blindspot_norelay(cb, table, &w);
    // Preserve exact 1v1 `linear.rs` (Toeplitz/Schur-Bellman coupled stopping-time race):
    // never dilute exact 1v1 valuation with FastGolf's Brownian approximation.
    if n0 <= 1 && n1 <= 1 {
        return bs_nr;
    }
    let fg = evaluate_fast_golf(cb, table, cache);
    let logit = |p: f64| -> f64 {
        let q = p.clamp(0.000_5, 0.999_5);
        (q / (1.0 - q)).ln()
    };
    // Normalized (`w_fg + w_bs == 1.00`) strictly-antisymmetric log-odds blend so DUCT MCTS
    // (`c = 0.4`) is never over-sharpened toward 0/1.
    let (w_fg, w_bs) = match (n0.min(n1), n0.max(n1)) {
        (1, 2) => (0.55, 0.45),
        (2, 2) => (0.50, 0.50),
        (3, 3) => (0.60, 0.40),
        _ => (0.65, 0.35),
    };
    let z = w_fg * logit(fg.win_prob) + w_bs * logit(bs_nr);
    let v_base = 1.0 / (1.0 + (-z).exp());
    crate::relay::apply_relay_leaf_lift(cb, table, &w, v_base).clamp(0.000_1, 0.999_9)
}

#[inline]
pub fn evaluate_hybrid_golf_temp(
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
    let bs_nr = crate::eval::compact_eval01_blindspot_norelay(cb, table, &w);
    if n0 <= 1 && n1 <= 1 {
        return bs_nr;
    }
    let fg = evaluate_fast_golf(cb, table, cache);
    let (w_fg, w_bs) = match (n0.min(n1), n0.max(n1)) {
        (1, 2) => (0.55, 0.45),
        (2, 2) => (0.50, 0.50),
        (3, 3) => (0.60, 0.40),
        _ => (0.65, 0.35),
    };
    let z = (w_fg * stage3_logit(fg.win_prob) + w_bs * stage3_logit(bs_nr)) / leaf_temp.max(0.1);
    let v_base = 1.0 / (1.0 + (-z.clamp(-20.0, 20.0)).exp());
    crate::relay::apply_relay_leaf_lift(cb, table, &w, v_base).clamp(0.000_1, 0.999_9)
}

// Stage-3 strictly antisymmetric value-head weights trained on 3,992 strong BlindAgent (3k)
// `team-pool-v2` self-play games (causally monotone: 0-ULP odd symmetry, zero team-confounded cols).
const STAGE3_LOG_W: [[f32; 20]; 3] = [
    [0.172490, 0.165056, 0.776459, 0.656873, 0.060817, 0.262741, 0.228946, 0.987666, 0.000000, 0.000000, 0.000000, 0.000000, 0.038072, 1.024143, 0.000000, 0.066815, 0.000000, 1.401289, 0.344979, 0.000000],
    [0.314939, 0.092967, 1.419599, 0.019691, 0.124642, 0.051862, 0.240639, 0.141182, 0.364278, 0.000000, 0.000000, 0.074280, 0.146601, 0.233786, 0.000000, 0.187076, 0.000000, 0.155128, 0.115697, 0.092967],
    [0.371568, 0.163857, 1.261636, 0.475417, 0.049469, 0.094951, 0.026140, 0.000000, 0.019046, 0.000000, 0.000000, 0.174369, 0.077783, 0.141131, 0.000000, 0.041447, 0.209782, 0.290184, 0.213897, 0.163857],
];
const STAGE3_MLP_SKIP: [f32; 20] = [0.223643, 0.116987, 0.994357, 0.241815, 0.082985, 0.062663, 0.068278, 0.139250, 0.089701, 0.000000, 0.000000, 0.047775, 0.081063, 0.146598, 0.000000, 0.069843, -0.002902, 0.196186, 0.098161, 0.139558];
const STAGE3_MLP_W2: [f32; 16] = [0.161152, -0.161187, 0.094266, -0.113487, -0.007915, -0.057771, 0.116091, -0.054518, 0.146989, 0.017536, 0.126540, 0.166339, -0.039143, -0.063993, 0.144016, 0.119273];
const STAGE3_MLP_W1: [[f32; 20]; 16] = [
    [0.059539, 0.031006, 0.212235, 0.034762, -0.001038, 0.012451, 0.035337, 0.027508, 0.008824, 0.000000, 0.000000, 0.060773, 0.027507, 0.022307, 0.000000, 0.018647, 0.009509, 0.009689, 0.022533, 0.032180],
    [-0.049388, -0.025349, -0.219455, -0.079552, -0.026939, -0.003507, -0.003496, -0.013532, -0.047286, 0.000000, 0.000000, -0.009470, -0.020316, -0.043814, 0.000000, -0.022986, -0.006578, -0.037670, -0.021214, -0.035058],
    [0.038310, 0.015905, 0.169606, 0.024797, 0.000568, -0.012558, -0.010583, -0.001004, 0.007063, 0.000000, 0.000000, 0.042513, 0.019178, 0.029083, 0.000000, 0.014257, -0.020662, -0.026805, 0.030944, 0.021073],
    [-0.042497, -0.019800, -0.166991, -0.055097, -0.028138, -0.010488, -0.030683, -0.019153, 0.013655, 0.000000, 0.000000, -0.005327, -0.006396, -0.029107, 0.000000, -0.010598, 0.002199, -0.048786, -0.014684, -0.025543],
    [-0.009846, -0.004864, -0.041027, -0.008840, 0.007813, -0.011873, -0.009301, -0.012277, -0.021037, 0.000000, 0.000000, 0.025027, -0.004993, 0.010327, 0.000000, 0.011889, 0.030214, -0.017812, -0.002226, -0.003537],
    [-0.023142, -0.016738, -0.096008, 0.027733, 0.000791, 0.001480, 0.003134, -0.030355, 0.024850, 0.000000, 0.000000, -0.013382, -0.008022, 0.014236, 0.000000, -0.013024, 0.001317, -0.026803, -0.018275, -0.015918],
    [0.040512, 0.019592, 0.140716, 0.002676, 0.006995, 0.017875, -0.011907, 0.046403, -0.040647, 0.000000, 0.000000, 0.065386, 0.011114, 0.034260, 0.000000, 0.013661, 0.014491, 0.052645, 0.021627, 0.036211],
    [-0.029586, -0.007387, -0.097120, -0.044955, 0.003178, 0.001011, 0.002697, -0.044352, -0.033615, 0.000000, 0.000000, 0.004729, -0.002238, -0.024821, 0.000000, -0.005004, 0.006808, -0.059526, -0.020575, -0.009200],
    [0.054051, 0.025978, 0.257129, 0.084040, 0.006077, 0.020179, -0.017746, 0.001586, 0.050134, 0.000000, 0.000000, 0.062592, 0.019382, 0.069158, 0.000000, 0.020763, -0.020087, -0.008254, 0.032361, 0.026386],
    [-0.001331, 0.004821, 0.013405, -0.023814, 0.008103, -0.018220, -0.011350, 0.019283, 0.055517, 0.000000, 0.000000, 0.015918, -0.000862, -0.024892, 0.000000, -0.007079, 0.007395, 0.003456, 0.003641, 0.001055],
    [0.039103, 0.018581, 0.165757, 0.067931, -0.007797, 0.009828, 0.037533, 0.047078, -0.024836, 0.000000, 0.000000, 0.064502, 0.014701, 0.030172, 0.000000, 0.005277, 0.018360, -0.035127, 0.029140, 0.032456],
    [0.045934, 0.026876, 0.225383, 0.086581, 0.018817, 0.014668, 0.038656, 0.018860, -0.033166, 0.000000, 0.000000, 0.029194, 0.024999, 0.019784, 0.000000, 0.013250, -0.054120, 0.028341, 0.029617, 0.034488],
    [-0.018012, -0.013576, -0.087572, 0.010153, 0.011016, 0.016822, 0.017435, -0.004450, 0.104672, 0.000000, 0.000000, -0.009730, -0.007300, 0.005871, 0.000000, -0.013414, -0.009089, 0.018229, -0.020402, -0.009677],
    [-0.018376, -0.015947, -0.089276, -0.027482, -0.004447, -0.017155, -0.029364, -0.025744, -0.027626, 0.000000, 0.000000, -0.031312, -0.007428, 0.011470, 0.000000, -0.014427, -0.015322, 0.022872, -0.020335, -0.015299],
    [0.047883, 0.025237, 0.185231, 0.055887, 0.030026, -0.010745, 0.021258, 0.039957, 0.037166, 0.000000, 0.000000, 0.007890, 0.005333, -0.019994, 0.000000, 0.015078, 0.001584, 0.058166, 0.026907, 0.033701],
    [0.041325, 0.020259, 0.155033, 0.055656, 0.027580, 0.022344, 0.029237, 0.005875, 0.006850, 0.000000, 0.000000, 0.034992, 0.010291, 0.052910, 0.000000, 0.014742, 0.013229, -0.013649, 0.016547, 0.032570],
];

#[inline(always)]
fn stage3_fast_tanh(x: f32) -> f32 {
    let x2 = x * x;
    (x * (27.0 + x2) / (27.0 + 9.0 * x2)).clamp(-1.0, 1.0)
}

/// Strictly antisymmetric Stage-3 Option-A (3-regime L2 logistic) leaf evaluator.
/// Preserves exact 1v1 `linear.rs` (`bs_nr`) and wraps `relay::apply_relay_leaf_lift` outside.
#[inline]
pub fn evaluate_stage3_log(
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
        return crate::eval::compact_eval01_blindspot_norelay(cb, table, &w);
    }
    let (phi, _, _, _) = extract_stage3_phi(cb, table, cache, &w);
    let reg = if n0 + n1 == 3 {
        0usize
    } else if n0 + n1 == 4 {
        1usize
    } else {
        2usize
    };
    let wt = &STAGE3_LOG_W[reg];
    let mut z = 0.0f32;
    for k in 0..20usize {
        z += wt[k] * phi[k];
    }
    let z64 = (f64::from(z) / leaf_temp.max(0.1)).clamp(-20.0, 20.0);
    let v_base = 1.0 / (1.0 + (-z64).exp());
    crate::relay::apply_relay_leaf_lift(cb, table, &w, v_base).clamp(0.000_1, 0.999_9)
}

/// Strictly antisymmetric Stage-3 Option-B (zero-bias odd residual MLP `20 -> 16 -> 1`) leaf evaluator.
/// Guarantees `v(A,B) + v(B,A) == 1.0` to 0 ULP, preserves exact 1v1 `linear.rs`, and wraps `relay` outside.
#[inline]
pub fn evaluate_stage3_mlp(
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
        return crate::eval::compact_eval01_blindspot_norelay(cb, table, &w);
    }
    let (phi, _, _, _) = extract_stage3_phi(cb, table, cache, &w);
    let mut z = 0.0f32;
    for k in 0..20usize {
        z += STAGE3_MLP_SKIP[k] * phi[k];
    }
    for h in 0..16usize {
        let row = &STAGE3_MLP_W1[h];
        let mut dot = 0.0f32;
        for k in 0..20usize {
            dot += row[k] * phi[k];
        }
        z += STAGE3_MLP_W2[h] * stage3_fast_tanh(dot);
    }
    let z64 = (f64::from(z) / leaf_temp.max(0.1)).clamp(-20.0, 20.0);
    let v_base = 1.0 / (1.0 + (-z64).exp());
    crate::relay::apply_relay_leaf_lift(cb, table, &w, v_base).clamp(0.000_1, 0.999_9)
}

/// Public wrapper over `mask::public_noop_reason` for data-collection exploration filtering.
pub fn is_public_noop_choice(b: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> bool {
    public_noop_reason(b, dex, side, c).is_some()
}

/// Strictly antisymmetric 20-feature Stage-3 leaf representation (`phi(B, A) == -phi(A, B)` to 0 ULP).
#[derive(Clone, Copy, Debug)]
pub struct Stage3Features {
    pub phi: [f32; 20],
    pub g_stag: f32,
    pub fg_p: f64,
    pub bs_p: f64,
    pub hy_p: f64,
    pub n0: u8,
    pub n1: u8,
    pub relay_elig: bool,
}

#[inline(always)]
pub(crate) fn stage3_logit(p: f64) -> f64 {
    let q = p.clamp(0.000_5, 0.999_5);
    (q / (1.0 - q)).ln()
}

/// Extract the 20 strictly antisymmetric Stage-3 features directly from `FastGolf`'s `ttk`/`entry_hps`
/// intermediates + O(1) `CompactBattle` / `TempoMatchCache` lookups (zero extra `DmgKernel1D` calls,
/// zero heap allocation, zero redundant `relay` calls on the hot path).
#[inline]
fn extract_stage3_phi(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
    w: &EvalWeights,
) -> ([f32; 20], f32, f64, f64) {
    let n0 = cb.sides[0].alive_mask.count_ones() as u8;
    let n1 = cb.sides[1].alive_mask.count_ones() as u8;
    let bs_nr = crate::eval::compact_eval01_blindspot_norelay(cb, table, w);
    let (fg, ttk, entry_hps) = compute_fast_golf_core(cb, table, cache);

    let logit_fg = stage3_logit(fg.win_prob);
    let logit_bs = stage3_logit(bs_nr);

    let view_opt = cache.and_then(|c| c.view_for(cb));
    let cview = view_opt.as_ref();
    let mid_switch = cb.sides[0].request == 2 || cb.sides[1].request == 2;
    let alive_cnt = [n0 as usize, n1 as usize];
    let acts = [cb.sides[0].active_idx as usize, cb.sides[1].active_idx as usize];
    let flying_bit = 1u32 << (table.flying_type & 31);
    let ground_bit = 1u32 << (table.ground_type & 31);

    // Precompute per-side symmetric helpers (identical formula for s=0 and s=1 -> 0-ULP antisymmetry).
    let mut entry_hp_sum = [0.0f64; 2];
    let mut wall_hard = [0.0f64; 2];
    let mut no_boom_best_ttk = [[0.0f64; 3]; 2];
    let mut rec_res = [0.0f64; 2];
    let mut tot_pp_res = [0.0f64; 2];
    let mut grounded_alive = [0.0f64; 2];
    let mut has_live_toxic = [0.0f64; 2];
    let mut has_live_phaze = [0.0f64; 2];
    let mut has_live_healbell = [0.0f64; 2];
    let mut opp_disrupt_cnt = [0u32; 2]; // how many healthy disruptors side `s` has
    let mut opp_boom_cnt = [0u32; 2]; // how many living boomers side `s` has
    let mut has_asleep = [false; 2];
    let mut status_berry_cnt = [0.0f64; 2];
    let mut unstatused_tox_targets = [0.0f64; 2];
    let mut unstatused_par_targets = [0.0f64; 2];
    let mut item_shield = [0.0f64; 2];

    for s in 0..2usize {
        let mask = cb.sides[s].alive_mask;
        for slot in 0..3usize {
            if (mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = s * 3 + slot;
            let mon = &cb.mons[idx];
            let maxhp = table.mons[idx].maxhp.max(1) as f64;
            let hp_frac = (mon.hp().max(0) as f64) / maxhp;
            entry_hp_sum[s] += (entry_hps[idx].max(0) as f64) / maxhp;

            let tmask = table.mons[idx].type_mask;
            if (tmask & flying_bit) == 0 {
                grounded_alive[s] += 1.0;
            }

            let st = mon.status_raw();
            if st == Status::Slp as u8 {
                has_asleep[s] = true;
            }
            let ispec = table.item_spec(mon.item);
            let hp_w = 0.4 + 0.6 * hp_frac;
            if ispec.is_miracleberry {
                status_berry_cnt[s] += 1.0;
                item_shield[s] += 1.0 * hp_w;
            } else if ispec.is_mintberry {
                status_berry_cnt[s] += 0.6;
                let has_rest = if let Some(cv) = cview {
                    cv.mon(s, slot).rest_slot < 4
                } else {
                    (0..table.mons[idx].num_moves as usize)
                        .any(|m| table.move_specs[idx][m].is_rest)
                };
                item_shield[s] += (if has_rest { 0.75 } else { 0.40 }) * hp_w;
            } else if ispec.is_przcure_berry {
                item_shield[s] += 0.40 * hp_w;
            } else if ispec.is_leftovers {
                item_shield[s] += 0.50 * hp_w;
            } else if ispec.is_goldberry || ispec.is_berry_juice {
                item_shield[s] += 0.35 * hp_w;
            }

            let mut half_pp = 0.0f64;
            let mut rest_pp = 0.0f64;
            let mut tot_pp = 0.0f64;
            let mut has_disrupt = false;
            let mut has_boom = false;
            let mut has_rest_move = false;
            let n_moves = table.mons[idx].num_moves as usize;
            if let Some(cv) = cview {
                let cm = cv.mon(s, slot);
                for m in 0..n_moves {
                    tot_pp += f64::from(mon.pp[m]);
                    if table.move_specs[idx][m].is_selfdestruct && mon.pp[m] > 0 {
                        has_boom = true;
                    }
                }
                if (cm.half_heal_slot as usize) < 4 {
                    half_pp = f64::from(mon.pp[cm.half_heal_slot as usize]);
                }
                if (cm.rest_slot as usize) < 4 {
                    rest_pp = f64::from(mon.pp[cm.rest_slot as usize]);
                    has_rest_move = mon.pp[cm.rest_slot as usize] > 0;
                }
                if cm.toxic_mask != 0
                    && (0..n_moves).any(|m| ((cm.toxic_mask >> m) & 1) != 0 && mon.pp[m] > 0)
                {
                    has_live_toxic[s] = 1.0;
                }
                if (cm.phaze_slot as usize) < 4 && mon.pp[cm.phaze_slot as usize] > 0 {
                    has_live_phaze[s] = 1.0;
                    has_disrupt = true;
                }
                if ((cm.haze_slot as usize) < 4 && mon.pp[cm.haze_slot as usize] > 0)
                    || ((cm.encore_slot as usize) < 4 && mon.pp[cm.encore_slot as usize] > 0)
                {
                    has_disrupt = true;
                }
                if (cm.healbell_slot as usize) < 4 && mon.pp[cm.healbell_slot as usize] > 0 {
                    has_live_healbell[s] = 1.0;
                }
            } else {
                for m in 0..n_moves {
                    let pp = mon.pp[m];
                    tot_pp += f64::from(pp);
                    let ms = &table.move_specs[idx][m];
                    if ms.is_half_heal || ms.special_kind == SpecialMoveKind::WeatherHeal {
                        half_pp = half_pp.max(f64::from(pp));
                    }
                    if ms.is_rest {
                        rest_pp = rest_pp.max(f64::from(pp));
                        if pp > 0 {
                            has_rest_move = true;
                        }
                    }
                    if pp > 0 {
                        if ms.is_selfdestruct {
                            has_boom = true;
                        }
                        if ms.primary_status == Status::Tox as u8 {
                            has_live_toxic[s] = 1.0;
                        }
                        match ms.special_kind {
                            SpecialMoveKind::Phaze => {
                                has_live_phaze[s] = 1.0;
                                has_disrupt = true;
                            }
                            SpecialMoveKind::Haze | SpecialMoveKind::Encore => {
                                has_disrupt = true;
                            }
                            SpecialMoveKind::HealBell => {
                                has_live_healbell[s] = 1.0;
                            }
                            _ => {}
                        }
                    }
                }
            }
            rec_res[s] += (1.0 + (half_pp + 2.2 * rest_pp) / 8.0).ln();
            tot_pp_res[s] += (tot_pp / 64.0).sqrt();
            if has_disrupt && hp_frac >= 0.30 && st != Status::Frz as u8 {
                opp_disrupt_cnt[s] += 1;
            }
            if has_boom && st != Status::Frz as u8 {
                opp_boom_cnt[s] += 1;
            }
            if st == Status::None as u8 && !ispec.is_miracleberry {
                if !has_rest_move && (tmask & table.poison_immune_mask) == 0 {
                    unstatused_tox_targets[s] += 1.0;
                }
                if (tmask & ground_bit) == 0 && !ispec.is_przcure_berry {
                    unstatused_par_targets[s] += 1.0;
                }
            }
        }
    }

    // Minimax wall hardness per side: hardest-to-break opponent mon for side `s` (both with and without Boom).
    for s in 0..2usize {
        let opp = 1 - s;
        let opp_mask = cb.sides[opp].alive_mask;
        let my_mask = cb.sides[s].alive_mask;
        let mut hard = 0.0f64;
        for e in 0..3usize {
            if (opp_mask & (1 << e)) == 0 {
                continue;
            }
            let mut best_break = 7.5f64;
            let mut best_no_boom = 7.5f64;
            for a in 0..3usize {
                if (my_mask & (1 << a)) == 0 {
                    continue;
                }
                let p = ttk[s][a][e];
                if p.eff_ttk < best_no_boom {
                    best_no_boom = p.eff_ttk;
                }
                let mut t = p.eff_ttk;
                if alive_cnt[s] >= 2 && p.boom_dmg_frac >= 0.72 {
                    t = t.min(1.6 / p.boom_dmg_frac.max(0.5));
                }
                if t < best_break {
                    best_break = t;
                }
            }
            no_boom_best_ttk[s][e] = best_no_boom;
            if best_break > hard {
                hard = best_break;
            }
        }
        wall_hard[s] = hard;
    }

    // Symmetric mutual-stagnation gate `g_stag ∈ [0, 1]` for Frontier #1 (`n0 + n1 >= 4`).
    let g_stag = if n0 + n1 >= 4 {
        let h_mut = wall_hard[0].min(wall_hard[1]);
        ((h_mut - 4.2) / 2.8).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Per-side interaction features: forced-switch trap, unphazed setup, sleep/status threat, boom leverage, trap/pursuit.
    let mut trap_press = [0.0f64; 2];
    let mut stag_prog = [0.0f64; 2];
    let mut setup_wincon = [0.0f64; 2];
    let mut sleep_thr = [0.0f64; 2];
    let mut status_thr = [0.0f64; 2];
    let mut boom_lev = [0.0f64; 2];
    let mut trap_pursuit = [0.0f64; 2];

    for s in 0..2usize {
        let opp = 1 - s;
        // Dead Spikes on a 1-alive opponent can never trigger again.
        let opp_spikes = if alive_cnt[opp] >= 2 {
            f64::from(spikes_layers(cb, opp))
        } else {
            0.0
        };
        stag_prog[s] = 0.50 * opp_spikes * (grounded_alive[opp] / 3.0)
            + 0.35 * has_live_toxic[s]
            + 0.25 * has_live_phaze[s]
            + 0.35 * has_live_healbell[s];

        // Forced-switch trap pressure (`s` forcing `opp` active out into heavy bench hit + Spikes).
        if !mid_switch && alive_cnt[opp] >= 2 {
            let a_s = acts[s];
            let a_o = acts[opp];
            if (cb.sides[s].alive_mask & (1 << a_s)) != 0
                && (cb.sides[opp].alive_mask & (1 << a_o)) != 0
            {
                let p_ao = ttk[s][a_s][a_o];
                let p_oa = ttk[opp][a_o][a_s];
                let slower_ohkod = p_ao.speed_cmp < 0 && p_oa.ohko_prob >= 0.5;
                let force = if !slower_ohkod
                    && (p_ao.raw_ttk <= 1.35 || (p_ao.speed_cmp > 0 && p_ao.ohko_prob >= 0.4))
                {
                    1.0
                } else if !slower_ohkod && p_ao.raw_ttk <= 2.15 && p_oa.raw_ttk >= 2.8 {
                    0.55
                } else {
                    0.0
                };
                if force > 0.0 {
                    let mut min_sw = f64::INFINITY;
                    for b in 0..3usize {
                        if b == a_o || (cb.sides[opp].alive_mask & (1 << b)) == 0 {
                            continue;
                        }
                        let b_idx = opp * 3 + b;
                        let spk_d = if opp_spikes > 0.0
                            && (table.mons[b_idx].type_mask & flying_bit) == 0
                        {
                            0.125
                        } else {
                            0.0
                        };
                        let sw_dmg = spk_d + ttk[s][a_s][b].raw_dmg.min(1.1);
                        if sw_dmg < min_sw {
                            min_sw = sw_dmg;
                        }
                    }
                    if min_sw.is_finite() {
                        trap_press[s] = force * min_sw.min(1.25);
                    }
                }
            }
        }

        // Unphazed setup win-con, sleep/status threats, boom leverage, MeanLook+Perish & Pursuit.
        let disrupt_disc = match opp_disrupt_cnt[opp] {
            0 => 1.0,
            1 => 0.30,
            _ => 0.0,
        };
        let boom_disc = match opp_boom_cnt[opp] {
            0 => 1.0,
            1 => 0.78,
            _ => 0.55,
        };
        let sleep_clause_open =
            ((cb.clause_flags & 1) == 0 && !table.ruleset.is_castle()) || !has_asleep[opp];
        let opp_berry_frac = status_berry_cnt[opp] / (alive_cnt[opp].max(1) as f64);
        let opp_hb_disc = if has_live_healbell[opp] > 0.0 { 0.45 } else { 1.0 };

        let my_mask = cb.sides[s].alive_mask;
        let opp_mask = cb.sides[opp].alive_mask;
        for slot in 0..3usize {
            if (my_mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = s * 3 + slot;
            let mon = &cb.mons[idx];
            let maxhp = table.mons[idx].maxhp.max(1) as f64;
            let hp_frac = (mon.hp().max(0) as f64) / maxhp;
            let st = mon.status_raw();
            let can_act = st != Status::Frz as u8 && st != Status::Slp as u8;
            let n_moves = table.mons[idx].num_moves as usize;
            let is_ghost = (table.mons[idx].type_mask & (1u32 << (table.ghost_type & 31))) != 0;

            let (has_setup, has_rec, has_sleep, has_tox, has_par, has_ml_perish, pursuit_slot) =
                if let Some(cv) = cview {
                    let cm = cv.mon(s, slot);
                    let su = cm.setup_mask != 0
                        && (0..n_moves).any(|m| ((cm.setup_mask >> m) & 1) != 0 && mon.pp[m] > 0);
                    let rc = ((cm.half_heal_slot as usize) < 4
                        && mon.pp[cm.half_heal_slot as usize] > 0)
                        || ((cm.rest_slot as usize) < 4 && mon.pp[cm.rest_slot as usize] > 0);
                    let sl = cm.sleep_mask != 0
                        && (0..n_moves).any(|m| ((cm.sleep_mask >> m) & 1) != 0 && mon.pp[m] > 0);
                    let tx = cm.toxic_mask != 0
                        && (0..n_moves).any(|m| ((cm.toxic_mask >> m) & 1) != 0 && mon.pp[m] > 0);
                    let pr = cm.twave_mask != 0
                        && (0..n_moves).any(|m| ((cm.twave_mask >> m) & 1) != 0 && mon.pp[m] > 0);
                    let mlp = (cm.meanlook_slot as usize) < 4
                        && mon.pp[cm.meanlook_slot as usize] > 0
                        && (cm.perish_slot as usize) < 4
                        && mon.pp[cm.perish_slot as usize] > 0;
                    let ps = if (cm.pursuit_slot as usize) < 4
                        && mon.pp[cm.pursuit_slot as usize] > 0
                    {
                        cm.pursuit_slot as usize
                    } else {
                        0xFF
                    };
                    (su, rc, sl, tx, pr, mlp, ps)
                } else {
                    let mut su = false;
                    let mut rc = false;
                    let mut sl = false;
                    let mut tx = false;
                    let mut pr = false;
                    let mut ml = false;
                    let mut pe = false;
                    let mut ps = 0xFFusize;
                    for m in 0..n_moves {
                        if mon.pp[m] == 0 {
                            continue;
                        }
                        let ms = &table.move_specs[idx][m];
                        if (ms.special_kind == SpecialMoveKind::Curse && !is_ghost)
                            || ms.special_kind == SpecialMoveKind::BellyDrum
                            || (ms.is_status
                                && ms.boosts_self
                                && ms.boosts.iter().any(|&(st_i, d)| st_i < 5 && d > 0))
                        {
                            su = true;
                        }
                        if ms.is_rest
                            || ms.is_half_heal
                            || ms.special_kind == SpecialMoveKind::WeatherHeal
                        {
                            rc = true;
                        }
                        if ms.primary_status == Status::Slp as u8 {
                            sl = true;
                        } else if ms.primary_status == Status::Tox as u8 {
                            tx = true;
                        } else if ms.primary_status == Status::Par as u8 {
                            pr = true;
                        }
                        if ms.special_kind == SpecialMoveKind::MeanLook {
                            ml = true;
                        }
                        if ms.special_kind == SpecialMoveKind::PerishSong {
                            pe = true;
                        }
                        if ms.special_kind == SpecialMoveKind::Pursuit && ps == 0xFF {
                            ps = m;
                        }
                    }
                    (su, rc, sl, tx, pr, ml && pe, ps)
                };

            if has_setup && st != Status::Frz as u8 {
                let rec_bonus = if has_rec { 1.35 } else { 1.0 };
                setup_wincon[s] += hp_frac * rec_bonus * disrupt_disc * boom_disc;
            }
            if has_sleep && can_act && sleep_clause_open {
                sleep_thr[s] = sleep_thr[s].max(hp_frac * (1.0 - 0.35 * opp_berry_frac));
            }
            if can_act {
                if has_tox {
                    status_thr[s] += 0.35
                        * (unstatused_tox_targets[opp] / 3.0)
                        * opp_hb_disc
                        * (0.5 + 0.5 * hp_frac);
                }
                if has_par {
                    status_thr[s] += 0.35
                        * (unstatused_par_targets[opp] / 3.0)
                        * opp_hb_disc
                        * (0.5 + 0.5 * hp_frac);
                }
            }
            if alive_cnt[s] >= 2 {
                // Permutation-invariant wall-breaking Boom leverage weighted by pre-Boom wall hardness.
                for e in 0..3usize {
                    if (opp_mask & (1 << e)) == 0 {
                        continue;
                    }
                    let bd = ttk[s][slot][e].boom_dmg_frac;
                    if bd > 0.0 {
                        let bval = bd * (no_boom_best_ttk[s][e] / 7.5) * (0.4 + 0.6 * hp_frac);
                        if bval > boom_lev[s] {
                            boom_lev[s] = bval;
                        }
                    }
                }
            }
            if has_ml_perish && hp_frac > 0.35 && st != Status::Frz as u8 {
                let opp_act_mon = &cb.mons[opp * 3 + acts[opp]];
                let active_trap = !mid_switch
                    && slot == acts[s]
                    && (opp_act_mon.vol_flags & vol_bit::TRAPPED_OR_LOCKED) != 0
                    && opp_act_mon.perish_turns() > 0;
                trap_pursuit[s] += if active_trap { 1.2 * hp_frac } else { 0.65 * hp_frac };
            }
            if pursuit_slot < 4 && alive_cnt[opp] >= 2 && can_act {
                let pm_type = table.move_specs[idx][pursuit_slot].move_type as usize;
                let is_weak = |e: usize| -> bool {
                    let e_idx = opp * 3 + e;
                    let eff = table.type_chart[pm_type][table.mons[e_idx].types[0] as usize]
                        + if table.mons[e_idx].num_types > 1 {
                            table.type_chart[pm_type][table.mons[e_idx].types[1] as usize]
                        } else {
                            0
                        };
                    eff > 0
                };
                let a_o = acts[opp];
                let act_weak =
                    !mid_switch && (opp_mask & (1 << a_o)) != 0 && is_weak(a_o);
                if act_weak && slot == acts[s] && ttk[s][slot][a_o].raw_ttk <= 2.2 {
                    trap_pursuit[s] += 0.45 * (0.5 + 0.5 * hp_frac);
                } else if (0..3usize).any(|e| (opp_mask & (1 << e)) != 0 && is_weak(e)) {
                    trap_pursuit[s] += 0.15 * (0.5 + 0.5 * hp_frac);
                }
            }
        }
    }

    let active_duel = if mid_switch
        || (cb.sides[0].alive_mask & (1 << acts[0])) == 0
        || (cb.sides[1].alive_mask & (1 << acts[1])) == 0
    {
        0.0
    } else {
        ((ttk[1][acts[1]][acts[0]].eff_ttk - ttk[0][acts[0]][acts[1]].eff_ttk) / 7.5)
            .clamp(-1.0, 1.0)
    };
    let spk0_live = if alive_cnt[0] >= 2 { f64::from(spikes_layers(cb, 0)) } else { 0.0 };
    let spk1_live = if alive_cnt[1] >= 2 { f64::from(spikes_layers(cb, 1)) } else { 0.0 };
    let spikes_gnd_diff =
        (spk1_live * grounded_alive[1] - spk0_live * grounded_alive[0]) / 3.0;
    let man_asym = ((i32::from(n0) - i32::from(n1)).abs() as f64) * 0.5;
    let multi4 = if n0 + n1 >= 4 { 1.0 } else { 0.0 };

    let phi: [f32; 20] = [
        logit_fg as f32,
        logit_bs as f32,
        ((fg.g_turns[1] - fg.g_turns[0]) / (fg.g_turns[0] + fg.g_turns[1]).max(1.0)) as f32,
        ((f64::from(n0) - f64::from(n1)) / 3.0) as f32,
        (entry_hp_sum[0] - entry_hp_sum[1]) as f32,
        active_duel as f32,
        (trap_press[0] - trap_press[1]) as f32,
        ((wall_hard[1] - wall_hard[0]) / 7.5).clamp(-1.0, 1.0) as f32,
        (g_stag * (rec_res[0] - rec_res[1])) as f32,
        (g_stag * (tot_pp_res[0] - tot_pp_res[1])) as f32,
        (g_stag * (stag_prog[0] - stag_prog[1])) as f32,
        spikes_gnd_diff as f32,
        (setup_wincon[0] - setup_wincon[1]) as f32,
        (sleep_thr[0] - sleep_thr[1]) as f32,
        (status_thr[0] - status_thr[1]) as f32,
        (item_shield[0] - item_shield[1]) as f32,
        (boom_lev[0] - boom_lev[1]) as f32,
        (trap_pursuit[0] - trap_pursuit[1]) as f32,
        (logit_fg * man_asym) as f32,
        (logit_bs * multi4) as f32,
    ];

    (phi, g_stag as f32, fg.win_prob, bs_nr)
}

/// Full Stage-3 feature extraction including `hy_p` and `relay_elig` for offline dataset generation.
#[inline]
pub fn extract_stage3_features(
    cb: &CompactBattle,
    table: &MatchTable,
    cache: Option<&TempoMatchCache>,
) -> Stage3Features {
    let w = EvalWeights::default();
    let n0 = cb.sides[0].alive_mask.count_ones() as u8;
    let n1 = cb.sides[1].alive_mask.count_ones() as u8;
    let (phi, g_stag, fg_p, bs_p) = extract_stage3_phi(cb, table, cache, &w);
    let relay_elig = crate::relay::is_relay_eligible(cb, table, 0)
        || crate::relay::is_relay_eligible(cb, table, 1);
    let hy_p = if n0 <= 1 && n1 <= 1 {
        bs_p
    } else {
        let (w_fg, w_bs) = match (n0.min(n1), n0.max(n1)) {
            (1, 2) => (0.55, 0.45),
            (2, 2) => (0.50, 0.50),
            (3, 3) => (0.60, 0.40),
            _ => (0.65, 0.35),
        };
        let vb = 1.0 / (1.0 + (-(w_fg * f64::from(phi[0]) + w_bs * f64::from(phi[1]))).exp());
        crate::relay::apply_relay_leaf_lift(cb, table, &w, vb).clamp(0.000_1, 0.999_9)
    };
    Stage3Features { phi, g_stag, fg_p, bs_p, hy_p, n0, n1, relay_elig }
}

/// Diagnostic breakdown of `evaluate_fast_golf` for offline calibration & autopsy (not used in hot path).
pub fn format_fast_golf_debug(cb: &CompactBattle, table: &MatchTable) -> String {
    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    let is_1v1 = n0 <= 1 && n1 <= 1;
    let mut entry_hps = [0i16; 6];
    let mut speeds = [1i32; 6];
    let mut sleep_turns = [0.0f64; 6];
    let mut sleep_eff_delays = [0.0f64; 6];
    for s in 0..2 {
        for slot in 0..3 {
            if (cb.sides[s].alive_mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = s * 3 + slot;
            entry_hps[idx] = entry_hp(cb, table, s, slot);
            speeds[idx] = effective_speed(cb, table, s, slot);
            if cb.mons[idx].status_raw() == Status::Slp as u8 {
                let slp = expected_sleep_turns(cb, s, slot);
                sleep_turns[idx] = slp;
                sleep_eff_delays[idx] = slp * (1.0 - sleep_talk_attack_rate(cb, table, s, slot));
            }
        }
    }
    let mut out = String::new();
    for s0 in 0..3 {
        if (cb.sides[0].alive_mask & (1 << s0)) == 0 {
            continue;
        }
        for s1 in 0..3 {
            if (cb.sides[1].alive_mask & (1 << s1)) == 0 {
                continue;
            }
            let mut a01 = compute_fast_pair(
                cb, table, 0, s0, 1, s1, &entry_hps, &speeds, &sleep_eff_delays, false, false,
                is_1v1, None,
            );
            let mut a10 = compute_fast_pair(
                cb, table, 1, s1, 0, s0, &entry_hps, &speeds, &sleep_eff_delays, false, false,
                is_1v1, None,
            );
            let mut b10 = a10;
            let mut b01 = a01;
            apply_fast_recovery(
                cb, table, 0, s0, 1, s1, &mut a01, &mut b10, &entry_hps, &sleep_turns, is_1v1, None,
            );
            apply_fast_recovery(
                cb, table, 1, s1, 0, s0, &mut a10, &mut b01, &entry_hps, &sleep_turns, is_1v1, None,
            );
            a01.eff_ttk = a01.eff_ttk.max(b01.eff_ttk);
            a10.eff_ttk = a10.eff_ttk.max(b10.eff_ttk);
            out.push_str(&format!(
                "    [p0#{s0}->p1#{s1}] raw={:.2} eff={:.2} dmg={:.0}% ohko={:.0}% boom={:.0}%/{:.0}% sus={} spd={:+} | [rev] raw={:.2} eff={:.2} dmg={:.0}% ohko={:.0}% boom={:.0}%/{:.0}% sus={}\n",
                a01.raw_ttk, a01.eff_ttk, a01.net_dmg * 100.0, a01.ohko_prob * 100.0,
                a01.boom_ko_prob * 100.0, a01.boom_dmg_frac * 100.0, a01.def_sustain as u8, a01.speed_cmp,
                a10.raw_ttk, a10.eff_ttk, a10.net_dmg * 100.0, a10.ohko_prob * 100.0,
                a10.boom_ko_prob * 100.0, a10.boom_dmg_frac * 100.0, a10.def_sustain as u8,
            ));
        }
    }
    out
}

/// Run zero-rollout CompactBattle DUCT SM-MCTS from `battle` for `side`.
pub fn search_tempo_mcts(
    battle: &Battle,
    dex: &Dex,
    side: usize,
    own_choices: &[SearchChoice],
    cfg: TempoMctsConfig,
    seed: u64,
) -> TempoMctsResult {
    let (table, base_cb) = CompactBattle::from_battle(battle, dex);
    let fast_cache = if matches!(cfg.leaf, TempoLeafMode::FastGolf | TempoLeafMode::HybridGolf) {
        Some(TempoMatchCache::build(&base_cb, &table))
    } else {
        None
    };
    let mut nodes: Vec<DuctNode> = Vec::with_capacity((cfg.iterations as usize / 2).max(16));
    let mut index: FxHashMap<u64, usize> = FxHashMap::default();

    // Build root node using exact caller `own_choices` order so indices align 1:1.
    let mut root = DuctNode::new(&base_cb, &table);
    let mut own_codes = [0u8; 6];
    let mut n_own = 0usize;
    for &c in own_choices {
        if n_own < 6 {
            if let Some(code) = encode_choice(&table, &base_cb, dex, side, c) {
                own_codes[n_own] = code;
                n_own += 1;
            }
        }
    }
    debug_assert_eq!(
        n_own,
        own_choices.len().min(6),
        "every legal SearchChoice at root must encode 1:1 without shifting indices"
    );
    if n_own > 0 {
        root.n_acts[side] = n_own as u8;
        root.acts[side] = own_codes;
    }
    let root_n0 = root.n_acts[0] as usize;
    let root_n1 = root.n_acts[1] as usize;

    let root_key = compact_duct_key(&base_cb, &table, 0);
    nodes.push(root);
    index.insert(root_key, 0);

    let mut joint_v = [[0u32; 7]; 7];
    let mut joint_s = [[0.0f64; 7]; 7];
    let mut rng = SplitMix64::new(seed ^ 0xD1B5_4A32_D192_ED03);
    let mut path: Vec<(usize, usize, usize)> = Vec::with_capacity(cfg.max_depth as usize + 1);
    let mut depth_sum = 0u64;

    let warmup_pairs = if cfg.argmax_root {
        0
    } else {
        (root_n0 * root_n1).min(cfg.iterations as usize)
    };

    for iter in 0..cfg.iterations as usize {
        let mut cb = base_cb;
        let mut depth = 0u8;
        let mut cur_id = 0usize;
        path.clear();

        let leaf_val = loop {
            let n0_cur = nodes[cur_id].n_acts[0] as usize;
            let n1_cur = nodes[cur_id].n_acts[1] as usize;
            let (i0, i1) = if depth == 0 && iter < warmup_pairs {
                (iter / root_n1.max(1), iter % root_n1.max(1))
            } else {
                let node = &nodes[cur_id];
                (
                    node.select_ucb(0, cfg.c, &mut rng, cfg.argmax_root),
                    node.select_ucb(1, cfg.c, &mut rng, cfg.argmax_root),
                )
            };
            let a0 = if n0_cur > 0 { nodes[cur_id].acts[0][i0] } else { 7 };
            let a1 = if n1_cur > 0 { nodes[cur_id].acts[1][i1] } else { 7 };
            path.push((cur_id, i0, i1));

            cb.prng_seed = rng.next();
            cb.apply_actions(&table, [a0, a1]);
            depth += 1;

            if cb.outcome != 0 {
                break eval_duct_leaf(&cb, &table, cfg.leaf, fast_cache.as_ref());
            }
            if let Some(v) = compact_turn_limit_outcome(&cb, &table) {
                break v;
            }
            if depth >= cfg.max_depth {
                break eval_duct_leaf(&cb, &table, cfg.leaf, fast_cache.as_ref());
            }

            let next_key = compact_duct_key(&cb, &table, depth);
            if let Some(&next_id) = index.get(&next_key) {
                cur_id = next_id;
            } else {
                let new_id = nodes.len();
                nodes.push(DuctNode::new(&cb, &table));
                index.insert(next_key, new_id);
                break eval_duct_leaf_with_rollout(
                    &mut cb,
                    &table,
                    cfg.leaf,
                    cfg.rollout_turns,
                    fast_cache.as_ref(),
                    &mut rng,
                );
            }
        };

        depth_sum += depth as u64;
        if cfg.minimax_backup {
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
                    val = n.backed_up_value(val);
                }
            }
            if let Some(&(_, r0, r1)) = path.first() {
                joint_v[r0][r1] += 1;
                joint_s[r0][r1] += val;
            }
        } else {
            for &(nid, i0, i1) in &path {
                let n = &mut nodes[nid];
                n.total += 1;
                if n.n_acts[0] > 0 {
                    n.visits[0][i0] += 1;
                    n.sums[0][i0] += leaf_val;
                }
                if n.n_acts[1] > 0 {
                    n.visits[1][i1] += 1;
                    n.sums[1][i1] += leaf_val;
                }
            }
            if let Some(&(_, r0, r1)) = path.first() {
                joint_v[r0][r1] += 1;
                joint_s[r0][r1] += leaf_val;
            }
        }
    }

    let root_node = &nodes[0];
    let n_my = own_choices.len();
    let opp = 1 - side;
    let n_op = root_node.n_acts[opp] as usize;

    let mut visits = vec![0u32; n_my];
    let mut means = vec![0.5f64; n_my];
    for i in 0..n_my {
        let v = root_node.visits[side][i];
        visits[i] = v;
        if v > 0 {
            let raw = root_node.sums[side][i] / (v as f64);
            means[i] = if side == 0 { raw } else { 1.0 - raw };
        }
    }

    // Root readout: either `argmax visits` + dominated guard (matching `skuct`) or shrunken Nash LP.
    let (own_policy, eq_win_prob) = if n_my <= 1 {
        (vec![1.0; n_my], means.first().copied().unwrap_or(0.5))
    } else if cfg.argmax_root || n_op <= 1 {
        let dom: Vec<bool> = own_choices
            .iter()
            .map(|&c| dominated_reason(battle, dex, side, c).is_some())
            .collect();
        let any_safe = dom.iter().any(|&d| !d);
        let best_i = (0..n_my)
            .filter(|&i| !any_safe || !dom[i])
            .max_by(|&a, &b| visits[a].cmp(&visits[b]).then_with(|| means[a].total_cmp(&means[b])))
            .unwrap_or(0);
        let mut pol = vec![0.0; n_my];
        pol[best_i] = 1.0;
        (pol, means[best_i])
    } else {
        let max_v = visits.iter().copied().max().unwrap_or(1);
        let min_own_v = (max_v / 8).max(1);
        let active_rows: Vec<usize> = (0..n_my).filter(|&i| visits[i] >= min_own_v).collect();
        let max_ov = (0..n_op)
            .map(|j| root_node.visits[opp][j])
            .max()
            .unwrap_or(1);
        let min_opp_v = (max_ov / 16).max(1);
        let active_cols: Vec<usize> = (0..n_op)
            .filter(|&j| root_node.visits[opp][j] >= min_opp_v)
            .collect();

        if active_rows.len() == 1 {
            let mut pol = vec![0.0; n_my];
            pol[active_rows[0]] = 1.0;
            (pol, means[active_rows[0]])
        } else if active_cols.is_empty() {
            let best_i = active_rows
                .iter()
                .copied()
                .max_by(|&a, &b| means[a].total_cmp(&means[b]))
                .unwrap_or(0);
            let mut pol = vec![0.0; n_my];
            pol[best_i] = 1.0;
            (pol, means[best_i])
        } else {
            let nr = active_rows.len();
            let nc = active_cols.len();
            let mut mat = vec![0.5; nr * nc];
            let k_shrink = 4.0f64;
            for (ri, &i) in active_rows.iter().enumerate() {
                let row_prior = means[i];
                for (cj, &j) in active_cols.iter().enumerate() {
                    let (r0, r1) = if side == 0 { (i, j) } else { (j, i) };
                    let nij = joint_v[r0][r1] as f64;
                    let val = if nij > 0.0 {
                        let raw_q = joint_s[r0][r1] / nij;
                        let q = if side == 0 { raw_q } else { 1.0 - raw_q };
                        (nij * q + k_shrink * row_prior) / (nij + k_shrink)
                    } else {
                        row_prior
                    };
                    mat[ri * nc + cj] = val;
                }
            }
            let sol = solve_matrix_full(&mat, nr, nc);
            let mut pol = vec![0.0; n_my];
            let mut total_p = 0.0;
            for (ri, &i) in active_rows.iter().enumerate() {
                let p = sol.x[ri].max(0.0);
                pol[i] = p;
                total_p += p;
            }
            if total_p > 1e-9 {
                for p in &mut pol {
                    *p /= total_p;
                }
            } else {
                let best_i = active_rows[0];
                pol[best_i] = 1.0;
            }
            (pol, sol.value)
        }
    };

    TempoMctsResult {
        own_choices: own_choices.to_vec(),
        own_policy,
        visits,
        means,
        eq_win_prob,
        mean_depth: (depth_sum as f64) / (cfg.iterations.max(1) as f64),
        nodes: nodes.len(),
    }
}

/// Standalone zero-rollout DUCT SM-MCTS Agent over `TempoEval` (or controlled `eval.rs` leaf).
pub struct TempoMctsAgent {
    pub cfg: TempoMctsConfig,
    rng: SplitMix64,
}

impl TempoMctsAgent {
    pub fn new(cfg: TempoMctsConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: SplitMix64::new(seed),
        }
    }
}

impl Agent for TempoMctsAgent {
    fn name(&self) -> String {
        let tag = match self.cfg.leaf {
            TempoLeafMode::TempoFull => "full",
            TempoLeafMode::TempoRaw => "raw",
            TempoLeafMode::FastGolf => "v2",
            TempoLeafMode::HybridGolf => "hybrid",
            TempoLeafMode::Stage3Log => "s3log",
            TempoLeafMode::Stage3Mlp => "s3mlp",
            TempoLeafMode::AzValue => "azv",
            TempoLeafMode::CfValue => "cfv",
            TempoLeafMode::EvalStatic => "eval",
            TempoLeafMode::EvalBlindspot => "blindspot",
        };
        format!("tempo-mcts-{tag}:{}", self.cfg.iterations)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        if matches!(choices[0], SearchChoice::Team(_)) {
            if self.cfg.fixed_preview {
                let d = SearchChoice::Team([1, 2, 3]);
                return if choices.contains(&d) { d } else { choices[0] };
            }
            let preview_variant = match self.cfg.leaf {
                TempoLeafMode::TempoRaw => TempoVariant::Raw,
                _ => TempoVariant::Full,
            };
            return fast_tempo_team_preview(
                battle,
                dex,
                side,
                choices,
                preview_variant,
                matches!(self.cfg.leaf, TempoLeafMode::FastGolf | TempoLeafMode::HybridGolf),
            );
        }

        let seed = self.rng.next();
        let res = search_tempo_mcts(battle, dex, side, choices, self.cfg, seed);

        if self.cfg.deterministic {
            let best_i = res
                .own_policy
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            choices[best_i]
        } else {
            let mut r = self.rng.next_f64();
            for (i, &p) in res.own_policy.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    return choices[i];
                }
            }
            choices[choices.len() - 1]
        }
    }

    fn root_policy(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> Vec<f64> {
        if choices.len() == 1 || matches!(choices[0], SearchChoice::Team(_)) {
            let pick = self.choose(battle, dex, side, choices);
            return choices.iter().map(|&c| if c == pick { 1.0 } else { 0.0 }).collect();
        }
        let seed = self.rng.next();
        search_tempo_mcts(battle, dex, side, choices, self.cfg, seed).own_policy
    }
}

// =========================================================================
// Exact `SkuctSearch` Wrapper (`SkuctGolfAgent`) — Zero-Edit Control Bridge
// =========================================================================

#[derive(Clone, Copy, Debug)]
pub struct SkuctGolfConfig {
    pub iterations: u32,
    pub c: f64,
    pub leaf: TempoLeafMode,
    pub rollout_turns: u16,
    pub fixed_preview: bool,
}

impl Default for SkuctGolfConfig {
    fn default() -> Self {
        Self {
            iterations: 1000,
            c: 0.4,
            leaf: TempoLeafMode::FastGolf,
            rollout_turns: 0,
            fixed_preview: true,
        }
    }
}

/// Runs the exact shipped `SkuctSearch` engine (`BattleSearchCache` keying, `backed_up_value`
/// minimax backup, `endgame_root`, `dominated_reason`) via `step_with_leaf`, swapping ONLY the
/// leaf evaluator (`FastGolf` vs `EvalBlindspot`) and rollout depth (`0, 2, 4, 8` turns).
pub struct SkuctGolfAgent {
    pub cfg: SkuctGolfConfig,
    rng: SplitMix64,
}

impl SkuctGolfAgent {
    pub fn new(cfg: SkuctGolfConfig, seed: u64) -> Self {
        Self {
            cfg,
            rng: SplitMix64::new(seed),
        }
    }
}

impl Agent for SkuctGolfAgent {
    fn name(&self) -> String {
        let tag = match self.cfg.leaf {
            TempoLeafMode::FastGolf => "golf",
            TempoLeafMode::HybridGolf => "hybrid",
            TempoLeafMode::Stage3Log => "s3log",
            TempoLeafMode::Stage3Mlp => "s3mlp",
            TempoLeafMode::AzValue => "azv",
            TempoLeafMode::CfValue => "cfv",
            TempoLeafMode::EvalBlindspot => "bs",
            TempoLeafMode::EvalStatic => "eval",
            TempoLeafMode::TempoFull => "full",
            TempoLeafMode::TempoRaw => "raw",
        };
        format!("skuct-{tag}-{}r:{}", self.cfg.rollout_turns, self.cfg.iterations)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        if choices.len() == 1 {
            return choices[0];
        }
        if matches!(choices[0], SearchChoice::Team(_)) {
            if self.cfg.fixed_preview {
                let d = SearchChoice::Team([1, 2, 3]);
                return if choices.contains(&d) { d } else { choices[0] };
            }
            return fast_tempo_team_preview(
                battle,
                dex,
                side,
                choices,
                TempoVariant::Full,
                matches!(self.cfg.leaf, TempoLeafMode::FastGolf | TempoLeafMode::HybridGolf),
            );
        }

        let rm_cfg = RmConfig {
            iterations: self.cfg.iterations,
            c: self.cfg.c,
            rule: SelRule::Ucb,
            ..RmConfig::default()
        };
        let roster_cache = RosterCompactCache::from_battle(battle, dex);
        let (root_table, root_cb) =
            CompactBattle::from_battle_with_roster_cache(battle, dex, Some(&roster_cache));
        let fast_cache =
            if matches!(self.cfg.leaf, TempoLeafMode::FastGolf | TempoLeafMode::HybridGolf) {
                Some(TempoMatchCache::build(&root_cb, &root_table))
            } else {
                None
            };

        let seed = self.rng.next();
        let mut search = SkuctSearch::new(battle, dex, rm_cfg, seed);
        let leaf_mode = self.cfg.leaf;
        let rollout_turns = self.cfg.rollout_turns;
        search.step_with_leaf(dex, self.cfg.iterations, &mut |sim, rng, do_rollout| {
            let (tbl, mut cb) =
                CompactBattle::from_battle_with_roster_cache(sim, dex, Some(&roster_cache));
            let turns = if do_rollout { rollout_turns } else { 0 };
            let v = eval_duct_leaf_with_rollout(&mut cb, &tbl, leaf_mode, turns, fast_cache.as_ref(), rng);
            record_compact_sim_end(&cb, &tbl, v);
            v
        });
        search.best(side).unwrap_or(choices[0])
    }
}

#[inline]
fn compact_sim_end(cb: &CompactBattle, table: &MatchTable, val: f64) -> SimEndInfo {
    let mut info = SimEndInfo::from_compact(cb);
    if info.outcome == 0 && compact_turn_limit_outcome(cb, table).is_some() {
        info.outcome = if (val - 1.0).abs() < 1e-9 {
            1
        } else if val.abs() < 1e-9 {
            2
        } else {
            3
        };
    }
    info
}

#[inline]
fn record_compact_sim_end(cb: &CompactBattle, table: &MatchTable, val: f64) {
    set_last_sim_end(compact_sim_end(cb, table, val));
}

#[inline]
fn battle_world_sig(sim: &Battle) -> u64 {
    let mut h = 0x517C_C1B7_2722_0A95u64;
    for side in 0..2usize {
        for &r in &sim.sides[side].party {
            h = mix64(h, (side as u64) << 8 | r as u64);
        }
        for p in &sim.sides[side].roster {
            let w0 = (p.species.0 as u64)
                | ((p.level as u64) << 16)
                | ((p.maxhp.max(0) as u64) << 24)
                | ((p.hp_type.0 as u64) << 40)
                | ((p.hp_power as u64) << 48)
                | ((p.happiness as u64) << 56);
            h = mix64(h, w0);
            let item_id = p.item.map_or(0xFFFFu64, |id| id.0 as u64);
            let mut tmask = 0u64;
            for (ti, t) in p.types.iter().take(2).enumerate() {
                tmask |= (t.0 as u64) << (ti * 8);
            }
            let w1 = (p.hp.max(0) as u64)
                | (item_id << 16)
                | (tmask << 32)
                | ((p.move_slots.len() as u64) << 56);
            h = mix64(h, w1);
            let s0 = (p.stored_stats[0].max(0) as u64)
                | ((p.stored_stats[1].max(0) as u64) << 16)
                | ((p.stored_stats[2].max(0) as u64) << 32)
                | ((p.stored_stats[3].max(0) as u64) << 48);
            h = mix64(h, s0);
            let mut m0 = (p.stored_stats[4].max(0) as u64) << 48;
            for (mi, sl) in p.move_slots.iter().take(4).enumerate() {
                m0 ^= (sl.id.0 as u64) << ((mi as u32) * 12);
            }
            h = mix64(h, m0);
        }
    }
    h = mix64(h, sim.queue.len() as u64);
    h
}

/// Exact picked-3v3 `CompactWorld` signature: distinguishes every field affecting `CompactBattle`
/// dynamics (items, exact HP, PP, status/volatiles/boosts, pending mid-turn BP flags, canonicalized
/// party roster, `table.match_fingerprint`) while ignoring unpicked roster slots so identical 3v3
/// worlds share one tree and `RootTypeStats`.
#[inline]
fn compact_world_sig(cb: &CompactBattle, table: &MatchTable) -> u64 {
    let mut h = mix64(0x517C_C1B7_2722_0A95, table.match_fingerprint);
    let side_hdr = (cb.sides[0].side_flags as u64)
        | ((cb.sides[1].side_flags as u64) << 8)
        | ((cb.bp_switch_flags as u64) << 16)
        | ((cb.sides[0].party_roster[0] as u64) << 24)
        | ((cb.sides[0].party_roster[1] as u64) << 28)
        | ((cb.sides[0].party_roster[2] as u64) << 32)
        | ((cb.sides[1].party_roster[0] as u64) << 36)
        | ((cb.sides[1].party_roster[1] as u64) << 40)
        | ((cb.sides[1].party_roster[2] as u64) << 44)
        | ((cb.sides[0].request as u64) << 48)
        | ((cb.sides[1].request as u64) << 52);
    h = mix64(h, side_hdr);
    for i in 0..6usize {
        let m = &cb.mons[i];
        let w0 = (m.hp().max(0) as u64)
            | ((m.item as u64) << 16)
            | ((m.status_packed as u64) << 24)
            | ((m.vol_flags as u64) << 32)
            | ((m.ext_flags as u64) << 48);
        h = mix64(h, w0);
        let w1 = (m.pp[0] as u64)
            | ((m.pp[1] as u64) << 8)
            | ((m.pp[2] as u64) << 16)
            | ((m.pp[3] as u64) << 24)
            | ((m.slot_refs as u64) << 32)
            | ((m.rollout_rage as u64) << 40);
        h = mix64(h, w1);
    }
    h
}

// =========================================================================
// Production Imperfect-Info Wrapper (`BlindGolfAgent`) via `BlindSearch::step_with_leaf`
// =========================================================================

#[derive(Clone, Debug)]
pub struct BlindGolfConfig {
    pub iterations: u32,
    pub c: f64,
    pub leaf: TempoLeafMode,
    pub rollout_turns: u16,
    pub preview_iterations: u32,
    pub own_preview_triple: Option<[u8; 3]>,
}

impl Default for BlindGolfConfig {
    fn default() -> Self {
        Self {
            iterations: 9_000,
            c: 0.4,
            leaf: TempoLeafMode::FastGolf,
            rollout_turns: 2,
            preview_iterations: 27_000,
            own_preview_triple: None,
        }
    }
}

struct BlindGolfGameState {
    side: usize,
    last_turn: u16,
    local: LocalPlayer,
    observer: Observer,
    belief: Belief,
}

/// Runs the exact shipped `BlindSearch` stack (`LocalPlayer`/`Observer`/`Belief`, 24-world
/// subsampling, `mask.rs` public root pruning, `root_policy.rs` Bayesian LP, `endgame_root.rs`)
/// via `BlindSearch::step_with_leaf`, swapping ONLY the leaf evaluator (`FastGolf` + `0r/2r`).
pub struct BlindGolfAgent {
    pub cfg: BlindGolfConfig,
    pool: Arc<MetaPool>,
    rng: SplitMix64,
    game: Option<BlindGolfGameState>,
}

impl BlindGolfAgent {
    pub fn new(cfg: BlindGolfConfig, pool: Arc<MetaPool>, seed: u64) -> Self {
        Self {
            cfg,
            pool,
            rng: SplitMix64::new(seed),
            game: None,
        }
    }

    pub fn with_own_preview_triple(mut self, triple: [u8; 3]) -> Self {
        self.cfg.own_preview_triple = Some(triple);
        self
    }

    fn sync_game(&mut self, battle: &Battle, dex: &Dex, side: usize, is_preview: bool) {
        let stale = match &self.game {
            None => true,
            Some(g) => {
                g.side != side
                    || is_preview
                    || battle.turn < g.last_turn
                    || !g.local.is_compatible(battle)
            }
        };
        if stale {
            let own_sets = submitted_team(battle, dex, side);
            let (local, observer) =
                LocalPlayer::new_with_own_sets(battle, dex, side, own_sets).expect("local player");
            let belief =
                Belief::with_fallback_policy(dex, &self.pool, &observer, FallbackPolicy::default());
            self.game = Some(BlindGolfGameState {
                side,
                last_turn: battle.turn,
                local,
                observer,
                belief,
            });
        }
        let g = self.game.as_mut().unwrap();
        g.local
            .observe(battle, dex, &mut g.observer)
            .expect("blind golf observe failed");
        g.belief.sync(dex, &g.observer);
        g.last_turn = battle.turn;
    }
}

impl Agent for BlindGolfAgent {
    fn requires_player_log(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        let tag = match self.cfg.leaf {
            TempoLeafMode::FastGolf => "golf",
            TempoLeafMode::HybridGolf => "hybrid",
            TempoLeafMode::Stage3Log => "s3log",
            TempoLeafMode::Stage3Mlp => "s3mlp",
            TempoLeafMode::AzValue => "azv",
            TempoLeafMode::CfValue => "cfv",
            TempoLeafMode::EvalBlindspot => "bs",
            TempoLeafMode::EvalStatic => "eval",
            TempoLeafMode::TempoFull => "full",
            TempoLeafMode::TempoRaw => "raw",
        };
        format!("blind-{tag}-{}r:{}", self.cfg.rollout_turns, self.cfg.iterations)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices.first(), Some(SearchChoice::Team(_)));
        self.sync_game(battle, dex, side, is_preview);
        if choices.len() == 1 {
            return choices[0];
        }
        if is_preview {
            if let Some(triple) = self.cfg.own_preview_triple {
                let _ = self.rng.next_f64();
                let pick = SearchChoice::Team(triple);
                return if choices.contains(&pick) { pick } else { choices[0] };
            }
        }

        let g = self.game.as_mut().unwrap();
        let pos = g
            .local
            .position(dex, &mut g.belief, &g.observer, &mut self.rng)
            .expect("blind golf position reconstruction failed");
        let has_prior = g.belief.has_preview_prior(&g.observer);
        let iters = if is_preview {
            self.cfg.preview_iterations
        } else {
            self.cfg.iterations
        };
        let rm_cfg = RmConfig {
            iterations: iters,
            c: self.cfg.c,
            rule: SelRule::Ucb,
            ..RmConfig::default()
        };
        let mut bs = BlindSearch::from_observation(&pos, dex, rm_cfg, side, self.rng.0);
        bs.prune_preview_aliases(dex);
        if has_prior {
            bs.set_opponent_preview_prior(true);
        }
        debug_assert_eq!(bs.actions(), choices);

        let leaf_mode = self.cfg.leaf;
        let rollout_turns = self.cfg.rollout_turns;
        let use_fast = matches!(leaf_mode, TempoLeafMode::FastGolf | TempoLeafMode::HybridGolf);
        // Per-world static cache (`<= 24` determinized worlds per decision): zero cross-world bleed!
        let mut world_caches: FxHashMap<u64, (RosterCompactCache, Option<TempoMatchCache>)> =
            FxHashMap::default();
        bs.step_with_leaf(dex, &g.belief, &g.observer, iters, &mut |sim, rng, do_rollout| {
            let sig = battle_world_sig(sim);
            let (rc, tc_opt) = world_caches.entry(sig).or_insert_with(|| {
                let rc = RosterCompactCache::from_battle(sim, dex);
                let tc = if use_fast {
                    let (t0, cb0) = CompactBattle::from_battle_with_roster_cache(sim, dex, Some(&rc));
                    Some(TempoMatchCache::build(&cb0, &t0))
                } else {
                    None
                };
                (rc, tc)
            });
            let (tbl, mut cb) = CompactBattle::from_battle_with_roster_cache(sim, dex, Some(rc));
            let turns = if do_rollout { rollout_turns } else { 0 };
            let v = eval_duct_leaf_with_rollout(&mut cb, &tbl, leaf_mode, turns, tc_opt.as_ref(), rng);
            record_compact_sim_end(&cb, &tbl, v);
            v
        });
        let pick = bs.best().unwrap_or(choices[0]);
        self.rng = SplitMix64::new(bs.search_rng_state());
        pick
    }
}

// =========================================================================
// CompactBattle-Native Imperfect-Information Search (`BlindCmctsAgent`)
// =========================================================================

#[inline]
fn draw_rem_uniform(rng: &mut SplitMix64, min_d: u8, max_d: u8, elapsed: u8) -> u8 {
    let lo = min_d.max(elapsed.saturating_add(1));
    if lo >= max_d {
        return 1;
    }
    let span = (max_d - lo + 1) as usize;
    let d = lo + rng.below(span) as u8;
    (d - elapsed).max(1)
}

/// Resample hidden runtime timers directly on a 256-byte `CompactBattle` (~10 ns, zero `Battle` touch).
#[inline]
fn resample_compact_hidden_timers(
    cb: &mut CompactBattle,
    table: &MatchTable,
    specs: &[HiddenTimerSpec],
    rng: &mut SplitMix64,
) {
    let is_castle = table.ruleset.is_castle();
    for &spec in specs {
        let side = spec.side as usize;
        if side >= 2 {
            continue;
        }
        let Some(p_slot) = cb.sides[side]
            .party_roster
            .iter()
            .position(|&r| r == spec.slot)
        else {
            continue;
        };
        if (cb.sides[side].alive_mask & (1 << p_slot)) == 0 {
            continue;
        }
        let mon = &mut cb.mons[side * 3 + p_slot];
        if mon.hp() <= 0 {
            continue;
        }
        match spec.kind {
            HiddenTimerKind::Sleep { slept } => {
                if mon.status_raw() == Status::Slp as u8 {
                    let max_d = if is_castle { 5 } else { 4 };
                    let rem = draw_rem_uniform(rng, 2, max_d, slept);
                    mon.set_status(Status::Slp as u8, rem);
                }
            }
            HiddenTimerKind::Confusion { ticks } => {
                if (mon.vol_flags & vol_bit::CONFUSION) != 0 {
                    mon.timers[0] = draw_rem_uniform(rng, 2, 5, ticks);
                }
            }
            HiddenTimerKind::Encore { upkeeps } => {
                if (mon.vol_flags & vol_bit::ENCORE) != 0 {
                    mon.timers[1] = draw_rem_uniform(rng, 3, 6, upkeeps);
                }
            }
            HiddenTimerKind::Disable { upkeeps, acted_on_start } => {
                if (mon.vol_flags & vol_bit::DISABLE) != 0 {
                    let bonus = u8::from(acted_on_start);
                    let (min_b, max_b) = if is_castle { (1, 7) } else { (2, 5) };
                    mon.timers[2] = draw_rem_uniform(rng, min_b + bonus, max_b + bonus, upkeeps);
                }
            }
            HiddenTimerKind::PartiallyTrapped { upkeeps } => {
                if mon.partially_trapped() {
                    mon.ext_timers[1] = draw_rem_uniform(rng, 3, 5, upkeeps);
                }
            }
            HiddenTimerKind::LockedMove { upkeeps } => {
                if (mon.ext_flags & ext_vol_bit::LOCKED_OUTRAGE) != 0 {
                    mon.ext_timers[0] = draw_rem_uniform(rng, 2, 3, upkeeps);
                }
            }
            HiddenTimerKind::Bide { .. } => {}
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ReplyPoolKey {
    Move(nc2000_engine::dex::MoveId),
    SwitchSpecies(nc2000_engine::dex::SpeciesId),
    Other(SearchChoice),
}

#[inline]
fn reply_pool_key(sim: &Battle, opp: usize, choice: SearchChoice) -> ReplyPoolKey {
    match choice {
        SearchChoice::Move(id) => ReplyPoolKey::Move(id),
        SearchChoice::Switch(pos) => sim.sides[opp]
            .party
            .get(pos.saturating_sub(1) as usize)
            .and_then(|&slot| sim.sides[opp].roster.get(slot as usize))
            .map_or(ReplyPoolKey::Other(choice), |p| ReplyPoolKey::SwitchSpecies(p.species)),
        other => ReplyPoolKey::Other(other),
    }
}

/// Deterministic forced-switch / mid-turn Baton Pass resolver for the 1-ply prior sweep and `0r` leaves.
#[inline]
pub(crate) fn resolve_compact_forced_switches(
    cb: &mut CompactBattle,
    table: &MatchTable,
    weights: &EvalWeights,
) {
    for _ in 0..4usize {
        if cb.outcome != 0
            || (cb.sides[0].request != 2 && cb.sides[1].request != 2)
            || compact_turn_limit_outcome(cb, table).is_some()
        {
            break;
        }
        let mut picks = [7u8; 2];
        for s in 0..2usize {
            if cb.sides[s].request != 2 {
                continue;
            }
            let mut cs = [0u8; 7];
            let n = cb.legal_actions(table, s, &mut cs);
            if n == 0 {
                continue;
            }
            if n == 1 {
                picks[s] = cs[0];
                continue;
            }
            let def_side = 1 - s;
            let def_slot = cb.sides[def_side].active_idx as usize;
            let foe_alive = (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
                && cb.mons[def_side * 3 + def_slot].hp() > 0;
            let att_slot = cb.sides[s].active_idx as usize;
            let att_idx = s * 3 + att_slot;
            let passing_alive = att_slot < 3
                && cb.mons[att_idx].hp() > 0
                && foe_alive
                && (cb.mons[att_idx].boosts[0] > 0
                    || cb.mons[att_idx].boosts[1] > 0
                    || cb.mons[att_idx].boosts[2] > 0
                    || cb.mons[att_idx].boosts[3] > 0
                    || cb.mons[att_idx].boosts[4] > 0
                    || cb.mons[att_idx].boosts[6] > 0
                    || (cb.mons[att_idx].vol_flags & vol_bit::SUBSTITUTE) != 0);
            if passing_alive {
                if let Some(sw) = crate::relay::best_midturn_bp_switch(cb, table, weights, s, &cs[..n])
                {
                    picks[s] = sw;
                    continue;
                }
            }
            let mut best_sw = cs[0];
            let mut best_sc = f64::NEG_INFINITY;
            for &c in &cs[..n] {
                if (4..7).contains(&c) {
                    let sl = (c - 4) as usize;
                    let sc = if foe_alive {
                        cb.best_hit_fraction(table, s, sl, def_side, def_slot).min(1.5)
                            - cb.best_hit_fraction(table, def_side, def_slot, s, sl).min(1.5)
                    } else {
                        (cb.mons[s * 3 + sl].hp() as f64)
                            / (table.mons[s * 3 + sl].maxhp.max(1) as f64)
                    };
                    if sc > best_sc {
                        best_sc = sc;
                        best_sw = c;
                    }
                }
            }
            picks[s] = best_sw;
        }
        crate::mcts::apply_compact_actions_with_bp_target(cb, table, picks);
    }
}

struct CompactWorld {
    table: MatchTable,
    cb: CompactBattle,
    cache: Option<TempoMatchCache>,
    sim: Option<Box<Battle>>,
    root_node_id: usize,
    type_index: usize,
}

struct CompactRootTypeStats {
    tkey: u64,
    samples: u32,
    late_samples: u32,
    replies: usize,
    reply_keys: Vec<ReplyPoolKey>,
    cells: Vec<(u32, f64)>,
    late_cells: Vec<(u32, f64)>,
    col_visits: Vec<u32>,
    rep_world_idx: usize,
    world_count: u32,
    prior_cells: Vec<f64>,
    cal_prior_buf: Vec<f64>,
    prior_valid: bool,
    prior_delta: f64,
    opp_eq_policy: Vec<f64>,
    active_cols: Vec<usize>,
    search_v: Vec<f64>,
    type_eq_value: f64,
}

/// Direct policy-evaluated opponent root UCB for Stage 2A (`depth == 0`, `replies > 1`).
#[inline]
fn select_stage2_opp_reply(
    stats: &CompactRootTypeStats,
    p_eval_sparse: &[(usize, f64)],
    c_ucb: f64,
) -> usize {
    let m = stats.replies;
    if m <= 1 {
        return 0;
    }
    // Untried pass: evaluate all unseen replies in descending 1-ply opponent payoff order.
    let mut best_untried: Option<(usize, f64)> = None;
    for j in 0..m {
        if stats.col_visits[j] == 0 {
            let mut v_opp = 0.5;
            if stats.prior_valid && !p_eval_sparse.is_empty() {
                let mut acc = 0.0;
                let mut p_sum = 0.0;
                for &(i, p_i) in p_eval_sparse {
                    acc += p_i * (1.0 - stats.prior_cells[i * m + j]);
                    p_sum += p_i;
                }
                if p_sum > 1e-9 {
                    v_opp = acc / p_sum;
                }
            }
            if best_untried.map_or(true, |(_, bv)| v_opp > bv) {
                best_untried = Some((j, v_opp));
            }
        }
    }
    if let Some((j, _)) = best_untried {
        return j;
    }

    let (src, n_t) = if stats.late_samples >= 64 {
        (&stats.late_cells, stats.late_samples)
    } else {
        (&stats.cells, stats.samples)
    };
    let ln_nt = (f64::from(n_t) + 1.0).ln().max(0.0);
    const KAPPA_REP: f64 = 4.0;
    let use_cal = stats.prior_valid && stats.cal_prior_buf.len() == stats.cells.len();
    let use_raw = stats.prior_valid && stats.prior_cells.len() == stats.cells.len();

    let mut best_j = 0usize;
    let mut best_u = f64::NEG_INFINITY;
    for j in 0..m {
        let mut v_opp = 0.0f64;
        let mut inv_neff = 0.0f64;
        let mut p_sum = 0.0f64;
        for &(i, p_i) in p_eval_sparse {
            let idx = i * m + j;
            let (n_ij, w_ij) = src[idx];
            let p_prior = if use_cal {
                stats.cal_prior_buf[idx]
            } else if use_raw {
                stats.prior_cells[idx]
            } else {
                0.5
            };
            let mu_ij = (w_ij + KAPPA_REP * p_prior) / (f64::from(n_ij) + KAPPA_REP);
            v_opp += p_i * (1.0 - mu_ij);
            inv_neff += (p_i * p_i) / (f64::from(n_ij) + 1.0);
            p_sum += p_i;
        }
        if p_sum > 1e-9 {
            v_opp /= p_sum;
            inv_neff /= p_sum * p_sum;
        } else {
            v_opp = 0.5;
            inv_neff = 1.0;
        }
        let n_eff = 1.0 / inv_neff.max(1e-9);
        let u = v_opp + c_ucb * (ln_nt / n_eff).sqrt();
        if u > best_u {
            best_u = u;
            best_j = j;
        }
    }
    best_j
}

#[derive(Clone, Debug)]
pub struct BlindCmctsConfig {
    pub iterations: u32,
    pub c: f64,
    pub max_depth: u8,
    pub leaf: TempoLeafMode,
    pub rollout_turns: u16,
    pub max_worlds: usize,
    pub preview_iterations: u32,
    pub own_preview_triple: Option<[u8; 3]>,
    /// Mirror shipped `AdaptiveBattleBudget`: check early stop every `adaptive_interval` iters.
    pub adaptive: bool,
    pub adaptive_interval: u32,
    pub adaptive_min_gap: f64,
    /// Enable Stage 2A equilibrium-targeted root architecture (1-ply prior sweep, direct
    /// policy-evaluated opponent root UCB, gap-gated threat cross-sampling, semantic pooling,
    /// and capped Stage-2 Bayesian LP).
    pub stage2_root: bool,
    /// Pool `Switch(pos)` opponent root priors by target `SpeciesId` (semantic pooling) without
    /// altering opponent root UCB.
    pub sempool: bool,
    /// Exponentially halve `late_cells` at power-of-2 milestones (`>= 1024`) so the readout LP
    /// reads mature-tree returns rather than early shallow iterations.
    pub mature_window: bool,
    /// On contested turns (when `adaptive` does not early-stop Tree 0, or on fixed budget when >1),
    /// aggregate root stats across `K` independent full-depth trees (`1` = single tree).
    pub contested_trees: u8,
    /// Post-fit logit temperature `T` applied to `Stage3Log`/`Stage3Mlp`/`HybridGolf`/`AzValue` (`1.0` = raw).
    pub leaf_temp: f64,
    /// Enable interior Simultaneous-Move PUCT (`depth >= 1`) with sound `compact_interior_noop_mask`
    /// and lazy species-free `P_net` policy prior (root remains 100% shipped UCB1 + Bayesian LP).
    pub use_puct: bool,
    /// Interior PUCT exploration constant `c_puct` (scaled for `Q ∈ [0, 1]`, default `1.0`).
    pub c_puct: f64,
    /// Interior PUCT First-Play Urgency reduction below node's running mean (`r_bar_s - fpu_red`).
    pub fpu_red: f64,
    /// Uniform floor `eps_p` mixed into `P_net` over safe unmasked interior actions (default `0.02`).
    pub puct_eps: f32,
    /// Empirical-Bayes minimax backup contraction cap (`0.85` = shipped, `0.0` = pure MC backup).
    pub backup_lambda: f64,
    /// Empirical-Bayes minimax backup pseudo-count (`1.5` = shipped, `4.0` = soft backup for `0r`).
    pub backup_tau: f64,
    /// Deterministic `EndgameSession` (`BoundSolver`) `decision_runs` cap on `n0 + n1 <= 3` (`0` = OFF).
    pub endgame_runs: u32,
    /// Post-warmup contested-turn counterfactual Top-K subgame verification iterations (`0` = OFF).
    pub verify_iters: u32,
    /// Use Counterfactual-trained `CF-P_net` instead of Gen-1 `P_net` for interior PUCT priors.
    pub use_cf_policy: bool,
}

impl Default for BlindCmctsConfig {
    fn default() -> Self {
        Self {
            iterations: 36_000,
            c: 0.4,
            max_depth: 250,
            leaf: TempoLeafMode::FastGolf,
            rollout_turns: 0,
            max_worlds: 24,
            preview_iterations: 27_000,
            own_preview_triple: None,
            adaptive: false,
            adaptive_interval: 13_500,
            adaptive_min_gap: 0.01,
            stage2_root: false,
            sempool: false,
            mature_window: false,
            contested_trees: 1,
            leaf_temp: 1.0,
            use_puct: false,
            c_puct: 1.0,
            fpu_red: 0.09,
            puct_eps: 0.02,
            backup_lambda: 0.85,
            backup_tau: 1.5,
            endgame_runs: 0,
            verify_iters: 0,
            use_cf_policy: false,
        }
    }
}

/// `CompactBattle`-native Imperfect-Information Minimax-UCT Search reusing the full production
/// observation/belief/mask/Bayesian-root-LP/endgame-root stack (`LocalPlayer`, `Observer`, `Belief`,
/// `public_noop_reason`, `EndgameRootStats`, `solve_bayesian_equilibrium_with_priors`,
/// `select_blind_root_choice`) while executing the entire search tree in 256-byte `CompactBattle`.
pub struct BlindCmctsAgent {
    pub cfg: BlindCmctsConfig,
    pool: Arc<MetaPool>,
    rng: SplitMix64,
    game: Option<BlindGolfGameState>,
    nodes_buf: Vec<DuctNode>,
    index_buf: FxHashMap<u64, usize>,
}

impl BlindCmctsAgent {
    pub fn new(cfg: BlindCmctsConfig, pool: Arc<MetaPool>, seed: u64) -> Self {
        Self {
            cfg,
            pool,
            rng: SplitMix64::new(seed),
            game: None,
            nodes_buf: Vec::new(),
            index_buf: FxHashMap::default(),
        }
    }

    pub fn with_own_preview_triple(mut self, triple: [u8; 3]) -> Self {
        self.cfg.own_preview_triple = Some(triple);
        self
    }

    fn sync_game(&mut self, battle: &Battle, dex: &Dex, side: usize, is_preview: bool) {
        let stale = match &self.game {
            None => true,
            Some(g) => {
                g.side != side
                    || is_preview
                    || battle.turn < g.last_turn
                    || !g.local.is_compatible(battle)
            }
        };
        if stale {
            let own_sets = submitted_team(battle, dex, side);
            let (local, observer) =
                LocalPlayer::new_with_own_sets(battle, dex, side, own_sets).expect("local player");
            let belief =
                Belief::with_fallback_policy(dex, &self.pool, &observer, FallbackPolicy::default());
            self.game = Some(BlindGolfGameState {
                side,
                last_turn: battle.turn,
                local,
                observer,
                belief,
            });
        }
        let g = self.game.as_mut().unwrap();
        g.local
            .observe(battle, dex, &mut g.observer)
            .expect("blind cmcts observe failed");
        g.belief.sync(dex, &g.observer);
        g.last_turn = battle.turn;
    }
}

impl Agent for BlindCmctsAgent {
    fn requires_player_log(&self) -> bool {
        true
    }

    fn name(&self) -> String {
        let tag = match self.cfg.leaf {
            TempoLeafMode::FastGolf => "golf",
            TempoLeafMode::HybridGolf => "hybrid",
            TempoLeafMode::Stage3Log => "s3log",
            TempoLeafMode::Stage3Mlp => "s3mlp",
            TempoLeafMode::AzValue => "azv",
            TempoLeafMode::CfValue => "cfv",
            TempoLeafMode::EvalBlindspot => "bs",
            TempoLeafMode::EvalStatic => "eval",
            TempoLeafMode::TempoFull => "full",
            TempoLeafMode::TempoRaw => "raw",
        };
        let eng = if self.cfg.stage2_root { "bcmcts2" } else { "bcmcts" };
        let pfx = if self.cfg.adaptive { "a" } else { "" };
        format!("{eng}-{tag}-{}r:{pfx}{}", self.cfg.rollout_turns, self.cfg.iterations)
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let is_preview = matches!(choices.first(), Some(SearchChoice::Team(_)));
        self.sync_game(battle, dex, side, is_preview);
        if choices.len() == 1 {
            return choices[0];
        }
        if is_preview {
            if let Some(triple) = self.cfg.own_preview_triple {
                let _ = self.rng.next_f64();
                let pick = SearchChoice::Team(triple);
                return if choices.contains(&pick) { pick } else { choices[0] };
            }
            // Fallback to shipped BlindSearch preview if no shared preview triple was supplied.
            let g = self.game.as_mut().unwrap();
            let pos = g
                .local
                .position(dex, &mut g.belief, &g.observer, &mut self.rng)
                .expect("preview position reconstruction failed");
            let has_prior = g.belief.has_preview_prior(&g.observer);
            let rm_cfg = RmConfig {
                iterations: self.cfg.preview_iterations,
                c: self.cfg.c,
                rule: SelRule::Ucb,
                ..RmConfig::default()
            };
            let mut bs = BlindSearch::from_observation(&pos, dex, rm_cfg, side, self.rng.0);
            bs.prune_preview_aliases(dex);
            if has_prior {
                bs.set_opponent_preview_prior(true);
            }
            bs.step(dex, &g.belief, &g.observer, self.cfg.preview_iterations);
            let pick = bs.best().unwrap_or(choices[0]);
            self.rng = SplitMix64::new(bs.search_rng_state());
            return pick;
        }

        let g = self.game.as_mut().unwrap();
        let pos = g
            .local
            .position(dex, &mut g.belief, &g.observer, &mut self.rng)
            .expect("blind cmcts position reconstruction failed");
        let hidden_timers = pos.hidden_timers().to_vec();
        let mut base = pos.model().clone();
        base.set_log_enabled(false);
        let base_idle = base.is_idle_move_request();
        if base_idle {
            base.quick_claw_rolls = [false, false];
            base.update_all_speeds(dex);
        }
        let my_acts = base.legal_choices(dex, side);
        debug_assert_eq!(my_acts.as_slice(), choices);
        let n_my = my_acts.len();
        if n_my <= 1 {
            return my_acts.first().copied().unwrap_or(choices[0]);
        }

        let opp = 1 - side;
        let my_dominated: Vec<bool> = my_acts
            .iter()
            .map(|&c| public_noop_reason(&base, dex, side, c).is_some())
            .collect();
        let has_safe = my_dominated.iter().any(|&d| !d);
        let eligible_flags: Vec<bool> = (0..n_my).map(|i| !has_safe || !my_dominated[i]).collect();
        if eligible_flags.iter().filter(|&&e| e).count() == 1 {
            let only = eligible_flags.iter().position(|&e| e).unwrap();
            return my_acts[only];
        }

        // Exact fallback to shipped `BlindSearch` whenever the public root board uses a mechanic
        // outside `CompactBattle`'s supported subset (e.g. `Solar Beam`, `Bide`, mid-turn `Counter`/
        // `Mirror Coat`, `Transform`, or `SleepClauseMod`), matching `blind.rs::try_build_compact_world`.
        let has_bide = hidden_timers
            .iter()
            .any(|t| matches!(t.kind, HiddenTimerKind::Bide { .. }));
        if has_bide || !nc2000_engine::compact::is_compact_tree_supported(&base, dex) {
            let rm_cfg = RmConfig {
                iterations: self.cfg.iterations,
                c: self.cfg.c,
                rule: SelRule::Ucb,
                ..RmConfig::default()
            };
            let mut bs = BlindSearch::from_observation(&pos, dex, rm_cfg, side, self.rng.0);
            if self.cfg.adaptive {
                let mut budget = crate::adaptive_budget::AdaptiveBattleBudget::default();
                budget.step(&mut bs, dex, &g.belief, &g.observer, self.cfg.iterations);
            } else {
                bs.step(dex, &g.belief, &g.observer, self.cfg.iterations);
            }
            let pick = bs.best().unwrap_or(my_acts[0]);
            self.rng = SplitMix64::new(bs.search_rng_state());
            return pick;
        }

        let stage2_root = self.cfg.stage2_root;
        let use_sempool = stage2_root || self.cfg.sempool;
        let mature_window = self.cfg.mature_window;
        let leaf_temp = self.cfg.leaf_temp;
        let max_trees = self.cfg.contested_trees.max(1) as usize;
        let decision_seed = self.rng.0 ^ 0xA24B_AED4_963E_E407;
        let mut timer_rng = SplitMix64::new(decision_seed ^ 0x714E_525F_726E_6701);
        let obs_rev = g.observer.revision();
        let target_k = self.cfg.max_worlds.max(1).min(self.cfg.iterations.max(1) as usize);

        let use_puct = self.cfg.use_puct;
        let c_puct = self.cfg.c_puct;
        let fpu_red = self.cfg.fpu_red;
        let puct_eps = self.cfg.puct_eps;
        let backup_lambda = self.cfg.backup_lambda;
        let backup_tau = self.cfg.backup_tau;
        let eg_decision_runs = self.cfg.endgame_runs as usize;
        let verify_iters = self.cfg.verify_iters;
        let use_cf_policy = self.cfg.use_cf_policy;
        let mut eg_session = if eg_decision_runs > 0 {
            let eg_cfg = crate::endgame::EndgameConfig {
                solver_fraction: 0.2,
                strategy_loss: 0.02,
                leaf_width: 0.02,
                max_alive: 2,
                max_nodes: 4_000,
                slice_runs: 64,
                state_runs: 512,
                decision_runs: eg_decision_runs,
            };
            crate::endgame::EndgameSession::new(eg_cfg, 0.0, 60_000.0).ok()
        } else {
            None
        };
        let mut eg_battle_buf: Option<Box<Battle>> = None;
        let use_fast = use_puct
            || verify_iters > 0
            || matches!(
                self.cfg.leaf,
                TempoLeafMode::FastGolf
                    | TempoLeafMode::HybridGolf
                    | TempoLeafMode::Stage3Log
                    | TempoLeafMode::Stage3Mlp
                    | TempoLeafMode::AzValue
                    | TempoLeafMode::CfValue
            );
        let target_cap = if self.cfg.adaptive {
            (self.cfg.adaptive_interval.saturating_mul(2) as usize).clamp(64, 32_768)
        } else {
            (self.cfg.iterations as usize / 2).clamp(64, 131_072)
        };
        let mut nodes = std::mem::take(&mut self.nodes_buf);
        let mut index = std::mem::take(&mut self.index_buf);

        let mut endgame = EndgameRootStats::new(&base, dex, side, &my_acts);
        let mut agg_my_n = vec![0u32; n_my];
        let mut agg_my_w = vec![0.0f64; n_my];
        let mut agg_ord_n = vec![0u32; n_my];
        let mut agg_ord_w = vec![0.0f64; n_my];
        let mut agg_root_types: Vec<CompactRootTypeStats> = Vec::new();
        let mut agg_root_type_index: FxHashMap<u64, usize> = FxHashMap::default();
        let mut agg_joint: Vec<(usize, ReplyPoolKey, u32, f64)> = Vec::new();
        let mut agg_joint_index: FxHashMap<(usize, ReplyPoolKey), usize> = FxHashMap::default();
        let mut total_done = 0u32;
        let mut chosen_opt: Option<SearchChoice> = None;

        for tree_k in 0..max_trees {
            let tree_mix = (tree_k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut plan_rng = SplitMix64::new(
                (decision_seed ^ tree_mix)
                    ^ obs_rev.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    ^ (target_k as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9),
            );
            let team_picks = g.belief.root_sampling_plan(target_k, &mut plan_rng);

            nodes.clear();
            if nodes.capacity() < target_cap {
                nodes.reserve(target_cap);
            }
            index.clear();
            if index.capacity() < target_cap {
                index.reserve(target_cap);
            }

            let mut worlds: Vec<CompactWorld> = Vec::with_capacity(target_k);
            let mut root_types: Vec<CompactRootTypeStats> = Vec::new();
            let mut root_type_index: FxHashMap<u64, usize> = FxHashMap::default();
            let mut world_root_index: FxHashMap<u64, usize> = FxHashMap::default();

            for plan_pick in team_picks {
                let mut sim = base.clone();
                g.belief.determinize_in_place_planned(
                    dex,
                    &mut sim,
                    &g.observer,
                    plan_pick,
                    &mut self.rng,
                );
                if base_idle && (sim.quick_claw_rolls[0] || sim.quick_claw_rolls[1]) {
                    sim.quick_claw_rolls = [false, false];
                    if let Some(a) = sim.active_id(opp) {
                        if !sim.poke(a).fainted {
                            sim.update_speed(dex, a);
                        }
                    }
                }
                // Collapse redundant `[lead, A, B]` / `[lead, B, A]` unseen bench permutations (parity with `blind.rs`).
                crate::blind::canonicalize_unseen_bench(dex, &mut sim, opp);
                // Per-world parity guard: if an imputed hidden bench/moveset contains a move outside
                // `CompactBattle`'s supported subset (e.g. hidden Solar Beam / Transform), delegate to
                // shipped `BlindSearch` just like `blind.rs::try_build_compact_world`.
                if !nc2000_engine::compact::is_compact_tree_supported(&sim, dex) {
                    self.nodes_buf = nodes;
                    self.index_buf = index;
                    let rm_cfg = RmConfig {
                        iterations: self.cfg.iterations,
                        c: self.cfg.c,
                        rule: SelRule::Ucb,
                        ..RmConfig::default()
                    };
                    let mut bs = BlindSearch::from_observation(&pos, dex, rm_cfg, side, self.rng.0);
                    if self.cfg.adaptive {
                        let mut budget = crate::adaptive_budget::AdaptiveBattleBudget::default();
                        budget.step(&mut bs, dex, &g.belief, &g.observer, self.cfg.iterations);
                    } else {
                        bs.step(dex, &g.belief, &g.observer, self.cfg.iterations);
                    }
                    let pick = bs.best().unwrap_or(my_acts[0]);
                    self.rng = SplitMix64::new(bs.search_rng_state());
                    return pick;
                }
                let (table, mut cb) = CompactBattle::from_battle(&sim, dex);
                if base_idle {
                    cb.quick_claw_roll = 0;
                }
                let rkey =
                    cb.search_key(&table, 16, base_idle || !cb.quick_claw_observable(&table));
                let tkey = cb.search_key(&table, 0, true);
                let opp_choices = sim.legal_choices(dex, opp);

                let mut encoded_opp: Vec<(SearchChoice, u8)> =
                    Vec::with_capacity(opp_choices.len().min(6));
                for &c in &opp_choices {
                    if encoded_opp.len() < 6 {
                        if let Some(code) = encode_choice(&table, &cb, dex, opp, c) {
                            encoded_opp.push((c, code));
                        }
                    }
                }
                let root_node_id = *world_root_index.entry(rkey).or_insert_with(|| {
                    let mut rnode = DuctNode::new(&cb, &table);
                    let mut own_codes = [0u8; 6];
                    let mut n_own = 0usize;
                    for &c in &my_acts {
                        if n_own < 6 {
                            if let Some(code) = encode_choice(&table, &cb, dex, side, c) {
                                own_codes[n_own] = code;
                                n_own += 1;
                            }
                        }
                    }
                    debug_assert_eq!(n_own, n_my.min(6));
                    rnode.n_acts[side] = n_own as u8;
                    rnode.acts[side] = own_codes;

                    if !encoded_opp.is_empty() {
                        let mut opp_codes = [0u8; 6];
                        for (j, &(_, code)) in encoded_opp.iter().enumerate() {
                            opp_codes[j] = code;
                        }
                        rnode.n_acts[opp] = encoded_opp.len() as u8;
                        rnode.acts[opp] = opp_codes;
                    }
                    let nid = nodes.len();
                    nodes.push(rnode);
                    index.insert(rkey, nid);
                    nid
                });

                let rep_world_idx = worlds.len();
                let type_index = *root_type_index.entry(tkey).or_insert_with(|| {
                    let replies = if cb.sides[opp].request == 0 {
                        0
                    } else {
                        encoded_opp.len()
                    };
                    let reply_keys: Vec<ReplyPoolKey> = encoded_opp
                        .iter()
                        .take(replies)
                        .map(|&(c, _)| {
                            if use_sempool {
                                reply_pool_key(&sim, opp, c)
                            } else {
                                ReplyPoolKey::Other(c)
                            }
                        })
                        .collect();
                    let ti = root_types.len();
                    root_types.push(CompactRootTypeStats {
                        tkey,
                        samples: 0,
                        late_samples: 0,
                        replies,
                        reply_keys,
                        cells: vec![(0, 0.0); n_my * replies],
                        late_cells: vec![(0, 0.0); n_my * replies],
                        col_visits: vec![0; replies],
                        rep_world_idx,
                        world_count: 0,
                        prior_cells: vec![0.5; n_my * replies.max(1)],
                        cal_prior_buf: Vec::new(),
                        prior_valid: false,
                        prior_delta: 0.0,
                        opp_eq_policy: vec![0.0; replies],
                        active_cols: Vec::new(),
                        search_v: vec![0.5; n_my * replies],
                        type_eq_value: 0.5,
                    });
                    ti
                });
                root_types[type_index].world_count += 1;

                let cache = if use_fast {
                    Some(TempoMatchCache::build(&cb, &table))
                } else {
                    None
                };
                let sim_box = if eg_decision_runs > 0 {
                    sim.listener_pool = Default::default();
                    Some(Box::new(sim))
                } else {
                    None
                };
                worlds.push(CompactWorld {
                    table,
                    cb,
                    cache,
                    sim: sim_box,
                    root_node_id,
                    type_index,
                });
            }

            // Deterministic single-world `n0 + n1 <= 3` root EndgameSession strategy probe.
            if tree_k == 0
                && hidden_timers.is_empty()
                && base.turn > 0
                && root_types.len() == 1
                && !worlds.is_empty()
            {
                if let Some(sess) = eg_session.as_mut() {
                    let w0 = &mut worlds[0];
                    let n_tot0 = w0.cb.sides[0].alive_mask.count_ones()
                        + w0.cb.sides[1].alive_mask.count_ones();
                    if n_tot0 <= 3
                        && sess.stats.runs < eg_decision_runs
                        && sess.eligible_compact(&w0.cb, &w0.table)
                    {
                        if let Some(w0_sim) = w0.sim.as_mut() {
                            std::mem::swap(&mut w0_sim.listener_pool, &mut base.listener_pool);
                            let cert = sess.probe_strategy(dex, w0_sim, side, 32, &mut || 0.0);
                            std::mem::swap(&mut w0_sim.listener_pool, &mut base.listener_pool);
                            if let Some(cert) = cert {
                                let mut crng =
                                    SplitMix64::new(decision_seed ^ 0x454e_4447_414d_4531);
                                let draw = crng.next_f64();
                                let mut cum = 0.0;
                                let mut pick_opt = None;
                                for &(act_opt, prob) in &cert.actions {
                                    if prob <= 0.0 {
                                        continue;
                                    }
                                    pick_opt = act_opt;
                                    cum += prob;
                                    if draw < cum {
                                        break;
                                    }
                                }
                                if let Some(pick) = pick_opt {
                                    if my_acts
                                        .iter()
                                        .position(|&c| c == pick)
                                        .is_some_and(|i| eligible_flags[i])
                                    {
                                        self.nodes_buf = nodes;
                                        self.index_buf = index;
                                        return pick;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            let mut my_n = vec![0u32; n_my];
            let mut my_w = vec![0.0f64; n_my];
            let mut ord_n = vec![0u32; n_my];
            let mut ord_w = vec![0.0f64; n_my];
            let mut joint: Vec<(usize, ReplyPoolKey, u32, f64)> = Vec::new();
            let mut joint_index: FxHashMap<(usize, ReplyPoolKey), usize> = FxHashMap::default();
            let mut bayesian_row_q: Option<Vec<f64>> = None;

        // Stage 2A state (strictly read-only prior sweep & smoothed search-guidance buffers).
        let n_elig_usize = eligible_flags.iter().filter(|&&e| e).count().max(1);
        let mut p_eval: Vec<f64> = eligible_flags
            .iter()
            .map(|&e| if e { 1.0 / (n_elig_usize as f64) } else { 0.0 })
            .collect();
        let mut p_eval_sparse: Vec<(usize, f64)> = p_eval
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| (eligible_flags[i] && p >= 0.01).then_some((i, p)))
            .collect();
        let mut q0_prior = vec![0.5f64; n_my];
        let mut has_stage2_prior = false;
        let mut has_stage2_refresh = false;
        let mut i_comp: Vec<usize> = (0..n_my).filter(|&i| eligible_flags[i]).collect();

        let leaf_mode = self.cfg.leaf;
        if stage2_root && self.cfg.iterations >= 128 && !root_types.is_empty() {
            let total_prior_cells: usize = root_types
                .iter()
                .map(|t| n_elig_usize * t.replies.max(1))
                .sum();
            let k_draw: usize = if total_prior_cells <= 200 { 4 } else { 2 };
            let mut sweep_rng = SplitMix64::new(decision_seed ^ 0x51A7_E109_C3B4_2D7F);
            let eval_weights = EvalWeights::default();

            for t in &mut root_types {
                let w = &worlds[t.rep_world_idx];
                let rnode = &nodes[w.root_node_id];
                let n_own_node = rnode.n_acts[side] as usize;
                let mut draws: Vec<(CompactBattle, u64)> = Vec::with_capacity(k_draw);
                for _ in 0..k_draw {
                    let mut cb_d = w.cb;
                    let mut prng = nc2000_engine::prng::Prng::new(sweep_rng.next());
                    let qc = Battle::roll_quick_claw_with(w.table.ruleset, &mut prng);
                    cb_d.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                    let step_seed = sweep_rng.next();
                    let t_seed = sweep_rng.next();
                    if !hidden_timers.is_empty() {
                        let mut trng = SplitMix64::new(t_seed);
                        resample_compact_hidden_timers(&mut cb_d, &w.table, &hidden_timers, &mut trng);
                    }
                    draws.push((cb_d, step_seed));
                }

                if t.replies > 0 {
                    let m = t.replies.min(rnode.n_acts[opp] as usize);
                    for i in 0..n_my.min(n_own_node) {
                        if !eligible_flags[i] {
                            continue;
                        }
                        let a_my = rnode.acts[side][i];
                        for j in 0..m {
                            let a_opp = rnode.acts[opp][j];
                            let actions = if side == 0 { [a_my, a_opp] } else { [a_opp, a_my] };
                            let mut sum = 0.0f64;
                            for &(cb_d, step_seed) in &draws {
                                let mut cb = cb_d;
                                cb.prng_seed = step_seed;
                                cb.apply_actions(&w.table, actions);
                                resolve_compact_forced_switches(&mut cb, &w.table, &eval_weights);
                                let v0 = eval_duct_leaf(&cb, &w.table, leaf_mode, w.cache.as_ref());
                                sum += if side == 0 { v0 } else { 1.0 - v0 };
                            }
                            t.prior_cells[i * t.replies + j] =
                                (sum / (k_draw as f64)).clamp(0.001, 0.999);
                        }
                    }
                    t.cal_prior_buf = t.prior_cells.clone();
                    t.prior_valid = true;
                } else {
                    for i in 0..n_my.min(n_own_node) {
                        if !eligible_flags[i] {
                            continue;
                        }
                        let a_my = rnode.acts[side][i];
                        let actions = if side == 0 { [a_my, 7] } else { [7, a_my] };
                        let mut sum = 0.0f64;
                        for &(cb_d, step_seed) in &draws {
                            let mut cb = cb_d;
                            cb.prng_seed = step_seed;
                            cb.apply_actions(&w.table, actions);
                            resolve_compact_forced_switches(&mut cb, &w.table, &eval_weights);
                            let v0 = eval_duct_leaf(&cb, &w.table, leaf_mode, w.cache.as_ref());
                            sum += if side == 0 { v0 } else { 1.0 - v0 };
                        }
                        t.prior_cells[i] = (sum / (k_draw as f64)).clamp(0.001, 0.999);
                    }
                    t.prior_valid = true;
                }
            }

            let mut t_map: Vec<usize> = Vec::new();
            let mut t_w: Vec<f64> = Vec::new();
            let mut t_m: Vec<usize> = Vec::new();
            let mut t_priors: Vec<&[f64]> = Vec::new();
            for (ti, t) in root_types.iter().enumerate() {
                if t.prior_valid && t.replies > 0 && t.world_count > 0 {
                    t_map.push(ti);
                    t_w.push(f64::from(t.world_count));
                    t_m.push(t.replies);
                    t_priors.push(&t.prior_cells);
                }
            }
            if !t_map.is_empty() {
                if let Some((p0, opp_pol, q0)) =
                    solve_initial_bayesian_prior(&eligible_flags, &t_w, &t_m, &t_priors)
                {
                    p_eval = p0;
                    p_eval_sparse.clear();
                    for (i, &p) in p_eval.iter().enumerate() {
                        if eligible_flags[i] && p >= 0.01 {
                            p_eval_sparse.push((i, p));
                        }
                    }
                    if p_eval_sparse.is_empty() {
                        if let Some(best_i) = (0..n_my)
                            .filter(|&i| eligible_flags[i])
                            .max_by(|&a, &b| p_eval[a].total_cmp(&p_eval[b]))
                        {
                            p_eval_sparse.push((best_i, 1.0));
                        }
                    }
                    for (k, &orig_t) in t_map.iter().enumerate() {
                        root_types[orig_t].opp_eq_policy = opp_pol[k].clone();
                    }
                    q0_prior = q0;
                    has_stage2_prior = true;
                }
            } else {
                // Forced-switch root (`replies == 0` for all types): world-weighted 1-ply prior.
                let mut total_wc = 0.0f64;
                let mut acc = vec![0.0f64; n_my];
                for t in &root_types {
                    if t.prior_valid && t.world_count > 0 {
                        let wc = f64::from(t.world_count);
                        total_wc += wc;
                        for i in 0..n_my {
                            if eligible_flags[i] {
                                acc[i] += wc * t.prior_cells[i];
                            }
                        }
                    }
                }
                if total_wc > 0.0 {
                    for i in 0..n_my {
                        if eligible_flags[i] {
                            q0_prior[i] = acc[i] / total_wc;
                        }
                    }
                    has_stage2_prior = true;
                }
            }
        }

        // Match shipped `RmConfig::default()` (`horizon = 100` turns, clamped by ruleset turn limit).
        let mut turn_cap = base.turn.saturating_add(100);
        if let Some(lim) = base.ruleset.turn_limit() {
            turn_cap = turn_cap.min(lim);
        }

        let build_eq_inputs = |my_n: &[u32],
                               my_w: &[f64],
                               root_types: &[CompactRootTypeStats],
                               joint: &[(usize, ReplyPoolKey, u32, f64)],
                               joint_index: &FxHashMap<(usize, ReplyPoolKey), usize>|
         -> Option<(Vec<RootType>, Vec<Vec<RootCell>>, Vec<f64>, Vec<usize>)> {
            if root_types.is_empty() {
                return None;
            }
            let total_late: u64 = root_types.iter().map(|t| u64::from(t.late_samples)).sum();
            let use_late = total_late >= 512;
            let mut pooled_late: FxHashMap<(usize, ReplyPoolKey), (u32, f64)> = FxHashMap::default();
            let mut late_row_n = vec![0u32; n_my];
            let mut late_row_w = vec![0.0f64; n_my];
            if use_late {
                for t in root_types {
                    if t.late_samples == 0 || t.replies == 0 {
                        continue;
                    }
                    for (i, &(n, sum)) in t.late_cells.iter().enumerate() {
                        if n == 0 {
                            continue;
                        }
                        let own = i / t.replies;
                        let reply = i % t.replies;
                        if let Some(&rkey) = t.reply_keys.get(reply) {
                            let entry = pooled_late.entry((own, rkey)).or_insert((0, 0.0));
                            entry.0 += n;
                            entry.1 += sum;
                        }
                        if own < n_my {
                            late_row_n[own] += n;
                            late_row_w[own] += sum;
                        }
                    }
                }
            }
            let mut types: Vec<RootType> = Vec::with_capacity(root_types.len());
            let mut pooled_priors: Vec<Vec<RootCell>> = Vec::with_capacity(root_types.len());
            let mut type_map: Vec<usize> = Vec::with_capacity(root_types.len());
            for (orig_t, t) in root_types.iter().enumerate() {
                let (samples, src_cells) = if use_late && t.late_samples >= 16 {
                    (t.late_samples, &t.late_cells)
                } else {
                    (t.samples, &t.cells)
                };
                if samples == 0 || t.replies == 0 {
                    continue;
                }
                let cells: Vec<RootCell> = src_cells
                    .iter()
                    .enumerate()
                    .filter(|(_, (n, _))| *n > 0)
                    .map(|(i, &(n, sum))| RootCell {
                        own: i / t.replies,
                        reply: i % t.replies,
                        samples: n,
                        mean: sum / f64::from(n),
                    })
                    .collect();
                let mut priors = Vec::new();
                for own in 0..n_my {
                    if !eligible_flags[own] {
                        continue;
                    }
                    for (reply, &rkey) in t.reply_keys.iter().enumerate() {
                        if let Some(&(ln, lw)) =
                            pooled_late.get(&(own, rkey)).filter(|&&(n, _)| n >= 8)
                        {
                            priors.push(RootCell {
                                own,
                                reply,
                                samples: ln,
                                mean: lw / f64::from(ln),
                            });
                        } else if let Some(&idx) = joint_index.get(&(own, rkey)) {
                            let (_, _, jn, jw) = joint[idx];
                            if jn > 0 {
                                priors.push(RootCell {
                                    own,
                                    reply,
                                    samples: jn,
                                    mean: jw / f64::from(jn),
                                });
                            }
                        }
                    }
                }
                types.push(RootType { samples, replies: t.replies, cells });
                pooled_priors.push(priors);
                type_map.push(orig_t);
            }
            if types.is_empty() {
                return None;
            }
            let mut means: Vec<f64> = (0..n_my)
                .map(|i| if my_n[i] > 0 { my_w[i] / f64::from(my_n[i]) } else { 0.5 })
                .collect();
            if use_late {
                for own in 0..n_my {
                    if late_row_n[own] >= 16 {
                        means[own] = late_row_w[own] / f64::from(late_row_n[own]);
                    }
                }
            }
            Some((types, pooled_priors, means, type_map))
        };

        let compute_bayesian_eq = |my_n: &[u32],
                                   my_w: &[f64],
                                   root_types: &[CompactRootTypeStats],
                                   joint: &[(usize, ReplyPoolKey, u32, f64)],
                                   joint_index: &FxHashMap<(usize, ReplyPoolKey), usize>|
         -> Option<crate::root_policy::BayesianEquilibrium> {
            let (types, pooled_priors, means, _) =
                build_eq_inputs(my_n, my_w, root_types, joint, joint_index)?;
            solve_bayesian_equilibrium_with_priors(
                &eligible_flags,
                my_n,
                &means,
                &types,
                &pooled_priors,
            )
        };

        let compute_stage2_eq = |my_n: &[u32],
                                 my_w: &[f64],
                                 root_types: &[CompactRootTypeStats],
                                 joint: &[(usize, ReplyPoolKey, u32, f64)],
                                 joint_index: &FxHashMap<(usize, ReplyPoolKey), usize>,
                                 use_cal_priors: bool|
         -> Option<(Stage2BayesianEquilibrium, Vec<usize>)> {
            let (types, pooled_priors, means, type_map) =
                build_eq_inputs(my_n, my_w, root_types, joint, joint_index)?;
            let cal_slices: Vec<&[f64]> = if use_cal_priors {
                type_map
                    .iter()
                    .map(|&orig_t| {
                        let t = &root_types[orig_t];
                        if t.prior_valid && t.cal_prior_buf.len() == n_my * t.replies {
                            t.cal_prior_buf.as_slice()
                        } else {
                            &[]
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let s2 = solve_bayesian_equilibrium_stage2(
                &eligible_flags,
                my_n,
                &means,
                &types,
                &pooled_priors,
                if use_cal_priors { Some(&cal_slices) } else { None },
            )?;
            Some((s2, type_map))
        };

        let select_root = |done: u32,
                           endgame: &EndgameRootStats,
                           my_n: &[u32],
                           my_w: &[f64],
                           ord_n: &[u32],
                           ord_w: &[f64],
                           root_types: &[CompactRootTypeStats],
                           joint: &[(usize, ReplyPoolKey, u32, f64)],
                           joint_index: &FxHashMap<(usize, ReplyPoolKey), usize>|
         -> SearchChoice {
            // Separate ordinary on-policy visits/rewards from forced cross-sampling probes
            // (matching `blind.rs::effective_visits` so off-policy Tier-2 refutations never
            // distort `argmax` visit-leader selection, Path-1 LP admission, or `EndgameRootStats`).
            let has_ord = (0..n_my).any(|i| eligible_flags[i] && ord_n[i] > 0);
            let eff_n = if has_ord { ord_n } else { my_n };
            let eff_w = if has_ord { ord_w } else { my_w };

            let reply_metrics = |own: usize, n_i: u32, mean_i: f64| -> (f64, f64) {
                if n_i == 0 {
                    return (0.0, mean_i);
                }
                let mut trap_risk = 0.0f64;
                let mut mistake_weighted_sum = 0.0f64;
                let mut mistake_weight_total = 0.0f64;
                for t in root_types {
                    if t.replies == 0 {
                        continue;
                    }
                    let rs = own * t.replies;
                    let re = rs + t.replies;
                    let Some(row) = t.cells.get(rs..re) else {
                        continue;
                    };
                    let type_n: u32 = row.iter().map(|&(n, _)| n).sum();
                    if type_n == 0 {
                        continue;
                    }
                    let type_w: f64 = row.iter().map(|&(_, w)| w).sum();
                    let type_mean = type_w / f64::from(type_n);
                    let (t_risk, t_mistake) =
                        reply_risk_and_mistake_ev(type_n, type_mean, row.iter().copied());
                    let w_t = f64::from(type_n);
                    mistake_weighted_sum += w_t * t_mistake;
                    mistake_weight_total += w_t;
                    if type_n >= 4 {
                        let type_shrink = (w_t / (w_t + 6.0)).clamp(0.0, 1.0);
                        let type_drop =
                            ((mean_i - type_mean).max(0.0) * type_shrink).clamp(0.0, 1.0);
                        trap_risk = trap_risk.max(type_drop.max(t_risk * type_shrink));
                    }
                }
                let mistake_ev = if mistake_weight_total > 0.0 {
                    let unattributed = (f64::from(n_i) - mistake_weight_total).max(0.0);
                    (mistake_weighted_sum + unattributed * mean_i)
                        / (mistake_weight_total + unattributed)
                } else {
                    mean_i
                };
                (trap_risk, mistake_ev)
            };

            select_blind_root_choice(
                endgame,
                &my_acts,
                eff_n,
                eff_n,
                eff_w,
                |i| eligible_flags[i],
                reply_metrics,
                |base_idx| {
                    if done >= 64 {
                        let eq_opt = if stage2_root {
                            // Strictly prior-free readout LP anchored to ordinary visits (`eff_n`/`eff_w`)
                            // so `base_idx = argmax eff_n` is guaranteed admitted in Path 1!
                            compute_stage2_eq(eff_n, eff_w, root_types, joint, joint_index, false)
                                .map(|(s2, _)| s2.eq)
                        } else {
                            compute_bayesian_eq(eff_n, eff_w, root_types, joint, joint_index)
                        };
                        if let Some(eq) = eq_opt {
                            if eq.override_value > eq.pure_guarantees[base_idx] + MIN_MIX_MARGIN {
                                let draw =
                                    SplitMix64::new(decision_seed ^ u64::from(done)).next_f64();
                                return crate::blind::draw_policy(&my_acts, &eq.policy, draw);
                            }
                        }
                    }
                    None
                },
            )
            .unwrap_or(my_acts[0])
        };

        let mut path: Vec<(usize, usize, usize)> = Vec::with_capacity(self.cfg.max_depth as usize + 2);
        let n_elig = n_elig_usize as u32;
        let world_warmup = (192 / root_types.len().max(2)).clamp(48, 96) as u32;
        let c_ucb = self.cfg.c;
        let max_depth = self.cfg.max_depth.min(250);
        let rollout_turns = self.cfg.rollout_turns;
        let rollout_playout = crate::mcts::Playout::Heavy {
            turns: rollout_turns,
            eps: 0.2,
            weights: EvalWeights::default(),
        };
        let cf_eg_playout = crate::mcts::Playout::Heavy {
            turns: 8,
            eps: 0.2,
            weights: EvalWeights::default(),
        };
        let rollout_rules = RmConfig::default().rollout_rules();
        let n_worlds = worlds.len();
        let adaptive = self.cfg.adaptive;
        let adaptive_interval = self.cfg.adaptive_interval.max(1);
        let adaptive_min_gap = self.cfg.adaptive_min_gap;
        let mut prev_adaptive: Option<(u32, SearchChoice)> = None;
        let mut done = 0u32;
        let base_turn = base.turn;
        let base_lp = &mut base.listener_pool;

        for iter in 0..self.cfg.iterations {
            if stage2_root {
                if iter >= 256 && iter % 256 == 0 {
                    // Refresh bias-corrected 1-ply calibrated priors per type.
                    for t in &mut root_types {
                        if !t.prior_valid || t.replies == 0 {
                            continue;
                        }
                        let src = if t.late_samples >= 64 { &t.late_cells } else { &t.cells };
                        let mut num = 0.0f64;
                        let mut den = 0.0f64;
                        for own in 0..n_my {
                            if !eligible_flags[own] {
                                continue;
                            }
                            for j in 0..t.replies {
                                let idx = own * t.replies + j;
                                let (n_ij, w_ij) = src[idx];
                                if n_ij >= 8 {
                                    let wt = f64::from(n_ij.min(32));
                                    let mean_ij = w_ij / f64::from(n_ij);
                                    num += wt * (mean_ij - t.prior_cells[idx]);
                                    den += wt;
                                }
                            }
                        }
                        t.prior_delta = if den > 0.0 {
                            (num / (den + 8.0)).clamp(-0.35, 0.35)
                        } else {
                            0.0
                        };
                        if t.cal_prior_buf.len() != n_my * t.replies {
                            t.cal_prior_buf.resize(n_my * t.replies, 0.5);
                        }
                        for (dst, &p0) in t.cal_prior_buf.iter_mut().zip(t.prior_cells.iter()) {
                            *dst = (p0 + t.prior_delta).clamp(0.02, 0.98);
                        }
                    }

                    let has_ord = (0..n_my).any(|i| eligible_flags[i] && ord_n[i] > 0);
                    let eff_n = if has_ord { ord_n.as_slice() } else { my_n.as_slice() };
                    let eff_w = if has_ord { ord_w.as_slice() } else { my_w.as_slice() };
                    if let Some((mut s2, type_map)) =
                        compute_stage2_eq(eff_n, eff_w, &root_types, &joint, &joint_index, true)
                    {
                        for (k, &orig_t) in type_map.iter().enumerate() {
                            let t = &mut root_types[orig_t];
                            t.opp_eq_policy = std::mem::take(&mut s2.opp_eq_policy[k]);
                            t.active_cols = std::mem::take(&mut s2.active_cols[k]);
                            t.search_v = std::mem::take(&mut s2.search_v[k]);
                            let m = t.replies;
                            let mut t_eq = 0.0f64;
                            let mut p_sum_eq = 0.0f64;
                            if t.search_v.len() == n_my * m && t.opp_eq_policy.len() == m {
                                for i in 0..n_my {
                                    let pi = s2.eq.policy[i];
                                    if pi <= 0.0 {
                                        continue;
                                    }
                                    p_sum_eq += pi;
                                    let mut tv = 0.0f64;
                                    for (j, &yj) in t.opp_eq_policy.iter().enumerate() {
                                        if yj > 0.0 {
                                            tv += yj * t.search_v[i * m + j];
                                        }
                                    }
                                    t_eq += pi * tv;
                                }
                            }
                            t.type_eq_value = if p_sum_eq > 1e-9 {
                                t_eq / p_sum_eq
                            } else {
                                s2.eq.equilibrium_value
                            };
                        }
                        i_comp.clear();
                        for i in 0..n_my {
                            if !eligible_flags[i] {
                                continue;
                            }
                            if s2.eq.policy[i] >= 0.02 {
                                i_comp.push(i);
                                continue;
                            }
                            // Threat-weighted optimistic cell UCB anchored to equilibrium_value
                            // prevents untested-threat starvation and 1-ply trap crowding.
                            let mut ucb_threat = 0.0f64;
                            let mut w_tot = 0.0f64;
                            for (k, &orig_t) in type_map.iter().enumerate() {
                                let wt = s2.type_weights[k];
                                if wt <= 0.0 {
                                    continue;
                                }
                                let t = &root_types[orig_t];
                                let m = t.replies;
                                if m == 0 || t.search_v.len() != n_my * m {
                                    continue;
                                }
                                let (src, n_t) = if t.late_samples >= 64 {
                                    (&t.late_cells, t.late_samples)
                                } else {
                                    (&t.cells, t.samples)
                                };
                                let ln_nt = (f64::from(n_t) + 1.0).ln().max(0.0);
                                let mut tv = 0.0f64;
                                for (j, &yj) in t.opp_eq_policy.iter().enumerate() {
                                    if yj <= 0.0 {
                                        continue;
                                    }
                                    let cn = src[i * m + j].0;
                                    let bonus = 0.35 * (ln_nt / (f64::from(cn) + 1.0)).sqrt();
                                    tv += yj * (t.search_v[i * m + j] + bonus).min(1.0);
                                }
                                ucb_threat += wt * tv;
                                w_tot += wt;
                            }
                            let ucb_val = if w_tot > 1e-9 {
                                ucb_threat / w_tot
                            } else {
                                s2.search_action_values[i]
                            };
                            if ucb_val >= s2.eq.equilibrium_value - 0.04 {
                                i_comp.push(i);
                            }
                        }
                        if i_comp.is_empty() {
                            i_comp.push(s2.eq.best_pure_action);
                        }

                        let sum_elig: u32 = (0..n_my)
                            .filter(|&i| eligible_flags[i])
                            .map(|i| eff_n[i])
                            .sum();
                        let sum_elig_f = f64::from(sum_elig.max(1));
                        let mut p_tot = 0.0f64;
                        for i in 0..n_my {
                            if eligible_flags[i] {
                                let p_emp = f64::from(eff_n[i]) / sum_elig_f;
                                let p_tgt = 0.7 * s2.eq.policy[i] + 0.3 * p_emp;
                                p_eval[i] = 0.5 * p_eval[i] + 0.5 * p_tgt;
                                p_tot += p_eval[i];
                            } else {
                                p_eval[i] = 0.0;
                            }
                        }
                        if p_tot > 1e-9 {
                            for p in &mut p_eval {
                                *p /= p_tot;
                            }
                        }
                        p_eval_sparse.clear();
                        for (i, &p) in p_eval.iter().enumerate() {
                            if eligible_flags[i] && p >= 0.01 {
                                p_eval_sparse.push((i, p));
                            }
                        }
                        if p_eval_sparse.is_empty() {
                            if let Some(best_i) = (0..n_my)
                                .filter(|&i| eligible_flags[i])
                                .max_by(|&a, &b| p_eval[a].total_cmp(&p_eval[b]))
                            {
                                p_eval_sparse.push((best_i, 1.0));
                            }
                        }

                        // Keep clean prior-free `KAPPA_CELL = 64` values for admitted LP actions
                        // (including the visit leader) while using unblocked `search_action_values`
                        // only for unadmitted rivals so Turn-35 buried counters stay reachable.
                        let mut row_q = vec![0.5f64; n_my];
                        for i in 0..n_my {
                            row_q[i] = if s2.lp_admitted[i] {
                                s2.eq.action_values[i]
                            } else {
                                s2.search_action_values[i]
                            };
                        }
                        bayesian_row_q = Some(row_q);
                        has_stage2_refresh = true;
                    }
                }
            } else if iter >= 512 && iter % 256 == 0 {
                if let Some(eq) = compute_bayesian_eq(&my_n, &my_w, &root_types, &joint, &joint_index)
                {
                    bayesian_row_q = Some(eq.action_values);
                }
            }

            let w = &worlds[(iter as usize) % n_worlds];
            let type_index = w.type_index;
            let root_id = w.root_node_id;

            // 1. Global own root selection (+ optional Stage 2A gap-gated targeted threat cross-sample).
            let mut forced_root_opp: Option<usize> = None;
            let forced_threat_pair = if stage2_root
                && has_stage2_refresh
                && iter >= 256
                && root_types[type_index].replies > 1
                && root_types[type_index].samples % 16 == 0
                && !i_comp.is_empty()
            {
                let stats = &root_types[type_index];
                let m = stats.replies;
                let src = if stats.late_samples >= 64 { &stats.late_cells } else { &stats.cells };
                let max_threat = (192u32 / (root_types.len().max(1) as u32)).clamp(32, 192);
                let n_threat =
                    ((2.0 * f64::from(stats.samples).sqrt()).floor() as u32).clamp(12, max_threat);
                let j_ucb = select_stage2_opp_reply(stats, &p_eval_sparse, c_ucb);
                let bq = bayesian_row_q.as_deref();

                // Tier 1: Active threats (`y* >= 0.05` or current `argmax UCB_opp`) for `i ∈ I_comp`.
                let mut pick_pair: Option<(usize, usize, u32, f64, f64)> = None;
                for &i in &i_comp {
                    let qi = bq.and_then(|q| q.get(i).copied()).unwrap_or(0.5);
                    for j in 0..m {
                        let yj = stats.opp_eq_policy.get(j).copied().unwrap_or(0.0);
                        if yj < 0.05 && j != j_ucb {
                            continue;
                        }
                        let cn = src[i * m + j].0;
                        if cn < n_threat {
                            let better = pick_pair.map_or(true, |(_, _, bn, by, bq_v)| {
                                cn < bn
                                    || (cn == bn
                                        && (yj > by + 1e-12
                                            || ((yj - by).abs() <= 1e-12 && qi > bq_v)))
                            });
                            if better {
                                pick_pair = Some((i, j, cn, yj, qi));
                            }
                        }
                    }
                }
                // Tier 2: Refutation guard (`j_ref = argmin search_v[i, j] < eq_value - 0.03`).
                if pick_pair.is_none() && stats.search_v.len() == n_my * m {
                    let n_ref = n_threat.min(48);
                    for &i in &i_comp {
                        let mut best_ref_j = 0usize;
                        let mut min_sv = f64::INFINITY;
                        for j in 0..m {
                            let sv = stats.search_v[i * m + j];
                            if sv < min_sv {
                                min_sv = sv;
                                best_ref_j = j;
                            }
                        }
                        if min_sv < stats.type_eq_value - 0.03 {
                            let cn = src[i * m + best_ref_j].0;
                            if cn < n_ref {
                                let better = pick_pair.map_or(true, |(_, _, bn, _, _)| cn < bn);
                                if better {
                                    pick_pair = Some((i, best_ref_j, cn, 0.0, 0.0));
                                }
                            }
                        }
                    }
                }
                // Tier 3: Other active LP columns (`min(24, n_threat)`).
                if pick_pair.is_none() && !stats.active_cols.is_empty() {
                    let n_act = 24u32.min(n_threat);
                    for &i in &i_comp {
                        for &j in &stats.active_cols {
                            let cn = src[i * m + j].0;
                            if cn < n_act {
                                let better = pick_pair.map_or(true, |(_, _, bn, _, _)| cn < bn);
                                if better {
                                    pick_pair = Some((i, j, cn, 0.0, 0.0));
                                }
                            }
                        }
                    }
                }
                pick_pair.map(|(i, j, _, _, _)| (i, j))
            } else {
                None
            };

            let is_forced_probe = forced_threat_pair.is_some();
            let my_pick = if let Some((ti, tj)) = forced_threat_pair {
                forced_root_opp = Some(tj);
                my_n[ti] += 1;
                ti
            } else {
                let min_floor_pick =
                    if !stage2_root && iter >= 512 && root_types[type_index].replies > 0 {
                        let stats = &root_types[type_index];
                        let replies = stats.replies;
                        let world_floor = (stats.samples / (3 * n_elig)).min(48);
                        (0..n_my)
                            .filter(|&i| eligible_flags[i])
                            .filter_map(|i| {
                                let rs = i * replies;
                                let w_n: u32 = stats
                                    .cells
                                    .get(rs..rs + replies)
                                    .map_or(0, |row| row.iter().map(|&(n, _)| n).sum());
                                (w_n < world_floor).then_some((w_n, my_n[i], i))
                            })
                            .min_by_key(|&(w_n, g_n, _)| (w_n, g_n))
                            .map(|(_, _, i)| i)
                    } else {
                        None
                    };
                if let Some(a) = min_floor_pick {
                    my_n[a] += 1;
                    ord_n[a] += 1;
                    a
                } else {
                    let mut n_untried = 0usize;
                    for a in 0..n_my {
                        if eligible_flags[a] && my_n[a] == 0 {
                            n_untried += 1;
                        }
                    }
                    let pick = if n_untried > 0 {
                        if stage2_root && has_stage2_prior {
                            let mut best_a = 0usize;
                            let mut best_q = f64::NEG_INFINITY;
                            for a in 0..n_my {
                                if eligible_flags[a] && my_n[a] == 0 && q0_prior[a] > best_q {
                                    best_q = q0_prior[a];
                                    best_a = a;
                                }
                            }
                            best_a
                        } else {
                            let mut target = self.rng.below(n_untried);
                            let mut chosen = 0usize;
                            for a in 0..n_my {
                                if eligible_flags[a] && my_n[a] == 0 {
                                    if target == 0 {
                                        chosen = a;
                                        break;
                                    }
                                    target -= 1;
                                }
                            }
                            chosen
                        }
                    } else {
                        let total: u32 = my_n.iter().sum();
                        let ln_total = (total.max(1) as f64).ln();
                        let mut best_a = eligible_flags.iter().position(|&e| e).unwrap_or(0);
                        let mut best_v = f64::NEG_INFINITY;
                        let bq = bayesian_row_q.as_deref();
                        let use_cold_prior = stage2_root && has_stage2_prior && (iter < 256 || bq.is_none());
                        for a in 0..n_my {
                            if !eligible_flags[a] {
                                continue;
                            }
                            let na = my_n[a] as f64;
                            let mean_a = if use_cold_prior {
                                (my_w[a] + 4.0 * q0_prior[a]) / (na + 4.0)
                            } else {
                                bq.and_then(|q| q.get(a).copied())
                                    .unwrap_or_else(|| my_w[a] / na)
                            };
                            let v = mean_a + c_ucb * (ln_total / na).sqrt();
                            if v > best_v {
                                best_v = v;
                                best_a = a;
                            }
                        }
                        best_a
                    };
                    my_n[pick] += 1;
                    ord_n[pick] += 1;
                    pick
                }
            };

            // 2. Prepare 256-byte CompactBattle for this iteration (~20 ns total!).
            // Resample Quick Claw and seed CompactBattle PRNG once per iteration (exact parity with `smmcts::run_compact_iteration`).
            let mut cb = w.cb;
            {
                let mut prng = nc2000_engine::prng::Prng::new(self.rng.next());
                let qc = Battle::roll_quick_claw_with(w.table.ruleset, &mut prng);
                cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                cb.prng_seed = prng.seed();
            }
            if !hidden_timers.is_empty() {
                resample_compact_hidden_timers(&mut cb, &w.table, &hidden_timers, &mut timer_rng);
            }

            // 3. Descend CompactBattle DUCT tree + settle leaf.
            let mut depth = 0u8;
            let mut cur_id = root_id;
            path.clear();
            let mut root_opp_idx = 0usize;

            let (leaf_val, end_info) = loop {
                if depth > 0 && use_puct {
                    nodes[cur_id].ensure_puct_priors(
                        &cb,
                        &w.table,
                        w.cache.as_ref(),
                        puct_eps,
                        use_cf_policy,
                    );
                }
                let node = &nodes[cur_id];
                let n0_cur = node.n_acts[0] as usize;
                let n1_cur = node.n_acts[1] as usize;
                let (i0, i1) = if depth == 0 {
                    let i_opp = if let Some(fj) = forced_root_opp {
                        fj
                    } else if stage2_root && root_types[type_index].replies > 1 {
                        select_stage2_opp_reply(&root_types[type_index], &p_eval_sparse, c_ucb)
                    } else {
                        node.select_ucb(opp, c_ucb, &mut self.rng, true)
                    };
                    root_opp_idx = i_opp;
                    if side == 0 { (my_pick, i_opp) } else { (i_opp, my_pick) }
                } else if use_puct {
                    (
                        node.select_puct(0, c_puct, fpu_red),
                        node.select_puct(1, c_puct, fpu_red),
                    )
                } else {
                    (
                        node.select_ucb(0, c_ucb, &mut self.rng, true),
                        node.select_ucb(1, c_ucb, &mut self.rng, true),
                    )
                };
                let a0 = if n0_cur > 0 { node.acts[0][i0] } else { 7 };
                let a1 = if n1_cur > 0 { node.acts[1][i1] } else { 7 };
                // Skip pushing 1x1 forced-action interior nodes so `backed_up_value` never dampens forced lines.
                if depth == 0 || n0_cur > 1 || n1_cur > 1 {
                    path.push((cur_id, i0, i1));
                }

                cb.apply_actions(&w.table, [a0, a1]);
                depth = depth.saturating_add(1);

                if cb.outcome != 0 {
                    let v = eval_duct_leaf_temp(&cb, &w.table, leaf_mode, w.cache.as_ref(), leaf_temp);
                    break (v, compact_sim_end(&cb, &w.table, v));
                }
                if let Some(v) = compact_turn_limit_outcome(&cb, &w.table) {
                    break (v, compact_sim_end(&cb, &w.table, v));
                }
                if depth >= max_depth || cb.turn > turn_cap {
                    if let Some(sess) = eg_session.as_mut() {
                        let n_tot = cb.sides[0].alive_mask.count_ones()
                            + cb.sides[1].alive_mask.count_ones();
                        if n_tot <= 3
                            && sess.stats.runs < eg_decision_runs
                            && sess.eligible_compact(&cb, &w.table)
                        {
                            if let Some(w_sim) = w.sim.as_deref() {
                                let mut bbuf = match eg_battle_buf.take() {
                                    Some(mut sim) => {
                                        sim.reset_search_sim(w_sim, [false, false]);
                                        sim
                                    }
                                    None => Box::new(w_sim.clone()),
                                };
                                std::mem::swap(&mut bbuf.listener_pool, base_lp);
                                cb.sync_to_battle(&w.table, dex, &mut bbuf);
                                let mut cert = None;
                                for _ in 0..4 {
                                    let before = sess.stats.runs;
                                    if let Some(cv) = sess.conditional_value(dex, &bbuf, &mut || 0.0) {
                                        cert = Some(cv);
                                        break;
                                    }
                                    if sess.stats.runs == before {
                                        break;
                                    }
                                }
                                std::mem::swap(&mut bbuf.listener_pool, base_lp);
                                eg_battle_buf = Some(bbuf);
                                if let Some(cv) = cert {
                                    let mut info = crate::mcts::SimEndInfo::from_compact(&cb);
                                    let v = if cv.bounds.lo >= 1.0 - 1e-6 {
                                        info.outcome = 1;
                                        1.0
                                    } else if cv.bounds.hi <= 1e-6 {
                                        info.outcome = 2;
                                        0.0
                                    } else {
                                        cv.estimate.clamp(0.000_002, 0.999_998)
                                    };
                                    break (v, info);
                                }
                            }
                        }
                    }
                    let v = eval_duct_leaf_temp(&cb, &w.table, leaf_mode, w.cache.as_ref(), leaf_temp);
                    break (v, compact_sim_end(&cb, &w.table, v));
                }

                let omit_qc = !cb.quick_claw_observable(&w.table);
                let next_key = cb.search_key(&w.table, 16, omit_qc);
                if let Some(&next_id) = index.get(&next_key) {
                    cur_id = next_id;
                } else {
                    let new_id = nodes.len();
                    nodes.push(DuctNode::new(&cb, &w.table));
                    index.insert(next_key, new_id);
                    if let Some(sess) = eg_session.as_mut() {
                        let n_tot = cb.sides[0].alive_mask.count_ones()
                            + cb.sides[1].alive_mask.count_ones();
                        if n_tot <= 3
                            && sess.stats.runs < eg_decision_runs
                            && sess.eligible_compact(&cb, &w.table)
                        {
                            if let Some(w_sim) = w.sim.as_deref() {
                                let mut bbuf = match eg_battle_buf.take() {
                                    Some(mut sim) => {
                                        sim.reset_search_sim(w_sim, [false, false]);
                                        sim
                                    }
                                    None => Box::new(w_sim.clone()),
                                };
                                std::mem::swap(&mut bbuf.listener_pool, base_lp);
                                cb.sync_to_battle(&w.table, dex, &mut bbuf);
                                let mut cert = None;
                                for _ in 0..4 {
                                    let before = sess.stats.runs;
                                    if let Some(cv) = sess.conditional_value(dex, &bbuf, &mut || 0.0) {
                                        cert = Some(cv);
                                        break;
                                    }
                                    if sess.stats.runs == before {
                                        break;
                                    }
                                }
                                std::mem::swap(&mut bbuf.listener_pool, base_lp);
                                eg_battle_buf = Some(bbuf);
                                if let Some(cv) = cert {
                                    let mut info = crate::mcts::SimEndInfo::from_compact(&cb);
                                    let v = if cv.bounds.lo >= 1.0 - 1e-6 {
                                        info.outcome = 1;
                                        1.0
                                    } else if cv.bounds.hi <= 1e-6 {
                                        info.outcome = 2;
                                        0.0
                                    } else {
                                        cv.estimate.clamp(0.000_002, 0.999_998)
                                    };
                                    break (v, info);
                                }
                            }
                        }
                    }
                    if rollout_turns == 0 {
                        if leaf_mode == TempoLeafMode::CfValue
                            && (cb.sides[0].alive_mask.count_ones()
                                + cb.sides[1].alive_mask.count_ones())
                                <= 3
                        {
                            let cutoff = turn_cap.min(cb.turn.saturating_add(7));
                            let ew = EvalWeights::default();
                            let v = crate::mcts::compact_playout_with_leaf(
                                &mut cb,
                                &w.table,
                                &cf_eg_playout,
                                cutoff,
                                &mut self.rng,
                                rollout_rules,
                                |cb_leaf, tbl_leaf| {
                                    compact_eval_leaf_blindspot(cb_leaf, tbl_leaf, &ew)
                                },
                            );
                            break (v, crate::mcts::last_sim_end());
                        }
                        if leaf_mode == TempoLeafMode::CfValue {
                            let v = crate::compact_az::evaluate_cf_value_resolving_switches(
                                &cb,
                                &w.table,
                                w.cache.as_ref(),
                                leaf_temp,
                            );
                            break (v, compact_sim_end(&cb, &w.table, v));
                        }
                        if leaf_mode == TempoLeafMode::AzValue
                            && (cb.sides[0].request == 2 || cb.sides[1].request == 2)
                        {
                            resolve_compact_forced_switches(&mut cb, &w.table, &EvalWeights::default());
                        }
                        let v = eval_duct_leaf_temp(&cb, &w.table, leaf_mode, w.cache.as_ref(), leaf_temp);
                        break (v, compact_sim_end(&cb, &w.table, v));
                    } else {
                        let cutoff =
                            turn_cap.min(cb.turn.saturating_add(rollout_turns.saturating_sub(1)));
                        let v = crate::mcts::compact_playout_with_leaf(
                            &mut cb,
                            &w.table,
                            &rollout_playout,
                            cutoff,
                            &mut self.rng,
                            rollout_rules,
                            |cb_leaf, tbl_leaf| {
                                eval_duct_leaf_temp(
                                    cb_leaf,
                                    tbl_leaf,
                                    leaf_mode,
                                    w.cache.as_ref(),
                                    leaf_temp,
                                )
                            },
                        );
                        break (v, crate::mcts::last_sim_end());
                    }
                }
            };

            // 4. Full-path empirical-Bayes minimax backup.
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
                if nid != root_id && val > 0.000_001 && val < 0.999_999 {
                    val = n.backed_up_value_cfg(val, backup_lambda, backup_tau);
                }
            }

            // 5. Record root stats (endgame_root, RootTypeStats, late_cells, pooled joint).
            let mine = if side == 0 { val } else { 1.0 - val };
            my_w[my_pick] += mine;
            if !is_forced_probe {
                ord_w[my_pick] += mine;
                endgame.record(base_turn, side, my_pick, mine, end_info);
            }
            let stats = &mut root_types[type_index];
            stats.samples += 1;
            if root_opp_idx < stats.replies {
                stats.col_visits[root_opp_idx] += 1;
                let ci = my_pick * stats.replies + root_opp_idx;
                stats.cells[ci].0 += 1;
                stats.cells[ci].1 += mine;
                if stats.samples > world_warmup {
                    stats.late_samples += 1;
                    stats.late_cells[ci].0 += 1;
                    stats.late_cells[ci].1 += mine;
                }
                if let Some(&rkey) = stats.reply_keys.get(root_opp_idx) {
                    let j_idx = *joint_index.entry((my_pick, rkey)).or_insert_with(|| {
                        let k = joint.len();
                        joint.push((my_pick, rkey, 0, 0.0));
                        k
                    });
                    joint[j_idx].2 += 1;
                    joint[j_idx].3 += mine;
                }
            }

            done = iter + 1;
            // Optional global mature-tree exponential window: halve ALL `root_types` simultaneously
            // on global `done` power-of-2 milestones (`>= 2048`) so relative Bayesian posterior weights
            // across common/rare types and across `contested_trees` remain strictly invariant.
            if mature_window && done >= 2048 && done.is_power_of_two() {
                for t in &mut root_types {
                    if t.late_samples < 32 {
                        continue;
                    }
                    let mut new_ls = 0u32;
                    for cell in &mut t.late_cells {
                        if cell.0 >= 2 {
                            let nn = cell.0 / 2;
                            cell.1 *= f64::from(nn) / f64::from(cell.0);
                            cell.0 = nn;
                        }
                        new_ls += cell.0;
                    }
                    t.late_samples = new_ls;
                }
            }
            // 6. Optional `AdaptiveBattleBudget`-compatible early-stop checkpoint (Tree 0 only).
            if tree_k == 0 && adaptive && done < self.cfg.iterations && done % adaptive_interval == 0 {
                let leader_opt = (0..n_my)
                    .filter(|&i| eligible_flags[i])
                    .max_by_key(|&i| ord_n[i]);
                let candidate = leader_opt.and_then(|leader| {
                    let nl = ord_n[leader];
                    if nl == 0 {
                        return None;
                    }
                    let cutoff = crate::root_policy::min_equilibrium_visits(nl);
                    let mean_l = ord_w[leader] / f64::from(nl);
                    for i in 0..n_my {
                        if i != leader && eligible_flags[i] {
                            let ni = ord_n[i];
                            let mean_i = if ni > 0 { ord_w[i] / f64::from(ni) } else { 0.5 };
                            if ni == 0
                                || ni >= cutoff
                                || mean_l - mean_i < adaptive_min_gap
                            {
                                return None;
                            }
                        }
                    }
                    let pick = select_root(
                        done,
                        &endgame,
                        &my_n,
                        &my_w,
                        &ord_n,
                        &ord_w,
                        &root_types,
                        &joint,
                        &joint_index,
                    );
                    (pick == my_acts[leader]).then_some(pick)
                });
                if candidate.is_some()
                    && prev_adaptive == candidate.map(|c| (done - adaptive_interval, c))
                {
                    break;
                }
                prev_adaptive = candidate.map(|c| (done, c));
            }
        }

            // Fast path: single-tree config OR Tree 0 early-stopped on a clear decision.
            if max_trees == 1 || (tree_k == 0 && adaptive && done < self.cfg.iterations) {
                let base_pick = select_root(
                    done,
                    &endgame,
                    &my_n,
                    &my_w,
                    &ord_n,
                    &ord_w,
                    &root_types,
                    &joint,
                    &joint_index,
                );
                let mut final_pick = base_pick;

                // Post-warmup Contested-Turn Counterfactual Top-K Subgame Verification (`@verify`):
                // Zero overhead on early-stopped clear turns; on contested full-cap idle roots,
                // detects close rivals or starved high-`P_net` opponent replies and runs balanced
                // round-robin forced-pair evaluations through the warm deep `P_net` PUCT tree.
                if verify_iters > 0
                    && max_trees == 1
                    && base_idle
                    && done == self.cfg.iterations
                    && n_elig_usize >= 2
                    && root_types.iter().any(|t| t.replies >= 2)
                {
                    let has_ord_v = (0..n_my).any(|i| eligible_flags[i] && ord_n[i] > 0);
                    let eff_n_v = if has_ord_v { ord_n.as_slice() } else { my_n.as_slice() };
                    let eff_w_v = if has_ord_v { ord_w.as_slice() } else { my_w.as_slice() };
                    let eg_normal = endgame
                        .analyze(
                            &my_acts,
                            eff_n_v,
                            eff_n_v,
                            eff_w_v,
                            |i| eligible_flags[i],
                            |_, _, m| (0.0, m),
                        )
                        .is_some_and(|info| info.regime == crate::endgame_root::EndgameRegime::Normal);

                    if eg_normal {
                        let leader = (0..n_my)
                            .filter(|&i| eligible_flags[i])
                            .max_by(|&a, &b| {
                                eff_n_v[a]
                                    .cmp(&eff_n_v[b])
                                    .then_with(|| {
                                        let ma = if eff_n_v[a] > 0 { eff_w_v[a] / f64::from(eff_n_v[a]) } else { -1.0 };
                                        let mb = if eff_n_v[b] > 0 { eff_w_v[b] / f64::from(eff_n_v[b]) } else { -1.0 };
                                        ma.total_cmp(&mb)
                                    })
                            })
                            .unwrap_or(0);
                        let nl = eff_n_v[leader].max(1);
                        let ml = eff_w_v[leader] / f64::from(nl);

                        // Compute root `P_net` priors per root_type (once per type) to catch starved replies.
                        let mut opp_priors: Vec<[f32; 7]> = Vec::with_capacity(root_types.len());
                        let mut starved_reply = false;
                        for t in &root_types {
                            if t.replies >= 2 {
                                let w_rep = &worlds[t.rep_world_idx];
                                let rnode = &nodes[w_rep.root_node_id];
                                let m = t.replies.min(rnode.n_acts[opp] as usize);
                                let (pr, _) = crate::compact_az::policy_priors_mode(
                                    &w_rep.cb,
                                    &w_rep.table,
                                    w_rep.cache.as_ref(),
                                    opp,
                                    &rnode.acts[opp],
                                    m,
                                    0.0,
                                    use_cf_policy,
                                );
                                if t.samples >= 128 {
                                    let floor = t.samples / 12;
                                    for j in 0..m {
                                        if pr[j] >= 0.12 && t.col_visits[j] < floor {
                                            starved_reply = true;
                                            break;
                                        }
                                    }
                                }
                                opp_priors.push(pr);
                            } else {
                                opp_priors.push([0.0; 7]);
                            }
                        }

                        let mut is_contested = starved_reply;
                        if !is_contested {
                            let bq_ref = bayesian_row_q.as_deref();
                            let bq_l = bq_ref.and_then(|q| q.get(leader).copied()).unwrap_or(ml);
                            for i in 0..n_my {
                                if i == leader || !eligible_flags[i] {
                                    continue;
                                }
                                let ni = eff_n_v[i];
                                let mi = if ni > 0 { eff_w_v[i] / f64::from(ni) } else { 0.0 };
                                let bq_i = bq_ref.and_then(|q| q.get(i).copied()).unwrap_or(mi);
                                if ni >= nl / 12
                                    || (ni >= 16 && (ml - mi) < 0.05)
                                    || (bq_l - bq_i) < 0.04
                                {
                                    is_contested = true;
                                    break;
                                }
                            }
                        }

                        if is_contested {
                            // Own root `P_net` priors on rep world 0 to surface starved tactical alternatives.
                            let w0 = &worlds[0];
                            let rn0 = &nodes[w0.root_node_id];
                            let n_own0 = n_my.min(rn0.n_acts[side] as usize);
                            let (own_pr, _) = crate::compact_az::policy_priors_mode(
                                &w0.cb,
                                &w0.table,
                                w0.cache.as_ref(),
                                side,
                                &rn0.acts[side],
                                n_own0,
                                0.0,
                                use_cf_policy,
                            );

                            // Select `s_own` (<= 3 eligible own candidates): guarantee BOTH `base_idx`
                            // (from `select_root`) and `leader` are present, plus top rival / P_net rival.
                            let base_idx = my_acts
                                .iter()
                                .position(|&c| c == base_pick)
                                .filter(|&i| eligible_flags[i])
                                .unwrap_or(leader);
                            let mut s_own: Vec<usize> = Vec::with_capacity(3);
                            s_own.push(base_idx);
                            if leader != base_idx && eligible_flags[leader] {
                                s_own.push(leader);
                            }
                            if s_own.len() < 3 {
                                if let Some(r1) = (0..n_my)
                                    .filter(|&i| eligible_flags[i] && !s_own.contains(&i))
                                    .max_by(|&a, &b| {
                                        eff_n_v[a].cmp(&eff_n_v[b]).then_with(|| {
                                            let ma = if eff_n_v[a] > 0 { eff_w_v[a] / f64::from(eff_n_v[a]) } else { -1.0 };
                                            let mb = if eff_n_v[b] > 0 { eff_w_v[b] / f64::from(eff_n_v[b]) } else { -1.0 };
                                            ma.total_cmp(&mb)
                                        })
                                    })
                                {
                                    s_own.push(r1);
                                }
                            }
                            if s_own.len() < 3 {
                                let bq_ref = bayesian_row_q.as_deref();
                                let cand_bq = (0..n_my)
                                    .filter(|&i| eligible_flags[i] && !s_own.contains(&i))
                                    .max_by(|&a, &b| {
                                        let sa = bq_ref
                                            .and_then(|q| q.get(a).copied())
                                            .unwrap_or_else(|| if eff_n_v[a] > 0 { eff_w_v[a] / f64::from(eff_n_v[a]) } else { -1.0 })
                                            + 0.06 * f64::from(own_pr.get(a).copied().unwrap_or(0.0));
                                        let sb = bq_ref
                                            .and_then(|q| q.get(b).copied())
                                            .unwrap_or_else(|| if eff_n_v[b] > 0 { eff_w_v[b] / f64::from(eff_n_v[b]) } else { -1.0 })
                                            + 0.06 * f64::from(own_pr.get(b).copied().unwrap_or(0.0));
                                        sa.total_cmp(&sb)
                                    });
                                if let Some(r2) = cand_bq {
                                    s_own.push(r2);
                                }
                            }

                            if s_own.len() >= 2 {
                                // Keep ALL legal opponent replies (`m <= 6` on 3v3 idle roots) so NO
                                // refutation against any candidate in `s_own` can ever be truncated!
                                let mut s_opp: Vec<Vec<usize>> = Vec::with_capacity(root_types.len());
                                for t in &root_types {
                                    let w_rep = &worlds[t.rep_world_idx];
                                    let m = t.replies.min(nodes[w_rep.root_node_id].n_acts[opp] as usize);
                                    s_opp.push((0..m).collect());
                                }

                                let k_own = s_own.len();
                                let mut ver_cells: Vec<Vec<(u32, f64)>> = s_opp
                                    .iter()
                                    .map(|cols| vec![(0u32, 0.0f64); k_own * cols.len()])
                                    .collect();
                                let mut ver_type_n: Vec<u32> = vec![0u32; root_types.len()];
                                let mut rr_cursor: Vec<usize> = vec![0usize; root_types.len()];
                                let mut ver_own_n: Vec<u32> = vec![0u32; n_my];
                                let mut ver_own_w: Vec<f64> = vec![0.0f64; n_my];

                                // Cycle worlds identically to main loop (`vi % n_worlds`) to preserve exact
                                // Bayesian posterior type weights, advancing per-type round-robin cursor.
                                for vi in 0..verify_iters {
                                    let w = &worlds[(vi as usize) % n_worlds];
                                    let ti = w.type_index;
                                    let cols = &s_opp[ti];
                                    let m_sub = cols.len();
                                    if m_sub == 0 {
                                        continue;
                                    }
                                    let cur = rr_cursor[ti];
                                    rr_cursor[ti] = cur + 1;
                                    let oi = (cur / m_sub) % k_own;
                                    let oj = cur % m_sub;
                                    let my_i = s_own[oi];
                                    let opp_j = cols[oj];
                                    let v_root_id = w.root_node_id;
                                    if my_i >= nodes[v_root_id].n_acts[side] as usize
                                        || opp_j >= nodes[v_root_id].n_acts[opp] as usize
                                    {
                                        continue;
                                    }

                                    let mut cb = w.cb;
                                    {
                                        let mut prng = nc2000_engine::prng::Prng::new(self.rng.next());
                                        let qc = Battle::roll_quick_claw_with(w.table.ruleset, &mut prng);
                                        cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                                        cb.prng_seed = prng.seed();
                                    }
                                    if !hidden_timers.is_empty() {
                                        resample_compact_hidden_timers(
                                            &mut cb,
                                            &w.table,
                                            &hidden_timers,
                                            &mut timer_rng,
                                        );
                                    }

                                    let mut depth = 0u8;
                                    let mut cur_id = v_root_id;
                                    path.clear();

                                    let v_leaf = loop {
                                        if depth > 0 && use_puct {
                                            nodes[cur_id].ensure_puct_priors(
                                                &cb,
                                                &w.table,
                                                w.cache.as_ref(),
                                                puct_eps,
                                                use_cf_policy,
                                            );
                                        }
                                        let node = &nodes[cur_id];
                                        let n0_cur = node.n_acts[0] as usize;
                                        let n1_cur = node.n_acts[1] as usize;
                                        let (i0, i1) = if depth == 0 {
                                            if side == 0 { (my_i, opp_j) } else { (opp_j, my_i) }
                                        } else if use_puct {
                                            (
                                                node.select_puct(0, c_puct, fpu_red),
                                                node.select_puct(1, c_puct, fpu_red),
                                            )
                                        } else {
                                            (
                                                node.select_ucb(0, c_ucb, &mut self.rng, true),
                                                node.select_ucb(1, c_ucb, &mut self.rng, true),
                                            )
                                        };
                                        let a0 = if n0_cur > 0 { node.acts[0][i0] } else { 7 };
                                        let a1 = if n1_cur > 0 { node.acts[1][i1] } else { 7 };
                                        // Skip depth == 0 so root UCB stats stay untouched; deepen interior nodes!
                                        if depth > 0 && (n0_cur > 1 || n1_cur > 1) {
                                            path.push((cur_id, i0, i1));
                                        }

                                        cb.apply_actions(&w.table, [a0, a1]);
                                        depth = depth.saturating_add(1);

                                        if cb.outcome != 0 {
                                            break eval_duct_leaf_temp(
                                                &cb,
                                                &w.table,
                                                leaf_mode,
                                                w.cache.as_ref(),
                                                leaf_temp,
                                            );
                                        }
                                        if let Some(v) = compact_turn_limit_outcome(&cb, &w.table) {
                                            break v;
                                        }
                                        if depth >= max_depth || cb.turn > turn_cap {
                                            break eval_duct_leaf_temp(
                                                &cb,
                                                &w.table,
                                                leaf_mode,
                                                w.cache.as_ref(),
                                                leaf_temp,
                                            );
                                        }

                                        let omit_qc = !cb.quick_claw_observable(&w.table);
                                        let next_key = cb.search_key(&w.table, 16, omit_qc);
                                        if let Some(&next_id) = index.get(&next_key) {
                                            cur_id = next_id;
                                        } else {
                                            let new_id = nodes.len();
                                            nodes.push(DuctNode::new(&cb, &w.table));
                                            index.insert(next_key, new_id);
                                            if rollout_turns == 0 {
                                                if leaf_mode == TempoLeafMode::CfValue
                                                    && (cb.sides[0].alive_mask.count_ones()
                                                        + cb.sides[1].alive_mask.count_ones())
                                                        <= 3
                                                {
                                                    let cutoff =
                                                        turn_cap.min(cb.turn.saturating_add(7));
                                                    let ew = EvalWeights::default();
                                                    break crate::mcts::compact_playout_with_leaf(
                                                        &mut cb,
                                                        &w.table,
                                                        &cf_eg_playout,
                                                        cutoff,
                                                        &mut self.rng,
                                                        rollout_rules,
                                                        |cb_leaf, tbl_leaf| {
                                                            compact_eval_leaf_blindspot(
                                                                cb_leaf, tbl_leaf, &ew,
                                                            )
                                                        },
                                                    );
                                                }
                                                if leaf_mode == TempoLeafMode::CfValue {
                                                    break crate::compact_az::evaluate_cf_value_resolving_switches(
                                                        &cb,
                                                        &w.table,
                                                        w.cache.as_ref(),
                                                        leaf_temp,
                                                    );
                                                }
                                                if leaf_mode == TempoLeafMode::AzValue
                                                    && (cb.sides[0].request == 2
                                                        || cb.sides[1].request == 2)
                                                {
                                                    resolve_compact_forced_switches(
                                                        &mut cb,
                                                        &w.table,
                                                        &EvalWeights::default(),
                                                    );
                                                }
                                                break eval_duct_leaf_temp(
                                                    &cb,
                                                    &w.table,
                                                    leaf_mode,
                                                    w.cache.as_ref(),
                                                    leaf_temp,
                                                );
                                            } else {
                                                let cutoff = turn_cap
                                                    .min(cb.turn.saturating_add(rollout_turns.saturating_sub(1)));
                                                break crate::mcts::compact_playout_with_leaf(
                                                    &mut cb,
                                                    &w.table,
                                                    &rollout_playout,
                                                    cutoff,
                                                    &mut self.rng,
                                                    rollout_rules,
                                                    |cb_leaf, tbl_leaf| {
                                                        eval_duct_leaf_temp(
                                                            cb_leaf,
                                                            tbl_leaf,
                                                            leaf_mode,
                                                            w.cache.as_ref(),
                                                            leaf_temp,
                                                        )
                                                    },
                                                );
                                            }
                                        }
                                    };

                                    let mut val = v_leaf;
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
                                        if val > 0.000_001 && val < 0.999_999 {
                                            val = n.backed_up_value_cfg(val, backup_lambda, backup_tau);
                                        }
                                    }

                                    let mine = if side == 0 { val } else { 1.0 - val };
                                    ver_type_n[ti] += 1;
                                    let cell = &mut ver_cells[ti][oi * m_sub + oj];
                                    cell.0 += 1;
                                    cell.1 += mine;
                                    ver_own_n[my_i] += 1;
                                    ver_own_w[my_i] += mine;
                                }

                                // Solve Bayesian LP over merged `late_cells + ver_cells` with cross-type
                                // verify-only `pooled_priors` so sparse minor types never suffer heavy
                                // uniform-reply shrinkage while every `s_own` action is unstarved & admitted.
                                let mut ver_elig = vec![false; n_my];
                                for &i in &s_own {
                                    if ver_own_n[i] >= 16 {
                                        ver_elig[i] = true;
                                    }
                                }
                                let mut ver_pooled: FxHashMap<(usize, ReplyPoolKey), (u32, f64)> =
                                    FxHashMap::default();
                                for (ti, cols) in s_opp.iter().enumerate() {
                                    let m_sub = cols.len();
                                    let t = &root_types[ti];
                                    for oi in 0..k_own {
                                        let own_i = s_own[oi];
                                        for (oj, &orig_j) in cols.iter().enumerate() {
                                            let (vn, vw) = ver_cells[ti][oi * m_sub + oj];
                                            if vn > 0 {
                                                if let Some(&rkey) = t.reply_keys.get(orig_j) {
                                                    let e = ver_pooled
                                                        .entry((own_i, rkey))
                                                        .or_insert((0, 0.0));
                                                    e.0 += vn;
                                                    e.1 += vw;
                                                }
                                            }
                                        }
                                    }
                                }
                                let ver_means: Vec<f64> = (0..n_my)
                                    .map(|i| {
                                        if ver_own_n[i] > 0 {
                                            ver_own_w[i] / f64::from(ver_own_n[i])
                                        } else {
                                            0.5
                                        }
                                    })
                                    .collect();
                                let mut ver_types: Vec<RootType> = Vec::with_capacity(root_types.len());
                                let mut ver_priors: Vec<Vec<RootCell>> =
                                    Vec::with_capacity(root_types.len());
                                for (ti, cols) in s_opp.iter().enumerate() {
                                    let m_sub = cols.len();
                                    if ver_type_n[ti] == 0 || m_sub == 0 {
                                        continue;
                                    }
                                    let t = &root_types[ti];
                                    let src_main = if t.late_samples >= 16 {
                                        &t.late_cells
                                    } else {
                                        &t.cells
                                    };
                                    let mut cells = Vec::with_capacity(k_own * m_sub);
                                    let mut priors = Vec::new();
                                    let mut merged_samples = 0u32;
                                    for oi in 0..k_own {
                                        let own_i = s_own[oi];
                                        for (oj, &orig_j) in cols.iter().enumerate() {
                                            let (vn, vw) = ver_cells[ti][oi * m_sub + oj];
                                            // Merge capped mature main-tree evidence (up to 4x verify cell)
                                            // so well-sampled replies retain full precision while starved replies
                                            // are dominated by fresh unconfounded deep-tree returns.
                                            let (mn, mw) = src_main[own_i * t.replies + orig_j];
                                            let cap_m = vn.saturating_mul(4);
                                            let (cn, cw) = if mn > 0 && cap_m > 0 {
                                                let use_m = mn.min(cap_m);
                                                let use_w = mw * (f64::from(use_m) / f64::from(mn));
                                                (vn + use_m, vw + use_w)
                                            } else {
                                                (vn, vw)
                                            };
                                            if cn > 0 {
                                                merged_samples = merged_samples.saturating_add(cn);
                                                cells.push(RootCell {
                                                    own: own_i,
                                                    reply: oj,
                                                    samples: cn,
                                                    mean: cw / f64::from(cn),
                                                });
                                            }
                                            if let Some(&rkey) = t.reply_keys.get(orig_j) {
                                                if let Some(&(pn, pw)) = ver_pooled.get(&(own_i, rkey)) {
                                                    if pn >= 8 {
                                                        priors.push(RootCell {
                                                            own: own_i,
                                                            reply: oj,
                                                            samples: pn,
                                                            mean: pw / f64::from(pn),
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if !cells.is_empty() && merged_samples > 0 {
                                        ver_types.push(RootType {
                                            samples: ver_type_n[ti],
                                            replies: m_sub,
                                            cells,
                                        });
                                        ver_priors.push(priors);
                                    }
                                }
                                if ver_elig[base_idx] && !ver_types.is_empty() {
                                    if let Some(eq) = solve_bayesian_equilibrium_with_priors(
                                        &ver_elig,
                                        &ver_own_n,
                                        &ver_means,
                                        &ver_types,
                                        &ver_priors,
                                    ) {
                                        if eq.override_value
                                            > eq.pure_guarantees[base_idx] + MIN_MIX_MARGIN
                                        {
                                            let draw = SplitMix64::new(
                                                decision_seed ^ u64::from(done) ^ 0x5645_5249_4659,
                                            )
                                            .next_f64();
                                            if let Some(vp) =
                                                crate::blind::draw_policy(&my_acts, &eq.policy, draw)
                                            {
                                                final_pick = vp;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                chosen_opt = Some(final_pick);
                break;
            }

            // Contested multi-tree root aggregation: merge full-depth tree stats across independent seeds/world plans.
            total_done = total_done.saturating_add(done);
            for i in 0..n_my {
                agg_my_n[i] = agg_my_n[i].saturating_add(my_n[i]);
                agg_my_w[i] += my_w[i];
                agg_ord_n[i] = agg_ord_n[i].saturating_add(ord_n[i]);
                agg_ord_w[i] += ord_w[i];
            }
            for t in root_types {
                if let Some(&ai) = agg_root_type_index.get(&t.tkey) {
                    let dst = &mut agg_root_types[ai];
                    if dst.replies == t.replies && dst.cells.len() == t.cells.len() {
                        dst.samples = dst.samples.saturating_add(t.samples);
                        dst.late_samples = dst.late_samples.saturating_add(t.late_samples);
                        for (dc, sc) in dst.cells.iter_mut().zip(t.cells.iter()) {
                            dc.0 = dc.0.saturating_add(sc.0);
                            dc.1 += sc.1;
                        }
                        for (dc, sc) in dst.late_cells.iter_mut().zip(t.late_cells.iter()) {
                            dc.0 = dc.0.saturating_add(sc.0);
                            dc.1 += sc.1;
                        }
                        for (dv, &sv) in dst.col_visits.iter_mut().zip(t.col_visits.iter()) {
                            *dv = dv.saturating_add(sv);
                        }
                        dst.world_count = dst.world_count.saturating_add(t.world_count);
                    }
                } else {
                    let ai = agg_root_types.len();
                    agg_root_type_index.insert(t.tkey, ai);
                    agg_root_types.push(t);
                }
            }
            for (own, rkey, jn, jw) in joint {
                let ji = *agg_joint_index.entry((own, rkey)).or_insert_with(|| {
                    let k = agg_joint.len();
                    agg_joint.push((own, rkey, 0, 0.0));
                    k
                });
                agg_joint[ji].2 = agg_joint[ji].2.saturating_add(jn);
                agg_joint[ji].3 += jw;
            }

            if tree_k + 1 == max_trees {
                chosen_opt = Some(select_root(
                    total_done,
                    &endgame,
                    &agg_my_n,
                    &agg_my_w,
                    &agg_ord_n,
                    &agg_ord_w,
                    &agg_root_types,
                    &agg_joint,
                    &agg_joint_index,
                ));
            }
        }

        self.nodes_buf = nodes;
        self.index_buf = index;
        chosen_opt.unwrap_or(my_acts[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use serde_json::json;

    fn test_dex() -> Dex {
        conformance::load_dex()
    }

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

    fn start(dex: &Dex, p1: &[PokemonSet], p2: &[PokemonSet]) -> Battle {
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
    fn tempo_poc_core_invariants() {
        let dex = test_dex();

        // 1. Exact mirror symmetry: G0 == G1, ΔG == 0, WinProb == 0.5.
        let zap = vec![mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"])];
        let b_mirror = start(&dex, &zap, &zap);
        let ev_mirror = evaluate_tempo(&b_mirror, &dex);
        assert!(ev_mirror.turn_advantage.abs() < 1e-9);
        assert!((ev_mirror.win_prob - 0.5).abs() < 1e-9);

        // 2. Recover threshold: <50% wall wins, >65% breaker wins, 50%-HP Starmie picks Recover.
        let starmie = vec![mk("Starmie", 50, "", &["Psychic", "Recover"])];
        let vap = vec![mk("Vaporeon", 50, "", &["Surf"])];
        let ev_wall = evaluate_tempo(&start(&dex, &vap, &starmie), &dex);
        assert!(ev_wall.turn_advantage < -5.0, "Starmie must wall Vaporeon: {:?}", ev_wall.g_turns);

        let zap_tbolt = vec![mk("Zapdos", 50, "", &["Thunderbolt"])];
        let ev_break = evaluate_tempo(&start(&dex, &zap_tbolt, &starmie), &dex);
        assert!(ev_break.turn_advantage > 1.0, "Zapdos must break Starmie: {:?}", ev_break.g_turns);

        // 3. Doomed vs Safe Setup: Marowak SDs vs Snorlax, never SDs vs OHKO Starmie.
        let wak_team = vec![
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake"]),
        ];
        let lax_team = vec![
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
        ];
        let mut b_safe = start(&dex, &wak_team, &lax_team);
        let cs_safe = b_safe.legal_choices(&dex, 0);
        let m_safe = analyze_turn_matrix(&b_safe, &dex, 0, &cs_safe, 16, 7);
        let sd_idx = cs_safe.iter().position(|c| c.to_input(&dex) == "move swordsdance").unwrap();
        assert!(m_safe.own_policy[sd_idx] > 0.5, "Marowak should Swords Dance vs Snorlax");

        let starmie_team = vec![
            mk("Starmie", 50, "Leftovers", &["Surf", "Recover"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Rest"]),
        ];
        let mut b_doom = start(&dex, &wak_team, &starmie_team);
        let cs_doom = b_doom.legal_choices(&dex, 0);
        let m_doom = analyze_turn_matrix(&b_doom, &dex, 0, &cs_doom, 16, 7);
        let sd_idx2 = cs_doom.iter().position(|c| c.to_input(&dex) == "move swordsdance").unwrap();
        assert!(m_doom.own_policy[sd_idx2] < 0.05, "Marowak must not SD vs faster OHKO Starmie");

        // 4. Phazing: Skarmory picks Whirlwind against +2 Curse Snorlax.
        let p1_phaze = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice"]),
        ];
        let p2_phaze = vec![
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest"]),
            mk("Machamp", 50, "Leftovers", &["Cross Chop", "Earthquake"]),
        ];
        let mut b_phaze = start(&dex, &p1_phaze, &p2_phaze);
        if let Some(id) = b_phaze.active_id(0) {
            let p = b_phaze.poke_mut(id);
            p.boosts[0] = 2;
            p.boosts[1] = 2;
        }
        let cs_phaze = b_phaze.legal_choices(&dex, 1);
        let m_phaze = analyze_turn_matrix(&b_phaze, &dex, 1, &cs_phaze, 16, 7);
        let ww_idx = cs_phaze.iter().position(|c| c.to_input(&dex) == "move whirlwind").unwrap();
        assert!(m_phaze.own_policy[ww_idx] > 0.8, "Skarmory must Whirlwind +2 Snorlax");

        // 5. Role Preservation & No Sac-Cliff: switching Cloyster -> Snorlax beats dying for free to Thunderbolt.
        let p1_r = vec![
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
        ];
        let p2_r = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest"]),
        ];
        let mut b_role = start(&dex, &p1_r, &p2_r);
        let ev_alive = evaluate_tempo(&b_role, &dex);
        let cs_role = b_role.legal_choices(&dex, 0);
        let m_role = analyze_turn_matrix(&b_role, &dex, 0, &cs_role, 16, 7);
        let surf_i = cs_role.iter().position(|c| c.to_input(&dex) == "move surf").unwrap();
        let sw_lax_i = cs_role.iter().position(|c| c.to_input(&dex) == "switch 2").unwrap();
        let tbolt_j = m_role.opp_choices.iter().position(|c| c.to_input(&dex) == "move thunderbolt").unwrap();
        let n_opp = m_role.opp_choices.len();
        assert!(
            m_role.turn_matrix[sw_lax_i * n_opp + tbolt_j]
                > m_role.turn_matrix[surf_i * n_opp + tbolt_j],
            "Switching to Snorlax must beat letting Cloyster die for free to Thunderbolt: sw={:+.2}t vs die={:+.2}t",
            m_role.turn_matrix[sw_lax_i * n_opp + tbolt_j],
            m_role.turn_matrix[surf_i * n_opp + tbolt_j],
        );

        let zap_roster = b_role.sides[0].party[2] as usize;
        b_role.sides[0].roster[zap_roster].hp = 0;
        b_role.sides[0].roster[zap_roster].status = Status::Fnt;
        b_role.sides[0].pokemon_left = 2;
        let ev_dead = evaluate_tempo(&b_role, &dex);
        assert!(
            ev_dead.turn_advantage < ev_alive.turn_advantage - 3.0,
            "Losing Zapdos vs Skarmory must severely worsen turn advantage"
        );

        // 6. HP Monotonicity: chipping a Rest Snorlax must never make Snorlax's side look better!
        let mut b_chip = start(&dex, &p1_r, &p2_r);
        let ev_full = evaluate_tempo(&b_chip, &dex);
        let lax_r = b_chip.sides[1].party[2] as usize;
        b_chip.sides[1].roster[lax_r].hp = (b_chip.sides[1].roster[lax_r].maxhp * 85) / 100;
        let ev_chipped = evaluate_tempo(&b_chip, &dex);
        assert!(
            ev_chipped.turn_advantage >= ev_full.turn_advantage - 0.05,
            "Chipping opponent Snorlax must not worsen P1 turn advantage: full={:+.2}t chipped={:+.2}t",
            ev_full.turn_advantage,
            ev_chipped.turn_advantage
        );

        // 7. DUCT MCTS search depth scales monotonically with iteration budget.
        let mut b_mcts = start(&dex, &p1_r, &p2_r);
        let cs_mcts = b_mcts.legal_choices(&dex, 0);
        let r30 = search_tempo_mcts(
            &b_mcts,
            &dex,
            0,
            &cs_mcts,
            TempoMctsConfig { iterations: 30, ..TempoMctsConfig::default() },
            11,
        );
        let r300 = search_tempo_mcts(
            &b_mcts,
            &dex,
            0,
            &cs_mcts,
            TempoMctsConfig { iterations: 300, ..TempoMctsConfig::default() },
            11,
        );
        let r1500 = search_tempo_mcts(
            &b_mcts,
            &dex,
            0,
            &cs_mcts,
            TempoMctsConfig { iterations: 1500, ..TempoMctsConfig::default() },
            11,
        );
        assert!(
            r1500.mean_depth > r300.mean_depth && r300.mean_depth > r30.mean_depth,
            "mean_depth must scale with budget: 30={:.2}, 300={:.2}, 1500={:.2}",
            r30.mean_depth,
            r300.mean_depth,
            r1500.mean_depth
        );
        let psum: f64 = r300.own_policy.iter().sum();
        assert!((psum - 1.0).abs() < 1e-6, "policy must sum to 1, got {psum}");

        // 8. FastGolf (v2) exact mirror symmetry & wall-chipping gradient (no 24t saturation!).
        let (t_m, cb_m) = CompactBattle::from_battle(&b_mirror, &dex);
        let c_m = TempoMatchCache::build(&cb_m, &t_m);
        let fg_m = evaluate_fast_golf(&cb_m, &t_m, Some(&c_m));
        assert!(fg_m.turn_advantage.abs() < 1e-9);
        assert!((fg_m.win_prob - 0.5).abs() < 1e-9);

        // Chipping a Rest Snorlax from 100% to 55% must produce strictly positive gradient (fixes 24t saturation!).
        let b_lax = start(&dex, &p1_r, &p2_r);
        let (t_l, cb_l) = CompactBattle::from_battle(&b_lax, &dex);
        let c_l = TempoMatchCache::build(&cb_l, &t_l);
        let fg_full = evaluate_fast_golf(&cb_l, &t_l, Some(&c_l));
        let mut cb_l_chip = cb_l;
        cb_l_chip.mons[5].hp_pool[0] = (t_l.mons[5].maxhp * 55) / 100;
        let fg_chip = evaluate_fast_golf(&cb_l_chip, &t_l, Some(&c_l));
        assert!(
            fg_chip.turn_advantage > fg_full.turn_advantage + 0.3,
            "FastGolf must give positive gradient for chipping a Rest wall: full={:+.2}t chip={:+.2}t",
            fg_full.turn_advantage,
            fg_chip.turn_advantage
        );

        // 9. FastGolf v8 invariants:
        // (a) Strict bench-HP monotonicity (losing HP on ANY living ally must never improve own score).
        let mut cb_own_chip = cb_l;
        cb_own_chip.mons[1].hp_pool[0] = (t_l.mons[1].maxhp * 50) / 100;
        let fg_own_chip = evaluate_fast_golf(&cb_own_chip, &t_l, Some(&c_l));
        assert!(
            fg_own_chip.turn_advantage < fg_full.turn_advantage - 0.05,
            "Chipping own bench mon must strictly worsen turn advantage: full={:+.2}t own_chip={:+.2}t",
            fg_full.turn_advantage,
            fg_own_chip.turn_advantage
        );

        // (b) Cached O(1) boost scaling stays accurate vs uncached DmgKernel1D on boosted states.
        let mut cb_boost = cb_l;
        cb_boost.mons[0].boosts[0] = 2;
        cb_boost.mons[0].boosts[1] = 2;
        let fg_cached_b = evaluate_fast_golf(&cb_boost, &t_l, Some(&c_l));
        let fg_uncached_b = evaluate_fast_golf(&cb_boost, &t_l, None);
        assert!(
            (fg_cached_b.win_prob - fg_uncached_b.win_prob).abs() < 0.05,
            "Cached O(1) boost scaling must closely match uncached kernel: cached={:.3} uncached={:.3}",
            fg_cached_b.win_prob,
            fg_uncached_b.win_prob
        );

        // (c) Last-enemy Explosion in 2v1 must win cleanly (zero sacrifice penalty on final enemy).
        let p1_boom = vec![
            mk("Forretress", 50, "Leftovers", &["Explosion", "Spikes"]),
            mk("Starmie", 50, "Leftovers", &["Surf", "Recover"]),
        ];
        let p2_sole = vec![mk("Tauros", 50, "Leftovers", &["Body Slam", "Earthquake"])];
        let b_boom = start(&dex, &p1_boom, &p2_sole);
        let (t_b, cb_b) = CompactBattle::from_battle(&b_boom, &dex);
        let c_b = TempoMatchCache::build(&cb_b, &t_b);
        let fg_boom = evaluate_fast_golf(&cb_b, &t_b, Some(&c_b));
        assert!(
            fg_boom.g_turns[0] < 2.2 && fg_boom.win_prob > 0.70,
            "2v1 with Forretress Explosion on last enemy must score fast win: {:?}",
            fg_boom
        );

        // (d) Last-enemy Perish Song / Destiny Bond checkmate in 2v1 vs Recover wall.
        let p1_perish = vec![
            mk("Gengar", 50, "Leftovers", &["Fire Punch", "Perish Song", "Destiny Bond", "Mean Look"]),
            mk("Starmie", 50, "Leftovers", &["Surf", "Recover"]),
        ];
        let p2_wall = vec![mk("Porygon2", 50, "Leftovers", &["Return", "Ice Beam", "Curse", "Recover"])];
        let b_perish = start(&dex, &p1_perish, &p2_wall);
        let (t_p, cb_p) = CompactBattle::from_battle(&b_perish, &dex);
        let c_p = TempoMatchCache::build(&cb_p, &t_p);
        let fg_perish = evaluate_fast_golf(&cb_p, &t_p, Some(&c_p));
        assert!(
            fg_perish.g_turns[0] < 3.6 && fg_perish.win_prob > 0.58,
            "2v1 with faster Gengar Perish/DBond vs lone wall must favor 2-mon side: {:?}",
            fg_perish
        );

        // (e) Duration-weighted Reflect: 1-turn remaining Reflect must not look as strong as 5-turn Reflect.
        let mut cb_r5 = cb_l;
        cb_r5.sides[1].reflect_turns = 5;
        let mut cb_r1 = cb_l;
        cb_r1.sides[1].reflect_turns = 1;
        let fg_r5 = evaluate_fast_golf(&cb_r5, &t_l, Some(&c_l));
        let fg_r1 = evaluate_fast_golf(&cb_r1, &t_l, Some(&c_l));
        assert!(
            fg_r1.turn_advantage > fg_r5.turn_advantage + 0.05,
            "Expiring 1-turn Reflect must give less foe defense than fresh 5-turn Reflect: r1={:+.2}t r5={:+.2}t",
            fg_r1.turn_advantage,
            fg_r5.turn_advantage
        );

        // 10. Minimax-backup + argmax-root + micro-rollout (`cmcts`) & `SkuctGolfAgent` smoke + dominated guard.
        let mut cmcts_agent = TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: 60,
                c: 0.4,
                max_depth: 64,
                leaf: TempoLeafMode::FastGolf,
                deterministic: true,
                fixed_preview: true,
                minimax_backup: true,
                argmax_root: true,
                rollout_turns: 2,
            },
            99,
        );
        let pick_cmcts = cmcts_agent.choose(&b_mcts, &dex, 0, &cs_mcts);
        assert!(cs_mcts.contains(&pick_cmcts));

        let mut skuct_golf = SkuctGolfAgent::new(
            SkuctGolfConfig {
                iterations: 40,
                c: 0.4,
                leaf: TempoLeafMode::FastGolf,
                rollout_turns: 2,
                fixed_preview: true,
            },
            99,
        );
        let pick_skuct = skuct_golf.choose(&b_mcts, &dex, 0, &cs_mcts);
        assert!(cs_mcts.contains(&pick_skuct));
    }

    #[test]
    fn tempo_engine_parity_fixes_ps_and_castle() {
        let dex = test_dex();

        // 1. Fresh Toxic packs `0x10` in `status_counter()` (`(step & 0x0F) | 0x10`).
        //    FastGolf & TempoFull must NOT treat fresh Toxic as 15/16 (93.75%/turn) instant KO!
        let p1_wall = vec![mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake"])];
        let p2_wall = vec![mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake"])];
        let b_tox = start(&dex, &p1_wall, &p2_wall);
        let (t_tox, cb_clean) = CompactBattle::from_battle(&b_tox, &dex);
        let c_tox = TempoMatchCache::build(&cb_clean, &t_tox);
        let mut cb_fresh_tox = cb_clean;
        cb_fresh_tox.mons[3].set_status(Status::Tox as u8, 0x10); // fresh Toxic: next tick is 1/16!
        let mut cb_deep_tox = cb_clean;
        cb_deep_tox.mons[3].set_status(Status::Tox as u8, 0x10 | 6); // 6 ticks in: next tick is 7/16!
        let fg_fresh = evaluate_fast_golf(&cb_fresh_tox, &t_tox, Some(&c_tox));
        let fg_deep = evaluate_fast_golf(&cb_deep_tox, &t_tox, Some(&c_tox));
        assert!(
            fg_fresh.g_turns[0] > 2.4 && fg_deep.g_turns[0] < fg_fresh.g_turns[0] - 0.5,
            "fresh Toxic (0x10) must not collapse to 1t KO and must be slower than deep Toxic: fresh={:.2}t deep={:.2}t",
            fg_fresh.g_turns[0],
            fg_deep.g_turns[0]
        );
        let te_fresh = evaluate_compact_tempo(&cb_fresh_tox, &t_tox);
        let te_deep = evaluate_compact_tempo(&cb_deep_tox, &t_tox);
        assert!(
            te_fresh.g_turns[0] > 2.4 && te_deep.g_turns[0] < te_fresh.g_turns[0] - 0.5,
            "TempoFull fresh Toxic must not clamp to 15/16: fresh={:.2}t deep={:.2}t",
            te_fresh.g_turns[0],
            te_deep.g_turns[0]
        );

        // 2. PS `PAR_SPEED_DROP` and `BRN_ATK_DROP` clearing vs Castle persistence.
        let p1_par = vec![mk("Zapdos", 50, "Leftovers", &["Drill Peck", "Thunderbolt"])];
        let p2_par = vec![mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake"])];
        let b_ps = start(&dex, &p1_par, &p2_par);
        let (t_ps, mut cb_ps) = CompactBattle::from_battle(&b_ps, &dex);
        cb_ps.mons[0].set_status(Status::Par as u8, 0);
        cb_ps.mons[0].vol_flags |= vol_bit::PAR_SPEED_DROP;
        let spe_par_drop = effective_speed(&cb_ps, &t_ps, 0, 0);
        cb_ps.mons[0].vol_flags &= !vol_bit::PAR_SPEED_DROP; // cleared by Haze/boost in PS
        let spe_par_cleared = effective_speed(&cb_ps, &t_ps, 0, 0);
        assert!(
            spe_par_cleared > spe_par_drop * 3,
            "PS cleared PAR_SPEED_DROP must restore full speed: drop={spe_par_drop} cleared={spe_par_cleared}"
        );

        let mut cb_ps_brn_drop = cb_ps;
        cb_ps_brn_drop.mons[0].set_status(Status::Brn as u8, 0);
        cb_ps_brn_drop.mons[0].vol_flags = vol_bit::BRN_ATK_DROP;
        let mut cb_ps_brn_clr = cb_ps_brn_drop;
        cb_ps_brn_clr.mons[0].vol_flags = 0; // BRN_ATK_DROP cleared in PS
        let c_ps = TempoMatchCache::build(&cb_ps, &t_ps);
        let fg_brn_drop = evaluate_fast_golf(&cb_ps_brn_drop, &t_ps, Some(&c_ps));
        let fg_brn_clr = evaluate_fast_golf(&cb_ps_brn_clr, &t_ps, Some(&c_ps));
        assert!(
            fg_brn_clr.g_turns[0] < fg_brn_drop.g_turns[0] + 0.01,
            "PS cleared BRN_ATK_DROP must not penalize physical attack more than active burn drop"
        );

        // 3. `encode_choice` + `search_tempo_mcts` exact alignment when active mon is forced to Struggle alongside bench switches!
        let p1_str = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt"]),
        ];
        let p2_str = vec![
            mk("Machamp", 50, "Leftovers", &["Cross Chop"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam"]),
        ];
        let mut b_str = start(&dex, &p1_str, &p2_str);
        if let Some(id) = b_str.active_id(0) {
            b_str.poke_mut(id).move_slots[0].pp = 0; // force Struggle!
        }
        let (t_str, cb_str) = CompactBattle::from_battle(&b_str, &dex);
        let cs_str = b_str.legal_choices(&dex, 0);
        assert_eq!(cs_str.len(), 2, "expected [Struggle, Switch(2)], got {:?}", cs_str);
        assert_eq!(encode_choice(&t_str, &cb_str, &dex, 0, cs_str[0]), Some(0));
        assert_eq!(encode_choice(&t_str, &cb_str, &dex, 0, cs_str[1]), Some(5));
        let res_str = search_tempo_mcts(
            &b_str,
            &dex,
            0,
            &cs_str,
            TempoMctsConfig {
                iterations: 80,
                c: 0.4,
                max_depth: 16,
                leaf: TempoLeafMode::FastGolf,
                deterministic: true,
                fixed_preview: true,
                minimax_backup: true,
                argmax_root: true,
                rollout_turns: 2,
            },
            123,
        );
        assert_eq!(res_str.visits.len(), 2);
        assert_eq!(res_str.visits[0] + res_str.visits[1], 80);
        assert!(res_str.visits[0] > 0 && res_str.visits[1] > 0);
        let psum_str: f64 = res_str.own_policy.iter().sum();
        assert!((psum_str - 1.0).abs() < 1e-6);

        // 4. `compact_duct_key` must distinguish legality-affecting states (Disable/Encore slots, PP==0,
        //    two_turn_locked, partially_trapped, bp_switch_flags, Castle 5-turn sleep).
        let (t_k, cb_k) = CompactBattle::from_battle(&b_ps, &dex);
        let mut cb_dis0 = cb_k;
        cb_dis0.mons[0].vol_flags |= vol_bit::DISABLE;
        cb_dis0.mons[0].set_disable_slot(0);
        let mut cb_dis1 = cb_dis0;
        cb_dis1.mons[0].set_disable_slot(1);
        assert_ne!(compact_duct_key(&cb_dis0, &t_k, 1), compact_duct_key(&cb_dis1, &t_k, 1));

        let mut cb_enc0 = cb_k;
        cb_enc0.mons[0].vol_flags |= vol_bit::ENCORE;
        cb_enc0.mons[0].set_encore_slot(0);
        let mut cb_enc1 = cb_enc0;
        cb_enc1.mons[0].set_encore_slot(1);
        assert_ne!(compact_duct_key(&cb_enc0, &t_k, 1), compact_duct_key(&cb_enc1, &t_k, 1));

        let mut cb_pp0 = cb_k;
        cb_pp0.mons[0].pp[0] = 0;
        assert_ne!(compact_duct_key(&cb_k, &t_k, 1), compact_duct_key(&cb_pp0, &t_k, 1));

        let mut cb_ptrap = cb_k;
        cb_ptrap.mons[0].set_partially_trapped(true);
        assert_ne!(compact_duct_key(&cb_k, &t_k, 1), compact_duct_key(&cb_ptrap, &t_k, 1));

        let mut cb_ttm = cb_k;
        cb_ttm.mons[0].set_two_turn_locked(true);
        cb_ttm.mons[0].set_locked_move_slot(1);
        assert_ne!(compact_duct_key(&cb_k, &t_k, 1), compact_duct_key(&cb_ttm, &t_k, 1));

        let mut cb_slp4 = cb_k;
        cb_slp4.mons[0].set_status(Status::Slp as u8, 4);
        let mut cb_slp5 = cb_k;
        cb_slp5.mons[0].set_status(Status::Slp as u8, 5);
        assert_ne!(compact_duct_key(&cb_slp4, &t_k, 1), compact_duct_key(&cb_slp5, &t_k, 1));

        // 5. Castle PBS ruleset smoke test: `effective_speed` keeps Par penalty, `evaluate_fast_golf`
        //    honors `L_TURN` terminal adjudication, and `cmcts` runs cleanly on Castle battles.
        let mut b_castle = Battle::from_fixture_with_ruleset(
            &dex,
            "1,2,3,4",
            &p1_str,
            &p2_str,
            Ruleset::CastlePbs(nc2000_engine::rules::CastleOverlayConfig::tournament()),
        )
        .unwrap();
        b_castle.choose(&dex, 0, "team 1,2").unwrap();
        b_castle.choose(&dex, 1, "team 1,2").unwrap();
        let (t_c, mut cb_c) = CompactBattle::from_battle(&b_castle, &dex);
        cb_c.mons[0].set_status(Status::Par as u8, 0);
        cb_c.mons[0].vol_flags = 0; // In Castle, Par penalty persists even without PAR_SPEED_DROP!
        let spe_c_par = effective_speed(&cb_c, &t_c, 0, 0);
        cb_c.mons[0].set_status(Status::None as u8, 0);
        let spe_c_clean = effective_speed(&cb_c, &t_c, 0, 0);
        assert_eq!(spe_c_par * 4, spe_c_clean);

        // Castle L_TURN terminal adjudication (`turn > 80`): higher alive count / HP ratio wins immediately.
        let mut cb_lim = cb_c;
        cb_lim.turn = 81;
        cb_lim.mons[3].hp_pool[0] = t_c.mons[3].maxhp / 2;
        assert!((evaluate_fast_golf(&cb_lim, &t_c, None).win_prob - 1.0).abs() < 1e-9);
        assert!((evaluate_compact_tempo(&cb_lim, &t_c).win_prob - 1.0).abs() < 1e-9);

        let cs_c = b_castle.legal_choices(&dex, 0);
        let mut cmcts_c = TempoMctsAgent::new(
            TempoMctsConfig {
                iterations: 80,
                c: 0.4,
                max_depth: 32,
                leaf: TempoLeafMode::FastGolf,
                deterministic: true,
                fixed_preview: true,
                minimax_backup: true,
                argmax_root: true,
                rollout_turns: 2,
            },
            77,
        );
        let pick_c = cmcts_c.choose(&b_castle, &dex, 0, &cs_c);
        assert!(cs_c.contains(&pick_c));

        // 6. `TempoMatchCache` remaps `party_roster` cleanly after a `Battle` switch (fixes `SkuctGolfAgent`
        //    post-switch cache corruption when `Battle` permutes `side.party`).
        let mut b_sw = start(&dex, &p1_str, &p2_str);
        let (t_root, cb_root) = CompactBattle::from_battle(&b_sw, &dex);
        let root_cache = TempoMatchCache::build(&cb_root, &t_root);
        b_sw.choose(&dex, 0, "switch 2").unwrap();
        b_sw.choose(&dex, 1, "switch 2").unwrap();
        let (t_post, cb_post) = CompactBattle::from_battle(&b_sw, &dex);
        assert_ne!(cb_post.sides[0].party_roster, cb_root.sides[0].party_roster);
        let fg_cached_sw = evaluate_fast_golf(&cb_post, &t_post, Some(&root_cache));
        let fg_uncached_sw = evaluate_fast_golf(&cb_post, &t_post, None);
        assert!(
            (fg_cached_sw.win_prob - fg_uncached_sw.win_prob).abs() < 1e-6,
            "TempoMatchCache must match uncached FastGolf exactly after Battle party permutation: cached={:.6} uncached={:.6}",
            fg_cached_sw.win_prob,
            fg_uncached_sw.win_prob
        );
    }

    #[test]
    fn blind_golf_agent_production_stack_smoke() {
        let dex = test_dex();
        let root = conformance::fixture::repo_root();
        let pool = Arc::new(crate::preview::load_meta_pool(
            &root.join("data/belief-pool-v3/belief-pool.json"),
        ));
        let t0 = &pool.teams[0].sets;
        let t1 = &pool.teams[1].sets;
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", t0, t1).unwrap();
        battle.set_log_enabled(true);
        let mut p1 = BlindGolfAgent::new(
            BlindGolfConfig {
                iterations: 200,
                c: 0.4,
                leaf: TempoLeafMode::FastGolf,
                rollout_turns: 2,
                preview_iterations: 200,
                own_preview_triple: Some([1, 2, 3]),
            },
            pool.clone(),
            42,
        );
        let mut p2 = crate::blind::BlindAgent::new(
            RmConfig { iterations: 200, c: 0.4, rule: SelRule::Ucb, ..RmConfig::default() },
            pool.clone(),
            None,
            99,
        );
        let res = crate::runner::play_game(&dex, &mut battle, &mut [&mut p1, &mut p2], 120).unwrap();
        assert!(
            matches!(
                res,
                crate::runner::GameResult::Outcome(_) | crate::runner::GameResult::TurnCapped
            ),
            "BlindGolfAgent vs BlindAgent must complete cleanly: {res:?}"
        );

        let mut battle2 = Battle::from_fixture(&dex, "5,6,7,8", t0, t1).unwrap();
        battle2.set_log_enabled(true);
        let mut bc1 = BlindCmctsAgent::new(
            BlindCmctsConfig {
                iterations: 600,
                c: 0.4,
                max_depth: 128,
                leaf: TempoLeafMode::HybridGolf,
                rollout_turns: 0,
                max_worlds: 12,
                preview_iterations: 200,
                own_preview_triple: Some([1, 2, 3]),
                adaptive: true,
                adaptive_interval: 200,
                adaptive_min_gap: 0.01,
                stage2_root: false,
                ..BlindCmctsConfig::default()
            },
            pool.clone(),
            42,
        );
        assert!(bc1.requires_player_log());
        assert!(p1.requires_player_log());
        let mut bc2 = crate::blind::BlindAgent::new(
            RmConfig { iterations: 200, c: 0.4, rule: SelRule::Ucb, ..RmConfig::default() },
            pool.clone(),
            None,
            99,
        );
        let res2 = crate::runner::play_game(&dex, &mut battle2, &mut [&mut bc1, &mut bc2], 120).unwrap();
        assert!(
            matches!(
                res2,
                crate::runner::GameResult::Outcome(_) | crate::runner::GameResult::TurnCapped
            ),
            "BlindCmctsAgent vs BlindAgent must complete cleanly: {res2:?}"
        );

        // Stage 2A smoke & determinism check (same seed -> identical pick; full game completes).
        let mut battle3 = Battle::from_fixture(&dex, "5,6,7,8", t0, t1).unwrap();
        battle3.set_log_enabled(true);
        let s2_cfg = BlindCmctsConfig {
            iterations: 800,
            c: 0.4,
            max_depth: 128,
            leaf: TempoLeafMode::HybridGolf,
            rollout_turns: 0,
            max_worlds: 12,
            preview_iterations: 200,
            own_preview_triple: Some([1, 2, 3]),
            adaptive: true,
            adaptive_interval: 256,
            adaptive_min_gap: 0.01,
            stage2_root: true,
            ..BlindCmctsConfig::default()
        };
        let mut s2_a = BlindCmctsAgent::new(s2_cfg.clone(), pool.clone(), 777);
        let mut s2_b = BlindCmctsAgent::new(s2_cfg.clone(), pool.clone(), 777);
        battle3.choose(&dex, 0, "team 1,2,3").unwrap();
        battle3.choose(&dex, 1, "team 1,2,3").unwrap();
        let cs0 = battle3.legal_choices(&dex, 0);
        let pick_a = s2_a.choose(&battle3, &dex, 0, &cs0);
        let pick_b = s2_b.choose(&battle3, &dex, 0, &cs0);
        assert_eq!(pick_a, pick_b, "Stage 2A BlindCmctsAgent must be deterministic given seed");

        let mut battle4 = Battle::from_fixture(&dex, "9,10,11,12", t0, t1).unwrap();
        battle4.set_log_enabled(true);
        let mut s2_play = BlindCmctsAgent::new(s2_cfg, pool.clone(), 123);
        let mut opp_play = crate::blind::BlindAgent::new(
            RmConfig { iterations: 200, c: 0.4, rule: SelRule::Ucb, ..RmConfig::default() },
            pool.clone(),
            None,
            456,
        );
        let res3 =
            crate::runner::play_game(&dex, &mut battle4, &mut [&mut s2_play, &mut opp_play], 120)
                .unwrap();
        assert!(
            matches!(
                res3,
                crate::runner::GameResult::Outcome(_) | crate::runner::GameResult::TurnCapped
            ),
            "Stage 2A BlindCmctsAgent vs BlindAgent must complete cleanly: {res3:?}"
        );
    }

    #[test]
    fn stage1_parity_antisymmetry_and_cache_edge_cases() {
        let dex = test_dex();

        // 1. Exact side-swap antisymmetry on dual-boosted + dual-Phazer state for both FastGolf and HybridGolf.
        let p1 = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
        ];
        let p2 = vec![
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Roar", "Rest"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
            mk("Scizor", 50, "Leftovers", &["Swords Dance", "Baton Pass", "Agility", "Steel Wing"]),
        ];
        let mut b_ab = start(&dex, &p1, &p2);
        let mut b_ba = start(&dex, &p2, &p1);
        for b in [&mut b_ab, &mut b_ba] {
            if let Some(a0) = b.active_id(0) {
                b.poke_mut(a0).boosts[0] = 2;
                b.poke_mut(a0).boosts[1] = 2;
            }
            if let Some(a1) = b.active_id(1) {
                b.poke_mut(a1).boosts[2] = 1;
                b.poke_mut(a1).boosts[3] = 2;
            }
        }
        // Swap boosts symmetrically on `b_ba`.
        if let Some(a0) = b_ba.active_id(0) {
            b_ba.poke_mut(a0).boosts = [0, 0, 1, 2, 0, 0, 0];
        }
        if let Some(a1) = b_ba.active_id(1) {
            b_ba.poke_mut(a1).boosts = [2, 2, 0, 0, 0, 0, 0];
        }
        let (t_ab, cb_ab) = CompactBattle::from_battle(&b_ab, &dex);
        let (t_ba, cb_ba) = CompactBattle::from_battle(&b_ba, &dex);
        let c_ab = TempoMatchCache::build(&cb_ab, &t_ab);
        let c_ba = TempoMatchCache::build(&cb_ba, &t_ba);
        let fg_ab = evaluate_fast_golf(&cb_ab, &t_ab, Some(&c_ab)).win_prob;
        let fg_ba = evaluate_fast_golf(&cb_ba, &t_ba, Some(&c_ba)).win_prob;
        assert!(
            (fg_ab + fg_ba - 1.0).abs() < 1e-11,
            "FastGolf dual-boosted dual-phazer antisymmetry failed: ab={fg_ab} ba={fg_ba}"
        );
        let hy_ab = evaluate_hybrid_golf(&cb_ab, &t_ab, Some(&c_ab));
        let hy_ba = evaluate_hybrid_golf(&cb_ba, &t_ba, Some(&c_ba));
        assert!(
            (hy_ab + hy_ba - 1.0).abs() < 1e-11,
            "HybridGolf antisymmetry failed: ab={hy_ab} ba={hy_ba}"
        );

        // 2. TempoMatchCache Drain-vs-Substitute invalidation: Giga Drain deals 0 vs active Sub!
        let p_drain = vec![mk("Exeggutor", 50, "Leftovers", &["Giga Drain"])];
        let p_sub = vec![mk("Vaporeon", 50, "Leftovers", &["Surf", "Substitute"])];
        let b_ds = start(&dex, &p_drain, &p_sub);
        let (t_ds, mut cb_ds) = CompactBattle::from_battle(&b_ds, &dex);
        let c_ds = TempoMatchCache::build(&cb_ds, &t_ds);
        cb_ds.mons[3].hp_pool[1] = 50; // active Substitute
        cb_ds.mons[3].vol_flags |= vol_bit::SUBSTITUTE;
        let fg_cached = evaluate_fast_golf(&cb_ds, &t_ds, Some(&c_ds));
        let fg_uncached = evaluate_fast_golf(&cb_ds, &t_ds, None);
        assert!(
            (fg_cached.win_prob - fg_uncached.win_prob).abs() < 1e-9,
            "Drain vs Sub cache must match uncached exact kernel: cached={} uncached={}",
            fg_cached.win_prob,
            fg_uncached.win_prob
        );

        // 3. `compact_world_sig` must separate worlds differing only in held item.
        let p_left = vec![mk("Snorlax", 55, "Leftovers", &["Body Slam", "Rest"])];
        let p_mint = vec![mk("Snorlax", 55, "Mint Berry", &["Body Slam", "Rest"])];
        let (tl, cbl) = CompactBattle::from_battle(&start(&dex, &p_left, &p_left), &dex);
        let (tm, cbm) = CompactBattle::from_battle(&start(&dex, &p_left, &p_mint), &dex);
        assert_ne!(
            compact_world_sig(&cbl, &tl),
            compact_world_sig(&cbm, &tm),
            "compact_world_sig must separate worlds differing only in item"
        );

        // 4. `compact_duct_key_seeded_qc` must distinguish Quick Claw roll when QC holder is alive
        //    AND never mask `two_turn_locked` with `mon.item` bits!
        let p_qc = vec![mk("Snorlax", 55, "Quick Claw", &["Body Slam", "Rest"])];
        let (tqc, mut cbqc0) = CompactBattle::from_battle(&start(&dex, &p_qc, &p_left), &dex);
        let has_qc = compact_world_has_qc(&cbqc0, &tqc);
        assert!(has_qc);
        cbqc0.quick_claw_roll = 0;
        let mut cbqc1 = cbqc0;
        cbqc1.quick_claw_roll = 1;
        let k0 = compact_duct_key_seeded_qc(
            &cbqc0,
            &tqc,
            42,
            compact_observable_qc_bits(&cbqc0, &tqc, has_qc),
        );
        let k1 = compact_duct_key_seeded_qc(
            &cbqc1,
            &tqc,
            42,
            compact_observable_qc_bits(&cbqc1, &tqc, has_qc),
        );
        assert_ne!(k0, k1, "Internal key must separate observable Quick Claw roll");

        let mut cb_tt0 = cbqc0;
        cb_tt0.mons[0].item = 0xFF; // all item bits set
        cb_tt0.mons[0].set_two_turn_locked(false);
        let mut cb_tt1 = cb_tt0;
        cb_tt1.mons[0].set_two_turn_locked(true);
        assert_ne!(
            compact_duct_key_seeded_qc(&cb_tt0, &tqc, 42, 0),
            compact_duct_key_seeded_qc(&cb_tt1, &tqc, 42, 0),
            "mon.item bits must never mask two_turn_locked"
        );

        // 5. Post-Screen-1 key & HybridGolf parity regressions:
        //    (a) Floor HP bucket separates 95% HP (Rest succeeds) from 100% HP (Rest fails),
        //    (b) Active & bench exact PP are hashed (pp=1 vs pp=16, bench pp=0 vs pp=16),
        //    (c) `last_move_slot` (Encore) is hashed,
        //    (d) `evaluate_hybrid_golf` matches `compact_eval_leaf_blindspot` 0-ULP on 1v1,
        //    (e) `RmConfig::default().rollout_rules()` enables `perish_combo` & `relay`.
        let p_key = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Rest", "Curse"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let (tk, mut cb_full) = CompactBattle::from_battle(&start(&dex, &p_key, &p_key), &dex);
        let mut cb_95 = cb_full;
        cb_95.mons[0].hp_pool[0] = (tk.mons[0].maxhp * 95) / 100;
        assert_ne!(
            cb_full.search_key(&tk, 16, true),
            cb_95.search_key(&tk, 16, true),
            "search_key must separate 100% HP vs 95% HP (Rest fail boundary)"
        );
        assert_ne!(
            compact_duct_key_seeded_qc(&cb_full, &tk, 42, 0),
            compact_duct_key_seeded_qc(&cb_95, &tk, 42, 0),
            "compact_duct_key_seeded_qc must separate 100% HP vs 95% HP"
        );

        let mut cb_pp_act = cb_full;
        cb_pp_act.mons[0].pp[2] = 1; // Rest PP 1 vs 16
        assert_ne!(cb_full.search_key(&tk, 16, true), cb_pp_act.search_key(&tk, 16, true));
        assert_ne!(
            compact_duct_key_seeded_qc(&cb_full, &tk, 42, 0),
            compact_duct_key_seeded_qc(&cb_pp_act, &tk, 42, 0),
        );

        let mut cb_pp_bench = cb_full;
        cb_pp_bench.mons[1].pp[2] = 0; // Bench Skarmory Rest PP 0 vs 16
        assert_ne!(cb_full.search_key(&tk, 16, true), cb_pp_bench.search_key(&tk, 16, true));
        assert_ne!(
            compact_duct_key_seeded_qc(&cb_full, &tk, 42, 0),
            compact_duct_key_seeded_qc(&cb_pp_bench, &tk, 42, 0),
        );

        let mut cb_lm = cb_full;
        cb_full.mons[0].set_last_move_slot(0);
        cb_lm.mons[0].set_last_move_slot(2);
        assert_ne!(cb_full.search_key(&tk, 16, true), cb_lm.search_key(&tk, 16, true));
        assert_ne!(
            compact_duct_key_seeded_qc(&cb_full, &tk, 42, 0),
            compact_duct_key_seeded_qc(&cb_lm, &tk, 42, 0),
        );

        let (t1, cb1) = CompactBattle::from_battle(&start(&dex, &p_left, &p_mint), &dex);
        let hy_1v1 = evaluate_hybrid_golf(&cb1, &t1, None);
        let bs_1v1 = compact_eval_leaf_blindspot(&cb1, &t1, &EvalWeights::default());
        assert_eq!(hy_1v1, bs_1v1, "HybridGolf 1v1 must match compact_eval_leaf_blindspot 0-ULP");

        let rr = RmConfig::default().rollout_rules();
        assert!(rr.perish_combo && rr.baton_pass && rr.settlement.uses_relay_bellman());

        // Stage 2A + 8r heavy rollout determinism check.
        let root = conformance::fixture::repo_root();
        let pool = Arc::new(crate::preview::load_meta_pool(
            &root.join("data/belief-pool-v3/belief-pool.json"),
        ));
        let mut b8 = Battle::from_fixture(&dex, "13,14,15,16", &pool.teams[0].sets, &pool.teams[1].sets).unwrap();
        b8.set_log_enabled(true);
        b8.choose(&dex, 0, "team 1,2,3").unwrap();
        b8.choose(&dex, 1, "team 1,2,3").unwrap();
        let cs8 = b8.legal_choices(&dex, 0);
        let cfg8 = BlindCmctsConfig {
            iterations: 600,
            c: 0.4,
            max_depth: 250,
            leaf: TempoLeafMode::EvalBlindspot,
            rollout_turns: 8,
            max_worlds: 12,
            preview_iterations: 200,
            own_preview_triple: Some([1, 2, 3]),
            adaptive: true,
            adaptive_interval: 256,
            adaptive_min_gap: 0.01,
            stage2_root: true,
            ..BlindCmctsConfig::default()
        };
        let mut a8 = BlindCmctsAgent::new(cfg8.clone(), pool.clone(), 9001);
        let mut b8_ag = BlindCmctsAgent::new(cfg8, pool, 9001);
        assert_eq!(a8.choose(&b8, &dex, 0, &cs8), b8_ag.choose(&b8, &dex, 0, &cs8));

        // 6. Stage-3 feature extraction: exact 0-ULP side-swap antisymmetry & Frontier-#1 PP monotonicity!
        let f_ab = extract_stage3_features(&cb_ab, &t_ab, Some(&c_ab));
        let f_ba = extract_stage3_features(&cb_ba, &t_ba, Some(&c_ba));
        assert_eq!(f_ab.g_stag, f_ba.g_stag, "g_stag must be exactly symmetric");
        for k in 0..20usize {
            assert!(
                (f_ab.phi[k] + f_ba.phi[k]).abs() <= 1e-6,
                "Stage3 phi[{k}] antisymmetry failed: ab={} ba={}",
                f_ab.phi[k],
                f_ba.phi[k]
            );
        }
        let f_unc = extract_stage3_features(&cb_ab, &t_ab, None);
        // Cached analytical boost scaling matches uncached integer DmgKernel1D closely on boosted states,
        // and matches exact 0-drift on non-TTK mechanical features (`phi[8..18]`).
        for k in 0..20usize {
            let tol = if (8..=17).contains(&k) { 1e-5 } else { 0.05 };
            assert!(
                (f_ab.phi[k] - f_unc.phi[k]).abs() <= tol,
                "Stage3 cached vs uncached phi[{k}] drift: cached={} uncached={}",
                f_ab.phi[k],
                f_unc.phi[k]
            );
        }
        // Frontier #1 stagnation-gated recovery PP monotonicity on 3v3 stall wall mirror:
        let wall_team = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Rest", "Curse", "Sleep Talk"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mk("Blissey", 50, "Leftovers", &["Seismic Toss", "Soft-Boiled", "Toxic", "Heal Bell"]),
        ];
        let (tw, cb_w_hi) = CompactBattle::from_battle(&start(&dex, &wall_team, &wall_team), &dex);
        let cw_hi = TempoMatchCache::build(&cb_w_hi, &tw);
        let mut cb_w_lo = cb_w_hi;
        cb_w_lo.mons[0].pp[1] = 2; // Snorlax Rest PP depleted (16 -> 2)
        cb_w_lo.mons[2].pp[1] = 4; // Blissey Soft-Boiled PP depleted (16 -> 4)
        let f_hi = extract_stage3_features(&cb_w_hi, &tw, None);
        let f_hi_c = extract_stage3_features(&cb_w_hi, &tw, Some(&cw_hi));
        for k in 0..20usize {
            assert!(
                (f_hi.phi[k] - f_hi_c.phi[k]).abs() <= 1e-5,
                "Stage3 unboosted cached vs uncached phi[{k}] drift: {} vs {}",
                f_hi.phi[k],
                f_hi_c.phi[k]
            );
        }
        let f_lo = extract_stage3_features(&cb_w_lo, &tw, None);
        assert!(f_hi.g_stag > 0.3, "3v3 wall mirror must activate stagnation gate: g_stag={}", f_hi.g_stag);
        assert!(
            f_hi.phi[8] > f_lo.phi[8] + 0.05,
            "Frontier #1 stag_rec_pp must be strictly monotone in recovery PP: hi={} lo={}",
            f_hi.phi[8],
            f_lo.phi[8]
        );
        // 7. Stage-3 heads (`Stage3Log` & `Stage3Mlp`): 0-ULP antisymmetry, 1v1 exact `bs_nr`,
        //    Frontier-#1 PP monotonicity, and `contested_trees = 2` determinism!
        let s3l_ab = evaluate_stage3_log(&cb_ab, &t_ab, Some(&c_ab), 1.0);
        let s3l_ba = evaluate_stage3_log(&cb_ba, &t_ba, Some(&c_ba), 1.0);
        assert!((s3l_ab + s3l_ba - 1.0).abs() < 1e-6, "Stage3Log antisymmetry: {s3l_ab} + {s3l_ba}");
        let s3m_ab = evaluate_stage3_mlp(&cb_ab, &t_ab, Some(&c_ab), 1.0);
        let s3m_ba = evaluate_stage3_mlp(&cb_ba, &t_ba, Some(&c_ba), 1.0);
        assert!((s3m_ab + s3m_ba - 1.0).abs() < 1e-6, "Stage3Mlp antisymmetry: {s3m_ab} + {s3m_ba}");
        assert_eq!(evaluate_stage3_log(&cb1, &t1, None, 1.0), bs_1v1, "Stage3Log 1v1 must match bs 0-ULP");
        assert_eq!(evaluate_stage3_mlp(&cb1, &t1, None, 1.0), bs_1v1, "Stage3Mlp 1v1 must match bs 0-ULP");
        assert!(
            evaluate_stage3_mlp(&cb_w_hi, &tw, None, 1.0)
                > evaluate_stage3_mlp(&cb_w_lo, &tw, None, 1.0) + 0.01,
            "Stage3Mlp must strictly reward higher recovery PP in 3v3 wall stall"
        );
        let cfg_k2 = BlindCmctsConfig {
            iterations: 400,
            c: 0.4,
            max_depth: 250,
            leaf: TempoLeafMode::Stage3Mlp,
            rollout_turns: 8,
            max_worlds: 12,
            preview_iterations: 200,
            own_preview_triple: Some([1, 2, 3]),
            adaptive: false,
            adaptive_interval: 200,
            adaptive_min_gap: 0.01,
            stage2_root: false,
            sempool: true,
            mature_window: true,
            contested_trees: 2,
            leaf_temp: 1.0,
            ..BlindCmctsConfig::default()
        };
        let pool2 = Arc::new(crate::preview::load_meta_pool(
            &root.join("data/belief-pool-v3/belief-pool.json"),
        ));
        let mut ak1 = BlindCmctsAgent::new(cfg_k2.clone(), pool2.clone(), 4242);
        let mut ak2 = BlindCmctsAgent::new(cfg_k2, pool2, 4242);
        assert_eq!(ak1.choose(&b8, &dex, 0, &cs8), ak2.choose(&b8, &dex, 0, &cs8));
    }
}
