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
    get_stat_branchless, role_bit, vol_bit, CompactBattle, CompactMoveSpec, CompactWeather,
    ItemStatKind, MatchTable, RosterCompactCache, SpecialMoveKind,
};
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::state::{stat_idx, Battle, Status};

use crate::agent::Agent;
use crate::eval::{
    compact_eval01, compact_eval_leaf_blindspot, compact_turn_limit_outcome, EvalWeights,
};
use crate::exact::solve_matrix_full;
use crate::linear::DmgKernel1D;
use crate::mask::dominated_reason;
use crate::rng::SplitMix64;
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
fn effective_speed(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> i32 {
    let idx = side * 3 + slot;
    let mon = &cb.mons[idx];
    let spec = &table.mons[idx];
    let has_par = mon.status_raw() == Status::Par as u8;
    get_stat_branchless(
        spec.stored_stats[stat_idx::SPE] as i32,
        mon.boosts[stat_idx::SPE],
        stat_idx::SPE,
        false,
        false,
        true,
        has_par,
        false,
        false,
        false,
        ItemStatKind::None,
    )
}

#[inline]
fn is_grounded(table: &MatchTable, idx: usize) -> bool {
    (table.mons[idx].type_mask & (1u32 << table.flying_type)) == 0
}

#[inline]
fn spikes_layers(cb: &CompactBattle, side: usize) -> u8 {
    cb.sides[side].side_flags & 0x03
}

/// Effective starting HP for a mon if it fights now (deducting Spikes if benched & grounded).
#[inline]
fn entry_hp(cb: &CompactBattle, table: &MatchTable, side: usize, slot: usize) -> i16 {
    let idx = side * 3 + slot;
    let mut hp = cb.mons[idx].hp();
    if hp <= 0 {
        return 0;
    }
    let is_active = cb.sides[side].active_idx as usize == slot;
    if !is_active && spikes_layers(cb, side) > 0 && is_grounded(table, idx) {
        let spike_dmg = (table.mons[idx].maxhp / 8).max(1);
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
fn expected_sleep_turns(cb: &CompactBattle, side: usize, slot: usize) -> f64 {
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
    if def_status == Status::Brn as u8 || def_status == Status::Psn as u8 {
        base_residual -= def_spec.quantum_1_8 as i32;
    }
    if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        base_residual -= def_spec.quantum_1_8 as i32;
    }
    if cb.weather_turns > 0
        && cb.weather_kind == CompactWeather::Sandstorm
        && (def_spec.type_mask & table.sandstorm_immune_mask) == 0
    {
        base_residual -= def_spec.quantum_1_8 as i32;
    }
    let is_toxic = def_status == Status::Tox as u8;
    let tox_base = if is_toxic {
        def_mon.status_counter().max(1) as i32
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

        // Residual for this turn (Toxic escalates each turn).
        let res_hp = if is_toxic {
            let step = (tox_base + turn as i32 - 1).min(15);
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
    if def_status == Status::Brn as u8 || def_status == Status::Psn as u8 {
        residual_frac -= (def_spec.quantum_1_8 as f64) / maxhp_f;
    } else if def_status == Status::Tox as u8 {
        let step = (def_mon.status_counter().max(1) as f64 + 1.0).min(15.0);
        residual_frac -= step * (def_spec.quantum_1_16 as f64) / maxhp_f;
    }
    if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        residual_frac -= (def_spec.quantum_1_8 as f64) / maxhp_f;
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
    if cb.outcome != 0 {
        let (g0, g1, wp) = match cb.outcome {
            1 => (0.0, 6.0, 1.0),
            2 => (6.0, 0.0, 0.0),
            _ => (0.0, 0.0, 0.5),
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
    side: usize,
    choice: SearchChoice,
) -> Option<u8> {
    let act_slot = cb.sides[side].active_idx as usize;
    let num_moves = table.mons[side * 3 + act_slot].num_moves as usize;
    match choice {
        SearchChoice::Move(id) => (0..num_moves)
            .find(|&m_i| table.mons[side * 3 + act_slot].moves[m_i] == id)
            .map(|m_i| m_i as u8),
        SearchChoice::Switch(pos) if (1..=3).contains(&pos) => Some(pos + 3),
        _ => None,
    }
}

/// Resolve any post-turn forced switch (`request == 2`) in `cb` cleanly using `evaluate_compact_tempo_variant`.
fn resolve_forced_switches_variant(cb: &mut CompactBattle, table: &MatchTable, variant: TempoVariant) {
    for _ in 0..2 {
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
        let opts0 = if r0 { bench_opts(0) } else { vec![0] };
        let opts1 = if r1 { bench_opts(1) } else { vec![0] };

        let (pick0, pick1) = if r0 && !r1 {
            let best = opts0
                .iter()
                .copied()
                .max_by(|&a, &b| {
                    let mut ca = *cb;
                    ca.step_switch(table, [a, 0]);
                    let mut cb_b = *cb;
                    cb_b.step_switch(table, [b, 0]);
                    evaluate_compact_tempo_variant(&ca, table, variant)
                        .win_prob
                        .total_cmp(&evaluate_compact_tempo_variant(&cb_b, table, variant).win_prob)
                })
                .unwrap();
            (best, 0)
        } else if !r0 && r1 {
            let best = opts1
                .iter()
                .copied()
                .min_by(|&a, &b| {
                    let mut ca = *cb;
                    ca.step_switch(table, [0, a]);
                    let mut cb_b = *cb;
                    cb_b.step_switch(table, [0, b]);
                    evaluate_compact_tempo_variant(&ca, table, variant)
                        .win_prob
                        .total_cmp(&evaluate_compact_tempo_variant(&cb_b, table, variant).win_prob)
                })
                .unwrap();
            (0, best)
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
        let Some(a_own) = encode_choice(&table, &base_cb, side, own_c) else {
            continue;
        };
        for (j, &opp_c) in opp_choices.iter().enumerate() {
            let Some(a_opp) = encode_choice(&table, &base_cb, opp, opp_c) else {
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
                        let Some(act) = encode_choice(&table, &base_cb, side, c) else {
                            return -1.0;
                        };
                        let mut cb = base_cb;
                        let sw = if side == 0 { [act, 0] } else { [0, act] };
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
    acts: [[u8; 7]; 2],
    visits: [[u32; 7]; 2],
    sums: [[f64; 7]; 2],
    total: u32,
}

impl DuctNode {
    fn new(cb: &CompactBattle, table: &MatchTable) -> Self {
        let mut acts = [[0u8; 7]; 2];
        let n0 = cb.legal_actions(table, 0, &mut acts[0]) as u8;
        let n1 = cb.legal_actions(table, 1, &mut acts[1]) as u8;
        Self {
            n_acts: [n0, n1],
            acts,
            visits: [[0; 7]; 2],
            sums: [[0.0; 7]; 2],
            total: 0,
        }
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
        const TAU: f64 = 1.5;
        let gain0 = if n0_len > 1 && n0 >= 2 {
            let mut mu0 = r_bar0;
            for i in 0..n0_len {
                let na = self.visits[0][i];
                if na > 0 {
                    let est = (self.sums[0][i] + TAU * r_bar0) / (f64::from(na) + TAU);
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
                    let est = (wb1 + TAU * r_bar1) / (f64::from(nb) + TAU);
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
        let lambda = 0.85 * (f64::from(n) / (f64::from(n) + 6.0));
        ((1.0 - lambda) * raw_val0 + lambda * v_star).clamp(0.000_002, 0.999_998)
    }
}

#[inline]
fn mix64(mut h: u64, v: u64) -> u64 {
    h ^= v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = h.rotate_left(27).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^ (h >> 31)
}

/// Compact 16-HP-bucket state hash for zero-alloc DUCT tree indexing.
#[inline]
fn compact_duct_key(cb: &CompactBattle, table: &MatchTable, depth: u8) -> u64 {
    let mut h = mix64(0x517C_C1B7_2722_0A95, depth as u64);
    let s0 = &cb.sides[0];
    let s1 = &cb.sides[1];
    let side_word = ((s0.active_idx as u64) << 0)
        | ((s0.alive_mask as u64) << 3)
        | ((s0.request as u64) << 6)
        | (((s0.side_flags & 0x07) as u64) << 9)
        | (((s0.reflect_turns > 0) as u64) << 12)
        | (((s0.lightscreen_turns > 0) as u64) << 13)
        | (((s0.safeguard_turns > 0) as u64) << 14)
        | ((s1.active_idx as u64) << 16)
        | ((s1.alive_mask as u64) << 19)
        | ((s1.request as u64) << 22)
        | (((s1.side_flags & 0x07) as u64) << 25)
        | (((s1.reflect_turns > 0) as u64) << 28)
        | (((s1.lightscreen_turns > 0) as u64) << 29)
        | (((s1.safeguard_turns > 0) as u64) << 30)
        | (if cb.weather_turns > 0 {
            ((cb.weather_kind as u64) << 32)
        } else {
            0
        });
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
            let hp_b = if hp == 0 {
                0u64
            } else {
                (((hp * 16 + maxhp - 1) / maxhp).clamp(1, 16)) as u64
            };
            let st = mon.status_raw() as u64;
            let st_c = if st == Status::Slp as u64 {
                mon.status_counter().min(4) as u64
            } else {
                0
            };
            let mut mword = ((idx as u64) << 0) | (hp_b << 4) | (st << 9) | (st_c << 12);
            if slot == act {
                let sub = (mon.sub_hp() > 0) as u64;
                let perish = mon.perish_turns() as u64;
                let vol = (mon.vol_flags
                    & (vol_bit::SUBSTITUTE
                        | vol_bit::CONFUSION
                        | vol_bit::ENCORE
                        | vol_bit::DISABLE
                        | vol_bit::TRAPPED_OR_LOCKED
                        | vol_bit::LEECH_SEED)) as u64;
                let rech = mon.must_recharge() as u64;
                mword |= (sub << 16) | (perish << 17) | (rech << 20) | (vol << 24);
                h = mix64(h, mword);
                let mut bword = 0u64;
                for (bi, &bv) in mon.boosts.iter().enumerate() {
                    bword |= (((bv + 6).clamp(0, 12) as u64) & 0xF) << (bi * 4);
                }
                h = mix64(h, bword);
            } else {
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
    match mode {
        TempoLeafMode::TempoFull => {
            evaluate_compact_tempo_variant(cb, table, TempoVariant::Full).win_prob
        }
        TempoLeafMode::TempoRaw => {
            evaluate_compact_tempo_variant(cb, table, TempoVariant::Raw).win_prob
        }
        TempoLeafMode::FastGolf => evaluate_fast_golf(cb, table, cache).win_prob,
        TempoLeafMode::EvalStatic => compact_eval01(cb, table, &EvalWeights::default()),
        TempoLeafMode::EvalBlindspot => {
            compact_eval_leaf_blindspot(cb, table, &EvalWeights::default())
        }
    }
}

/// Fast `CompactBattle`-native ε-greedy (`eps=0.20`) heavy rollout action picker for controlled
/// rollout-vs-zero-rollout ablations (`rollout_turns = 0, 2, 4, 8`).
#[inline]
fn compact_rollout_pick_fast(
    cb: &CompactBattle,
    table: &MatchTable,
    side: usize,
    cs: &[u8],
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

    // Perish Song last-turn escape (matches mcts.rs heavy playout).
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
                let inc = if foe_alive {
                    cb.best_hit_fraction(table, def_side, def_slot, side, sl)
                } else {
                    0.0
                };
                let sc = 1.0 - inc;
                if sc > best_score {
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

    if cb.sides[side].request == 1
        && (cb.sides[side].alive_mask & (1 << att_slot)) != 0
        && (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
    {
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

    // Forced switch: pick best immediate matchup (`own_hit - foe_hit`).
    let foe_alive = (cb.sides[def_side].alive_mask & (1 << def_slot)) != 0
        && cb.mons[def_side * 3 + def_slot].hp() > 0;
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
    let raw_cutoff = cb.turn.saturating_add(rollout_turns.saturating_sub(1));
    let cutoff = table
        .ruleset
        .turn_limit()
        .map_or(raw_cutoff, |lim| raw_cutoff.min(lim));
    let mut safety = 0u8;
    while cb.outcome == 0 && cb.turn <= cutoff && safety < 32 {
        safety += 1;
        if let Some(v) = compact_turn_limit_outcome(cb, table) {
            return v;
        }
        let mut picks = [7u8; 2];
        for s in 0..2usize {
            if cb.sides[s].request == 0 {
                continue;
            }
            let mut cs = [0u8; 7];
            let n = cb.legal_actions(table, s, &mut cs);
            if n > 0 {
                picks[s] = compact_rollout_pick_fast(cb, table, s, &cs[..n], rng);
            }
        }
        cb.prng_seed = rng.next();
        cb.apply_actions(table, picks);
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

const MOVE_FLAG_USABLE: u8 = 1 << 0;
const MOVE_FLAG_PHYSICAL: u8 = 1 << 1;
const MOVE_FLAG_FIXED: u8 = 1 << 2;
const MOVE_FLAG_WATER: u8 = 1 << 3;
const MOVE_FLAG_FIRE: u8 = 1 << 4;
const MOVE_FLAG_SOLARBEAM: u8 = 1 << 5;
const MOVE_FLAG_SELFDESTRUCT: u8 = 1 << 6;
const MOVE_FLAG_RECHARGE: u8 = 1 << 7;

#[derive(Clone, Copy, Default)]
struct CachedMoveDmg {
    mean_dmg: f32,
    min_norm: f32,
    max_norm: f32,
    max_crit: f32,
    p_hit: f32,
    p_crit: f32,
    flags: u8,
}

#[derive(Clone, Copy)]
struct StaticMonCache {
    half_heal_slot: u8,
    half_heal_is_weather: bool,
    rest_slot: u8,
    sleep_talk_slot: u8,
    phaze_slot: u8,
    haze_slot: u8,
    bp_slot: u8,
    perish_slot: u8,
    dbond_slot: u8,
    st_attack_rate: f64,
    inv_maxhp: f64,
    q16_frac: f64,
    q8_frac: f64,
    sand_immune: bool,
    init_boost_type: u8,
    init_castle_boost_type: u8,
    init_stat_kind: nc2000_engine::compact::ItemStatKind,
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
            st_attack_rate: 0.0,
            inv_maxhp: 1.0,
            q16_frac: 0.0625,
            q8_frac: 0.125,
            sand_immune: false,
            init_boost_type: 0xFF,
            init_castle_boost_type: 0xFF,
            init_stat_kind: nc2000_engine::compact::ItemStatKind::None,
        }
    }
}

#[derive(Clone, Copy)]
struct CachedPairEntry {
    moves: [CachedMoveDmg; 4],
    best_norm_slot: u8,
    best_phys_slot: u8,
    best_spec_slot: u8,
    boom_slot: u8,
    any_dynamic_bp: bool,
}

impl Default for CachedPairEntry {
    fn default() -> Self {
        Self {
            moves: [CachedMoveDmg::default(); 4],
            best_norm_slot: 0xFF,
            best_phys_slot: 0xFF,
            best_spec_slot: 0xFF,
            boom_slot: 0xFF,
            any_dynamic_bp: false,
        }
    }
}

/// Per-search precomputed static damage & trait table across all 6 mons and 18 directed pairs.
/// Built once at search root (~2 us) and reused across thousands of MCTS leaves with O(1) lookup,
/// including O(1) analytical Gen 2 boost/burn/screen/weather scaling (zero `DmgKernel1D` on leaves).
#[derive(Clone)]
pub struct TempoMatchCache {
    mons: [StaticMonCache; 6],
    entries: [[[CachedPairEntry; 3]; 3]; 2],
}

impl TempoMatchCache {
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
                    let mut best_phys_dmg = 0.0f32;
                    let mut best_spec_dmg = 0.0f32;
                    let mut best_boom_dmg = 0.0f32;
                    for m_i in 0..n_moves {
                        let ms = &table.move_specs[att_idx][m_i];
                        if ms.is_status {
                            continue;
                        }
                        if matches!(
                            ms.special_kind,
                            SpecialMoveKind::FlailReversal
                                | SpecialMoveKind::SuperFang
                                | SpecialMoveKind::PainSplit
                                | SpecialMoveKind::Counter
                                | SpecialMoveKind::MirrorCoat
                        ) {
                            entry.any_dynamic_bp = true;
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
                            if (sm.flags & MOVE_FLAG_PHYSICAL) != 0 {
                                if eff > best_phys_dmg {
                                    best_phys_dmg = eff;
                                    entry.best_phys_slot = m_i as u8;
                                }
                            } else if eff > best_spec_dmg {
                                best_spec_dmg = eff;
                                entry.best_spec_slot = m_i as u8;
                            }
                        }
                    }
                    entries[att_side][att_slot][def_slot] = entry;
                }
            }
        }
        Self { mons, entries }
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
fn scale_cached_move(
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
    if acc_b != 0 || eva_b != 0 {
        let ratio = GEN2_STAGE_MULT[(acc_b + 6).clamp(0, 12) as usize]
            / GEN2_STAGE_MULT[(eva_b + 6).clamp(0, 12) as usize];
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
fn fast_ko_prob(m: &CachedMoveDmg, hp: f32) -> f64 {
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
struct FastPair {
    raw_ttk: f64,
    eff_ttk: f64,
    net_dmg: f64,
    raw_dmg: f64,
    max_norm_frac: f64,
    ohko_prob: f64,
    boom_ko_prob: f64,
    boom_dmg_frac: f64,
    heal_duty: f64,
    def_sustain: bool,
    speed_cmp: i8,
    perish_capped: bool,
    best_pp: u8,
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
    cache: Option<&TempoMatchCache>,
) -> FastPair {
    let att_idx = att_side * 3 + att_slot;
    let def_idx = def_side * 3 + def_slot;
    let att_mon = &cb.mons[att_idx];
    let def_mon = &cb.mons[def_idx];
    let att_spec = &table.mons[att_idx];
    let def_spec = &table.mons[def_idx];

    let s_att = if ignore_att_boosts && att_mon.boosts[4] != 0 {
        let base = (att_spec.stored_stats[stat_idx::SPE] as i32).max(1);
        if att_mon.status_raw() == Status::Par as u8 { (base / 4).max(1) } else { base }
    } else {
        speeds[att_idx]
    };
    let s_def = if ignore_def_boosts && def_mon.boosts[4] != 0 {
        let base = (def_spec.stored_stats[stat_idx::SPE] as i32).max(1);
        if def_mon.status_raw() == Status::Par as u8 { (base / 4).max(1) } else { base }
    } else {
        speeds[def_idx]
    };
    let speed_cmp = if s_att > s_def { 1 } else if s_att < s_def { -1 } else { 0 };

    let start_hp = entry_hps[def_idx];
    if start_hp <= 0 || att_mon.hp() <= 0 {
        return FastPair {
            raw_ttk: if start_hp <= 0 { 0.0 } else { FAST_WALL_CAP },
            eff_ttk: if start_hp <= 0 { 0.0 } else { FAST_WALL_CAP },
            speed_cmp,
            ..FastPair::default()
        };
    }

    let (inv_maxhp, q16_frac, q8_frac, sand_immune) = if let Some(c) = cache {
        let m = &c.mons[def_idx];
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
    if def_status == Status::Brn as u8 || def_status == Status::Psn as u8 {
        res_frac -= q8_frac;
    } else if def_status == Status::Tox as u8 {
        let step = (def_mon.status_counter().max(1) as f64 + 1.5).min(15.0);
        res_frac -= step * q16_frac;
    }
    if (def_mon.vol_flags & vol_bit::LEECH_SEED) != 0 {
        res_frac -= q8_frac;
    }
    if cb.weather_turns > 0 && cb.weather_kind == CompactWeather::Sandstorm && !sand_immune {
        res_frac -= q8_frac;
    }

    let zero_boosts = [0i8; 7];
    let eff_att_boosts = if ignore_att_boosts { &zero_boosts } else { &att_mon.boosts };
    let eff_def_boosts = if ignore_def_boosts { &zero_boosts } else { &def_mon.boosts };
    let att_status = att_mon.status_raw();
    let att_burned = att_status == Status::Brn as u8;
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
    let cached_entry = cache.and_then(|c| {
        let e = &c.entries[att_side][att_slot][def_slot];
        if e.any_dynamic_bp || (def_mon.vol_flags & vol_bit::FORESIGHT) != 0 {
            return None;
        }
        let att_is = table.item_spec(att_mon.item);
        let cm_a = &c.mons[att_idx];
        let cm_d = &c.mons[def_idx];
        let item_ok = att_is.boost_type == cm_a.init_boost_type
            && att_is.castle_boost_type == cm_a.init_castle_boost_type
            && att_is.stat_kind == cm_a.init_stat_kind
            && def_item.stat_kind == cm_d.init_stat_kind;
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

    // Fast path: O(1) precomputed best-move lookup (handles clean AND boosted/burned/screened/weathered states!).
    let mut used_fast_slot = false;
    if let Some(ce) = cached_entry {
        if !needs_scale {
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
        } else {
            // Boosted/screened/burned/weathered O(1) path: check at most `best_phys_slot` + `best_spec_slot`!
            let ps = ce.best_phys_slot as usize;
            let ss = ce.best_spec_slot as usize;
            let p_ok = ps == 0xFF || (att_mon.pp[ps] > 0 && !att_mon.is_slot_disabled(ps));
            let s_ok = ss == 0xFF || (att_mon.pp[ss] > 0 && !att_mon.is_slot_disabled(ss));
            if p_ok && s_ok {
                used_fast_slot = true;
                let mut best_score = 0.0f64;
                for &slot_m in &[ps, ss] {
                    if slot_m >= 4 {
                        continue;
                    }
                    let md = scale_cached_move(
                        ce.moves[slot_m],
                        eff_att_boosts,
                        eff_def_boosts,
                        att_burned,
                        def_reflect_turns,
                        def_lightscreen_turns,
                        cb.weather_kind,
                        weather_active,
                    );
                    let p_ko = fast_ko_prob(&md, ko_hp_f);
                    let mut d_mean = (md.mean_dmg as f64) * inv_maxhp;
                    if (md.flags & MOVE_FLAG_RECHARGE) != 0 && p_ko < 0.85 {
                        d_mean *= 0.55;
                    }
                    let sc = d_mean + 0.15 * p_ko;
                    if sc > best_score {
                        best_score = sc;
                        best_mean = d_mean;
                        best_max_norm = (md.max_norm as f64) * inv_maxhp;
                        best_ohko = p_ko;
                        best_pp = att_mon.pp[slot_m];
                    }
                }
                let bms = ce.boom_slot as usize;
                if bms < 4 && att_mon.pp[bms] > 0 && !att_mon.is_slot_disabled(bms) {
                    let md = scale_cached_move(
                        ce.moves[bms],
                        eff_att_boosts,
                        eff_def_boosts,
                        att_burned,
                        def_reflect_turns,
                        def_lightscreen_turns,
                        cb.weather_kind,
                        weather_active,
                    );
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
        (FAST_WALL_CAP - 1.5 * (1.0 - hp_frac)).max(1.0)
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
        (t_hit + delay + sub_turns).min(FAST_WALL_CAP)
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

    FastPair {
        raw_ttk,
        eff_ttk: raw_ttk,
        net_dmg: d_net,
        raw_dmg: best_mean,
        max_norm_frac: best_max_norm,
        ohko_prob,
        boom_ko_prob: boom_ko,
        boom_dmg_frac: boom_dmg,
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
    cache: Option<&TempoMatchCache>,
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
            let m = &c.mons[def_idx];
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
    cache: Option<&TempoMatchCache>,
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
            let ms = match kind {
                SpecialMoveKind::PerishSong => c.mons[idx].perish_slot as usize,
                SpecialMoveKind::DestinyBond => c.mons[idx].dbond_slot as usize,
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
            |c| c.mons[idx].inv_maxhp,
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
    if cb.outcome != 0 {
        let (g0, g1, wp) = match cb.outcome {
            1 => (0.0, 6.0, 1.0),
            2 => (6.0, 0.0, 0.0),
            _ => (0.0, 0.0, 0.5),
        };
        return FastGolfEval { g_turns: [g0, g1], turn_advantage: g1 - g0, win_prob: wp };
    }

    let n0 = cb.sides[0].alive_mask.count_ones();
    let n1 = cb.sides[1].alive_mask.count_ones();
    let is_1v1 = n0 <= 1 && n1 <= 1;

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
                    let st_s = c.mons[idx].sleep_talk_slot as usize;
                    if st_s < 4
                        && cb.mons[idx].pp[st_s] > 0
                        && !cb.mons[idx].is_slot_disabled(st_s)
                    {
                        c.mons[idx].st_attack_rate
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
                cache,
            );
            let r10 = compute_fast_pair(
                cb, table, 1, s1, 0, s0, &entry_hps, &speeds, &sleep_eff_delays, false, false,
                cache,
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
                cache,
            );
            let r_ea = compute_fast_pair(
                cb, table, opp, e, s, act, &entry_hps, &speeds, &sleep_eff_delays, false, true,
                cache,
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
    for s in 0..2usize {
        let unb_opt = if s == 0 { &unb0 } else { &unb1 };
        let Some((ref unb_att, ref unb_def)) = *unb_opt else {
            continue;
        };
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
                let bs = c.mons[act_idx].bp_slot as usize;
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
                let ps = c.mons[e_idx].phaze_slot as usize;
                let hs = c.mons[e_idx].haze_slot as usize;
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

        if blend_active > 0.0 || blend_bench > 0.0 {
            for e in 0..3usize {
                if (cb.sides[opp].alive_mask & (1 << e)) == 0 {
                    continue;
                }
                let w = if e == opp_act { blend_active } else { blend_bench };
                if w > 0.0 {
                    ttk[s][act][e] = blend_fast_pair(ttk[s][act][e], unb_att[e], w);
                    ttk[opp][e][act] = blend_fast_pair(ttk[opp][e][act], unb_def[e], w);
                }
            }
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
        // Empirically calibrated Brownian race variance per turn across 18k+ real NC2000 states:
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
    FastGolfEval { g_turns: [g0, g1], turn_advantage: adv, win_prob: wp }
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
                cb, table, 0, s0, 1, s1, &entry_hps, &speeds, &sleep_eff_delays, false, false, None,
            );
            let mut a10 = compute_fast_pair(
                cb, table, 1, s1, 0, s0, &entry_hps, &speeds, &sleep_eff_delays, false, false, None,
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
    let fast_cache = if cfg.leaf == TempoLeafMode::FastGolf {
        Some(TempoMatchCache::build(&base_cb, &table))
    } else {
        None
    };
    let mut nodes: Vec<DuctNode> = Vec::with_capacity((cfg.iterations as usize / 2).max(16));
    let mut index: FxHashMap<u64, usize> = FxHashMap::default();

    // Build root node using exact caller `own_choices` order so indices align 1:1.
    let mut root = DuctNode::new(&base_cb, &table);
    let mut own_codes = [0u8; 7];
    let mut n_own = 0usize;
    for &c in own_choices {
        if n_own < 7 {
            if let Some(code) = encode_choice(&table, &base_cb, side, c) {
                own_codes[n_own] = code;
                n_own += 1;
            }
        }
    }
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
                self.cfg.leaf == TempoLeafMode::FastGolf,
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
                self.cfg.leaf == TempoLeafMode::FastGolf,
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
        let fast_cache = if self.cfg.leaf == TempoLeafMode::FastGolf {
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
            eval_duct_leaf_with_rollout(&mut cb, &tbl, leaf_mode, turns, fast_cache.as_ref(), rng)
        });
        search.best(side).unwrap_or(choices[0])
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
}
