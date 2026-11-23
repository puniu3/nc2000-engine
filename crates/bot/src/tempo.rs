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
    ItemStatKind, MatchTable, SpecialMoveKind,
};
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::state::{stat_idx, Battle, Status};

use crate::agent::Agent;
use crate::eval::{compact_eval01, compact_eval_leaf_blindspot, EvalWeights};
use crate::exact::solve_matrix_full;
use crate::linear::DmgKernel1D;
use crate::rng::SplitMix64;

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

    // Precompute each ally's available HP budget (entry HP fraction).
    let mut base_hp_budget = [0.0f64; 3];
    for &a in &allies[..n_allies] {
        let hp = entry_hp(cb, table, side, a) as f64;
        let maxhp = table.mons[side * 3 + a].maxhp.max(1) as f64;
        base_hp_budget[a] = (hp / maxhp).clamp(0.0, 1.0);
    }

    // Active matchup speed tie-break (`-0.25` if active ally faster than active enemy, `+0.25` if slower).
    let active_speed_adj = if cb.mons[side * 3 + act_ally].hp() > 0
        && cb.mons[opp * 3 + act_enemy].hp() > 0
    {
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
            // - Active enemy handled by benched ally: +1.0 immediate switch turn (+ 1 free hit on entry).
            // - Bench enemy handled by a NEW distinct ally beyond the first: +0.5 pivot friction.
            let mut entry_hit_hp = 0.0f64;
            if e == act_enemy && a != act_ally {
                switch_turns += 1.0;
                // Incoming free hit from active enemy when switching in.
                entry_hit_hp = pair_ea.net_dmg_frac.max(0.0);
            } else if e != act_enemy && used_mask != 0 && (used_mask & (1 << a)) == 0 {
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
                sum += evaluate_tempo_variant(&sim, dex, variant).win_prob_for(side);
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
            return fast_tempo_team_preview(battle, dex, side, choices, self.variant);
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
    fn select_ucb(&self, side: usize, c: f64) -> usize {
        let n = self.n_acts[side] as usize;
        if n <= 1 {
            return 0;
        }
        for i in 0..n {
            if self.visits[side][i] == 0 {
                return i;
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
fn eval_duct_leaf(cb: &CompactBattle, table: &MatchTable, mode: TempoLeafMode) -> f64 {
    if cb.outcome != 0 {
        return match cb.outcome {
            1 => 1.0,
            2 => 0.0,
            _ => 0.5,
        };
    }
    match mode {
        TempoLeafMode::TempoFull => {
            evaluate_compact_tempo_variant(cb, table, TempoVariant::Full).win_prob
        }
        TempoLeafMode::TempoRaw => {
            evaluate_compact_tempo_variant(cb, table, TempoVariant::Raw).win_prob
        }
        TempoLeafMode::EvalStatic => compact_eval01(cb, table, &EvalWeights::default()),
        TempoLeafMode::EvalBlindspot => {
            compact_eval_leaf_blindspot(cb, table, &EvalWeights::default())
        }
    }
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

    let warmup_pairs = (root_n0 * root_n1).min(cfg.iterations as usize);

    for iter in 0..cfg.iterations as usize {
        let mut cb = base_cb;
        let mut depth = 0u8;
        let mut cur_id = 0usize;
        path.clear();

        let leaf_val = loop {
            let (i0, i1) = if depth == 0 && iter < warmup_pairs {
                (iter / root_n1.max(1), iter % root_n1.max(1))
            } else {
                let node = &nodes[cur_id];
                (node.select_ucb(0, cfg.c), node.select_ucb(1, cfg.c))
            };
            let a0 = nodes[cur_id].acts[0][i0];
            let a1 = nodes[cur_id].acts[1][i1];
            path.push((cur_id, i0, i1));

            cb.prng_seed = rng.next();
            cb.apply_actions(&table, [a0, a1]);
            depth += 1;

            if cb.outcome != 0 {
                break eval_duct_leaf(&cb, &table, cfg.leaf);
            }
            if depth >= cfg.max_depth {
                break eval_duct_leaf(&cb, &table, cfg.leaf);
            }

            let next_key = compact_duct_key(&cb, &table, depth);
            if let Some(&next_id) = index.get(&next_key) {
                cur_id = next_id;
            } else {
                let new_id = nodes.len();
                nodes.push(DuctNode::new(&cb, &table));
                index.insert(next_key, new_id);
                break eval_duct_leaf(&cb, &table, cfg.leaf);
            }
        };

        depth_sum += depth as u64;
        for &(nid, i0, i1) in &path {
            let n = &mut nodes[nid];
            n.total += 1;
            n.visits[0][i0] += 1;
            n.sums[0][i0] += leaf_val;
            n.visits[1][i1] += 1;
            n.sums[1][i1] += leaf_val;
        }
        if let Some(&(_, r0, r1)) = path.first() {
            joint_v[r0][r1] += 1;
            joint_s[r0][r1] += leaf_val;
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

    // Root readout: solve shrunken joint matrix over active support (`visits >= max_v / 8`) via exact Nash LP.
    let (own_policy, eq_win_prob) = if n_my <= 1 {
        (vec![1.0; n_my], means.first().copied().unwrap_or(0.5))
    } else if n_op <= 1 {
        let best_i = (0..n_my)
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
            return fast_tempo_team_preview(battle, dex, side, choices, preview_variant);
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
    }
}
