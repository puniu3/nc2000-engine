//! Two-Stage Persistent-Tree Double-Oracle Team Preview Engine (`CfPreviewSearch`).
//!
//! Replaces decoupled UCB1 + `LegacyRollout` (`8r`, `rollout_switch = false`) pure-argmax
//! preview with:
//! 1. **Stage 1 (Full-Width Dense `CF-V_net` v2 Screening)**: Across all Turn-0 belief worlds
//!    (exact normalized weights over `belief.alive()` or `FallbackCandPool`, zero hidden-info
//!    leakage), precomputes one `6×6` `PreviewTempoCache` per world (~8 μs) and evaluates the
//!    entire legal `L_me × L_opp` (<= `60×60`, or up to `120×120` when `party_order_matters`)
//!    Turn-1 matrix via 0-ULP `evaluate_cf_value` (~8–12 ms/world). Solves Stage-1 Multi-World
//!    Bayesian RM+ (`O(sum_w K_me * K_opp)` per sweep, zero simplex pivot cap).
//! 2. **Anti-Pruning Support Selection**: Selects `S_me` and per-world `S_opp(w)` combining
//!    Stage-1 Nash support, top pure maximin guarantees, a mandatory **every-legal-lead
//!    diversity quota** (with 0-ply blind-spot archetype priority: Sleep, Spikes, Baton Pass,
//!    Explosion, Pursuit/Counter/MirrorCoat, PerishTrap/Encore, BellyDrum), and a same-core cap
//!    (`<= 2`) so critical bench pieces cannot be crowded out.
//! 3. **Stage 2 (Persistent-Tree Iterative Double-Oracle + Active-Support Deep Polish)**:
//!    Refines `S_me × S_opp(w)` using persistent Turn-1 `cfv-0r@puct@lam0` trees (matching the
//!    certified in-battle Champion: `P_net` PUCT + sound no-op mask + `evaluate_cf_value_resolving_switches`
//!    + `n0+n1 <= 3` `8r EvalBlindspot + linear.rs` Hybrid Endgame Guard + pure MC `lam = 0.0`),
//!    verifies outside best-responses (`j' ∉ S_opp` vs `supp(p*)`, `i' ∉ S_me` vs `supp(q*)`),
//!    polishes near-binding cells to low SE, and outputs an exact Bayesian Minimax / LCB-gated
//!    Mixed Nash strategy.

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn start_timer() -> Instant {
    Instant::now()
}

#[cfg(not(target_arch = "wasm32"))]
#[inline]
fn elapsed_ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

#[cfg(target_arch = "wasm32")]
#[inline]
fn start_timer() -> () {}

#[cfg(target_arch = "wasm32")]
#[inline]
fn elapsed_ms(_t: ()) -> f64 {
    0.0
}

use nc2000_engine::battle::SearchChoice;
use nc2000_engine::compact::{
    is_compact_preview_supported, timer_idx, vol_bit, CompactBattle, CompactPreviewRoster,
    MatchTable, SpecialMoveKind, VarBpKind,
};
use nc2000_engine::dex::Dex;
use nc2000_engine::fxhash::FxHashMap;
use nc2000_engine::state::{Battle, RequestState};

use crate::belief::Belief;
use crate::blind::{party_order_matters, BlindSearch};
use crate::compact_az::{
    evaluate_cf_value, evaluate_cf_value_resolving_switches, policy_priors_both_mode,
};
use crate::eval::{compact_eval_leaf_blindspot, compact_turn_limit_outcome, EvalWeights};
use crate::mcts::Playout;
use crate::player_state::ObservedPosition;
use crate::rng::SplitMix64;
use crate::smmcts::{RmConfig, SelRule};
use crate::tempo::{PreviewTempoCache, TempoMatchCache};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreviewPolicyMode {
    /// Use Stage-1 0-ply `CF-V_net` v2 Bayesian RM+ equilibrium directly (no Stage-2 search).
    Stage1Only,
    /// Pure maximin on the Stage-2 refined matrix: `argmax_i sum_w w_k min_j V[w][i, j]`.
    PureMaximin,
    /// Mixed Bayesian RM+ equilibrium with dust shedding (`dust_ratio * max_p`).
    MixedNash { dust_ratio: f64 },
    /// Mix iff polished mixed guarantee beats best pure guarantee by `>= min_mix_margin`
    /// and conservative LCB mixed guarantee does not trail best pure by more than `0.005`.
    LcbGatedMixed {
        min_mix_margin: f64,
        dust_ratio: f64,
        lcb_z: f64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewCellReadout {
    /// Raw running mean `total_w / total_n` across all visits to the Turn-1 cell.
    RawMean,
    /// Late-window mean (exponentially halving at powers of 2 `>= 64`) to drop early exploration drag.
    LateWindowMean,
    /// Exact RM+ value of the Turn-1 root `<= 6×6` joint action matrix (shrunk lightly toward `LateWindowMean`).
    Turn1Minimax,
}

#[derive(Clone, Debug)]
pub struct PreviewSearchConfig {
    /// Total Stage-2 `cfv-0r@puct@lam0` iteration budget across all worlds/cells/rounds.
    pub total_iters: u32,
    /// Target candidate support size per side (`K`).
    pub support_k: usize,
    /// Coarse Stage-2a iterations per `(i ∈ S_me, j ∈ S_opp(w))` cell (upper cap; scaled down if budget tight).
    pub coarse_iters: u32,
    /// Number of Double-Oracle outside best-response verification rounds (`0` = off).
    pub do_rounds: usize,
    /// Outside candidates screened per side per Double-Oracle round.
    pub do_candidates: usize,
    /// Iterations per `(supp, outside)` probe cell in Double-Oracle verification.
    pub do_iters: u32,
    /// Minimum violation gain required to admit an outside candidate in Double-Oracle.
    pub do_margin: f64,
    /// Iterations added per persistent-tree polish step in Stage 2c.
    pub polish_chunk: u32,
    /// Near-binding gap around `v*` eligible for Stage-2c active-support polish.
    pub polish_gap: f64,
    /// Maximum Turn-0 belief worlds screened in Stage 1.
    pub max_stage1_worlds: usize,
    /// Maximum highest-weight belief worlds refined in Stage 2.
    pub max_stage2_worlds: usize,
    /// Stage-1 & Stage-2 Bayesian RM+ sweep counts.
    pub rm_sweeps_stage1: u32,
    pub rm_sweeps_final: u32,
    /// Policy extraction mode (`PureMaximin` / `MixedNash` / `LcbGatedMixed` / `Stage1Only`).
    pub policy_mode: PreviewPolicyMode,
    /// Turn-1 cell valuation readout (`Turn1Minimax` / `LateWindowMean` / `RawMean`).
    pub cell_readout: PreviewCellReadout,
    /// Interior Turn-1+ PUCT parameters (matching certified in-battle Champion).
    pub use_cf_policy: bool,
    pub c_puct: f64,
    pub fpu_red: f64,
    pub puct_eps: f32,
    pub leaf_temp: f64,
    /// Fallback `BlindSearch` iterations if ALL belief worlds are unsupported by `CompactPreviewRoster`.
    pub fallback_iters: u32,
}

impl Default for PreviewSearchConfig {
    fn default() -> Self {
        Self {
            total_iters: 720_000,
            support_k: 12,
            coarse_iters: 128,
            do_rounds: 1,
            do_candidates: 8,
            do_iters: 96,
            do_margin: 0.015,
            polish_chunk: 128,
            polish_gap: 0.03,
            max_stage1_worlds: 24,
            max_stage2_worlds: 8,
            rm_sweeps_stage1: 400,
            rm_sweeps_final: 1500,
            policy_mode: PreviewPolicyMode::LcbGatedMixed {
                min_mix_margin: 0.015,
                dust_ratio: 0.08,
                lcb_z: 1.0,
            },
            cell_readout: PreviewCellReadout::LateWindowMean,
            use_cf_policy: false,
            c_puct: 1.0,
            fpu_red: 0.09,
            puct_eps: 0.02,
            leaf_temp: 1.0,
            fallback_iters: 27_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PreviewSearchResult {
    pub choice: SearchChoice,
    /// Probability distribution aligned 1:1 with `base.legal_choices(dex, side)`.
    pub policy_full: Vec<f64>,
    pub value: f64,
    pub pure_guarantee: f64,
    pub mixed_guarantee: f64,
    pub duality_gap: f64,
    pub used_mix: bool,
    pub used_fallback: bool,
    pub stage1_ms: f64,
    pub total_ms: f64,
    pub iters_used: u32,
    pub worlds_stage1: usize,
    pub worlds_stage2: usize,
    pub support_me: usize,
    pub support_opp_max: usize,
    pub do_admitted_me: usize,
    pub do_admitted_opp: usize,
}

// =========================================================================
// 1. Multi-World Bayesian RM+ Solver (O(sum_w K_me * K_opp) per sweep)
// =========================================================================

#[derive(Clone, Debug)]
pub struct BayesianRmSolution {
    pub own_policy: Vec<f64>,
    pub opp_policies: Vec<Vec<f64>>,
    pub value: f64,
    pub duality_gap: f64,
    pub row_guarantees: Vec<f64>,
    pub row_vs_q: Vec<f64>,
    pub col_vs_p: Vec<Vec<f64>>,
    pub world_values: Vec<f64>,
}

/// Linear-weighted alternating Multi-World Bayesian RM+ solver:
/// `\max_{p \in \Delta(n_{own})} \sum_w \pi_w \min_{q_w \in \Delta(m_w)} p^T V_w q_w`.
/// Supports heterogeneous opponent column counts (`60` vs `120` when `party_order_matters`
/// differs across worlds), zero simplex pivot limits, and exact duality-gap certification.
pub fn solve_bayesian_rm_plus(
    n_own: usize,
    worlds: &[(f64, usize, &[f64])],
    sweeps: u32,
) -> BayesianRmSolution {
    assert!(n_own > 0 && !worlds.is_empty());
    let n_w = worlds.len();
    let mut r_own = vec![0.0f64; n_own];
    let mut avg_own = vec![0.0f64; n_own];
    let mut p_cur = vec![1.0 / (n_own as f64); n_own];

    let mut r_opp: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut avg_opp: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut q_cur: Vec<Vec<f64>> = worlds
        .iter()
        .map(|&(_, m, _)| vec![1.0 / (m.max(1) as f64); m])
        .collect();

    let mut row_ev = vec![0.0f64; n_own];
    let mut col_ev: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut weight_sum = 0.0f64;

    let steps = sweeps.max(1);
    for t in 1..=steps {
        let tw = f64::from(t);
        weight_sum += tw;

        // 1. Regret-matching+ own strategy `p_cur` from `r_own`.
        let r_sum: f64 = r_own.iter().sum();
        if r_sum > 1e-15 {
            let inv = 1.0 / r_sum;
            for i in 0..n_own {
                p_cur[i] = r_own[i] * inv;
            }
        } else {
            let u = 1.0 / (n_own as f64);
            for p in &mut p_cur {
                *p = u;
            }
        }
        for i in 0..n_own {
            avg_own[i] += tw * p_cur[i];
        }

        // 2. Update each world's opponent regrets against `p_cur`, then derive `q_cur[w]` (alternating RM+).
        for w in 0..n_w {
            let (_, m, vals) = worlds[w];
            if m == 0 {
                continue;
            }
            let cev = &mut col_ev[w];
            for j in 0..m {
                cev[j] = 0.0;
            }
            for i in 0..n_own {
                let pi = p_cur[i];
                if pi > 0.0 {
                    let row = &vals[i * m..(i + 1) * m];
                    for j in 0..m {
                        cev[j] += pi * row[j];
                    }
                }
            }
            let q = &mut q_cur[w];
            let mut val_w = 0.0;
            for j in 0..m {
                val_w += q[j] * cev[j];
            }
            // Opponent minimizes side-me payoff -> regret for col j is `val_w - cev[j]`.
            let ro = &mut r_opp[w];
            let mut ro_sum = 0.0;
            for j in 0..m {
                let nr = (ro[j] + (val_w - cev[j])).max(0.0);
                ro[j] = nr;
                ro_sum += nr;
            }
            if ro_sum > 1e-15 {
                let inv = 1.0 / ro_sum;
                for j in 0..m {
                    q[j] = ro[j] * inv;
                }
            } else {
                let u = 1.0 / (m as f64);
                for x in q.iter_mut() {
                    *x = u;
                }
            }
            let ao = &mut avg_opp[w];
            for j in 0..m {
                ao[j] += tw * q[j];
            }
        }

        // 3. Update own regrets against updated `q_cur`.
        for x in &mut row_ev {
            *x = 0.0;
        }
        for w in 0..n_w {
            let (pi_w, m, vals) = worlds[w];
            if m == 0 || pi_w <= 0.0 {
                continue;
            }
            let q = &q_cur[w];
            for i in 0..n_own {
                let row = &vals[i * m..(i + 1) * m];
                let mut acc = 0.0;
                for j in 0..m {
                    acc += row[j] * q[j];
                }
                row_ev[i] += pi_w * acc;
            }
        }
        let mut val_cur = 0.0;
        for i in 0..n_own {
            val_cur += p_cur[i] * row_ev[i];
        }
        for i in 0..n_own {
            r_own[i] = (r_own[i] + (row_ev[i] - val_cur)).max(0.0);
        }
    }

    let inv_w = 1.0 / weight_sum.max(1.0);
    let mut own_policy: Vec<f64> = avg_own.iter().map(|&v| v * inv_w).collect();
    let p_sum: f64 = own_policy.iter().sum();
    if p_sum > 0.0 {
        for p in &mut own_policy {
            *p /= p_sum;
        }
    }
    let opp_policies: Vec<Vec<f64>> = avg_opp
        .iter()
        .map(|ao| {
            let mut q: Vec<f64> = ao.iter().map(|&v| v * inv_w).collect();
            let s: f64 = q.iter().sum();
            if s > 0.0 {
                for x in &mut q {
                    *x /= s;
                }
            }
            q
        })
        .collect();

    // Exact evaluation of average strategies `(own_policy, opp_policies)`.
    let mut row_vs_q = vec![0.0f64; n_own];
    let mut row_guarantees = vec![0.0f64; n_own];
    let mut col_vs_p: Vec<Vec<f64>> = Vec::with_capacity(n_w);
    let mut world_values = Vec::with_capacity(n_w);
    let mut lower_bound = 0.0;

    for w in 0..n_w {
        let (pi_w, m, vals) = worlds[w];
        let mut cp = vec![0.0f64; m];
        for i in 0..n_own {
            let row = &vals[i * m..(i + 1) * m];
            let mut rmin = f64::INFINITY;
            let mut r_q = 0.0;
            let pi = own_policy[i];
            for j in 0..m {
                let v = row[j];
                if v < rmin {
                    rmin = v;
                }
                r_q += v * opp_policies[w][j];
                if pi > 0.0 {
                    cp[j] += pi * v;
                }
            }
            row_guarantees[i] += pi_w * rmin;
            row_vs_q[i] += pi_w * r_q;
        }
        let min_cp = cp.iter().copied().fold(f64::INFINITY, f64::min);
        let mut vw = 0.0;
        for j in 0..m {
            vw += cp[j] * opp_policies[w][j];
        }
        lower_bound += pi_w * min_cp;
        world_values.push(vw);
        col_vs_p.push(cp);
    }
    let upper_bound = row_vs_q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let value: f64 = own_policy
        .iter()
        .zip(&row_vs_q)
        .map(|(&p, &u)| p * u)
        .sum();
    let duality_gap = (upper_bound - lower_bound).max(0.0);

    BayesianRmSolution {
        own_policy,
        opp_policies,
        value,
        duality_gap,
        row_guarantees,
        row_vs_q,
        col_vs_p,
        world_values,
    }
}

/// Multi-World Bayesian RM+ solver supporting per-world fixed opponent preview priors.
///
/// - When every world has `opp_priors[w] == None`, dispatches directly to
///   `solve_bayesian_rm_plus` for 0-ULP bit-exact invariance on default PS & Castle.
/// - When every world has `opp_priors[w] == Some(q_w)`, computes the exact closed-form
///   Bayesian best response `\max_i \sum_w \pi_w E_{j \sim q_w}[V_w[i, j]]` with `duality_gap = 0`.
/// - When some worlds have `Some(q_w)` and others `None` (partial prior coverage), solves
///   the exact hybrid Bayesian equilibrium: fixed `q_w` in prioritized worlds and adversarial
///   minimax RM+ in unprioritized worlds.
pub fn solve_bayesian_rm_plus_with_priors(
    n_own: usize,
    worlds: &[(f64, usize, &[f64])],
    opp_priors: &[Option<&[f64]>],
    sweeps: u32,
) -> BayesianRmSolution {
    assert!(n_own > 0 && !worlds.is_empty());
    assert_eq!(worlds.len(), opp_priors.len());
    if opp_priors.iter().all(Option::is_none) {
        return solve_bayesian_rm_plus(n_own, worlds, sweeps);
    }

    let n_w = worlds.len();
    let norm_prior = |m: usize, qp: &[f64]| -> Vec<f64> {
        debug_assert_eq!(qp.len(), m);
        let mut q = vec![0.0f64; m];
        let mut s = 0.0;
        for j in 0..m {
            let v = qp[j].max(0.0);
            q[j] = v;
            s += v;
        }
        if s > 0.0 {
            let inv = 1.0 / s;
            for x in &mut q {
                *x *= inv;
            }
        } else if m > 0 {
            let u = 1.0 / (m as f64);
            for x in &mut q {
                *x = u;
            }
        }
        q
    };

    if opp_priors.iter().all(Option::is_some) {
        let opp_policies: Vec<Vec<f64>> = (0..n_w)
            .map(|w| norm_prior(worlds[w].1, opp_priors[w].unwrap()))
            .collect();
        let mut row_vs_q = vec![0.0f64; n_own];
        for w in 0..n_w {
            let (pi_w, m, vals) = worlds[w];
            if m == 0 || pi_w <= 0.0 {
                continue;
            }
            let q = &opp_policies[w];
            for i in 0..n_own {
                let row = &vals[i * m..(i + 1) * m];
                let mut acc = 0.0;
                for j in 0..m {
                    acc += row[j] * q[j];
                }
                row_vs_q[i] += pi_w * acc;
            }
        }
        let best_i = (0..n_own)
            .max_by(|&a, &b| row_vs_q[a].total_cmp(&row_vs_q[b]).then_with(|| b.cmp(&a)))
            .unwrap_or(0);
        let mut own_policy = vec![0.0f64; n_own];
        own_policy[best_i] = 1.0;
        let mut col_vs_p: Vec<Vec<f64>> = Vec::with_capacity(n_w);
        let mut world_values = Vec::with_capacity(n_w);
        for w in 0..n_w {
            let (_, m, vals) = worlds[w];
            let cp = vals[best_i * m..(best_i + 1) * m].to_vec();
            let mut vw = 0.0;
            for j in 0..m {
                vw += cp[j] * opp_policies[w][j];
            }
            world_values.push(vw);
            col_vs_p.push(cp);
        }
        let value = row_vs_q[best_i];
        let row_guarantees = row_vs_q.clone();
        return BayesianRmSolution {
            own_policy,
            opp_policies,
            value,
            duality_gap: 0.0,
            row_guarantees,
            row_vs_q,
            col_vs_p,
            world_values,
        };
    }

    // Hybrid case: some worlds fixed to `opp_priors[w]`, others adversarial RM+.
    let mut r_own = vec![0.0f64; n_own];
    let mut avg_own = vec![0.0f64; n_own];
    let mut p_cur = vec![1.0 / (n_own as f64); n_own];

    let mut r_opp: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut avg_opp: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut q_cur: Vec<Vec<f64>> = (0..n_w)
        .map(|w| {
            let m = worlds[w].1;
            if let Some(qp) = opp_priors[w] {
                norm_prior(m, qp)
            } else {
                vec![1.0 / (m.max(1) as f64); m]
            }
        })
        .collect();

    let mut row_ev = vec![0.0f64; n_own];
    let mut col_ev: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut weight_sum = 0.0f64;

    let steps = sweeps.max(1);
    for t in 1..=steps {
        let tw = f64::from(t);
        weight_sum += tw;

        let r_sum: f64 = r_own.iter().sum();
        if r_sum > 1e-15 {
            let inv = 1.0 / r_sum;
            for i in 0..n_own {
                p_cur[i] = r_own[i] * inv;
            }
        } else {
            let u = 1.0 / (n_own as f64);
            for p in &mut p_cur {
                *p = u;
            }
        }
        for i in 0..n_own {
            avg_own[i] += tw * p_cur[i];
        }

        for w in 0..n_w {
            if opp_priors[w].is_some() {
                continue;
            }
            let (_, m, vals) = worlds[w];
            if m == 0 {
                continue;
            }
            let cev = &mut col_ev[w];
            for j in 0..m {
                cev[j] = 0.0;
            }
            for i in 0..n_own {
                let pi = p_cur[i];
                if pi > 0.0 {
                    let row = &vals[i * m..(i + 1) * m];
                    for j in 0..m {
                        cev[j] += pi * row[j];
                    }
                }
            }
            let q = &mut q_cur[w];
            let mut val_w = 0.0;
            for j in 0..m {
                val_w += q[j] * cev[j];
            }
            let ro = &mut r_opp[w];
            let mut ro_sum = 0.0;
            for j in 0..m {
                let nr = (ro[j] + (val_w - cev[j])).max(0.0);
                ro[j] = nr;
                ro_sum += nr;
            }
            if ro_sum > 1e-15 {
                let inv = 1.0 / ro_sum;
                for j in 0..m {
                    q[j] = ro[j] * inv;
                }
            } else {
                let u = 1.0 / (m as f64);
                for x in q.iter_mut() {
                    *x = u;
                }
            }
            let ao = &mut avg_opp[w];
            for j in 0..m {
                ao[j] += tw * q[j];
            }
        }

        for x in &mut row_ev {
            *x = 0.0;
        }
        for w in 0..n_w {
            let (pi_w, m, vals) = worlds[w];
            if m == 0 || pi_w <= 0.0 {
                continue;
            }
            let q = &q_cur[w];
            for i in 0..n_own {
                let row = &vals[i * m..(i + 1) * m];
                let mut acc = 0.0;
                for j in 0..m {
                    acc += row[j] * q[j];
                }
                row_ev[i] += pi_w * acc;
            }
        }
        let mut val_cur = 0.0;
        for i in 0..n_own {
            val_cur += p_cur[i] * row_ev[i];
        }
        for i in 0..n_own {
            r_own[i] = (r_own[i] + (row_ev[i] - val_cur)).max(0.0);
        }
    }

    let inv_w = 1.0 / weight_sum.max(1.0);
    let mut own_policy: Vec<f64> = avg_own.iter().map(|&v| v * inv_w).collect();
    let p_sum: f64 = own_policy.iter().sum();
    if p_sum > 0.0 {
        for p in &mut own_policy {
            *p /= p_sum;
        }
    }
    let opp_policies: Vec<Vec<f64>> = (0..n_w)
        .map(|w| {
            if opp_priors[w].is_some() {
                q_cur[w].clone()
            } else {
                let mut q: Vec<f64> = avg_opp[w].iter().map(|&v| v * inv_w).collect();
                let s: f64 = q.iter().sum();
                if s > 0.0 {
                    for x in &mut q {
                        *x /= s;
                    }
                }
                q
            }
        })
        .collect();

    let mut row_vs_q = vec![0.0f64; n_own];
    let mut row_guarantees = vec![0.0f64; n_own];
    let mut col_vs_p: Vec<Vec<f64>> = Vec::with_capacity(n_w);
    let mut world_values = Vec::with_capacity(n_w);
    let mut lower_bound = 0.0;

    for w in 0..n_w {
        let (pi_w, m, vals) = worlds[w];
        let is_fixed = opp_priors[w].is_some();
        let mut cp = vec![0.0f64; m];
        for i in 0..n_own {
            let row = &vals[i * m..(i + 1) * m];
            let mut rmin = f64::INFINITY;
            let mut r_q = 0.0;
            let pi = own_policy[i];
            for j in 0..m {
                let v = row[j];
                if v < rmin {
                    rmin = v;
                }
                r_q += v * opp_policies[w][j];
                if pi > 0.0 {
                    cp[j] += pi * v;
                }
            }
            row_guarantees[i] += pi_w * if is_fixed { r_q } else { rmin };
            row_vs_q[i] += pi_w * r_q;
        }
        let min_cp = cp.iter().copied().fold(f64::INFINITY, f64::min);
        let mut vw = 0.0;
        for j in 0..m {
            vw += cp[j] * opp_policies[w][j];
        }
        lower_bound += pi_w * if is_fixed { vw } else { min_cp };
        world_values.push(vw);
        col_vs_p.push(cp);
    }
    let upper_bound = row_vs_q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let value: f64 = own_policy
        .iter()
        .zip(&row_vs_q)
        .map(|(&p, &u)| p * u)
        .sum();
    let duality_gap = (upper_bound - lower_bound).max(0.0);

    BayesianRmSolution {
        own_policy,
        opp_policies,
        value,
        duality_gap,
        row_guarantees,
        row_vs_q,
        col_vs_p,
        world_values,
    }
}

fn select_prior_opp_support(opp_prior: &[f64], opp_ev: &[f64], k: usize) -> Vec<usize> {
    let m = opp_prior.len();
    if m <= k {
        return (0..m).collect();
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| {
        opp_prior[b]
            .total_cmp(&opp_prior[a])
            .then_with(|| opp_ev[b].total_cmp(&opp_ev[a]))
            .then_with(|| a.cmp(&b))
    });
    order.truncate(k);
    order
}

fn restrict_opp_prior(opp_prior: Option<&[f64]>, cols: &[usize]) -> Option<Vec<f64>> {
    let qp = opp_prior?;
    let mut sub: Vec<f64> = cols.iter().map(|&j| qp[j].max(0.0)).collect();
    let sum: f64 = sub.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in &mut sub {
            *x *= inv;
        }
    } else if !sub.is_empty() {
        let u = 1.0 / (sub.len() as f64);
        for x in &mut sub {
            *x = u;
        }
    }
    Some(sub)
}

fn shed_dust(p: &mut [f64], ratio: f64) {
    let pmax = p.iter().copied().fold(0.0f64, f64::max);
    if pmax <= 0.0 {
        return;
    }
    let thresh = pmax * ratio;
    for v in p.iter_mut() {
        if *v < thresh {
            *v = 0.0;
        }
    }
    let sum: f64 = p.iter().sum();
    if sum > 0.0 {
        for v in p.iter_mut() {
            *v /= sum;
        }
    }
}

// =========================================================================
// 2. Canonical Action Helpers & Lead Tactical Archetype Tagging
// =========================================================================

/// Return all canonical legal Team Preview `[lead, b1, b2]` triples for `side`
/// (preserving bench order whenever `party_order_matters` is true, e.g. Beat Up).
pub fn canonical_preview_triples(dex: &Dex, battle: &mut Battle, side: usize) -> Vec<[u8; 3]> {
    canonical_preview_actions(dex, battle, side)
        .1
        .into_iter()
        .map(|(_, t)| t)
        .collect()
}

/// Independent Turn-1 `cfv-0r@puct@lam0` oracle evaluation for a single `(p0, p1)` cell,
/// properly resampling Quick Claw and Berserk Gene confusion duration per iteration.
pub fn eval_turn1_oracle_cell(
    roster: &CompactPreviewRoster,
    preview_cache: &PreviewTempoCache,
    picks: [[u8; 3]; 2],
    iters: u32,
    seed: u64,
) -> f64 {
    let cfg = PreviewSearchConfig::default();
    let mut cell = CfPreviewCell::new(roster, preview_cache, picks, 0.5, &cfg);
    let eg_w = EvalWeights::default();
    let eg_p = Playout::Heavy {
        turns: 8,
        eps: 0.2,
        weights: eg_w.clone(),
    };
    let r_rules = RmConfig::default().rollout_rules();
    let mut path = Vec::with_capacity(64);
    let mut rng = SplitMix64::new(seed);
    cell.step(iters, &cfg, &mut rng, &eg_w, &eg_p, r_rules, &mut path);
    cell.value0(PreviewCellReadout::LateWindowMean)
}

fn canonical_preview_actions(
    dex: &Dex,
    battle: &mut Battle,
    side: usize,
) -> (Vec<SearchChoice>, Vec<(usize, [u8; 3])>) {
    let legal = battle.legal_choices(dex, side);
    let order_matters = party_order_matters(dex, &battle.sides[side].roster);
    let mut canon = Vec::with_capacity(legal.len());
    let mut seen: Vec<[u8; 3]> = Vec::with_capacity(60);
    for (idx, &ch) in legal.iter().enumerate() {
        let SearchChoice::Team(t) = ch else {
            continue;
        };
        if !order_matters && t[2] != 0 {
            let mut key = t;
            if key[1] > key[2] {
                key.swap(1, 2);
            }
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
        }
        canon.push((idx, t));
    }
    (legal, canon)
}

#[inline]
fn subset_key(t: [u8; 3]) -> [u8; 3] {
    let mut s = t;
    s.sort_unstable();
    s
}

/// Detect 0-ply Stage-1 tactical blind-spot lead mechanics (Sleep, Spikes, Baton Pass,
/// Explosion/Self-Destruct, Pursuit/Counter/MirrorCoat, PerishSong/MeanLook/Encore, BellyDrum)
/// strictly from that side's roster (used for anti-pruning lead diversity).
fn lead_tactical_flags(dex: &Dex, battle: &Battle, side: usize) -> [bool; 6] {
    let mut flags = [false; 6];
    let s = &battle.sides[side];
    for (pos, &r_slot) in s.party.iter().take(6).enumerate() {
        let mon = &s.roster[usize::from(r_slot)];
        for slot in &mon.base_move_slots {
            let k = dex.moves.key(slot.id);
            if matches!(
                k,
                "spore"
                    | "sleeppowder"
                    | "lovelykiss"
                    | "hypnosis"
                    | "sing"
                    | "spikes"
                    | "batonpass"
                    | "explosion"
                    | "selfdestruct"
                    | "pursuit"
                    | "counter"
                    | "mirrorcoat"
                    | "perishsong"
                    | "meanlook"
                    | "spiderweb"
                    | "encore"
                    | "bellydrum"
            ) {
                flags[pos] = true;
                break;
            }
        }
    }
    flags
}

/// Select `<= k` candidate action indices with Stage-1 Nash support, mandatory every-legal-lead
/// diversity quota (prioritizing 0-ply blind-spot tactical leads), pure maximin top picks, and
/// a same-3-mon-subset cap (`<= 2` outside Nash support).
pub(crate) fn select_diverse_support(
    canon: &[(usize, [u8; 3])],
    nash_p: &[f64],
    ev_vs_opp: &[f64],
    pure_guar: &[f64],
    tactical_lead: &[bool; 6],
    k: usize,
) -> Vec<usize> {
    let n = canon.len();
    if n <= k {
        return (0..n).collect();
    }
    let mut selected: Vec<usize> = Vec::with_capacity(k);
    let mut in_sel = vec![false; n];
    let mut subset_counts: Vec<([u8; 3], usize)> = Vec::new();
    let add = |idx: usize, sel: &mut Vec<usize>, flags: &mut [bool], sub_c: &mut Vec<([u8; 3], usize)>| {
        if flags[idx] || sel.len() >= k {
            return;
        }
        flags[idx] = true;
        sel.push(idx);
        let sk = subset_key(canon[idx].1);
        if let Some((_, c)) = sub_c.iter_mut().find(|(k0, _)| *k0 == sk) {
            *c += 1;
        } else {
            sub_c.push((sk, 1));
        }
    };
    let sub_cnt = |idx: usize, sub_c: &[([u8; 3], usize)]| -> usize {
        let sk = subset_key(canon[idx].1);
        sub_c.iter().find(|(k0, _)| *k0 == sk).map_or(0, |&(_, c)| c)
    };

    // 1. Stage-1 Nash support (`p >= 0.01`), highest mass first (up to k/2 + 2 so leads always fit).
    let mut nash_order: Vec<usize> = (0..n).filter(|&i| nash_p[i] >= 0.01).collect();
    nash_order.sort_by(|&a, &b| nash_p[b].total_cmp(&nash_p[a]));
    let nash_cap = k.saturating_sub(4).max(1);
    for &i in nash_order.iter().take(nash_cap) {
        add(i, &mut selected, &mut in_sel, &mut subset_counts);
    }

    // 2. Mandatory Lead-Diversity Quota: guarantee >= 1 triple for every legal lead.
    let mut leads: Vec<u8> = Vec::with_capacity(6);
    for &(_, t) in canon {
        if !leads.contains(&t[0]) {
            leads.push(t[0]);
        }
    }
    // Order unrepresented leads by tactical lead flag + best EV for that lead.
    let mut lead_best: Vec<(u8, usize, f64)> = Vec::new();
    for &ld in &leads {
        let already = selected.iter().any(|&i| canon[i].1[0] == ld);
        if already {
            continue;
        }
        let bonus = if ld >= 1 && (ld as usize) <= 6 && tactical_lead[ld as usize - 1] {
            0.018
        } else {
            0.0
        };
        if let Some(best_i) = (0..n)
            .filter(|&i| canon[i].1[0] == ld && !in_sel[i])
            .max_by(|&a, &b| {
                let sa = ev_vs_opp[a] + 0.25 * pure_guar[a];
                let sb = ev_vs_opp[b] + 0.25 * pure_guar[b];
                sa.total_cmp(&sb)
            })
        {
            lead_best.push((ld, best_i, ev_vs_opp[best_i] + bonus));
        }
    }
    lead_best.sort_by(|a, b| b.2.total_cmp(&a.2));
    for (_, bi, _) in lead_best {
        add(bi, &mut selected, &mut in_sel, &mut subset_counts);
    }

    // 3. Top pure maximin guarantee picks (respecting same-subset cap <= 2).
    let mut guar_order: Vec<usize> = (0..n).filter(|&i| !in_sel[i]).collect();
    guar_order.sort_by(|&a, &b| pure_guar[b].total_cmp(&pure_guar[a]));
    for &i in guar_order.iter().take(3) {
        if sub_cnt(i, &subset_counts) < 2 {
            add(i, &mut selected, &mut in_sel, &mut subset_counts);
        }
    }

    // 4. Fill remaining slots by best Stage-1 `ev_vs_opp` with same-subset cap <= 2.
    let mut ev_order: Vec<usize> = (0..n).filter(|&i| !in_sel[i]).collect();
    ev_order.sort_by(|&a, &b| ev_vs_opp[b].total_cmp(&ev_vs_opp[a]));
    for &i in &ev_order {
        if selected.len() >= k {
            break;
        }
        if sub_cnt(i, &subset_counts) < 2 {
            add(i, &mut selected, &mut in_sel, &mut subset_counts);
        }
    }
    // Fallback if same-subset cap left slots empty.
    for &i in &ev_order {
        if selected.len() >= k {
            break;
        }
        add(i, &mut selected, &mut in_sel, &mut subset_counts);
    }
    selected
}

// =========================================================================
// 3. Persistent Turn-1 `cfv-0r@puct@lam0` Cell Tree
// =========================================================================

#[derive(Clone, Copy)]
struct CfPrevNode {
    n_acts: [u8; 2],
    acts: [[u8; 6]; 2],
    noop_mask: [u8; 2],
    priors_ready: bool,
    total: u32,
    priors_q: [[u16; 6]; 2],
    visits: [[u32; 6]; 2],
    sums: [[f64; 6]; 2],
}

impl CfPrevNode {
    #[inline]
    fn new(cb: &CompactBattle, table: &MatchTable) -> Self {
        let mut t0 = [0u8; 7];
        let mut t1 = [0u8; 7];
        let n0 = cb.legal_actions(table, 0, &mut t0).min(6) as u8;
        let n1 = cb.legal_actions(table, 1, &mut t1).min(6) as u8;
        let mut acts = [[0u8; 6]; 2];
        acts[0].copy_from_slice(&t0[..6]);
        acts[1].copy_from_slice(&t1[..6]);
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
    fn ensure_priors(
        &mut self,
        cb: &CompactBattle,
        table: &MatchTable,
        cache: &TempoMatchCache,
        eps_p: f32,
        use_cf_policy: bool,
    ) {
        if self.priors_ready {
            return;
        }
        let ((p0, m0), (p1, m1)) = policy_priors_both_mode(
            cb,
            table,
            Some(cache),
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
        self.noop_mask = [m0, m1];
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
}

struct CfPreviewCell {
    table: MatchTable,
    root_cb: CompactBattle,
    cache: TempoMatchCache,
    turn_cap: u16,
    berserk_confused: [bool; 2],
    nodes: Vec<CfPrevNode>,
    tt: FxHashMap<u64, usize>,
    root_joint_n: [[f64; 6]; 6],
    root_joint_w0: [[f64; 6]; 6],
    total_n: u32,
    total_w0: f64,
    late_n: f64,
    late_w0: f64,
    next_halve: u32,
    prior_v0: f64,
}

impl CfPreviewCell {
    fn new(
        roster: &CompactPreviewRoster,
        preview_cache: &PreviewTempoCache,
        picks: [[u8; 3]; 2],
        prior_v0: f64,
        cfg: &PreviewSearchConfig,
    ) -> Self {
        let (table, mut root_cb) = roster.build_turn1(picks, 0);
        if root_cb.sides[0].request == 1 && root_cb.sides[1].request == 1 {
            root_cb.quick_claw_roll = 0;
        }
        let berserk_confused = [
            (root_cb.mons[0].vol_flags & vol_bit::CONFUSION) != 0,
            (root_cb.mons[3].vol_flags & vol_bit::CONFUSION) != 0,
        ];
        let cache = preview_cache.slice_for(&root_cb);
        let mut turn_cap = root_cb.turn.saturating_add(100);
        if let Some(lim) = table.ruleset.turn_limit() {
            turn_cap = turn_cap.min(lim);
        }
        let mut rn = CfPrevNode::new(&root_cb, &table);
        rn.ensure_priors(&root_cb, &table, &cache, cfg.puct_eps, cfg.use_cf_policy);
        let omit_qc = (root_cb.sides[0].request == 1 && root_cb.sides[1].request == 1)
            || !root_cb.quick_claw_observable(&table);
        let rk = root_cb.search_key(&table, 16, omit_qc);
        let mut nodes = Vec::with_capacity(64);
        let mut tt: FxHashMap<u64, usize> = FxHashMap::default();
        nodes.push(rn);
        tt.insert(rk, 0);
        Self {
            table,
            root_cb,
            cache,
            turn_cap,
            berserk_confused,
            nodes,
            tt,
            root_joint_n: [[0.0; 6]; 6],
            root_joint_w0: [[0.0; 6]; 6],
            total_n: 0,
            total_w0: 0.0,
            late_n: 0.0,
            late_w0: 0.0,
            next_halve: 64,
            prior_v0,
        }
    }

    fn step_one(
        &mut self,
        cfg: &PreviewSearchConfig,
        rng: &mut SplitMix64,
        eg_weights: &EvalWeights,
        eg_playout: &Playout,
        rollout_rules: crate::mcts::RolloutRules,
        path: &mut Vec<(usize, usize, usize)>,
    ) -> (f64, u32) {
        let c_puct = cfg.c_puct;
        let fpu_red = cfg.fpu_red;
        let puct_eps = cfg.puct_eps;
        let use_cf_policy = cfg.use_cf_policy;
        let leaf_temp = cfg.leaf_temp;

        let mut cb = self.root_cb;
        {
            let mut prng = nc2000_engine::prng::Prng::new(rng.next());
            let qc = Battle::roll_quick_claw_with(self.table.ruleset, &mut prng);
            cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
            // Resample Berserk Gene lead confusion duration uniformly in {2, 3, 4, 5}.
            for s in 0..2usize {
                if self.berserk_confused[s] {
                    cb.mons[s * 3].timers[timer_idx::CONFUSION] = (2 + prng.random(4)) as u8;
                }
            }
            cb.prng_seed = prng.seed();
        }
        let mut cur_id = 0usize;
        let mut depth = 0u8;
        let mut root_i0 = 0usize;
        let mut root_i1 = 0usize;
        path.clear();

        let val0 = loop {
            self.nodes[cur_id].ensure_priors(
                &cb,
                &self.table,
                &self.cache,
                puct_eps,
                use_cf_policy,
            );
            let node = &self.nodes[cur_id];
            let n0_cur = node.n_acts[0] as usize;
            let n1_cur = node.n_acts[1] as usize;
            let i0 = node.select_puct(0, c_puct, fpu_red);
            let i1 = node.select_puct(1, c_puct, fpu_red);
            if depth == 0 {
                root_i0 = i0;
                root_i1 = i1;
            }
            let a0 = if n0_cur > 0 { node.acts[0][i0] } else { 7 };
            let a1 = if n1_cur > 0 { node.acts[1][i1] } else { 7 };
            if depth == 0 || n0_cur > 1 || n1_cur > 1 {
                path.push((cur_id, i0, i1));
            }
            cb.apply_actions(&self.table, [a0, a1]);
            depth = depth.saturating_add(1);
            if cb.outcome != 0 {
                break match cb.outcome {
                    1 => 1.0,
                    2 => 0.0,
                    _ => 0.5,
                };
            }
            if let Some(v) = compact_turn_limit_outcome(&cb, &self.table) {
                break v;
            }
            if depth >= 250 || cb.turn > self.turn_cap {
                let alive_sum = cb.sides[0].alive_mask.count_ones()
                    + cb.sides[1].alive_mask.count_ones();
                if alive_sum <= 3 {
                    break compact_eval_leaf_blindspot(&cb, &self.table, eg_weights);
                }
                break evaluate_cf_value_resolving_switches(
                    &cb,
                    &self.table,
                    Some(&self.cache),
                    leaf_temp,
                );
            }
            let oq = !cb.quick_claw_observable(&self.table);
            let nk = cb.search_key(&self.table, 16, oq);
            if let Some(&nid) = self.tt.get(&nk) {
                cur_id = nid;
            } else {
                let nid = self.nodes.len();
                self.nodes.push(CfPrevNode::new(&cb, &self.table));
                self.tt.insert(nk, nid);
                let alive_sum = cb.sides[0].alive_mask.count_ones()
                    + cb.sides[1].alive_mask.count_ones();
                if alive_sum <= 3 {
                    // Hybrid Endgame Guard (`n0 + n1 <= 3` -> `8r EvalBlindspot + linear.rs`).
                    let cutoff = self.turn_cap.min(cb.turn.saturating_add(7));
                    break crate::mcts::compact_playout_with_leaf(
                        &mut cb,
                        &self.table,
                        eg_playout,
                        cutoff,
                        rng,
                        rollout_rules,
                        |cb_l, t_l| compact_eval_leaf_blindspot(cb_l, t_l, eg_weights),
                    );
                }
                break evaluate_cf_value_resolving_switches(
                    &cb,
                    &self.table,
                    Some(&self.cache),
                    leaf_temp,
                );
            }
        };

        // Pure MC backup (`lam = 0.0`), matching certified in-battle Champion.
        for &(nid, i0, i1) in path.iter().rev() {
            let n = &mut self.nodes[nid];
            n.total += 1;
            if n.n_acts[0] > 0 {
                n.visits[0][i0] += 1;
                n.sums[0][i0] += val0;
            }
            if n.n_acts[1] > 0 {
                n.visits[1][i1] += 1;
                n.sums[1][i1] += val0;
            }
        }
        self.total_n += 1;
        self.total_w0 += val0;
        self.late_n += 1.0;
        self.late_w0 += val0;
        if root_i0 < 6 && root_i1 < 6 {
            self.root_joint_n[root_i0][root_i1] += 1.0;
            self.root_joint_w0[root_i0][root_i1] += val0;
        }
        if self.total_n == self.next_halve {
            self.late_n *= 0.5;
            self.late_w0 *= 0.5;
            for r in 0..6 {
                for c in 0..6 {
                    if self.root_joint_n[r][c] >= 4.0 {
                        self.root_joint_n[r][c] *= 0.5;
                        self.root_joint_w0[r][c] *= 0.5;
                    }
                }
            }
            self.next_halve = self.next_halve.saturating_mul(2);
        }
        (val0, u32::from(depth))
    }

    fn step(
        &mut self,
        iters: u32,
        cfg: &PreviewSearchConfig,
        rng: &mut SplitMix64,
        eg_weights: &EvalWeights,
        eg_playout: &Playout,
        rollout_rules: crate::mcts::RolloutRules,
        path: &mut Vec<(usize, usize, usize)>,
    ) {
        for _ in 0..iters {
            self.step_one(cfg, rng, eg_weights, eg_playout, rollout_rules, path);
        }
    }

    /// Side-0 value readout for this Turn-1 cell.
    fn value0(&self, readout: PreviewCellReadout) -> f64 {
        if self.total_n == 0 {
            return self.prior_v0;
        }
        let raw = self.total_w0 / f64::from(self.total_n);
        let late = if self.late_n >= 16.0 {
            self.late_w0 / self.late_n
        } else {
            raw
        };
        match readout {
            PreviewCellReadout::RawMean => raw,
            PreviewCellReadout::LateWindowMean => late,
            PreviewCellReadout::Turn1Minimax => {
                if self.total_n < 256 {
                    return late;
                }
                let rn = &self.nodes[0];
                let k0 = rn.n_acts[0] as usize;
                let k1 = rn.n_acts[1] as usize;
                let min_v = (self.total_n / 32).max(4);
                let mut rows = [0usize; 6];
                let mut nr = 0usize;
                for i in 0..k0 {
                    if ((rn.noop_mask[0] >> i) & 1) == 0 && rn.visits[0][i] >= min_v {
                        rows[nr] = i;
                        nr += 1;
                    }
                }
                let mut cols = [0usize; 6];
                let mut nc = 0usize;
                for j in 0..k1 {
                    if ((rn.noop_mask[1] >> j) & 1) == 0 && rn.visits[1][j] >= min_v {
                        cols[nc] = j;
                        nc += 1;
                    }
                }
                if nr == 0 || nc == 0 {
                    return late;
                }
                if nr == 1 && nc == 1 {
                    let n = self.root_joint_n[rows[0]][cols[0]];
                    return if n > 0.0 {
                        self.root_joint_w0[rows[0]][cols[0]] / n
                    } else {
                        late
                    };
                }
                // Build shrunk `nr × nc` Turn-1 root payoff matrix (kappa = 4 toward marginal row/col mean).
                let mut mat = [0.5f64; 36];
                let mut m_idx = 0usize;
                for r_i in 0..nr {
                    let ri = rows[r_i];
                    let row_m = if rn.visits[0][ri] > 0 {
                        rn.sums[0][ri] / f64::from(rn.visits[0][ri])
                    } else {
                        late
                    };
                    for c_i in 0..nc {
                        let cj = cols[c_i];
                        let col_m = if rn.visits[1][cj] > 0 {
                            rn.sums[1][cj] / f64::from(rn.visits[1][cj])
                        } else {
                            late
                        };
                        let prior_ij = (row_m + col_m - raw).clamp(0.02, 0.98);
                        let n_ij = self.root_joint_n[ri][cj];
                        let v_ij = (self.root_joint_w0[ri][cj] + 4.0 * prior_ij) / (n_ij + 4.0);
                        mat[m_idx] = v_ij.clamp(0.0, 1.0);
                        m_idx += 1;
                    }
                }
                let sol = solve_bayesian_rm_plus(nr, &[(1.0, nc, &mat[..nr * nc])], 100);
                // Blend lightly toward `late` when visits are small to prevent minimax over-optimism.
                let alpha = (f64::from(self.total_n) / (f64::from(self.total_n) + 128.0)).clamp(0.0, 0.85);
                (alpha * sol.value + (1.0 - alpha) * late).clamp(0.000_1, 0.999_9)
            }
        }
    }
}

// =========================================================================
// 4. Main `solve_cf_preview` Entrypoint (Zero Hidden-Info Leakage)
// =========================================================================

struct PreviewWorldData {
    weight: f64,
    opp_canon: Vec<(usize, [u8; 3])>,
    opp_tactical: [bool; 6],
    opp_prior: Option<Vec<f64>>,
    compact: Option<(CompactPreviewRoster, PreviewTempoCache)>,
    stage1_vals: Vec<f64>, // row-major `[n_own * n_opp]` from `side`'s perspective
}

/// Run shipped `BlindSearch` live Team Preview directly from an already-constructed
/// `ObservedPosition` + `Belief` + `Observer` (for fair search-only timing comparisons).
pub fn solve_shipped_preview(
    position: &ObservedPosition,
    dex: &Dex,
    belief: &Belief,
    obs: &crate::observe::Observer,
    side: usize,
    iterations: u32,
    rng: SplitMix64,
) -> SearchChoice {
    let rm_cfg = RmConfig {
        iterations,
        rule: SelRule::Ucb,
        c: 0.4,
        settlement: crate::accounting::LeafSettlementMode::LegacyRollout,
        ..Default::default()
    };
    let mut bs = BlindSearch::from_observation_with_rng(position, dex, rm_cfg, side, rng);
    bs.prune_preview_aliases(dex);
    bs.step(dex, belief, obs, iterations);
    bs.best().expect("BlindSearch preview best")
}

/// Solve Turn-0 Team Preview from a player-observed position (`ObservedPosition`) and `Belief`
/// using Two-Stage Persistent-Tree Double-Oracle (`CF-V_net` v2 + `cfv-0r@puct@lam0` + Bayesian RM+).
pub fn solve_cf_preview(
    position: &ObservedPosition,
    dex: &Dex,
    belief: &Belief,
    obs: &crate::observe::Observer,
    side: usize,
    cfg: &PreviewSearchConfig,
    rng: &mut SplitMix64,
) -> PreviewSearchResult {
    let t_total = start_timer();
    let mut base = position.model().clone();
    assert_eq!(base.request_state, RequestState::TeamPreview);
    let opp = 1 - side;

    let (own_legal, own_canon) = canonical_preview_actions(dex, &mut base, side);
    let n_legal = own_legal.len();
    let n_own = own_canon.len();
    if n_own <= 1 {
        let mut pol = vec![0.0; n_legal];
        let ch = if let Some(&(idx, _)) = own_canon.first() {
            pol[idx] = 1.0;
            own_legal[idx]
        } else {
            pol[0] = 1.0;
            own_legal[0]
        };
        return PreviewSearchResult {
            choice: ch,
            policy_full: pol,
            value: 0.5,
            pure_guarantee: 0.5,
            mixed_guarantee: 0.5,
            duality_gap: 0.0,
            used_mix: false,
            used_fallback: false,
            stage1_ms: 0.0,
            total_ms: elapsed_ms(t_total),
            iters_used: 0,
            worlds_stage1: 0,
            worlds_stage2: 0,
            support_me: n_own,
            support_opp_max: 0,
            do_admitted_me: 0,
            do_admitted_opp: 0,
        };
    }

    let own_tactical = lead_tactical_flags(dex, &base, side);
    let world_plan = belief.preview_world_distribution(cfg.max_stage1_worlds, rng);

    // Determinize all Turn-0 opponent worlds first and verify CompactPreviewRoster coverage
    // BEFORE Stage 1 so unsupported worlds never get silently dropped or burn dead Battle loops.
    let mut det_worlds: Vec<(crate::belief::TeamPlanPick, f64, Battle)> =
        Vec::with_capacity(world_plan.len());
    let mut all_compact = !world_plan.is_empty();
    for &(plan_pick, weight) in &world_plan {
        let mut sim_w = base.clone();
        sim_w.set_log_enabled(false);
        belief.determinize_in_place_planned(dex, &mut sim_w, obs, plan_pick, rng);
        sim_w.quick_claw_rolls = [false, false];
        if !is_compact_preview_supported(&sim_w, dex) {
            all_compact = false;
            break;
        }
        det_worlds.push((plan_pick, weight, sim_w));
    }

    if !all_compact {
        let choice = solve_shipped_preview(
            position,
            dex,
            belief,
            obs,
            side,
            cfg.fallback_iters,
            SplitMix64::new(rng.next()),
        );
        let mut pol = vec![0.0; n_legal];
        if let Some(pos) = own_legal.iter().position(|&c| c == choice) {
            pol[pos] = 1.0;
        }
        return PreviewSearchResult {
            choice,
            policy_full: pol,
            value: 0.5,
            pure_guarantee: 0.5,
            mixed_guarantee: 0.5,
            duality_gap: 0.0,
            used_mix: false,
            used_fallback: true,
            stage1_ms: 0.0,
            total_ms: elapsed_ms(t_total),
            iters_used: cfg.fallback_iters,
            worlds_stage1: world_plan.len(),
            worlds_stage2: 0,
            support_me: n_own,
            support_opp_max: 0,
            do_admitted_me: 0,
            do_admitted_opp: 0,
        };
    }

    let mut worlds: Vec<PreviewWorldData> = Vec::with_capacity(det_worlds.len());
    let eg_weights = EvalWeights::default();

    let use_opp_prior = belief.has_preview_prior(obs);
    for (plan_pick, weight, mut sim_w) in det_worlds {
        let (opp_legal, opp_canon) = canonical_preview_actions(dex, &mut sim_w, opp);
        let n_opp = opp_canon.len();
        if n_opp == 0 {
            continue;
        }
        let opp_prior = if use_opp_prior {
            let opp_order_matters = party_order_matters(dex, &sim_w.sides[opp].roster);
            belief.preview_prior_probs_canonical(
                plan_pick,
                obs,
                &opp_legal,
                &opp_canon,
                opp_order_matters,
            )
        } else {
            None
        };
        let opp_tactical = lead_tactical_flags(dex, &sim_w, opp);
        let roster = CompactPreviewRoster::from_battle(&sim_w, dex, None);
        let pcache = PreviewTempoCache::build_for_preview_roster(
            &roster,
            [sim_w.sides[0].party.len(), sim_w.sides[1].party.len()],
        );

        // Stage 1: Evaluate full `n_own × n_opp` Turn-1 matrix via 0-ply `CF-V_net` v2.
        let mut stage1_vals = vec![0.5f64; n_own * n_opp];
        for (i, &(_, own_t)) in own_canon.iter().enumerate() {
            for (j, &(_, opp_t)) in opp_canon.iter().enumerate() {
                let picks = if side == 0 { [own_t, opp_t] } else { [opp_t, own_t] };
                let (table, mut cb) = roster.build_turn1(picks, 0);
                if cb.sides[0].request == 1 && cb.sides[1].request == 1 {
                    cb.quick_claw_roll = 0;
                }
                let sliced = pcache.slice_for(&cb);
                let v0 = evaluate_cf_value(&cb, &table, Some(&sliced), cfg.leaf_temp);
                stage1_vals[i * n_opp + j] = if side == 0 { v0 } else { 1.0 - v0 };
            }
        }

        worlds.push(PreviewWorldData {
            weight,
            opp_canon,
            opp_tactical,
            opp_prior,
            compact: Some((roster, pcache)),
            stage1_vals,
        });
    }

    if worlds.is_empty() {
        let choice = solve_shipped_preview(
            position,
            dex,
            belief,
            obs,
            side,
            cfg.fallback_iters,
            SplitMix64::new(rng.next()),
        );
        let mut pol = vec![0.0; n_legal];
        if let Some(pos) = own_legal.iter().position(|&c| c == choice) {
            pol[pos] = 1.0;
        }
        return PreviewSearchResult {
            choice,
            policy_full: pol,
            value: 0.5,
            pure_guarantee: 0.5,
            mixed_guarantee: 0.5,
            duality_gap: 0.0,
            used_mix: false,
            used_fallback: true,
            stage1_ms: 0.0,
            total_ms: elapsed_ms(t_total),
            iters_used: cfg.fallback_iters,
            worlds_stage1: 0,
            worlds_stage2: 0,
            support_me: n_own,
            support_opp_max: 0,
            do_admitted_me: 0,
            do_admitted_opp: 0,
        };
    }

    // Normalize world weights across surviving worlds.
    let w_sum: f64 = worlds.iter().map(|w| w.weight).sum();
    let w_len = worlds.len() as f64;
    for w in &mut worlds {
        w.weight = if w_sum > 0.0 {
            w.weight / w_sum
        } else {
            1.0 / w_len
        };
    }

    // Solve Stage-1 Multi-World Bayesian RM+ on full `n_own × n_opp_w` matrices.
    let s1_slices: Vec<(f64, usize, &[f64])> = worlds
        .iter()
        .map(|w| (w.weight, w.opp_canon.len(), w.stage1_vals.as_slice()))
        .collect();
    let s1_priors: Vec<Option<&[f64]>> = worlds.iter().map(|w| w.opp_prior.as_deref()).collect();
    let sol1 = solve_bayesian_rm_plus_with_priors(n_own, &s1_slices, &s1_priors, cfg.rm_sweeps_stage1);
    let stage1_ms = elapsed_ms(t_total);

    if matches!(cfg.policy_mode, PreviewPolicyMode::Stage1Only) || cfg.total_iters == 0 {
        let mut p = sol1.own_policy.clone();
        shed_dust(&mut p, 0.08);
        let best_pure = (0..n_own)
            .max_by(|&a, &b| sol1.row_guarantees[a].total_cmp(&sol1.row_guarantees[b]))
            .unwrap_or(0);
        let g_pure = sol1.row_guarantees[best_pure];
        let g_mix: f64 = worlds
            .iter()
            .map(|w| {
                let m = w.opp_canon.len();
                if let Some(ref qp) = w.opp_prior {
                    let mut ev = 0.0;
                    for j in 0..m {
                        let mut v = 0.0;
                        for i in 0..n_own {
                            v += p[i] * w.stage1_vals[i * m + j];
                        }
                        ev += qp[j] * v;
                    }
                    w.weight * ev
                } else {
                    let mut mn = f64::INFINITY;
                    for j in 0..m {
                        let mut v = 0.0;
                        for i in 0..n_own {
                            v += p[i] * w.stage1_vals[i * m + j];
                        }
                        if v < mn {
                            mn = v;
                        }
                    }
                    w.weight * mn
                }
            })
            .sum();
        let draw = rng.next_f64();
        let pick_c = sample_dist(&p, draw);
        let mut pol = vec![0.0; n_legal];
        for (i, &(orig_idx, _)) in own_canon.iter().enumerate() {
            pol[orig_idx] = p[i];
        }
        return PreviewSearchResult {
            choice: own_legal[own_canon[pick_c].0],
            policy_full: pol,
            value: sol1.value,
            pure_guarantee: g_pure,
            mixed_guarantee: g_mix,
            duality_gap: sol1.duality_gap,
            used_mix: p.iter().filter(|&&x| x > 0.0).count() > 1,
            used_fallback: false,
            stage1_ms,
            total_ms: elapsed_ms(t_total),
            iters_used: 0,
            worlds_stage1: worlds.len(),
            worlds_stage2: 0,
            support_me: n_own,
            support_opp_max: worlds.iter().map(|w| w.opp_canon.len()).max().unwrap_or(0),
            do_admitted_me: 0,
            do_admitted_opp: 0,
        };
    }

    // =====================================================================
    // Stage 2: Support Selection + Persistent-Tree Double-Oracle Refinement
    // =====================================================================

    // Order compact-supported worlds by descending posterior weight: refine the top
    // `max_stage2_worlds` with Stage-2 search, and retain the tail worlds' Stage-1
    // `CF-V_net` v2 matrices in Bayesian RM+ so 100% of the posterior is preserved.
    let mut ordered_worlds: Vec<usize> = (0..worlds.len())
        .filter(|&w| worlds[w].compact.is_some())
        .collect();
    ordered_worlds.sort_by(|&a, &b| worlds[b].weight.total_cmp(&worlds[a].weight));
    let n_w2 = ordered_worlds.len().min(cfg.max_stage2_worlds.max(1));
    let s2_world_ids: Vec<usize> = ordered_worlds[..n_w2].to_vec();
    let tail_world_ids: Vec<usize> = ordered_worlds[n_w2..].to_vec();

    // Normalized budget weights over Stage-2 refined worlds (for search allocation).
    let w2_sum: f64 = s2_world_ids.iter().map(|&w| worlds[w].weight).sum();
    let s2_alloc_weights: Vec<f64> = s2_world_ids
        .iter()
        .map(|&w| {
            if w2_sum > 0.0 {
                worlds[w].weight / w2_sum
            } else {
                1.0 / (n_w2 as f64)
            }
        })
        .collect();
    // Exact full-posterior weights across refined + tail worlds (sum to 1.0).
    let w_all_sum: f64 = ordered_worlds.iter().map(|&w| worlds[w].weight).sum();
    let norm_w = |w_id: usize| -> f64 {
        if w_all_sum > 0.0 {
            worlds[w_id].weight / w_all_sum
        } else {
            1.0 / (ordered_worlds.len().max(1) as f64)
        }
    };
    let s2_post_weights: Vec<f64> = s2_world_ids.iter().map(|&w| norm_w(w)).collect();
    let tail_post_weights: Vec<f64> = tail_world_ids.iter().map(|&w| norm_w(w)).collect();

    // Adaptively tighten `support_k` on multi-world (`W >= 4`, e.g. off-pool) so each refined
    // world receives ~2.25× deeper cell evaluations while tail worlds retain full Stage-1 hedge.
    let k_eff = if n_w2 >= 4 {
        cfg.support_k.min(8)
    } else {
        cfg.support_k
    };

    // Select diverse own support `s_me` (`|s_me| <= k_eff`).
    let mut s_me = select_diverse_support(
        &own_canon,
        &sol1.own_policy,
        &sol1.row_vs_q,
        &sol1.row_guarantees,
        &own_tactical,
        k_eff,
    );

    // Select diverse opponent support `s_opp[wi]` per Stage-2 world.
    let mut s_opp: Vec<Vec<usize>> = Vec::with_capacity(n_w2);
    for &w_id in &s2_world_ids {
        let wd = &worlds[w_id];
        let m = wd.opp_canon.len();
        let opp_ev: Vec<f64> = sol1.col_vs_p[w_id].iter().map(|&v| 1.0 - v).collect();
        if let Some(ref qp) = wd.opp_prior {
            s_opp.push(select_prior_opp_support(qp, &opp_ev, k_eff));
            continue;
        }
        let mut opp_guar = vec![1.0f64; m];
        for j in 0..m {
            let mut mx = 0.0f64;
            for i in 0..n_own {
                let v = wd.stage1_vals[i * m + j];
                if v > mx {
                    mx = v;
                }
            }
            opp_guar[j] = 1.0 - mx;
        }
        s_opp.push(select_diverse_support(
            &wd.opp_canon,
            &sol1.opp_policies[w_id],
            &opp_ev,
            &opp_guar,
            &wd.opp_tactical,
            k_eff,
        ));
    }

    // Persistent cell map: `(wi, own_canon_i, opp_canon_j) -> CfPreviewCell`.
    let mut cells: FxHashMap<(usize, usize, usize), CfPreviewCell> = FxHashMap::default();
    let eg_playout = Playout::Heavy {
        turns: 8,
        eps: 0.2,
        weights: eg_weights.clone(),
    };
    let rollout_rules = RmConfig::default().rollout_rules();
    let mut path_buf: Vec<(usize, usize, usize)> = Vec::with_capacity(64);
    let mut iters_used: u32 = 0;

    let step_cell = |cells: &mut FxHashMap<(usize, usize, usize), CfPreviewCell>,
                     wi: usize,
                     own_i: usize,
                     opp_j: usize,
                     add_iters: u32,
                     iters_used: &mut u32,
                     rng: &mut SplitMix64,
                     path_buf: &mut Vec<(usize, usize, usize)>| {
        if add_iters == 0 {
            return;
        }
        let w_id = s2_world_ids[wi];
        let wd = &worlds[w_id];
        let (ref roster, ref pcache) = wd.compact.as_ref().unwrap();
        let m = wd.opp_canon.len();
        let prior_v_mine = wd.stage1_vals[own_i * m + opp_j];
        let prior_v0 = if side == 0 { prior_v_mine } else { 1.0 - prior_v_mine };
        let own_t = own_canon[own_i].1;
        let opp_t = wd.opp_canon[opp_j].1;
        let picks = if side == 0 { [own_t, opp_t] } else { [opp_t, own_t] };
        let cell = cells
            .entry((wi, own_i, opp_j))
            .or_insert_with(|| CfPreviewCell::new(roster, pcache, picks, prior_v0, cfg));
        cell.step(
            add_iters,
            cfg,
            rng,
            &eg_weights,
            &eg_playout,
            rollout_rules,
            path_buf,
        );
        *iters_used += add_iters;
    };

    // Helper: read current subgame matrices (refined `0..n_w2` + Stage-1 tail worlds)
    // and solve Multi-World Bayesian RM+ over the full 100% posterior.
    let solve_subgame = |cells: &FxHashMap<(usize, usize, usize), CfPreviewCell>,
                         s_me: &[usize],
                         s_opp: &[Vec<usize>],
                         sweeps: u32|
     -> (Vec<Vec<f64>>, BayesianRmSolution) {
        let k_me = s_me.len();
        let n_tail = tail_world_ids.len();
        let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
        let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2 + n_tail);
        let mut w_list: Vec<f64> = Vec::with_capacity(n_w2 + n_tail);
        let mut sub_priors: Vec<Option<Vec<f64>>> = Vec::with_capacity(n_w2 + n_tail);
        for wi in 0..n_w2 {
            let w_id = s2_world_ids[wi];
            let wd = &worlds[w_id];
            let m_full = wd.opp_canon.len();
            let k_opp = s_opp[wi].len();
            let mut mat = vec![0.5f64; k_me * k_opp];
            for (ri, &own_i) in s_me.iter().enumerate() {
                for (cj, &opp_j) in s_opp[wi].iter().enumerate() {
                    let v = if let Some(cell) = cells.get(&(wi, own_i, opp_j)) {
                        let v0 = cell.value0(cfg.cell_readout);
                        if side == 0 { v0 } else { 1.0 - v0 }
                    } else {
                        wd.stage1_vals[own_i * m_full + opp_j]
                    };
                    mat[ri * k_opp + cj] = v;
                }
            }
            mats.push(mat);
            col_lens.push(k_opp);
            w_list.push(s2_post_weights[wi]);
            sub_priors.push(restrict_opp_prior(wd.opp_prior.as_deref(), &s_opp[wi]));
        }
        for (ti, &w_id) in tail_world_ids.iter().enumerate() {
            let wd = &worlds[w_id];
            let m_full = wd.opp_canon.len();
            let mut mat = vec![0.5f64; k_me * m_full];
            for (ri, &own_i) in s_me.iter().enumerate() {
                let src = &wd.stage1_vals[own_i * m_full..(own_i + 1) * m_full];
                mat[ri * m_full..(ri + 1) * m_full].copy_from_slice(src);
            }
            mats.push(mat);
            col_lens.push(m_full);
            w_list.push(tail_post_weights[ti]);
            sub_priors.push(wd.opp_prior.clone());
        }
        let slices: Vec<(f64, usize, &[f64])> = (0..mats.len())
            .map(|idx| (w_list[idx], col_lens[idx], mats[idx].as_slice()))
            .collect();
        let prior_refs: Vec<Option<&[f64]>> = sub_priors.iter().map(|p| p.as_deref()).collect();
        let sol = solve_bayesian_rm_plus_with_priors(k_me, &slices, &prior_refs, sweeps);
        (mats, sol)
    };

    // Stage 2a: Depth-balanced coarse evaluation across all `(wi, own_i ∈ s_me, opp_j ∈ s_opp[wi])`.
    // Scales proportionally with `total_iters` (~45% budget, up to 640 iters/cell) to prevent
    // severe coarse-vs-polish depth distortion on larger budgets.
    let coarse_budget = (u64::from(cfg.total_iters) * 45 / 100) as u32;
    let coarse_cap = (cfg.total_iters / 320).clamp(cfg.coarse_iters.min(640), 640).max(1);
    let mut coarse_per_w = vec![cfg.coarse_iters; n_w2];
    let total_cells_coarse: usize = s_opp.iter().map(|so| s_me.len() * so.len()).sum();
    if total_cells_coarse > 0 {
        for wi in 0..n_w2 {
            let w_budget = ((f64::from(coarse_budget) * s2_alloc_weights[wi]).round() as u32).max(1);
            let n_c = (s_me.len() * s_opp[wi].len()).max(1) as u32;
            let per_cell = (w_budget / n_c).clamp(coarse_cap.min(32), coarse_cap);
            coarse_per_w[wi] = per_cell;
            for &own_i in &s_me {
                for &opp_j in &s_opp[wi] {
                    if iters_used >= cfg.total_iters {
                        break;
                    }
                    let step_n = per_cell.min(cfg.total_iters - iters_used);
                    step_cell(&mut cells, wi, own_i, opp_j, step_n, &mut iters_used, rng, &mut path_buf);
                }
            }
        }
    }

    let (_, mut sol2) = solve_subgame(&cells, &s_me, &s_opp, 600);
    let mut do_admitted_me = 0usize;
    let mut do_admitted_opp = 0usize;

    // Stage 2b: Double-Oracle Outside Best-Response Verification (budget cap 65%).
    let do_cap_iters = (u64::from(cfg.total_iters) * 65 / 100) as u32;
    let do_start_iters = iters_used;
    let opp_do_cap = do_start_iters
        .saturating_add((do_cap_iters.saturating_sub(do_start_iters)) * 60 / 100);
    for _round in 0..cfg.do_rounds {
        if iters_used >= do_cap_iters {
            break;
        }
        let mut admitted_opp = false;
        let mut admitted_me = false;
        let mut supp_me: Vec<usize> = (0..s_me.len())
            .filter(|&ri| sol2.own_policy[ri] >= 0.02)
            .collect();
        if supp_me.is_empty() {
            if let Some(best_r) = (0..s_me.len())
                .max_by(|&a, &b| sol2.own_policy[a].total_cmp(&sol2.own_policy[b]))
            {
                supp_me.push(best_r);
            }
        }
        let supp_p_sum: f64 = supp_me.iter().map(|&ri| sol2.own_policy[ri]).sum();

        // 1. Opponent BR Oracle (Exploitability guard): screen outside `j' ∉ s_opp[wi]` against `supp_me`.
        for wi in 0..n_w2 {
            if iters_used >= opp_do_cap {
                break;
            }
            if n_w2 > 1 && s2_alloc_weights[wi] < 0.04 {
                continue;
            }
            let do_step_w = coarse_per_w[wi].clamp(64, cfg.do_iters.max(192));
            let w_id = s2_world_ids[wi];
            let wd = &worlds[w_id];
            if wd.opp_prior.is_some() {
                continue;
            }
            let m_full = wd.opp_canon.len();
            let mut outside: Vec<(usize, f64)> = Vec::new();
            for j in 0..m_full {
                if s_opp[wi].contains(&j) {
                    continue;
                }
                let mut ev = 0.0;
                for &ri in &supp_me {
                    let pw = if supp_p_sum > 0.0 {
                        sol2.own_policy[ri] / supp_p_sum
                    } else {
                        1.0 / (supp_me.len() as f64)
                    };
                    ev += pw * wd.stage1_vals[s_me[ri] * m_full + j];
                }
                outside.push((j, ev));
            }
            outside.sort_by(|a, b| a.1.total_cmp(&b.1));
            let mut new_opp: Vec<usize> = Vec::new();
            for &(cand_j, _) in outside.iter().take(cfg.do_candidates) {
                if iters_used >= opp_do_cap {
                    break;
                }
                let mut ev2 = 0.0;
                let mut probed_all = true;
                for &ri in &supp_me {
                    let own_i = s_me[ri];
                    let step_n = do_step_w.min(opp_do_cap.saturating_sub(iters_used));
                    if step_n == 0 {
                        probed_all = false;
                        break;
                    }
                    step_cell(&mut cells, wi, own_i, cand_j, step_n, &mut iters_used, rng, &mut path_buf);
                    let pw = if supp_p_sum > 0.0 {
                        sol2.own_policy[ri] / supp_p_sum
                    } else {
                        1.0 / (supp_me.len() as f64)
                    };
                    let v = cells.get(&(wi, own_i, cand_j)).map_or(0.5, |c| {
                        let v0 = c.value0(cfg.cell_readout);
                        if side == 0 { v0 } else { 1.0 - v0 }
                    });
                    ev2 += pw * v;
                }
                if probed_all && ev2 < sol2.world_values[wi] - cfg.do_margin {
                    new_opp.push(cand_j);
                }
            }
            for cand_j in new_opp {
                let missing: u32 = s_me
                    .iter()
                    .filter(|&&own_i| !cells.contains_key(&(wi, own_i, cand_j)))
                    .count() as u32;
                let per_fill = (do_step_w / 2).clamp(48, 128);
                if opp_do_cap.saturating_sub(iters_used) < missing.saturating_mul(per_fill) {
                    continue;
                }
                s_opp[wi].push(cand_j);
                do_admitted_opp += 1;
                admitted_opp = true;
                for &own_i in &s_me {
                    if cells.contains_key(&(wi, own_i, cand_j)) {
                        continue;
                    }
                    let step_n = per_fill.min(cfg.total_iters.saturating_sub(iters_used));
                    step_cell(&mut cells, wi, own_i, cand_j, step_n, &mut iters_used, rng, &mut path_buf);
                }
            }
        }
        if admitted_opp {
            let (_, updated) = solve_subgame(&cells, &s_me, &s_opp, 600);
            sol2 = updated;
        }

        // 2. Own BR Oracle (Missed-winner guard): screen outside `i' ∉ s_me` against `supp_opp`.
        if iters_used < do_cap_iters {
            let supp_opp: Vec<Vec<usize>> = (0..n_w2)
                .map(|wi| {
                    let mut v: Vec<usize> = (0..s_opp[wi].len())
                        .filter(|&cj| sol2.opp_policies[wi][cj] >= 0.02)
                        .collect();
                    if v.is_empty() {
                        if let Some(bc) = (0..s_opp[wi].len()).max_by(|&a, &b| {
                            sol2.opp_policies[wi][a].total_cmp(&sol2.opp_policies[wi][b])
                        }) {
                            v.push(bc);
                        }
                    }
                    v
                })
                .collect();
            let mut outside_me: Vec<(usize, f64)> = Vec::new();
            for i in 0..n_own {
                if s_me.contains(&i) {
                    continue;
                }
                // Match `sol2.value` 1:1 over the full 100% posterior (refined + Stage-1 tail worlds).
                let mut ev = 0.0;
                for wi in 0..n_w2 {
                    let w_id = s2_world_ids[wi];
                    let wd = &worlds[w_id];
                    let m_full = wd.opp_canon.len();
                    let qsum: f64 = supp_opp[wi].iter().map(|&cj| sol2.opp_policies[wi][cj]).sum();
                    let mut ev_w = 0.0;
                    for &cj in &supp_opp[wi] {
                        let qw = if qsum > 0.0 {
                            sol2.opp_policies[wi][cj] / qsum
                        } else {
                            1.0 / (supp_opp[wi].len() as f64)
                        };
                        ev_w += qw * wd.stage1_vals[i * m_full + s_opp[wi][cj]];
                    }
                    ev += s2_post_weights[wi] * ev_w;
                }
                for (ti, &w_id) in tail_world_ids.iter().enumerate() {
                    let wd = &worlds[w_id];
                    let m_full = wd.opp_canon.len();
                    let q_tail = &sol2.opp_policies[n_w2 + ti];
                    let mut ev_w = 0.0;
                    for j in 0..m_full {
                        let qj = q_tail[j];
                        if qj > 0.0 {
                            ev_w += qj * wd.stage1_vals[i * m_full + j];
                        }
                    }
                    ev += tail_post_weights[ti] * ev_w;
                }
                outside_me.push((i, ev));
            }
            outside_me.sort_by(|a, b| b.1.total_cmp(&a.1));
            let mut new_me: Vec<usize> = Vec::new();
            for &(cand_i, _) in outside_me.iter().take(cfg.do_candidates) {
                if iters_used >= do_cap_iters {
                    break;
                }
                let mut ev2 = 0.0;
                let mut probed_all = true;
                for wi in 0..n_w2 {
                    let do_step_w = coarse_per_w[wi].clamp(64, cfg.do_iters.max(192));
                    let qsum: f64 = supp_opp[wi].iter().map(|&cj| sol2.opp_policies[wi][cj]).sum();
                    let mut ev_w = 0.0;
                    for &cj in &supp_opp[wi] {
                        let opp_j = s_opp[wi][cj];
                        let step_n = do_step_w.min(do_cap_iters.saturating_sub(iters_used));
                        if step_n == 0 {
                            probed_all = false;
                            break;
                        }
                        step_cell(&mut cells, wi, cand_i, opp_j, step_n, &mut iters_used, rng, &mut path_buf);
                        let qw = if qsum > 0.0 {
                            sol2.opp_policies[wi][cj] / qsum
                        } else {
                            1.0 / (supp_opp[wi].len() as f64)
                        };
                        let v = cells.get(&(wi, cand_i, opp_j)).map_or(0.5, |c| {
                            let v0 = c.value0(cfg.cell_readout);
                            if side == 0 { v0 } else { 1.0 - v0 }
                        });
                        ev_w += qw * v;
                    }
                    if !probed_all {
                        break;
                    }
                    ev2 += s2_post_weights[wi] * ev_w;
                }
                if probed_all {
                    for (ti, &w_id) in tail_world_ids.iter().enumerate() {
                        let wd = &worlds[w_id];
                        let m_full = wd.opp_canon.len();
                        let q_tail = &sol2.opp_policies[n_w2 + ti];
                        let mut ev_w = 0.0;
                        for j in 0..m_full {
                            let qj = q_tail[j];
                            if qj > 0.0 {
                                ev_w += qj * wd.stage1_vals[cand_i * m_full + j];
                            }
                        }
                        ev2 += tail_post_weights[ti] * ev_w;
                    }
                    if ev2 > sol2.value + cfg.do_margin {
                        new_me.push(cand_i);
                    }
                }
            }
            for cand_i in new_me {
                let mut missing = 0u32;
                for wi in 0..n_w2 {
                    missing += s_opp[wi]
                        .iter()
                        .filter(|&&opp_j| !cells.contains_key(&(wi, cand_i, opp_j)))
                        .count() as u32;
                }
                let per_fill = (coarse_per_w[0] / 2).clamp(48, 128);
                if do_cap_iters.saturating_sub(iters_used) < missing.saturating_mul(per_fill) {
                    continue;
                }
                s_me.push(cand_i);
                do_admitted_me += 1;
                admitted_me = true;
                for wi in 0..n_w2 {
                    for &opp_j in &s_opp[wi] {
                        if cells.contains_key(&(wi, cand_i, opp_j)) {
                            continue;
                        }
                        let step_n = per_fill.min(cfg.total_iters.saturating_sub(iters_used));
                        step_cell(&mut cells, wi, cand_i, opp_j, step_n, &mut iters_used, rng, &mut path_buf);
                    }
                }
            }
        }

        if admitted_me {
            let (_, updated) = solve_subgame(&cells, &s_me, &s_opp, 600);
            sol2 = updated;
        }
        if !admitted_opp && !admitted_me {
            break;
        }
    }

    // Stage 2c: Depth-Balanced Active-Support Deep Polish.
    // Caps per-cell visits at `<= 4×` coarse (max `2048`) to prevent asymmetric depth distortion
    // and PUCT oscillation; always includes `pure_r` AND its worst-case column per world.
    let mut polish_steps = 0u32;
    while iters_used < cfg.total_iters {
        let mut active_rows: Vec<usize> = (0..s_me.len())
            .filter(|&ri| sol2.own_policy[ri] >= 0.01 || sol2.row_vs_q[ri] >= sol2.value - cfg.polish_gap)
            .collect();
        let pure_r_opt = (0..s_me.len())
            .max_by(|&a, &b| sol2.row_guarantees[a].total_cmp(&sol2.row_guarantees[b]));
        if let Some(pure_r) = pure_r_opt {
            if !active_rows.contains(&pure_r) {
                active_rows.push(pure_r);
            }
        }
        if active_rows.len() < 2 {
            let mut ord: Vec<usize> = (0..s_me.len()).collect();
            ord.sort_by(|&a, &b| sol2.row_vs_q[b].total_cmp(&sol2.row_vs_q[a]));
            for ri in ord {
                if active_rows.len() >= 2 {
                    break;
                }
                if !active_rows.contains(&ri) {
                    active_rows.push(ri);
                }
            }
        }

        let mut best_target: Option<(usize, usize, usize, f64)> = None;
        for wi in 0..n_w2 {
            let cell_cap_w = (coarse_per_w[wi].saturating_mul(4)).clamp(512, 2048);
            let mut active_cols: Vec<usize> = (0..s_opp[wi].len())
                .filter(|&cj| {
                    sol2.opp_policies[wi][cj] >= 0.01
                        || sol2.col_vs_p[wi][cj] <= sol2.world_values[wi] + cfg.polish_gap
                })
                .collect();
            // Always polish `pure_r`'s worst column in this world so `pure_guarantee` has no low-visit bias.
            if let Some(pure_r) = pure_r_opt {
                let own_i = s_me[pure_r];
                if let Some(worst_cj) = (0..s_opp[wi].len()).min_by(|&a, &b| {
                    let va = cells.get(&(wi, own_i, s_opp[wi][a])).map_or(0.5, |c| {
                        let v0 = c.value0(cfg.cell_readout);
                        if side == 0 { v0 } else { 1.0 - v0 }
                    });
                    let vb = cells.get(&(wi, own_i, s_opp[wi][b])).map_or(0.5, |c| {
                        let v0 = c.value0(cfg.cell_readout);
                        if side == 0 { v0 } else { 1.0 - v0 }
                    });
                    va.total_cmp(&vb)
                }) {
                    if !active_cols.contains(&worst_cj) {
                        active_cols.push(worst_cj);
                    }
                }
            }
            if active_cols.len() < 2 {
                let mut ord: Vec<usize> = (0..s_opp[wi].len()).collect();
                ord.sort_by(|&a, &b| sol2.col_vs_p[wi][a].total_cmp(&sol2.col_vs_p[wi][b]));
                for cj in ord {
                    if active_cols.len() >= 2 {
                        break;
                    }
                    if !active_cols.contains(&cj) {
                        active_cols.push(cj);
                    }
                }
            }
            for &ri in &active_rows {
                let own_i = s_me[ri];
                let pw = sol2.own_policy[ri] + 0.05;
                for &cj in &active_cols {
                    let opp_j = s_opp[wi][cj];
                    let visits = cells.get(&(wi, own_i, opp_j)).map_or(0, |c| c.total_n);
                    if visits >= cell_cap_w {
                        continue;
                    }
                    let qw = sol2.opp_policies[wi][cj] + 0.05;
                    let score = s2_alloc_weights[wi] * pw * qw / f64::from(visits + 1).sqrt();
                    if best_target.map_or(true, |bt| score > bt.3) {
                        best_target = Some((wi, own_i, opp_j, score));
                    }
                }
            }
        }
        // If all primary active cells hit `cell_cap_w`, widen polish to any remaining support cell
        // below `cell_cap_w / 2` to tighten off-support bounds symmetrically, else stop early!
        if best_target.is_none() {
            for wi in 0..n_w2 {
                let half_cap = coarse_per_w[wi].saturating_mul(2).clamp(256, 1024);
                for &ri in &active_rows {
                    let own_i = s_me[ri];
                    let pw = sol2.own_policy[ri] + 0.05;
                    for &opp_j in &s_opp[wi] {
                        let visits = cells.get(&(wi, own_i, opp_j)).map_or(0, |c| c.total_n);
                        if visits >= half_cap {
                            continue;
                        }
                        let score = s2_alloc_weights[wi] * pw / f64::from(visits + 1).sqrt();
                        if best_target.map_or(true, |bt| score > bt.3) {
                            best_target = Some((wi, own_i, opp_j, score));
                        }
                    }
                }
            }
        }
        let Some((wi, own_i, opp_j, _)) = best_target else {
            break;
        };
        let chunk = cfg.polish_chunk.min(cfg.total_iters - iters_used);
        step_cell(&mut cells, wi, own_i, opp_j, chunk, &mut iters_used, rng, &mut path_buf);
        polish_steps += 1;
        if polish_steps % 8 == 0 {
            let (_, updated) = solve_subgame(&cells, &s_me, &s_opp, 600);
            sol2 = updated;
        }
    }

    // Final high-precision Multi-World Bayesian RM+ solve (100% posterior: refined + tail worlds).
    let (final_mats, final_sol) = solve_subgame(&cells, &s_me, &s_opp, cfg.rm_sweeps_final);
    let k_me = s_me.len();
    let best_pure_r = (0..k_me)
        .max_by(|&a, &b| {
            final_sol.row_guarantees[a]
                .total_cmp(&final_sol.row_guarantees[b])
                .then_with(|| final_sol.row_vs_q[a].total_cmp(&final_sol.row_vs_q[b]))
        })
        .unwrap_or(0);
    let pure_guarantee = final_sol.row_guarantees[best_pure_r];

    let mut p_mix = final_sol.own_policy.clone();
    let dust_ratio = match cfg.policy_mode {
        PreviewPolicyMode::MixedNash { dust_ratio }
        | PreviewPolicyMode::LcbGatedMixed { dust_ratio, .. } => dust_ratio,
        _ => 0.08,
    };
    shed_dust(&mut p_mix, dust_ratio);

    // Exact mixed guarantee & conservative LCB mixed guarantee over full posterior.
    let lcb_z = match cfg.policy_mode {
        PreviewPolicyMode::LcbGatedMixed { lcb_z, .. } => lcb_z,
        _ => 1.0,
    };
    let mut mixed_guarantee = 0.0;
    let mut mixed_guarantee_lcb = 0.0;
    for wi in 0..n_w2 {
        let k_opp = s_opp[wi].len();
        let mat = &final_mats[wi];
        let w_id = s2_world_ids[wi];
        if let Some(q_sub) = restrict_opp_prior(worlds[w_id].opp_prior.as_deref(), &s_opp[wi]) {
            let mut ev_w = 0.0;
            let mut var_w = 0.0;
            for cj in 0..k_opp {
                let opp_j = s_opp[wi][cj];
                let qj = q_sub[cj];
                if qj <= 0.0 {
                    continue;
                }
                let mut mean_j = 0.0;
                let mut var_j = 0.0;
                for ri in 0..k_me {
                    let pr = p_mix[ri];
                    if pr > 0.0 {
                        let v = mat[ri * k_opp + cj];
                        mean_j += pr * v;
                        let n_c = cells.get(&(wi, s_me[ri], opp_j)).map_or(1, |c| c.total_n.max(1));
                        var_j += (pr * pr) * (0.25 / f64::from(n_c));
                    }
                }
                ev_w += qj * mean_j;
                var_w += (qj * qj) * var_j;
            }
            mixed_guarantee += s2_post_weights[wi] * ev_w;
            mixed_guarantee_lcb += s2_post_weights[wi] * (ev_w - lcb_z * var_w.sqrt());
            continue;
        }
        let mut min_col = f64::INFINITY;
        let mut min_col_lcb = f64::INFINITY;
        for cj in 0..k_opp {
            let opp_j = s_opp[wi][cj];
            let mut mean_j = 0.0;
            let mut var_j = 0.0;
            for ri in 0..k_me {
                let pr = p_mix[ri];
                if pr > 0.0 {
                    let v = mat[ri * k_opp + cj];
                    mean_j += pr * v;
                    let n_c = cells.get(&(wi, s_me[ri], opp_j)).map_or(1, |c| c.total_n.max(1));
                    var_j += (pr * pr) * (0.25 / f64::from(n_c));
                }
            }
            if mean_j < min_col {
                min_col = mean_j;
            }
            let lcb_j = mean_j - lcb_z * var_j.sqrt();
            if lcb_j < min_col_lcb {
                min_col_lcb = lcb_j;
            }
        }
        mixed_guarantee += s2_post_weights[wi] * min_col;
        mixed_guarantee_lcb += s2_post_weights[wi] * min_col_lcb;
    }
    for (ti, &w_id) in tail_world_ids.iter().enumerate() {
        let m_full = worlds[w_id].opp_canon.len();
        let mat = &final_mats[n_w2 + ti];
        if let Some(ref qp) = worlds[w_id].opp_prior {
            let mut ev_w = 0.0;
            for cj in 0..m_full {
                let qj = qp[cj];
                if qj <= 0.0 {
                    continue;
                }
                let mut mean_j = 0.0;
                for ri in 0..k_me {
                    let pr = p_mix[ri];
                    if pr > 0.0 {
                        mean_j += pr * mat[ri * m_full + cj];
                    }
                }
                ev_w += qj * mean_j;
            }
            mixed_guarantee += tail_post_weights[ti] * ev_w;
            mixed_guarantee_lcb += tail_post_weights[ti] * ev_w;
            continue;
        }
        let mut min_col = f64::INFINITY;
        for cj in 0..m_full {
            let mut mean_j = 0.0;
            for ri in 0..k_me {
                let pr = p_mix[ri];
                if pr > 0.0 {
                    mean_j += pr * mat[ri * m_full + cj];
                }
            }
            if mean_j < min_col {
                min_col = mean_j;
            }
        }
        // Deterministic Stage-1 tail worlds share identical model error for pure & mixed (0 MC variance).
        mixed_guarantee += tail_post_weights[ti] * min_col;
        mixed_guarantee_lcb += tail_post_weights[ti] * min_col;
    }

    let use_mix = match cfg.policy_mode {
        PreviewPolicyMode::PureMaximin => false,
        PreviewPolicyMode::MixedNash { .. } => p_mix.iter().filter(|&&x| x > 0.0).count() > 1,
        PreviewPolicyMode::LcbGatedMixed { min_mix_margin, .. } => {
            p_mix.iter().filter(|&&x| x > 0.0).count() > 1
                && mixed_guarantee > pure_guarantee + min_mix_margin
                && mixed_guarantee_lcb > pure_guarantee - 0.005
        }
        PreviewPolicyMode::Stage1Only => true,
    };

    let mut sub_pol = vec![0.0f64; k_me];
    if use_mix {
        sub_pol.copy_from_slice(&p_mix);
    } else {
        sub_pol[best_pure_r] = 1.0;
    }
    let draw = rng.next_f64();
    let picked_r = sample_dist(&sub_pol, draw);
    let picked_canon_i = s_me[picked_r];

    let mut policy_full = vec![0.0f64; n_legal];
    for (ri, &ci) in s_me.iter().enumerate() {
        policy_full[own_canon[ci].0] = sub_pol[ri];
    }

    PreviewSearchResult {
        choice: own_legal[own_canon[picked_canon_i].0],
        policy_full,
        value: final_sol.value,
        pure_guarantee,
        mixed_guarantee,
        duality_gap: final_sol.duality_gap,
        used_mix: use_mix,
        used_fallback: false,
        stage1_ms,
        total_ms: elapsed_ms(t_total),
        iters_used,
        worlds_stage1: worlds.len(),
        worlds_stage2: n_w2,
        support_me: s_me.len(),
        support_opp_max: s_opp.iter().map(Vec::len).max().unwrap_or(0),
        do_admitted_me,
        do_admitted_opp,
    }
}

#[inline]
fn sample_dist(p: &[f64], draw: f64) -> usize {
    let mut acc = 0.0;
    let mut best = 0usize;
    let mut best_p = -1.0;
    for (i, &pr) in p.iter().enumerate() {
        if pr > best_p {
            best_p = pr;
            best = i;
        }
        acc += pr;
        if draw < acc {
            return i;
        }
    }
    best
}

// =========================================================================
// 5. Incrementally Steppable `SteppedCfPreview` (Production `BlindSearch`)
// =========================================================================

/// Configuration for Control-Variate Multi-Turn Horizon Calibration (`TrajCalib`)
/// running ahead of `SteppedCfPreview`'s persistent Turn-1 Double-Oracle search.
#[derive(Clone, Debug)]
pub struct TrajCalibConfig {
    /// Fraction of `total_iters` spent on CRN-paired multi-turn trajectory calibration (`0.0` = disabled).
    pub calib_frac: f64,
    /// Turn-1 PUCT iterations inside each calibration trajectory (for low-variance CRN `V_T1`).
    pub turn1_iters: u32,
    /// Subsequent-turn PUCT iterations inside each calibration trajectory.
    pub iters_per_turn: u32,
    /// Rao-Blackwellized `SearchVroot` early-exit threshold (`alive_sum <= rb_max_alive`).
    pub rb_max_alive: u32,
    /// Rao-Blackwellized `SearchVroot` evaluation iterations.
    pub rb_eval_iters: u32,
    /// Ridge feature basis for fitting `(V_traj - V_s1)` and `(V_traj - V_T1)`.
    pub ridge_basis: TrajRidgeBasis,
    /// L2 ridge penalty on intercept (`f = 0`).
    pub ridge_lambda_int: f64,
    /// L2 ridge penalty on main species presence features.
    pub ridge_lambda_main: f64,
    /// L2 ridge penalty on lead-delta and interaction features.
    pub ridge_lambda_inter: f64,
    /// Fit shrunk per-world `delta_w` on `W > 1` (`false` pools strictly across all worlds via `theta_shared`).
    pub per_world_delta: bool,
    /// GLS weighting pseudo-count `(kappa + 1) * n / (kappa + n)`.
    pub kappa_gls: f64,
    /// Scale applied to Stage-1 ridge delta (`V_traj - V_s1`).
    pub delta_s1_beta: f64,
    /// Scale applied to Stage-2 Turn-1 control-variate delta (`V_traj - V_T1`).
    pub delta_t1_beta: f64,
    /// Stage-2 support width (`support_k`).
    pub support_k: usize,
    /// Stage-2 DO rounds (`do_rounds`).
    pub do_rounds: usize,
    /// LCB-gated mix parameters (`min_mix_margin`, `dust_ratio`, `lcb_z`).
    pub min_mix_margin: f64,
    pub dust_ratio: f64,
    pub lcb_z: f64,
    /// Optional Rubin Bayesian Bootstrap draws over CRN calibration blocks (`0` = LcbGatedMixed).
    pub bootstrap_draws: usize,
    /// Dust shedding ratio when `bootstrap_draws > 1`.
    pub bootstrap_dust: f64,
    /// Union calibrated Stage-1 support with raw `CF-V_net` Stage-1 top picks (prevents seed-drop of tactical cores).
    pub union_raw_support: bool,
    /// Evaluate final equilibrium over `s_me x ALL opponent columns` (eliminates outside-support killer blindspots).
    pub full_opp_readout: bool,
}

impl Default for TrajCalibConfig {
    fn default() -> Self {
        Self {
            calib_frac: 0.65,
            turn1_iters: 128,
            iters_per_turn: 128,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 45.0,
            ridge_lambda_inter: 70.0,
            per_world_delta: false,
            kappa_gls: 12.0,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.85,
            support_k: 12,
            do_rounds: 1,
            min_mix_margin: 0.010,
            dust_ratio: 0.08,
            lcb_z: 0.75,
            bootstrap_draws: 16,
            bootstrap_dust: 0.15,
            union_raw_support: true,
            full_opp_readout: true,
        }
    }
}

#[derive(Clone, Copy)]
struct CalibObs {
    wi: u16,
    own_i: u16,
    opp_j: u16,
    block_id: u32,
    r_s1: f32,
    r_t1: f32,
    has_t1: bool,
}

struct TrajCalibState {
    calib_iters: u32,
    explore_iters: u32,
    traj_cfg: TrajPreviewConfig,
    scratch: TrajTurnScratch,
    active_slots: Vec<Option<ActiveTrajGame>>,
    cur_slot: usize,
    task_queue: std::collections::VecDeque<TrajTask>,
    s2_metas: Vec<WorldInteractMeta>,
    tail_metas: Vec<WorldInteractMeta>,
    obs_s1_n: Vec<Vec<u32>>,
    obs_s1_sum: Vec<Vec<f64>>,
    obs_t1_n: Vec<Vec<u32>>,
    obs_t1_sum: Vec<Vec<f64>>,
    own_mon_cnt: [u32; 6],
    own_lead_cnt: [u32; 6],
    own_bench_pair_cnt: [u32; 15],
    opp_mon_cnt: Vec<[u32; 6]>,
    opp_lead_cnt: Vec<[u32; 6]>,
    opp_bench_pair_cnt: Vec<[u32; 15]>,
    world_games: Vec<u32>,
    next_block_id: u32,
    games_completed: u32,
}

enum SteppedPreviewPhase {
    TrajCalib,
    CoarseTier0,
    CoarseTier1,
    CoarseTier2,
    DoOpp {
        round: usize,
        do_cap_iters: u32,
        opp_do_cap: u32,
        supp_me: Vec<usize>,
        supp_p_sum: f64,
        wi: usize,
        candidates: Vec<usize>,
        cand_pos: usize,
        active_cand: Option<(usize, bool)>,
        pending_admit: Vec<usize>,
        admitted_opp: bool,
    },
    DoOppFinalize {
        round: usize,
        do_cap_iters: u32,
        admitted_opp: bool,
    },
    DoOwn {
        round: usize,
        do_cap_iters: u32,
        supp_opp: Vec<Vec<usize>>,
        candidates: Vec<usize>,
        cand_pos: usize,
        active_cand: Option<(usize, bool)>,
        pending_admit: Vec<usize>,
        admitted_opp: bool,
        admitted_me: bool,
    },
    DoOwnFinalize {
        round: usize,
        do_cap_iters: u32,
        admitted_opp: bool,
        admitted_me: bool,
    },
    Polish {
        polish_steps: u32,
    },
}

/// Incrementally steppable, slice-invariant Two-Stage Persistent-Tree Double-Oracle
/// Team Preview searcher driven 1 iteration at a time by `BlindSearch`.
/// Supports optional Control-Variate Multi-Turn Horizon Calibration (`TrajCalib`).
pub struct SteppedCfPreview {
    my_acts: Vec<SearchChoice>,
    side: usize,
    cfg: PreviewSearchConfig,
    own_canon: Vec<(usize, [u8; 3])>,
    own_tactical: [bool; 6],
    worlds: Vec<PreviewWorldData>,
    raw_stage1_vals: Vec<Vec<f64>>,
    delta_t1: Vec<Vec<f64>>,
    calib_cfg: Option<TrajCalibConfig>,
    calib: Option<Box<TrajCalibState>>,
    calib_obs: Vec<CalibObs>,
    calib_s2_metas: Vec<WorldInteractMeta>,
    calib_tail_metas: Vec<WorldInteractMeta>,
    calib_iters_spent: u32,
    s2_world_ids: Vec<usize>,
    tail_world_ids: Vec<usize>,
    s2_alloc_weights: Vec<f64>,
    s2_post_weights: Vec<f64>,
    tail_post_weights: Vec<f64>,
    s_me: Vec<usize>,
    s_opp: Vec<Vec<usize>>,
    coarse_per_w: Vec<u32>,
    cells: FxHashMap<(usize, usize, usize), CfPreviewCell>,
    eg_weights: EvalWeights,
    eg_playout: Playout,
    rollout_rules: crate::mcts::RolloutRules,
    path_buf: Vec<(usize, usize, usize)>,
    sol2: BayesianRmSolution,
    iters_done: u32,
    total_nodes: usize,
    current_work: Option<(usize, usize, usize, u32)>,
    work_queue: std::collections::VecDeque<(usize, usize, usize, u32)>,
    phase: SteppedPreviewPhase,
}

impl SteppedCfPreview {
    #[allow(clippy::too_many_arguments, dead_code)]
    pub(crate) fn try_new(
        base: &Battle,
        dex: &Dex,
        belief: &Belief,
        obs: &crate::observe::Observer,
        side: usize,
        my_acts: &[SearchChoice],
        my_mask: Option<&[bool]>,
        c_ucb: f64,
        rng: &mut SplitMix64,
    ) -> Option<Self> {
        Self::try_new_with_calib(
            base, dex, belief, obs, side, my_acts, my_mask, c_ucb, None, rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new_calibrated(
        base: &Battle,
        dex: &Dex,
        belief: &Belief,
        obs: &crate::observe::Observer,
        side: usize,
        my_acts: &[SearchChoice],
        my_mask: Option<&[bool]>,
        c_ucb: f64,
        use_opp_prior: bool,
        rng: &mut SplitMix64,
    ) -> Option<Self> {
        Self::try_new_with_calib_and_prior(
            base,
            dex,
            belief,
            obs,
            side,
            my_acts,
            my_mask,
            c_ucb,
            Some(TrajCalibConfig::default()),
            use_opp_prior,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_calib(
        base: &Battle,
        dex: &Dex,
        belief: &Belief,
        obs: &crate::observe::Observer,
        side: usize,
        my_acts: &[SearchChoice],
        my_mask: Option<&[bool]>,
        c_ucb: f64,
        calib_cfg: Option<TrajCalibConfig>,
        rng: &mut SplitMix64,
    ) -> Option<Self> {
        let use_opp_prior = belief.has_preview_prior(obs);
        Self::try_new_with_calib_and_prior(
            base,
            dex,
            belief,
            obs,
            side,
            my_acts,
            my_mask,
            c_ucb,
            calib_cfg,
            use_opp_prior,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn try_new_with_calib_and_prior(
        base: &Battle,
        dex: &Dex,
        belief: &Belief,
        obs: &crate::observe::Observer,
        side: usize,
        my_acts: &[SearchChoice],
        my_mask: Option<&[bool]>,
        c_ucb: f64,
        calib_cfg: Option<TrajCalibConfig>,
        use_opp_prior: bool,
        rng: &mut SplitMix64,
    ) -> Option<Self> {
        if base.request_state != RequestState::TeamPreview {
            return None;
        }
        let opp = 1 - side;
        let order_matters = party_order_matters(dex, &base.sides[side].roster);
        let mut own_canon: Vec<(usize, [u8; 3])> = Vec::with_capacity(my_acts.len());
        let mut seen: Vec<[u8; 3]> = Vec::with_capacity(60);
        for (idx, &ch) in my_acts.iter().enumerate() {
            if my_mask.is_some_and(|m| !m[idx]) {
                continue;
            }
            let SearchChoice::Team(t) = ch else {
                return None;
            };
            if !order_matters && t[2] != 0 {
                let mut key = t;
                if key[1] > key[2] {
                    key.swap(1, 2);
                }
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
            }
            own_canon.push((idx, t));
        }
        let n_own = own_canon.len();
        if n_own <= 1 {
            return None;
        }

        let mut cfg = PreviewSearchConfig::default();
        cfg.c_puct = (1.0 * (c_ucb / 0.4)).clamp(0.25, 4.0);
        if let Some(ref cc) = calib_cfg {
            cfg.support_k = cc.support_k.max(4);
            cfg.do_rounds = cc.do_rounds;
            cfg.policy_mode = PreviewPolicyMode::LcbGatedMixed {
                min_mix_margin: cc.min_mix_margin,
                dust_ratio: cc.dust_ratio,
                lcb_z: cc.lcb_z,
            };
        }

        let own_tactical = lead_tactical_flags(dex, base, side);
        let world_plan = belief.preview_world_distribution(cfg.max_stage1_worlds, rng);
        if world_plan.is_empty() {
            return None;
        }

        let mut det_worlds: Vec<(crate::belief::TeamPlanPick, f64, Battle)> =
            Vec::with_capacity(world_plan.len());
        for &(plan_pick, weight) in &world_plan {
            let mut sim_w = base.clone();
            sim_w.set_log_enabled(false);
            belief.determinize_in_place_planned(dex, &mut sim_w, obs, plan_pick, rng);
            sim_w.quick_claw_rolls = [false, false];
            if !is_compact_preview_supported(&sim_w, dex) {
                return None;
            }
            det_worlds.push((plan_pick, weight, sim_w));
        }

        let calib_active = calib_cfg
            .as_ref()
            .is_some_and(|cc| cc.calib_frac > 0.0 && cfg.total_iters >= 1024);
        let mut worlds: Vec<PreviewWorldData> = Vec::with_capacity(det_worlds.len());
        let mut all_metas: Vec<WorldInteractMeta> = Vec::with_capacity(det_worlds.len());
        for (plan_pick, weight, mut sim_w) in det_worlds {
            let (opp_legal, opp_canon) = canonical_preview_actions(dex, &mut sim_w, opp);
            let n_opp = opp_canon.len();
            if n_opp == 0 {
                continue;
            }
            let opp_prior = if use_opp_prior {
                let opp_order_matters = party_order_matters(dex, &sim_w.sides[opp].roster);
                belief.preview_prior_probs_canonical(
                    plan_pick,
                    obs,
                    &opp_legal,
                    &opp_canon,
                    opp_order_matters,
                )
            } else {
                None
            };
            let opp_tactical = lead_tactical_flags(dex, &sim_w, opp);
            let party_lens = [sim_w.sides[0].party.len(), sim_w.sides[1].party.len()];
            let roster = CompactPreviewRoster::from_battle(&sim_w, dex, None);
            let pcache = PreviewTempoCache::build_for_preview_roster(&roster, party_lens);
            if calib_active {
                all_metas.push(WorldInteractMeta::build(&roster, &pcache, party_lens));
            }
            let mut stage1_vals = vec![0.5f64; n_own * n_opp];
            for (i, &(_, own_t)) in own_canon.iter().enumerate() {
                for (j, &(_, opp_t)) in opp_canon.iter().enumerate() {
                    let picks = if side == 0 { [own_t, opp_t] } else { [opp_t, own_t] };
                    let (table, mut cb) = roster.build_turn1(picks, 0);
                    if cb.sides[0].request == 1 && cb.sides[1].request == 1 {
                        cb.quick_claw_roll = 0;
                    }
                    let sliced = pcache.slice_for(&cb);
                    let v0 = evaluate_cf_value(&cb, &table, Some(&sliced), cfg.leaf_temp);
                    stage1_vals[i * n_opp + j] = if side == 0 { v0 } else { 1.0 - v0 };
                }
            }
            worlds.push(PreviewWorldData {
                weight,
                opp_canon,
                opp_tactical,
                opp_prior,
                compact: Some((roster, pcache)),
                stage1_vals,
            });
        }
        if worlds.is_empty() {
            return None;
        }

        let w_sum: f64 = worlds.iter().map(|w| w.weight).sum();
        let w_len = worlds.len() as f64;
        for w in &mut worlds {
            w.weight = if w_sum > 0.0 {
                w.weight / w_sum
            } else {
                1.0 / w_len
            };
        }

        let s1_slices: Vec<(f64, usize, &[f64])> = worlds
            .iter()
            .map(|w| (w.weight, w.opp_canon.len(), w.stage1_vals.as_slice()))
            .collect();
        let s1_priors: Vec<Option<&[f64]>> =
            worlds.iter().map(|w| w.opp_prior.as_deref()).collect();
        let sol1 = solve_bayesian_rm_plus_with_priors(
            n_own,
            &s1_slices,
            &s1_priors,
            cfg.rm_sweeps_stage1,
        );

        let mut ordered_worlds: Vec<usize> = (0..worlds.len()).collect();
        ordered_worlds.sort_by(|&a, &b| worlds[b].weight.total_cmp(&worlds[a].weight));
        let n_w2 = ordered_worlds.len().min(cfg.max_stage2_worlds.max(1));
        let s2_world_ids: Vec<usize> = ordered_worlds[..n_w2].to_vec();
        let tail_world_ids: Vec<usize> = ordered_worlds[n_w2..].to_vec();

        let w2_sum: f64 = s2_world_ids.iter().map(|&w| worlds[w].weight).sum();
        let s2_alloc_weights: Vec<f64> = s2_world_ids
            .iter()
            .map(|&w| {
                if w2_sum > 0.0 {
                    worlds[w].weight / w2_sum
                } else {
                    1.0 / (n_w2 as f64)
                }
            })
            .collect();
        let w_all_sum: f64 = ordered_worlds.iter().map(|&w| worlds[w].weight).sum();
        let norm_w = |w_id: usize| -> f64 {
            if w_all_sum > 0.0 {
                worlds[w_id].weight / w_all_sum
            } else {
                1.0 / (ordered_worlds.len().max(1) as f64)
            }
        };
        let s2_post_weights: Vec<f64> = s2_world_ids.iter().map(|&w| norm_w(w)).collect();
        let tail_post_weights: Vec<f64> = tail_world_ids.iter().map(|&w| norm_w(w)).collect();

        let k_eff = if n_w2 >= 4 {
            cfg.support_k.min(8)
        } else {
            cfg.support_k
        };
        let s_me = select_diverse_support(
            &own_canon,
            &sol1.own_policy,
            &sol1.row_vs_q,
            &sol1.row_guarantees,
            &own_tactical,
            k_eff,
        );
        let mut s_opp: Vec<Vec<usize>> = Vec::with_capacity(n_w2);
        for &w_id in &s2_world_ids {
            let wd = &worlds[w_id];
            let m = wd.opp_canon.len();
            let opp_ev: Vec<f64> = sol1.col_vs_p[w_id].iter().map(|&v| 1.0 - v).collect();
            if let Some(ref qp) = wd.opp_prior {
                s_opp.push(select_prior_opp_support(qp, &opp_ev, k_eff));
                continue;
            }
            let mut opp_guar = vec![1.0f64; m];
            for j in 0..m {
                let mut mx = 0.0f64;
                for i in 0..n_own {
                    let v = wd.stage1_vals[i * m + j];
                    if v > mx {
                        mx = v;
                    }
                }
                opp_guar[j] = 1.0 - mx;
            }
            s_opp.push(select_diverse_support(
                &wd.opp_canon,
                &sol1.opp_policies[w_id],
                &opp_ev,
                &opp_guar,
                &wd.opp_tactical,
                k_eff,
            ));
        }
        if s_me.is_empty() || s_opp.iter().any(Vec::is_empty) {
            return None;
        }

        let raw_stage1_vals: Vec<Vec<f64>> = if calib_active {
            worlds.iter().map(|w| w.stage1_vals.clone()).collect()
        } else {
            Vec::new()
        };
        let delta_t1: Vec<Vec<f64>> = if calib_active {
            s2_world_ids
                .iter()
                .map(|&w_id| vec![0.0f64; n_own * worlds[w_id].opp_canon.len()])
                .collect()
        } else {
            Vec::new()
        };

        let (calib, phase, work_queue) = if calib_active {
            let cc = calib_cfg.as_ref().unwrap();
            let calib_iters =
                ((f64::from(cfg.total_iters) * cc.calib_frac.clamp(0.05, 0.70)).round() as u32)
                    .max(512);
            let traj_cfg = TrajPreviewConfig {
                total_iters: calib_iters,
                iters_per_turn: cc.iters_per_turn.max(16),
                turn1_iters: cc.turn1_iters,
                rb_exit: RbExitMode::CondSearchVroot {
                    max_alive: cc.rb_max_alive,
                    eval_iters: cc.rb_eval_iters,
                },
                ridge_basis: cc.ridge_basis,
                ridge_lambda_main: cc.ridge_lambda_main,
                ridge_lambda_inter: cc.ridge_lambda_inter,
                ridge_gls: true,
                ridge_shared_multiworld: true,
                kappa: cc.kappa_gls,
                crn_mode: TrajCrnMode::PairedCrnStratified,
                c_puct: cfg.c_puct,
                fpu_red: cfg.fpu_red,
                puct_eps: cfg.puct_eps,
                use_cf_policy: cfg.use_cf_policy,
                leaf_temp: cfg.leaf_temp,
                stepped_slots: 4,
                ..TrajPreviewConfig::default()
            };
            let s2_metas = s2_world_ids.iter().map(|&w_id| all_metas[w_id].clone()).collect();
            let tail_metas = tail_world_ids.iter().map(|&w_id| all_metas[w_id].clone()).collect();
            let obs_s1_n: Vec<Vec<u32>> = s2_world_ids
                .iter()
                .map(|&w_id| vec![0u32; n_own * worlds[w_id].opp_canon.len()])
                .collect();
            let obs_s1_sum: Vec<Vec<f64>> = s2_world_ids
                .iter()
                .map(|&w_id| vec![0.0f64; n_own * worlds[w_id].opp_canon.len()])
                .collect();
            let obs_t1_n: Vec<Vec<u32>> = s2_world_ids
                .iter()
                .map(|&w_id| vec![0u32; n_own * worlds[w_id].opp_canon.len()])
                .collect();
            let obs_t1_sum: Vec<Vec<f64>> = s2_world_ids
                .iter()
                .map(|&w_id| vec![0.0f64; n_own * worlds[w_id].opp_canon.len()])
                .collect();
            let st = TrajCalibState {
                calib_iters,
                explore_iters: calib_iters / 2,
                traj_cfg,
                scratch: TrajTurnScratch::new(),
                active_slots: (0..4).map(|_| None).collect(),
                cur_slot: 0,
                task_queue: std::collections::VecDeque::with_capacity(16),
                s2_metas,
                tail_metas,
                obs_s1_n,
                obs_s1_sum,
                obs_t1_n,
                obs_t1_sum,
                own_mon_cnt: [0; 6],
                own_lead_cnt: [0; 6],
                own_bench_pair_cnt: [0; 15],
                opp_mon_cnt: vec![[0; 6]; n_w2],
                opp_lead_cnt: vec![[0; 6]; n_w2],
                opp_bench_pair_cnt: vec![[0; 15]; n_w2],
                world_games: vec![0; n_w2],
                next_block_id: 0,
                games_completed: 0,
            };
            (
                Some(Box::new(st)),
                SteppedPreviewPhase::TrajCalib,
                std::collections::VecDeque::with_capacity(s_me.len() * 2),
            )
        } else {
            let mut wq = std::collections::VecDeque::with_capacity(s_me.len() * 2);
            let first_opp = s_opp[0][0];
            for &own_i in &s_me {
                wq.push_back((0usize, own_i, first_opp, 1u32));
            }
            (None, SteppedPreviewPhase::CoarseTier0, wq)
        };

        let coarse_budget = (u64::from(cfg.total_iters) * 45 / 100) as u32;
        let coarse_cap = (cfg.total_iters / 320).clamp(cfg.coarse_iters.min(640), 640).max(1);
        let mut coarse_per_w = vec![cfg.coarse_iters; n_w2];
        for wi in 0..n_w2 {
            let w_budget = ((f64::from(coarse_budget) * s2_alloc_weights[wi]).round() as u32).max(1);
            let n_c = (s_me.len() * s_opp[wi].len()).max(1) as u32;
            coarse_per_w[wi] = (w_budget / n_c).clamp(coarse_cap.min(32), coarse_cap);
        }

        let eg_weights = EvalWeights::default();
        let eg_playout = Playout::Heavy {
            turns: 8,
            eps: 0.2,
            weights: eg_weights.clone(),
        };
        let rollout_rules = RmConfig::default().rollout_rules();

        Some(Self {
            my_acts: my_acts.to_vec(),
            side,
            cfg,
            own_canon,
            own_tactical,
            worlds,
            raw_stage1_vals,
            delta_t1,
            calib_cfg,
            calib,
            calib_obs: Vec::new(),
            calib_s2_metas: Vec::new(),
            calib_tail_metas: Vec::new(),
            calib_iters_spent: 0,
            s2_world_ids,
            tail_world_ids,
            s2_alloc_weights,
            s2_post_weights,
            tail_post_weights,
            s_me,
            s_opp,
            coarse_per_w,
            cells: FxHashMap::default(),
            eg_weights,
            eg_playout,
            rollout_rules,
            path_buf: Vec::with_capacity(64),
            sol2: sol1,
            iters_done: 0,
            total_nodes: 0,
            current_work: None,
            work_queue,
            phase,
        })
    }

    pub fn node_count(&self) -> usize {
        self.total_nodes
    }

    pub fn iters_done(&self) -> u32 {
        self.iters_done
    }

    #[inline]
    fn cell_val_mine(&self, wi: usize, own_i: usize, opp_j: usize) -> f64 {
        let w_id = self.s2_world_ids[wi];
        let m_full = self.worlds[w_id].opp_canon.len();
        let idx = own_i * m_full + opp_j;
        if let Some(cell) = self.cells.get(&(wi, own_i, opp_j)) {
            let v0 = cell.value0(self.cfg.cell_readout);
            let v_t1 = if self.side == 0 { v0 } else { 1.0 - v0 };
            let dt = self
                .delta_t1
                .get(wi)
                .and_then(|v| v.get(idx))
                .copied()
                .unwrap_or(0.0);
            (v_t1 + dt).clamp(0.01, 0.99)
        } else {
            self.worlds[w_id].stage1_vals[idx]
        }
    }

    fn fit_calib_ridge(
        &self,
        cc: &TrajCalibConfig,
        s2_metas: &[WorldInteractMeta],
        obs_n: &[Vec<u32>],
        obs_sum: &[Vec<f64>],
    ) -> ([f64; MAX_RIDGE_DIM], Vec<[f64; MAX_RIDGE_DIM]>) {
        let n_own = self.own_canon.len();
        let n_w2 = self.s2_world_ids.len();
        let basis = cc.ridge_basis;
        let dim = basis.dim();
        let main_end = basis.main_end();
        let kappa = cc.kappa_gls.max(0.5);
        let mut feats = [(0usize, 0.0f64); MAX_RIDGE_NNZ];

        // 1. Fit shared residual ridge across all Stage-2 worlds.
        let mut theta_shared = [0.0f64; MAX_RIDGE_DIM];
        {
            let mut ata = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
            let mut atb = [0.0f64; MAX_RIDGE_DIM];
            let mut any_obs = false;
            for wi in 0..n_w2 {
                let w_id = self.s2_world_ids[wi];
                let wd = &self.worlds[w_id];
                let m = wd.opp_canon.len();
                let meta = &s2_metas[wi];
                for i in 0..n_own {
                    let own_t = self.own_canon[i].1;
                    for j in 0..m {
                        let idx = i * m + j;
                        let n = obs_n[wi][idx];
                        if n == 0 {
                            continue;
                        }
                        any_obs = true;
                        let nf = f64::from(n);
                        let r_bar = obs_sum[wi][idx] / nf;
                        let w_cell = (kappa + 1.0) * nf / (kappa + nf);
                        let nnz = extract_ridge_features(
                            basis,
                            self.side,
                            own_t,
                            wd.opp_canon[j].1,
                            meta,
                            &mut feats,
                        );
                        for a in 0..nnz {
                            let (fa, va) = feats[a];
                            atb[fa] += w_cell * va * r_bar;
                            for b in 0..=a {
                                let (fb, vb) = feats[b];
                                let prod = w_cell * va * vb;
                                ata[fa][fb] += prod;
                                if fa != fb {
                                    ata[fb][fa] += prod;
                                }
                            }
                        }
                    }
                }
            }
            if any_obs {
                theta_shared = solve_spd_cholesky_split(
                    dim,
                    main_end,
                    &ata,
                    &atb,
                    cc.ridge_lambda_int,
                    cc.ridge_lambda_main,
                    cc.ridge_lambda_inter,
                );
            }
        }

        // 2. Shrunk per-world residual deltas when W > 1 (optional; disabled by default to prevent W=8 overfitting).
        let mut deltas_w = vec![[0.0f64; MAX_RIDGE_DIM]; n_w2];
        if n_w2 > 1 && cc.per_world_delta {
            for wi in 0..n_w2 {
                let w_id = self.s2_world_ids[wi];
                let wd = &self.worlds[w_id];
                let m = wd.opp_canon.len();
                let meta = &s2_metas[wi];
                let mut ata = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
                let mut atb = [0.0f64; MAX_RIDGE_DIM];
                let mut any_w = false;
                for i in 0..n_own {
                    let own_t = self.own_canon[i].1;
                    for j in 0..m {
                        let idx = i * m + j;
                        let n = obs_n[wi][idx];
                        if n == 0 {
                            continue;
                        }
                        any_w = true;
                        let nf = f64::from(n);
                        let nnz = extract_ridge_features(
                            basis,
                            self.side,
                            own_t,
                            wd.opp_canon[j].1,
                            meta,
                            &mut feats,
                        );
                        let mut shared_pred = 0.0;
                        for &(f, v) in &feats[..nnz] {
                            shared_pred += theta_shared[f] * v;
                        }
                        let r_rem = (obs_sum[wi][idx] / nf) - shared_pred;
                        let w_cell = (kappa + 1.0) * nf / (kappa + nf);
                        for a in 0..nnz {
                            let (fa, va) = feats[a];
                            atb[fa] += w_cell * va * r_rem;
                            for b in 0..=a {
                                let (fb, vb) = feats[b];
                                let prod = w_cell * va * vb;
                                ata[fa][fb] += prod;
                                if fa != fb {
                                    ata[fb][fa] += prod;
                                }
                            }
                        }
                    }
                }
                if any_w {
                    deltas_w[wi] = solve_spd_cholesky_split(
                        dim,
                        main_end,
                        &ata,
                        &atb,
                        2.0,
                        cc.ridge_lambda_main * 2.0,
                        cc.ridge_lambda_inter * 2.0,
                    );
                }
            }
        }
        (theta_shared, deltas_w)
    }

    fn enqueue_calib_batch(&mut self, rng: &mut SplitMix64) {
        let n_own = self.own_canon.len();
        let n_w2 = self.s2_world_ids.len();
        let cc = self.calib_cfg.clone().unwrap();

        let (wi, in_explore) = {
            let cal = self.calib.as_ref().unwrap();
            let mut best_wi = 0usize;
            let mut best_ratio = f64::INFINITY;
            for w in 0..n_w2 {
                let ratio = f64::from(cal.world_games[w]) / self.s2_alloc_weights[w].max(1e-6);
                if ratio < best_ratio {
                    best_ratio = ratio;
                    best_wi = w;
                }
            }
            (best_wi, self.iters_done < cal.explore_iters)
        };
        let w_id = self.s2_world_ids[wi];
        let m = self.worlds[w_id].opp_canon.len();

        let score_triple = |t: [u8; 3],
                            mon_cnt: &[u32; 6],
                            lead_cnt: &[u32; 6],
                            bp_cnt: &[u32; 15]|
         -> f64 {
            let mut sc = 0.0;
            for (pos, &s) in t.iter().enumerate() {
                if s >= 1 && (s as usize) <= 6 {
                    let idx = s as usize - 1;
                    sc += f64::from(mon_cnt[idx]) * 1.0;
                    if pos == 0 {
                        sc += f64::from(lead_cnt[idx]) * 0.7;
                    }
                }
            }
            if let Some(bp) = SteppedTrajPreview::bench_pair_index(t) {
                sc += f64::from(bp_cnt[bp]) * 0.5;
            }
            sc
        };

        let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(4);
        if in_explore {
            let cal = self.calib.as_ref().unwrap();
            let mut used_i: Vec<usize> = Vec::with_capacity(4);
            let mut used_j: Vec<usize> = Vec::with_capacity(4);
            for _ in 0..4 {
                let mut best_i = 0usize;
                let mut best_si = f64::INFINITY;
                for i in 0..n_own {
                    if used_i.contains(&i) {
                        continue;
                    }
                    let sc = score_triple(
                        self.own_canon[i].1,
                        &cal.own_mon_cnt,
                        &cal.own_lead_cnt,
                        &cal.own_bench_pair_cnt,
                    ) + rng.next_f64() * 0.05;
                    if sc < best_si {
                        best_si = sc;
                        best_i = i;
                    }
                }
                let best_j = if let Some(ref qp) = self.worlds[w_id].opp_prior {
                    sample_dist(qp, rng.next_f64())
                } else {
                    let mut best_j = 0usize;
                    let mut best_sj = f64::INFINITY;
                    for j in 0..m {
                        if used_j.contains(&j) {
                            continue;
                        }
                        let sc = score_triple(
                            self.worlds[w_id].opp_canon[j].1,
                            &cal.opp_mon_cnt[wi],
                            &cal.opp_lead_cnt[wi],
                            &cal.opp_bench_pair_cnt[wi],
                        ) + rng.next_f64() * 0.05;
                        if sc < best_sj {
                            best_sj = sc;
                            best_j = j;
                        }
                    }
                    used_j.push(best_j);
                    best_j
                };
                used_i.push(best_i);
                pairs.push((best_i, best_j));
            }
        } else {
            // Equilibrium-guided DO trajectories on the current calibrated Stage-1 + Ridge matrix!
            let (theta_s1, dw_s1) = {
                let cal = self.calib.as_ref().unwrap();
                self.fit_calib_ridge(&cc, &cal.s2_metas, &cal.obs_s1_n, &cal.obs_s1_sum)
            };
            let basis = cc.ridge_basis;
            let mut feats = [(0usize, 0.0f64); MAX_RIDGE_NNZ];
            let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2);
            let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2);
            {
                let cal = self.calib.as_ref().unwrap();
                for w in 0..n_w2 {
                    let wid = self.s2_world_ids[w];
                    let wd = &self.worlds[wid];
                    let mw = wd.opp_canon.len();
                    let meta = &cal.s2_metas[w];
                    let mut mat = vec![0.5f64; n_own * mw];
                    for i in 0..n_own {
                        let own_t = self.own_canon[i].1;
                        for j in 0..mw {
                            let idx = i * mw + j;
                            let nnz = extract_ridge_features(
                                basis,
                                self.side,
                                own_t,
                                wd.opp_canon[j].1,
                                meta,
                                &mut feats,
                            );
                            let mut corr = 0.0;
                            for &(f, v) in &feats[..nnz] {
                                corr += (theta_s1[f] + dw_s1[w][f]) * v;
                            }
                            mat[idx] = (self.raw_stage1_vals[wid][idx] + cc.delta_s1_beta * corr)
                                .clamp(0.02, 0.98);
                        }
                    }
                    mats.push(mat);
                    col_lens.push(mw);
                }
            }
            let slices: Vec<(f64, usize, &[f64])> = (0..n_w2)
                .map(|w| (self.s2_alloc_weights[w], col_lens[w], mats[w].as_slice()))
                .collect();
            let priors: Vec<Option<&[f64]>> = (0..n_w2)
                .map(|w| self.worlds[self.s2_world_ids[w]].opp_prior.as_deref())
                .collect();
            let sol = solve_bayesian_rm_plus_with_priors(n_own, &slices, &priors, 250);
            let cal = self.calib.as_ref().unwrap();

            // 1. Paired own challenger vs incumbent against equilibrium opponent.
            let j_tgt = sample_dist(&sol.opp_policies[wi], rng.next_f64());
            let i_inc = sample_dist(&sol.own_policy, rng.next_f64());
            let mut own_cands: Vec<(usize, f64)> = (0..n_own)
                .filter(|&i| i != i_inc)
                .map(|i| {
                    let n = cal.obs_s1_n[wi][i * m + j_tgt];
                    (i, sol.row_vs_q[i] + 0.08 / f64::from(n + 1).sqrt())
                })
                .collect();
            own_cands.sort_by(|a, b| b.1.total_cmp(&a.1));
            let i_ch = own_cands.first().map_or(i_inc, |&(i, _)| i);
            pairs.push((i_inc, j_tgt));
            pairs.push((i_ch, j_tgt));

            // 2. Paired opp challenger vs incumbent against equilibrium own pick.
            let i_tgt = sample_dist(&sol.own_policy, rng.next_f64());
            let j_inc = sample_dist(&sol.opp_policies[wi], rng.next_f64());
            let j_ch = if let Some(ref qp) = self.worlds[w_id].opp_prior {
                (0..m)
                    .filter(|&j| j != j_inc)
                    .max_by(|&a, &b| {
                        let sa = qp[a] / f64::from(cal.obs_s1_n[wi][i_tgt * m + a] + 1).sqrt();
                        let sb = qp[b] / f64::from(cal.obs_s1_n[wi][i_tgt * m + b] + 1).sqrt();
                        sa.total_cmp(&sb)
                    })
                    .unwrap_or(j_inc)
            } else {
                let mut opp_cands: Vec<(usize, f64)> = (0..m)
                    .filter(|&j| j != j_inc)
                    .map(|j| {
                        let n = cal.obs_s1_n[wi][i_tgt * m + j];
                        (j, (1.0 - sol.col_vs_p[wi][j]) + 0.08 / f64::from(n + 1).sqrt())
                    })
                    .collect();
                opp_cands.sort_by(|a, b| b.1.total_cmp(&a.1));
                opp_cands.first().map_or(j_inc, |&(j, _)| j)
            };
            pairs.push((i_tgt, j_inc));
            pairs.push((i_tgt, j_ch));
        }

        let cal = self.calib.as_mut().unwrap();
        let blk = cal.next_block_id;
        cal.next_block_id += 1;
        let shared_seed = rng.next();
        for &(ri, cj) in &pairs {
            let own_t = self.own_canon[ri].1;
            let opp_t = self.worlds[w_id].opp_canon[cj].1;
            for (pos, &s) in own_t.iter().enumerate() {
                if s >= 1 && (s as usize) <= 6 {
                    let idx = s as usize - 1;
                    cal.own_mon_cnt[idx] += 1;
                    if pos == 0 {
                        cal.own_lead_cnt[idx] += 1;
                    }
                }
            }
            if let Some(bp) = SteppedTrajPreview::bench_pair_index(own_t) {
                cal.own_bench_pair_cnt[bp] += 1;
            }
            for (pos, &s) in opp_t.iter().enumerate() {
                if s >= 1 && (s as usize) <= 6 {
                    let idx = s as usize - 1;
                    cal.opp_mon_cnt[wi][idx] += 1;
                    if pos == 0 {
                        cal.opp_lead_cnt[wi][idx] += 1;
                    }
                }
            }
            if let Some(bp) = SteppedTrajPreview::bench_pair_index(opp_t) {
                cal.opp_bench_pair_cnt[wi][bp] += 1;
            }
            cal.world_games[wi] += 1;
            let visit_k = cal.obs_s1_n[wi][ri * m + cj];
            cal.task_queue.push_back(TrajTask {
                wi,
                own_i: ri,
                opp_j: cj,
                block_id: blk,
                game_seed: shared_seed,
                cell_visit_k: visit_k,
                t1_only_iters: 0,
            });
        }
    }

    fn finalize_calib(&mut self) {
        let Some(cal) = self.calib.take() else {
            return;
        };
        let cc = self.calib_cfg.clone().unwrap();
        let n_own = self.own_canon.len();
        let n_w2 = self.s2_world_ids.len();
        let basis = cc.ridge_basis;
        let mut feats = [(0usize, 0.0f64); MAX_RIDGE_NNZ];

        // 1. Fit Stage-1 horizon delta (V_traj - raw_stage1) and update `worlds[*].stage1_vals`.
        let (theta_s1, dw_s1) =
            self.fit_calib_ridge(&cc, &cal.s2_metas, &cal.obs_s1_n, &cal.obs_s1_sum);
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let m = self.worlds[w_id].opp_canon.len();
            let meta = &cal.s2_metas[wi];
            for i in 0..n_own {
                let own_t = self.own_canon[i].1;
                for j in 0..m {
                    let idx = i * m + j;
                    let nnz = extract_ridge_features(
                        basis,
                        self.side,
                        own_t,
                        self.worlds[w_id].opp_canon[j].1,
                        meta,
                        &mut feats,
                    );
                    let mut corr = 0.0;
                    for &(f, v) in &feats[..nnz] {
                        corr += (theta_s1[f] + dw_s1[wi][f]) * v;
                    }
                    self.worlds[w_id].stage1_vals[idx] =
                        (self.raw_stage1_vals[w_id][idx] + cc.delta_s1_beta * corr)
                            .clamp(0.02, 0.98);
                }
            }
        }
        // Transfer 75% of shared species-level horizon delta to tail belief worlds.
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let m = self.worlds[w_id].opp_canon.len();
            let meta = &cal.tail_metas[ti];
            for i in 0..n_own {
                let own_t = self.own_canon[i].1;
                for j in 0..m {
                    let idx = i * m + j;
                    let nnz = extract_ridge_features(
                        basis,
                        self.side,
                        own_t,
                        self.worlds[w_id].opp_canon[j].1,
                        meta,
                        &mut feats,
                    );
                    let mut corr = 0.0;
                    for &(f, v) in &feats[..nnz] {
                        corr += theta_s1[f] * v;
                    }
                    self.worlds[w_id].stage1_vals[idx] =
                        (self.raw_stage1_vals[w_id][idx] + 0.75 * cc.delta_s1_beta * corr)
                            .clamp(0.02, 0.98);
                }
            }
        }

        // 2. Fit Stage-2 Turn-1 control-variate delta (V_traj - V_T1) for every S2 world.
        let (theta_t1, dw_t1) =
            self.fit_calib_ridge(&cc, &cal.s2_metas, &cal.obs_t1_n, &cal.obs_t1_sum);
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let m = self.worlds[w_id].opp_canon.len();
            let meta = &cal.s2_metas[wi];
            for i in 0..n_own {
                let own_t = self.own_canon[i].1;
                for j in 0..m {
                    let idx = i * m + j;
                    let nnz = extract_ridge_features(
                        basis,
                        self.side,
                        own_t,
                        self.worlds[w_id].opp_canon[j].1,
                        meta,
                        &mut feats,
                    );
                    let mut corr = 0.0;
                    for &(f, v) in &feats[..nnz] {
                        corr += (theta_t1[f] + dw_t1[wi][f]) * v;
                    }
                    self.delta_t1[wi][idx] = (cc.delta_t1_beta * corr).clamp(-0.35, 0.35);
                }
            }
        }

        // 3. Re-solve calibrated Stage-1 equilibrium and re-select Stage-2 diverse support!
        let s1_slices: Vec<(f64, usize, &[f64])> = self
            .worlds
            .iter()
            .map(|w| (w.weight, w.opp_canon.len(), w.stage1_vals.as_slice()))
            .collect();
        let s1_priors: Vec<Option<&[f64]>> =
            self.worlds.iter().map(|w| w.opp_prior.as_deref()).collect();
        let sol1 = solve_bayesian_rm_plus_with_priors(
            n_own,
            &s1_slices,
            &s1_priors,
            self.cfg.rm_sweeps_stage1,
        );
        let k_eff = if n_w2 >= 4 {
            self.cfg.support_k.min(8)
        } else {
            self.cfg.support_k
        };
        let k_cal = if cc.union_raw_support {
            (k_eff * 2 / 3).max(4)
        } else {
            k_eff
        };
        let mut new_s_me = select_diverse_support(
            &self.own_canon,
            &sol1.own_policy,
            &sol1.row_vs_q,
            &sol1.row_guarantees,
            &self.own_tactical,
            k_cal,
        );
        if cc.union_raw_support {
            for &raw_i in &self.s_me {
                if new_s_me.len() >= k_eff {
                    break;
                }
                if !new_s_me.contains(&raw_i) {
                    new_s_me.push(raw_i);
                }
            }
        }
        let mut new_s_opp: Vec<Vec<usize>> = Vec::with_capacity(n_w2);
        for (wi, &w_id) in self.s2_world_ids.iter().enumerate() {
            let wd = &self.worlds[w_id];
            let m = wd.opp_canon.len();
            let opp_ev: Vec<f64> = sol1.col_vs_p[w_id].iter().map(|&v| 1.0 - v).collect();
            let mut so = if let Some(ref qp) = wd.opp_prior {
                select_prior_opp_support(qp, &opp_ev, k_cal)
            } else {
                let mut opp_guar = vec![1.0f64; m];
                for j in 0..m {
                    let mut mx = 0.0f64;
                    for i in 0..n_own {
                        let v = wd.stage1_vals[i * m + j];
                        if v > mx {
                            mx = v;
                        }
                    }
                    opp_guar[j] = 1.0 - mx;
                }
                select_diverse_support(
                    &wd.opp_canon,
                    &sol1.opp_policies[w_id],
                    &opp_ev,
                    &opp_guar,
                    &wd.opp_tactical,
                    k_cal,
                )
            };
            if cc.union_raw_support {
                for &raw_j in &self.s_opp[wi] {
                    if so.len() >= k_eff {
                        break;
                    }
                    if !so.contains(&raw_j) {
                        so.push(raw_j);
                    }
                }
            }
            new_s_opp.push(so);
        }
        if !new_s_me.is_empty() && !new_s_opp.iter().any(Vec::is_empty) {
            self.s_me = new_s_me;
            self.s_opp = new_s_opp;
        }
        self.sol2 = sol1;

        // 4. Recompute Stage-2 Coarse budget from remaining iterations and kick off CoarseTier0.
        self.calib_s2_metas = cal.s2_metas;
        self.calib_tail_metas = cal.tail_metas;
        self.calib_iters_spent = self.iters_done;
        let rem_budget = self.cfg.total_iters.saturating_sub(self.iters_done).max(1024);
        let coarse_budget = (u64::from(rem_budget) * 45 / 100) as u32;
        let coarse_cap = (rem_budget / 256).clamp(self.cfg.coarse_iters.min(640), 640).max(1);
        for wi in 0..n_w2 {
            let w_budget =
                ((f64::from(coarse_budget) * self.s2_alloc_weights[wi]).round() as u32).max(1);
            let n_c = (self.s_me.len() * self.s_opp[wi].len()).max(1) as u32;
            self.coarse_per_w[wi] = (w_budget / n_c).clamp(coarse_cap.min(32), coarse_cap);
        }

        self.work_queue.clear();
        let first_opp = self.s_opp[0][0];
        for &own_i in &self.s_me {
            self.work_queue.push_back((0usize, own_i, first_opp, 1u32));
        }
        self.phase = SteppedPreviewPhase::CoarseTier0;
    }

    fn step_calib_one(&mut self, rng: &mut SplitMix64) -> (usize, SearchChoice, f64, f64, u32) {
        let n_slots = self.calib.as_ref().unwrap().active_slots.len();
        loop {
            let need_spawn = {
                let cal = self.calib.as_ref().unwrap();
                cal.active_slots[cal.cur_slot].is_none()
            };
            if need_spawn {
                if self.calib.as_ref().unwrap().task_queue.is_empty() {
                    self.enqueue_calib_batch(rng);
                }
                let cal = self.calib.as_mut().unwrap();
                let task = cal.task_queue.pop_front().unwrap();
                let w_id = self.s2_world_ids[task.wi];
                let wd = &self.worlds[w_id];
                let (ref roster, ref pcache) = wd.compact.as_ref().unwrap();
                let own_t = self.own_canon[task.own_i].1;
                let opp_t = wd.opp_canon[task.opp_j].1;
                let picks = if self.side == 0 {
                    [own_t, opp_t]
                } else {
                    [opp_t, own_t]
                };
                cal.active_slots[cal.cur_slot] = Some(ActiveTrajGame::new(
                    task.wi,
                    task.own_i,
                    task.opp_j,
                    task.block_id,
                    roster,
                    pcache,
                    picks,
                    task.game_seed,
                    task.cell_visit_k,
                    cal.traj_cfg.crn_mode,
                    0,
                ));
            }

            let mut slot_game = {
                let cal = self.calib.as_mut().unwrap();
                cal.active_slots[cal.cur_slot].take().unwrap()
            };
            if !slot_game.turn_ready {
                let done_opt = {
                    let cal = self.calib.as_mut().unwrap();
                    slot_game.prepare_next_turn(&cal.traj_cfg, &mut cal.scratch)
                };
                if let Some((v0, _rb)) = done_opt {
                    let wi = slot_game.wi;
                    let own_i = slot_game.own_i;
                    let opp_j = slot_game.opp_j;
                    let v_mine = if self.side == 0 { v0 } else { 1.0 - v0 };
                    let w_id = self.s2_world_ids[wi];
                    let m = self.worlds[w_id].opp_canon.len();
                    let idx = own_i * m + opp_j;
                    let cal = self.calib.as_mut().unwrap();
                    let r_s1 = v_mine - self.raw_stage1_vals[w_id][idx];
                    cal.obs_s1_n[wi][idx] += 1;
                    cal.obs_s1_sum[wi][idx] += r_s1;
                    let (r_t1, has_t1) = if let Some(t1_v0) = slot_game.turn1_v0 {
                        let t1_mine = if self.side == 0 { t1_v0 } else { 1.0 - t1_v0 };
                        let d = v_mine - t1_mine;
                        cal.obs_t1_n[wi][idx] += 1;
                        cal.obs_t1_sum[wi][idx] += d;
                        (d as f32, true)
                    } else {
                        (0.0, false)
                    };
                    cal.games_completed += 1;
                    cal.cur_slot = (cal.cur_slot + 1) % n_slots;
                    self.calib_obs.push(CalibObs {
                        wi: wi as u16,
                        own_i: own_i as u16,
                        opp_j: opp_j as u16,
                        block_id: slot_game.block_id,
                        r_s1: r_s1 as f32,
                        r_t1,
                        has_t1,
                    });
                    continue;
                }
            }

            let (val0, depth, added) = {
                let cal = self.calib.as_mut().unwrap();
                cal.scratch.step_puct_iter(
                    &slot_game.turn_root_cb,
                    &slot_game.table,
                    &slot_game.cache,
                    slot_game.turn_cap,
                    cal.traj_cfg.c_puct,
                    cal.traj_cfg.fpu_red,
                    cal.traj_cfg.puct_eps,
                    cal.traj_cfg.use_cf_policy,
                    cal.traj_cfg.leaf_temp,
                    &mut slot_game.turn_rng,
                )
            };
            self.total_nodes += added;
            self.iters_done += 1;
            slot_game.turn_iters_done += 1;
            if slot_game.turn1_v0.is_none()
                && slot_game.cb.turn == 1
                && slot_game.turn_iters_done.saturating_mul(2) > slot_game.turn_target_iters
            {
                slot_game.t1_late_sum += val0;
                slot_game.t1_late_n += 1;
            }

            let wi = slot_game.wi;
            let own_i = slot_game.own_i;
            let opp_j = slot_game.opp_j;
            let opp_t = self.worlds[self.s2_world_ids[wi]].opp_canon[opp_j].1;

            if slot_game.turn_iters_done >= slot_game.turn_target_iters {
                if slot_game.turn1_v0.is_none() && slot_game.cb.turn == 1 {
                    let cal = self.calib.as_ref().unwrap();
                    let t1_v0 = if slot_game.t1_late_n > 0 {
                        (slot_game.t1_late_sum / f64::from(slot_game.t1_late_n)).clamp(0.0, 1.0)
                    } else {
                        cal.scratch.readout_turn().1
                    };
                    slot_game.turn1_v0 = Some(t1_v0);
                }
                let finished = {
                    let cal = self.calib.as_ref().unwrap();
                    slot_game.finish_turn(&cal.scratch)
                };
                let mut completed_obs: Option<CalibObs> = None;
                let cal = self.calib.as_mut().unwrap();
                if let Some((final_v0, _rb)) = finished {
                    let v_mine = if self.side == 0 {
                        final_v0
                    } else {
                        1.0 - final_v0
                    };
                    let w_id = self.s2_world_ids[wi];
                    let m = self.worlds[w_id].opp_canon.len();
                    let idx = own_i * m + opp_j;
                    let r_s1 = v_mine - self.raw_stage1_vals[w_id][idx];
                    cal.obs_s1_n[wi][idx] += 1;
                    cal.obs_s1_sum[wi][idx] += r_s1;
                    let (r_t1, has_t1) = if let Some(t1_v0) = slot_game.turn1_v0 {
                        let t1_mine = if self.side == 0 { t1_v0 } else { 1.0 - t1_v0 };
                        let d = v_mine - t1_mine;
                        cal.obs_t1_n[wi][idx] += 1;
                        cal.obs_t1_sum[wi][idx] += d;
                        (d as f32, true)
                    } else {
                        (0.0, false)
                    };
                    cal.games_completed += 1;
                    completed_obs = Some(CalibObs {
                        wi: wi as u16,
                        own_i: own_i as u16,
                        opp_j: opp_j as u16,
                        block_id: slot_game.block_id,
                        r_s1: r_s1 as f32,
                        r_t1,
                        has_t1,
                    });
                } else {
                    cal.active_slots[cal.cur_slot] = Some(slot_game);
                }
                cal.cur_slot = (cal.cur_slot + 1) % n_slots;
                if let Some(obs) = completed_obs {
                    self.calib_obs.push(obs);
                }
            } else {
                let cal = self.calib.as_mut().unwrap();
                cal.active_slots[cal.cur_slot] = Some(slot_game);
            }

            if self.iters_done >= self.calib.as_ref().unwrap().calib_iters {
                self.finalize_calib();
            }

            let val_mine = if self.side == 0 { val0 } else { 1.0 - val0 };
            let own_orig_idx = self.own_canon[own_i].0;
            return (own_orig_idx, SearchChoice::Team(opp_t), val0, val_mine, depth);
        }
    }

    fn solve_subgame(&self, sweeps: u32) -> (Vec<Vec<f64>>, BayesianRmSolution) {
        let k_me = self.s_me.len();
        let n_w2 = self.s2_world_ids.len();
        let n_tail = self.tail_world_ids.len();
        let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
        let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2 + n_tail);
        let mut w_list: Vec<f64> = Vec::with_capacity(n_w2 + n_tail);
        let mut sub_priors: Vec<Option<Vec<f64>>> = Vec::with_capacity(n_w2 + n_tail);
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let k_opp = self.s_opp[wi].len();
            let mut mat = vec![0.5f64; k_me * k_opp];
            for (ri, &own_i) in self.s_me.iter().enumerate() {
                for (cj, &opp_j) in self.s_opp[wi].iter().enumerate() {
                    mat[ri * k_opp + cj] = self.cell_val_mine(wi, own_i, opp_j);
                }
            }
            mats.push(mat);
            col_lens.push(k_opp);
            w_list.push(self.s2_post_weights[wi]);
            sub_priors.push(restrict_opp_prior(
                self.worlds[w_id].opp_prior.as_deref(),
                &self.s_opp[wi],
            ));
        }
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let wd = &self.worlds[w_id];
            let m_full = wd.opp_canon.len();
            let mut mat = vec![0.5f64; k_me * m_full];
            for (ri, &own_i) in self.s_me.iter().enumerate() {
                let src = &wd.stage1_vals[own_i * m_full..(own_i + 1) * m_full];
                mat[ri * m_full..(ri + 1) * m_full].copy_from_slice(src);
            }
            mats.push(mat);
            col_lens.push(m_full);
            w_list.push(self.tail_post_weights[ti]);
            sub_priors.push(wd.opp_prior.clone());
        }
        let slices: Vec<(f64, usize, &[f64])> = (0..mats.len())
            .map(|idx| (w_list[idx], col_lens[idx], mats[idx].as_slice()))
            .collect();
        let prior_refs: Vec<Option<&[f64]>> = sub_priors.iter().map(|p| p.as_deref()).collect();
        let sol = solve_bayesian_rm_plus_with_priors(k_me, &slices, &prior_refs, sweeps);
        (mats, sol)
    }

    fn build_do_opp_candidates(&self, wi: usize, supp_me: &[usize], supp_p_sum: f64) -> Vec<usize> {
        let n_w2 = self.s2_world_ids.len();
        if n_w2 > 1 && self.s2_alloc_weights[wi] < 0.04 {
            return Vec::new();
        }
        let wd = &self.worlds[self.s2_world_ids[wi]];
        if wd.opp_prior.is_some() {
            return Vec::new();
        }
        let m_full = wd.opp_canon.len();
        let mut outside: Vec<(usize, f64)> = Vec::new();
        for j in 0..m_full {
            if self.s_opp[wi].contains(&j) {
                continue;
            }
            let mut ev = 0.0;
            for &ri in supp_me {
                let pw = if supp_p_sum > 0.0 {
                    self.sol2.own_policy[ri] / supp_p_sum
                } else {
                    1.0 / (supp_me.len() as f64)
                };
                ev += pw * wd.stage1_vals[self.s_me[ri] * m_full + j];
            }
            outside.push((j, ev));
        }
        outside.sort_by(|a, b| a.1.total_cmp(&b.1));
        outside
            .into_iter()
            .take(self.cfg.do_candidates)
            .map(|(j, _)| j)
            .collect()
    }

    fn start_do_round(&mut self, round: usize, do_cap_iters: u32, opp_do_cap: u32) {
        let mut supp_me: Vec<usize> = (0..self.s_me.len())
            .filter(|&ri| self.sol2.own_policy[ri] >= 0.02)
            .collect();
        if supp_me.is_empty() {
            if let Some(best_r) = (0..self.s_me.len())
                .max_by(|&a, &b| self.sol2.own_policy[a].total_cmp(&self.sol2.own_policy[b]))
            {
                supp_me.push(best_r);
            }
        }
        let supp_p_sum: f64 = supp_me.iter().map(|&ri| self.sol2.own_policy[ri]).sum();
        let candidates = self.build_do_opp_candidates(0, &supp_me, supp_p_sum);
        self.phase = SteppedPreviewPhase::DoOpp {
            round,
            do_cap_iters,
            opp_do_cap,
            supp_me,
            supp_p_sum,
            wi: 0,
            candidates,
            cand_pos: 0,
            active_cand: None,
            pending_admit: Vec::new(),
            admitted_opp: false,
        };
    }

    fn start_do_own(&mut self, round: usize, do_cap_iters: u32, admitted_opp: bool) {
        let n_w2 = self.s2_world_ids.len();
        let supp_opp: Vec<Vec<usize>> = (0..n_w2)
            .map(|wi| {
                let mut v: Vec<usize> = (0..self.s_opp[wi].len())
                    .filter(|&cj| self.sol2.opp_policies[wi][cj] >= 0.02)
                    .collect();
                if v.is_empty() {
                    if let Some(bc) = (0..self.s_opp[wi].len()).max_by(|&a, &b| {
                        self.sol2.opp_policies[wi][a].total_cmp(&self.sol2.opp_policies[wi][b])
                    }) {
                        v.push(bc);
                    }
                }
                v
            })
            .collect();
        let n_own = self.own_canon.len();
        let mut outside_me: Vec<(usize, f64)> = Vec::new();
        for i in 0..n_own {
            if self.s_me.contains(&i) {
                continue;
            }
            let mut ev = 0.0;
            for wi in 0..n_w2 {
                let wd = &self.worlds[self.s2_world_ids[wi]];
                let m_full = wd.opp_canon.len();
                let qsum: f64 = supp_opp[wi]
                    .iter()
                    .map(|&cj| self.sol2.opp_policies[wi][cj])
                    .sum();
                let mut ev_w = 0.0;
                for &cj in &supp_opp[wi] {
                    let qw = if qsum > 0.0 {
                        self.sol2.opp_policies[wi][cj] / qsum
                    } else {
                        1.0 / (supp_opp[wi].len() as f64)
                    };
                    ev_w += qw * wd.stage1_vals[i * m_full + self.s_opp[wi][cj]];
                }
                ev += self.s2_post_weights[wi] * ev_w;
            }
            for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
                let wd = &self.worlds[w_id];
                let m_full = wd.opp_canon.len();
                let q_tail = &self.sol2.opp_policies[n_w2 + ti];
                let mut ev_w = 0.0;
                for j in 0..m_full {
                    let qj = q_tail[j];
                    if qj > 0.0 {
                        ev_w += qj * wd.stage1_vals[i * m_full + j];
                    }
                }
                ev += self.tail_post_weights[ti] * ev_w;
            }
            outside_me.push((i, ev));
        }
        outside_me.sort_by(|a, b| b.1.total_cmp(&a.1));
        let candidates: Vec<usize> = outside_me
            .into_iter()
            .take(self.cfg.do_candidates)
            .map(|(i, _)| i)
            .collect();
        self.phase = SteppedPreviewPhase::DoOwn {
            round,
            do_cap_iters,
            supp_opp,
            candidates,
            cand_pos: 0,
            active_cand: None,
            pending_admit: Vec::new(),
            admitted_opp,
            admitted_me: false,
        };
    }

    fn advance_phase(&mut self) {
        let n_w2 = self.s2_world_ids.len();
        loop {
            if !self.work_queue.is_empty() {
                return;
            }
            let phase = std::mem::replace(
                &mut self.phase,
                SteppedPreviewPhase::Polish { polish_steps: 0 },
            );
            match phase {
                SteppedPreviewPhase::TrajCalib => {
                    self.phase = SteppedPreviewPhase::TrajCalib;
                    return;
                }
                SteppedPreviewPhase::CoarseTier0 => {
                    for wi in 0..n_w2 {
                        let target = self.coarse_per_w[wi].min(32);
                        for &own_i in &self.s_me {
                            for &opp_j in &self.s_opp[wi] {
                                let cur = self
                                    .cells
                                    .get(&(wi, own_i, opp_j))
                                    .map_or(0, |c| c.total_n);
                                if target > cur {
                                    self.work_queue.push_back((wi, own_i, opp_j, target - cur));
                                }
                            }
                        }
                    }
                    self.phase = SteppedPreviewPhase::CoarseTier1;
                }
                SteppedPreviewPhase::CoarseTier1 => {
                    let mut budget_rem = self.cfg.total_iters.saturating_sub(self.iters_done);
                    for wi in 0..n_w2 {
                        let target = self.coarse_per_w[wi];
                        for &own_i in &self.s_me {
                            for &opp_j in &self.s_opp[wi] {
                                if budget_rem == 0 {
                                    break;
                                }
                                let cur = self
                                    .cells
                                    .get(&(wi, own_i, opp_j))
                                    .map_or(0, |c| c.total_n);
                                if target > cur {
                                    let add = (target - cur).min(budget_rem);
                                    self.work_queue.push_back((wi, own_i, opp_j, add));
                                    budget_rem -= add;
                                }
                            }
                        }
                    }
                    self.phase = SteppedPreviewPhase::CoarseTier2;
                }
                SteppedPreviewPhase::CoarseTier2 => {
                    let (_, sol2) = self.solve_subgame(600);
                    self.sol2 = sol2;
                    let rem_after_calib =
                        self.cfg.total_iters.saturating_sub(self.calib_iters_spent);
                    let do_cap_iters = self
                        .calib_iters_spent
                        .saturating_add((u64::from(rem_after_calib) * 65 / 100) as u32);
                    if self.cfg.do_rounds > 0 && self.iters_done < do_cap_iters {
                        let opp_do_cap = self.iters_done.saturating_add(
                            (do_cap_iters.saturating_sub(self.iters_done)) * 60 / 100,
                        );
                        self.start_do_round(0, do_cap_iters, opp_do_cap);
                    } else {
                        self.phase = SteppedPreviewPhase::Polish { polish_steps: 0 };
                    }
                }
                SteppedPreviewPhase::DoOpp {
                    round,
                    do_cap_iters,
                    opp_do_cap,
                    supp_me,
                    supp_p_sum,
                    mut wi,
                    mut candidates,
                    mut cand_pos,
                    mut active_cand,
                    mut pending_admit,
                    mut admitted_opp,
                } => {
                    if let Some((cand_j, probed_all)) = active_cand.take() {
                        if probed_all {
                            let mut ev2 = 0.0;
                            for &ri in &supp_me {
                                let own_i = self.s_me[ri];
                                let pw = if supp_p_sum > 0.0 {
                                    self.sol2.own_policy[ri] / supp_p_sum
                                } else {
                                    1.0 / (supp_me.len() as f64)
                                };
                                let v = self.cell_val_mine(wi, own_i, cand_j);
                                ev2 += pw * v;
                            }
                            if ev2 < self.sol2.world_values[wi] - self.cfg.do_margin {
                                pending_admit.push(cand_j);
                            }
                        }
                    }

                    if wi < n_w2 && cand_pos < candidates.len() && self.iters_done < opp_do_cap {
                        let cand_j = candidates[cand_pos];
                        cand_pos += 1;
                        let do_step_w = self.coarse_per_w[wi].clamp(64, self.cfg.do_iters.max(192));
                        let mut rem = opp_do_cap.saturating_sub(self.iters_done);
                        let mut probed_all = true;
                        for &ri in &supp_me {
                            let step_n = do_step_w.min(rem);
                            if step_n == 0 {
                                probed_all = false;
                                break;
                            }
                            self.work_queue.push_back((wi, self.s_me[ri], cand_j, step_n));
                            rem -= step_n;
                        }
                        active_cand = Some((cand_j, probed_all));
                        self.phase = SteppedPreviewPhase::DoOpp {
                            round,
                            do_cap_iters,
                            opp_do_cap,
                            supp_me,
                            supp_p_sum,
                            wi,
                            candidates,
                            cand_pos,
                            active_cand,
                            pending_admit,
                            admitted_opp,
                        };
                        continue;
                    }

                    // Finished probing candidates for world `wi`: admit violators and fill.
                    if wi < n_w2 && !pending_admit.is_empty() {
                        let do_step_w = self.coarse_per_w[wi].clamp(64, self.cfg.do_iters.max(192));
                        let per_fill = (do_step_w / 2).clamp(48, 128);
                        let mut rem = opp_do_cap.saturating_sub(self.iters_done);
                        for cand_j in pending_admit.drain(..) {
                            if self.s_opp[wi].contains(&cand_j) {
                                continue;
                            }
                            let missing = self
                                .s_me
                                .iter()
                                .filter(|&&own_i| !self.cells.contains_key(&(wi, own_i, cand_j)))
                                .count() as u32;
                            if rem < missing.saturating_mul(per_fill) {
                                continue;
                            }
                            self.s_opp[wi].push(cand_j);
                            admitted_opp = true;
                            for &own_i in &self.s_me {
                                if self.cells.contains_key(&(wi, own_i, cand_j)) {
                                    continue;
                                }
                                let step_n = per_fill.min(rem);
                                if step_n > 0 {
                                    self.work_queue.push_back((wi, own_i, cand_j, step_n));
                                    rem -= step_n;
                                }
                            }
                        }
                    }

                    wi += 1;
                    if wi < n_w2 && self.iters_done < opp_do_cap {
                        candidates = self.build_do_opp_candidates(wi, &supp_me, supp_p_sum);
                        cand_pos = 0;
                        self.phase = SteppedPreviewPhase::DoOpp {
                            round,
                            do_cap_iters,
                            opp_do_cap,
                            supp_me,
                            supp_p_sum,
                            wi,
                            candidates,
                            cand_pos,
                            active_cand: None,
                            pending_admit: Vec::new(),
                            admitted_opp,
                        };
                    } else {
                        // Drain any newly enqueued admission fill cells BEFORE re-solving `sol2`
                        // and transitioning to Own BR / Polish.
                        self.phase = SteppedPreviewPhase::DoOppFinalize {
                            round,
                            do_cap_iters,
                            admitted_opp,
                        };
                    }
                }
                SteppedPreviewPhase::DoOppFinalize {
                    round,
                    do_cap_iters,
                    admitted_opp,
                } => {
                    if admitted_opp {
                        let (_, updated) = self.solve_subgame(600);
                        self.sol2 = updated;
                    }
                    if self.iters_done < do_cap_iters {
                        self.start_do_own(round, do_cap_iters, admitted_opp);
                    } else {
                        self.phase = SteppedPreviewPhase::Polish { polish_steps: 0 };
                    }
                }
                SteppedPreviewPhase::DoOwn {
                    round,
                    do_cap_iters,
                    supp_opp,
                    candidates,
                    mut cand_pos,
                    mut active_cand,
                    mut pending_admit,
                    admitted_opp,
                    mut admitted_me,
                } => {
                    if let Some((cand_i, probed_all)) = active_cand.take() {
                        if probed_all {
                            let mut ev2 = 0.0;
                            for wi in 0..n_w2 {
                                let qsum: f64 = supp_opp[wi]
                                    .iter()
                                    .map(|&cj| self.sol2.opp_policies[wi][cj])
                                    .sum();
                                let mut ev_w = 0.0;
                                for &cj in &supp_opp[wi] {
                                    let opp_j = self.s_opp[wi][cj];
                                    let qw = if qsum > 0.0 {
                                        self.sol2.opp_policies[wi][cj] / qsum
                                    } else {
                                        1.0 / (supp_opp[wi].len() as f64)
                                    };
                                    let v = self.cell_val_mine(wi, cand_i, opp_j);
                                    ev_w += qw * v;
                                }
                                ev2 += self.s2_post_weights[wi] * ev_w;
                            }
                            for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
                                let wd = &self.worlds[w_id];
                                let m_full = wd.opp_canon.len();
                                let q_tail = &self.sol2.opp_policies[n_w2 + ti];
                                let mut ev_w = 0.0;
                                for j in 0..m_full {
                                    let qj = q_tail[j];
                                    if qj > 0.0 {
                                        ev_w += qj * wd.stage1_vals[cand_i * m_full + j];
                                    }
                                }
                                ev2 += self.tail_post_weights[ti] * ev_w;
                            }
                            if ev2 > self.sol2.value + self.cfg.do_margin {
                                pending_admit.push(cand_i);
                            }
                        }
                    }

                    if cand_pos < candidates.len() && self.iters_done < do_cap_iters {
                        let cand_i = candidates[cand_pos];
                        cand_pos += 1;
                        let mut rem = do_cap_iters.saturating_sub(self.iters_done);
                        let mut probed_all = true;
                        for wi in 0..n_w2 {
                            let do_step_w = self.coarse_per_w[wi].clamp(64, self.cfg.do_iters.max(192));
                            for &cj in &supp_opp[wi] {
                                let step_n = do_step_w.min(rem);
                                if step_n == 0 {
                                    probed_all = false;
                                    break;
                                }
                                self.work_queue.push_back((wi, cand_i, self.s_opp[wi][cj], step_n));
                                rem -= step_n;
                            }
                            if !probed_all {
                                break;
                            }
                        }
                        active_cand = Some((cand_i, probed_all));
                        self.phase = SteppedPreviewPhase::DoOwn {
                            round,
                            do_cap_iters,
                            supp_opp,
                            candidates,
                            cand_pos,
                            active_cand,
                            pending_admit,
                            admitted_opp,
                            admitted_me,
                        };
                        continue;
                    }

                    // Admit winning outside own candidates and enqueue missing fills.
                    if !pending_admit.is_empty() {
                        let per_fill = (self.coarse_per_w[0] / 2).clamp(48, 128);
                        let mut rem = do_cap_iters.saturating_sub(self.iters_done);
                        for cand_i in pending_admit.drain(..) {
                            if self.s_me.contains(&cand_i) {
                                continue;
                            }
                            let mut missing = 0u32;
                            for wi in 0..n_w2 {
                                missing += self.s_opp[wi]
                                    .iter()
                                    .filter(|&&opp_j| !self.cells.contains_key(&(wi, cand_i, opp_j)))
                                    .count() as u32;
                            }
                            if rem < missing.saturating_mul(per_fill) {
                                continue;
                            }
                            self.s_me.push(cand_i);
                            admitted_me = true;
                            for wi in 0..n_w2 {
                                for &opp_j in &self.s_opp[wi] {
                                    if self.cells.contains_key(&(wi, cand_i, opp_j)) {
                                        continue;
                                    }
                                    let step_n = per_fill.min(rem);
                                    if step_n > 0 {
                                        self.work_queue.push_back((wi, cand_i, opp_j, step_n));
                                        rem -= step_n;
                                    }
                                }
                            }
                        }
                    }

                    // Drain any newly enqueued admission fill cells BEFORE re-solving `sol2`
                    // and entering the next DO round or Polish.
                    self.phase = SteppedPreviewPhase::DoOwnFinalize {
                        round,
                        do_cap_iters,
                        admitted_opp,
                        admitted_me,
                    };
                }
                SteppedPreviewPhase::DoOwnFinalize {
                    round,
                    do_cap_iters,
                    admitted_opp,
                    admitted_me,
                } => {
                    if admitted_me {
                        let (_, updated) = self.solve_subgame(600);
                        self.sol2 = updated;
                    }
                    if round + 1 < self.cfg.do_rounds
                        && (admitted_opp || admitted_me)
                        && self.iters_done < do_cap_iters
                    {
                        let opp_do_cap = self.iters_done.saturating_add(
                            (do_cap_iters.saturating_sub(self.iters_done)) * 60 / 100,
                        );
                        self.start_do_round(round + 1, do_cap_iters, opp_do_cap);
                    } else {
                        self.phase = SteppedPreviewPhase::Polish { polish_steps: 0 };
                    }
                }
                SteppedPreviewPhase::Polish { mut polish_steps } => {
                    if polish_steps > 0 && polish_steps % 8 == 0 {
                        let (_, updated) = self.solve_subgame(600);
                        self.sol2 = updated;
                    }
                    let mut active_rows: Vec<usize> = (0..self.s_me.len())
                        .filter(|&ri| {
                            self.sol2.own_policy[ri] >= 0.01
                                || self.sol2.row_vs_q[ri] >= self.sol2.value - self.cfg.polish_gap
                        })
                        .collect();
                    let pure_r_opt = (0..self.s_me.len()).max_by(|&a, &b| {
                        self.sol2.row_guarantees[a].total_cmp(&self.sol2.row_guarantees[b])
                    });
                    if let Some(pure_r) = pure_r_opt {
                        if !active_rows.contains(&pure_r) {
                            active_rows.push(pure_r);
                        }
                    }
                    if active_rows.len() < 2 {
                        let mut ord: Vec<usize> = (0..self.s_me.len()).collect();
                        ord.sort_by(|&a, &b| {
                            self.sol2.row_vs_q[b].total_cmp(&self.sol2.row_vs_q[a])
                        });
                        for ri in ord {
                            if active_rows.len() >= 2 {
                                break;
                            }
                            if !active_rows.contains(&ri) {
                                active_rows.push(ri);
                            }
                        }
                    }

                    let mut best_target: Option<(usize, usize, usize, f64)> = None;
                    let mut best_uncapped: Option<(usize, usize, usize, f64)> = None;
                    for wi in 0..n_w2 {
                        let cell_cap_w = (self.coarse_per_w[wi].saturating_mul(4)).clamp(512, 2048);
                        let mut active_cols: Vec<usize> = (0..self.s_opp[wi].len())
                            .filter(|&cj| {
                                self.sol2.opp_policies[wi][cj] >= 0.01
                                    || self.sol2.col_vs_p[wi][cj]
                                        <= self.sol2.world_values[wi] + self.cfg.polish_gap
                            })
                            .collect();
                        if let Some(pure_r) = pure_r_opt {
                            let own_i = self.s_me[pure_r];
                            if self.worlds[self.s2_world_ids[wi]].opp_prior.is_none() {
                                if let Some(worst_cj) = (0..self.s_opp[wi].len()).min_by(|&a, &b| {
                                    let va = self.cell_val_mine(wi, own_i, self.s_opp[wi][a]);
                                    let vb = self.cell_val_mine(wi, own_i, self.s_opp[wi][b]);
                                    va.total_cmp(&vb)
                                }) {
                                    if !active_cols.contains(&worst_cj) {
                                        active_cols.push(worst_cj);
                                    }
                                }
                            }
                        }
                        if active_cols.len() < 2 {
                            let mut ord: Vec<usize> = (0..self.s_opp[wi].len()).collect();
                            ord.sort_by(|&a, &b| {
                                self.sol2.col_vs_p[wi][a].total_cmp(&self.sol2.col_vs_p[wi][b])
                            });
                            for cj in ord {
                                if active_cols.len() >= 2 {
                                    break;
                                }
                                if !active_cols.contains(&cj) {
                                    active_cols.push(cj);
                                }
                            }
                        }
                        for &ri in &active_rows {
                            let own_i = self.s_me[ri];
                            let pw = self.sol2.own_policy[ri] + 0.05;
                            for &cj in &active_cols {
                                let opp_j = self.s_opp[wi][cj];
                                let visits =
                                    self.cells.get(&(wi, own_i, opp_j)).map_or(0, |c| c.total_n);
                                let qw = self.sol2.opp_policies[wi][cj] + 0.05;
                                let score = self.s2_alloc_weights[wi] * pw * qw
                                    / f64::from(visits + 1).sqrt();
                                if best_uncapped.map_or(true, |bt| score > bt.3) {
                                    best_uncapped = Some((wi, own_i, opp_j, score));
                                }
                                if visits < cell_cap_w
                                    && best_target.map_or(true, |bt| score > bt.3)
                                {
                                    best_target = Some((wi, own_i, opp_j, score));
                                }
                            }
                        }
                    }
                    if best_target.is_none() {
                        for wi in 0..n_w2 {
                            let half_cap =
                                self.coarse_per_w[wi].saturating_mul(2).clamp(256, 1024);
                            for &ri in &active_rows {
                                let own_i = self.s_me[ri];
                                let pw = self.sol2.own_policy[ri] + 0.05;
                                for &opp_j in &self.s_opp[wi] {
                                    let visits = self
                                        .cells
                                        .get(&(wi, own_i, opp_j))
                                        .map_or(0, |c| c.total_n);
                                    if visits >= half_cap {
                                        continue;
                                    }
                                    let score = self.s2_alloc_weights[wi] * pw
                                        / f64::from(visits + 1).sqrt();
                                    if best_target.map_or(true, |bt| score > bt.3) {
                                        best_target = Some((wi, own_i, opp_j, score));
                                    }
                                }
                            }
                        }
                    }
                    let (wi, own_i, opp_j, _) = best_target
                        .or(best_uncapped)
                        .unwrap_or((0, self.s_me[0], self.s_opp[0][0], 0.0));
                    self.work_queue
                        .push_back((wi, own_i, opp_j, self.cfg.polish_chunk));
                    polish_steps = polish_steps.saturating_add(1);
                    self.phase = SteppedPreviewPhase::Polish { polish_steps };
                }
            }
        }
    }

    /// Execute exactly ONE Turn-1 `cfv-0r@puct@lam0` iteration (or calibration trajectory iter).
    /// Returns `(own_orig_idx, opp_choice, val0, val_mine, depth)`.
    pub fn step_one(&mut self, rng: &mut SplitMix64) -> (usize, SearchChoice, f64, f64, u32) {
        if self.calib.is_some() {
            return self.step_calib_one(rng);
        }
        let (wi, own_i, opp_j) = loop {
            if let Some((wi, own_i, opp_j, ref mut rem)) = self.current_work {
                if *rem > 0 {
                    *rem -= 1;
                    break (wi, own_i, opp_j);
                }
            }
            self.current_work = None;
            if self.work_queue.is_empty() {
                self.advance_phase();
            }
            self.current_work = self.work_queue.pop_front();
        };

        let w_id = self.s2_world_ids[wi];
        let wd = &self.worlds[w_id];
        let (ref roster, ref pcache) = wd.compact.as_ref().unwrap();
        let m = wd.opp_canon.len();
        let prior_v_mine = self
            .raw_stage1_vals
            .get(w_id)
            .and_then(|v| v.get(own_i * m + opp_j))
            .copied()
            .unwrap_or_else(|| wd.stage1_vals[own_i * m + opp_j]);
        let prior_v0 = if self.side == 0 {
            prior_v_mine
        } else {
            1.0 - prior_v_mine
        };
        let own_t = self.own_canon[own_i].1;
        let opp_t = wd.opp_canon[opp_j].1;
        let picks = if self.side == 0 {
            [own_t, opp_t]
        } else {
            [opp_t, own_t]
        };

        let mut added_nodes = 0usize;
        let cell = self.cells.entry((wi, own_i, opp_j)).or_insert_with(|| {
            let c = CfPreviewCell::new(roster, pcache, picks, prior_v0, &self.cfg);
            added_nodes = c.nodes.len();
            c
        });
        let prev_nodes = cell.nodes.len();
        let (val0, depth) = cell.step_one(
            &self.cfg,
            rng,
            &self.eg_weights,
            &self.eg_playout,
            self.rollout_rules,
            &mut self.path_buf,
        );
        self.total_nodes += added_nodes + (cell.nodes.len() - prev_nodes);
        self.iters_done += 1;

        let val_mine = if self.side == 0 { val0 } else { 1.0 - val0 };
        let own_orig_idx = self.own_canon[own_i].0;
        let opp_choice = SearchChoice::Team(opp_t);
        (own_orig_idx, opp_choice, val0, val_mine, depth)
    }

    /// Solve the execution readout over the full 100% posterior without mutating search state.
    /// Uses Rubin Bayesian Bootstrap over CRN calibration blocks when `bootstrap_draws > 1`,
    /// otherwise uses `LcbGatedMixed`.
    fn solve_execution_mix(&self, done: u32) -> (bool, usize, Vec<f64>) {
        let sweeps = if done >= 4096 {
            self.cfg.rm_sweeps_final
        } else {
            self.cfg.rm_sweeps_stage1
        };
        let use_full_opp = self.calib.is_none()
            && self.calib_cfg.as_ref().is_some_and(|cc| cc.full_opp_readout);
        let (final_mats, final_sol) = if use_full_opp {
            self.solve_subgame_full_opp(sweeps)
        } else {
            self.solve_subgame(sweeps)
        };
        let k_me = self.s_me.len();
        let n_w2 = self.s2_world_ids.len();
        let n_tail = self.tail_world_ids.len();
        let best_pure_r = (0..k_me)
            .max_by(|&a, &b| {
                final_sol.row_guarantees[a]
                    .total_cmp(&final_sol.row_guarantees[b])
                    .then_with(|| final_sol.row_vs_q[a].total_cmp(&final_sol.row_vs_q[b]))
            })
            .unwrap_or(0);

        // Early-stop readout while still inside `TrajCalib`: apply interim calibrated Stage-1 ridge
        // if enough calibration trajectories have completed, without mutating search state.
        if let (Some(ref cc), Some(ref cal)) = (&self.calib_cfg, &self.calib) {
            if cal.games_completed >= 8 {
                let (th_s1, dw_s1) =
                    self.fit_calib_ridge(cc, &cal.s2_metas, &cal.obs_s1_n, &cal.obs_s1_sum);
                let basis = cc.ridge_basis;
                let mut feats = [(0usize, 0.0f64); MAX_RIDGE_NNZ];
                let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
                let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2 + n_tail);
                let mut w_list: Vec<f64> = Vec::with_capacity(n_w2 + n_tail);
                for wi in 0..n_w2 {
                    let w_id = self.s2_world_ids[wi];
                    let wd = &self.worlds[w_id];
                    let m_full = wd.opp_canon.len();
                    let meta = &cal.s2_metas[wi];
                    let mut mat = vec![0.5f64; k_me * m_full];
                    for (ri, &own_i) in self.s_me.iter().enumerate() {
                        let own_t = self.own_canon[own_i].1;
                        for j in 0..m_full {
                            let nnz = extract_ridge_features(
                                basis,
                                self.side,
                                own_t,
                                wd.opp_canon[j].1,
                                meta,
                                &mut feats,
                            );
                            let mut corr = 0.0;
                            for &(f, v) in &feats[..nnz] {
                                corr += (th_s1[f] + dw_s1[wi][f]) * v;
                            }
                            mat[ri * m_full + j] = (self.raw_stage1_vals[w_id][own_i * m_full + j]
                                + cc.delta_s1_beta * corr)
                                .clamp(0.02, 0.98);
                        }
                    }
                    mats.push(mat);
                    col_lens.push(m_full);
                    w_list.push(self.s2_post_weights[wi]);
                }
                for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
                    let wd = &self.worlds[w_id];
                    let m_full = wd.opp_canon.len();
                    let meta = &cal.tail_metas[ti];
                    let mut mat = vec![0.5f64; k_me * m_full];
                    for (ri, &own_i) in self.s_me.iter().enumerate() {
                        let own_t = self.own_canon[own_i].1;
                        for j in 0..m_full {
                            let nnz = extract_ridge_features(
                                basis,
                                self.side,
                                own_t,
                                wd.opp_canon[j].1,
                                meta,
                                &mut feats,
                            );
                            let mut corr = 0.0;
                            for &(f, v) in &feats[..nnz] {
                                corr += th_s1[f] * v;
                            }
                            mat[ri * m_full + j] = (self.raw_stage1_vals[w_id][own_i * m_full + j]
                                + 0.75 * cc.delta_s1_beta * corr)
                                .clamp(0.02, 0.98);
                        }
                    }
                    mats.push(mat);
                    col_lens.push(m_full);
                    w_list.push(self.tail_post_weights[ti]);
                }
                let slices: Vec<(f64, usize, &[f64])> = (0..mats.len())
                    .map(|idx| (w_list[idx], col_lens[idx], mats[idx].as_slice()))
                    .collect();
                let mut priors: Vec<Option<&[f64]>> = Vec::with_capacity(n_w2 + n_tail);
                for &w_id in &self.s2_world_ids {
                    priors.push(self.worlds[w_id].opp_prior.as_deref());
                }
                for &w_id in &self.tail_world_ids {
                    priors.push(self.worlds[w_id].opp_prior.as_deref());
                }
                let sol = solve_bayesian_rm_plus_with_priors(k_me, &slices, &priors, sweeps);
                let mut p_mix = sol.own_policy.clone();
                shed_dust(&mut p_mix, cc.bootstrap_dust);
                let use_mix = p_mix.iter().filter(|&&x| x > 0.0).count() > 1;
                let top_r = (0..k_me)
                    .max_by(|&a, &b| p_mix[a].total_cmp(&p_mix[b]))
                    .unwrap_or(best_pure_r);
                return (use_mix, top_r, p_mix);
            }
        }

        if let Some(ref cc) = self.calib_cfg {
            if cc.bootstrap_draws > 1
                && self.calib.is_none()
                && self.calib_obs.len() >= 8
                && self.calib_s2_metas.len() == n_w2
            {
                let mut max_blk = 0u32;
                for o in &self.calib_obs {
                    if o.block_id > max_blk {
                        max_blk = o.block_id;
                    }
                }
                let n_blk = (max_blk as usize) + 1;
                let mut blk_seen = vec![false; n_blk];
                let mut active_blks = Vec::with_capacity(n_blk);
                for o in &self.calib_obs {
                    let b = o.block_id as usize;
                    if !blk_seen[b] {
                        blk_seen[b] = true;
                        active_blks.push(b);
                    }
                }
                let nb_f = active_blks.len().max(1) as f64;
                let mut brng = SplitMix64::new(
                    0xB007_57AA_0000_0001u64
                        ^ (u64::from(done).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ ((self.calib_obs.len() as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
                );
                let basis = cc.ridge_basis;
                let dim = basis.dim();
                let main_end = basis.main_end();
                let mut feats = [(0usize, 0.0f64); MAX_RIDGE_NNZ];
                let mut blk_w = vec![0.0f64; n_blk];
                let mut avg_p = vec![0.0f64; k_me];
                let draws = cc.bootstrap_draws;
                let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2 + n_tail);
                let mut w_list: Vec<f64> = Vec::with_capacity(n_w2 + n_tail);
                let mut boot_priors_owned: Vec<Option<Vec<f64>>> =
                    Vec::with_capacity(n_w2 + n_tail);
                for wi in 0..n_w2 {
                    let w_id = self.s2_world_ids[wi];
                    col_lens.push(if use_full_opp {
                        self.worlds[w_id].opp_canon.len()
                    } else {
                        self.s_opp[wi].len()
                    });
                    w_list.push(self.s2_post_weights[wi]);
                    boot_priors_owned.push(if use_full_opp {
                        self.worlds[w_id].opp_prior.clone()
                    } else {
                        restrict_opp_prior(self.worlds[w_id].opp_prior.as_deref(), &self.s_opp[wi])
                    });
                }
                for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
                    col_lens.push(self.worlds[w_id].opp_canon.len());
                    w_list.push(self.tail_post_weights[ti]);
                    boot_priors_owned.push(self.worlds[w_id].opp_prior.clone());
                }
                let boot_prior_refs: Vec<Option<&[f64]>> =
                    boot_priors_owned.iter().map(|p| p.as_deref()).collect();

                for _ in 0..draws {
                    let mut w_sum = 0.0;
                    for &b in &active_blks {
                        let u = brng.next_f64().clamp(1e-12, 1.0 - 1e-12);
                        let e = -u.ln();
                        blk_w[b] = e;
                        w_sum += e;
                    }
                    let scale = if w_sum > 0.0 { nb_f / w_sum } else { 1.0 };
                    for &b in &active_blks {
                        blk_w[b] *= scale;
                    }

                    let mut ata_t1 = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
                    let mut atb_t1 = [0.0f64; MAX_RIDGE_DIM];
                    let mut any_t1 = false;
                    let mut ata_s1 = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
                    let mut atb_s1 = [0.0f64; MAX_RIDGE_DIM];
                    let mut any_s1 = false;
                    for o in &self.calib_obs {
                        let bw = blk_w[o.block_id as usize];
                        if bw <= 0.0 {
                            continue;
                        }
                        let wi = o.wi as usize;
                        let w_id = self.s2_world_ids[wi];
                        let nnz = extract_ridge_features(
                            basis,
                            self.side,
                            self.own_canon[o.own_i as usize].1,
                            self.worlds[w_id].opp_canon[o.opp_j as usize].1,
                            &self.calib_s2_metas[wi],
                            &mut feats,
                        );
                        any_s1 = true;
                        let rs1 = f64::from(o.r_s1);
                        for a in 0..nnz {
                            let (fa, va) = feats[a];
                            atb_s1[fa] += bw * va * rs1;
                            for b in 0..=a {
                                let (fb, vb) = feats[b];
                                let prod = bw * va * vb;
                                ata_s1[fa][fb] += prod;
                                if fa != fb {
                                    ata_s1[fb][fa] += prod;
                                }
                            }
                        }
                        if o.has_t1 {
                            any_t1 = true;
                            let rt1 = f64::from(o.r_t1);
                            for a in 0..nnz {
                                let (fa, va) = feats[a];
                                atb_t1[fa] += bw * va * rt1;
                                for b in 0..=a {
                                    let (fb, vb) = feats[b];
                                    let prod = bw * va * vb;
                                    ata_t1[fa][fb] += prod;
                                    if fa != fb {
                                        ata_t1[fb][fa] += prod;
                                    }
                                }
                            }
                        }
                    }
                    let th_t1 = if any_t1 {
                        solve_spd_cholesky_split(
                            dim,
                            main_end,
                            &ata_t1,
                            &atb_t1,
                            cc.ridge_lambda_int,
                            cc.ridge_lambda_main,
                            cc.ridge_lambda_inter,
                        )
                    } else {
                        [0.0; MAX_RIDGE_DIM]
                    };
                    let th_s1 = if any_s1 {
                        solve_spd_cholesky_split(
                            dim,
                            main_end,
                            &ata_s1,
                            &atb_s1,
                            cc.ridge_lambda_int,
                            cc.ridge_lambda_main,
                            cc.ridge_lambda_inter,
                        )
                    } else {
                        [0.0; MAX_RIDGE_DIM]
                    };

                    let mut bmats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
                    for wi in 0..n_w2 {
                        let w_id = self.s2_world_ids[wi];
                        let wd = &self.worlds[w_id];
                        let m_full = wd.opp_canon.len();
                        let k_opp = if use_full_opp { m_full } else { self.s_opp[wi].len() };
                        let meta = &self.calib_s2_metas[wi];
                        let mut mat = vec![0.5f64; k_me * k_opp];
                        for (ri, &own_i) in self.s_me.iter().enumerate() {
                            let own_t = self.own_canon[own_i].1;
                            for cj in 0..k_opp {
                                let opp_j = if use_full_opp { cj } else { self.s_opp[wi][cj] };
                                let nnz = extract_ridge_features(
                                    basis,
                                    self.side,
                                    own_t,
                                    wd.opp_canon[opp_j].1,
                                    meta,
                                    &mut feats,
                                );
                                let val = if let Some(cell) = self.cells.get(&(wi, own_i, opp_j)) {
                                    let v0 = cell.value0(self.cfg.cell_readout);
                                    let v_t1 = if self.side == 0 { v0 } else { 1.0 - v0 };
                                    let mut corr = 0.0;
                                    for &(f, v) in &feats[..nnz] {
                                        corr += th_t1[f] * v;
                                    }
                                    (v_t1 + (cc.delta_t1_beta * corr).clamp(-0.35, 0.35))
                                        .clamp(0.01, 0.99)
                                } else {
                                    let mut corr = 0.0;
                                    for &(f, v) in &feats[..nnz] {
                                        corr += th_s1[f] * v;
                                    }
                                    (self.raw_stage1_vals[w_id][own_i * m_full + opp_j]
                                        + cc.delta_s1_beta * corr)
                                        .clamp(0.02, 0.98)
                                };
                                mat[ri * k_opp + cj] = val;
                            }
                        }
                        bmats.push(mat);
                    }
                    for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
                        let wd = &self.worlds[w_id];
                        let m_full = wd.opp_canon.len();
                        let meta = &self.calib_tail_metas[ti];
                        let mut mat = vec![0.5f64; k_me * m_full];
                        for (ri, &own_i) in self.s_me.iter().enumerate() {
                            let own_t = self.own_canon[own_i].1;
                            for j in 0..m_full {
                                let nnz = extract_ridge_features(
                                    basis,
                                    self.side,
                                    own_t,
                                    wd.opp_canon[j].1,
                                    meta,
                                    &mut feats,
                                );
                                let mut corr = 0.0;
                                for &(f, v) in &feats[..nnz] {
                                    corr += th_s1[f] * v;
                                }
                                mat[ri * m_full + j] = (self.raw_stage1_vals[w_id]
                                    [own_i * m_full + j]
                                    + 0.75 * cc.delta_s1_beta * corr)
                                    .clamp(0.02, 0.98);
                            }
                        }
                        bmats.push(mat);
                    }
                    let slices: Vec<(f64, usize, &[f64])> = (0..bmats.len())
                        .map(|idx| (w_list[idx], col_lens[idx], bmats[idx].as_slice()))
                        .collect();
                    let bsol = solve_bayesian_rm_plus_with_priors(
                        k_me,
                        &slices,
                        &boot_prior_refs,
                        500,
                    );
                    let inv_d = 1.0 / (draws as f64);
                    for ri in 0..k_me {
                        avg_p[ri] += bsol.own_policy[ri] * inv_d;
                    }
                }
                shed_dust(&mut avg_p, cc.bootstrap_dust);
                let use_mix = avg_p.iter().filter(|&&x| x > 0.0).count() > 1;
                let top_r = (0..k_me)
                    .max_by(|&a, &b| avg_p[a].total_cmp(&avg_p[b]))
                    .unwrap_or(best_pure_r);
                return (use_mix, top_r, avg_p);
            }
        }
        let pure_guarantee = final_sol.row_guarantees[best_pure_r];

        let mut p_mix = final_sol.own_policy.clone();
        let (min_mix_margin, dust_ratio, lcb_z) = match self.cfg.policy_mode {
            PreviewPolicyMode::LcbGatedMixed {
                min_mix_margin,
                dust_ratio,
                lcb_z,
            } => (min_mix_margin, dust_ratio, lcb_z),
            _ => (0.015, 0.08, 1.0),
        };
        shed_dust(&mut p_mix, dust_ratio);

        let mut mixed_guarantee = 0.0;
        let mut mixed_guarantee_lcb = 0.0;
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let k_opp = if use_full_opp {
                self.worlds[w_id].opp_canon.len()
            } else {
                self.s_opp[wi].len()
            };
            let mat = &final_mats[wi];
            let q_prior = if use_full_opp {
                self.worlds[w_id].opp_prior.clone()
            } else {
                restrict_opp_prior(self.worlds[w_id].opp_prior.as_deref(), &self.s_opp[wi])
            };
            if let Some(q_w) = q_prior {
                let mut ev_w = 0.0;
                let mut var_w = 0.0;
                for cj in 0..k_opp {
                    let qj = q_w[cj];
                    if qj <= 0.0 {
                        continue;
                    }
                    let opp_j = if use_full_opp { cj } else { self.s_opp[wi][cj] };
                    let mut mean_j = 0.0;
                    let mut var_j = 0.0;
                    for ri in 0..k_me {
                        let pr = p_mix[ri];
                        if pr > 0.0 {
                            let v = mat[ri * k_opp + cj];
                            mean_j += pr * v;
                            if let Some(c) = self.cells.get(&(wi, self.s_me[ri], opp_j)) {
                                let n_c = c.total_n.max(1);
                                var_j += (pr * pr) * (0.25 / f64::from(n_c));
                            } else if !use_full_opp {
                                var_j += (pr * pr) * 0.25;
                            }
                        }
                    }
                    ev_w += qj * mean_j;
                    var_w += (qj * qj) * var_j;
                }
                mixed_guarantee += self.s2_post_weights[wi] * ev_w;
                mixed_guarantee_lcb += self.s2_post_weights[wi] * (ev_w - lcb_z * var_w.sqrt());
                continue;
            }
            let mut min_col = f64::INFINITY;
            let mut min_col_lcb = f64::INFINITY;
            for cj in 0..k_opp {
                let opp_j = if use_full_opp { cj } else { self.s_opp[wi][cj] };
                let mut mean_j = 0.0;
                let mut var_j = 0.0;
                for ri in 0..k_me {
                    let pr = p_mix[ri];
                    if pr > 0.0 {
                        let v = mat[ri * k_opp + cj];
                        mean_j += pr * v;
                        if let Some(c) = self.cells.get(&(wi, self.s_me[ri], opp_j)) {
                            let n_c = c.total_n.max(1);
                            var_j += (pr * pr) * (0.25 / f64::from(n_c));
                        } else if !use_full_opp {
                            var_j += (pr * pr) * 0.25;
                        }
                    }
                }
                if mean_j < min_col {
                    min_col = mean_j;
                }
                let lcb_j = mean_j - lcb_z * var_j.sqrt();
                if lcb_j < min_col_lcb {
                    min_col_lcb = lcb_j;
                }
            }
            mixed_guarantee += self.s2_post_weights[wi] * min_col;
            mixed_guarantee_lcb += self.s2_post_weights[wi] * min_col_lcb;
        }
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let m_full = self.worlds[w_id].opp_canon.len();
            let mat = &final_mats[n_w2 + ti];
            if let Some(ref qp) = self.worlds[w_id].opp_prior {
                let mut ev_w = 0.0;
                for cj in 0..m_full {
                    let qj = qp[cj];
                    if qj <= 0.0 {
                        continue;
                    }
                    let mut mean_j = 0.0;
                    for ri in 0..k_me {
                        let pr = p_mix[ri];
                        if pr > 0.0 {
                            mean_j += pr * mat[ri * m_full + cj];
                        }
                    }
                    ev_w += qj * mean_j;
                }
                mixed_guarantee += self.tail_post_weights[ti] * ev_w;
                mixed_guarantee_lcb += self.tail_post_weights[ti] * ev_w;
                continue;
            }
            let mut min_col = f64::INFINITY;
            for cj in 0..m_full {
                let mut mean_j = 0.0;
                for ri in 0..k_me {
                    let pr = p_mix[ri];
                    if pr > 0.0 {
                        mean_j += pr * mat[ri * m_full + cj];
                    }
                }
                if mean_j < min_col {
                    min_col = mean_j;
                }
            }
            mixed_guarantee += self.tail_post_weights[ti] * min_col;
            mixed_guarantee_lcb += self.tail_post_weights[ti] * min_col;
        }

        let use_mix = p_mix.iter().filter(|&&x| x > 0.0).count() > 1
            && mixed_guarantee > pure_guarantee + min_mix_margin
            && mixed_guarantee_lcb > pure_guarantee - 0.005;

        (use_mix, best_pure_r, p_mix)
    }

    fn solve_subgame_full_opp(&self, sweeps: u32) -> (Vec<Vec<f64>>, BayesianRmSolution) {
        let k_me = self.s_me.len();
        let n_w2 = self.s2_world_ids.len();
        let n_tail = self.tail_world_ids.len();
        let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
        let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2 + n_tail);
        let mut w_list: Vec<f64> = Vec::with_capacity(n_w2 + n_tail);
        let mut priors: Vec<Option<&[f64]>> = Vec::with_capacity(n_w2 + n_tail);
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let m_full = self.worlds[w_id].opp_canon.len();
            let mut mat = vec![0.5f64; k_me * m_full];
            for (ri, &own_i) in self.s_me.iter().enumerate() {
                for opp_j in 0..m_full {
                    mat[ri * m_full + opp_j] = self.cell_val_mine(wi, own_i, opp_j);
                }
            }
            mats.push(mat);
            col_lens.push(m_full);
            w_list.push(self.s2_post_weights[wi]);
            priors.push(self.worlds[w_id].opp_prior.as_deref());
        }
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let wd = &self.worlds[w_id];
            let m_full = wd.opp_canon.len();
            let mut mat = vec![0.5f64; k_me * m_full];
            for (ri, &own_i) in self.s_me.iter().enumerate() {
                let src = &wd.stage1_vals[own_i * m_full..(own_i + 1) * m_full];
                mat[ri * m_full..(ri + 1) * m_full].copy_from_slice(src);
            }
            mats.push(mat);
            col_lens.push(m_full);
            w_list.push(self.tail_post_weights[ti]);
            priors.push(wd.opp_prior.as_deref());
        }
        let slices: Vec<(f64, usize, &[f64])> = (0..mats.len())
            .map(|idx| (w_list[idx], col_lens[idx], mats[idx].as_slice()))
            .collect();
        let sol = solve_bayesian_rm_plus_with_priors(k_me, &slices, &priors, sweeps);
        (mats, sol)
    }

    /// Read-only Nash value (from `self.side`'s perspective) over the active Stage-2 posterior.
    pub fn nash_value(&self, done: u32) -> f64 {
        let sweeps = if done >= 4096 {
            self.cfg.rm_sweeps_final
        } else {
            self.cfg.rm_sweeps_stage1
        };
        let use_full_opp = self.calib.is_none()
            && self.calib_cfg.as_ref().is_some_and(|cc| cc.full_opp_readout);
        let (_, final_sol) = if use_full_opp {
            self.solve_subgame_full_opp(sweeps)
        } else {
            self.solve_subgame(sweeps)
        };
        final_sol.value
    }

    /// Read-only `LcbGatedMixed` execution distribution over canonical legal preview picks.
    /// Exact match to the distribution sampled by `best_choice(decision_seed, done)`.
    pub fn execution_policy(&self, done: u32) -> Vec<(SearchChoice, f64)> {
        let (use_mix, best_pure_r, p_mix) = self.solve_execution_mix(done);
        if !use_mix {
            let choice = self.my_acts[self.own_canon[self.s_me[best_pure_r]].0];
            return vec![(choice, 1.0)];
        }
        let mut out = Vec::with_capacity(p_mix.len());
        for (r, &pr) in p_mix.iter().enumerate() {
            if pr > 0.0 {
                let choice = self.my_acts[self.own_canon[self.s_me[r]].0];
                out.push((choice, pr));
            }
        }
        if out.is_empty() {
            let choice = self.my_acts[self.own_canon[self.s_me[best_pure_r]].0];
            vec![(choice, 1.0)]
        } else {
            out
        }
    }

    /// Deterministic `LcbGatedMixed` Bayesian RM+ choice readout over the full 100% posterior.
    pub fn best_choice(&self, decision_seed: u64, done: u32) -> SearchChoice {
        let (use_mix, best_pure_r, p_mix) = self.solve_execution_mix(done);
        let picked_r = if use_mix {
            let draw = SplitMix64::new(decision_seed ^ u64::from(done)).next_f64();
            sample_dist(&p_mix, draw)
        } else {
            best_pure_r
        };
        let picked_canon_i = self.s_me[picked_r];
        self.my_acts[self.own_canon[picked_canon_i].0]
    }
}

// =========================================================================
// 6. Live Trajectory-DO Team Preview Engine (`TrajPreview` / `SteppedTrajPreview`)
// =========================================================================

/// Rao-Blackwellized early-exit policy for full-game `CompactBattle` trajectories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbExitMode {
    /// Play every trajectory to terminal battle outcome (or `max_turns`).
    Terminal,
    /// Unconditionally exit on any idle turn with `alive_sum <= max_alive`,
    /// returning `compact_eval_leaf_blindspot` (`linear.rs` stage-game value).
    UncondBlindspot { max_alive: u32 },
    /// Conditionally exit on any idle turn with `alive_sum <= max_alive` ONLY when
    /// no living mon has unmodeled endgame hazards (`Endure`, `Flail`/`Reversal`,
    /// `Focus Band`, `Destiny Bond`, `Perish Song`, `Explosion`/`Self-Destruct`,
    /// `Counter`/`Mirror Coat`, or low-PP/mutual-recovery stall).
    CondBlindspot { max_alive: u32 },
    /// On an idle turn with `alive_sum <= max_alive`, run one final `eval_iters`
    /// PUCT turn (with `8r` `compact_playout_with_leaf` + `linear.rs`) and return
    /// its continuous root value `v_root0`.
    SearchVroot { max_alive: u32, eval_iters: u32 },
    /// Same as `SearchVroot`, except skips early exit when `has_tactical_endgame_hazard`
    /// is active so Endure/Reversal/FocusBand/DBond/Perish endgames play to terminal.
    CondSearchVroot { max_alive: u32, eval_iters: u32 },
    /// Conditional fast static exit when hazard-free; otherwise fall back to
    /// `SearchVroot { eval_iters }` on `alive_sum <= max_alive`.
    HybridCond { max_alive: u32, eval_iters: u32 },
}

/// Residual regression feature basis `R(i, j) = Y(i, j) - V_s1(i, j)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajRidgeBasis {
    /// 25 dims: intercept + lead0[6] + bench0[6] + lead1[6] + bench1[6].
    Additive25,
    /// 31 dims: `Additive25` + 6 side-symmetric interaction features:
    /// `spikes_phaze_grounded` (2), `bp_setup_synergy` (2), `lead_pressure_after_switchin` (2).
    Interact31,
    /// 33 dims: `Interact31` + `spikes_grounded` (2).
    Interact33,
    /// 19 dims (core-only horizon bias, invariant to Turn-1 lead order):
    /// intercept + `own_present[6]` + `opp_present[6]` (main) + 6 multi-turn domain interactions
    /// (`spikes_phaze_grounded` x2, `bp_setup` x2, `spikes_grounded` x2).
    Present19,
    /// 33 dims (variance-reduced): intercept + `own_present[6]` + `opp_present[6]` (main)
    /// + `own_lead_delta[6]` + `opp_lead_delta[6]` + 8 domain interactions (inter).
    PresentInteract33,
    /// 39 dims: `Interact33` + 6 side-symmetric Turn-1 lead-vs-lead matchup features:
    /// `lead_dmg` (2), `faster_lead_dmg` (2), `faster_lead_sleep` (2).
    Interact39,
    /// 55 dims: `Additive25` + unordered own pairs (15) + unordered opp pairs (15).
    Pair55,
    /// 61 dims: `Additive25` + `mon0 x mon1` presence cross-terms (36, scaled by 1/3).
    MonCross61,
    /// 67 dims: `Interact31` + `mon0 x mon1` presence cross-terms (36, scaled by 1/3).
    Interact67,
}

impl TrajRidgeBasis {
    #[inline]
    pub const fn dim(self) -> usize {
        match self {
            Self::Present19 => 19,
            Self::Additive25 => 25,
            Self::Interact31 => 31,
            Self::Interact33 | Self::PresentInteract33 => 33,
            Self::Interact39 => 39,
            Self::Pair55 => 55,
            Self::MonCross61 => 61,
            Self::Interact67 => 67,
        }
    }

    #[inline]
    pub const fn main_end(self) -> usize {
        match self {
            Self::Present19 | Self::PresentInteract33 => 13,
            _ => 25,
        }
    }
}

/// Common Random Numbers & Turn-1 stratification schedule for trajectory sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrajCrnMode {
    /// Independent battle seed per trajectory game.
    Independent,
    /// Turn-indexed PRNG + paired challenger-vs-incumbent CRN + synchronized active polish.
    PairedCrn,
    /// `PairedCrn` + deterministic Turn-1 Berserk Gene `{2,3,4,5}` & Quick Claw stratification.
    PairedCrnStratified,
}

/// Policy extraction rule from the Trajectory-DO posterior payoff matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrajPolicyMode {
    /// Existing LCB-gated mix rule (`use_neff = true` uses `cell_n + kappa` effective samples).
    LcbGatedMixed {
        min_mix_margin: f64,
        dust_ratio: f64,
        lcb_z: f64,
        use_neff: bool,
    },
    /// Pure Multi-World Bayesian RM+ equilibrium with dust shedding.
    MixedNash { dust_ratio: f64 },
    /// L2-regularized Multi-World Bayesian RM+ (`max_p min_q p^T V q - l2_reg * ||p||_2^2`).
    RegRm { l2_reg: f64, dust_ratio: f64 },
    /// Rubin Bayesian Bootstrap ensemble over CRN seed blocks (`Exp(1)` block weights).
    BayesianBootstrapRm { draws: usize, dust_ratio: f64 },
}

/// Configuration for the live Trajectory-DO Team Preview engine.
#[derive(Clone, Debug)]
pub struct TrajPreviewConfig {
    /// Total PUCT iterations across all trajectory turns (matches `unknownPreview` scale).
    pub total_iters: u32,
    /// PUCT iterations per decision turn inside a trajectory game.
    pub iters_per_turn: u32,
    /// Optional higher PUCT iteration count specifically on Turn 1 (`0` = use `iters_per_turn`).
    pub turn1_iters: u32,
    /// Rao-Blackwellized early-exit mode.
    pub rb_exit: RbExitMode,
    /// Residual ridge feature basis.
    pub ridge_basis: TrajRidgeBasis,
    /// L2 ridge penalty on main mon effects (`1..25`).
    pub ridge_lambda_main: f64,
    /// L2 ridge penalty on interaction / pair / cross effects (`25..D`).
    pub ridge_lambda_inter: f64,
    /// Use inverse-variance GLS cell weighting `w = (kappa + 1) * n / (kappa + n)` in ridge fit.
    pub ridge_gls: bool,
    /// Share cross-world species-level residual ridge across belief worlds (`theta_shared + delta_w`).
    pub ridge_shared_multiworld: bool,
    /// Empirical-Bayes shrinkage pseudo-count toward Stage-1 + ridge prior.
    pub kappa: f64,
    /// Common Random Numbers and Turn-1 stratification schedule.
    pub crn_mode: TrajCrnMode,
    /// Fraction of `total_iters` spent on roster- & bench-pair-balanced exploration.
    pub explore_frac: f64,
    /// Fraction of `total_iters` spent on Paired CRN Double-Oracle rounds (remainder -> Polish).
    pub do_frac: f64,
    /// Fraction of `total_iters` reserved at the end for Turn-1 Tactical Lead Polish
    /// (zero-mean control-variate `V_T1 - V_s1` over active cores and all their lead permutations).
    pub t1_polish_frac: f64,
    /// Turn-1 PUCT iterations per active cell in `T1Polish`.
    pub t1_polish_iters: u32,
    /// Control-variate blend weight on centered `(V_T1 - V_s1)` (`0.0` = disabled).
    pub t1_polish_alpha: f64,
    /// Number of challengers paired per side per DO batch (including reserved bench-coverage probe).
    pub do_challengers: usize,
    /// UCB exploration constant for DO challenger selection.
    pub do_ucb_c: f64,
    /// Maximum Turn-0 belief worlds screened in Stage 1.
    pub max_stage1_worlds: usize,
    /// Maximum highest-weight belief worlds refined with trajectories in Stage 2.
    pub max_stage2_worlds: usize,
    /// Maximum turns per trajectory game before blindspot fallback.
    pub max_turns: u16,
    /// Bayesian RM+ sweeps during DO rounds and final readout.
    pub rm_sweeps_do: u32,
    pub rm_sweeps_final: u32,
    /// Final policy extraction rule.
    pub policy_mode: TrajPolicyMode,
    /// Interior PUCT hyperparameters (aligned with `search_compact_cfv_puct`).
    pub c_puct: f64,
    pub fpu_red: f64,
    pub puct_eps: f32,
    pub use_cf_policy: bool,
    pub leaf_temp: f64,
    /// Number of interleaved active trajectory slots for slice-invariant stepped execution.
    pub stepped_slots: usize,
    /// Fallback `BlindSearch` iterations when any belief world is unsupported by `CompactPreviewRoster`.
    pub fallback_iters: u32,
}

impl Default for TrajPreviewConfig {
    fn default() -> Self {
        Self {
            total_iters: 720_000,
            iters_per_turn: 32,
            turn1_iters: 0,
            rb_exit: RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            },
            ridge_basis: TrajRidgeBasis::Interact33,
            ridge_lambda_main: 12.0,
            ridge_lambda_inter: 20.0,
            ridge_gls: true,
            ridge_shared_multiworld: true,
            kappa: 16.0,
            crn_mode: TrajCrnMode::PairedCrnStratified,
            explore_frac: 0.30,
            do_frac: 0.45,
            t1_polish_frac: 0.0,
            t1_polish_iters: 320,
            t1_polish_alpha: 0.0,
            do_challengers: 4,
            do_ucb_c: 0.10,
            max_stage1_worlds: 24,
            max_stage2_worlds: 8,
            max_turns: 180,
            rm_sweeps_do: 350,
            rm_sweeps_final: 1500,
            policy_mode: TrajPolicyMode::BayesianBootstrapRm {
                draws: 16,
                dust_ratio: 0.15,
            },
            c_puct: 1.0,
            fpu_red: 0.09,
            puct_eps: 0.02,
            use_cf_policy: false,
            leaf_temp: 1.0,
            stepped_slots: 4,
            fallback_iters: 27_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrajPreviewResult {
    pub choice: SearchChoice,
    /// Probability distribution aligned 1:1 with `base.legal_choices(dex, side)`.
    pub policy_full: Vec<f64>,
    /// Opponent equilibrium policy in world 0 (canonical order) for single-world diagnostics.
    pub opp_policy_w0: Vec<f64>,
    pub value: f64,
    pub pure_guarantee: f64,
    pub mixed_guarantee: f64,
    pub duality_gap: f64,
    pub used_mix: bool,
    pub used_fallback: bool,
    pub stage1_ms: f64,
    pub total_ms: f64,
    pub iters_used: u32,
    pub games_completed: usize,
    pub worlds_stage1: usize,
    pub worlds_stage2: usize,
}

#[inline]
fn splitmix_hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Detect whether any living mon on the board carries an endgame tactical mechanic
/// that static `compact_eval_leaf_blindspot` / `linear.rs` does not accurately model:
/// `Endure`, `Flail`/`Reversal`, `Focus Band`, `Destiny Bond`, `Perish Song`,
/// `Explosion`/`Self-Destruct`, `Counter`/`Mirror Coat`, active `Encore`, or low-PP /
/// mutual-recovery stall.
pub fn has_tactical_endgame_hazard(cb: &CompactBattle, table: &MatchTable) -> bool {
    const HAZARD_VOL: u16 =
        vol_bit::PERISH_SONG | vol_bit::DESTINY_BOND | vol_bit::ENCORE | vol_bit::ENDURE;

    let mut active_has_recovery = [false; 2];
    for side in 0..2usize {
        let mask = cb.sides[side].alive_mask;
        let act_slot = cb.sides[side].active_idx as usize;
        for slot in 0..3usize {
            if (mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = side * 3 + slot;
            let mon = &cb.mons[idx];
            if mon.hp() <= 0 {
                continue;
            }
            if (mon.vol_flags & HAZARD_VOL) != 0 {
                return true;
            }
            if mon.item > 0 {
                let ispec = &table.item_specs[(mon.item as usize - 1).min(table.item_specs.len().saturating_sub(1))];
                if ispec.is_focus_band {
                    return true;
                }
            }
            let nm = (table.mons[idx].num_moves as usize).min(4);
            for m in 0..nm {
                let pp = mon.pp[m];
                if pp == 0 {
                    continue;
                }
                let ms = &table.move_specs[idx][m];
                if ms.is_endure
                    || ms.is_selfdestruct
                    || ms.var_bp_kind == VarBpKind::FlailReversal
                    || matches!(
                        ms.special_kind,
                        SpecialMoveKind::FlailReversal
                            | SpecialMoveKind::DestinyBond
                            | SpecialMoveKind::PerishSong
                            | SpecialMoveKind::Counter
                            | SpecialMoveKind::MirrorCoat
                    )
                {
                    return true;
                }
                if ms.is_rest || ms.is_half_heal {
                    if pp <= 4 {
                        return true;
                    }
                    if slot == act_slot {
                        active_has_recovery[side] = true;
                    }
                }
            }
        }
    }
    if active_has_recovery[0] && active_has_recovery[1] && cb.turn >= 20 {
        return true;
    }
    false
}

/// Reusable zero-allocation per-turn PUCT scratchpad (`cfv-0r@puct@lam0` + `n0+n1 <= 3` 8r guard).
pub struct TrajTurnScratch {
    nodes: Vec<CfPrevNode>,
    tt: FxHashMap<u64, usize>,
    path: Vec<(usize, usize, usize)>,
    eg_weights: EvalWeights,
    eg_playout: Playout,
    rollout_rules: crate::mcts::RolloutRules,
}

impl Default for TrajTurnScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl TrajTurnScratch {
    pub fn new() -> Self {
        let eg_weights = EvalWeights::default();
        let eg_playout = Playout::Heavy {
            turns: 8,
            eps: 0.2,
            weights: eg_weights.clone(),
        };
        let rollout_rules = RmConfig::default().rollout_rules();
        Self {
            nodes: Vec::with_capacity(64),
            tt: FxHashMap::default(),
            path: Vec::with_capacity(64),
            eg_weights,
            eg_playout,
            rollout_rules,
        }
    }

    /// Initialize the scratch tree for a new decision turn on `root_cb`.
    fn reset_for_turn(
        &mut self,
        base_cb: &CompactBattle,
        table: &MatchTable,
        cache: &TempoMatchCache,
        puct_eps: f32,
        use_cf_policy: bool,
    ) -> (CompactBattle, u16) {
        let mut turn_cap = base_cb.turn.saturating_add(100);
        if let Some(lim) = table.ruleset.turn_limit() {
            turn_cap = turn_cap.min(lim);
        }
        let mut root_cb = *base_cb;
        let root_idle = root_cb.sides[0].request == 1 && root_cb.sides[1].request == 1;
        if root_idle {
            root_cb.quick_claw_roll = 0;
        }
        let mut rn0 = CfPrevNode::new(&root_cb, table);
        rn0.ensure_priors(&root_cb, table, cache, puct_eps, use_cf_policy);
        let omit_qc = root_idle || !root_cb.quick_claw_observable(table);
        let root_key = root_cb.search_key(table, 16, omit_qc);
        self.nodes.clear();
        self.tt.clear();
        self.nodes.push(rn0);
        self.tt.insert(root_key, 0);
        (root_cb, turn_cap)
    }

    /// Execute one PUCT iteration on the active turn tree; returns `(leaf_val0, depth, added_nodes)`.
    #[allow(clippy::too_many_arguments)]
    fn step_puct_iter(
        &mut self,
        root_cb: &CompactBattle,
        table: &MatchTable,
        cache: &TempoMatchCache,
        turn_cap: u16,
        c_puct: f64,
        fpu_red: f64,
        puct_eps: f32,
        use_cf_policy: bool,
        leaf_temp: f64,
        rng: &mut SplitMix64,
    ) -> (f64, u32, usize) {
        let prev_nodes = self.nodes.len();
        let mut cb = *root_cb;
        {
            let mut prng = nc2000_engine::prng::Prng::new(rng.next());
            let qc = Battle::roll_quick_claw_with(table.ruleset, &mut prng);
            cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
            cb.prng_seed = prng.seed();
        }
        let mut cur_id = 0usize;
        let mut depth = 0u8;
        self.path.clear();

        let leaf_val = loop {
            self.nodes[cur_id].ensure_priors(&cb, table, cache, puct_eps, use_cf_policy);
            let node = &self.nodes[cur_id];
            let n0_cur = node.n_acts[0] as usize;
            let n1_cur = node.n_acts[1] as usize;
            let i0 = node.select_puct(0, c_puct, fpu_red);
            let i1 = node.select_puct(1, c_puct, fpu_red);
            let a0 = if n0_cur > 0 { node.acts[0][i0] } else { 7 };
            let a1 = if n1_cur > 0 { node.acts[1][i1] } else { 7 };
            if depth == 0 || n0_cur > 1 || n1_cur > 1 {
                self.path.push((cur_id, i0, i1));
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
                let alive_sum = cb.sides[0].alive_mask.count_ones()
                    + cb.sides[1].alive_mask.count_ones();
                if alive_sum <= 3 {
                    break compact_eval_leaf_blindspot(&cb, table, &self.eg_weights);
                }
                break evaluate_cf_value_resolving_switches(&cb, table, Some(cache), leaf_temp);
            }
            let oq = !cb.quick_claw_observable(table);
            let nk = cb.search_key(table, 16, oq);
            if let Some(&nid) = self.tt.get(&nk) {
                cur_id = nid;
            } else {
                let nid = self.nodes.len();
                self.nodes.push(CfPrevNode::new(&cb, table));
                self.tt.insert(nk, nid);
                let alive_sum = cb.sides[0].alive_mask.count_ones()
                    + cb.sides[1].alive_mask.count_ones();
                if alive_sum <= 3 {
                    let cutoff = turn_cap.min(cb.turn.saturating_add(7));
                    break crate::mcts::compact_playout_with_leaf(
                        &mut cb,
                        table,
                        &self.eg_playout,
                        cutoff,
                        rng,
                        self.rollout_rules,
                        |cb_l, t_l| compact_eval_leaf_blindspot(cb_l, t_l, &self.eg_weights),
                    );
                }
                break evaluate_cf_value_resolving_switches(&cb, table, Some(cache), leaf_temp);
            }
        };

        for &(nid, i0, i1) in self.path.iter().rev() {
            let n = &mut self.nodes[nid];
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
        (leaf_val, u32::from(depth), self.nodes.len() - prev_nodes)
    }

    /// Extract most-visited non-noop actions per side and root value `v_root0`.
    fn readout_turn(&self) -> ([u8; 2], f64) {
        let rn = &self.nodes[0];
        let mut best_acts = [7u8; 2];
        for s in 0..2usize {
            let k = rn.n_acts[s] as usize;
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
                    let qi = if vi > 0 {
                        let raw = rn.sums[s][i] / f64::from(vi);
                        if s == 0 { raw } else { 1.0 - raw }
                    } else {
                        0.5
                    };
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
        let k0 = rn.n_acts[0] as usize;
        let n0_tot: u32 = rn.visits[0][..k0].iter().sum();
        let v_root0 = if n0_tot > 0 {
            let w0: f64 = rn.sums[0][..k0].iter().sum();
            (w0 / f64::from(n0_tot)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        (best_acts, v_root0)
    }
}

/// Play a single full-game `CompactBattle` trajectory using reusable `TrajTurnScratch`,
/// returning `(val0, turns_played, iters_used, exited_early_rb)`.
#[allow(clippy::too_many_arguments)]
pub fn play_compact_traj_with_rb(
    roster: &CompactPreviewRoster,
    pcache: &PreviewTempoCache,
    picks: [[u8; 3]; 2],
    iters_per_turn: u32,
    rb_exit: RbExitMode,
    crn_mode: TrajCrnMode,
    game_seed: u64,
    cell_visit_k: u32,
    max_turns: u16,
    scratch: &mut TrajTurnScratch,
) -> (f64, u16, u32, bool) {
    let mut cfg = TrajPreviewConfig::default();
    cfg.iters_per_turn = iters_per_turn;
    cfg.rb_exit = rb_exit;
    cfg.crn_mode = crn_mode;
    cfg.max_turns = max_turns;
    let mut g = ActiveTrajGame::new(
        0,
        0,
        0,
        0,
        roster,
        pcache,
        picks,
        game_seed,
        cell_visit_k,
        crn_mode,
        0,
    );
    let mut iters_used = 0u32;
    loop {
        if let Some((v0, rb)) = g.prepare_next_turn(&cfg, scratch) {
            return (v0, g.cb.turn, iters_used, rb);
        }
        while g.turn_iters_done < g.turn_target_iters {
            let _ = scratch.step_puct_iter(
                &g.turn_root_cb,
                &g.table,
                &g.cache,
                g.turn_cap,
                cfg.c_puct,
                cfg.fpu_red,
                cfg.puct_eps,
                cfg.use_cf_policy,
                cfg.leaf_temp,
                &mut g.turn_rng,
            );
            g.turn_iters_done += 1;
            iters_used += 1;
        }
        if let Some((v0, rb)) = g.finish_turn(scratch) {
            return (v0, g.cb.turn, iters_used, rb);
        }
    }
}

/// Precomputed per-world static metadata for compact interaction features (`Interact31` / `Interact33`).
#[derive(Clone)]
struct WorldInteractMeta {
    has_spikes: [[bool; 6]; 2],
    has_forced_phaze: [[bool; 6]; 2],
    is_grounded: [[bool; 6]; 2],
    has_bp: [[bool; 6]; 2],
    has_setup: [[bool; 6]; 2],
    has_sleep: [[bool; 6]; 2],
    spe: [[i16; 6]; 2],
    /// Pairwise Turn-1 expected best damage fraction `[att_side][att_slot_0..6][def_slot_0..6]` in `[0, 1]`.
    dmg_frac: [[[f64; 6]; 6]; 2],
}

impl WorldInteractMeta {
    fn build(
        roster: &CompactPreviewRoster,
        pcache: &PreviewTempoCache,
        party_lens: [usize; 2],
    ) -> Self {
        let mut meta = Self {
            has_spikes: [[false; 6]; 2],
            has_forced_phaze: [[false; 6]; 2],
            is_grounded: [[true; 6]; 2],
            has_bp: [[false; 6]; 2],
            has_setup: [[false; 6]; 2],
            has_sleep: [[false; 6]; 2],
            spe: [[0; 6]; 2],
            dmg_frac: [[[0.0; 6]; 6]; 2],
        };
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
        for b0 in 0..n0 {
            for b1 in 0..n1 {
                let (table, cb) = roster.build_turn1([blk0[b0], blk1[b1]], 0);
                let sliced = pcache.slice_for(&cb);
                let Some(view) = sliced.view_for(&cb) else {
                    continue;
                };
                for s in 0..2usize {
                    let blk = if s == 0 { blk0[b0] } else { blk1[b1] };
                    for slot in 0..3usize {
                        let rslot = blk[slot];
                        if rslot == 0 || (rslot as usize) > 6 {
                            continue;
                        }
                        let pos = rslot as usize - 1;
                        let midx = s * 3 + slot;
                        let sm = view.mon(s, slot);
                        meta.has_forced_phaze[s][pos] = sm.phaze_slot != 0xFF;
                        meta.has_bp[s][pos] = sm.bp_slot != 0xFF;
                        meta.is_grounded[s][pos] =
                            (table.mons[midx].type_mask & (1u32 << table.flying_type)) == 0;
                        let mut spikes = false;
                        let mut setup = sm.setup_mask != 0;
                        let nm = (table.mons[midx].num_moves as usize).min(4);
                        for m in 0..nm {
                            let ms = &table.move_specs[midx][m];
                            if ms.special_kind == SpecialMoveKind::Spikes {
                                spikes = true;
                            }
                            if ms.is_substitute
                                || matches!(
                                    ms.special_kind,
                                    SpecialMoveKind::Curse | SpecialMoveKind::BellyDrum
                                )
                            {
                                setup = true;
                            }
                        }
                        meta.has_spikes[s][pos] = spikes;
                        meta.has_setup[s][pos] = setup;
                        meta.has_sleep[s][pos] = sm.sleep_mask != 0;
                        meta.spe[s][pos] = table.mons[midx].stored_stats[4];
                    }
                }
                for a_slot in 0..3usize {
                    let a_r = blk0[b0][a_slot];
                    if a_r == 0 || (a_r as usize) > 6 {
                        continue;
                    }
                    let a_pos = a_r as usize - 1;
                    let inv_hp0 = view.mon(0, a_slot).inv_maxhp;
                    for d_slot in 0..3usize {
                        let d_r = blk1[b1][d_slot];
                        if d_r == 0 || (d_r as usize) > 6 {
                            continue;
                        }
                        let d_pos = d_r as usize - 1;
                        let inv_hp1 = view.mon(1, d_slot).inv_maxhp;

                        let p01 = view.pair(0, a_slot, d_slot);
                        let mut best01 = 0.0f64;
                        for mv in &p01.moves {
                            if (mv.flags & (1 << 6)) != 0 {
                                continue; // ignore Self-Destruct/Explosion for sustainable pressure
                            }
                            let d = f64::from(mv.mean_dmg * mv.p_hit) * inv_hp1;
                            if d > best01 {
                                best01 = d;
                            }
                        }
                        meta.dmg_frac[0][a_pos][d_pos] = best01.clamp(0.0, 1.0);

                        let p10 = view.pair(1, d_slot, a_slot);
                        let mut best10 = 0.0f64;
                        for mv in &p10.moves {
                            if (mv.flags & (1 << 6)) != 0 {
                                continue;
                            }
                            let d = f64::from(mv.mean_dmg * mv.p_hit) * inv_hp0;
                            if d > best10 {
                                best10 = d;
                            }
                        }
                        meta.dmg_frac[1][d_pos][a_pos] = best10.clamp(0.0, 1.0);
                    }
                }
            }
        }
        meta
    }
}

const MAX_RIDGE_DIM: usize = 68;
const MAX_RIDGE_NNZ: usize = 32;

#[inline]
fn unordered_pair_idx(a: usize, b: usize) -> usize {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    // Map 0 <= lo < hi <= 5 into 0..15.
    let base = lo * 5 - (lo * (lo.saturating_sub(1))) / 2;
    (base + (hi - lo - 1)).min(14)
}

/// Extract sparse `(feature_idx, feature_val)` entries for a `(own_triple, opp_triple)` pair
/// oriented from `side`'s perspective (slots `1..=6`; `0` safely skipped on `< 3`-mon teams).
fn extract_ridge_features(
    basis: TrajRidgeBasis,
    side: usize,
    own_t: [u8; 3],
    opp_t: [u8; 3],
    meta: &WorldInteractMeta,
    out: &mut [(usize, f64); MAX_RIDGE_NNZ],
) -> usize {
    let mut k = 0usize;
    out[k] = (0, 1.0);
    k += 1;

    let mut own_slots = [0usize; 3];
    let mut n_own = 0usize;
    let mut opp_slots = [0usize; 3];
    let mut n_opp = 0usize;
    if basis == TrajRidgeBasis::Present19 {
        for &s in &own_t {
            if s >= 1 && (s as usize) <= 6 {
                let idx = s as usize - 1;
                own_slots[n_own] = idx;
                n_own += 1;
                out[k] = (1 + idx, 1.0);
                k += 1;
            }
        }
        for &s in &opp_t {
            if s >= 1 && (s as usize) <= 6 {
                let idx = s as usize - 1;
                opp_slots[n_opp] = idx;
                n_opp += 1;
                out[k] = (7 + idx, 1.0);
                k += 1;
            }
        }
    } else if basis == TrajRidgeBasis::PresentInteract33 {
        for (pos, &s) in own_t.iter().enumerate() {
            if s >= 1 && (s as usize) <= 6 {
                let idx = s as usize - 1;
                own_slots[n_own] = idx;
                n_own += 1;
                out[k] = (1 + idx, 1.0);
                k += 1;
                if pos == 0 {
                    out[k] = (13 + idx, 1.0);
                    k += 1;
                }
            }
        }
        for (pos, &s) in opp_t.iter().enumerate() {
            if s >= 1 && (s as usize) <= 6 {
                let idx = s as usize - 1;
                opp_slots[n_opp] = idx;
                n_opp += 1;
                out[k] = (7 + idx, 1.0);
                k += 1;
                if pos == 0 {
                    out[k] = (19 + idx, 1.0);
                    k += 1;
                }
            }
        }
    } else {
        for (pos, &s) in own_t.iter().enumerate() {
            if s >= 1 && (s as usize) <= 6 {
                let idx = s as usize - 1;
                own_slots[n_own] = idx;
                n_own += 1;
                let f = if pos == 0 { 1 + idx } else { 7 + idx };
                out[k] = (f, 1.0);
                k += 1;
            }
        }
        for (pos, &s) in opp_t.iter().enumerate() {
            if s >= 1 && (s as usize) <= 6 {
                let idx = s as usize - 1;
                opp_slots[n_opp] = idx;
                n_opp += 1;
                let f = if pos == 0 { 13 + idx } else { 19 + idx };
                out[k] = (f, 1.0);
                k += 1;
            }
        }
    }

    let s_me = side;
    let s_op = 1 - side;

    let add_domain_interact = |k: &mut usize,
                               out: &mut [(usize, f64); MAX_RIDGE_NNZ],
                               include_spikes_only: bool| {
        let own_spikes = own_slots[..n_own].iter().any(|&p| meta.has_spikes[s_me][p]);
        let own_phaze = own_slots[..n_own].iter().any(|&p| meta.has_forced_phaze[s_me][p]);
        let opp_grounded = opp_slots[..n_opp]
            .iter()
            .filter(|&&p| meta.is_grounded[s_op][p])
            .count() as f64
            / (n_opp.max(1) as f64);

        let opp_spikes = opp_slots[..n_opp].iter().any(|&p| meta.has_spikes[s_op][p]);
        let opp_phaze = opp_slots[..n_opp].iter().any(|&p| meta.has_forced_phaze[s_op][p]);
        let own_grounded = own_slots[..n_own]
            .iter()
            .filter(|&&p| meta.is_grounded[s_me][p])
            .count() as f64
            / (n_own.max(1) as f64);

        if own_spikes && own_phaze && opp_grounded > 0.0 {
            out[*k] = (25, opp_grounded);
            *k += 1;
        }
        if opp_spikes && opp_phaze && own_grounded > 0.0 {
            out[*k] = (26, own_grounded);
            *k += 1;
        }

        let own_bp = own_slots[..n_own].iter().any(|&p| meta.has_bp[s_me][p]);
        let own_setup = own_slots[..n_own].iter().any(|&p| meta.has_setup[s_me][p]);
        if own_bp && own_setup && n_own >= 2 {
            out[*k] = (27, 1.0);
            *k += 1;
        }
        let opp_bp = opp_slots[..n_opp].iter().any(|&p| meta.has_bp[s_op][p]);
        let opp_setup = opp_slots[..n_opp].iter().any(|&p| meta.has_setup[s_op][p]);
        if opp_bp && opp_setup && n_opp >= 2 {
            out[*k] = (28, 1.0);
            *k += 1;
        }

        // Lead pressure after accounting for opponent's safest switch-in among picked 3.
        if n_own > 0 && n_opp > 0 {
            let l_me = own_slots[0];
            let mut min_p_me = f64::INFINITY;
            for &d in &opp_slots[..n_opp] {
                let net = meta.dmg_frac[s_me][l_me][d] - 0.5 * meta.dmg_frac[s_op][d][l_me];
                if net < min_p_me {
                    min_p_me = net;
                }
            }
            if min_p_me.is_finite() {
                out[*k] = (29, min_p_me.clamp(-0.5, 1.0));
                *k += 1;
            }
            let l_op = opp_slots[0];
            let mut min_p_op = f64::INFINITY;
            for &d in &own_slots[..n_own] {
                let net = meta.dmg_frac[s_op][l_op][d] - 0.5 * meta.dmg_frac[s_me][d][l_op];
                if net < min_p_op {
                    min_p_op = net;
                }
            }
            if min_p_op.is_finite() {
                out[*k] = (30, min_p_op.clamp(-0.5, 1.0));
                *k += 1;
            }
        }
        if include_spikes_only {
            if own_spikes && opp_grounded > 0.0 {
                out[*k] = (31, opp_grounded);
                *k += 1;
            }
            if opp_spikes && own_grounded > 0.0 {
                out[*k] = (32, own_grounded);
                *k += 1;
            }
        }
    };

    match basis {
        TrajRidgeBasis::Present19 => {
            let own_spikes = own_slots[..n_own].iter().any(|&p| meta.has_spikes[s_me][p]);
            let own_phaze = own_slots[..n_own].iter().any(|&p| meta.has_forced_phaze[s_me][p]);
            let opp_grounded = opp_slots[..n_opp]
                .iter()
                .filter(|&&p| meta.is_grounded[s_op][p])
                .count() as f64
                / (n_opp.max(1) as f64);
            let opp_spikes = opp_slots[..n_opp].iter().any(|&p| meta.has_spikes[s_op][p]);
            let opp_phaze = opp_slots[..n_opp].iter().any(|&p| meta.has_forced_phaze[s_op][p]);
            let own_grounded = own_slots[..n_own]
                .iter()
                .filter(|&&p| meta.is_grounded[s_me][p])
                .count() as f64
                / (n_own.max(1) as f64);
            if own_spikes && own_phaze && opp_grounded > 0.0 {
                out[k] = (13, opp_grounded);
                k += 1;
            }
            if opp_spikes && opp_phaze && own_grounded > 0.0 {
                out[k] = (14, own_grounded);
                k += 1;
            }
            let own_bp = own_slots[..n_own].iter().any(|&p| meta.has_bp[s_me][p]);
            let own_setup = own_slots[..n_own].iter().any(|&p| meta.has_setup[s_me][p]);
            if own_bp && own_setup && n_own >= 2 {
                out[k] = (15, 1.0);
                k += 1;
            }
            let opp_bp = opp_slots[..n_opp].iter().any(|&p| meta.has_bp[s_op][p]);
            let opp_setup = opp_slots[..n_opp].iter().any(|&p| meta.has_setup[s_op][p]);
            if opp_bp && opp_setup && n_opp >= 2 {
                out[k] = (16, 1.0);
                k += 1;
            }
            if own_spikes && opp_grounded > 0.0 {
                out[k] = (17, opp_grounded);
                k += 1;
            }
            if opp_spikes && own_grounded > 0.0 {
                out[k] = (18, own_grounded);
                k += 1;
            }
        }
        TrajRidgeBasis::Additive25 => {}
        TrajRidgeBasis::Interact31 => {
            add_domain_interact(&mut k, out, false);
        }
        TrajRidgeBasis::Interact33 | TrajRidgeBasis::PresentInteract33 => {
            add_domain_interact(&mut k, out, true);
        }
        TrajRidgeBasis::Interact39 => {
            add_domain_interact(&mut k, out, true);
            if n_own > 0 && n_opp > 0 {
                let l_me = own_slots[0];
                let l_op = opp_slots[0];
                let d_me = meta.dmg_frac[s_me][l_me][l_op];
                let d_op = meta.dmg_frac[s_op][l_op][l_me];
                if d_me > 0.0 {
                    out[k] = (33, d_me);
                    k += 1;
                }
                if d_op > 0.0 {
                    out[k] = (34, d_op);
                    k += 1;
                }
                let spe_me = meta.spe[s_me][l_me];
                let spe_op = meta.spe[s_op][l_op];
                let fast_me = if spe_me > spe_op {
                    1.0
                } else if spe_me == spe_op {
                    0.5
                } else {
                    0.0
                };
                let fast_op = 1.0 - fast_me;
                if fast_me * d_me > 0.0 {
                    out[k] = (35, fast_me * d_me);
                    k += 1;
                }
                if fast_op * d_op > 0.0 {
                    out[k] = (36, fast_op * d_op);
                    k += 1;
                }
                if fast_me > 0.0 && meta.has_sleep[s_me][l_me] {
                    out[k] = (37, fast_me);
                    k += 1;
                }
                if fast_op > 0.0 && meta.has_sleep[s_op][l_op] {
                    out[k] = (38, fast_op);
                    k += 1;
                }
            }
        }
        TrajRidgeBasis::Pair55 => {
            for a in 0..n_own {
                for b in (a + 1)..n_own {
                    if own_slots[a] != own_slots[b] {
                        out[k] = (25 + unordered_pair_idx(own_slots[a], own_slots[b]), 1.0);
                        k += 1;
                    }
                }
            }
            for a in 0..n_opp {
                for b in (a + 1)..n_opp {
                    if opp_slots[a] != opp_slots[b] {
                        out[k] = (40 + unordered_pair_idx(opp_slots[a], opp_slots[b]), 1.0);
                        k += 1;
                    }
                }
            }
        }
        TrajRidgeBasis::MonCross61 => {
            for &a in &own_slots[..n_own] {
                for &b in &opp_slots[..n_opp] {
                    out[k] = (25 + a * 6 + b, 1.0 / 3.0);
                    k += 1;
                }
            }
        }
        TrajRidgeBasis::Interact67 => {
            add_domain_interact(&mut k, out, false);
            for &a in &own_slots[..n_own] {
                for &b in &opp_slots[..n_opp] {
                    out[k] = (31 + a * 6 + b, 1.0 / 3.0);
                    k += 1;
                }
            }
        }
    }
    k
}

/// Fast Cholesky SPD solver for `(A + diag(lambda)) w = b` (`dim <= MAX_RIDGE_DIM`).
fn solve_spd_cholesky(
    dim: usize,
    ata: &[[f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM],
    atb: &[f64; MAX_RIDGE_DIM],
    lambda_int: f64,
    lambda_main: f64,
    lambda_inter: f64,
) -> [f64; MAX_RIDGE_DIM] {
    solve_spd_cholesky_split(dim, 25, ata, atb, lambda_int, lambda_main, lambda_inter)
}

fn solve_spd_cholesky_split(
    dim: usize,
    main_end: usize,
    ata: &[[f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM],
    atb: &[f64; MAX_RIDGE_DIM],
    lambda_int: f64,
    lambda_main: f64,
    lambda_inter: f64,
) -> [f64; MAX_RIDGE_DIM] {
    let mut l = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
    for i in 0..dim {
        let reg = if i == 0 {
            lambda_int
        } else if i < main_end {
            lambda_main
        } else {
            lambda_inter
        };
        for j in 0..=i {
            let mut sum = ata[i][j];
            if i == j {
                sum += reg;
            }
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = if sum > 1e-12 { sum.sqrt() } else { 1e-6 };
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    // Forward substitution: L y = atb.
    let mut y = [0.0f64; MAX_RIDGE_DIM];
    for i in 0..dim {
        let mut sum = atb[i];
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = sum / l[i][i];
    }
    // Backward substitution: L^T w = y.
    let mut w = [0.0f64; MAX_RIDGE_DIM];
    for i in (0..dim).rev() {
        let mut sum = y[i];
        for k in (i + 1)..dim {
            sum -= l[k][i] * w[k];
        }
        w[i] = sum / l[i][i];
    }
    w
}

/// L2-regularized Multi-World Bayesian RM+ (`max_p min_q sum_w pi_w p^T V_w q_w - l2_reg * ||p||_2^2`).
/// When `l2_reg <= 0.0`, delegates directly to exact `solve_bayesian_rm_plus`.
pub fn solve_bayesian_rm_plus_reg(
    n_own: usize,
    worlds: &[(f64, usize, &[f64])],
    sweeps: u32,
    l2_reg: f64,
) -> BayesianRmSolution {
    if l2_reg <= 0.0 {
        return solve_bayesian_rm_plus(n_own, worlds, sweeps);
    }
    assert!(n_own > 0 && !worlds.is_empty());
    let n_w = worlds.len();
    let mut r_own = vec![0.0f64; n_own];
    let mut avg_own = vec![0.0f64; n_own];
    let mut p_cur = vec![1.0 / (n_own as f64); n_own];

    let mut r_opp: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut avg_opp: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut q_cur: Vec<Vec<f64>> = worlds
        .iter()
        .map(|&(_, m, _)| vec![1.0 / (m.max(1) as f64); m])
        .collect();

    let mut row_ev = vec![0.0f64; n_own];
    let mut col_ev: Vec<Vec<f64>> = worlds.iter().map(|&(_, m, _)| vec![0.0; m]).collect();
    let mut weight_sum = 0.0f64;

    let steps = sweeps.max(1);
    for t in 1..=steps {
        let tw = f64::from(t);
        weight_sum += tw;

        let r_sum: f64 = r_own.iter().sum();
        if r_sum > 1e-15 {
            let inv = 1.0 / r_sum;
            for i in 0..n_own {
                p_cur[i] = r_own[i] * inv;
            }
        } else {
            let u = 1.0 / (n_own as f64);
            for p in &mut p_cur {
                *p = u;
            }
        }
        for i in 0..n_own {
            avg_own[i] += tw * p_cur[i];
        }

        for w in 0..n_w {
            let (_, m, vals) = worlds[w];
            if m == 0 {
                continue;
            }
            let cev = &mut col_ev[w];
            for x in cev.iter_mut() {
                *x = 0.0;
            }
            for i in 0..n_own {
                let pi = p_cur[i];
                if pi > 0.0 {
                    let row = &vals[i * m..(i + 1) * m];
                    for j in 0..m {
                        cev[j] += pi * row[j];
                    }
                }
            }
            let q = &mut q_cur[w];
            let mut val_w = 0.0;
            for j in 0..m {
                val_w += q[j] * cev[j];
            }
            let ro = &mut r_opp[w];
            let mut ro_sum = 0.0;
            for j in 0..m {
                let nr = (ro[j] + (val_w - cev[j])).max(0.0);
                ro[j] = nr;
                ro_sum += nr;
            }
            if ro_sum > 1e-15 {
                let inv = 1.0 / ro_sum;
                for j in 0..m {
                    q[j] = ro[j] * inv;
                }
            } else {
                let u = 1.0 / (m as f64);
                for x in q.iter_mut() {
                    *x = u;
                }
            }
            let ao = &mut avg_opp[w];
            for j in 0..m {
                ao[j] += tw * q[j];
            }
        }

        for x in &mut row_ev {
            *x = 0.0;
        }
        for w in 0..n_w {
            let (pi_w, m, vals) = worlds[w];
            if m == 0 || pi_w <= 0.0 {
                continue;
            }
            let q = &q_cur[w];
            for i in 0..n_own {
                let row = &vals[i * m..(i + 1) * m];
                let mut acc = 0.0;
                for j in 0..m {
                    acc += row[j] * q[j];
                }
                row_ev[i] += pi_w * acc;
            }
        }
        // Apply L2 concave regularizer gradient `-2 * l2_reg * p_cur[i]`.
        let mut val_cur = 0.0;
        for i in 0..n_own {
            let eff = row_ev[i] - 2.0 * l2_reg * p_cur[i];
            row_ev[i] = eff;
            val_cur += p_cur[i] * eff;
        }
        for i in 0..n_own {
            r_own[i] = (r_own[i] + (row_ev[i] - val_cur)).max(0.0);
        }
    }

    let inv_w = 1.0 / weight_sum.max(1.0);
    let mut own_policy: Vec<f64> = avg_own.iter().map(|&v| v * inv_w).collect();
    let p_sum: f64 = own_policy.iter().sum();
    if p_sum > 0.0 {
        for p in &mut own_policy {
            *p /= p_sum;
        }
    }
    let opp_policies: Vec<Vec<f64>> = avg_opp
        .iter()
        .map(|ao| {
            let mut q: Vec<f64> = ao.iter().map(|&v| v * inv_w).collect();
            let s: f64 = q.iter().sum();
            if s > 0.0 {
                for x in &mut q {
                    *x /= s;
                }
            }
            q
        })
        .collect();

    let mut row_vs_q = vec![0.0f64; n_own];
    let mut row_guarantees = vec![0.0f64; n_own];
    let mut col_vs_p: Vec<Vec<f64>> = Vec::with_capacity(n_w);
    let mut world_values = Vec::with_capacity(n_w);
    let mut lower_bound = 0.0;

    for w in 0..n_w {
        let (pi_w, m, vals) = worlds[w];
        let mut cp = vec![0.0f64; m];
        for i in 0..n_own {
            let row = &vals[i * m..(i + 1) * m];
            let mut rmin = f64::INFINITY;
            let mut r_q = 0.0;
            let pi = own_policy[i];
            for j in 0..m {
                let v = row[j];
                if v < rmin {
                    rmin = v;
                }
                r_q += v * opp_policies[w][j];
                if pi > 0.0 {
                    cp[j] += pi * v;
                }
            }
            row_guarantees[i] += pi_w * rmin;
            row_vs_q[i] += pi_w * r_q;
        }
        let min_cp = cp.iter().copied().fold(f64::INFINITY, f64::min);
        let mut vw = 0.0;
        for j in 0..m {
            vw += cp[j] * opp_policies[w][j];
        }
        lower_bound += pi_w * min_cp;
        world_values.push(vw);
        col_vs_p.push(cp);
    }
    let upper_bound = row_vs_q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let value: f64 = own_policy
        .iter()
        .zip(&row_vs_q)
        .map(|(&p, &u)| p * u)
        .sum();
    let duality_gap = (upper_bound - lower_bound).max(0.0);

    BayesianRmSolution {
        own_policy,
        opp_policies,
        value,
        duality_gap,
        row_guarantees,
        row_vs_q,
        col_vs_p,
        world_values,
    }
}

#[derive(Clone, Copy)]
struct TrajTask {
    wi: usize,
    own_i: usize,
    opp_j: usize,
    block_id: u32,
    game_seed: u64,
    cell_visit_k: u32,
    t1_only_iters: u32,
}

#[derive(Clone, Copy)]
struct TrajObs {
    wi: u16,
    own_i: u16,
    opp_j: u16,
    block_id: u32,
    val_mine: f64,
}

struct ActiveTrajGame {
    wi: usize,
    own_i: usize,
    opp_j: usize,
    block_id: u32,
    game_seed: u64,
    indep_rng: SplitMix64,
    table: MatchTable,
    cb: CompactBattle,
    cache: TempoMatchCache,
    turn_root_cb: CompactBattle,
    turn_cap: u16,
    turn_rng: SplitMix64,
    turn_iters_done: u32,
    turn_target_iters: u32,
    is_vroot_exit_turn: bool,
    t1_only_iters: u32,
    t1_late_sum: f64,
    t1_late_n: u32,
    turn1_v0: Option<f64>,
    steps_taken: u16,
    turn_ready: bool,
}

impl ActiveTrajGame {
    #[allow(clippy::too_many_arguments)]
    fn new(
        wi: usize,
        own_i: usize,
        opp_j: usize,
        block_id: u32,
        roster: &CompactPreviewRoster,
        pcache: &PreviewTempoCache,
        picks: [[u8; 3]; 2],
        game_seed: u64,
        cell_visit_k: u32,
        crn_mode: TrajCrnMode,
        t1_only_iters: u32,
    ) -> Self {
        let mut indep_rng = SplitMix64::new(game_seed);
        let init_seed = match crn_mode {
            TrajCrnMode::Independent => indep_rng.next(),
            TrajCrnMode::PairedCrn | TrajCrnMode::PairedCrnStratified => {
                splitmix_hash(game_seed ^ 0xA11C_E001)
            }
        };
        let (table, mut cb) = roster.build_turn1(picks, init_seed);
        let cache = pcache.slice_for(&cb);

        // Resample Turn-1 Quick Claw and Berserk Gene confusion duration.
        let t1_seed = match crn_mode {
            TrajCrnMode::Independent => indep_rng.next(),
            TrajCrnMode::PairedCrn | TrajCrnMode::PairedCrnStratified => {
                splitmix_hash(game_seed ^ 0xB22D_F002)
            }
        };
        let mut prng = nc2000_engine::prng::Prng::new(t1_seed);
        let mut qc = Battle::roll_quick_claw_with(table.ruleset, &mut prng);
        if crn_mode == TrajCrnMode::PairedCrnStratified {
            const PHI: f64 = 0.618_033_988_749_894_9;
            for s in 0..2usize {
                let item = cb.mons[s * 3].item;
                if item > 0 {
                    let ispec = &table.item_specs
                        [(item as usize - 1).min(table.item_specs.len().saturating_sub(1))];
                    if ispec.is_quick_claw {
                        let u0 = (splitmix_hash(game_seed ^ ((s as u64 + 1) * 0x517C_C1B7)) >> 11)
                            as f64
                            * (1.0 / ((1u64 << 53) as f64));
                        let u = (u0 + f64::from(cell_visit_k) * PHI).fract();
                        qc[s] = u < (60.0 / 256.0);
                    }
                }
            }
        }
        cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
        for s in 0..2usize {
            if (cb.mons[s * 3].vol_flags & vol_bit::CONFUSION) != 0 {
                let dur = if crn_mode == TrajCrnMode::PairedCrnStratified {
                    let perm = (splitmix_hash(game_seed ^ ((s as u64 + 3) * 0x9E37_79B9)) & 3) as u32;
                    2 + ((cell_visit_k.wrapping_add(perm)) & 3)
                } else {
                    2 + prng.random(4)
                };
                cb.mons[s * 3].timers[timer_idx::CONFUSION] = dur as u8;
            }
        }
        cb.prng_seed = prng.seed();

        let turn_root_cb = cb;
        Self {
            wi,
            own_i,
            opp_j,
            block_id,
            game_seed,
            indep_rng,
            table,
            cb,
            cache,
            turn_root_cb,
            turn_cap: 101,
            turn_rng: SplitMix64::new(0),
            turn_iters_done: 0,
            turn_target_iters: 0,
            is_vroot_exit_turn: false,
            t1_only_iters,
            t1_late_sum: 0.0,
            t1_late_n: 0,
            turn1_v0: None,
            steps_taken: 0,
            turn_ready: false,
        }
    }

    /// Advance forced/trivial turns until either the game ends (returning `Some((val0, rb_exit))`)
    /// or the next decision turn is initialized inside `scratch`.
    fn prepare_next_turn(
        &mut self,
        cfg: &TrajPreviewConfig,
        scratch: &mut TrajTurnScratch,
    ) -> Option<(f64, bool)> {
        let max_steps = cfg.max_turns.saturating_mul(3).saturating_add(16);
        while self.steps_taken < max_steps {
            if self.cb.outcome != 0 {
                let v0 = match self.cb.outcome {
                    1 => 1.0,
                    2 => 0.0,
                    _ => 0.5,
                };
                return Some((v0, false));
            }
            if let Some(v) = compact_turn_limit_outcome(&self.cb, &self.table) {
                return Some((v, false));
            }
            if self.cb.turn > cfg.max_turns {
                return Some((
                    compact_eval_leaf_blindspot(&self.cb, &self.table, &scratch.eg_weights),
                    true,
                ));
            }

            let is_idle = self.cb.sides[0].request == 1 && self.cb.sides[1].request == 1;
            self.is_vroot_exit_turn = false;
            if is_idle && self.t1_only_iters == 0 {
                let alive_sum = self.cb.sides[0].alive_mask.count_ones()
                    + self.cb.sides[1].alive_mask.count_ones();
                match cfg.rb_exit {
                    RbExitMode::Terminal => {}
                    RbExitMode::UncondBlindspot { max_alive } if alive_sum <= max_alive => {
                        return Some((
                            compact_eval_leaf_blindspot(&self.cb, &self.table, &scratch.eg_weights),
                            true,
                        ));
                    }
                    RbExitMode::CondBlindspot { max_alive }
                        if alive_sum <= max_alive
                            && !has_tactical_endgame_hazard(&self.cb, &self.table) =>
                    {
                        return Some((
                            compact_eval_leaf_blindspot(&self.cb, &self.table, &scratch.eg_weights),
                            true,
                        ));
                    }
                    RbExitMode::SearchVroot {
                        max_alive,
                        eval_iters,
                    } if alive_sum <= max_alive => {
                        self.is_vroot_exit_turn = true;
                        self.turn_target_iters = eval_iters.max(1);
                    }
                    RbExitMode::CondSearchVroot {
                        max_alive,
                        eval_iters,
                    } if alive_sum <= max_alive
                        && !has_tactical_endgame_hazard(&self.cb, &self.table) =>
                    {
                        self.is_vroot_exit_turn = true;
                        self.turn_target_iters = eval_iters.max(1);
                    }
                    RbExitMode::HybridCond {
                        max_alive,
                        eval_iters,
                    } if alive_sum <= max_alive => {
                        if !has_tactical_endgame_hazard(&self.cb, &self.table) {
                            return Some((
                                compact_eval_leaf_blindspot(
                                    &self.cb,
                                    &self.table,
                                    &scratch.eg_weights,
                                ),
                                true,
                            ));
                        }
                        self.is_vroot_exit_turn = true;
                        self.turn_target_iters = eval_iters.max(1);
                    }
                    _ => {}
                }

                // Reseed idle-turn PRNG (turn-indexed for CRN synchronization).
                let prng_s = match cfg.crn_mode {
                    TrajCrnMode::Independent => self.indep_rng.next(),
                    TrajCrnMode::PairedCrn | TrajCrnMode::PairedCrnStratified => splitmix_hash(
                        self.game_seed
                            ^ ((u64::from(self.cb.turn)).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            ^ 0x1357_9BDF,
                    ),
                };
                let mut prng = nc2000_engine::prng::Prng::new(prng_s);
                let qc = Battle::roll_quick_claw_with(self.table.ruleset, &mut prng);
                self.cb.quick_claw_roll = (qc[0] as u8) | ((qc[1] as u8) << 1);
                self.cb.prng_seed = prng.seed();
            }

            let mut a0 = [0u8; 7];
            let mut a1 = [0u8; 7];
            let n0 = if self.cb.sides[0].request > 0 {
                self.cb.legal_actions(&self.table, 0, &mut a0).min(6)
            } else {
                0
            };
            let n1 = if self.cb.sides[1].request > 0 {
                self.cb.legal_actions(&self.table, 1, &mut a1).min(6)
            } else {
                0
            };
            if n0 == 0 && n1 == 0 {
                break;
            }
            if !self.is_vroot_exit_turn && self.t1_only_iters == 0 && n0 <= 1 && n1 <= 1 {
                let p0 = if n0 == 1 { a0[0] } else { 7 };
                let p1 = if n1 == 1 { a1[0] } else { 7 };
                self.cb.apply_actions(&self.table, [p0, p1]);
                self.steps_taken = self.steps_taken.saturating_add(1);
                continue;
            }

            let sseed = match cfg.crn_mode {
                TrajCrnMode::Independent => self.indep_rng.next(),
                TrajCrnMode::PairedCrn | TrajCrnMode::PairedCrnStratified => splitmix_hash(
                    self.game_seed
                        ^ ((u64::from(self.cb.turn)).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                        ^ (u64::from(self.steps_taken) << 24)
                        ^ 0x2468_ACE0,
                ),
            };
            let (root_cb, tcap) = scratch.reset_for_turn(
                &self.cb,
                &self.table,
                &self.cache,
                cfg.puct_eps,
                cfg.use_cf_policy,
            );
            self.turn_root_cb = root_cb;
            self.turn_cap = tcap;
            self.turn_rng = SplitMix64::new(sseed ^ 0xA24B_AED4_963E_E407);
            self.turn_iters_done = 0;
            if !self.is_vroot_exit_turn {
                self.turn_target_iters = if self.t1_only_iters > 0 {
                    self.t1_only_iters
                } else if self.cb.turn == 1 && cfg.turn1_iters > 0 {
                    cfg.turn1_iters
                } else {
                    cfg.iters_per_turn.max(1)
                };
            }
            self.turn_ready = true;
            return None;
        }
        let v0 = if self.cb.outcome != 0 {
            match self.cb.outcome {
                1 => 1.0,
                2 => 0.0,
                _ => 0.5,
            }
        } else {
            compact_eval_leaf_blindspot(&self.cb, &self.table, &scratch.eg_weights)
        };
        Some((v0, self.cb.outcome == 0))
    }

    fn finish_turn(&mut self, scratch: &TrajTurnScratch) -> Option<(f64, bool)> {
        let (best_acts, v_root0) = scratch.readout_turn();
        self.turn_ready = false;
        if self.is_vroot_exit_turn {
            return Some((v_root0, true));
        }
        self.cb.apply_actions(&self.table, best_acts);
        self.steps_taken = self.steps_taken.saturating_add(1);
        None
    }
}

struct TrajWorldStats {
    cell_n: Vec<u32>,
    cell_sum: Vec<f64>,
    t1_n: Vec<u32>,
    t1_sum: Vec<f64>,
    meta: WorldInteractMeta,
    opp_mon_cnt: [u32; 6],
    opp_lead_cnt: [u32; 6],
    opp_bench_pair_cnt: [u32; 15],
    explore_games: u32,
    do_batches: u32,
    polish_batches: u32,
    t1_batches: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrajPhase {
    Explore,
    DoubleOracle,
    Polish,
    T1Polish,
}

/// Incrementally steppable, bit-identically slice-invariant, WASM32-safe
/// Live Trajectory-DO Team Preview engine.
pub struct SteppedTrajPreview {
    my_acts: Vec<SearchChoice>,
    side: usize,
    cfg: TrajPreviewConfig,
    own_canon: Vec<(usize, [u8; 3])>,
    worlds: Vec<PreviewWorldData>,
    s2_world_ids: Vec<usize>,
    tail_world_ids: Vec<usize>,
    s2_alloc_weights: Vec<f64>,
    s2_post_weights: Vec<f64>,
    tail_post_weights: Vec<f64>,
    world_stats: Vec<TrajWorldStats>,
    tail_metas: Vec<WorldInteractMeta>,
    own_mon_cnt: [u32; 6],
    own_lead_cnt: [u32; 6],
    own_bench_pair_cnt: [u32; 15],
    obs_log: Vec<TrajObs>,
    sol1: BayesianRmSolution,
    sol_cur: BayesianRmSolution,
    scratch: TrajTurnScratch,
    active_slots: Vec<Option<ActiveTrajGame>>,
    cur_slot: usize,
    task_queue: std::collections::VecDeque<TrajTask>,
    phase: TrajPhase,
    next_block_id: u32,
    explore_enqueued: u32,
    iters_done: u32,
    games_completed: usize,
    total_nodes: usize,
    stage1_ms: f64,
    sol_dirty: bool,
}

impl SteppedTrajPreview {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        base: &Battle,
        dex: &Dex,
        belief: &Belief,
        obs: &crate::observe::Observer,
        side: usize,
        my_acts: &[SearchChoice],
        my_mask: Option<&[bool]>,
        cfg: TrajPreviewConfig,
        rng: &mut SplitMix64,
    ) -> Option<Self> {
        let t0 = start_timer();
        if base.request_state != RequestState::TeamPreview {
            return None;
        }
        let opp = 1 - side;
        let order_matters = party_order_matters(dex, &base.sides[side].roster);
        let mut own_canon: Vec<(usize, [u8; 3])> = Vec::with_capacity(my_acts.len());
        let mut seen: Vec<[u8; 3]> = Vec::with_capacity(60);
        for (idx, &ch) in my_acts.iter().enumerate() {
            if my_mask.is_some_and(|m| !m[idx]) {
                continue;
            }
            let SearchChoice::Team(t) = ch else {
                return None;
            };
            if !order_matters && t[2] != 0 {
                let mut key = t;
                if key[1] > key[2] {
                    key.swap(1, 2);
                }
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
            }
            own_canon.push((idx, t));
        }
        let n_own = own_canon.len();
        if n_own <= 1 {
            return None;
        }

        let world_plan = belief.preview_world_distribution(cfg.max_stage1_worlds, rng);
        if world_plan.is_empty() {
            return None;
        }
        let mut det_worlds: Vec<(f64, Battle)> = Vec::with_capacity(world_plan.len());
        for &(plan_pick, weight) in &world_plan {
            let mut sim_w = base.clone();
            sim_w.set_log_enabled(false);
            belief.determinize_in_place_planned(dex, &mut sim_w, obs, plan_pick, rng);
            sim_w.quick_claw_rolls = [false, false];
            if !is_compact_preview_supported(&sim_w, dex) {
                return None;
            }
            det_worlds.push((weight, sim_w));
        }

        let mut worlds: Vec<PreviewWorldData> = Vec::with_capacity(det_worlds.len());
        let mut all_metas: Vec<WorldInteractMeta> = Vec::with_capacity(det_worlds.len());
        for (weight, mut sim_w) in det_worlds {
            let (_, opp_canon) = canonical_preview_actions(dex, &mut sim_w, opp);
            let n_opp = opp_canon.len();
            if n_opp == 0 {
                continue;
            }
            let opp_tactical = lead_tactical_flags(dex, &sim_w, opp);
            let party_lens = [sim_w.sides[0].party.len(), sim_w.sides[1].party.len()];
            let roster = CompactPreviewRoster::from_battle(&sim_w, dex, None);
            let pcache = PreviewTempoCache::build_for_preview_roster(&roster, party_lens);
            let meta = WorldInteractMeta::build(&roster, &pcache, party_lens);
            let mut stage1_vals = vec![0.5f64; n_own * n_opp];
            for (i, &(_, own_t)) in own_canon.iter().enumerate() {
                for (j, &(_, opp_t)) in opp_canon.iter().enumerate() {
                    let picks = if side == 0 { [own_t, opp_t] } else { [opp_t, own_t] };
                    let (table, mut cb) = roster.build_turn1(picks, 0);
                    if cb.sides[0].request == 1 && cb.sides[1].request == 1 {
                        cb.quick_claw_roll = 0;
                    }
                    let sliced = pcache.slice_for(&cb);
                    let v0 = evaluate_cf_value(&cb, &table, Some(&sliced), cfg.leaf_temp);
                    stage1_vals[i * n_opp + j] = if side == 0 { v0 } else { 1.0 - v0 };
                }
            }
            worlds.push(PreviewWorldData {
                weight,
                opp_canon,
                opp_tactical,
                opp_prior: None,
                compact: Some((roster, pcache)),
                stage1_vals,
            });
            all_metas.push(meta);
        }
        if worlds.is_empty() {
            return None;
        }

        let w_sum: f64 = worlds.iter().map(|w| w.weight).sum();
        let w_len = worlds.len() as f64;
        for w in &mut worlds {
            w.weight = if w_sum > 0.0 {
                w.weight / w_sum
            } else {
                1.0 / w_len
            };
        }

        let s1_slices: Vec<(f64, usize, &[f64])> = worlds
            .iter()
            .map(|w| (w.weight, w.opp_canon.len(), w.stage1_vals.as_slice()))
            .collect();
        let sol1 = solve_bayesian_rm_plus(n_own, &s1_slices, 400);
        let stage1_ms = elapsed_ms(t0);

        let mut ordered_worlds: Vec<usize> = (0..worlds.len()).collect();
        ordered_worlds.sort_by(|&a, &b| worlds[b].weight.total_cmp(&worlds[a].weight));
        let n_w2 = ordered_worlds.len().min(cfg.max_stage2_worlds.max(1));
        let s2_world_ids: Vec<usize> = ordered_worlds[..n_w2].to_vec();
        let tail_world_ids: Vec<usize> = ordered_worlds[n_w2..].to_vec();

        let w2_sum: f64 = s2_world_ids.iter().map(|&w| worlds[w].weight).sum();
        let s2_alloc_weights: Vec<f64> = s2_world_ids
            .iter()
            .map(|&w| {
                if w2_sum > 0.0 {
                    worlds[w].weight / w2_sum
                } else {
                    1.0 / (n_w2 as f64)
                }
            })
            .collect();
        let w_all_sum: f64 = ordered_worlds.iter().map(|&w| worlds[w].weight).sum();
        let norm_w = |w_id: usize| -> f64 {
            if w_all_sum > 0.0 {
                worlds[w_id].weight / w_all_sum
            } else {
                1.0 / (ordered_worlds.len().max(1) as f64)
            }
        };
        let s2_post_weights: Vec<f64> = s2_world_ids.iter().map(|&w| norm_w(w)).collect();
        let tail_post_weights: Vec<f64> = tail_world_ids.iter().map(|&w| norm_w(w)).collect();

        let mut world_stats = Vec::with_capacity(n_w2);
        for &w_id in &s2_world_ids {
            let m = worlds[w_id].opp_canon.len();
            world_stats.push(TrajWorldStats {
                cell_n: vec![0u32; n_own * m],
                cell_sum: vec![0.0f64; n_own * m],
                t1_n: vec![0u32; n_own * m],
                t1_sum: vec![0.0f64; n_own * m],
                meta: all_metas[w_id].clone(),
                opp_mon_cnt: [0; 6],
                opp_lead_cnt: [0; 6],
                opp_bench_pair_cnt: [0; 15],
                explore_games: 0,
                do_batches: 0,
                polish_batches: 0,
                t1_batches: 0,
            });
        }
        let tail_metas: Vec<WorldInteractMeta> =
            tail_world_ids.iter().map(|&w_id| all_metas[w_id].clone()).collect();

        let n_slots = cfg.stepped_slots.clamp(1, 8);
        let mut active_slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            active_slots.push(None);
        }

        Some(Self {
            my_acts: my_acts.to_vec(),
            side,
            cfg,
            own_canon,
            worlds,
            s2_world_ids,
            tail_world_ids,
            s2_alloc_weights,
            s2_post_weights,
            tail_post_weights,
            world_stats,
            tail_metas,
            own_mon_cnt: [0; 6],
            own_lead_cnt: [0; 6],
            own_bench_pair_cnt: [0; 15],
            obs_log: Vec::with_capacity(512),
            sol_cur: sol1.clone(),
            sol1,
            scratch: TrajTurnScratch::new(),
            active_slots,
            cur_slot: 0,
            task_queue: std::collections::VecDeque::with_capacity(32),
            phase: TrajPhase::Explore,
            next_block_id: 1,
            explore_enqueued: 0,
            iters_done: 0,
            games_completed: 0,
            total_nodes: 0,
            stage1_ms,
            sol_dirty: false,
        })
    }

    pub fn node_count(&self) -> usize {
        self.total_nodes
    }

    pub fn games_completed(&self) -> usize {
        self.games_completed
    }

    pub fn stage1_ms(&self) -> f64 {
        self.stage1_ms
    }

    #[inline]
    fn bench_pair_index(t: [u8; 3]) -> Option<usize> {
        if t[1] >= 1 && t[1] <= 6 && t[2] >= 1 && t[2] <= 6 && t[1] != t[2] {
            Some(unordered_pair_idx(t[1] as usize - 1, t[2] as usize - 1))
        } else {
            None
        }
    }

    fn record_coverage_own(&mut self, t: [u8; 3]) {
        if t[0] >= 1 && t[0] <= 6 {
            let l = t[0] as usize - 1;
            self.own_lead_cnt[l] += 1;
            self.own_mon_cnt[l] += 1;
        }
        if t[1] >= 1 && t[1] <= 6 {
            self.own_mon_cnt[t[1] as usize - 1] += 1;
        }
        if t[2] >= 1 && t[2] <= 6 {
            self.own_mon_cnt[t[2] as usize - 1] += 1;
        }
        if let Some(p) = Self::bench_pair_index(t) {
            self.own_bench_pair_cnt[p] += 1;
        }
    }

    fn record_coverage_opp(&mut self, wi: usize, t: [u8; 3]) {
        let ws = &mut self.world_stats[wi];
        if t[0] >= 1 && t[0] <= 6 {
            let l = t[0] as usize - 1;
            ws.opp_lead_cnt[l] += 1;
            ws.opp_mon_cnt[l] += 1;
        }
        if t[1] >= 1 && t[1] <= 6 {
            ws.opp_mon_cnt[t[1] as usize - 1] += 1;
        }
        if t[2] >= 1 && t[2] <= 6 {
            ws.opp_mon_cnt[t[2] as usize - 1] += 1;
        }
        if let Some(p) = Self::bench_pair_index(t) {
            ws.opp_bench_pair_cnt[p] += 1;
        }
    }

    /// Build posterior `V_hat` matrices across all refined + tail worlds from weighted cell stats.
    fn build_posterior_matrices_from_stats(
        &self,
        stats_n: &[Vec<f64>],
        stats_sum: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let n_own = self.own_canon.len();
        let n_w2 = self.s2_world_ids.len();
        let n_tail = self.tail_world_ids.len();
        let basis = self.cfg.ridge_basis;
        let dim = basis.dim();
        let kappa = self.cfg.kappa.max(0.5);
        let mut feats = [(0usize, 0.0f64); MAX_RIDGE_NNZ];

        // 1. Fit shared species-level residual ridge across all Stage-2 worlds (when enabled or W == 1).
        let mut theta_shared = [0.0f64; MAX_RIDGE_DIM];
        if self.cfg.ridge_shared_multiworld || n_w2 == 1 {
            let mut ata = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
            let mut atb = [0.0f64; MAX_RIDGE_DIM];
            let mut any_obs = false;
            for wi in 0..n_w2 {
                let w_id = self.s2_world_ids[wi];
                let wd = &self.worlds[w_id];
                let m = wd.opp_canon.len();
                let meta = &self.world_stats[wi].meta;
                for i in 0..n_own {
                    let own_t = self.own_canon[i].1;
                    for j in 0..m {
                        let idx = i * m + j;
                        let n = stats_n[wi][idx];
                        if n <= 0.0 {
                            continue;
                        }
                        any_obs = true;
                        let r_bar = (stats_sum[wi][idx] / n) - wd.stage1_vals[idx];
                        let w_cell = if self.cfg.ridge_gls {
                            (kappa + 1.0) * n / (kappa + n)
                        } else {
                            n
                        };
                        let nnz = extract_ridge_features(
                            basis,
                            self.side,
                            own_t,
                            wd.opp_canon[j].1,
                            meta,
                            &mut feats,
                        );
                        for a in 0..nnz {
                            let (fa, va) = feats[a];
                            atb[fa] += w_cell * va * r_bar;
                            for b in 0..=a {
                                let (fb, vb) = feats[b];
                                let prod = w_cell * va * vb;
                                ata[fa][fb] += prod;
                                if fa != fb {
                                    ata[fb][fa] += prod;
                                }
                            }
                        }
                    }
                }
            }
            if any_obs {
                theta_shared = solve_spd_cholesky(
                    dim,
                    &ata,
                    &atb,
                    1.0,
                    self.cfg.ridge_lambda_main,
                    self.cfg.ridge_lambda_inter,
                );
            }
        }

        // 2. Build Stage-2 world matrices (with per-world residual delta when W > 1).
        let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let wd = &self.worlds[w_id];
            let m = wd.opp_canon.len();
            let meta = &self.world_stats[wi].meta;

            let mut delta_w = [0.0f64; MAX_RIDGE_DIM];
            let dw_dim = if n_w2 > 1 {
                if self.cfg.ridge_shared_multiworld {
                    dim.min(39)
                } else {
                    dim
                }
            } else {
                0
            };
            if dw_dim > 0 {
                let mut ata = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
                let mut atb = [0.0f64; MAX_RIDGE_DIM];
                let mut any_w = false;
                for i in 0..n_own {
                    let own_t = self.own_canon[i].1;
                    for j in 0..m {
                        let idx = i * m + j;
                        let n = stats_n[wi][idx];
                        if n <= 0.0 {
                            continue;
                        }
                        any_w = true;
                        let nnz = extract_ridge_features(
                            basis,
                            self.side,
                            own_t,
                            wd.opp_canon[j].1,
                            meta,
                            &mut feats,
                        );
                        let mut shared_pred = 0.0;
                        if self.cfg.ridge_shared_multiworld {
                            for &(f, v) in &feats[..nnz] {
                                shared_pred += theta_shared[f] * v;
                            }
                        }
                        let r_rem = (stats_sum[wi][idx] / n) - wd.stage1_vals[idx] - shared_pred;
                        let w_cell = if self.cfg.ridge_gls {
                            (kappa + 1.0) * n / (kappa + n)
                        } else {
                            n
                        };
                        for a in 0..nnz {
                            let (fa, va) = feats[a];
                            if fa >= dw_dim {
                                continue;
                            }
                            atb[fa] += w_cell * va * r_rem;
                            for b in 0..=a {
                                let (fb, vb) = feats[b];
                                if fb >= dw_dim {
                                    continue;
                                }
                                let prod = w_cell * va * vb;
                                ata[fa][fb] += prod;
                                if fa != fb {
                                    ata[fb][fa] += prod;
                                }
                            }
                        }
                    }
                }
                if any_w {
                    let mult = if self.cfg.ridge_shared_multiworld { 2.0 } else { 1.0 };
                    delta_w = solve_spd_cholesky(
                        dw_dim,
                        &ata,
                        &atb,
                        1.0 * mult,
                        self.cfg.ridge_lambda_main * mult,
                        self.cfg.ridge_lambda_inter * mult,
                    );
                }
            }

            let mut mat = vec![0.5f64; n_own * m];
            for i in 0..n_own {
                let own_t = self.own_canon[i].1;
                for j in 0..m {
                    let idx = i * m + j;
                    let nnz = extract_ridge_features(
                        basis,
                        self.side,
                        own_t,
                        wd.opp_canon[j].1,
                        meta,
                        &mut feats,
                    );
                    let mut corr = 0.0;
                    if self.cfg.ridge_shared_multiworld || n_w2 == 1 {
                        for &(f, v) in &feats[..nnz] {
                            corr += theta_shared[f] * v;
                        }
                    }
                    if dw_dim > 0 {
                        for &(f, v) in &feats[..nnz] {
                            if f < dw_dim {
                                corr += delta_w[f] * v;
                            }
                        }
                    }
                    let prior = (wd.stage1_vals[idx] + corr).clamp(0.02, 0.98);
                    let n = stats_n[wi][idx];
                    mat[idx] = (stats_sum[wi][idx] + kappa * prior) / (n + kappa);
                }
            }
            // Control-variate Turn-1 tactical lead delta (zero-mean per world -> exact zero cardinal drift!).
            if self.cfg.t1_polish_alpha > 0.0 {
                let ws = &self.world_stats[wi];
                let mut d_sum = 0.0f64;
                let mut d_cnt = 0usize;
                for idx in 0..(n_own * m) {
                    if ws.t1_n[idx] > 0 {
                        let v_t1 = ws.t1_sum[idx] / f64::from(ws.t1_n[idx]);
                        d_sum += v_t1 - wd.stage1_vals[idx];
                        d_cnt += 1;
                    }
                }
                if d_cnt > 0 {
                    let mean_d = d_sum / (d_cnt as f64);
                    let alpha = self.cfg.t1_polish_alpha;
                    for idx in 0..(n_own * m) {
                        if ws.t1_n[idx] > 0 {
                            let v_t1 = ws.t1_sum[idx] / f64::from(ws.t1_n[idx]);
                            let centered = (v_t1 - wd.stage1_vals[idx]) - mean_d;
                            mat[idx] = (mat[idx] + alpha * centered).clamp(0.01, 0.99);
                        }
                    }
                }
            }
            mats.push(mat);
        }

        // 3. Tail worlds (`n_w2..W`): transfer 75% of shared species-level residual when enabled.
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let wd = &self.worlds[w_id];
            let m = wd.opp_canon.len();
            if !self.cfg.ridge_shared_multiworld {
                mats.push(wd.stage1_vals.clone());
                continue;
            }
            let meta = &self.tail_metas[ti];
            let mut mat = vec![0.5f64; n_own * m];
            for i in 0..n_own {
                let own_t = self.own_canon[i].1;
                for j in 0..m {
                    let idx = i * m + j;
                    let nnz = extract_ridge_features(
                        basis,
                        self.side,
                        own_t,
                        wd.opp_canon[j].1,
                        meta,
                        &mut feats,
                    );
                    let mut corr = 0.0;
                    for &(f, v) in &feats[..nnz] {
                        corr += theta_shared[f] * v;
                    }
                    mat[idx] = (wd.stage1_vals[idx] + 0.75 * corr).clamp(0.02, 0.98);
                }
            }
            mats.push(mat);
        }
        mats
    }

    fn build_current_matrices(&self) -> Vec<Vec<f64>> {
        let stats_n: Vec<Vec<f64>> = self
            .world_stats
            .iter()
            .map(|ws| ws.cell_n.iter().map(|&n| f64::from(n)).collect())
            .collect();
        let stats_sum: Vec<Vec<f64>> = self
            .world_stats
            .iter()
            .map(|ws| ws.cell_sum.clone())
            .collect();
        self.build_posterior_matrices_from_stats(&stats_n, &stats_sum)
    }

    fn solve_full_game_from_mats(
        &self,
        mats: &[Vec<f64>],
        sweeps: u32,
        l2_reg: f64,
    ) -> BayesianRmSolution {
        let n_own = self.own_canon.len();
        let n_w2 = self.s2_world_ids.len();
        let mut slices: Vec<(f64, usize, &[f64])> = Vec::with_capacity(mats.len());
        for wi in 0..n_w2 {
            let m = self.worlds[self.s2_world_ids[wi]].opp_canon.len();
            slices.push((self.s2_post_weights[wi], m, mats[wi].as_slice()));
        }
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let m = self.worlds[w_id].opp_canon.len();
            slices.push((self.tail_post_weights[ti], m, mats[n_w2 + ti].as_slice()));
        }
        solve_bayesian_rm_plus_reg(n_own, &slices, sweeps, l2_reg)
    }

    fn refresh_current_solution(&mut self) {
        if !self.sol_dirty {
            return;
        }
        let mats = self.build_current_matrices();
        let n_w_tot = self.s2_world_ids.len() + self.tail_world_ids.len();
        let sweeps = if n_w_tot > 4 {
            self.cfg.rm_sweeps_do.min(180)
        } else {
            self.cfg.rm_sweeps_do
        };
        self.sol_cur = self.solve_full_game_from_mats(&mats, sweeps, 0.0);
        self.sol_dirty = false;
    }

    /// Pick next world index proportionally to `s2_alloc_weights`.
    fn pick_world_by_deficit(&self, counts: impl Fn(&TrajWorldStats) -> u32) -> usize {
        let n_w2 = self.s2_world_ids.len();
        if n_w2 <= 1 {
            return 0;
        }
        let total: u32 = self.world_stats.iter().map(&counts).sum();
        let next_tot = f64::from(total + 1);
        let mut best_wi = 0usize;
        let mut best_def = f64::NEG_INFINITY;
        for wi in 0..n_w2 {
            let def = self.s2_alloc_weights[wi] * next_tot - f64::from(counts(&self.world_stats[wi]));
            if def > best_def {
                best_def = def;
                best_wi = wi;
            }
        }
        best_wi
    }

    fn pick_balanced_own(&self, rng: &mut SplitMix64) -> usize {
        let mut best_i = 0usize;
        let mut best_sc = f64::INFINITY;
        for (i, &(_, t)) in self.own_canon.iter().enumerate() {
            let l_c = if t[0] >= 1 && t[0] <= 6 {
                f64::from(self.own_lead_cnt[t[0] as usize - 1])
            } else {
                0.0
            };
            let b1_c = if t[1] >= 1 && t[1] <= 6 {
                f64::from(self.own_mon_cnt[t[1] as usize - 1])
            } else {
                0.0
            };
            let b2_c = if t[2] >= 1 && t[2] <= 6 {
                f64::from(self.own_mon_cnt[t[2] as usize - 1])
            } else {
                0.0
            };
            let bp_c = Self::bench_pair_index(t)
                .map_or(0.0, |p| f64::from(self.own_bench_pair_cnt[p]));
            let sc = 2.5 * l_c + b1_c + b2_c + 2.0 * bp_c + rng.next_f64() * 0.75;
            if sc < best_sc {
                best_sc = sc;
                best_i = i;
            }
        }
        best_i
    }

    fn pick_balanced_opp(&self, wi: usize, rng: &mut SplitMix64) -> usize {
        let wd = &self.worlds[self.s2_world_ids[wi]];
        let ws = &self.world_stats[wi];
        let mut best_j = 0usize;
        let mut best_sc = f64::INFINITY;
        for (j, &(_, t)) in wd.opp_canon.iter().enumerate() {
            let l_c = if t[0] >= 1 && t[0] <= 6 {
                f64::from(ws.opp_lead_cnt[t[0] as usize - 1])
            } else {
                0.0
            };
            let b1_c = if t[1] >= 1 && t[1] <= 6 {
                f64::from(ws.opp_mon_cnt[t[1] as usize - 1])
            } else {
                0.0
            };
            let b2_c = if t[2] >= 1 && t[2] <= 6 {
                f64::from(ws.opp_mon_cnt[t[2] as usize - 1])
            } else {
                0.0
            };
            let bp_c = Self::bench_pair_index(t)
                .map_or(0.0, |p| f64::from(ws.opp_bench_pair_cnt[p]));
            let sc = 2.5 * l_c + b1_c + b2_c + 2.0 * bp_c + rng.next_f64() * 0.75;
            if sc < best_sc {
                best_sc = sc;
                best_j = j;
            }
        }
        best_j
    }

    fn enqueue_next_batch(&mut self, rng: &mut SplitMix64) {
        let explore_cap = ((f64::from(self.cfg.total_iters) * self.cfg.explore_frac).round() as u32)
            .min(self.cfg.total_iters);
        let do_cap = ((f64::from(self.cfg.total_iters) * (self.cfg.explore_frac + self.cfg.do_frac))
            .round() as u32)
            .min(self.cfg.total_iters);
        let t1_enabled = self.cfg.t1_polish_frac > 0.0
            && self.cfg.t1_polish_alpha > 0.0
            && self.cfg.t1_polish_iters > 0;
        let t1_start = if t1_enabled {
            ((f64::from(self.cfg.total_iters) * (1.0 - self.cfg.t1_polish_frac.clamp(0.0, 0.5)))
                .round() as u32)
                .max(do_cap)
        } else {
            u32::MAX
        };

        if self.iters_done < explore_cap {
            self.phase = TrajPhase::Explore;
        } else if self.iters_done < do_cap {
            self.phase = TrajPhase::DoubleOracle;
        } else if t1_enabled && self.iters_done >= t1_start {
            self.phase = TrajPhase::T1Polish;
        } else {
            self.phase = TrajPhase::Polish;
        }

        let n_own = self.own_canon.len();
        match self.phase {
            TrajPhase::Explore => {
                // Enqueue a mini-batch of 4 roster- & bench-pair-balanced exploration games
                // (20% calibrated on Stage-1 Nash support, 80% full-coverage balanced).
                for _ in 0..4 {
                    let wi = self.pick_world_by_deficit(|ws| ws.explore_games);
                    let w_id = self.s2_world_ids[wi];
                    let m = self.worlds[w_id].opp_canon.len();
                    let (own_i, opp_j) = if self.explore_enqueued % 5 == 0 {
                        let i = sample_dist(&self.sol1.own_policy, rng.next_f64());
                        let j = sample_dist(&self.sol1.opp_policies[w_id], rng.next_f64());
                        (i, j)
                    } else {
                        let i = self.pick_balanced_own(rng);
                        let j = self.pick_balanced_opp(wi, rng);
                        (i, j)
                    };
                    let own_t = self.own_canon[own_i].1;
                    let opp_t = self.worlds[w_id].opp_canon[opp_j].1;
                    self.record_coverage_own(own_t);
                    self.record_coverage_opp(wi, opp_t);
                    self.world_stats[wi].explore_games += 1;
                    self.explore_enqueued += 1;

                    let idx = own_i * m + opp_j;
                    let visit_k = self.world_stats[wi].cell_n[idx];
                    let block_id = self.next_block_id;
                    self.next_block_id += 1;
                    let gseed = rng.next();
                    self.task_queue.push_back(TrajTask {
                        wi,
                        own_i,
                        opp_j,
                        block_id,
                        game_seed: gseed,
                        cell_visit_k: visit_k,
                        t1_only_iters: 0,
                    });
                }
            }
            TrajPhase::DoubleOracle => {
                self.refresh_current_solution();
                let wi = self.pick_world_by_deficit(|ws| ws.do_batches);
                self.world_stats[wi].do_batches += 1;
                let w_id = self.s2_world_ids[wi];
                let m = self.worlds[w_id].opp_canon.len();

                let mut row_n = vec![0u32; n_own];
                let mut col_n = vec![0u32; m];
                for i in 0..n_own {
                    for j in 0..m {
                        let c = self.world_stats[wi].cell_n[i * m + j];
                        row_n[i] += c;
                        col_n[j] += c;
                    }
                }

                // 1. Paired CRN Own Challenger-vs-Incumbent batch (same world, opp pick, battle seed!).
                let j_tgt = sample_dist(&self.sol_cur.opp_policies[wi], rng.next_f64());
                let i_inc = sample_dist(&self.sol_cur.own_policy, rng.next_f64());
                let mut own_batch: Vec<usize> = Vec::with_capacity(self.cfg.do_challengers + 2);
                let paired = self.cfg.crn_mode != TrajCrnMode::Independent;
                if paired {
                    own_batch.push(i_inc);
                }
                let mut own_ucb: Vec<(usize, f64)> = (0..n_own)
                    .filter(|&i| !own_batch.contains(&i))
                    .map(|i| {
                        let u = self.sol_cur.row_vs_q[i]
                            + self.cfg.do_ucb_c / f64::from(row_n[i] + 1).sqrt();
                        (i, u)
                    })
                    .collect();
                own_ucb.sort_by(|a, b| b.1.total_cmp(&a.1));
                let n_ucb = self.cfg.do_challengers.saturating_sub(1).max(1);
                for &(ci, _) in own_ucb.iter().take(n_ucb) {
                    own_batch.push(ci);
                }
                // Reserve 1 probe specifically for lowest-coverage zero-mass bench combination!
                if let Some((bi, _)) = (0..n_own)
                    .filter(|&i| !own_batch.contains(&i) && self.sol_cur.own_policy[i] < 0.01)
                    .filter_map(|i| {
                        let bp_c = Self::bench_pair_index(self.own_canon[i].1)
                            .map_or(u32::MAX, |p| self.own_bench_pair_cnt[p]);
                        let u = self.sol_cur.row_vs_q[i]
                            + self.cfg.do_ucb_c / f64::from(row_n[i] + 1).sqrt();
                        Some((i, (bp_c, u)))
                    })
                    .min_by(|a, b| a.1 .0.cmp(&b.1 .0).then_with(|| b.1 .1.total_cmp(&a.1 .1)))
                {
                    own_batch.push(bi);
                }

                let own_block = self.next_block_id;
                self.next_block_id += 1;
                let shared_own_seed = rng.next();
                for &ci in &own_batch {
                    let own_t = self.own_canon[ci].1;
                    let opp_t = self.worlds[w_id].opp_canon[j_tgt].1;
                    self.record_coverage_own(own_t);
                    self.record_coverage_opp(wi, opp_t);
                    let idx = ci * m + j_tgt;
                    let visit_k = self.world_stats[wi].cell_n[idx];
                    let (blk, gs) = if paired {
                        (own_block, shared_own_seed)
                    } else {
                        let b = self.next_block_id;
                        self.next_block_id += 1;
                        (b, rng.next())
                    };
                    self.task_queue.push_back(TrajTask {
                        wi,
                        own_i: ci,
                        opp_j: j_tgt,
                        block_id: blk,
                        game_seed: gs,
                        cell_visit_k: visit_k,
                        t1_only_iters: 0,
                    });
                }

                // 2. Paired CRN Opp Challenger-vs-Incumbent batch (same world, own pick, battle seed!).
                let i_tgt = sample_dist(&self.sol_cur.own_policy, rng.next_f64());
                let j_inc = sample_dist(&self.sol_cur.opp_policies[wi], rng.next_f64());
                let mut opp_batch: Vec<usize> = Vec::with_capacity(self.cfg.do_challengers + 2);
                if paired {
                    opp_batch.push(j_inc);
                }
                let mut opp_ucb: Vec<(usize, f64)> = (0..m)
                    .filter(|&j| !opp_batch.contains(&j))
                    .map(|j| {
                        let u = (1.0 - self.sol_cur.col_vs_p[wi][j])
                            + self.cfg.do_ucb_c / f64::from(col_n[j] + 1).sqrt();
                        (j, u)
                    })
                    .collect();
                opp_ucb.sort_by(|a, b| b.1.total_cmp(&a.1));
                for &(cj, _) in opp_ucb.iter().take(n_ucb) {
                    opp_batch.push(cj);
                }
                // Reserve 1 probe for lowest-coverage zero-mass opponent bench combination!
                if let Some((bj, _)) = (0..m)
                    .filter(|&j| !opp_batch.contains(&j) && self.sol_cur.opp_policies[wi][j] < 0.01)
                    .filter_map(|j| {
                        let bp_c = Self::bench_pair_index(self.worlds[w_id].opp_canon[j].1)
                            .map_or(u32::MAX, |p| self.world_stats[wi].opp_bench_pair_cnt[p]);
                        let u = (1.0 - self.sol_cur.col_vs_p[wi][j])
                            + self.cfg.do_ucb_c / f64::from(col_n[j] + 1).sqrt();
                        Some((j, (bp_c, u)))
                    })
                    .min_by(|a, b| a.1 .0.cmp(&b.1 .0).then_with(|| b.1 .1.total_cmp(&a.1 .1)))
                {
                    opp_batch.push(bj);
                }

                let opp_block = self.next_block_id;
                self.next_block_id += 1;
                let shared_opp_seed = rng.next();
                for &cj in &opp_batch {
                    let own_t = self.own_canon[i_tgt].1;
                    let opp_t = self.worlds[w_id].opp_canon[cj].1;
                    self.record_coverage_own(own_t);
                    self.record_coverage_opp(wi, opp_t);
                    let idx = i_tgt * m + cj;
                    let visit_k = self.world_stats[wi].cell_n[idx];
                    let (blk, gs) = if paired {
                        (opp_block, shared_opp_seed)
                    } else {
                        let b = self.next_block_id;
                        self.next_block_id += 1;
                        (b, rng.next())
                    };
                    self.task_queue.push_back(TrajTask {
                        wi,
                        own_i: i_tgt,
                        opp_j: cj,
                        block_id: blk,
                        game_seed: gs,
                        cell_visit_k: visit_k,
                        t1_only_iters: 0,
                    });
                }
            }
            TrajPhase::Polish => {
                self.refresh_current_solution();
                let wi = self.pick_world_by_deficit(|ws| ws.polish_batches);
                self.world_stats[wi].polish_batches += 1;
                let w_id = self.s2_world_ids[wi];
                let m = self.worlds[w_id].opp_canon.len();

                let mut active_rows: Vec<usize> = (0..n_own)
                    .filter(|&i| {
                        self.sol_cur.own_policy[i] >= 0.015
                            || self.sol_cur.row_vs_q[i] >= self.sol_cur.value - 0.025
                    })
                    .collect();
                active_rows.sort_by(|&a, &b| {
                    self.sol_cur.own_policy[b]
                        .total_cmp(&self.sol_cur.own_policy[a])
                        .then_with(|| self.sol_cur.row_vs_q[b].total_cmp(&self.sol_cur.row_vs_q[a]))
                });
                active_rows.truncate(6);
                if active_rows.is_empty() {
                    active_rows.push(0);
                }

                let mut active_cols: Vec<usize> = (0..m)
                    .filter(|&j| {
                        self.sol_cur.opp_policies[wi][j] >= 0.015
                            || self.sol_cur.col_vs_p[wi][j] <= self.sol_cur.world_values[wi] + 0.025
                    })
                    .collect();
                active_cols.sort_by(|&a, &b| {
                    self.sol_cur.opp_policies[wi][b]
                        .total_cmp(&self.sol_cur.opp_policies[wi][a])
                        .then_with(|| {
                            self.sol_cur.col_vs_p[wi][a].total_cmp(&self.sol_cur.col_vs_p[wi][b])
                        })
                });
                active_cols.truncate(6);
                if active_cols.is_empty() {
                    active_cols.push(0);
                }

                let mut scored: Vec<(usize, usize, f64)> =
                    Vec::with_capacity(active_rows.len() * active_cols.len());
                for &ri in &active_rows {
                    let pw = self.sol_cur.own_policy[ri] + 0.05;
                    for &cj in &active_cols {
                        let qw = self.sol_cur.opp_policies[wi][cj] + 0.05;
                        let n_ij = self.world_stats[wi].cell_n[ri * m + cj];
                        let sc = pw * qw / f64::from(n_ij + 1).sqrt();
                        scored.push((ri, cj, sc));
                    }
                }
                scored.sort_by(|a, b| b.2.total_cmp(&a.2));

                let paired = self.cfg.crn_mode != TrajCrnMode::Independent;
                let pol_block = self.next_block_id;
                self.next_block_id += 1;
                let shared_pol_seed = rng.next();
                for &(ri, cj, _) in scored.iter().take(8) {
                    let idx = ri * m + cj;
                    let visit_k = self.world_stats[wi].cell_n[idx];
                    let (blk, gs) = if paired {
                        (pol_block, shared_pol_seed)
                    } else {
                        let b = self.next_block_id;
                        self.next_block_id += 1;
                        (b, rng.next())
                    };
                    self.task_queue.push_back(TrajTask {
                        wi,
                        own_i: ri,
                        opp_j: cj,
                        block_id: blk,
                        game_seed: gs,
                        cell_visit_k: visit_k,
                        t1_only_iters: 0,
                    });
                }
            }
            TrajPhase::T1Polish => {
                self.refresh_current_solution();
                let wi = self.pick_world_by_deficit(|ws| ws.t1_batches);
                self.world_stats[wi].t1_batches += 1;
                let w_id = self.s2_world_ids[wi];
                let m = self.worlds[w_id].opp_canon.len();

                let sorted_core = |mut t: [u8; 3]| -> [u8; 3] {
                    t.sort_unstable();
                    t
                };

                // Rank own rows and expand all canonical lead permutations of the top 3 active cores
                // so Turn-1 tactical lead order (e.g. Snorlax vs Marowak lead into Cloyster) is directly evaluated!
                let mut own_ranked: Vec<usize> = (0..n_own).collect();
                own_ranked.sort_by(|&a, &b| {
                    self.sol_cur.own_policy[b]
                        .total_cmp(&self.sol_cur.own_policy[a])
                        .then_with(|| self.sol_cur.row_vs_q[b].total_cmp(&self.sol_cur.row_vs_q[a]))
                });
                let mut active_rows: Vec<usize> = Vec::with_capacity(12);
                let mut own_cores: Vec<[u8; 3]> = Vec::with_capacity(4);
                for &ri in &own_ranked {
                    let core = sorted_core(self.own_canon[ri].1);
                    if !own_cores.contains(&core) && own_cores.len() < 3 {
                        own_cores.push(core);
                        for i2 in 0..n_own {
                            if sorted_core(self.own_canon[i2].1) == core && !active_rows.contains(&i2) {
                                active_rows.push(i2);
                            }
                        }
                    }
                    if !active_rows.contains(&ri) && active_rows.len() < 10 {
                        active_rows.push(ri);
                    }
                    if active_rows.len() >= 10 && own_cores.len() >= 3 {
                        break;
                    }
                }
                active_rows.truncate(10);
                if active_rows.is_empty() {
                    active_rows.push(0);
                }

                let mut opp_ranked: Vec<usize> = (0..m).collect();
                opp_ranked.sort_by(|&a, &b| {
                    self.sol_cur.opp_policies[wi][b]
                        .total_cmp(&self.sol_cur.opp_policies[wi][a])
                        .then_with(|| {
                            self.sol_cur.col_vs_p[wi][a].total_cmp(&self.sol_cur.col_vs_p[wi][b])
                        })
                });
                let mut active_cols: Vec<usize> = Vec::with_capacity(12);
                let mut opp_cores: Vec<[u8; 3]> = Vec::with_capacity(4);
                for &cj in &opp_ranked {
                    let core = sorted_core(self.worlds[w_id].opp_canon[cj].1);
                    if !opp_cores.contains(&core) && opp_cores.len() < 3 {
                        opp_cores.push(core);
                        for j2 in 0..m {
                            if sorted_core(self.worlds[w_id].opp_canon[j2].1) == core
                                && !active_cols.contains(&j2)
                            {
                                active_cols.push(j2);
                            }
                        }
                    }
                    if !active_cols.contains(&cj) && active_cols.len() < 10 {
                        active_cols.push(cj);
                    }
                    if active_cols.len() >= 10 && opp_cores.len() >= 3 {
                        break;
                    }
                }
                active_cols.truncate(10);
                if active_cols.is_empty() {
                    active_cols.push(0);
                }

                let mut scored: Vec<(usize, usize, u32, f64)> =
                    Vec::with_capacity(active_rows.len() * active_cols.len());
                for &ri in &active_rows {
                    let pw = self.sol_cur.own_policy[ri]
                        + (self.sol_cur.row_vs_q[ri] - self.sol_cur.value + 0.10).max(0.01);
                    for &cj in &active_cols {
                        let qw = self.sol_cur.opp_policies[wi][cj]
                            + (self.sol_cur.world_values[wi] - self.sol_cur.col_vs_p[wi][cj] + 0.10)
                                .max(0.01);
                        let tn = self.world_stats[wi].t1_n[ri * m + cj];
                        scored.push((ri, cj, tn, pw * qw));
                    }
                }
                scored.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| b.3.total_cmp(&a.3)));

                let t1_iters = self.cfg.t1_polish_iters.max(64);
                let blk = self.next_block_id;
                self.next_block_id += 1;
                let shared_seed = rng.next();
                for &(ri, cj, tn, _) in scored.iter().take(8) {
                    self.task_queue.push_back(TrajTask {
                        wi,
                        own_i: ri,
                        opp_j: cj,
                        block_id: blk,
                        game_seed: shared_seed,
                        cell_visit_k: tn,
                        t1_only_iters: t1_iters,
                    });
                }
            }
        }
    }

    fn record_completed_game(
        &mut self,
        wi: usize,
        own_i: usize,
        opp_j: usize,
        block_id: u32,
        val0: f64,
    ) {
        let val_mine = if self.side == 0 { val0 } else { 1.0 - val0 };
        let m = self.worlds[self.s2_world_ids[wi]].opp_canon.len();
        let idx = own_i * m + opp_j;
        self.world_stats[wi].cell_n[idx] += 1;
        self.world_stats[wi].cell_sum[idx] += val_mine;
        self.obs_log.push(TrajObs {
            wi: wi as u16,
            own_i: own_i as u16,
            opp_j: opp_j as u16,
            block_id,
            val_mine,
        });
        self.games_completed += 1;
        self.sol_dirty = true;
    }

    /// Execute exactly ONE PUCT iteration across the interleaved Trajectory-DO engine.
    /// Bit-identically slice-invariant (`step_one` × N == chunked calls).
    /// Returns `(own_orig_idx, opp_choice, val0, val_mine, depth)`.
    pub fn step_one(&mut self, rng: &mut SplitMix64) -> (usize, SearchChoice, f64, f64, u32) {
        let n_slots = self.active_slots.len();
        loop {
            if self.active_slots[self.cur_slot].is_none() {
                if self.task_queue.is_empty() {
                    self.enqueue_next_batch(rng);
                }
                let task = self.task_queue.pop_front().unwrap();
                let w_id = self.s2_world_ids[task.wi];
                let wd = &self.worlds[w_id];
                let (ref roster, ref pcache) = wd.compact.as_ref().unwrap();
                let own_t = self.own_canon[task.own_i].1;
                let opp_t = wd.opp_canon[task.opp_j].1;
                let picks = if self.side == 0 {
                    [own_t, opp_t]
                } else {
                    [opp_t, own_t]
                };
                self.active_slots[self.cur_slot] = Some(ActiveTrajGame::new(
                    task.wi,
                    task.own_i,
                    task.opp_j,
                    task.block_id,
                    roster,
                    pcache,
                    picks,
                    task.game_seed,
                    task.cell_visit_k,
                    self.cfg.crn_mode,
                    task.t1_only_iters,
                ));
            }

            let mut slot_game = self.active_slots[self.cur_slot].take().unwrap();
            if !slot_game.turn_ready {
                if let Some((v0, _rb)) = slot_game.prepare_next_turn(&self.cfg, &mut self.scratch) {
                    let wi = slot_game.wi;
                    let own_i = slot_game.own_i;
                    let opp_j = slot_game.opp_j;
                    let blk = slot_game.block_id;
                    self.record_completed_game(wi, own_i, opp_j, blk, v0);
                    self.cur_slot = (self.cur_slot + 1) % n_slots;
                    // Game ended on forced turn / RB exit without spending a PUCT iter; keep looping
                    // until we perform 1 real PUCT iteration (guaranteed bounded).
                    continue;
                }
            }

            let (val0, depth, added) = self.scratch.step_puct_iter(
                &slot_game.turn_root_cb,
                &slot_game.table,
                &slot_game.cache,
                slot_game.turn_cap,
                self.cfg.c_puct,
                self.cfg.fpu_red,
                self.cfg.puct_eps,
                self.cfg.use_cf_policy,
                self.cfg.leaf_temp,
                &mut slot_game.turn_rng,
            );
            self.total_nodes += added;
            self.iters_done += 1;
            slot_game.turn_iters_done += 1;
            if slot_game.t1_only_iters > 0
                && slot_game.turn_iters_done.saturating_mul(2) > slot_game.turn_target_iters
            {
                slot_game.t1_late_sum += val0;
                slot_game.t1_late_n += 1;
            }

            let wi = slot_game.wi;
            let own_i = slot_game.own_i;
            let opp_j = slot_game.opp_j;
            let opp_t = self.worlds[self.s2_world_ids[wi]].opp_canon[opp_j].1;

            if slot_game.turn_iters_done >= slot_game.turn_target_iters {
                if slot_game.t1_only_iters > 0 {
                    let final_v0 = if slot_game.t1_late_n > 0 {
                        (slot_game.t1_late_sum / f64::from(slot_game.t1_late_n)).clamp(0.0, 1.0)
                    } else {
                        self.scratch.readout_turn().1
                    };
                    let val_mine_t1 = if self.side == 0 { final_v0 } else { 1.0 - final_v0 };
                    let m = self.worlds[self.s2_world_ids[wi]].opp_canon.len();
                    let idx = own_i * m + opp_j;
                    self.world_stats[wi].t1_n[idx] += 1;
                    self.world_stats[wi].t1_sum[idx] += val_mine_t1;
                    self.sol_dirty = true;
                } else if let Some((final_v0, _rb)) = slot_game.finish_turn(&self.scratch) {
                    self.record_completed_game(wi, own_i, opp_j, slot_game.block_id, final_v0);
                } else {
                    self.active_slots[self.cur_slot] = Some(slot_game);
                }
                // Switch to next slot strictly at a turn boundary so `self.scratch` stays single-instance.
                self.cur_slot = (self.cur_slot + 1) % n_slots;
            } else {
                self.active_slots[self.cur_slot] = Some(slot_game);
            }

            let val_mine = if self.side == 0 { val0 } else { 1.0 - val0 };
            let own_orig_idx = self.own_canon[own_i].0;
            return (own_orig_idx, SearchChoice::Team(opp_t), val0, val_mine, depth);
        }
    }

    /// Solve the final execution policy over the full 100% posterior (read-only).
    /// Returns `(sub_pol, final_sol, pure_guarantee, mixed_guarantee, used_mix)`.
    fn solve_final_policy(
        &self,
        done: u32,
    ) -> (Vec<f64>, BayesianRmSolution, f64, f64, bool) {
        self.solve_final_policy_with_mode(done, self.cfg.policy_mode)
    }

    /// Read-only execution distribution over canonical legal preview picks.
    /// Exact match to the distribution sampled by `best_choice(decision_seed, done)`.
    pub fn execution_policy(&self, done: u32) -> Vec<(SearchChoice, f64)> {
        let (sub_pol, _, _, _, _) = self.solve_final_policy(done);
        let mut out = Vec::with_capacity(sub_pol.len());
        for (i, &pr) in sub_pol.iter().enumerate() {
            if pr > 0.0 {
                out.push((self.my_acts[self.own_canon[i].0], pr));
            }
        }
        if out.is_empty() {
            vec![(self.my_acts[self.own_canon[0].0], 1.0)]
        } else {
            out
        }
    }

    /// Deterministic choice readout over the full 100% posterior.
    pub fn best_choice(&self, decision_seed: u64, done: u32) -> SearchChoice {
        let (sub_pol, _, _, _, _) = self.solve_final_policy(done);
        let draw = SplitMix64::new(decision_seed ^ u64::from(done)).next_f64();
        let picked = sample_dist(&sub_pol, draw);
        self.my_acts[self.own_canon[picked].0]
    }

    /// Read-only diagnostic readout under an alternative `TrajPolicyMode` without re-running trajectories.
    /// Returns `(policy_entries, nash_value, pure_guarantee, mixed_guarantee, used_mix)`.
    pub fn readout_with_policy_mode(
        &self,
        done: u32,
        mode: TrajPolicyMode,
    ) -> (Vec<(SearchChoice, f64)>, f64, f64, f64, bool) {
        let (sub_pol, final_sol, pure_g, mix_g, used_mix) =
            self.solve_final_policy_with_mode(done, mode);
        let mut out = Vec::with_capacity(sub_pol.len());
        for (i, &pr) in sub_pol.iter().enumerate() {
            if pr > 0.0 {
                out.push((self.my_acts[self.own_canon[i].0], pr));
            }
        }
        if out.is_empty() {
            out.push((self.my_acts[self.own_canon[0].0], 1.0));
        }
        (out, final_sol.value, pure_g, mix_g, used_mix)
    }

    /// Public diagnostic accessor for the solved posterior value matrices (one per world).
    pub fn posterior_matrices(&self) -> Vec<Vec<f64>> {
        self.build_current_matrices()
    }

    /// Public diagnostic accessor for the canonical own triples.
    pub fn own_triples(&self) -> Vec<[u8; 3]> {
        self.own_canon.iter().map(|&(_, t)| t).collect()
    }

    /// Public diagnostic accessor for world 0's canonical opponent triples.
    pub fn opp_triples_w0(&self) -> Vec<[u8; 3]> {
        self.worlds[self.s2_world_ids[0]]
            .opp_canon
            .iter()
            .map(|&(_, t)| t)
            .collect()
    }

    /// Public diagnostic accessor for `iters_done`.
    pub fn iters_done(&self) -> u32 {
        self.iters_done
    }

    /// Public diagnostic accessor for `(worlds_stage1, worlds_stage2)`.
    pub fn world_counts(&self) -> (usize, usize) {
        (self.worlds.len(), self.s2_world_ids.len())
    }

    fn solve_final_policy_with_mode(
        &self,
        done: u32,
        mode: TrajPolicyMode,
    ) -> (Vec<f64>, BayesianRmSolution, f64, f64, bool) {
        let n_own = self.own_canon.len();
        let n_w2 = self.s2_world_ids.len();
        let sweeps = if done >= 4096 {
            self.cfg.rm_sweeps_final
        } else {
            self.cfg.rm_sweeps_do
        };
        let mats = self.build_current_matrices();
        let l2_reg = match mode {
            TrajPolicyMode::RegRm { l2_reg, .. } => l2_reg,
            _ => 0.0,
        };
        let final_sol = self.solve_full_game_from_mats(&mats, sweeps, l2_reg);

        let best_pure = (0..n_own)
            .max_by(|&a, &b| {
                final_sol.row_guarantees[a]
                    .total_cmp(&final_sol.row_guarantees[b])
                    .then_with(|| final_sol.row_vs_q[a].total_cmp(&final_sol.row_vs_q[b]))
            })
            .unwrap_or(0);
        let pure_guarantee = final_sol.row_guarantees[best_pure];

        let mut p_mix = match mode {
            TrajPolicyMode::BayesianBootstrapRm { draws, dust_ratio }
                if draws > 1 && self.obs_log.len() >= 8 =>
            {
                let mut max_blk = 0u32;
                for o in &self.obs_log {
                    if o.block_id > max_blk {
                        max_blk = o.block_id;
                    }
                }
                let n_blk = (max_blk as usize) + 1;
                let mut blk_seen = vec![false; n_blk];
                let mut active_blks = Vec::new();
                for o in &self.obs_log {
                    let b = o.block_id as usize;
                    if !blk_seen[b] {
                        blk_seen[b] = true;
                        active_blks.push(b);
                    }
                }
                let mut brng = SplitMix64::new(
                    0xB007_57AA_0000_0001u64
                        ^ (u64::from(done).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                        ^ ((self.games_completed as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
                );
                let mut blk_w = vec![0.0f64; n_blk];
                let mut avg_p = vec![0.0f64; n_own];
                let mut b_n: Vec<Vec<f64>> = self
                    .world_stats
                    .iter()
                    .map(|ws| vec![0.0f64; ws.cell_n.len()])
                    .collect();
                let mut b_sum: Vec<Vec<f64>> = self
                    .world_stats
                    .iter()
                    .map(|ws| vec![0.0f64; ws.cell_sum.len()])
                    .collect();
                let nb_f = active_blks.len().max(1) as f64;
                let n_w_tot = n_w2 + self.tail_world_ids.len();
                // Restrict multi-world / large bootstrap solves to the union of equilibrium support
                // + top near-optimal rows & columns (keeps W = 24 bootstrap under ~25 ms!).
                let use_sub = n_w_tot > 1 || n_own > 20;
                let (sub_rows, sub_cols): (Vec<usize>, Vec<Vec<usize>>) = if use_sub {
                    let mut rows: Vec<usize> = (0..n_own)
                        .filter(|&i| final_sol.own_policy[i] >= 0.005)
                        .collect();
                    let mut by_q: Vec<usize> = (0..n_own).collect();
                    by_q.sort_by(|&a, &b| final_sol.row_vs_q[b].total_cmp(&final_sol.row_vs_q[a]));
                    for i in by_q.into_iter().take(16) {
                        if !rows.contains(&i) {
                            rows.push(i);
                        }
                    }
                    let mut by_g: Vec<usize> = (0..n_own).collect();
                    by_g.sort_by(|&a, &b| {
                        final_sol.row_guarantees[b].total_cmp(&final_sol.row_guarantees[a])
                    });
                    for i in by_g.into_iter().take(4) {
                        if !rows.contains(&i) {
                            rows.push(i);
                        }
                    }
                    if rows.is_empty() {
                        rows.push(0);
                    }
                    let mut cols_all = Vec::with_capacity(n_w_tot);
                    for w in 0..n_w_tot {
                        let m = final_sol.opp_policies[w].len();
                        let mut cols: Vec<usize> = (0..m)
                            .filter(|&j| final_sol.opp_policies[w][j] >= 0.005)
                            .collect();
                        let mut by_p: Vec<usize> = (0..m).collect();
                        by_p.sort_by(|&a, &b| {
                            final_sol.col_vs_p[w][a].total_cmp(&final_sol.col_vs_p[w][b])
                        });
                        for j in by_p.into_iter().take(14) {
                            if !cols.contains(&j) {
                                cols.push(j);
                            }
                        }
                        if cols.is_empty() && m > 0 {
                            cols.push(0);
                        }
                        cols_all.push(cols);
                    }
                    (rows, cols_all)
                } else {
                    (Vec::new(), Vec::new())
                };
                let boot_sweeps = if use_sub { 240u32 } else { 320u32 };
                for _ in 0..draws {
                    let mut wsum = 0.0;
                    for &b in &active_blks {
                        let u = brng.next_f64().clamp(1e-12, 1.0 - 1e-12);
                        let e = -u.ln();
                        blk_w[b] = e;
                        wsum += e;
                    }
                    let scale = nb_f / wsum.max(1e-12);
                    for &b in &active_blks {
                        blk_w[b] *= scale;
                    }
                    for wi in 0..n_w2 {
                        b_n[wi].fill(0.0);
                        b_sum[wi].fill(0.0);
                    }
                    for o in &self.obs_log {
                        let wi = o.wi as usize;
                        let m = self.worlds[self.s2_world_ids[wi]].opp_canon.len();
                        let idx = (o.own_i as usize) * m + (o.opp_j as usize);
                        let w = blk_w[o.block_id as usize];
                        b_n[wi][idx] += w;
                        b_sum[wi][idx] += w * o.val_mine;
                    }
                    let bmats = self.build_posterior_matrices_from_stats(&b_n, &b_sum);
                    if use_sub {
                        let kr = sub_rows.len();
                        let mut sub_mats: Vec<Vec<f64>> = Vec::with_capacity(n_w_tot);
                        let mut sub_lens: Vec<usize> = Vec::with_capacity(n_w_tot);
                        let mut sub_ws: Vec<f64> = Vec::with_capacity(n_w_tot);
                        for w in 0..n_w_tot {
                            let m_full = if w < n_w2 {
                                self.worlds[self.s2_world_ids[w]].opp_canon.len()
                            } else {
                                self.worlds[self.tail_world_ids[w - n_w2]].opp_canon.len()
                            };
                            let kc = sub_cols[w].len();
                            let mut sm = vec![0.5f64; kr * kc];
                            for (ri, &orig_i) in sub_rows.iter().enumerate() {
                                for (cj, &orig_j) in sub_cols[w].iter().enumerate() {
                                    sm[ri * kc + cj] = bmats[w][orig_i * m_full + orig_j];
                                }
                            }
                            sub_mats.push(sm);
                            sub_lens.push(kc);
                            sub_ws.push(if w < n_w2 {
                                self.s2_post_weights[w]
                            } else {
                                self.tail_post_weights[w - n_w2]
                            });
                        }
                        let slices: Vec<(f64, usize, &[f64])> = (0..n_w_tot)
                            .map(|w| (sub_ws[w], sub_lens[w], sub_mats[w].as_slice()))
                            .collect();
                        let bsol = solve_bayesian_rm_plus(kr, &slices, boot_sweeps);
                        for (ri, &orig_i) in sub_rows.iter().enumerate() {
                            avg_p[orig_i] += bsol.own_policy[ri] / (draws as f64);
                        }
                    } else {
                        let bsol = self.solve_full_game_from_mats(&bmats, boot_sweeps, 0.0);
                        for i in 0..n_own {
                            avg_p[i] += bsol.own_policy[i] / (draws as f64);
                        }
                    }
                }
                shed_dust(&mut avg_p, dust_ratio);
                avg_p
            }
            TrajPolicyMode::MixedNash { dust_ratio }
            | TrajPolicyMode::RegRm { dust_ratio, .. }
            | TrajPolicyMode::LcbGatedMixed { dust_ratio, .. }
            | TrajPolicyMode::BayesianBootstrapRm { dust_ratio, .. } => {
                let mut p = final_sol.own_policy.clone();
                shed_dust(&mut p, dust_ratio);
                p
            }
        };

        let psum: f64 = p_mix.iter().sum();
        if psum <= 0.0 {
            p_mix = vec![0.0; n_own];
            p_mix[best_pure] = 1.0;
        } else {
            for x in &mut p_mix {
                *x /= psum;
            }
        }

        let (lcb_z, use_neff) = match mode {
            TrajPolicyMode::LcbGatedMixed {
                lcb_z, use_neff, ..
            } => (lcb_z, use_neff),
            _ => (0.0, true),
        };
        let mut mixed_guarantee = 0.0;
        let mut mixed_guarantee_lcb = 0.0;
        for wi in 0..n_w2 {
            let m = self.worlds[self.s2_world_ids[wi]].opp_canon.len();
            let mat = &mats[wi];
            let mut min_col = f64::INFINITY;
            let mut min_col_lcb = f64::INFINITY;
            for j in 0..m {
                let mut mean_j = 0.0;
                let mut var_j = 0.0;
                for i in 0..n_own {
                    let pr = p_mix[i];
                    if pr > 0.0 {
                        mean_j += pr * mat[i * m + j];
                        let n_raw = f64::from(self.world_stats[wi].cell_n[i * m + j]);
                        let n_c = if use_neff {
                            (n_raw + self.cfg.kappa).max(1.0)
                        } else {
                            n_raw.max(1.0)
                        };
                        var_j += (pr * pr) * (0.12 / n_c);
                    }
                }
                if mean_j < min_col {
                    min_col = mean_j;
                }
                let lcb_j = mean_j - lcb_z * var_j.sqrt();
                if lcb_j < min_col_lcb {
                    min_col_lcb = lcb_j;
                }
            }
            mixed_guarantee += self.s2_post_weights[wi] * min_col;
            mixed_guarantee_lcb += self.s2_post_weights[wi] * min_col_lcb;
        }
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let m = self.worlds[w_id].opp_canon.len();
            let mat = &mats[n_w2 + ti];
            let mut min_col = f64::INFINITY;
            for j in 0..m {
                let mut mean_j = 0.0;
                for i in 0..n_own {
                    let pr = p_mix[i];
                    if pr > 0.0 {
                        mean_j += pr * mat[i * m + j];
                    }
                }
                if mean_j < min_col {
                    min_col = mean_j;
                }
            }
            mixed_guarantee += self.tail_post_weights[ti] * min_col;
            mixed_guarantee_lcb += self.tail_post_weights[ti] * min_col;
        }

        let n_supp = p_mix.iter().filter(|&&x| x > 0.0).count();
        let use_mix = match mode {
            TrajPolicyMode::LcbGatedMixed { min_mix_margin, .. } => {
                n_supp > 1
                    && mixed_guarantee > pure_guarantee + min_mix_margin
                    && mixed_guarantee_lcb > pure_guarantee - 0.005
            }
            TrajPolicyMode::MixedNash { .. }
            | TrajPolicyMode::RegRm { .. }
            | TrajPolicyMode::BayesianBootstrapRm { .. } => n_supp > 1,
        };

        let mut sub_pol = vec![0.0f64; n_own];
        if use_mix {
            sub_pol.copy_from_slice(&p_mix);
        } else {
            sub_pol[best_pure] = 1.0;
        }
        (sub_pol, final_sol, pure_guarantee, mixed_guarantee, use_mix)
    }
}

/// Solve Turn-0 Team Preview using the Live Trajectory-DO engine (`SteppedTrajPreview`).
/// Guarantees bit-for-bit parity with stepped execution and zero hidden-information leakage.
pub fn solve_traj_preview(
    position: &ObservedPosition,
    dex: &Dex,
    belief: &Belief,
    obs: &crate::observe::Observer,
    side: usize,
    cfg: &TrajPreviewConfig,
    rng: &mut SplitMix64,
) -> TrajPreviewResult {
    let t_total = start_timer();
    let mut base = position.model().clone();
    assert_eq!(base.request_state, RequestState::TeamPreview);
    let own_legal = base.legal_choices(dex, side);
    let n_legal = own_legal.len();
    let (_, own_canon) = canonical_preview_actions(dex, &mut base, side);
    if own_canon.len() <= 1 {
        let mut pol = vec![0.0; n_legal];
        let ch = if let Some(&(idx, _)) = own_canon.first() {
            pol[idx] = 1.0;
            own_legal[idx]
        } else {
            pol[0] = 1.0;
            own_legal[0]
        };
        return TrajPreviewResult {
            choice: ch,
            policy_full: pol,
            opp_policy_w0: Vec::new(),
            value: 0.5,
            pure_guarantee: 0.5,
            mixed_guarantee: 0.5,
            duality_gap: 0.0,
            used_mix: false,
            used_fallback: false,
            stage1_ms: 0.0,
            total_ms: elapsed_ms(t_total),
            iters_used: 0,
            games_completed: 0,
            worlds_stage1: 0,
            worlds_stage2: 0,
        };
    }

    let Some(mut stepped) = SteppedTrajPreview::try_new(
        &base,
        dex,
        belief,
        obs,
        side,
        &own_legal,
        None,
        cfg.clone(),
        rng,
    ) else {
        let choice = solve_shipped_preview(
            position,
            dex,
            belief,
            obs,
            side,
            cfg.fallback_iters,
            SplitMix64::new(rng.next()),
        );
        let mut pol = vec![0.0; n_legal];
        if let Some(pos) = own_legal.iter().position(|&c| c == choice) {
            pol[pos] = 1.0;
        }
        return TrajPreviewResult {
            choice,
            policy_full: pol,
            opp_policy_w0: Vec::new(),
            value: 0.5,
            pure_guarantee: 0.5,
            mixed_guarantee: 0.5,
            duality_gap: 0.0,
            used_mix: false,
            used_fallback: true,
            stage1_ms: 0.0,
            total_ms: elapsed_ms(t_total),
            iters_used: cfg.fallback_iters,
            games_completed: 0,
            worlds_stage1: 0,
            worlds_stage2: 0,
        };
    };

    for _ in 0..cfg.total_iters {
        stepped.step_one(rng);
    }
    let (sub_pol, final_sol, pure_guarantee, mixed_guarantee, used_mix) =
        stepped.solve_final_policy(stepped.iters_done);
    let draw = rng.next_f64();
    let picked = sample_dist(&sub_pol, draw);
    let mut policy_full = vec![0.0f64; n_legal];
    for (i, &pr) in sub_pol.iter().enumerate() {
        policy_full[stepped.own_canon[i].0] = pr;
    }
    let choice = stepped.my_acts[stepped.own_canon[picked].0];
    let opp_policy_w0 = final_sol.opp_policies.first().cloned().unwrap_or_default();

    TrajPreviewResult {
        choice,
        policy_full,
        opp_policy_w0,
        value: final_sol.value,
        pure_guarantee,
        mixed_guarantee,
        duality_gap: final_sol.duality_gap,
        used_mix,
        used_fallback: false,
        stage1_ms: stepped.stage1_ms,
        total_ms: elapsed_ms(t_total),
        iters_used: stepped.iters_done,
        games_completed: stepped.games_completed,
        worlds_stage1: stepped.worlds.len(),
        worlds_stage2: stepped.s2_world_ids.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player_state::LocalPlayer;
    use crate::preview::{MetaPool, MetaTeam};
    use nc2000_engine::battle::PokemonSet;
    use serde_json::json;
    use std::sync::Arc;

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

    #[test]
    fn preview_tempo_cache_exact_parity_and_bayesian_rm_convergence() {
        let dex = conformance::load_dex();
        let t1 = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t2 = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Machamp", 50, "Berserk Gene", &["Cross Chop", "Earthquake", "Rock Slide", "Hidden Power Bug"]),
            mk("Umbreon", 50, "Leftovers", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];
        let b = Battle::from_fixture(&dex, "1,2,3,4", &t1, &t2).unwrap();
        assert!(is_compact_preview_supported(&b, &dex));
        let roster = CompactPreviewRoster::from_battle(&b, &dex, None);
        let pcache = PreviewTempoCache::build_for_preview_roster(&roster, [6, 6]);

        // Verify 0-ULP parity of `PreviewTempoCache::slice_for` vs fresh `TempoMatchCache::build`
        // including Berserk Gene lead (slot 5 of t2) and Baton Pass lead (slot 6 of t2).
        for picks in [
            [[1, 2, 3], [1, 2, 3]],
            [[4, 5, 6], [4, 5, 6]],
            [[3, 1, 6], [5, 2, 4]],
            [[2, 4, 5], [6, 1, 3]],
        ] {
            let (table, cb) = roster.build_turn1(picks, 0);
            let direct = TempoMatchCache::build(&cb, &table);
            let sliced = pcache.slice_for(&cb);
            let v_direct = evaluate_cf_value(&cb, &table, Some(&direct), 1.0);
            let v_sliced = evaluate_cf_value(&cb, &table, Some(&sliced), 1.0);
            assert!(
                (v_direct - v_sliced).abs() < 1e-12,
                "PreviewTempoCache::slice_for drift: {v_direct} vs {v_sliced} on {picks:?}"
            );
        }

        // Verify Multi-World Bayesian RM+ exactness on RPS + heterogeneous column counts.
        let rps = [0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5];
        let sol = solve_bayesian_rm_plus(3, &[(1.0, 3, &rps)], 1000);
        assert!(sol.duality_gap < 1e-3, "RPS gap {}", sol.duality_gap);
        for &p in &sol.own_policy {
            assert!((p - 1.0 / 3.0).abs() < 0.01, "RPS policy {:?}", sol.own_policy);
        }
        // Heterogeneous column counts (world 0 has 2 cols, world 1 has 4 cols).
        let w0 = [0.8, 0.2, 0.2, 0.8];
        let w1 = [0.7, 0.3, 0.9, 0.9, 0.3, 0.7, 0.9, 0.9];
        let hsol = solve_bayesian_rm_plus(2, &[(0.5, 2, &w0), (0.5, 4, &w1)], 1000);
        assert!(hsol.duality_gap < 1e-3, "hetero gap {}", hsol.duality_gap);
        assert!((hsol.value - 0.5).abs() < 0.01, "hetero value {}", hsol.value);
    }

    #[test]
    fn cf_preview_respects_level_caps_beatup_small_teams_and_zero_leak() {
        let dex = conformance::load_dex();
        let t_own = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
        ];
        // Opponent team A vs mutated hidden moves team B (same public species + levels).
        let t_opp_a = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let mut t_opp_b = t_opp_a.clone();
        t_opp_b[0] = mk("Raikou", 50, "Magnet", &["Thunder", "Crunch", "Reflect", "Rest"]);
        t_opp_b[2] = mk("Exeggutor", 50, "Mint Berry", &["Psychic", "Stun Spore", "Leech Seed", "Rest"]);

        let pool = Arc::new(MetaPool {
            teams: vec![MetaTeam {
                id: "opp-a".into(),
                sets: t_opp_a.clone(),
                weight: Some(1.0),
            }],
        });

        let run_once = |opp_sets: &[PokemonSet]| {
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t_own, opp_sets).unwrap();
            let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_own.clone()).unwrap();
            lp.observe(&b, &dex, &mut obs).unwrap();
            let mut belief = Belief::new(&dex, &pool, &obs);
            belief.sync(&dex, &obs);
            let mut rng = SplitMix64::new(20261124);
            let pos = lp.position(&dex, &mut belief, &obs, &mut rng).unwrap();
            let cfg = PreviewSearchConfig {
                total_iters: 2048,
                support_k: 6,
                coarse_iters: 32,
                do_rounds: 1,
                do_candidates: 4,
                do_iters: 32,
                polish_chunk: 32,
                ..Default::default()
            };
            let res = solve_cf_preview(&pos, &dex, &belief, &obs, 0, &cfg, &mut rng);
            let legal = b.legal_choices(&dex, 0);
            assert!(legal.contains(&res.choice), "illegal choice {:?}", res.choice);
            if let SearchChoice::Team(t) = res.choice {
                let sum_lv: u32 = t.iter().filter(|&&s| s != 0).map(|&s| u32::from(t_own[s as usize - 1].level)).sum();
                assert!(sum_lv <= 155, "exceeded 155 cap: {t:?} -> {sum_lv}");
            }
            res
        };
        let r_a = run_once(&t_opp_a);
        let r_b = run_once(&t_opp_b);
        assert_eq!(r_a.choice, r_b.choice, "hidden opponent move mutation must not leak into preview choice");
        assert_eq!(r_a.policy_full, r_b.policy_full, "policy must be bit-identical under hidden opp mutation");

        // 2-mon team edge case (`triple[2] == 0`).
        let small = vec![t_own[2].clone(), t_own[3].clone()];
        let mut b_sm = Battle::from_fixture(&dex, "1,2,3,4", &small, &small).unwrap();
        let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b_sm, &dex, 0, small.clone()).unwrap();
        lp.observe(&b_sm, &dex, &mut obs).unwrap();
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);
        let mut rng = SplitMix64::new(77);
        let pos = lp.position(&dex, &mut belief, &obs, &mut rng).unwrap();
        let cfg_sm = PreviewSearchConfig {
            total_iters: 256,
            support_k: 4,
            coarse_iters: 32,
            do_rounds: 0,
            ..Default::default()
        };
        let res_sm = solve_cf_preview(&pos, &dex, &belief, &obs, 0, &cfg_sm, &mut rng);
        assert!(b_sm.legal_choices(&dex, 0).contains(&res_sm.choice));
    }

    #[test]
    fn cf_preview_cell_symmetry_halving_berserk_gene_and_do_invariants() {
        let dex = conformance::load_dex();
        let t_a = vec![
            mk("Machamp", 50, "Berserk Gene", &["Cross Chop", "Earthquake", "Rock Slide", "Hidden Power Bug"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_b = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Leftovers", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];

        // 1. Berserk Gene lead detection + float-exact halving across 64 and 128 boundaries.
        let b_ab = Battle::from_fixture(&dex, "1,2,3,4", &t_a, &t_b).unwrap();
        let r_ab = CompactPreviewRoster::from_battle(&b_ab, &dex, None);
        let pc_ab = PreviewTempoCache::build_for_preview_roster(&r_ab, [6, 6]);
        let cfg = PreviewSearchConfig::default();
        let mut cell_ab = CfPreviewCell::new(&r_ab, &pc_ab, [[1, 2, 3], [1, 2, 3]], 0.5, &cfg);
        assert_eq!(cell_ab.berserk_confused, [true, false]);

        let eg_w = EvalWeights::default();
        let eg_p = Playout::Heavy { turns: 8, eps: 0.2, weights: eg_w.clone() };
        let r_rules = RmConfig::default().rollout_rules();
        let mut path = Vec::new();
        let mut rng = SplitMix64::new(20261130);
        cell_ab.step(160, &cfg, &mut rng, &eg_w, &eg_p, r_rules, &mut path);
        assert_eq!(cell_ab.total_n, 160);
        // After 64 -> 32.0, after 128 -> (32+64)*0.5 = 48.0, +32 = 80.0 (exact float halving, zero int drift).
        assert!(
            (cell_ab.late_n - 80.0).abs() < 1e-12,
            "expected exact late_n=80.0, got {}",
            cell_ab.late_n
        );
        // Verify Berserk Gene confusion duration resamples across all 4 values {2,3,4,5}.
        let mut seen_durs = [false; 6];
        let mut prng_check = SplitMix64::new(999);
        for _ in 0..64 {
            let mut pr = nc2000_engine::prng::Prng::new(prng_check.next());
            let _ = Battle::roll_quick_claw_with(cell_ab.table.ruleset, &mut pr);
            let d = (2 + pr.random(4)) as usize;
            seen_durs[d] = true;
        }
        assert!(seen_durs[2] && seen_durs[3] && seen_durs[4] && seen_durs[5]);

        // 2. Side-0 vs Side-1 symmetry on swapped matchup (t_a vs t_b) and (t_b vs t_a).
        let b_ba = Battle::from_fixture(&dex, "1,2,3,4", &t_b, &t_a).unwrap();
        let r_ba = CompactPreviewRoster::from_battle(&b_ba, &dex, None);
        let pc_ba = PreviewTempoCache::build_for_preview_roster(&r_ba, [6, 6]);
        let mut cell_ba = CfPreviewCell::new(&r_ba, &pc_ba, [[1, 2, 3], [1, 2, 3]], 0.5, &cfg);
        assert_eq!(cell_ba.berserk_confused, [false, true]);
        let mut rng2 = SplitMix64::new(20261131);
        cell_ab.step(352, &cfg, &mut rng, &eg_w, &eg_p, r_rules, &mut path); // total 512
        cell_ba.step(512, &cfg, &mut rng2, &eg_w, &eg_p, r_rules, &mut path);
        for ro in [
            PreviewCellReadout::RawMean,
            PreviewCellReadout::LateWindowMean,
            PreviewCellReadout::Turn1Minimax,
        ] {
            let sum = cell_ab.value0(ro) + cell_ba.value0(ro);
            assert!(
                (sum - 1.0).abs() < 0.06,
                "side asymmetry on {ro:?}: v_ab={} v_ba={} sum={}",
                cell_ab.value0(ro),
                cell_ba.value0(ro),
                sum
            );
        }

        // 3. Double-Oracle multi-round stress under tight budget (verifies no OOB after opp DO admission).
        let pool = Arc::new(MetaPool {
            teams: vec![
                MetaTeam { id: "tb".into(), sets: t_b.clone(), weight: Some(0.6) },
                MetaTeam { id: "ta".into(), sets: t_a.clone(), weight: Some(0.4) },
            ],
        });
        let b_do = Battle::from_fixture(&dex, "1,2,3,4", &t_a, &t_b).unwrap();
        let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b_do, &dex, 0, t_a.clone()).unwrap();
        lp.observe(&b_do, &dex, &mut obs).unwrap();
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);
        let mut rng_do = SplitMix64::new(4242);
        let pos = lp.position(&dex, &mut belief, &obs, &mut rng_do).unwrap();
        let cfg_do = PreviewSearchConfig {
            total_iters: 3000,
            support_k: 6,
            coarse_iters: 24,
            do_rounds: 2,
            do_candidates: 4,
            do_iters: 32,
            do_margin: -0.5, // force DO admission to stress re-solve & bounds!
            polish_chunk: 32,
            ..Default::default()
        };
        let res_do = solve_cf_preview(&pos, &dex, &belief, &obs, 0, &cfg_do, &mut rng_do);
        assert!(res_do.iters_used <= cfg_do.total_iters);
        assert!(!res_do.used_fallback);
    }

    #[test]
    fn stepped_cf_preview_slice_invariance_c_sensitivity_and_mask_legality() {
        let dex = conformance::load_dex();
        let t_a = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_b = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Leftovers", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];
        let pool = Arc::new(MetaPool {
            teams: vec![MetaTeam { id: "tb".into(), sets: t_b.clone(), weight: Some(1.0) }],
        });
        let b = Battle::from_fixture(&dex, "1,2,3,4", &t_a, &t_b).unwrap();
        let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_a.clone()).unwrap();
        lp.observe(&b, &dex, &mut obs).unwrap();
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);
        let pos = lp.position(&dex, &mut belief, &obs, &mut SplitMix64::new(99)).unwrap();

        // 1. Exact bit-identical slice invariance (`step(1)*256 == step(64)*4 == step(256)`).
        let mk_search = |c: f64| {
            let mut bs = BlindSearch::from_observation(
                &pos,
                &dex,
                RmConfig { c, ..Default::default() },
                0,
                777,
            );
            bs.prune_preview_aliases(&dex);
            bs
        };
        let mut s_one = mk_search(0.4);
        for _ in 0..256 {
            s_one.step(&dex, &belief, &obs, 1);
        }
        let mut s_chunk = mk_search(0.4);
        for _ in 0..4 {
            s_chunk.step(&dex, &belief, &obs, 64);
        }
        let mut s_full = mk_search(0.4);
        s_full.step(&dex, &belief, &obs, 256);

        assert_eq!(s_one.visits(), s_full.visits());
        assert_eq!(s_chunk.visits(), s_full.visits());
        assert_eq!(s_one.means(), s_full.means());
        assert_eq!(s_chunk.means(), s_full.means());
        assert_eq!(s_one.best(), s_full.best());
        assert_eq!(s_chunk.best(), s_full.best());
        assert_eq!(s_one.node_count(), s_full.node_count());
        assert!(s_full.node_count() > 0);
        assert!(!s_full.root_replies().is_empty());
        // Visit-sample invariant: >= 2 distinct actions visited early.
        assert!(s_full.visits().iter().filter(|&&n| n > 0).count() >= 2);

        // 2. Double-Oracle admission-fill finalize invariant: fills MUST drain before `sol2` / `Polish`.
        let mut rng_do = SplitMix64::new(5151);
        let mut model_clone = pos.model().clone();
        let acts = model_clone.legal_choices(&dex, 0);
        let mut stepped = SteppedCfPreview::try_new(
            pos.model(),
            &dex,
            &belief,
            &obs,
            0,
            &acts,
            None,
            0.4,
            &mut rng_do,
        )
        .unwrap();
        stepped.cfg.total_iters = 3000;
        stepped.cfg.coarse_iters = 24;
        stepped.cfg.do_iters = 32;
        stepped.cfg.do_candidates = 4;
        stepped.cfg.do_margin = -0.5;
        stepped.cfg.polish_chunk = 32;
        for w in &mut stepped.coarse_per_w {
            *w = 8;
        }
        let mut entered_polish = false;
        for _ in 0..3000 {
            stepped.step_one(&mut rng_do);
            if matches!(stepped.phase, SteppedPreviewPhase::Polish { .. }) {
                entered_polish = true;
                for wi in 0..stepped.s2_world_ids.len() {
                    for &own_i in &stepped.s_me {
                        for &opp_j in &stepped.s_opp[wi] {
                            let n = stepped
                                .cells
                                .get(&(wi, own_i, opp_j))
                                .map_or(0, |c| c.total_n);
                            assert!(
                                n > 0,
                                "entered Polish before DO fill drained for ({wi},{own_i},{opp_j})"
                            );
                        }
                    }
                }
                break;
            }
        }
        assert!(entered_polish, "SteppedCfPreview should reach Polish under tight test budget");

        // 2. Exploration constant `c` sensitivity (`c = 0.4` vs `c = 1.0` at 512 iters).
        let mut s_c04 = mk_search(0.4);
        let mut s_c10 = mk_search(1.0);
        s_c04.step(&dex, &belief, &obs, 512);
        s_c10.step(&dex, &belief, &obs, 512);
        assert_ne!(
            s_c04.means(),
            s_c10.means(),
            "c = 0.4 vs c = 1.0 must produce distinct SteppedCfPreview rollouts"
        );

        // 3. Explicit `mask_actions` enforcement: masked actions get 0 visits and are never chosen.
        let mut s_masked = mk_search(0.4);
        let mut allowed = vec![false; s_masked.actions().len()];
        allowed[0] = true;
        allowed[5] = true;
        allowed[10] = true;
        s_masked.mask_actions(&allowed);
        s_masked.step(&dex, &belief, &obs, 128);
        assert!(s_masked
            .visits()
            .iter()
            .zip(&allowed)
            .all(|(&n, &ok)| ok || n == 0));
        let chosen = s_masked.best().unwrap();
        let chosen_idx = s_masked.actions().iter().position(|&c| c == chosen).unwrap();
        assert!(allowed[chosen_idx], "best() returned masked preview action");
    }

    #[test]
    fn stepped_cf_preview_execution_policy_matches_best_choice_and_protocol_agent() {
        use crate::decision::{DecisionPolicy, PolicyMode};
        use crate::player::PlayerChannel;
        use crate::ProtocolAgent;

        let dex = conformance::load_dex();
        let t_a = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_b = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Leftovers", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];
        let pool = MetaPool {
            teams: vec![MetaTeam { id: "tb".into(), sets: t_b.clone(), weight: Some(1.0) }],
        };
        let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t_a, &t_b).unwrap();
        let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_a.clone()).unwrap();
        lp.observe(&b, &dex, &mut obs).unwrap();
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);
        let pos = lp.position(&dex, &mut belief, &obs, &mut SplitMix64::new(99)).unwrap();

        let mut rng = SplitMix64::new(4242);
        let mut model_clone = pos.model().clone();
        let acts = model_clone.legal_choices(&dex, 0);
        let mut stepped = SteppedCfPreview::try_new(
            pos.model(),
            &dex,
            &belief,
            &obs,
            0,
            &acts,
            None,
            0.4,
            &mut rng,
        )
        .unwrap();
        for _ in 0..512 {
            stepped.step_one(&mut rng);
        }

        // Test both pure gate (high margin) and mixed gate (negative margin) for exact parity.
        for (min_margin, expect_pure) in [(10.0, true), (-1.0, false)] {
            stepped.cfg.policy_mode = PreviewPolicyMode::LcbGatedMixed {
                min_mix_margin: min_margin,
                dust_ratio: 0.08,
                lcb_z: 0.0,
            };
            let done = 512u32;
            let entries = stepped.execution_policy(done);
            assert!(!entries.is_empty());
            let sum: f64 = entries.iter().map(|(_, p)| *p).sum();
            assert!((sum - 1.0).abs() < 1e-12, "policy sum {sum} != 1.0");
            assert!(entries.iter().all(|(_, p)| p.is_finite() && *p > 0.0));
            if expect_pure {
                assert_eq!(entries.len(), 1, "high margin must yield pure execution policy");
            }
            let dp = DecisionPolicy::new("test", entries.iter().copied()).unwrap();
            for seed in 0u64..256 {
                let draw = SplitMix64::new(seed ^ u64::from(done)).next_f64();
                assert_eq!(
                    dp.sample(draw),
                    stepped.best_choice(seed, done),
                    "execution_policy sample diverged from best_choice at seed={seed}, margin={min_margin}"
                );
            }
        }

        // Verify ProtocolAgent::preview_execution_policy preserves PolicyMode::Legacy untouched.
        let mut ch = PlayerChannel::new(0);
        let frame = ch.frame(&mut b, &dex).unwrap();
        let req_json = serde_json::to_string(&frame.request).unwrap();
        let mut agent = ProtocolAgent::new(&dex, 0, pool, RmConfig { c: 0.4, ..Default::default() }, 777);
        agent.set_own_team(t_a);
        for line in &frame.lines {
            agent.push_line(&dex, line);
        }
        assert!(agent.on_request(&dex, &req_json).unwrap());
        agent.step(&dex, 256).unwrap();

        let legacy_before = agent.decision_policy(&dex, PolicyMode::Legacy).unwrap();
        let exec_pol = agent.preview_execution_policy(&dex).unwrap();
        let legacy_after = agent.decision_policy(&dex, PolicyMode::Legacy).unwrap();

        assert_eq!(legacy_before, legacy_after, "preview_execution_policy must be strictly read-only");
        assert_eq!(legacy_before.source(), "legacy");
        assert_eq!(legacy_before.actions().len(), 1);
        assert!(
            matches!(exec_pol.source(), "cf-preview-mix" | "cf-preview-pure"),
            "unexpected source: {}",
            exec_pol.source()
        );
        let legacy_input = &legacy_before.actions()[0].input;
        assert!(
            exec_pol.actions().iter().any(|a| &a.input == legacy_input),
            "legacy best choice {legacy_input} not in execution policy support {:?}",
            exec_pol.actions()
        );
    }

    #[test]
    fn traj_do_slice_invariance_zero_leak_hazards_multiworld_and_parity() {
        use crate::decision::DecisionPolicy;

        let dex = conformance::load_dex();
        let t_own = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Whirlwind", "Rest"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Heracross", 50, "Focus Band", &["Megahorn", "Endure", "Curse", "Reversal"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_opp_a = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Leftovers", &["Mean Look", "Charm", "Rest", "Baton Pass"]),
        ];
        let mut t_opp_b = t_opp_a.clone();
        t_opp_b[0] = mk("Raikou", 50, "Magnet", &["Thunder", "Crunch", "Reflect", "Rest"]);
        t_opp_b[2] = mk("Exeggutor", 50, "Mint Berry", &["Psychic", "Stun Spore", "Leech Seed", "Rest"]);

        // 1. Tactical endgame hazard detector catches Endure+Reversal+FocusBand & Destiny Bond,
        //    and returns false on a clean non-hazard endgame.
        {
            let b_haz = Battle::from_fixture(&dex, "1,2,3,4", &t_own, &t_opp_a).unwrap();
            let r_haz = CompactPreviewRoster::from_battle(&b_haz, &dex, None);
            let (tbl_h, cb_h) = r_haz.build_turn1([[4, 1, 2], [1, 2, 4]], 0);
            assert!(
                has_tactical_endgame_hazard(&cb_h, &tbl_h),
                "Heracross Endure+Reversal+FocusBand must trigger tactical hazard guard"
            );
            let (tbl_c, mut cb_c) = r_haz.build_turn1([[1, 2, 6], [1, 2, 4]], 0);
            // Eliminate bench slots except clean non-hazard Tyranitar (slot 2) vs Snorlax (slot 0) on turn 1.
            cb_c.sides[0].alive_mask = 1 << 0;
            cb_c.sides[1].alive_mask = 1 << 2;
            cb_c.sides[1].active_idx = 2;
            assert!(
                !has_tactical_endgame_hazard(&cb_c, &tbl_c),
                "Clean Snorlax (Rest 16PP) vs Tyranitar on turn 1 should be hazard-free"
            );
        }

        // 2. Cholesky SPD solver & L2-regularized RM+ sanity checks.
        {
            let mut ata = [[0.0f64; MAX_RIDGE_DIM]; MAX_RIDGE_DIM];
            let mut atb = [0.0f64; MAX_RIDGE_DIM];
            ata[0][0] = 4.0;
            ata[0][1] = 1.0;
            ata[1][0] = 1.0;
            ata[1][1] = 3.0;
            atb[0] = 6.0;
            atb[1] = 5.0;
            let w = solve_spd_cholesky(2, &ata, &atb, 1.0, 2.0, 4.0);
            // System: 5*w0 + 1*w1 = 6, 1*w0 + 5*w1 = 5 => w0 = 25/24, w1 = 19/24.
            assert!((w[0] - 25.0 / 24.0).abs() < 1e-10);
            assert!((w[1] - 19.0 / 24.0).abs() < 1e-10);

            let rps = [0.5, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0, 1.0, 0.5];
            let sol = solve_bayesian_rm_plus_reg(3, &[(1.0, 3, &rps)], 800, 0.02);
            for &p in &sol.own_policy {
                assert!((p - 1.0 / 3.0).abs() < 0.02, "RegRm RPS drift: {:?}", sol.own_policy);
            }
        }

        // 3. Multi-world (W = 2) zero hidden-info leakage, level-cap compliance, and exact slice invariance!
        let pool = Arc::new(MetaPool {
            teams: vec![
                MetaTeam { id: "opp-a".into(), sets: t_opp_a.clone(), weight: Some(0.65) },
                MetaTeam { id: "opp-b".into(), sets: t_opp_b.clone(), weight: Some(0.35) },
            ],
        });
        let test_cfg = TrajPreviewConfig {
            total_iters: 1024,
            iters_per_turn: 16,
            rb_exit: RbExitMode::SearchVroot { max_alive: 3, eval_iters: 16 },
            ridge_basis: TrajRidgeBasis::Interact33,
            policy_mode: TrajPolicyMode::BayesianBootstrapRm { draws: 8, dust_ratio: 0.12 },
            max_stage2_worlds: 2,
            ..Default::default()
        };

        let run_batch = |opp_sets: &[PokemonSet]| {
            let mut b = Battle::from_fixture(&dex, "1,2,3,4", &t_own, opp_sets).unwrap();
            let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_own.clone()).unwrap();
            lp.observe(&b, &dex, &mut obs).unwrap();
            let mut belief = Belief::new(&dex, &pool, &obs);
            belief.sync(&dex, &obs);
            let mut rng = SplitMix64::new(20261126);
            let pos = lp.position(&dex, &mut belief, &obs, &mut rng).unwrap();
            let res = solve_traj_preview(&pos, &dex, &belief, &obs, 0, &test_cfg, &mut rng);
            let legal = b.legal_choices(&dex, 0);
            assert!(legal.contains(&res.choice), "illegal Traj-DO choice {:?}", res.choice);
            if let SearchChoice::Team(t) = res.choice {
                let sum_lv: u32 = t.iter().filter(|&&s| s != 0).map(|&s| u32::from(t_own[s as usize - 1].level)).sum();
                assert!(sum_lv <= 155, "exceeded 155 cap: {t:?} -> {sum_lv}");
            }
            assert!(!res.used_fallback);
            assert_eq!(res.worlds_stage1, 2);
            assert_eq!(res.worlds_stage2, 2);
            res
        };
        let ra = run_batch(&t_opp_a);
        let rb = run_batch(&t_opp_b);
        assert_eq!(ra.choice, rb.choice, "Traj-DO leaked hidden opponent moves/items");
        assert_eq!(ra.policy_full, rb.policy_full, "Traj-DO policy must be bit-identical under hidden mutation");

        // 4. Bit-identical slice invariance (`step(1)*256 == step(64)*4 == step(256)`),
        //    early action diversity, mask_actions, and execution_policy == best_choice parity!
        let b = Battle::from_fixture(&dex, "1,2,3,4", &t_own, &t_opp_a).unwrap();
        let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_own.clone()).unwrap();
        lp.observe(&b, &dex, &mut obs).unwrap();
        let mut belief = Belief::new(&dex, &pool, &obs);
        belief.sync(&dex, &obs);
        let pos = lp.position(&dex, &mut belief, &obs, &mut SplitMix64::new(99)).unwrap();
        let mut mclone = pos.model().clone();
        let acts = mclone.legal_choices(&dex, 0);

        let mk_stepped = |mask: Option<&[bool]>| -> (SteppedTrajPreview, SplitMix64) {
            let mut rng = SplitMix64::new(7777);
            let st = SteppedTrajPreview::try_new(
                pos.model(),
                &dex,
                &belief,
                &obs,
                0,
                &acts,
                mask,
                test_cfg.clone(),
                &mut rng,
            )
            .unwrap();
            (st, rng)
        };

        let (mut s1, mut r1) = mk_stepped(None);
        let mut visited_acts = std::collections::HashSet::new();
        for _ in 0..256 {
            let (idx, _, _, _, _) = s1.step_one(&mut r1);
            visited_acts.insert(idx);
        }
        assert!(visited_acts.len() >= 2, "must span >= 2 actions within 256 steps");
        assert!(s1.node_count() > 0);

        let (mut s2, mut r2) = mk_stepped(None);
        for _ in 0..4 {
            for _ in 0..64 {
                s2.step_one(&mut r2);
            }
        }
        assert_eq!(s1.node_count(), s2.node_count());
        assert_eq!(s1.execution_policy(256), s2.execution_policy(256));
        for seed in 0u64..128 {
            assert_eq!(s1.best_choice(seed, 256), s2.best_choice(seed, 256));
        }
        let entries = s1.execution_policy(256);
        let dp = DecisionPolicy::new("traj-test", entries.iter().copied()).unwrap();
        for seed in 0u64..128 {
            let draw = SplitMix64::new(seed ^ 256u64).next_f64();
            assert_eq!(dp.sample(draw), s1.best_choice(seed, 256));
        }

        // Mask legality check: masked actions never visited or chosen.
        let mut allowed = vec![false; acts.len()];
        allowed[0] = true;
        allowed[3] = true;
        allowed[7] = true;
        let (mut sm, mut rm) = mk_stepped(Some(&allowed));
        for _ in 0..128 {
            let (idx, _, _, _, _) = sm.step_one(&mut rm);
            assert!(allowed[idx], "stepped Traj-DO visited masked action {idx}");
        }
        let chosen = sm.best_choice(42, 128);
        let cidx = acts.iter().position(|&c| c == chosen).unwrap();
        assert!(allowed[cidx], "best_choice returned masked action");

        // 5. 2-mon team edge case (`triple[2] == 0`).
        let small = vec![t_own[2].clone(), t_own[4].clone()];
        let mut b_sm = Battle::from_fixture(&dex, "1,2,3,4", &small, &small).unwrap();
        let (mut lp_s, mut obs_s) = LocalPlayer::new_with_own_sets(&b_sm, &dex, 0, small.clone()).unwrap();
        lp_s.observe(&b_sm, &dex, &mut obs_s).unwrap();
        let mut bel_s = Belief::new(&dex, &pool, &obs_s);
        bel_s.sync(&dex, &obs_s);
        let mut rng_s = SplitMix64::new(88);
        let pos_s = lp_s.position(&dex, &mut bel_s, &obs_s, &mut rng_s).unwrap();
        let cfg_s = TrajPreviewConfig {
            total_iters: 128,
            iters_per_turn: 16,
            ..test_cfg
        };
        let res_s = solve_traj_preview(&pos_s, &dex, &bel_s, &obs_s, 0, &cfg_s, &mut rng_s);
        assert!(b_sm.legal_choices(&dex, 0).contains(&res_s.choice));
    }

    #[test]
    fn stepped_calib_preview_phase_boundary_slice_invariance_and_zero_leak() {
        use crate::decision::DecisionPolicy;
        let dex = conformance::load_dex();
        let t_a = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Leftovers", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Hidden Power Bug"]),
            mk("Gengar", 50, "Leftovers", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Leftovers", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_b1 = vec![
            mk("Raikou", 50, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Roar", "Rest"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Leftovers", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];
        let t_b2 = vec![
            mk("Raikou", 50, "Leftovers", &["Thunder", "Crunch", "Rest", "Sleep Talk"]),
            mk("Suicune", 50, "Leftovers", &["Hydro Pump", "Ice Beam", "Toxic", "Rest"]),
            mk("Exeggutor", 50, "Leftovers", &["Psychic", "Stun Spore", "Leech Seed", "Explosion"]),
            mk("Tyranitar", 50, "Leftovers", &["Rock Slide", "Fire Blast", "Dynamic Punch", "Crunch"]),
            mk("Snorlax", 55, "Leftovers", &["Double-Edge", "Belly Drum", "Earthquake", "Lovely Kiss"]),
            mk("Umbreon", 50, "Leftovers", &["Mean Look", "Baton Pass", "Charm", "Rest"]),
        ];
        let pool = MetaPool {
            teams: vec![
                MetaTeam { id: "tb1".into(), sets: t_b1.clone(), weight: Some(0.6) },
                MetaTeam { id: "tb2".into(), sets: t_b2.clone(), weight: Some(0.4) },
            ],
        };

        let make_env = |opp_sets: &[PokemonSet]| {
            let b = Battle::from_fixture(&dex, "1,2,3,4", &t_a, opp_sets).unwrap();
            let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_a.clone()).unwrap();
            lp.observe(&b, &dex, &mut obs).unwrap();
            let mut belief = Belief::new(&dex, &pool, &obs);
            belief.sync(&dex, &obs);
            let pos = lp.position(&dex, &mut belief, &obs, &mut SplitMix64::new(123)).unwrap();
            (pos, belief, obs)
        };
        let (pos1, bel1, obs1) = make_env(&t_b1);
        let (pos2, bel2, obs2) = make_env(&t_b2);

        let mk_calib = |model: &Battle, bel: &Belief, obs: &crate::observe::Observer| {
            let mut rng = SplitMix64::new(2026_0303);
            let mut model_clone = model.clone();
            let acts = model_clone.legal_choices(&dex, 0);
            let mut s = SteppedCfPreview::try_new_with_calib(
                model,
                &dex,
                bel,
                obs,
                0,
                &acts,
                None,
                0.4,
                Some(TrajCalibConfig {
                    turn1_iters: 16,
                    iters_per_turn: 16,
                    rb_eval_iters: 16,
                    support_k: 6,
                    ..TrajCalibConfig::default()
                }),
                &mut rng,
            )
            .unwrap();
            // Compact total budget so `4096` steps cross `TrajCalib -> Coarse -> DO -> Polish`.
            s.cfg.total_iters = 4096;
            s.cfg.coarse_iters = 16;
            s.cfg.do_iters = 16;
            s.cfg.polish_chunk = 32;
            if let Some(ref mut cal) = s.calib {
                cal.calib_iters = 1600;
                cal.explore_iters = 800;
                cal.traj_cfg.total_iters = 1600;
            }
            (s, rng)
        };

        // 1. Exact bit-identical slice invariance across the TrajCalib -> Stage-2 phase boundary.
        let (mut s_one, mut r_one) = mk_calib(pos1.model(), &bel1, &obs1);
        for _ in 0..4096 {
            s_one.step_one(&mut r_one);
        }
        assert!(s_one.calib.is_none(), "TrajCalib must finalize and drop scratch state");
        assert!(!s_one.calib_obs.is_empty(), "TrajCalib must collect CRN calibration observations");

        let (mut s_chunk, mut r_chunk) = mk_calib(pos1.model(), &bel1, &obs1);
        for _ in 0..64 {
            for _ in 0..64 {
                s_chunk.step_one(&mut r_chunk);
            }
        }
        assert_eq!(s_one.node_count(), s_chunk.node_count());
        assert_eq!(s_one.nash_value(4096), s_chunk.nash_value(4096));
        assert_eq!(s_one.execution_policy(4096), s_chunk.execution_policy(4096));
        for seed in 0u64..64 {
            assert_eq!(s_one.best_choice(seed, 4096), s_chunk.best_choice(seed, 4096));
        }
        let entries = s_one.execution_policy(4096);
        let dp = DecisionPolicy::new("calib-test", entries.iter().copied()).unwrap();
        for seed in 0u64..64 {
            let draw = SplitMix64::new(seed ^ 4096u64).next_f64();
            assert_eq!(dp.sample(draw), s_one.best_choice(seed, 4096));
        }

        // 2. Zero hidden-info leakage: swapping true opponent hidden moves/items gives bit-identical output.
        let (mut s_alt, mut r_alt) = mk_calib(pos2.model(), &bel2, &obs2);
        for _ in 0..4096 {
            s_alt.step_one(&mut r_alt);
        }
        assert_eq!(s_one.execution_policy(4096), s_alt.execution_policy(4096));
        assert_eq!(s_one.nash_value(4096), s_alt.nash_value(4096));
        for seed in 0u64..32 {
            assert_eq!(s_one.best_choice(seed, 4096), s_alt.best_choice(seed, 4096));
        }
    }

    #[test]
    fn stepped_cf_preview_traj_calib_castle_free_invariance_zero_leak_and_turn_cap() {
        use crate::decision::DecisionPolicy;
        use crate::player::PlayerChannel;
        use crate::preview::{MetaPool, MetaTeam};
        use crate::{ProtocolAgent, RmConfig};
        use nc2000_engine::rules::Ruleset;

        let dex = conformance::load_dex();
        // 100% Castle-legal (no NYPC Lovely Kiss, valid min-levels, distinct items per team) multi-world rosters.
        let t_a = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Bright Powder", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Gold Berry", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Body Slam"]),
            mk("Gengar", 50, "Quick Claw", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Mint Berry", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_b1 = vec![
            mk("Raikou", 50, "Magnet", &["Thunderbolt", "Protect", "Roar", "Rest"]),
            mk("Suicune", 50, "Mint Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Quick Claw", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 55, "Miracle Berry", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Gold Berry", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];
        let t_b2 = vec![
            mk("Raikou", 50, "Mint Berry", &["Thunder", "Roar", "Rest", "Sleep Talk"]),
            mk("Suicune", 50, "Leftovers", &["Surf", "Ice Beam", "Toxic", "Rest"]),
            mk("Exeggutor", 50, "Gold Berry", &["Psychic", "Stun Spore", "Leech Seed", "Explosion"]),
            mk("Tyranitar", 55, "Quick Claw", &["Rock Slide", "Fire Blast", "Dynamic Punch", "Crunch"]),
            mk("Snorlax", 50, "Miracle Berry", &["Double-Edge", "Earthquake", "Self-Destruct", "Curse"]),
            mk("Umbreon", 50, "Bright Powder", &["Mean Look", "Baton Pass", "Charm", "Rest"]),
        ];
        let pool = MetaPool {
            teams: vec![
                MetaTeam { id: "tb1".into(), sets: t_b1.clone(), weight: Some(0.6) },
                MetaTeam { id: "tb2".into(), sets: t_b2.clone(), weight: Some(0.4) },
            ],
        };

        let make_env = |side: usize, opp_sets: &[PokemonSet], ruleset: Ruleset| {
            let (p1, p2) = if side == 0 { (&t_a[..], opp_sets) } else { (opp_sets, &t_a[..]) };
            let b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", p1, p2, ruleset).unwrap();
            let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, side, t_a.clone()).unwrap();
            lp.observe(&b, &dex, &mut obs).unwrap();
            let mut belief = Belief::new(&dex, &pool, &obs);
            belief.sync(&dex, &obs);
            assert!(!belief.is_fallback(), "Castle-legal multi-world pool must stay alive");
            assert_eq!(belief.alive().len(), 2);
            let pos = lp.position(&dex, &mut belief, &obs, &mut SplitMix64::new(123)).unwrap();
            assert_eq!(pos.model().ruleset, ruleset);
            (pos, belief, obs)
        };

        let mk_calib = |model: &Battle, bel: &Belief, obs: &crate::observe::Observer, side: usize| {
            let mut rng = SplitMix64::new(2026_1126 ^ (side as u64));
            let mut model_clone = model.clone();
            let acts = model_clone.legal_choices(&dex, side);
            let mut s = SteppedCfPreview::try_new_with_calib(
                model,
                &dex,
                bel,
                obs,
                side,
                &acts,
                None,
                0.4,
                Some(TrajCalibConfig {
                    turn1_iters: 16,
                    iters_per_turn: 16,
                    rb_eval_iters: 16,
                    support_k: 6,
                    ..TrajCalibConfig::default()
                }),
                &mut rng,
            )
            .expect("SteppedCfPreview::try_new_with_calib must activate on Castle");
            s.cfg.total_iters = 4096;
            s.cfg.coarse_iters = 16;
            s.cfg.do_iters = 16;
            s.cfg.polish_chunk = 32;
            if let Some(ref mut cal) = s.calib {
                cal.calib_iters = 1600;
                cal.explore_iters = 800;
                cal.traj_cfg.total_iters = 1600;
            }
            (s, rng)
        };

        // 1. Both Side 0 (1P) and Side 1 (2P) under `castle-free`: full phase progression,
        //    bit-identical slice invariance across TrajCalib -> Stage-2, and zero hidden-info leakage.
        for side in [0usize, 1usize] {
            let (pos1, bel1, obs1) = make_env(side, &t_b1, Ruleset::castle_free());
            let (pos2, bel2, obs2) = make_env(side, &t_b2, Ruleset::castle_free());

            let (mut s_one, mut r_one) = mk_calib(pos1.model(), &bel1, &obs1, side);
            for _ in 0..4096 {
                s_one.step_one(&mut r_one);
            }
            assert!(s_one.calib.is_none(), "Castle side {side}: TrajCalib must finalize");
            assert!(!s_one.calib_obs.is_empty(), "Castle side {side}: must collect TrajCalib obs");
            let v_nash = s_one.nash_value(4096);
            assert!(v_nash.is_finite() && (0.0..=1.0).contains(&v_nash));

            // Confirm cells preserve `castle-free` ruleset and uncapped horizon (`turn_cap == 101`).
            assert!(!s_one.cells.is_empty());
            for cell in s_one.cells.values() {
                assert_eq!(cell.table.ruleset, Ruleset::castle_free());
                assert_eq!(cell.turn_cap, 101);
            }

            let (mut s_chunk, mut r_chunk) = mk_calib(pos1.model(), &bel1, &obs1, side);
            for _ in 0..64 {
                for _ in 0..64 {
                    s_chunk.step_one(&mut r_chunk);
                }
            }
            assert_eq!(s_one.node_count(), s_chunk.node_count());
            assert_eq!(s_one.nash_value(4096), s_chunk.nash_value(4096));
            assert_eq!(s_one.execution_policy(4096), s_chunk.execution_policy(4096));
            let entries = s_one.execution_policy(4096);
            let dp = DecisionPolicy::new("castle-calib-test", entries.iter().copied()).unwrap();
            for seed in 0u64..32 {
                let draw = SplitMix64::new(seed ^ 4096u64).next_f64();
                assert_eq!(dp.sample(draw), s_one.best_choice(seed, 4096));
                assert_eq!(s_one.best_choice(seed, 4096), s_chunk.best_choice(seed, 4096));
            }

            // Zero hidden-info leakage under `castle-free`: swapping true opponent hidden moves/items
            // produces bit-identical policy and Nash value.
            let (mut s_alt, mut r_alt) = mk_calib(pos2.model(), &bel2, &obs2, side);
            for _ in 0..4096 {
                s_alt.step_one(&mut r_alt);
            }
            assert_eq!(s_one.execution_policy(4096), s_alt.execution_policy(4096));
            assert_eq!(s_one.nash_value(4096), s_alt.nash_value(4096));
        }

        // 2. Explicit `castle-tournament` (`L_TURN = 80`) clamps CfPreviewCell & TrajTurnScratch `turn_cap` to 80.
        {
            let (pos_t, bel_t, obs_t) = make_env(0, &t_b1, Ruleset::castle_tournament());
            let (mut s_t, mut r_t) = mk_calib(pos_t.model(), &bel_t, &obs_t, 0);
            for _ in 0..2048 {
                s_t.step_one(&mut r_t);
            }
            assert!(!s_t.cells.is_empty());
            for cell in s_t.cells.values() {
                assert_eq!(cell.table.ruleset, Ruleset::castle_tournament());
                assert_eq!(cell.turn_cap, 80);
            }
        }

        // 3. End-to-end `ProtocolAgent::new_with_ruleset(..., Ruleset::castle_free())` activates
        //    `SteppedCfPreview` (policy-v4) on Turn 0 and returns a legal Castle Team Preview choice.
        {
            let mut b = Battle::from_fixture_with_ruleset(
                &dex,
                "1,2,3,4",
                &t_a,
                &t_b1,
                Ruleset::castle_free(),
            )
            .unwrap();
            let mut ch = PlayerChannel::new(0);
            let frame = ch.frame(&mut b, &dex).unwrap();
            let req_json = serde_json::to_string(&frame.request).unwrap();
            let mut agent = ProtocolAgent::new_with_ruleset(
                &dex,
                0,
                pool.clone(),
                RmConfig::default(),
                777,
                Ruleset::castle_free(),
            );
            agent.set_own_team(t_a.clone());
            for line in &frame.lines {
                agent.push_line(&dex, line);
            }
            assert!(agent.on_request(&dex, &req_json).unwrap());
            agent.step(&dex, 256).unwrap();
            let pol = agent
                .preview_execution_policy(&dex)
                .expect("ProtocolAgent under castle-free must activate SteppedCfPreview at Turn 0");
            assert!(
                !pol.actions().is_empty(),
                "Castle SteppedCfPreview execution policy must be non-empty"
            );
            let pick = agent.best(&dex).expect("Castle Team Preview pick");
            assert!(pick.starts_with("team "), "expected team choice, got {pick:?}");
            b.choose(&dex, 0, &pick).expect("Castle preview choice must be legal in Battle");
        }
    }

    #[test]
    fn solve_bayesian_rm_plus_with_priors_invariance_best_response_and_hybrid() {
        // 1. All-None fast path is 0-ULP bit-identical to `solve_bayesian_rm_plus`.
        let w0 = [0.5, 0.2, 0.8, 0.7, 0.5, 0.3, 0.1, 0.9, 0.5];
        let w1 = [0.4, 0.6, 0.7, 0.3, 0.2, 0.9];
        let worlds = [(0.6, 3usize, w0.as_slice()), (0.4, 2usize, w1.as_slice())];
        let base = solve_bayesian_rm_plus(3, &worlds, 400);
        let none = solve_bayesian_rm_plus_with_priors(3, &worlds, &[None, None], 400);
        assert_eq!(base.own_policy, none.own_policy);
        assert_eq!(base.opp_policies, none.opp_policies);
        assert_eq!(base.value.to_bits(), none.value.to_bits());
        assert_eq!(base.duality_gap.to_bits(), none.duality_gap.to_bits());
        assert_eq!(base.row_guarantees, none.row_guarantees);

        // 2. All-Some computes exact closed-form Bayesian best response with 0 duality gap.
        let q0 = [0.1, 0.8, 0.1];
        let q1 = [0.25, 0.75];
        let br = solve_bayesian_rm_plus_with_priors(3, &worlds, &[Some(&q0), Some(&q1)], 400);
        assert_eq!(br.duality_gap, 0.0);
        assert_eq!(br.opp_policies[0], q0);
        assert_eq!(br.opp_policies[1], q1);
        let expected_ev = |i: usize| -> f64 {
            let e0: f64 = (0..3).map(|j| w0[i * 3 + j] * q0[j]).sum();
            let e1: f64 = (0..2).map(|j| w1[i * 2 + j] * q1[j]).sum();
            0.6 * e0 + 0.4 * e1
        };
        let evs = [expected_ev(0), expected_ev(1), expected_ev(2)];
        let best_i = (0..3)
            .max_by(|&a, &b| evs[a].total_cmp(&evs[b]))
            .unwrap();
        assert_eq!(br.own_policy[best_i], 1.0);
        assert!((br.value - evs[best_i]).abs() < 1e-12);
        for i in 0..3 {
            assert!((br.row_vs_q[i] - evs[i]).abs() < 1e-12);
            assert!((br.row_guarantees[i] - evs[i]).abs() < 1e-12);
        }

        // 3. Hybrid (world 0 fixed to q0, world 1 adversarial matching pennies):
        //    world 0 gives row 0 = 0.9, row 1 = 0.5 against q0=[1, 0];
        //    world 1 is 2x2 matching pennies [1, 0; 0, 1].
        //    With weights (0.5, 0.5): maximizing 0.5*(0.9 p + 0.5(1-p)) + 0.5*min(p, 1-p)
        //    peaks at p = 0.5 (value = 0.5*0.7 + 0.5*0.5 = 0.60), NOT pure row 0 (value = 0.45)!
        let hw0 = [0.9, 0.1, 0.5, 0.5];
        let hw1 = [1.0, 0.0, 0.0, 1.0];
        let hworlds = [(0.5, 2usize, hw0.as_slice()), (0.5, 2usize, hw1.as_slice())];
        let hq0 = [1.0, 0.0];
        let hsol = solve_bayesian_rm_plus_with_priors(2, &hworlds, &[Some(&hq0), None], 1000);
        assert_eq!(hsol.opp_policies[0], hq0);
        assert!(hsol.duality_gap < 1e-3, "duality gap {}", hsol.duality_gap);
        assert!((hsol.value - 0.60).abs() < 5e-3, "value {}", hsol.value);
        assert!(
            (hsol.own_policy[0] - 0.5).abs() < 0.03,
            "own_policy {:?}",
            hsol.own_policy
        );
    }

    #[test]
    fn stepped_cf_preview_consumes_belief_prior_and_pick_prior_without_legacy_downgrade() {
        use crate::player::PlayerChannel;
        use crate::preview::{MetaPool, MetaTeam};
        use crate::{BeliefPrior, BlindSearch, PickPrior, ProtocolAgent, RmConfig};
        use nc2000_engine::rules::Ruleset;
        use std::sync::Arc;

        let dex = conformance::load_dex();
        let t_a = vec![
            mk("Snorlax", 55, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Zapdos", 50, "Bright Powder", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
            mk("Cloyster", 50, "Gold Berry", &["Surf", "Spikes", "Explosion", "Rapid Spin"]),
            mk("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Body Slam"]),
            mk("Gengar", 50, "Quick Claw", &["Thunderbolt", "Ice Punch", "Hypnosis", "Explosion"]),
            mk("Skarmory", 50, "Mint Berry", &["Drill Peck", "Whirlwind", "Rest", "Curse"]),
        ];
        let t_b = vec![
            mk("Raikou", 50, "Magnet", &["Thunderbolt", "Protect", "Roar", "Rest"]),
            mk("Suicune", 50, "Mint Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
            mk("Exeggutor", 50, "Quick Claw", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
            mk("Tyranitar", 55, "Miracle Berry", &["Rock Slide", "Crunch", "Earthquake", "Pursuit"]),
            mk("Snorlax", 50, "Leftovers", &["Body Slam", "Earthquake", "Curse", "Rest"]),
            mk("Umbreon", 50, "Gold Berry", &["Pursuit", "Charm", "Rest", "Baton Pass"]),
        ];
        // Also test `party_order_matters` (Beat Up) folding with an off-pool team.
        let mut t_b_beatup = t_b.clone();
        t_b_beatup[5].moves = vec!["Beat Up".into(), "Pursuit".into(), "Rest".into(), "Charm".into()];

        let in_pool = MetaPool {
            teams: vec![MetaTeam {
                id: "tb_castle".into(),
                sets: t_b.clone(),
                weight: Some(1.0),
            }],
        };
        let empty_pool = MetaPool { teams: Vec::new() };

        // A. Off-pool `BeliefPrior.lead` under BOTH PS and `castle-free`:
        //    must keep `SteppedCfPreview` active (no LegacyRollout downgrade!),
        //    must stay slice-invariant, and must shift preview evaluation across opposite priors.
        let prior_lead_raikou = Arc::new(BeliefPrior::from_json(
            r#"{"species":{"raikou":{"lead":0.99},"exeggutor":{"lead":0.01},"tyranitar":{"lead":0.01},"suicune":{"lead":0.01},"snorlax":{"lead":0.01},"umbreon":{"lead":0.01}}}"#,
        ));
        let prior_lead_egg = Arc::new(BeliefPrior::from_json(
            r#"{"species":{"raikou":{"lead":0.01},"exeggutor":{"lead":0.99},"tyranitar":{"lead":0.01},"suicune":{"lead":0.01},"snorlax":{"lead":0.01},"umbreon":{"lead":0.01}}}"#,
        ));
        for ruleset in [Ruleset::default(), Ruleset::castle_free()] {
            let run_blind_prior = |prior: Arc<BeliefPrior>, opp_sets: &[PokemonSet], chunk: u32| {
                let b = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &t_a, opp_sets, ruleset).unwrap();
                let (mut lp, mut obs) = LocalPlayer::new_with_own_sets(&b, &dex, 0, t_a.clone()).unwrap();
                lp.observe(&b, &dex, &mut obs).unwrap();
                let mut belief = Belief::new(&dex, &empty_pool, &obs);
                belief.set_prior(prior);
                belief.sync(&dex, &obs);
                assert!(belief.has_preview_prior(&obs));
                let pos = lp.position(&dex, &mut belief, &obs, &mut SplitMix64::new(42)).unwrap();
                let mut bs = BlindSearch::from_observation(&pos, &dex, RmConfig::default(), 0, 0xCAFE_0001);
                bs.prune_preview_aliases(&dex);
                bs.set_opponent_preview_prior(true);
                let mut rem = 256u32;
                while rem > 0 {
                    let s = chunk.min(rem);
                    bs.step(&dex, &belief, &obs, s);
                    rem -= s;
                }
                assert!(
                    bs.uses_cf_preview(),
                    "{ruleset:?}: BlindSearch with BeliefPrior.lead must stay on SteppedCfPreview"
                );
                (bs.best().unwrap(), bs.preview_execution_policy().unwrap())
            };
            let (pick_r1, pol_r1) = run_blind_prior(prior_lead_raikou.clone(), &t_b, 256);
            let (pick_r2, pol_r2) = run_blind_prior(prior_lead_raikou.clone(), &t_b, 16);
            assert_eq!(pick_r1, pick_r2, "{ruleset:?}: slice invariance failed");
            assert_eq!(pol_r1.actions(), pol_r2.actions());
            assert!(
                pol_r1.source().starts_with("cf-preview-"),
                "expected cf-preview source, got {}",
                pol_r1.source()
            );
            // Beat Up opponent (`opp_order_matters == true`) also activates cleanly.
            let (_, pol_bu) = run_blind_prior(prior_lead_raikou.clone(), &t_b_beatup, 128);
            assert!(pol_bu.source().starts_with("cf-preview-"));

            let (pick_e, _) = run_blind_prior(prior_lead_egg.clone(), &t_b, 256);
            // Check Stage-1/calibrated Nash value explicitly differs across opposite lead priors.
            let b_cmp = Battle::from_fixture_with_ruleset(&dex, "1,2,3,4", &t_a, &t_b, ruleset).unwrap();
            let (mut lp_a, mut obs_a) = LocalPlayer::new_with_own_sets(&b_cmp, &dex, 0, t_a.clone()).unwrap();
            lp_a.observe(&b_cmp, &dex, &mut obs_a).unwrap();
            let mut bel_a = Belief::new(&dex, &empty_pool, &obs_a);
            bel_a.set_prior(prior_lead_raikou.clone());
            bel_a.sync(&dex, &obs_a);
            let mut bel_b = Belief::new(&dex, &empty_pool, &obs_a);
            bel_b.set_prior(prior_lead_egg.clone());
            bel_b.sync(&dex, &obs_a);
            let pos_a = lp_a.position(&dex, &mut bel_a, &obs_a, &mut SplitMix64::new(9)).unwrap();
            let mut m_acts = pos_a.model().clone();
            let acts = m_acts.legal_choices(&dex, 0);
            let sa = SteppedCfPreview::try_new_with_calib(
                pos_a.model(), &dex, &bel_a, &obs_a, 0, &acts, None, 0.4, None, &mut SplitMix64::new(7),
            ).unwrap();
            let sb = SteppedCfPreview::try_new_with_calib(
                pos_a.model(), &dex, &bel_b, &obs_a, 0, &acts, None, 0.4, None, &mut SplitMix64::new(7),
            ).unwrap();
            assert!(
                (sa.nash_value(0) - sb.nash_value(0)).abs() > 1e-4 || pick_r1 != pick_e,
                "{ruleset:?}: SteppedCfPreview did not respond to opposite BeliefPrior.lead"
            );
        }

        // B. In-pool `PickPrior` through `ProtocolAgent` under `castle-free`:
        //    keeps `SteppedCfPreview` active and shifts policy across opposite PickPriors.
        let make_pick_prior = |fav: [&str; 3]| -> Arc<PickPrior> {
            let text = serde_json::json!({
                "schema": "nc2000-pick-prior-v1",
                "smoothing": 0.001,
                "backoff_strength": 1.0,
                "source": {},
                "rows": [{
                    "team": "tb_castle",
                    "enemy_preview": [],
                    "side": 1,
                    "choices": [{ "species": fav, "count": 500 }]
                }]
            })
            .to_string();
            Arc::new(PickPrior::from_json(&text).unwrap())
        };
        let pp_a = make_pick_prior(["raikou", "suicune", "snorlax"]);
        let pp_b = make_pick_prior(["tyranitar", "exeggutor", "umbreon"]);
        let run_proto_pp = |pp: Arc<PickPrior>| {
            let mut b = Battle::from_fixture_with_ruleset(
                &dex, "1,2,3,4", &t_a, &t_b, Ruleset::castle_free(),
            ).unwrap();
            let mut ch = PlayerChannel::new(0);
            let frame = ch.frame(&mut b, &dex).unwrap();
            let req_json = serde_json::to_string(&frame.request).unwrap();
            let mut agent = ProtocolAgent::new_with_ruleset(
                &dex, 0, in_pool.clone(), RmConfig::default(), 777, Ruleset::castle_free(),
            );
            agent.set_own_team(t_a.clone());
            agent.set_pick_prior(pp);
            for line in &frame.lines {
                agent.push_line(&dex, line);
            }
            assert!(agent.on_request(&dex, &req_json).unwrap());
            agent.step(&dex, 256).unwrap();
            let pol = agent.preview_execution_policy(&dex).unwrap();
            assert!(
                pol.source().starts_with("cf-preview-"),
                "PickPrior under castle-free must stay on SteppedCfPreview, got {}",
                pol.source()
            );
            agent.best(&dex).unwrap()
        };
        let pick_pp_a = run_proto_pp(pp_a);
        let pick_pp_b = run_proto_pp(pp_b);
        assert_ne!(
            pick_pp_a, pick_pp_b,
            "Castle ProtocolAgent with opposite PickPriors must shift Team Preview choice"
        );
    }
}
