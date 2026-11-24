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
    MatchTable,
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
            total_iters: 240_000,
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
    let mut det_worlds: Vec<(f64, Battle)> = Vec::with_capacity(world_plan.len());
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
        det_worlds.push((weight, sim_w));
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

    for (weight, mut sim_w) in det_worlds {
        let (_, opp_canon) = canonical_preview_actions(dex, &mut sim_w, opp);
        let n_opp = opp_canon.len();
        if n_opp == 0 {
            continue;
        }
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
    let sol1 = solve_bayesian_rm_plus(n_own, &s1_slices, cfg.rm_sweeps_stage1);
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
        }
        let slices: Vec<(f64, usize, &[f64])> = (0..mats.len())
            .map(|idx| (w_list[idx], col_lens[idx], mats[idx].as_slice()))
            .collect();
        let sol = solve_bayesian_rm_plus(k_me, &slices, sweeps);
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

enum SteppedPreviewPhase {
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
pub(crate) struct SteppedCfPreview {
    my_acts: Vec<SearchChoice>,
    side: usize,
    cfg: PreviewSearchConfig,
    own_canon: Vec<(usize, [u8; 3])>,
    worlds: Vec<PreviewWorldData>,
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
    #[allow(clippy::too_many_arguments)]
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

        let own_tactical = lead_tactical_flags(dex, base, side);
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
        for (weight, mut sim_w) in det_worlds {
            let (_, opp_canon) = canonical_preview_actions(dex, &mut sim_w, opp);
            let n_opp = opp_canon.len();
            if n_opp == 0 {
                continue;
            }
            let opp_tactical = lead_tactical_flags(dex, &sim_w, opp);
            let roster = CompactPreviewRoster::from_battle(&sim_w, dex, None);
            let pcache = PreviewTempoCache::build_for_preview_roster(
                &roster,
                [sim_w.sides[0].party.len(), sim_w.sides[1].party.len()],
            );
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
        let sol1 = solve_bayesian_rm_plus(n_own, &s1_slices, cfg.rm_sweeps_stage1);

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

        let coarse_budget = (u64::from(cfg.total_iters) * 45 / 100) as u32;
        let coarse_cap = (cfg.total_iters / 320).clamp(cfg.coarse_iters.min(640), 640).max(1);
        let mut coarse_per_w = vec![cfg.coarse_iters; n_w2];
        for wi in 0..n_w2 {
            let w_budget = ((f64::from(coarse_budget) * s2_alloc_weights[wi]).round() as u32).max(1);
            let n_c = (s_me.len() * s_opp[wi].len()).max(1) as u32;
            coarse_per_w[wi] = (w_budget / n_c).clamp(coarse_cap.min(32), coarse_cap);
        }

        // Tier 0: 1 visit per own support action so small budgets (e.g. visit-sample)
        // immediately span all candidate actions.
        let mut work_queue = std::collections::VecDeque::with_capacity(s_me.len() * 2);
        let first_opp = s_opp[0][0];
        for &own_i in &s_me {
            work_queue.push_back((0usize, own_i, first_opp, 1u32));
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
            worlds,
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
            phase: SteppedPreviewPhase::CoarseTier0,
        })
    }

    pub(crate) fn node_count(&self) -> usize {
        self.total_nodes
    }

    fn solve_subgame(&self, sweeps: u32) -> (Vec<Vec<f64>>, BayesianRmSolution) {
        let k_me = self.s_me.len();
        let n_w2 = self.s2_world_ids.len();
        let n_tail = self.tail_world_ids.len();
        let mut mats: Vec<Vec<f64>> = Vec::with_capacity(n_w2 + n_tail);
        let mut col_lens: Vec<usize> = Vec::with_capacity(n_w2 + n_tail);
        let mut w_list: Vec<f64> = Vec::with_capacity(n_w2 + n_tail);
        for wi in 0..n_w2 {
            let w_id = self.s2_world_ids[wi];
            let wd = &self.worlds[w_id];
            let m_full = wd.opp_canon.len();
            let k_opp = self.s_opp[wi].len();
            let mut mat = vec![0.5f64; k_me * k_opp];
            for (ri, &own_i) in self.s_me.iter().enumerate() {
                for (cj, &opp_j) in self.s_opp[wi].iter().enumerate() {
                    let v = if let Some(cell) = self.cells.get(&(wi, own_i, opp_j)) {
                        let v0 = cell.value0(self.cfg.cell_readout);
                        if self.side == 0 { v0 } else { 1.0 - v0 }
                    } else {
                        wd.stage1_vals[own_i * m_full + opp_j]
                    };
                    mat[ri * k_opp + cj] = v;
                }
            }
            mats.push(mat);
            col_lens.push(k_opp);
            w_list.push(self.s2_post_weights[wi]);
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
        }
        let slices: Vec<(f64, usize, &[f64])> = (0..mats.len())
            .map(|idx| (w_list[idx], col_lens[idx], mats[idx].as_slice()))
            .collect();
        let sol = solve_bayesian_rm_plus(k_me, &slices, sweeps);
        (mats, sol)
    }

    fn build_do_opp_candidates(&self, wi: usize, supp_me: &[usize], supp_p_sum: f64) -> Vec<usize> {
        let n_w2 = self.s2_world_ids.len();
        if n_w2 > 1 && self.s2_alloc_weights[wi] < 0.04 {
            return Vec::new();
        }
        let wd = &self.worlds[self.s2_world_ids[wi]];
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
                    let do_cap_iters = (u64::from(self.cfg.total_iters) * 65 / 100) as u32;
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
                                let v = self.cells.get(&(wi, own_i, cand_j)).map_or(0.5, |c| {
                                    let v0 = c.value0(self.cfg.cell_readout);
                                    if self.side == 0 { v0 } else { 1.0 - v0 }
                                });
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
                                    let v = self.cells.get(&(wi, cand_i, opp_j)).map_or(0.5, |c| {
                                        let v0 = c.value0(self.cfg.cell_readout);
                                        if self.side == 0 { v0 } else { 1.0 - v0 }
                                    });
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
                            if let Some(worst_cj) = (0..self.s_opp[wi].len()).min_by(|&a, &b| {
                                let va = self
                                    .cells
                                    .get(&(wi, own_i, self.s_opp[wi][a]))
                                    .map_or(0.5, |c| {
                                        let v0 = c.value0(self.cfg.cell_readout);
                                        if self.side == 0 { v0 } else { 1.0 - v0 }
                                    });
                                let vb = self
                                    .cells
                                    .get(&(wi, own_i, self.s_opp[wi][b]))
                                    .map_or(0.5, |c| {
                                        let v0 = c.value0(self.cfg.cell_readout);
                                        if self.side == 0 { v0 } else { 1.0 - v0 }
                                    });
                                va.total_cmp(&vb)
                            }) {
                                if !active_cols.contains(&worst_cj) {
                                    active_cols.push(worst_cj);
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

    /// Execute exactly ONE Turn-1 `cfv-0r@puct@lam0` iteration.
    /// Returns `(own_orig_idx, opp_choice, val0, val_mine, depth)`.
    pub(crate) fn step_one(&mut self, rng: &mut SplitMix64) -> (usize, SearchChoice, f64, f64, u32) {
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
        let prior_v_mine = wd.stage1_vals[own_i * m + opp_j];
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

    /// Solve the `LcbGatedMixed` Bayesian RM+ readout over the full 100% posterior.
    /// Returns `(use_mix, best_pure_r, p_mix)` without mutating search state.
    fn solve_execution_mix(&self, done: u32) -> (bool, usize, Vec<f64>) {
        let sweeps = if done >= 4096 {
            self.cfg.rm_sweeps_final
        } else {
            self.cfg.rm_sweeps_stage1
        };
        let (final_mats, final_sol) = self.solve_subgame(sweeps);
        let k_me = self.s_me.len();
        let n_w2 = self.s2_world_ids.len();
        let best_pure_r = (0..k_me)
            .max_by(|&a, &b| {
                final_sol.row_guarantees[a]
                    .total_cmp(&final_sol.row_guarantees[b])
                    .then_with(|| final_sol.row_vs_q[a].total_cmp(&final_sol.row_vs_q[b]))
            })
            .unwrap_or(0);
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
            let k_opp = self.s_opp[wi].len();
            let mat = &final_mats[wi];
            let mut min_col = f64::INFINITY;
            let mut min_col_lcb = f64::INFINITY;
            for cj in 0..k_opp {
                let opp_j = self.s_opp[wi][cj];
                let mut mean_j = 0.0;
                let mut var_j = 0.0;
                for ri in 0..k_me {
                    let pr = p_mix[ri];
                    if pr > 0.0 {
                        let v = mat[ri * k_opp + cj];
                        mean_j += pr * v;
                        let n_c = self
                            .cells
                            .get(&(wi, self.s_me[ri], opp_j))
                            .map_or(1, |c| c.total_n.max(1));
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
            mixed_guarantee += self.s2_post_weights[wi] * min_col;
            mixed_guarantee_lcb += self.s2_post_weights[wi] * min_col_lcb;
        }
        for (ti, &w_id) in self.tail_world_ids.iter().enumerate() {
            let m_full = self.worlds[w_id].opp_canon.len();
            let mat = &final_mats[n_w2 + ti];
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

    /// Read-only `LcbGatedMixed` execution distribution over canonical legal preview picks.
    /// Exact match to the distribution sampled by `best_choice(decision_seed, done)`.
    pub(crate) fn execution_policy(&self, done: u32) -> Vec<(SearchChoice, f64)> {
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
    pub(crate) fn best_choice(&self, decision_seed: u64, done: u32) -> SearchChoice {
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
}
