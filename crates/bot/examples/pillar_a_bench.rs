//! Pillar A: Live Trajectory-DO Team Preview Ablation & Independent Verification Harness.
//!
//! Emits a single-line compact JSON summary to stdout and writes full traces to `/tmp/pillar_a_*.json`.
//!
//! Modes:
//! - `--mode a1-rb`: Lever A1 Rao-Blackwellized early-exit bias, variance & speed audit on 10 key cells.
//! - `--mode a234-ablate`: Levers A1–A4 systematic ablation on `m1, m2, m3` (8 independent seeds each)
//!   plus held-out cell-swap residual RMSE validation.
//! - `--mode suite1-blind9k`: Independent Verification Suite 1 — real `BlindAgent 9k` H2H and 48-arm
//!   unilateral exploitability on `m1, m2, m3` vs shipped `policy-v2` (`SteppedCfPreview 240k`).
//! - `--mode suite2-multiworld`: Independent Verification Suite 2 — unseen `belief-pool-v3`
//!   (`hash % 5 == 0`) multi-world (`W = 1..24`) latency, zero-leakage, and real `BlindAgent 9k` H2H.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use nc2000_bot::accounting::LeafSettlementMode;
use nc2000_bot::agent::Agent;
use nc2000_bot::belief::Belief;
use nc2000_bot::blind::{BlindAgent, BlindSearch};
use nc2000_bot::player_state::LocalPlayer;
use nc2000_bot::preview::{MetaPool, MetaTeam};
use nc2000_bot::preview_search::{
    canonical_preview_triples, play_compact_traj_with_rb, solve_traj_preview, RbExitMode,
    SteppedCfPreview, SteppedTrajPreview, TrajCalibConfig, TrajCrnMode, TrajPolicyMode,
    TrajPreviewConfig, TrajRidgeBasis, TrajTurnScratch,
};
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_bot::tempo::PreviewTempoCache;
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::compact::CompactPreviewRoster;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, RequestState};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
struct CatalogFile {
    teams: Vec<CatalogTeam>,
}

#[derive(Clone, Deserialize)]
struct CatalogTeam {
    id: String,
    sets: Vec<PokemonSet>,
    #[serde(default)]
    weight: Option<f64>,
}

fn repo_root() -> PathBuf {
    conformance::fixture::repo_root()
}

fn canonical_triple(t: [u8; 3]) -> [u8; 3] {
    if t[2] == 0 || t[1] <= t[2] {
        t
    } else {
        [t[0], t[2], t[1]]
    }
}

fn canonical_team_hash(sets: &[PokemonSet]) -> u64 {
    let mut keys: Vec<String> = sets
        .iter()
        .map(|s| {
            let mut mv = s.moves.clone();
            mv.sort();
            format!("{}|{}|{}|{}", s.species, s.level, s.item, mv.join(","))
        })
        .collect();
    keys.sort();
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for k in &keys {
        for b in k.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x1000_0000_01B3);
        }
        h ^= 0xFF;
        h = h.wrapping_mul(0x1000_0000_01B3);
    }
    h
}

fn r4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn mean_sd(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let m = xs.iter().sum::<f64>() / (xs.len() as f64);
    if xs.len() <= 1 {
        return (m, 0.0);
    }
    let var = xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / ((xs.len() - 1) as f64);
    (m, var.sqrt())
}

fn median_p95(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let med = if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    };
    let p95_idx = (((n as f64) * 0.95).ceil() as usize).saturating_sub(1).min(n - 1);
    (med, s[p95_idx])
}

fn mean_pairwise_tv(pols: &[Vec<([u8; 3], f64)>]) -> f64 {
    let n = pols.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for a in 0..n {
        for b in (a + 1)..n {
            let mut map: HashMap<[u8; 3], f64> = HashMap::new();
            for &(t, p) in &pols[a] {
                *map.entry(t).or_insert(0.0) += p;
            }
            for &(t, p) in &pols[b] {
                *map.entry(t).or_insert(0.0) -= p;
            }
            let tv = 0.5 * map.values().map(|v| v.abs()).sum::<f64>();
            sum += tv;
            cnt += 1;
        }
    }
    sum / (cnt as f64)
}

fn avg_policy_dist(pols: &[Vec<([u8; 3], f64)>]) -> Vec<([u8; 3], f64)> {
    let mut map: HashMap<[u8; 3], f64> = HashMap::new();
    let inv = 1.0 / (pols.len().max(1) as f64);
    for p in pols {
        for &(t, pr) in p {
            *map.entry(t).or_insert(0.0) += pr * inv;
        }
    }
    let mut out: Vec<([u8; 3], f64)> = map.into_iter().filter(|&(_, p)| p > 1e-6).collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

fn sample_pick(dist: &[([u8; 3], f64)], draw: f64) -> [u8; 3] {
    let mut acc = 0.0;
    for &(t, p) in dist {
        acc += p;
        if draw < acc {
            return t;
        }
    }
    dist[0].0
}

fn fmt_triple(sets: &[PokemonSet], t: [u8; 3]) -> String {
    let sp = |s: u8| -> &str {
        if s == 0 || (s as usize) > sets.len() {
            "?"
        } else {
            sets[s as usize - 1].species.as_str()
        }
    };
    format!("[{},{},{}]({}/{}/{})", t[0], t[1], t[2], sp(t[0]), sp(t[1]), sp(t[2]))
}

fn one_hot_preview_policy(pick: [u8; 3]) -> Vec<f64> {
    let ct = canonical_triple(pick);
    let acts = nc2000_bot::preview::preview_actions();
    let mut pol = vec![0.0f64; acts.len()];
    if let Some(pos) = acts.iter().position(|&a| a == ct) {
        pol[pos] = 1.0;
    }
    pol
}

struct LegalGuardBlind {
    inner: BlindAgent,
}

impl Agent for LegalGuardBlind {
    fn name(&self) -> String {
        self.inner.name()
    }

    fn requires_player_log(&self) -> bool {
        true
    }

    fn choose(
        &mut self,
        battle: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let ch = self.inner.choose(battle, dex, side, choices);
        if choices.contains(&ch) {
            ch
        } else {
            choices[0]
        }
    }
}

fn play_blind_battle_with_picks(
    dex: &Dex,
    sets_p0: &[PokemonSet],
    sets_p1: &[PokemonSet],
    pool_p0: Arc<MetaPool>,
    pool_p1: Arc<MetaPool>,
    pick_p0: [u8; 3],
    pick_p1: [u8; 3],
    battle_iters: u32,
    battle_seed: u64,
    agent_seed_p0: u64,
    agent_seed_p1: u64,
) -> f64 {
    let mut brng = SplitMix64::new(battle_seed);
    let bseed = brng.battle_seed();
    let mut b = Battle::from_fixture(dex, &bseed, sets_p0, sets_p1).unwrap();
    b.set_log_enabled(true);
    let cfg = RmConfig {
        iterations: battle_iters,
        rule: SelRule::Ucb,
        c: 0.4,
        settlement: LeafSettlementMode::CfValuePuct,
        ..Default::default()
    };
    let mut a0 = LegalGuardBlind {
        inner: BlindAgent::new(cfg.clone(), pool_p0, None, agent_seed_p0)
            .with_own_preview_policy(one_hot_preview_policy(pick_p0)),
    };
    let mut a1 = LegalGuardBlind {
        inner: BlindAgent::new(cfg, pool_p1, None, agent_seed_p1)
            .with_own_preview_policy(one_hot_preview_policy(pick_p1)),
    };
    let mut agents: [&mut dyn Agent; 2] = [&mut a0, &mut a1];
    match play_game(dex, &mut b, &mut agents, 300).unwrap() {
        GameResult::Outcome(Outcome::P1Win) => 1.0,
        GameResult::Outcome(Outcome::P2Win) => 0.0,
        _ => 0.5,
    }
}

fn make_calib_cfg(tag: &str) -> TrajCalibConfig {
    match tag {
        "cal2_p19_lam55_le3" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 48,
            iters_per_turn: 24,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_t1_beta: 1.0,
            ..TrajCalibConfig::default()
        },
        "cal2_p19_lam35_le3" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 48,
            iters_per_turn: 24,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 35.0,
            ridge_lambda_inter: 60.0,
            per_world_delta: false,
            delta_t1_beta: 1.0,
            ..TrajCalibConfig::default()
        },
        "cal2_p19_lam55_le4" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 48,
            iters_per_turn: 24,
            rb_max_alive: 4,
            rb_eval_iters: 48,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_t1_beta: 1.0,
            ..TrajCalibConfig::default()
        },
        "cal2_p19_lam80_le3_soft" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 48,
            iters_per_turn: 24,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 80.0,
            ridge_lambda_inter: 100.0,
            per_world_delta: false,
            delta_t1_beta: 1.0,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            ..TrajCalibConfig::default()
        },
        "cal2_p19_lam55_b080" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 48,
            iters_per_turn: 24,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_t1_beta: 0.80,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            ..TrajCalibConfig::default()
        },
        "cal3_ipt32_lam55_b080" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.80,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            ..TrajCalibConfig::default()
        },
        "cal3_ipt32_lam45_b085" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 45.0,
            ridge_lambda_inter: 70.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.85,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            ..TrajCalibConfig::default()
        },
        "cal3_ipt32_lam55_b065" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_s1_beta: 0.85,
            delta_t1_beta: 0.65,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            ..TrajCalibConfig::default()
        },
        "cal3_ipt48_lam55_b080" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 64,
            iters_per_turn: 48,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.80,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            ..TrajCalibConfig::default()
        },
        "bs_lam45_b085_d10" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 45.0,
            ridge_lambda_inter: 70.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.85,
            bootstrap_draws: 16,
            bootstrap_dust: 0.10,
            ..TrajCalibConfig::default()
        },
        "bs_lam35_b100_d12" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 35.0,
            ridge_lambda_inter: 60.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 1.00,
            bootstrap_draws: 16,
            bootstrap_dust: 0.12,
            ..TrajCalibConfig::default()
        },
        "bs_lam55_b080_d10" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.80,
            bootstrap_draws: 16,
            bootstrap_dust: 0.10,
            ..TrajCalibConfig::default()
        },
        "bs_lam45_b085_ipt24_d10" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 48,
            iters_per_turn: 24,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 45.0,
            ridge_lambda_inter: 70.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.85,
            bootstrap_draws: 16,
            bootstrap_dust: 0.10,
            ..TrajCalibConfig::default()
        },
        "cal4_fullopp_lcb_lam55_b080" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.80,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            bootstrap_draws: 0,
            union_raw_support: true,
            full_opp_readout: true,
            ..TrajCalibConfig::default()
        },
        "cal4_fullopp_lcb_lam45_b085" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 45.0,
            ridge_lambda_inter: 70.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.85,
            min_mix_margin: 0.010,
            lcb_z: 0.75,
            bootstrap_draws: 0,
            union_raw_support: true,
            full_opp_readout: true,
            ..TrajCalibConfig::default()
        },
        "cal4_fullopp_bs_lam45_b085_d15" => TrajCalibConfig {
            calib_frac: 0.40,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 45.0,
            ridge_lambda_inter: 70.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.85,
            bootstrap_draws: 16,
            bootstrap_dust: 0.15,
            union_raw_support: true,
            full_opp_readout: true,
            ..TrajCalibConfig::default()
        },
        "cal4_fullopp_bs_lam55_b080_d15" => TrajCalibConfig {
            calib_frac: 0.35,
            turn1_iters: 64,
            iters_per_turn: 32,
            rb_max_alive: 3,
            rb_eval_iters: 64,
            ridge_basis: TrajRidgeBasis::Present19,
            ridge_lambda_int: 4.0,
            ridge_lambda_main: 55.0,
            ridge_lambda_inter: 80.0,
            per_world_delta: false,
            delta_s1_beta: 1.0,
            delta_t1_beta: 0.80,
            bootstrap_draws: 16,
            bootstrap_dust: 0.15,
            union_raw_support: true,
            full_opp_readout: true,
            ..TrajCalibConfig::default()
        },
        _ => TrajCalibConfig::default(),
    }
}

/// Run Control-Variate Trajectory-Calibrated `SteppedCfPreview` (240k) once.
fn run_calib_once(
    dex: &Dex,
    sets_a: &[PokemonSet],
    sets_b: &[PokemonSet],
    pool: &Arc<MetaPool>,
    side: usize,
    calib_cfg: &TrajCalibConfig,
    iters: u32,
    seed: u64,
) -> (Vec<([u8; 3], f64)>, f64, f64) {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", sets_a, sets_b).unwrap();
    b.set_log_enabled(true);
    let own_sets = if side == 0 { sets_a } else { sets_b };
    let (mut lp, mut obs) =
        LocalPlayer::new_with_own_sets(&b, dex, side, own_sets.to_vec()).unwrap();
    lp.observe(&b, dex, &mut obs).unwrap();
    let mut belief = Belief::new(dex, pool, &obs);
    belief.sync(dex, &obs);
    let mut rng = SplitMix64::new(seed);
    let pos = lp.position(dex, &mut belief, &obs, &mut rng).unwrap();
    let mut base = pos.model().clone();
    let own_legal = base.legal_choices(dex, side);

    let t0 = Instant::now();
    let mut stepped = SteppedCfPreview::try_new_with_calib(
        &base,
        dex,
        &belief,
        &obs,
        side,
        &own_legal,
        None,
        0.4,
        Some(calib_cfg.clone()),
        &mut rng,
    )
    .expect("compact preview supported");
    for _ in 0..iters {
        stepped.step_one(&mut rng);
    }
    let entries = stepped.execution_policy(iters);
    let val_mine = stepped.nash_value(iters);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let val0 = if side == 0 { val_mine } else { 1.0 - val_mine };
    let mut out = Vec::new();
    for (ch, pr) in entries {
        if let SearchChoice::Team(tr) = ch {
            out.push((canonical_triple(tr), pr));
        }
    }
    out.sort_by(|x, y| y.1.total_cmp(&x.1));
    (out, val0, ms)
}

/// Run shipped `SteppedCfPreview` (240k) for a single side & seed, returning `(policy_dist, ms_1t)`.
fn run_shipped_policy_v2(
    dex: &Dex,
    sets_a: &[PokemonSet],
    sets_b: &[PokemonSet],
    pool: &Arc<MetaPool>,
    side: usize,
    iters: u32,
    seed: u64,
) -> (Vec<([u8; 3], f64)>, f64) {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", sets_a, sets_b).unwrap();
    b.set_log_enabled(true);
    let own_sets = if side == 0 { sets_a } else { sets_b };
    let (mut lp, mut obs) =
        LocalPlayer::new_with_own_sets(&b, dex, side, own_sets.to_vec()).unwrap();
    lp.observe(&b, dex, &mut obs).unwrap();
    let mut belief = Belief::new(dex, pool, &obs);
    belief.sync(dex, &obs);
    let mut rng = SplitMix64::new(seed);
    let pos = lp.position(dex, &mut belief, &obs, &mut rng).unwrap();
    let cfg = RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c: 0.4,
        ..Default::default()
    };
    let t0 = Instant::now();
    let mut bs = BlindSearch::from_observation(&pos, dex, cfg, side, seed);
    bs.prune_preview_aliases(dex);
    bs.step(dex, &belief, &obs, iters);
    let pol = bs.preview_execution_policy().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut out = Vec::new();
    for row in pol.actions() {
        if let SearchChoice::Team(tr) = row.input {
            out.push((canonical_triple(tr), row.probability));
        }
    }
    out.sort_by(|x, y| y.1.total_cmp(&x.1));
    (out, ms)
}

fn disclosed_pool(opp_sets: &[PokemonSet]) -> Arc<MetaPool> {
    Arc::new(MetaPool {
        teams: vec![MetaTeam {
            id: "opp-disclosed".into(),
            sets: opp_sets.to_vec(),
            weight: Some(1.0),
        }],
    })
}

/// Run `Traj-DO` once and return `(policy_dist, value_side0, games_completed, stage1_ms, total_ms, stepped_opt)`.
fn run_traj_once(
    dex: &Dex,
    sets_a: &[PokemonSet],
    sets_b: &[PokemonSet],
    pool: &Arc<MetaPool>,
    side: usize,
    cfg: &TrajPreviewConfig,
    seed: u64,
) -> (Vec<([u8; 3], f64)>, f64, usize, f64, f64, SteppedTrajPreview) {
    let mut b = Battle::from_fixture(dex, "1,2,3,4", sets_a, sets_b).unwrap();
    b.set_log_enabled(true);
    let own_sets = if side == 0 { sets_a } else { sets_b };
    let (mut lp, mut obs) =
        LocalPlayer::new_with_own_sets(&b, dex, side, own_sets.to_vec()).unwrap();
    lp.observe(&b, dex, &mut obs).unwrap();
    let mut belief = Belief::new(dex, pool, &obs);
    belief.sync(dex, &obs);
    let mut rng = SplitMix64::new(seed);
    let pos = lp.position(dex, &mut belief, &obs, &mut rng).unwrap();
    let mut base = pos.model().clone();
    let own_legal = base.legal_choices(dex, side);

    let t0 = Instant::now();
    let mut stepped = SteppedTrajPreview::try_new(
        &base,
        dex,
        &belief,
        &obs,
        side,
        &own_legal,
        None,
        cfg.clone(),
        &mut rng,
    )
    .expect("compact preview supported");
    for _ in 0..cfg.total_iters {
        stepped.step_one(&mut rng);
    }
    let (entries, val_mine, _, _, _) =
        stepped.readout_with_policy_mode(stepped.iters_done(), cfg.policy_mode);
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let val0 = if side == 0 { val_mine } else { 1.0 - val_mine };
    let mut dist = Vec::new();
    for (ch, p) in entries {
        if let SearchChoice::Team(tr) = ch {
            dist.push((canonical_triple(tr), p));
        }
    }
    dist.sort_by(|x, y| y.1.total_cmp(&x.1));
    let s1_ms = stepped.stage1_ms();
    let games = stepped.games_completed();
    (dist, val0, games, s1_ms, total_ms, stepped)
}

// =========================================================================
// Mode 1: Lever A1 Rao-Blackwellized Early-Exit Bias, Variance & Speed Audit
// =========================================================================

fn run_a1_rb(dex: &Dex, team_map: &HashMap<String, Vec<PokemonSet>>, threads: usize) {
    let t0 = Instant::now();
    struct CellCase {
        tag: &'static str,
        team_a: &'static str,
        team_b: &'static str,
        p0: [u8; 3],
        p1: [u8; 3],
        hazard_note: &'static str,
    }
    let cases = vec![
        CellCase {
            tag: "m1_killer_raik_cloy_skarm",
            team_a: "pb84-cln-skarm50",
            team_b: "mjj-2014-thdrcomb",
            p0: [2, 4, 6],
            p1: [3, 1, 4],
            hazard_note: "Spikes+Whirlwind+RestTalk+Explosion",
        },
        CellCase {
            tag: "m1_base_lax_champ_cloy",
            team_a: "pb84-cln-skarm50",
            team_b: "mjj-2014-thdrcomb",
            p0: [1, 3, 4],
            p1: [3, 1, 4],
            hazard_note: "RestTalk+Explosion",
        },
        CellCase {
            tag: "m1_mjj_venu_ttar_lax",
            team_a: "pb84-cln-skarm50",
            team_b: "mjj-2014-thdrcomb",
            p0: [1, 3, 4],
            p1: [3, 2, 6],
            hazard_note: "Sleep+LeechSeed+SelfDestruct",
        },
        CellCase {
            tag: "m1_mjj_steelix_zap_cloy",
            team_a: "pb84-cln-skarm50",
            team_b: "mjj-2014-thdrcomb",
            p0: [2, 3, 6],
            p1: [4, 1, 5],
            hazard_note: "Spikes+Roar+Whirlwind+RestTalk",
        },
        CellCase {
            tag: "m2_elec_wak_lax_vs_ttar",
            team_a: "sample-07",
            team_b: "mjj-2014-thdrcomb",
            p0: [1, 4, 5],
            p1: [2, 5, 6],
            hazard_note: "Explosion+SelfDestruct",
        },
        CellCase {
            tag: "m2_bg_tauros_vs_steelix",
            team_a: "sample-07",
            team_b: "mjj-2014-thdrcomb",
            p0: [2, 3, 5],
            p1: [4, 2, 5],
            hazard_note: "BerserkGene+Explosion",
        },
        CellCase {
            tag: "m3_hera_endrev_gar_dbond_vs_cloy",
            team_a: "pb63-m-gar55",
            team_b: "pb-6",
            p0: [6, 2, 5],
            p1: [5, 1, 3],
            hazard_note: "Endure+Reversal+FocusBand+DestinyBond",
        },
        CellCase {
            tag: "m3_hera_endrev_gar_dbond_vs_champ",
            team_a: "pb63-m-gar55",
            team_b: "pb-6",
            p0: [6, 2, 5],
            p1: [2, 3, 5],
            hazard_note: "Endure+Reversal+FocusBand+DestinyBond",
        },
        CellCase {
            tag: "m3_wak_lax_zap_vs_perish_gar",
            team_a: "pb63-m-gar55",
            team_b: "pb-6",
            p0: [4, 1, 6],
            p1: [5, 1, 6],
            hazard_note: "PerishSong+DestinyBond+MeanLook",
        },
        CellCase {
            tag: "m3_hera_vs_perish_bp_umb",
            team_a: "pb63-m-gar55",
            team_b: "pb-6",
            p0: [6, 2, 5],
            p1: [6, 3, 4],
            hazard_note: "EndureRev+FocusBand+Perish+DBond+BP",
        },
    ];

    let modes: Vec<(&str, RbExitMode)> = vec![
        ("terminal", RbExitMode::Terminal),
        ("uncond_le3", RbExitMode::UncondBlindspot { max_alive: 3 }),
        ("cond_le3", RbExitMode::CondBlindspot { max_alive: 3 }),
        ("uncond_le2", RbExitMode::UncondBlindspot { max_alive: 2 }),
        ("cond_le2", RbExitMode::CondBlindspot { max_alive: 2 }),
        (
            "vroot64_le3",
            RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            },
        ),
        (
            "hybrid64_le3",
            RbExitMode::HybridCond {
                max_alive: 3,
                eval_iters: 64,
            },
        ),
        (
            "vroot64_le2",
            RbExitMode::SearchVroot {
                max_alive: 2,
                eval_iters: 64,
            },
        ),
    ];

    let n_games = 128usize;
    let ipt = 32u32;
    let mut cell_reports = Vec::new();
    let mut mode_agg_bias: HashMap<String, Vec<f64>> = HashMap::new();
    let mut mode_agg_var_ratio: HashMap<String, Vec<f64>> = HashMap::new();
    let mut mode_agg_iters: HashMap<String, Vec<f64>> = HashMap::new();
    let mut mode_agg_ms: HashMap<String, Vec<f64>> = HashMap::new();
    let mut mode_agg_rb_rate: HashMap<String, Vec<f64>> = HashMap::new();

    for (ci, cc) in cases.iter().enumerate() {
        let b = Battle::from_fixture(dex, "1,2,3,4", &team_map[cc.team_a], &team_map[cc.team_b])
            .unwrap();
        let roster = CompactPreviewRoster::from_battle(&b, dex, None);
        let pcache = PreviewTempoCache::build_for_preview_roster(&roster, [6, 6]);
        let base_s = 0x2026_A100_0000_0001u64 ^ ((ci as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));

        let mut term_mean = 0.5;
        let mut term_var = 0.25;
        let mut term_iters = 1.0;
        let mut m_map = serde_json::Map::new();

        for &(mname, rbm) in &modes {
            let cur = AtomicUsize::new(0);
            let res: Vec<(f64, u32, bool)> = std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for _ in 0..threads.min(n_games).max(1) {
                    let (cur_r, ros_r, pc_r) = (&cur, &roster, &pcache);
                    hs.push(sc.spawn(move || {
                        let mut scratch = TrajTurnScratch::new();
                        let mut out = Vec::new();
                        loop {
                            let k = cur_r.fetch_add(1, Ordering::Relaxed);
                            if k >= n_games {
                                break;
                            }
                            let gs = base_s
                                ^ ((k as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                            let (v, _t, it, rb) = play_compact_traj_with_rb(
                                ros_r,
                                pc_r,
                                [cc.p0, cc.p1],
                                ipt,
                                rbm,
                                TrajCrnMode::PairedCrnStratified,
                                gs,
                                k as u32,
                                180,
                                &mut scratch,
                            );
                            out.push((v, it, rb));
                        }
                        out
                    }));
                }
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });

            // Clean 1-thread timing on 16 games (zero thread contention).
            let mut scratch1 = TrajTurnScratch::new();
            let t_1t = Instant::now();
            let n_time = 16usize;
            for k in 0..n_time {
                let gs = base_s ^ ((k as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let _ = play_compact_traj_with_rb(
                    &roster,
                    &pcache,
                    [cc.p0, cc.p1],
                    ipt,
                    rbm,
                    TrajCrnMode::PairedCrnStratified,
                    gs,
                    k as u32,
                    180,
                    &mut scratch1,
                );
            }
            let ms_1t = t_1t.elapsed().as_secs_f64() * 1000.0 / (n_time as f64);

            let vals: Vec<f64> = res.iter().map(|x| x.0).collect();
            let (m, sd) = mean_sd(&vals);
            let var = sd * sd;
            let avg_it = res.iter().map(|x| f64::from(x.1)).sum::<f64>() / (n_games as f64);
            let rb_rate = res.iter().filter(|x| x.2).count() as f64 / (n_games as f64);

            if mname == "terminal" {
                term_mean = m;
                term_var = var.max(1e-6);
                term_iters = avg_it.max(1.0);
            }
            let bias = m - term_mean;
            let vratio = var / term_var;
            let eff_gain = (term_var * term_iters) / (var.max(1e-6) * avg_it.max(1.0));
            mode_agg_bias.entry(mname.into()).or_default().push(bias);
            mode_agg_var_ratio.entry(mname.into()).or_default().push(vratio);
            mode_agg_iters.entry(mname.into()).or_default().push(avg_it);
            mode_agg_ms.entry(mname.into()).or_default().push(ms_1t);
            mode_agg_rb_rate.entry(mname.into()).or_default().push(rb_rate);

            m_map.insert(
                mname.into(),
                json!({
                    "mean": r4(m),
                    "bias": r4(bias),
                    "var": r4(var),
                    "var_ratio": r4(vratio),
                    "iters": r2(avg_it),
                    "ms1t": r2(ms_1t),
                    "rb_rate": r4(rb_rate),
                    "eff_gain": r2(eff_gain),
                }),
            );
        }

        cell_reports.push(json!({
            "tag": cc.tag,
            "hazard": cc.hazard_note,
            "modes": m_map,
        }));
    }

    let mut summary = Vec::new();
    for &(mname, _) in &modes {
        let bs = &mode_agg_bias[mname];
        let vrs = &mode_agg_var_ratio[mname];
        let its = &mode_agg_iters[mname];
        let mss = &mode_agg_ms[mname];
        let rbs = &mode_agg_rb_rate[mname];
        let mae = bs.iter().map(|x| x.abs()).sum::<f64>() / (bs.len() as f64);
        let max_abs_bias = bs.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let signed_bias = bs.iter().sum::<f64>() / (bs.len() as f64);
        let avg_vr = vrs.iter().sum::<f64>() / (vrs.len() as f64);
        let avg_it = its.iter().sum::<f64>() / (its.len() as f64);
        let avg_ms = mss.iter().sum::<f64>() / (mss.len() as f64);
        let avg_rb = rbs.iter().sum::<f64>() / (rbs.len() as f64);
        summary.push(json!({
            "mode": mname,
            "signed_bias": r4(signed_bias),
            "mae_bias": r4(mae),
            "max_abs_bias": r4(max_abs_bias),
            "avg_var_ratio": r4(avg_vr),
            "avg_iters": r2(avg_it),
            "avg_ms1t": r2(avg_ms),
            "avg_rb_rate": r4(avg_rb),
        }));
    }

    let full = json!({
        "mode": "a1_rb",
        "ipt": ipt,
        "games_per_cell": n_games,
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "summary": summary,
        "cells": cell_reports,
    });
    std::fs::write("/tmp/pillar_a_a1_rb.json", serde_json::to_string_pretty(&full).unwrap()).unwrap();
    let compact = json!({"mode":"a1_rb","wall_s":r2(t0.elapsed().as_secs_f64()),"summary":summary});
    println!("{}", compact);
}

// =========================================================================
// Mode 2: Levers A1–A4 Systematic Ablation on m1, m2, m3 (8 seeds each)
// =========================================================================

struct MatchupDef {
    code: &'static str,
    team_a: &'static str,
    team_b: &'static str,
    true_nash_a: f64,
    killers: Vec<(&'static str, usize, [u8; 3])>,
}

fn pilot_matchups() -> Vec<MatchupDef> {
    vec![
        MatchupDef {
            code: "m1",
            team_a: "pb84-cln-skarm50",
            team_b: "mjj-2014-thdrcomb",
            true_nash_a: 0.3385,
            killers: vec![
                ("pb84_RaikCloySkarm", 0, [2, 4, 6]),
                ("mjj_VenuTtarLax", 1, [3, 2, 6]),
                ("mjj_SteelixZapCloy", 1, [4, 1, 5]),
            ],
        },
        MatchupDef {
            code: "m2",
            team_a: "sample-07",
            team_b: "mjj-2014-thdrcomb",
            true_nash_a: 0.4008,
            killers: vec![("mjj_SteelixTtarCloy", 1, [4, 2, 5])],
        },
        MatchupDef {
            code: "m3",
            team_a: "pb63-m-gar55",
            team_b: "pb-6",
            true_nash_a: 0.3859,
            killers: vec![("pb63_ZapGarHera", 0, [6, 2, 5])],
        },
    ]
}

fn run_a234_ablate(dex: &Dex, team_map: &HashMap<String, Vec<PokemonSet>>, threads: usize) {
    let t0 = Instant::now();
    let matchups = pilot_matchups();
    let n_seeds = 8usize;

    // 0. Build held-out 16-cell ground truth (fg64 terminal, 64 games/cell) per matchup to score
    //    out-of-sample cell-swap residual RMSE of each ridge basis!
    let mut heldout_cells: Vec<Vec<(usize, usize, f64)>> = Vec::new();
    for (mi, ms) in matchups.iter().enumerate() {
        let sets_a = &team_map[ms.team_a];
        let sets_b = &team_map[ms.team_b];
        let mut b = Battle::from_fixture(dex, "1,2,3,4", sets_a, sets_b).unwrap();
        let c0 = canonical_preview_triples(dex, &mut b, 0);
        let c1 = canonical_preview_triples(dex, &mut b, 1);
        let roster = CompactPreviewRoster::from_battle(&b, dex, None);
        let pcache = PreviewTempoCache::build_for_preview_roster(&roster, [6, 6]);
        let mut hrng = SplitMix64::new(0x2026_1126_9000_0001u64 ^ ((mi as u64) << 16));
        let mut pairs: Vec<(usize, usize)> = Vec::new();
        // Include killer swaps + diverse unseen triples.
        for &(_, side, tr) in &ms.killers {
            let ct = canonical_triple(tr);
            if side == 0 {
                if let Some(i) = c0.iter().position(|&x| x == ct) {
                    pairs.push((i, hrng.below(c1.len())));
                    pairs.push((i, hrng.below(c1.len())));
                }
            } else if let Some(j) = c1.iter().position(|&x| x == ct) {
                pairs.push((hrng.below(c0.len()), j));
                pairs.push((hrng.below(c0.len()), j));
            }
        }
        while pairs.len() < 16 {
            let p = (hrng.below(c0.len()), hrng.below(c1.len()));
            if !pairs.contains(&p) {
                pairs.push(p);
            }
        }
        let cur = AtomicUsize::new(0);
        let evals: Vec<(usize, usize, f64)> = std::thread::scope(|sc| {
            let mut hs = Vec::new();
            for _ in 0..threads.min(pairs.len()).max(1) {
                let (cur_r, prs_r, ros_r, pc_r, c0_r, c1_r) =
                    (&cur, &pairs, &roster, &pcache, &c0, &c1);
                hs.push(sc.spawn(move || {
                    let mut scratch = TrajTurnScratch::new();
                    let mut out = Vec::new();
                    loop {
                        let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                        if idx >= prs_r.len() {
                            break;
                        }
                        let (i, j) = prs_r[idx];
                        let mut acc = 0.0;
                        let ng = 48usize;
                        for k in 0..ng {
                            let gs = 0x7777_0001u64
                                ^ ((mi as u64) << 32)
                                ^ ((idx as u64) << 12)
                                ^ ((k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                            let (v, _, _, _) = play_compact_traj_with_rb(
                                ros_r,
                                pc_r,
                                [c0_r[i], c1_r[j]],
                                64,
                                RbExitMode::CondBlindspot { max_alive: 2 },
                                TrajCrnMode::PairedCrnStratified,
                                gs,
                                k as u32,
                                180,
                                &mut scratch,
                            );
                            acc += v;
                        }
                        out.push((i, j, acc / (ng as f64)));
                    }
                    out
                }));
            }
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        heldout_cells.push(evals);
    }

    // Build candidate ablation grid anchored around a clean base (`180k`, `ipt=32`, `CondLe3`, `Interact31`, `PairedCrnStratified`, `GLS`, `kappa=6`).
    let base_cfg = TrajPreviewConfig {
        total_iters: 180_000,
        iters_per_turn: 32,
        rb_exit: RbExitMode::CondBlindspot { max_alive: 3 },
        ridge_basis: TrajRidgeBasis::Interact31,
        ridge_lambda_main: 8.0,
        ridge_lambda_inter: 16.0,
        ridge_gls: true,
        ridge_shared_multiworld: true,
        kappa: 6.0,
        crn_mode: TrajCrnMode::PairedCrnStratified,
        explore_frac: 0.30,
        do_frac: 0.45,
        do_challengers: 4,
        do_ucb_c: 0.10,
        policy_mode: TrajPolicyMode::BayesianBootstrapRm {
            draws: 16,
            dust_ratio: 0.08,
        },
        ..Default::default()
    };

    let mut variants: Vec<(&str, TrajPreviewConfig)> = Vec::new();
    // Base reference
    variants.push(("base_180k_cond3_int31_crnS_k6", base_cfg.clone()));

    // Lever A1 inside full Traj-DO
    variants.push((
        "a1_terminal",
        TrajPreviewConfig {
            rb_exit: RbExitMode::Terminal,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a1_uncond3",
        TrajPreviewConfig {
            rb_exit: RbExitMode::UncondBlindspot { max_alive: 3 },
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a1_cond2",
        TrajPreviewConfig {
            rb_exit: RbExitMode::CondBlindspot { max_alive: 2 },
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a1_hybrid64_le3",
        TrajPreviewConfig {
            rb_exit: RbExitMode::HybridCond {
                max_alive: 3,
                eval_iters: 64,
            },
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a1_vroot64_le3",
        TrajPreviewConfig {
            rb_exit: RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            },
            ..base_cfg.clone()
        },
    ));

    // Lever A2: CRN & Stratification
    variants.push((
        "a2_indep_nocrn",
        TrajPreviewConfig {
            crn_mode: TrajCrnMode::Independent,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a2_paired_crn",
        TrajPreviewConfig {
            crn_mode: TrajCrnMode::PairedCrn,
            ..base_cfg.clone()
        },
    ));

    // Lever A3a: Ridge Basis & GLS
    variants.push((
        "a3_add25_raw",
        TrajPreviewConfig {
            ridge_basis: TrajRidgeBasis::Additive25,
            ridge_gls: false,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_add25_gls",
        TrajPreviewConfig {
            ridge_basis: TrajRidgeBasis::Additive25,
            ridge_gls: true,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_int33_gls",
        TrajPreviewConfig {
            ridge_basis: TrajRidgeBasis::Interact33,
            ridge_gls: true,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_pair55_gls",
        TrajPreviewConfig {
            ridge_basis: TrajRidgeBasis::Pair55,
            ridge_lambda_inter: 24.0,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_moncross61_gls",
        TrajPreviewConfig {
            ridge_basis: TrajRidgeBasis::MonCross61,
            ridge_lambda_inter: 32.0,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_int67_gls",
        TrajPreviewConfig {
            ridge_basis: TrajRidgeBasis::Interact67,
            ridge_lambda_inter: 32.0,
            ..base_cfg.clone()
        },
    ));

    // Lever A3b: ipt, kappa, budget & split
    variants.push((
        "a3_ipt24",
        TrajPreviewConfig {
            iters_per_turn: 24,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_ipt48",
        TrajPreviewConfig {
            iters_per_turn: 48,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_ipt64",
        TrajPreviewConfig {
            iters_per_turn: 64,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_k4",
        TrajPreviewConfig {
            kappa: 4.0,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_k10",
        TrajPreviewConfig {
            kappa: 10.0,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_budget140k_wasm",
        TrajPreviewConfig {
            total_iters: 140_000,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_budget240k_parity",
        TrajPreviewConfig {
            total_iters: 240_000,
            ..base_cfg.clone()
        },
    ));
    variants.push((
        "a3_split_25_40_35",
        TrajPreviewConfig {
            explore_frac: 0.25,
            do_frac: 0.40,
            ..base_cfg.clone()
        },
    ));

    let mut variant_reports = Vec::new();
    let mut a4_report = Value::Null;

    for (vname, vcfg) in &variants {
        // Run all 3 matchups x 8 seeds in parallel for accuracy/SD/RMSE, PLUS 1 clean serial seed per matchup for true 1-thread latency!
        let mut jobs: Vec<(usize, usize)> = Vec::new();
        for mi in 0..matchups.len() {
            for s in 0..n_seeds {
                jobs.push((mi, s));
            }
        }
        let cur = AtomicUsize::new(0);
        let par_res: Vec<(usize, usize, f64, usize, f64, Vec<([u8; 3], f64)>, Option<SteppedTrajPreview>)> =
            std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for _ in 0..threads.min(jobs.len()).max(1) {
                    let (cur_r, jobs_r, ms_r, ho_r) = (&cur, &jobs, &matchups, &heldout_cells);
                    let keep_stepped = *vname == "base_180k_cond3_int31_crnS_k6";
                    hs.push(sc.spawn(move || {
                        let mut out = Vec::new();
                        loop {
                            let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                            if idx >= jobs_r.len() {
                                break;
                            }
                            let (mi, s) = jobs_r[idx];
                            let ms = &ms_r[mi];
                            let sets_a = &team_map[ms.team_a];
                            let sets_b = &team_map[ms.team_b];
                            let pool = disclosed_pool(sets_b);
                            let seed = 0x2026_1126_A234_0001u64
                                ^ ((mi as u64) << 24)
                                ^ ((s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                            let (dist, val0, games, _s1ms, _tms, stepped) =
                                run_traj_once(dex, sets_a, sets_b, &pool, 0, vcfg, seed);
                            // Compute held-out cell RMSE from posterior matrix world 0.
                            let mats = stepped.posterior_matrices();
                            let m_opp = stepped.opp_triples_w0().len();
                            let mut sq_err = 0.0;
                            for &(hi, hj, htrue) in &ho_r[mi] {
                                let pred = mats[0][hi * m_opp + hj];
                                sq_err += (pred - htrue) * (pred - htrue);
                            }
                            let rmse = (sq_err / (ho_r[mi].len().max(1) as f64)).sqrt();
                            out.push((
                                mi,
                                s,
                                val0,
                                games,
                                rmse,
                                dist,
                                if keep_stepped { Some(stepped) } else { None },
                            ));
                        }
                        out
                    }));
                }
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });

        // Clean 1-thread latency timing (2 seeds x 3 matchups = 6 uncontended timings).
        let mut ms_1t_all = Vec::new();
        for (mi, ms) in matchups.iter().enumerate() {
            let sets_a = &team_map[ms.team_a];
            let sets_b = &team_map[ms.team_b];
            let pool = disclosed_pool(sets_b);
            for s in 0..2usize {
                let seed = 0x2026_1126_7130_0001u64
                    ^ ((mi as u64) << 24)
                    ^ ((s as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let (_, _, _, _, tms, _) =
                    run_traj_once(dex, sets_a, sets_b, &pool, 0, vcfg, seed);
                ms_1t_all.push(tms);
            }
        }
        let (med_ms, p95_ms) = median_p95(&ms_1t_all);

        let mut maes = Vec::new();
        let mut sds = Vec::new();
        let mut tvs = Vec::new();
        let mut rmses = Vec::new();
        let mut games_all = Vec::new();
        let mut per_m = serde_json::Map::new();

        for (mi, ms) in matchups.iter().enumerate() {
            let mut vals = Vec::new();
            let mut pols = Vec::new();
            let mut r_list = Vec::new();
            for row in &par_res {
                if row.0 == mi {
                    vals.push(row.2);
                    games_all.push(row.3 as f64);
                    r_list.push(row.4);
                    pols.push(row.5.clone());
                }
            }
            let (vm, vsd) = mean_sd(&vals);
            let err = (vm - ms.true_nash_a).abs();
            let tv = mean_pairwise_tv(&pols);
            let avg_rmse = r_list.iter().sum::<f64>() / (r_list.len().max(1) as f64);
            maes.push(err);
            sds.push(vsd);
            tvs.push(tv);
            rmses.push(avg_rmse);
            per_m.insert(
                ms.code.into(),
                json!({
                    "val_mean": r4(vm),
                    "true": ms.true_nash_a,
                    "abs_err": r4(err),
                    "seed_sd": r4(vsd),
                    "policy_tv": r4(tv),
                    "heldout_rmse": r4(avg_rmse),
                }),
            );
        }

        let avg_mae = maes.iter().sum::<f64>() / (maes.len() as f64);
        let avg_sd = sds.iter().sum::<f64>() / (sds.len() as f64);
        let avg_tv = tvs.iter().sum::<f64>() / (tvs.len() as f64);
        let avg_rmse = rmses.iter().sum::<f64>() / (rmses.len() as f64);
        let avg_games = games_all.iter().sum::<f64>() / (games_all.len().max(1) as f64);

        variant_reports.push(json!({
            "name": vname,
            "mae_3m": r4(avg_mae),
            "sd_8seed": r4(avg_sd),
            "policy_tv": r4(avg_tv),
            "heldout_rmse": r4(avg_rmse),
            "avg_games": r2(avg_games),
            "med_ms1t": r2(med_ms),
            "p95_ms1t": r2(p95_ms),
            "matchups": per_m,
        }));

        // Lever A4: evaluate all policy extraction modes on identical solved base_180k trajectories!
        if *vname == "base_180k_cond3_int31_crnS_k6" {
            let a4_modes: Vec<(&str, TrajPolicyMode)> = vec![
                (
                    "lcb_raw_n",
                    TrajPolicyMode::LcbGatedMixed {
                        min_mix_margin: 0.015,
                        dust_ratio: 0.08,
                        lcb_z: 1.0,
                        use_neff: false,
                    },
                ),
                (
                    "lcb_neff",
                    TrajPolicyMode::LcbGatedMixed {
                        min_mix_margin: 0.015,
                        dust_ratio: 0.08,
                        lcb_z: 1.0,
                        use_neff: true,
                    },
                ),
                ("mixed_nash_08", TrajPolicyMode::MixedNash { dust_ratio: 0.08 }),
                (
                    "reg_rm_015",
                    TrajPolicyMode::RegRm {
                        l2_reg: 0.015,
                        dust_ratio: 0.08,
                    },
                ),
                (
                    "reg_rm_030",
                    TrajPolicyMode::RegRm {
                        l2_reg: 0.030,
                        dust_ratio: 0.08,
                    },
                ),
                (
                    "boot_rm_16",
                    TrajPolicyMode::BayesianBootstrapRm {
                        draws: 16,
                        dust_ratio: 0.08,
                    },
                ),
            ];
            let mut a4_rows = Vec::new();
            for &(aname, amode) in &a4_modes {
                let mut mix_cnt = 0usize;
                let mut tot_cnt = 0usize;
                let mut supp_sum = 0.0;
                let mut tv_sum = 0.0;
                for mi in 0..matchups.len() {
                    let mut pols = Vec::new();
                    for row in &par_res {
                        if row.0 == mi {
                            if let Some(ref st) = row.6 {
                                let (entries, _, _, _, um) =
                                    st.readout_with_policy_mode(st.iters_done(), amode);
                                if um {
                                    mix_cnt += 1;
                                }
                                tot_cnt += 1;
                                supp_sum += entries.len() as f64;
                                let d: Vec<([u8; 3], f64)> = entries
                                    .into_iter()
                                    .filter_map(|(c, p)| match c {
                                        SearchChoice::Team(t) => Some((canonical_triple(t), p)),
                                        _ => None,
                                    })
                                    .collect();
                                pols.push(d);
                            }
                        }
                    }
                    tv_sum += mean_pairwise_tv(&pols);
                }
                a4_rows.push(json!({
                    "mode": aname,
                    "mix_rate": r4(mix_cnt as f64 / (tot_cnt.max(1) as f64)),
                    "avg_support_size": r2(supp_sum / (tot_cnt.max(1) as f64)),
                    "policy_tv_8seed": r4(tv_sum / (matchups.len() as f64)),
                }));
            }
            a4_report = json!(a4_rows);
        }
    }

    let compact_rows: Vec<Value> = variant_reports
        .iter()
        .map(|v| {
            json!({
                "name": v["name"],
                "mae": v["mae_3m"],
                "sd": v["sd_8seed"],
                "tv": v["policy_tv"],
                "rmse": v["heldout_rmse"],
                "games": v["avg_games"],
                "ms_med": v["med_ms1t"],
                "ms_p95": v["p95_ms1t"],
            })
        })
        .collect();

    let full = json!({
        "mode": "a234_ablate",
        "n_seeds": n_seeds,
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "variants": variant_reports,
        "lever_a4_readout": a4_report,
    });
    std::fs::write(
        "/tmp/pillar_a_a234_ablate.json",
        serde_json::to_string_pretty(&full).unwrap(),
    )
    .unwrap();
    let compact = json!({
        "mode": "a234_ablate",
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "variants": compact_rows,
        "a4": a4_report,
    });
    println!("{}", compact);
}

// =========================================================================
// Mode 3: Suite 1 — Real BlindAgent 9k H2H & 48-arm Exploitability on m1, m2, m3
// =========================================================================

fn make_suite_cfg(tag: &str) -> TrajPreviewConfig {
    let mut cfg = TrajPreviewConfig {
        total_iters: 180_000,
        iters_per_turn: 32,
        rb_exit: RbExitMode::CondBlindspot { max_alive: 3 },
        ridge_basis: TrajRidgeBasis::Interact31,
        ridge_lambda_main: 8.0,
        ridge_lambda_inter: 16.0,
        ridge_gls: true,
        ridge_shared_multiworld: true,
        kappa: 6.0,
        crn_mode: TrajCrnMode::PairedCrnStratified,
        explore_frac: 0.30,
        do_frac: 0.45,
        do_challengers: 4,
        do_ucb_c: 0.10,
        policy_mode: TrajPolicyMode::BayesianBootstrapRm {
            draws: 16,
            dust_ratio: 0.08,
        },
        ..Default::default()
    };
    match tag {
        "wasm150k" => {
            cfg.total_iters = 150_000;
        }
        "wasm180k" => {
            cfg.total_iters = 180_000;
        }
        "parity240k" => {
            cfg.total_iters = 240_000;
        }
        "mixed_nash" => {
            cfg.policy_mode = TrajPolicyMode::MixedNash { dust_ratio: 0.08 };
        }
        "lcb_neff" => {
            cfg.policy_mode = TrajPolicyMode::LcbGatedMixed {
                min_mix_margin: 0.015,
                dust_ratio: 0.08,
                lcb_z: 1.0,
                use_neff: true,
            };
        }
        "reg_rm" => {
            cfg.policy_mode = TrajPolicyMode::RegRm {
                l2_reg: 0.02,
                dust_ratio: 0.08,
            };
        }
        "cond2_240k" => {
            cfg.total_iters = 240_000;
            cfg.rb_exit = RbExitMode::CondBlindspot { max_alive: 2 };
        }
        "hybrid64_240k" => {
            cfg.total_iters = 240_000;
            cfg.rb_exit = RbExitMode::HybridCond {
                max_alive: 3,
                eval_iters: 64,
            };
        }
        "k10_240k" => {
            cfg.total_iters = 240_000;
            cfg.kappa = 10.0;
        }
        "ipt48_240k" => {
            cfg.total_iters = 240_000;
            cfg.iters_per_turn = 48;
        }
        "vroot_180k_k16" => {
            cfg.total_iters = 180_000;
            cfg.rb_exit = RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            };
            cfg.kappa = 16.0;
            cfg.ridge_lambda_main = 12.0;
            cfg.ridge_lambda_inter = 20.0;
            cfg.ridge_basis = TrajRidgeBasis::Interact33;
        }
        "vroot_240k_k16" => {
            cfg.total_iters = 240_000;
            cfg.rb_exit = RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            };
            cfg.kappa = 16.0;
            cfg.ridge_lambda_main = 12.0;
            cfg.ridge_lambda_inter = 20.0;
            cfg.ridge_basis = TrajRidgeBasis::Interact33;
        }
        "vroot_240k_k24" => {
            cfg.total_iters = 240_000;
            cfg.rb_exit = RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            };
            cfg.kappa = 24.0;
            cfg.ridge_lambda_main = 14.0;
            cfg.ridge_lambda_inter = 24.0;
            cfg.ridge_basis = TrajRidgeBasis::Interact33;
        }
        "vroot_180k_k16_mixed" => {
            cfg.total_iters = 180_000;
            cfg.rb_exit = RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            };
            cfg.kappa = 16.0;
            cfg.ridge_lambda_main = 12.0;
            cfg.ridge_lambda_inter = 20.0;
            cfg.ridge_basis = TrajRidgeBasis::Interact33;
            cfg.policy_mode = TrajPolicyMode::MixedNash { dust_ratio: 0.08 };
        }
        "vroot_180k_k16_lcb" => {
            cfg.total_iters = 180_000;
            cfg.rb_exit = RbExitMode::SearchVroot {
                max_alive: 3,
                eval_iters: 64,
            };
            cfg.kappa = 16.0;
            cfg.ridge_lambda_main = 12.0;
            cfg.ridge_lambda_inter = 20.0;
            cfg.ridge_basis = TrajRidgeBasis::Interact33;
            cfg.policy_mode = TrajPolicyMode::LcbGatedMixed {
                min_mix_margin: 0.015,
                dust_ratio: 0.08,
                lcb_z: 1.0,
                use_neff: true,
            };
        }
        "term_240k_k16" => {
            cfg.total_iters = 240_000;
            cfg.rb_exit = RbExitMode::Terminal;
            cfg.kappa = 16.0;
            cfg.ridge_lambda_main = 12.0;
            cfg.ridge_lambda_inter = 20.0;
            cfg.ridge_basis = TrajRidgeBasis::Interact33;
        }
        "final_default" => {
            return TrajPreviewConfig::default();
        }
        "final_wasm180k" => {
            return TrajPreviewConfig {
                total_iters: 180_000,
                ..TrajPreviewConfig::default()
            };
        }
        "i39_k6_boot20" => {
            return TrajPreviewConfig {
                ridge_basis: TrajRidgeBasis::Interact39,
                ridge_lambda_main: 8.0,
                ridge_lambda_inter: 14.0,
                kappa: 6.0,
                policy_mode: TrajPolicyMode::BayesianBootstrapRm {
                    draws: 16,
                    dust_ratio: 0.20,
                },
                ..TrajPreviewConfig::default()
            };
        }
        "i39_k6_reg015" => {
            return TrajPreviewConfig {
                ridge_basis: TrajRidgeBasis::Interact39,
                ridge_lambda_main: 8.0,
                ridge_lambda_inter: 14.0,
                kappa: 6.0,
                policy_mode: TrajPolicyMode::RegRm {
                    l2_reg: 0.015,
                    dust_ratio: 0.10,
                },
                ..TrajPreviewConfig::default()
            };
        }
        "i39_k6_t1_64_boot20" => {
            return TrajPreviewConfig {
                turn1_iters: 64,
                ridge_basis: TrajRidgeBasis::Interact39,
                ridge_lambda_main: 8.0,
                ridge_lambda_inter: 14.0,
                kappa: 6.0,
                policy_mode: TrajPolicyMode::BayesianBootstrapRm {
                    draws: 16,
                    dust_ratio: 0.20,
                },
                ..TrajPreviewConfig::default()
            };
        }
        "i39_k6_t1_64_reg015" => {
            return TrajPreviewConfig {
                turn1_iters: 64,
                ridge_basis: TrajRidgeBasis::Interact39,
                ridge_lambda_main: 8.0,
                ridge_lambda_inter: 14.0,
                kappa: 6.0,
                policy_mode: TrajPolicyMode::RegRm {
                    l2_reg: 0.015,
                    dust_ratio: 0.10,
                },
                ..TrajPreviewConfig::default()
            };
        }
        "i39_k4_t1_64_reg010" => {
            return TrajPreviewConfig {
                turn1_iters: 64,
                ridge_basis: TrajRidgeBasis::Interact39,
                ridge_lambda_main: 6.0,
                ridge_lambda_inter: 12.0,
                kappa: 4.0,
                policy_mode: TrajPolicyMode::RegRm {
                    l2_reg: 0.010,
                    dust_ratio: 0.10,
                },
                ..TrajPreviewConfig::default()
            };
        }
        "i39_k6_t1_64_mixed10" => {
            return TrajPreviewConfig {
                turn1_iters: 64,
                ridge_basis: TrajRidgeBasis::Interact39,
                ridge_lambda_main: 8.0,
                ridge_lambda_inter: 14.0,
                kappa: 6.0,
                policy_mode: TrajPolicyMode::MixedNash { dust_ratio: 0.10 },
                ..TrajPreviewConfig::default()
            };
        }
        "hyb_20_a075" => {
            return TrajPreviewConfig {
                explore_frac: 0.25,
                do_frac: 0.45,
                t1_polish_frac: 0.20,
                t1_polish_iters: 320,
                t1_polish_alpha: 0.75,
                ..TrajPreviewConfig::default()
            };
        }
        "hyb_20_a100" => {
            return TrajPreviewConfig {
                explore_frac: 0.25,
                do_frac: 0.45,
                t1_polish_frac: 0.20,
                t1_polish_iters: 320,
                t1_polish_alpha: 1.0,
                ..TrajPreviewConfig::default()
            };
        }
        "hyb_25_a100_d20" => {
            return TrajPreviewConfig {
                explore_frac: 0.25,
                do_frac: 0.40,
                t1_polish_frac: 0.25,
                t1_polish_iters: 384,
                t1_polish_alpha: 1.0,
                policy_mode: TrajPolicyMode::BayesianBootstrapRm {
                    draws: 16,
                    dust_ratio: 0.20,
                },
                ..TrajPreviewConfig::default()
            };
        }
        "hyb_20_a100_reg015" => {
            return TrajPreviewConfig {
                explore_frac: 0.25,
                do_frac: 0.45,
                t1_polish_frac: 0.20,
                t1_polish_iters: 320,
                t1_polish_alpha: 1.0,
                policy_mode: TrajPolicyMode::RegRm {
                    l2_reg: 0.015,
                    dust_ratio: 0.12,
                },
                ..TrajPreviewConfig::default()
            };
        }
        _ => {}
    }
    cfg
}

fn run_a3_focus(dex: &Dex, team_map: &HashMap<String, Vec<PokemonSet>>, threads: usize) {
    let t0 = Instant::now();
    let matchups = pilot_matchups();
    let n_seeds = 8usize;

    let vroot_base = TrajPreviewConfig {
        total_iters: 180_000,
        iters_per_turn: 32,
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
        do_challengers: 4,
        do_ucb_c: 0.10,
        policy_mode: TrajPolicyMode::BayesianBootstrapRm {
            draws: 16,
            dust_ratio: 0.08,
        },
        ..Default::default()
    };

    let variants: Vec<(&str, TrajPreviewConfig)> = vec![
        ("vroot64_180k_k6_l8", TrajPreviewConfig { kappa: 6.0, ridge_lambda_main: 8.0, ridge_lambda_inter: 16.0, ..vroot_base.clone() }),
        ("vroot64_180k_k12_l12", TrajPreviewConfig { kappa: 12.0, ..vroot_base.clone() }),
        ("vroot64_180k_k16_l12", vroot_base.clone()),
        ("vroot64_180k_k24_l14", TrajPreviewConfig { kappa: 24.0, ridge_lambda_main: 14.0, ridge_lambda_inter: 24.0, ..vroot_base.clone() }),
        ("vroot64_180k_k32_l16", TrajPreviewConfig { kappa: 32.0, ridge_lambda_main: 16.0, ridge_lambda_inter: 28.0, ..vroot_base.clone() }),
        ("vroot64_180k_k16_int31", TrajPreviewConfig { ridge_basis: TrajRidgeBasis::Interact31, ..vroot_base.clone() }),
        ("vroot64_180k_k16_add25", TrajPreviewConfig { ridge_basis: TrajRidgeBasis::Additive25, ..vroot_base.clone() }),
        ("vroot64_180k_k16_nocrn", TrajPreviewConfig { crn_mode: TrajCrnMode::Independent, ..vroot_base.clone() }),
        ("vroot48_180k_k16", TrajPreviewConfig { rb_exit: RbExitMode::SearchVroot { max_alive: 3, eval_iters: 48 }, ..vroot_base.clone() }),
        ("vroot64_240k_k16", TrajPreviewConfig { total_iters: 240_000, ..vroot_base.clone() }),
        ("vroot64_240k_k24", TrajPreviewConfig { total_iters: 240_000, kappa: 24.0, ridge_lambda_main: 14.0, ridge_lambda_inter: 24.0, ..vroot_base.clone() }),
        ("term_240k_k16", TrajPreviewConfig { total_iters: 240_000, rb_exit: RbExitMode::Terminal, ..vroot_base.clone() }),
    ];

    let mut rows = Vec::new();
    for (vname, vcfg) in &variants {
        let mut jobs = Vec::new();
        for mi in 0..matchups.len() {
            for s in 0..n_seeds {
                jobs.push((mi, s));
            }
        }
        let cur = AtomicUsize::new(0);
        let res: Vec<(usize, f64, usize, Vec<([u8; 3], f64)>)> = std::thread::scope(|sc| {
            let mut hs = Vec::new();
            for _ in 0..threads.min(jobs.len()).max(1) {
                let (cur_r, jobs_r, ms_r) = (&cur, &jobs, &matchups);
                hs.push(sc.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                        if idx >= jobs_r.len() {
                            break;
                        }
                        let (mi, s) = jobs_r[idx];
                        let ms = &ms_r[mi];
                        let sets_a = &team_map[ms.team_a];
                        let sets_b = &team_map[ms.team_b];
                        let pool = disclosed_pool(sets_b);
                        let seed = 0x2026_1126_F0C5_0001u64
                            ^ ((mi as u64) << 24)
                            ^ ((s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let (dist, val0, games, _, _, _) =
                            run_traj_once(dex, sets_a, sets_b, &pool, 0, vcfg, seed);
                        out.push((mi, val0, games, dist));
                    }
                    out
                }));
            }
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mut ms1t = Vec::new();
        for (mi, ms) in matchups.iter().enumerate() {
            let sets_a = &team_map[ms.team_a];
            let sets_b = &team_map[ms.team_b];
            let pool = disclosed_pool(sets_b);
            let (_, _, _, _, tms, _) = run_traj_once(dex, sets_a, sets_b, &pool, 0, vcfg, 0x7777 ^ (mi as u64));
            ms1t.push(tms);
        }
        let (med_ms, p95_ms) = median_p95(&ms1t);
        let mut maes = Vec::new();
        let mut sds = Vec::new();
        let mut tvs = Vec::new();
        let mut gs = Vec::new();
        for (mi, ms) in matchups.iter().enumerate() {
            let vals: Vec<f64> = res.iter().filter(|x| x.0 == mi).map(|x| x.1).collect();
            let pols: Vec<Vec<([u8; 3], f64)>> = res.iter().filter(|x| x.0 == mi).map(|x| x.3.clone()).collect();
            for x in res.iter().filter(|x| x.0 == mi) {
                gs.push(x.2 as f64);
            }
            let (vm, vsd) = mean_sd(&vals);
            maes.push((vm - ms.true_nash_a).abs());
            sds.push(vsd);
            tvs.push(mean_pairwise_tv(&pols));
        }
        rows.push(json!({
            "name": vname,
            "mae": r4(maes.iter().sum::<f64>() / 3.0),
            "sd8": r4(sds.iter().sum::<f64>() / 3.0),
            "tv": r4(tvs.iter().sum::<f64>() / 3.0),
            "games": r2(gs.iter().sum::<f64>() / (gs.len().max(1) as f64)),
            "ms_med": r2(med_ms),
            "ms_p95": r2(p95_ms),
        }));
    }
    let out = json!({"mode":"a3_focus","wall_s":r2(t0.elapsed().as_secs_f64()),"rows":rows});
    std::fs::write("/tmp/pillar_a_a3_focus.json", serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("{}", out);
}

fn run_suite1_blind9k(
    dex: &Dex,
    team_map: &HashMap<String, Vec<PokemonSet>>,
    threads: usize,
    cfg_tag: &str,
    h2h_pairs: usize,
    screen_g: usize,
    exploit_9k_games: usize,
    skip_v2_exploit: bool,
) {
    let t0 = Instant::now();
    let matchups = pilot_matchups();
    let traj_cfg = make_suite_cfg(cfg_tag);
    let n_seeds = 8usize;

    let mut matchup_reports = Vec::new();
    let mut h2h_total_w = 0.0;
    let mut h2h_total_n = 0usize;
    let mut h2h_pair_diffs: Vec<f64> = Vec::new();
    let mut traj_all_ms = Vec::new();
    let mut v2_all_ms = Vec::new();
    let mut traj_nash_maes = Vec::new();
    let mut traj_nash_sds = Vec::new();
    let mut expl_traj_max_gains = Vec::new();
    let mut expl_v2_max_gains = Vec::new();

    for (mi, ms) in matchups.iter().enumerate() {
        let sets_a = &team_map[ms.team_a];
        let sets_b = &team_map[ms.team_b];
        let pool_a = disclosed_pool(sets_b);
        let pool_b = disclosed_pool(sets_a);

        // 1. Collect 8 independent preview policies for BOTH sides (`Traj-DO` and `policy-v2`),
        //    plus uncontended 1-thread timings!
        let mut traj_pols_a = Vec::with_capacity(n_seeds);
        let mut traj_pols_b = Vec::with_capacity(n_seeds);
        let mut traj_vals_a = Vec::with_capacity(n_seeds);
        let mut v2_pols_a = Vec::with_capacity(n_seeds);
        let mut v2_pols_b = Vec::with_capacity(n_seeds);

        // Serial 1-thread timing on seeds 0..2, parallel on seeds 2..8.
        for s in 0..2usize {
            let sa = 0x2026_5100_0001u64
                ^ ((mi as u64) << 24)
                ^ ((s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let sb = sa ^ 0xABCD_EF01_1234_5678;
            let (pa, va, _, _, ms_ta, _) =
                run_traj_once(dex, sets_a, sets_b, &pool_a, 0, &traj_cfg, sa);
            let (pb, _, _, _, ms_tb, _) =
                run_traj_once(dex, sets_a, sets_b, &pool_b, 1, &traj_cfg, sb);
            traj_all_ms.push(ms_ta);
            traj_all_ms.push(ms_tb);
            traj_pols_a.push(pa);
            traj_pols_b.push(pb);
            traj_vals_a.push(va);

            let (v2a, ms_va) = run_shipped_policy_v2(dex, sets_a, sets_b, &pool_a, 0, 240_000, sa);
            let (v2b, ms_vb) = run_shipped_policy_v2(dex, sets_a, sets_b, &pool_b, 1, 240_000, sb);
            v2_all_ms.push(ms_va);
            v2_all_ms.push(ms_vb);
            v2_pols_a.push(v2a);
            v2_pols_b.push(v2b);
        }

        let rem_seeds: Vec<(usize, usize)> = (2..n_seeds)
            .flat_map(|s| [(0usize, s), (1usize, s)])
            .collect();
        let cur = AtomicUsize::new(0);
        let par_pols: Vec<(usize, Vec<([u8; 3], f64)>, f64, Vec<([u8; 3], f64)>)> =
            std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for _ in 0..threads.min(rem_seeds.len()).max(1) {
                    let (cur_r, rs_r, pa_r, pb_r, cfg_r) =
                        (&cur, &rem_seeds, &pool_a, &pool_b, &traj_cfg);
                    hs.push(sc.spawn(move || {
                        let mut out = Vec::new();
                        loop {
                            let idx = cur_r.fetch_add(1, Ordering::Relaxed);
                            if idx >= rs_r.len() {
                                break;
                            }
                            let (side, s) = rs_r[idx];
                            let sa = 0x2026_5100_0001u64
                                ^ ((mi as u64) << 24)
                                ^ ((s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                            let seed = if side == 0 {
                                sa
                            } else {
                                sa ^ 0xABCD_EF01_1234_5678
                            };
                            let pool = if side == 0 { pa_r } else { pb_r };
                            let (pt, vt, _, _, _, _) =
                                run_traj_once(dex, sets_a, sets_b, pool, side, cfg_r, seed);
                            let (pv, _) =
                                run_shipped_policy_v2(dex, sets_a, sets_b, pool, side, 240_000, seed);
                            out.push((side, pt, vt, pv));
                        }
                        out
                    }));
                }
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
        for (side, pt, vt, pv) in par_pols {
            if side == 0 {
                traj_pols_a.push(pt);
                traj_vals_a.push(vt);
                v2_pols_a.push(pv);
            } else {
                traj_pols_b.push(pt);
                v2_pols_b.push(pv);
            }
        }

        let (val_mean, val_sd) = mean_sd(&traj_vals_a);
        traj_nash_maes.push((val_mean - ms.true_nash_a).abs());
        traj_nash_sds.push(val_sd);
        let avg_traj_a = avg_policy_dist(&traj_pols_a);
        let avg_traj_b = avg_policy_dist(&traj_pols_b);
        let avg_v2_a = avg_policy_dist(&v2_pols_a);
        let avg_v2_b = avg_policy_dist(&v2_pols_b);

        // 2. Real BlindAgent 9k seat-paired CRN H2H: Traj-DO vs policy-v2!
        //    Pair k plays 2 games:
        //      Game A: Traj-DO on side 0 (A) vs policy-v2 on side 1 (B)
        //      Game B: policy-v2 on side 0 (A) vs Traj-DO on side 1 (B)
        //    Sharing battle seed + agent seeds across the seat pair (CRN!).
        let cur_h = AtomicUsize::new(0);
        let h2h_res: Vec<(f64, f64)> = std::thread::scope(|sc| {
            let mut hs = Vec::new();
            for _ in 0..threads.min(h2h_pairs).max(1) {
                let (cur_r, pa_r, pb_r, ta_r, tb_r, va_r, vb_r) = (
                    &cur_h,
                    &pool_a,
                    &pool_b,
                    &traj_pols_a,
                    &traj_pols_b,
                    &v2_pols_a,
                    &v2_pols_b,
                );
                hs.push(sc.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let k = cur_r.fetch_add(1, Ordering::Relaxed);
                        if k >= h2h_pairs {
                            break;
                        }
                        let mut drng = SplitMix64::new(
                            0x2026_1126_4240_0001u64
                                ^ ((mi as u64) << 28)
                                ^ ((k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                        );
                        let bseed = drng.next();
                        let aseed0 = drng.next();
                        let aseed1 = drng.next();
                        let si = k % n_seeds;
                        let draw_a = drng.next_f64();
                        let draw_b = drng.next_f64();

                        // Seat 0: Traj-DO(A) vs policy-v2(B)
                        let p_ta = sample_pick(&ta_r[si], draw_a);
                        let p_vb = sample_pick(&vb_r[si], draw_b);
                        let g0 = play_blind_battle_with_picks(
                            dex,
                            sets_a,
                            sets_b,
                            Arc::clone(pa_r),
                            Arc::clone(pb_r),
                            p_ta,
                            p_vb,
                            9000,
                            bseed,
                            aseed0,
                            aseed1,
                        );
                        // Seat 1: policy-v2(A) vs Traj-DO(B) -> Traj-DO score is (1 - g1_side0)
                        let p_va = sample_pick(&va_r[si], draw_a);
                        let p_tb = sample_pick(&tb_r[si], draw_b);
                        let g1_s0 = play_blind_battle_with_picks(
                            dex,
                            sets_a,
                            sets_b,
                            Arc::clone(pa_r),
                            Arc::clone(pb_r),
                            p_va,
                            p_tb,
                            9000,
                            bseed,
                            aseed0,
                            aseed1,
                        );
                        out.push((g0, 1.0 - g1_s0));
                    }
                    out
                }));
            }
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });

        let mut m_traj_w = 0.0;
        let mut m_pair_diffs = Vec::with_capacity(h2h_pairs);
        for &(s0, s1) in &h2h_res {
            m_traj_w += s0 + s1;
            // Average of seat pair minus 0.5 -> paired advantage of Traj-DO over policy-v2.
            let diff = 0.5 * (s0 + s1) - 0.5;
            m_pair_diffs.push(diff);
            h2h_pair_diffs.push(diff);
        }
        let m_h2h_n = h2h_pairs * 2;
        h2h_total_w += m_traj_w;
        h2h_total_n += m_h2h_n;
        let m_h2h_wr = m_traj_w / (m_h2h_n as f64);

        // 3. Real BlindAgent 9k 48-arm unilateral exploitability (both contexts: side 0 defending, side 1 defending)!
        //    Stage A: Screen ALL legal opposing arms with 12 BlindAgent 3k CRN games against defender policy.
        //    Stage B: Confirm top 4 screened arms + ALL known killer comps with `exploit_9k_games` independent
        //             BlindAgent 9k CRN games against defender policy, AND compute baseline policy-vs-policy EV!
        let mut b_tmp = Battle::from_fixture(dex, "1,2,3,4", sets_a, sets_b).unwrap();
        let c0 = canonical_preview_triples(dex, &mut b_tmp, 0);
        let c1 = canonical_preview_triples(dex, &mut b_tmp, 1);

        let eval_context_exploit = |def_side: usize,
                                    def_name: &str,
                                    def_dist: &[([u8; 3], f64)],
                                    opp_dist_base: &[([u8; 3], f64)],
                                    salt: u64|
         -> (f64, Value) {
            let atk_side = 1 - def_side;
            let atk_cands = if atk_side == 0 { &c0 } else { &c1 };
            let atk_sets = if atk_side == 0 { sets_a } else { sets_b };

            // Stage A: BlindAgent 3k screen over all legal attacker triples.
            let cur_s = AtomicUsize::new(0);
            let screen_scores: Vec<(usize, f64)> = std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for _ in 0..threads.min(atk_cands.len()).max(1) {
                    let (cur_r, ac_r, pa_r, pb_r) = (&cur_s, atk_cands, &pool_a, &pool_b);
                    hs.push(sc.spawn(move || {
                        let mut out = Vec::new();
                        loop {
                            let ai = cur_r.fetch_add(1, Ordering::Relaxed);
                            if ai >= ac_r.len() {
                                break;
                            }
                            let mut acc = 0.0;
                            for g in 0..screen_g {
                                let mut grng = SplitMix64::new(
                                    0x2026_E300_0001u64
                                        ^ salt
                                        ^ ((mi as u64) << 24)
                                        ^ ((def_side as u64) << 20)
                                        ^ ((g as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                                );
                                let bseed = grng.next();
                                let as0 = grng.next();
                                let as1 = grng.next();
                                let def_pick = sample_pick(def_dist, grng.next_f64());
                                let atk_pick = ac_r[ai];
                                let (p0, p1) = if def_side == 0 {
                                    (def_pick, atk_pick)
                                } else {
                                    (atk_pick, def_pick)
                                };
                                let v0 = play_blind_battle_with_picks(
                                    dex,
                                    sets_a,
                                    sets_b,
                                    Arc::clone(pa_r),
                                    Arc::clone(pb_r),
                                    p0,
                                    p1,
                                    1500,
                                    bseed,
                                    as0,
                                    as1,
                                );
                                let v_atk = if atk_side == 0 { v0 } else { 1.0 - v0 };
                                acc += v_atk;
                            }
                            out.push((ai, acc / (screen_g as f64)));
                        }
                        out
                    }));
                }
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });
            let mut ranked = screen_scores;
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

            let mut finalists: Vec<[u8; 3]> = ranked
                .iter()
                .take(3)
                .map(|&(ai, _)| atk_cands[ai])
                .collect();
            for &(_, kside, ktr) in &ms.killers {
                if kside == atk_side {
                    let ct = canonical_triple(ktr);
                    if !finalists.contains(&ct) {
                        finalists.push(ct);
                    }
                }
            }

            // Stage B: Independent BlindAgent 9k confirmation for baseline + all finalists (CRN seeds!).
            // Index 0 = baseline opposing policy (`opp_dist_base`), indices 1..=F = finalists.
            let n_arms = 1 + finalists.len();
            let total_jobs = n_arms * exploit_9k_games;
            let cur_f = AtomicUsize::new(0);
            let fin_res: Vec<(usize, f64)> = std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for _ in 0..threads.min(total_jobs).max(1) {
                    let (cur_r, fin_r, pa_r, pb_r) = (&cur_f, &finalists, &pool_a, &pool_b);
                    hs.push(sc.spawn(move || {
                        let mut out = Vec::new();
                        loop {
                            let jid = cur_r.fetch_add(1, Ordering::Relaxed);
                            if jid >= total_jobs {
                                break;
                            }
                            let arm = jid / exploit_9k_games;
                            let g = jid % exploit_9k_games;
                            // Shared CRN seed per game index `g` across baseline & all finalist arms!
                            let mut grng = SplitMix64::new(
                                0x2026_E900_0001u64
                                    ^ salt
                                    ^ ((mi as u64) << 24)
                                    ^ ((def_side as u64) << 20)
                                    ^ ((g as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
                            );
                            let bseed = grng.next();
                            let as0 = grng.next();
                            let as1 = grng.next();
                            let def_pick = sample_pick(def_dist, grng.next_f64());
                            let atk_pick = if arm == 0 {
                                sample_pick(opp_dist_base, grng.next_f64())
                            } else {
                                fin_r[arm - 1]
                            };
                            let (p0, p1) = if def_side == 0 {
                                (def_pick, atk_pick)
                            } else {
                                (atk_pick, def_pick)
                            };
                            let v0 = play_blind_battle_with_picks(
                                dex,
                                sets_a,
                                sets_b,
                                Arc::clone(pa_r),
                                Arc::clone(pb_r),
                                p0,
                                p1,
                                9000,
                                bseed,
                                as0,
                                as1,
                            );
                            let v_atk = if atk_side == 0 { v0 } else { 1.0 - v0 };
                            out.push((arm, v_atk));
                        }
                        out
                    }));
                }
                hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
            });

            let mut arm_sums = vec![0.0; n_arms];
            let mut arm_ns = vec![0usize; n_arms];
            for (arm, v) in fin_res {
                arm_sums[arm] += v;
                arm_ns[arm] += 1;
            }
            let base_atk_ev = arm_sums[0] / (arm_ns[0].max(1) as f64);
            let true_def_nash = if def_side == 0 {
                ms.true_nash_a
            } else {
                1.0 - ms.true_nash_a
            };
            let true_atk_nash = 1.0 - true_def_nash;

            let mut best_atk_ev = 0.0f64;
            let mut best_atk_tr = finalists[0];
            let mut fin_json = Vec::new();
            for (fi, &tr) in finalists.iter().enumerate() {
                let ev = arm_sums[fi + 1] / (arm_ns[fi + 1].max(1) as f64);
                if ev > best_atk_ev {
                    best_atk_ev = ev;
                    best_atk_tr = tr;
                }
                let is_known_killer = ms
                    .killers
                    .iter()
                    .find(|&&(_, ks, kt)| ks == atk_side && canonical_triple(kt) == tr)
                    .map(|&(lbl, _, _)| lbl);
                fin_json.push(json!({
                    "pick": fmt_triple(atk_sets, tr),
                    "atk_9k_ev": r4(ev),
                    "gain_vs_base": r4(ev - base_atk_ev),
                    "gain_vs_true_nash": r4(ev - true_atk_nash),
                    "killer_tag": is_known_killer,
                }));
            }
            let max_gain_vs_nash = (best_atk_ev - true_atk_nash).max(0.0);
            let report = json!({
                "defender": def_name,
                "def_side": def_side,
                "base_atk_9k_ev": r4(base_atk_ev),
                "true_atk_nash": r4(true_atk_nash),
                "worst_atk_pick": fmt_triple(atk_sets, best_atk_tr),
                "worst_atk_9k_ev": r4(best_atk_ev),
                "exploit_vs_true_nash": r4(best_atk_ev - true_atk_nash),
                "exploit_vs_base": r4(best_atk_ev - base_atk_ev),
                "finalists": fin_json,
            });
            (max_gain_vs_nash, report)
        };

        let (et0, rep_t0) =
            eval_context_exploit(0, "traj_do", &avg_traj_a, &avg_traj_b, 0x1111);
        let (et1, rep_t1) =
            eval_context_exploit(1, "traj_do", &avg_traj_b, &avg_traj_a, 0x2222);
        let (ev0, rep_v0) = if skip_v2_exploit {
            (0.0, Value::Null)
        } else {
            eval_context_exploit(0, "policy_v2", &avg_v2_a, &avg_v2_b, 0x3333)
        };
        let (ev1, rep_v1) = if skip_v2_exploit {
            (0.0, Value::Null)
        } else {
            eval_context_exploit(1, "policy_v2", &avg_v2_b, &avg_v2_a, 0x4444)
        };

        expl_traj_max_gains.push(et0);
        expl_traj_max_gains.push(et1);
        expl_v2_max_gains.push(ev0);
        expl_v2_max_gains.push(ev1);

        let top_fmt = |sets: &[PokemonSet], d: &[([u8; 3], f64)]| -> Vec<Value> {
            d.iter()
                .take(5)
                .map(|&(t, p)| json!({"pick": fmt_triple(sets, t), "p": r4(p)}))
                .collect()
        };

        matchup_reports.push(json!({
            "code": ms.code,
            "true_243k_nash_a": ms.true_nash_a,
            "traj_nash_a_mean": r4(val_mean),
            "traj_nash_a_sd8": r4(val_sd),
            "h2h_traj_vs_v2_9k": r4(m_h2h_wr),
            "h2h_games": m_h2h_n,
            "traj_top_a": top_fmt(sets_a, &avg_traj_a),
            "traj_top_b": top_fmt(sets_b, &avg_traj_b),
            "exploit_traj_s0": rep_t0,
            "exploit_traj_s1": rep_t1,
            "exploit_v2_s0": rep_v0,
            "exploit_v2_s1": rep_v1,
        }));
    }

    let pooled_h2h = h2h_total_w / (h2h_total_n.max(1) as f64);
    let (diff_m, diff_sd) = mean_sd(&h2h_pair_diffs);
    let h2h_se = diff_sd / (h2h_pair_diffs.len().max(1) as f64).sqrt();
    let h2h_z = if h2h_se > 1e-9 { diff_m / h2h_se } else { 0.0 };

    let (traj_med_ms, traj_p95_ms) = median_p95(&traj_all_ms);
    let (v2_med_ms, v2_p95_ms) = median_p95(&v2_all_ms);
    let avg_mae = traj_nash_maes.iter().sum::<f64>() / (traj_nash_maes.len() as f64);
    let avg_sd = traj_nash_sds.iter().sum::<f64>() / (traj_nash_sds.len() as f64);
    let avg_expl_traj =
        expl_traj_max_gains.iter().sum::<f64>() / (expl_traj_max_gains.len() as f64);
    let avg_expl_v2 = expl_v2_max_gains.iter().sum::<f64>() / (expl_v2_max_gains.len() as f64);

    let full = json!({
        "mode": "suite1_blind9k",
        "cfg_tag": cfg_tag,
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "h2h_traj_vs_v2_9k": r4(pooled_h2h),
        "h2h_delta_pp": r2(diff_m * 100.0),
        "h2h_se_pp": r2(h2h_se * 100.0),
        "h2h_z": r2(h2h_z),
        "h2h_total_games": h2h_total_n,
        "traj_nash_mae": r4(avg_mae),
        "traj_nash_sd8": r4(avg_sd),
        "traj_avg_exploit_vs_nash": r4(avg_expl_traj),
        "v2_avg_exploit_vs_nash": r4(avg_expl_v2),
        "traj_ms1t_med": r2(traj_med_ms),
        "traj_ms1t_p95": r2(traj_p95_ms),
        "v2_ms1t_med": r2(v2_med_ms),
        "v2_ms1t_p95": r2(v2_p95_ms),
        "matchups": matchup_reports,
    });
    let out_file = format!("/tmp/pillar_a_suite1_{}.json", cfg_tag);
    std::fs::write(&out_file, serde_json::to_string_pretty(&full).unwrap()).unwrap();
    let compact = json!({
        "mode": "suite1_blind9k",
        "cfg": cfg_tag,
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "h2h_9k": r4(pooled_h2h),
        "delta_pp": r2(diff_m * 100.0),
        "se_pp": r2(h2h_se * 100.0),
        "z": r2(h2h_z),
        "mae": r4(avg_mae),
        "sd8": r4(avg_sd),
        "expl_traj": r4(avg_expl_traj),
        "expl_v2": r4(avg_expl_v2),
        "traj_ms_med": r2(traj_med_ms),
        "traj_ms_p95": r2(traj_p95_ms),
        "v2_ms_med": r2(v2_med_ms),
    });
    println!("{}", compact);
}

// =========================================================================
// Mode 4: Suite 2 — Multi-World Unseen `belief-pool-v3` (`hash % 5 == 0`)
// =========================================================================

fn run_suite2_multiworld(
    dex: &Dex,
    threads: usize,
    cfg_tag: &str,
    n_matchups: usize,
    pairs_per_matchup: usize,
) {
    let t0 = Instant::now();
    let root = repo_root();
    let bp_txt = std::fs::read_to_string(root.join("data/belief-pool-v3/belief-pool.json")).unwrap();
    let bp_cat: CatalogFile = serde_json::from_str(&bp_txt).unwrap();
    let full_pool = Arc::new(MetaPool {
        teams: bp_cat
            .teams
            .iter()
            .map(|t| MetaTeam {
                id: t.id.clone(),
                sets: t.sets.clone(),
                weight: t.weight,
            })
            .collect(),
    });

    // Unseen held-out teams: `canonical_team_hash % 5 == 0`.
    let unseen: Vec<usize> = (0..bp_cat.teams.len())
        .filter(|&i| (canonical_team_hash(&bp_cat.teams[i].sets) % 5) == 0)
        .collect();

    // Also inspect world counts (`alive.len()`) per unseen team under `belief-pool-v3` so our
    // 20-matchup panel naturally includes rich multi-world (W >= 2) cases AND we stress W = 1, 8, 24!
    let mut rng = SplitMix64::new(0x2026_1126_5200_0001);
    let mut team_worlds: Vec<(usize, usize)> = Vec::new();
    for &idx in &unseen {
        let opp_sets = &bp_cat.teams[idx].sets;
        let b = Battle::from_fixture(dex, "1,2,3,4", &bp_cat.teams[unseen[0]].sets, opp_sets).unwrap();
        let (mut lp, mut obs) =
            LocalPlayer::new_with_own_sets(&b, dex, 0, bp_cat.teams[unseen[0]].sets.clone()).unwrap();
        lp.observe(&b, dex, &mut obs).unwrap();
        let mut bel = Belief::new(dex, &full_pool, &obs);
        bel.sync(dex, &obs);
        let w_cnt = bel.alive().len().min(24).max(1);
        team_worlds.push((idx, w_cnt));
    }
    // Prefer multi-world opponents (`W >= 2`) for at least half the panel, plus general unseen pairs.
    let multi_idx: Vec<usize> = team_worlds.iter().filter(|&&(_, w)| w >= 2).map(|&(i, _)| i).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for k in 0..n_matchups {
        let a = unseen[rng.below(unseen.len())];
        let b = if !multi_idx.is_empty() && k < (n_matchups * 2 / 3) {
            let mut pick = multi_idx[rng.below(multi_idx.len())];
            if pick == a {
                pick = multi_idx[(k + 1) % multi_idx.len()];
            }
            pick
        } else {
            let mut pick = unseen[rng.below(unseen.len())];
            if pick == a {
                pick = unseen[(k + 1) % unseen.len()];
            }
            pick
        };
        pairs.push((a, b));
    }

    let traj_cfg = make_suite_cfg(cfg_tag);
    let traj_cfg_noshare = TrajPreviewConfig {
        ridge_shared_multiworld: false,
        ..traj_cfg.clone()
    };

    // 1. Clean 1-thread latency & zero-leakage check across all `n_matchups` (both sides) + explicit W = 1, 8, 24 stress test!
    let mut traj_ms_all = Vec::new();
    let mut traj_s1_ms_all = Vec::new();
    let mut v2_ms_all = Vec::new();
    let mut w_counts = Vec::new();
    let mut leak_failures = 0usize;
    let mut precomputed_pols: Vec<(
        Vec<([u8; 3], f64)>,
        Vec<([u8; 3], f64)>,
        Vec<([u8; 3], f64)>,
        Vec<([u8; 3], f64)>,
        Vec<([u8; 3], f64)>,
        Vec<([u8; 3], f64)>,
    )> = Vec::with_capacity(n_matchups);

    for (mi, &(ia, ib)) in pairs.iter().enumerate() {
        let sets_a = &bp_cat.teams[ia].sets;
        let sets_b = &bp_cat.teams[ib].sets;
        let sa = 0x2026_5210_0001u64 ^ ((mi as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let sb = sa ^ 0x5555_AAAA_3333_CCCC;

        let (pa, _, _, s1a, msa, st_a) =
            run_traj_once(dex, sets_a, sets_b, &full_pool, 0, &traj_cfg, sa);
        let (pb, _, _, s1b, msb, st_b) =
            run_traj_once(dex, sets_a, sets_b, &full_pool, 1, &traj_cfg, sb);
        let (pa_ns, _, _, _, _, _) =
            run_traj_once(dex, sets_a, sets_b, &full_pool, 0, &traj_cfg_noshare, sa);
        let (pb_ns, _, _, _, _, _) =
            run_traj_once(dex, sets_a, sets_b, &full_pool, 1, &traj_cfg_noshare, sb);
        let (v2a, msva) = run_shipped_policy_v2(dex, sets_a, sets_b, &full_pool, 0, 240_000, sa);
        let (v2b, msvb) = run_shipped_policy_v2(dex, sets_a, sets_b, &full_pool, 1, 240_000, sb);

        traj_ms_all.push(msa);
        traj_ms_all.push(msb);
        traj_s1_ms_all.push(s1a);
        traj_s1_ms_all.push(s1b);
        v2_ms_all.push(msva);
        v2_ms_all.push(msvb);
        w_counts.push(st_a.world_counts().0);
        w_counts.push(st_b.world_counts().0);

        // Zero-leakage check on first 6 matchups: mutate opponent's hidden moves & verify bit-identical policy!
        if mi < 6 {
            let mut mut_b = sets_b.clone();
            for mon in &mut mut_b {
                if !mon.moves.is_empty() {
                    mon.moves.reverse();
                }
                mon.item = "Leftovers".to_string();
            }
            let (pa_mut, _, _, _, _, _) =
                run_traj_once(dex, sets_a, &mut_b, &full_pool, 0, &traj_cfg, sa);
            if pa != pa_mut {
                leak_failures += 1;
            }
        }
        precomputed_pols.push((pa, pb, v2a, v2b, pa_ns, pb_ns));
    }

    // Explicit W = 1, 8, 24 synthetic multi-world stress benchmark (exact 1-thread latency scaling!).
    let mut w_stress = Vec::new();
    {
        let sets_a = &bp_cat.teams[unseen[0]].sets;
        let sets_b = &bp_cat.teams[unseen[1]].sets;
        for &target_w in &[1usize, 8usize, 24usize] {
            let mut synth_teams = Vec::with_capacity(target_w);
            for w in 0..target_w {
                let mut cloned = sets_b.clone();
                // Keep identical public species & levels while varying hidden item/move order across worlds!
                if w > 0 {
                    let slot = w % cloned.len();
                    cloned[slot].item = match w % 4 {
                        0 => "Leftovers",
                        1 => "Miracle Berry",
                        2 => "Mint Berry",
                        _ => "Gold Berry",
                    }
                    .to_string();
                }
                synth_teams.push(MetaTeam {
                    id: format!("w{w}"),
                    sets: cloned,
                    weight: Some(1.0 / ((w + 1) as f64).powf(0.5)),
                });
            }
            let synth_pool = Arc::new(MetaPool { teams: synth_teams });
            let mut t_ms = Vec::new();
            let mut s1_ms = Vec::new();
            let mut v2_ms = Vec::new();
            let mut g_cnt = 0usize;
            for s in 0..3usize {
                let sd = 0x2026_9999_0001u64 ^ ((target_w as u64) << 16) ^ (s as u64);
                let (_, _, gc, s1, tot, st) =
                    run_traj_once(dex, sets_a, sets_b, &synth_pool, 0, &traj_cfg, sd);
                assert_eq!(st.world_counts().0, target_w);
                t_ms.push(tot);
                s1_ms.push(s1);
                g_cnt = gc;
                let (_, vm) = run_shipped_policy_v2(dex, sets_a, sets_b, &synth_pool, 0, 240_000, sd);
                v2_ms.push(vm);
            }
            let (tm, tp) = median_p95(&t_ms);
            let (sm, _) = median_p95(&s1_ms);
            let (vm, _) = median_p95(&v2_ms);
            w_stress.push(json!({
                "W": target_w,
                "traj_med_ms1t": r2(tm),
                "traj_p95_ms1t": r2(tp),
                "stage1_med_ms1t": r2(sm),
                "games_completed": g_cnt,
                "v2_med_ms1t": r2(vm),
            }));
        }
    }

    // 2. Real BlindAgent 9k seat-paired CRN H2H across all `n_matchups` (`Traj-DO` vs `policy-v2` AND `Traj-DO shared` vs `noshare`)!
    let total_pairs = n_matchups * pairs_per_matchup;
    let cur = AtomicUsize::new(0);
    let h2h_res: Vec<(usize, f64, f64)> = std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(total_pairs).max(1) {
            let (cur_r, prs_r, pol_r, fp_r, bpc_r) =
                (&cur, &pairs, &precomputed_pols, &full_pool, &bp_cat.teams);
            hs.push(sc.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let pid = cur_r.fetch_add(1, Ordering::Relaxed);
                    if pid >= total_pairs {
                        break;
                    }
                    let mi = pid / pairs_per_matchup;
                    let k = pid % pairs_per_matchup;
                    let (ia, ib) = prs_r[mi];
                    let sets_a = &bpc_r[ia].sets;
                    let sets_b = &bpc_r[ib].sets;
                    let (ref pa, ref pb, ref v2a, ref v2b, _, _) = pol_r[mi];

                    let mut drng = SplitMix64::new(
                        0x2026_5220_0001u64
                            ^ ((mi as u64) << 24)
                            ^ ((k as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                    );
                    let bseed = drng.next();
                    let as0 = drng.next();
                    let as1 = drng.next();
                    let da = drng.next_f64();
                    let db = drng.next_f64();

                    let g0 = play_blind_battle_with_picks(
                        dex,
                        sets_a,
                        sets_b,
                        Arc::clone(fp_r),
                        Arc::clone(fp_r),
                        sample_pick(pa, da),
                        sample_pick(v2b, db),
                        9000,
                        bseed,
                        as0,
                        as1,
                    );
                    let g1_s0 = play_blind_battle_with_picks(
                        dex,
                        sets_a,
                        sets_b,
                        Arc::clone(fp_r),
                        Arc::clone(fp_r),
                        sample_pick(v2a, da),
                        sample_pick(pb, db),
                        9000,
                        bseed,
                        as0,
                        as1,
                    );
                    out.push((mi, g0, 1.0 - g1_s0));
                }
                out
            }));
        }
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });

    let mut tot_w = 0.0;
    let mut tot_n = 0usize;
    let mut pair_diffs = Vec::new();
    let mut per_m_w = vec![0.0; n_matchups];
    let mut per_m_n = vec![0usize; n_matchups];
    for (mi, s0, s1) in h2h_res {
        tot_w += s0 + s1;
        tot_n += 2;
        pair_diffs.push(0.5 * (s0 + s1) - 0.5);
        per_m_w[mi] += s0 + s1;
        per_m_n[mi] += 2;
    }
    let wr = tot_w / (tot_n.max(1) as f64);
    let (dm, dsd) = mean_sd(&pair_diffs);
    let se = dsd / (pair_diffs.len().max(1) as f64).sqrt();
    let z = if se > 1e-9 { dm / se } else { 0.0 };

    let (t_med, t_p95) = median_p95(&traj_ms_all);
    let (s1_med, _) = median_p95(&traj_s1_ms_all);
    let (v2_med, v2_p95) = median_p95(&v2_ms_all);
    let mw_frac = w_counts.iter().filter(|&&w| w >= 2).count() as f64 / (w_counts.len().max(1) as f64);

    let full = json!({
        "mode": "suite2_multiworld",
        "cfg_tag": cfg_tag,
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "n_matchups": n_matchups,
        "total_games": tot_n,
        "leak_failures": leak_failures,
        "multiworld_side_frac": r4(mw_frac),
        "world_counts": w_counts,
        "h2h_traj_vs_v2_9k": r4(wr),
        "delta_pp": r2(dm * 100.0),
        "se_pp": r2(se * 100.0),
        "z": r2(z),
        "traj_med_ms1t": r2(t_med),
        "traj_p95_ms1t": r2(t_p95),
        "stage1_med_ms1t": r2(s1_med),
        "v2_med_ms1t": r2(v2_med),
        "v2_p95_ms1t": r2(v2_p95),
        "w_stress": w_stress,
        "per_matchup_wr": per_m_w
            .iter()
            .zip(&per_m_n)
            .enumerate()
            .map(|(mi, (&w, &n))| json!({
                "m": mi,
                "team_a": bp_cat.teams[pairs[mi].0].id,
                "team_b": bp_cat.teams[pairs[mi].1].id,
                "wr": r4(w / (n.max(1) as f64)),
                "games": n,
            }))
            .collect::<Vec<_>>(),
    });
    let out_file = format!("/tmp/pillar_a_suite2_{}.json", cfg_tag);
    std::fs::write(&out_file, serde_json::to_string_pretty(&full).unwrap()).unwrap();
    let compact = json!({
        "mode": "suite2_multiworld",
        "cfg": cfg_tag,
        "wall_s": r2(t0.elapsed().as_secs_f64()),
        "games": tot_n,
        "leaks": leak_failures,
        "h2h_9k": r4(wr),
        "delta_pp": r2(dm * 100.0),
        "se_pp": r2(se * 100.0),
        "z": r2(z),
        "traj_ms_med": r2(t_med),
        "traj_ms_p95": r2(t_p95),
        "v2_ms_med": r2(v2_med),
        "w_stress": w_stress,
    });
    println!("{}", compact);
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct DumpedMatchupPols {
    code: String,
    team_a: String,
    team_b: String,
    pols_a: Vec<Vec<([u8; 3], f64)>>,
    pols_b: Vec<Vec<([u8; 3], f64)>>,
    vals_a: Vec<f64>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct DumpedMwPols {
    team_a: String,
    team_b: String,
    pol_a: Vec<([u8; 3], f64)>,
    pol_b: Vec<([u8; 3], f64)>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct DumpedPoliciesFile {
    tag: String,
    pilot: Vec<DumpedMatchupPols>,
    mw: Vec<DumpedMwPols>,
    pilot_ms_med: f64,
    mw_ms_med: f64,
}

fn select_unseen_mw_pairs(dex: &Dex, bp_cat: &CatalogFile, n_matchups: usize) -> Vec<(usize, usize)> {
    let full_pool = Arc::new(MetaPool {
        teams: bp_cat
            .teams
            .iter()
            .map(|t| MetaTeam {
                id: t.id.clone(),
                sets: t.sets.clone(),
                weight: t.weight,
            })
            .collect(),
    });
    let unseen: Vec<usize> = (0..bp_cat.teams.len())
        .filter(|&i| (canonical_team_hash(&bp_cat.teams[i].sets) % 5) == 0)
        .collect();
    let mut rng = SplitMix64::new(0x2026_1126_5200_0001);
    let mut team_worlds: Vec<(usize, usize)> = Vec::new();
    for &idx in &unseen {
        let opp_sets = &bp_cat.teams[idx].sets;
        let b = Battle::from_fixture(dex, "1,2,3,4", &bp_cat.teams[unseen[0]].sets, opp_sets).unwrap();
        let (mut lp, mut obs) =
            LocalPlayer::new_with_own_sets(&b, dex, 0, bp_cat.teams[unseen[0]].sets.clone()).unwrap();
        lp.observe(&b, dex, &mut obs).unwrap();
        let mut bel = Belief::new(dex, &full_pool, &obs);
        bel.sync(dex, &obs);
        let w_cnt = bel.alive().len().min(24).max(1);
        team_worlds.push((idx, w_cnt));
    }
    let multi_idx: Vec<usize> = team_worlds.iter().filter(|&&(_, w)| w >= 2).map(|&(i, _)| i).collect();
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for k in 0..n_matchups {
        let a = unseen[rng.below(unseen.len())];
        let b = if !multi_idx.is_empty() && k < (n_matchups * 2 / 3) {
            let mut pick = multi_idx[rng.below(multi_idx.len())];
            if pick == a {
                pick = multi_idx[(k + 1) % multi_idx.len()];
            }
            pick
        } else {
            let mut pick = unseen[rng.below(unseen.len())];
            if pick == a {
                pick = unseen[(k + 1) % unseen.len()];
            }
            pick
        };
        pairs.push((a, b));
    }
    pairs
}

fn run_dump_policies(
    dex: &Dex,
    team_map: &HashMap<String, Vec<PokemonSet>>,
    which: &str,
    cfg_tag: &str,
    threads: usize,
    out_path: &str,
) {
    let t0 = Instant::now();
    let matchups = pilot_matchups();
    let n_seeds = 8usize;
    let traj_cfg = make_suite_cfg(cfg_tag);
    let calib_cfg = make_calib_cfg(cfg_tag);
    let use_traj = which == "traj";
    let use_calib = which == "calib";

    let mut pilot_ms = Vec::new();
    let mut pilot_out = Vec::new();
    for (mi, ms) in matchups.iter().enumerate() {
        let sets_a = &team_map[ms.team_a];
        let sets_b = &team_map[ms.team_b];
        let pool_a = disclosed_pool(sets_b);
        let pool_b = disclosed_pool(sets_a);
        let jobs: Vec<(usize, usize)> = (0..n_seeds).flat_map(|s| [(0usize, s), (1usize, s)]).collect();
        let cur = AtomicUsize::new(0);
        let res: Vec<(usize, usize, Vec<([u8; 3], f64)>, f64, f64)> = std::thread::scope(|sc| {
            let mut hs = Vec::new();
            for _ in 0..threads.min(jobs.len()).max(1) {
                let (cr, jr, pa, pb, cfg_r, cal_r) =
                    (&cur, &jobs, &pool_a, &pool_b, &traj_cfg, &calib_cfg);
                hs.push(sc.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let k = cr.fetch_add(1, Ordering::Relaxed);
                        if k >= jr.len() {
                            break;
                        }
                        let (side, s) = jr[k];
                        let sa = 0x2026_7300_0001u64
                            ^ ((mi as u64) << 24)
                            ^ ((s as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let seed = if side == 0 { sa } else { sa ^ 0xABCD_EF01_1234_5678 };
                        let pool = if side == 0 { pa } else { pb };
                        if use_calib {
                            let (p, v, ms) =
                                run_calib_once(dex, sets_a, sets_b, pool, side, cal_r, 240_000, seed);
                            out.push((side, s, p, v, ms));
                        } else if use_traj {
                            let (p, v, _, _, ms, _) =
                                run_traj_once(dex, sets_a, sets_b, pool, side, cfg_r, seed);
                            out.push((side, s, p, v, ms));
                        } else {
                            let (p, ms) =
                                run_shipped_policy_v2(dex, sets_a, sets_b, pool, side, 240_000, seed);
                            out.push((side, s, p, 0.5, ms));
                        }
                    }
                    out
                }));
            }
            hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        let mut pols_a = vec![Vec::new(); n_seeds];
        let mut pols_b = vec![Vec::new(); n_seeds];
        let mut vals_a = vec![0.5; n_seeds];
        for (side, s, p, v, ms_t) in res {
            pilot_ms.push(ms_t);
            if side == 0 {
                pols_a[s] = p;
                vals_a[s] = v;
            } else {
                pols_b[s] = p;
            }
        }
        pilot_out.push(DumpedMatchupPols {
            code: ms.code.to_string(),
            team_a: ms.team_a.to_string(),
            team_b: ms.team_b.to_string(),
            pols_a,
            pols_b,
            vals_a,
        });
    }

    let root = repo_root();
    let bp_txt = std::fs::read_to_string(root.join("data/belief-pool-v3/belief-pool.json")).unwrap();
    let bp_cat: CatalogFile = serde_json::from_str(&bp_txt).unwrap();
    let full_pool = Arc::new(MetaPool {
        teams: bp_cat
            .teams
            .iter()
            .map(|t| MetaTeam {
                id: t.id.clone(),
                sets: t.sets.clone(),
                weight: t.weight,
            })
            .collect(),
    });
    let mw_pairs = select_unseen_mw_pairs(dex, &bp_cat, 20);
    let mw_jobs: Vec<(usize, usize)> = (0..mw_pairs.len()).flat_map(|m| [(m, 0usize), (m, 1usize)]).collect();
    let cur_m = AtomicUsize::new(0);
    let mw_res: Vec<(usize, usize, Vec<([u8; 3], f64)>, f64)> = std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(mw_jobs.len()).max(1) {
            let (cr, jr, mp_r, fp_r, cfg_r, cal_r, cat_r) =
                (&cur_m, &mw_jobs, &mw_pairs, &full_pool, &traj_cfg, &calib_cfg, &bp_cat);
            hs.push(sc.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let k = cr.fetch_add(1, Ordering::Relaxed);
                    if k >= jr.len() {
                        break;
                    }
                    let (mi, side) = jr[k];
                    let (ia, ib) = mp_r[mi];
                    let sa = 0x2026_7400_0001u64 ^ ((mi as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let seed = if side == 0 { sa } else { sa ^ 0x5555_AAAA_3333_CCCC };
                    let sets_a = &cat_r.teams[ia].sets;
                    let sets_b = &cat_r.teams[ib].sets;
                    if use_calib {
                        let (p, _, ms) =
                            run_calib_once(dex, sets_a, sets_b, fp_r, side, cal_r, 240_000, seed);
                        out.push((mi, side, p, ms));
                    } else if use_traj {
                        let (p, _, _, _, ms, _) =
                            run_traj_once(dex, sets_a, sets_b, fp_r, side, cfg_r, seed);
                        out.push((mi, side, p, ms));
                    } else {
                        let (p, ms) =
                            run_shipped_policy_v2(dex, sets_a, sets_b, fp_r, side, 240_000, seed);
                        out.push((mi, side, p, ms));
                    }
                }
                out
            }));
        }
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut mw_ms = Vec::new();
    let mut mw_a = vec![Vec::new(); mw_pairs.len()];
    let mut mw_b = vec![Vec::new(); mw_pairs.len()];
    for (mi, side, p, ms_t) in mw_res {
        mw_ms.push(ms_t);
        if side == 0 {
            mw_a[mi] = p;
        } else {
            mw_b[mi] = p;
        }
    }
    let mw_out: Vec<DumpedMwPols> = mw_pairs
        .iter()
        .enumerate()
        .map(|(mi, &(ia, ib))| DumpedMwPols {
            team_a: bp_cat.teams[ia].id.clone(),
            team_b: bp_cat.teams[ib].id.clone(),
            pol_a: mw_a[mi].clone(),
            pol_b: mw_b[mi].clone(),
        })
        .collect();

    let (p_med, _) = median_p95(&pilot_ms);
    let (m_med, _) = median_p95(&mw_ms);
    let doc = DumpedPoliciesFile {
        tag: format!("{which}:{cfg_tag}"),
        pilot: pilot_out,
        mw: mw_out,
        pilot_ms_med: r2(p_med),
        mw_ms_med: r2(m_med),
    };
    std::fs::write(out_path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    println!(
        "{}",
        json!({"mode":"dump_policies","tag":doc.tag,"pilot":doc.pilot.len(),"mw":doc.mw.len(),"wall_s":r2(t0.elapsed().as_secs_f64()),"out":out_path})
    );
}

#[allow(clippy::too_many_arguments)]
fn run_cert_shard(
    dex: &Dex,
    team_map: &HashMap<String, Vec<PokemonSet>>,
    pol_a_path: &str,
    pol_b_path: &str,
    battle_iters: u32,
    h2h_pairs: usize,
    expl_games: usize,
    include_mw: bool,
    mw_pairs: usize,
    shard: usize,
    shards: usize,
    threads: usize,
    out_path: &str,
) {
    let t0 = Instant::now();
    let pa_doc: DumpedPoliciesFile =
        serde_json::from_str(&std::fs::read_to_string(pol_a_path).unwrap()).unwrap();
    let pb_doc: DumpedPoliciesFile =
        serde_json::from_str(&std::fs::read_to_string(pol_b_path).unwrap()).unwrap();
    let matchups = pilot_matchups();

    let root = repo_root();
    let bp_txt = std::fs::read_to_string(root.join("data/belief-pool-v3/belief-pool.json")).unwrap();
    let bp_cat: CatalogFile = serde_json::from_str(&bp_txt).unwrap();
    let bp_map: HashMap<String, Vec<PokemonSet>> =
        bp_cat.teams.iter().map(|t| (t.id.clone(), t.sets.clone())).collect();
    let full_pool = Arc::new(MetaPool {
        teams: bp_cat
            .teams
            .iter()
            .map(|t| MetaTeam {
                id: t.id.clone(),
                sets: t.sets.clone(),
                weight: t.weight,
            })
            .collect(),
    });

    // Certified killer comps + top screened challengers across all 6 contexts.
    // Tuple: (matchup_idx, def_side, atk_tag, atk_triple)
    let expl_arms: Vec<(usize, usize, &'static str, [u8; 3])> = vec![
        // m1 s0 defending (pb84 defending vs mjj)
        (0, 0, "mjj_VenuTtarLax", [3, 2, 6]),
        (0, 0, "mjj_SteelixZapCloy", [4, 1, 5]),
        (0, 0, "mjj_ZapSteelixCloy", [1, 4, 5]),
        // m1 s1 defending (mjj defending vs pb84)
        (0, 1, "pb84_RaikCloySkarm", [2, 4, 6]),
        (0, 1, "pb84_GligRaikCloy", [5, 2, 4]),
        (0, 1, "pb84_RaikChampGlig", [2, 3, 5]),
        // m2 s0 defending (sample-07 defending vs mjj)
        (1, 0, "mjj_SteelixTtarCloy", [4, 2, 5]),
        (1, 0, "mjj_VenuTtarSteelix", [3, 2, 4]),
        (1, 0, "mjj_CloyTtarSteelix", [5, 2, 4]),
        // m2 s1 defending (mjj defending vs sample-07)
        (1, 1, "s07_ElecTaurosLax", [1, 2, 5]),
        (1, 1, "s07_ElecCloyWak", [1, 3, 4]),
        // m3 s0 defending (pb63 defending vs pb-6)
        (2, 0, "pb6_ZapLaxCloy", [1, 3, 5]),
        (2, 0, "pb6_GarZapLax", [6, 1, 3]),
        // m3 s1 defending (pb-6 defending vs pb63)
        (2, 1, "pb63_ZapGarHera", [6, 2, 5]),
        (2, 1, "pb63_ZapLaxBliss", [6, 1, 3]),
    ];

    #[derive(Clone)]
    enum WorkItem {
        PilotH2h { mi: usize, pair: usize },
        Expl { arm_idx: usize, game: usize },
        MwH2h { mi: usize, pair: usize },
    }
    let mut work = Vec::new();
    for mi in 0..matchups.len() {
        for pair in 0..h2h_pairs {
            work.push(WorkItem::PilotH2h { mi, pair });
        }
    }
    for arm_idx in 0..expl_arms.len() {
        for game in 0..expl_games {
            work.push(WorkItem::Expl { arm_idx, game });
        }
    }
    if include_mw {
        for mi in 0..pa_doc.mw.len() {
            for pair in 0..mw_pairs {
                work.push(WorkItem::MwH2h { mi, pair });
            }
        }
    }

    let my_items: Vec<(usize, WorkItem)> = work
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| idx % shards.max(1) == shard)
        .collect();

    let pilot_pools: Vec<(Arc<MetaPool>, Arc<MetaPool>)> = matchups
        .iter()
        .map(|ms| (disclosed_pool(&team_map[ms.team_b]), disclosed_pool(&team_map[ms.team_a])))
        .collect();
    let avg_a_s0: Vec<Vec<([u8; 3], f64)>> = pa_doc.pilot.iter().map(|m| avg_policy_dist(&m.pols_a)).collect();
    let avg_a_s1: Vec<Vec<([u8; 3], f64)>> = pa_doc.pilot.iter().map(|m| avg_policy_dist(&m.pols_b)).collect();
    let avg_b_s0: Vec<Vec<([u8; 3], f64)>> = pb_doc.pilot.iter().map(|m| avg_policy_dist(&m.pols_a)).collect();
    let avg_b_s1: Vec<Vec<([u8; 3], f64)>> = pb_doc.pilot.iter().map(|m| avg_policy_dist(&m.pols_b)).collect();

    let cur = AtomicUsize::new(0);
    let rows: Vec<Value> = std::thread::scope(|sc| {
        let mut hs = Vec::new();
        for _ in 0..threads.min(my_items.len()).max(1) {
            let (cr, ir, ms_r, pp_r, pa_r, pb_r, ea_r, a0_r, a1_r, b0_r, b1_r, bpm_r, fp_r) = (
                &cur, &my_items, &matchups, &pilot_pools, &pa_doc, &pb_doc, &expl_arms,
                &avg_a_s0, &avg_a_s1, &avg_b_s0, &avg_b_s1, &bp_map, &full_pool,
            );
            hs.push(sc.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let k = cr.fetch_add(1, Ordering::Relaxed);
                    if k >= ir.len() {
                        break;
                    }
                    let (widx, ref item) = ir[k];
                    match *item {
                        WorkItem::PilotH2h { mi, pair } => {
                            let ms = &ms_r[mi];
                            let sets_a = &team_map[ms.team_a];
                            let sets_b = &team_map[ms.team_b];
                            let (ref pool_a, ref pool_b) = pp_r[mi];
                            let mut drng = SplitMix64::new(
                                0x2026_9900_0001u64
                                    ^ ((mi as u64) << 28)
                                    ^ ((pair as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                            );
                            let bseed = drng.next();
                            let as0 = drng.next();
                            let as1 = drng.next();
                            let si = pair % pa_r.pilot[mi].pols_a.len();
                            let da = drng.next_f64();
                            let db = drng.next_f64();
                            let p_aa = sample_pick(&pa_r.pilot[mi].pols_a[si], da);
                            let p_bb = sample_pick(&pb_r.pilot[mi].pols_b[si], db);
                            let g0 = play_blind_battle_with_picks(
                                dex, sets_a, sets_b, Arc::clone(pool_a), Arc::clone(pool_b),
                                p_aa, p_bb, battle_iters, bseed, as0, as1,
                            );
                            let p_ba = sample_pick(&pb_r.pilot[mi].pols_a[si], da);
                            let p_ab = sample_pick(&pa_r.pilot[mi].pols_b[si], db);
                            let g1_s0 = play_blind_battle_with_picks(
                                dex, sets_a, sets_b, Arc::clone(pool_a), Arc::clone(pool_b),
                                p_ba, p_ab, battle_iters, bseed, as0, as1,
                            );
                            out.push(json!({
                                "kind": "pilot_h2h", "widx": widx, "code": ms.code, "pair": pair,
                                "a_s0": g0, "a_s1": 1.0 - g1_s0, "pair_diff": 0.5 * (g0 + (1.0 - g1_s0)) - 0.5
                            }));
                        }
                        WorkItem::Expl { arm_idx, game } => {
                            let (mi, def_side, atk_tag, atk_pick) = ea_r[arm_idx];
                            let ms = &ms_r[mi];
                            let sets_a = &team_map[ms.team_a];
                            let sets_b = &team_map[ms.team_b];
                            let (ref pool_a, ref pool_b) = pp_r[mi];
                            let mut grng = SplitMix64::new(
                                0x2026_8800_0001u64
                                    ^ ((arm_idx as u64) << 24)
                                    ^ ((game as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                            );
                            let bseed = grng.next();
                            let as0 = grng.next();
                            let as1 = grng.next();
                            let draw = grng.next_f64();
                            let def_a = if def_side == 0 { &a0_r[mi] } else { &a1_r[mi] };
                            let def_b = if def_side == 0 { &b0_r[mi] } else { &b1_r[mi] };
                            let pick_def_a = sample_pick(def_a, draw);
                            let pick_def_b = sample_pick(def_b, draw);
                            let (p0_a, p1_a) = if def_side == 0 { (pick_def_a, atk_pick) } else { (atk_pick, pick_def_a) };
                            let (p0_b, p1_b) = if def_side == 0 { (pick_def_b, atk_pick) } else { (atk_pick, pick_def_b) };
                            let v0_a = play_blind_battle_with_picks(
                                dex, sets_a, sets_b, Arc::clone(pool_a), Arc::clone(pool_b),
                                p0_a, p1_a, battle_iters, bseed, as0, as1,
                            );
                            let v0_b = play_blind_battle_with_picks(
                                dex, sets_a, sets_b, Arc::clone(pool_a), Arc::clone(pool_b),
                                p0_b, p1_b, battle_iters, bseed, as0, as1,
                            );
                            let atk_vs_a = if def_side == 0 { 1.0 - v0_a } else { v0_a };
                            let atk_vs_b = if def_side == 0 { 1.0 - v0_b } else { v0_b };
                            out.push(json!({
                                "kind": "expl", "widx": widx, "code": ms.code, "def_side": def_side,
                                "atk_tag": atk_tag, "true_atk_nash": if def_side == 0 { 1.0 - ms.true_nash_a } else { ms.true_nash_a },
                                "game": game, "atk_vs_a": atk_vs_a, "atk_vs_b": atk_vs_b
                            }));
                        }
                        WorkItem::MwH2h { mi, pair } => {
                            let ma = &pa_r.mw[mi];
                            let mb = &pb_r.mw[mi];
                            let sets_a = &bpm_r[&ma.team_a];
                            let sets_b = &bpm_r[&ma.team_b];
                            let mut drng = SplitMix64::new(
                                0x2026_7700_0001u64
                                    ^ ((mi as u64) << 24)
                                    ^ ((pair as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
                            );
                            let bseed = drng.next();
                            let as0 = drng.next();
                            let as1 = drng.next();
                            let da = drng.next_f64();
                            let db = drng.next_f64();
                            let p_aa = sample_pick(&ma.pol_a, da);
                            let p_bb = sample_pick(&mb.pol_b, db);
                            let g0 = play_blind_battle_with_picks(
                                dex, sets_a, sets_b, Arc::clone(fp_r), Arc::clone(fp_r),
                                p_aa, p_bb, battle_iters, bseed, as0, as1,
                            );
                            let p_ba = sample_pick(&mb.pol_a, da);
                            let p_ab = sample_pick(&ma.pol_b, db);
                            let g1_s0 = play_blind_battle_with_picks(
                                dex, sets_a, sets_b, Arc::clone(fp_r), Arc::clone(fp_r),
                                p_ba, p_ab, battle_iters, bseed, as0, as1,
                            );
                            out.push(json!({
                                "kind": "mw_h2h", "widx": widx, "mi": mi, "pair": pair,
                                "a_s0": g0, "a_s1": 1.0 - g1_s0, "pair_diff": 0.5 * (g0 + (1.0 - g1_s0)) - 0.5
                            }));
                        }
                    }
                }
                out
            }));
        }
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });

    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    for r in &rows {
        serde_json::to_writer(&mut f, r).unwrap();
        writeln!(f).unwrap();
    }
    f.flush().unwrap();
    println!(
        "{}",
        json!({"mode":"cert_shard","shard":shard,"shards":shards,"items":rows.len(),"iters":battle_iters,"wall_s":r2(t0.elapsed().as_secs_f64())})
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| -> Option<String> {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let mode = flag("--mode").unwrap_or_else(|| "a1-rb".to_string());
    let threads: usize = flag("--threads").map_or(12, |v| v.parse().unwrap());
    let cfg_tag = flag("--cfg").unwrap_or_else(|| "wasm180k".to_string());
    let h2h_pairs: usize = flag("--h2h-pairs").map_or(48, |v| v.parse().unwrap());
    let screen_games: usize = flag("--screen-games").map_or(8, |v| v.parse().unwrap());
    let exploit_9k_games: usize = flag("--exploit-games").map_or(48, |v| v.parse().unwrap());
    let skip_v2_exploit = args.iter().any(|a| a == "--skip-v2-exploit");
    let mw_matchups: usize = flag("--matchups").map_or(20, |v| v.parse().unwrap());
    let mw_pairs: usize = flag("--mw-pairs").map_or(12, |v| v.parse().unwrap());

    let root = repo_root();
    let dex = conformance::load_dex();
    let cat_txt =
        std::fs::read_to_string(root.join("data/team-pool-v4/environment-pool-32.json")).unwrap();
    let cat: CatalogFile = serde_json::from_str(&cat_txt).unwrap();
    let team_map: HashMap<String, Vec<PokemonSet>> =
        cat.teams.into_iter().map(|t| (t.id, t.sets)).collect();

    match mode.as_str() {
        "a1-rb" => run_a1_rb(&dex, &team_map, threads),
        "a234-ablate" => run_a234_ablate(&dex, &team_map, threads),
        "a3-focus" => run_a3_focus(&dex, &team_map, threads),
        "suite1-blind9k" => run_suite1_blind9k(
            &dex,
            &team_map,
            threads,
            &cfg_tag,
            h2h_pairs,
            screen_games,
            exploit_9k_games,
            skip_v2_exploit,
        ),
        "suite2-multiworld" => {
            run_suite2_multiworld(&dex, threads, &cfg_tag, mw_matchups, mw_pairs)
        }
        "dump-policies" => {
            let which = flag("--which").unwrap_or_else(|| "traj".to_string());
            let out = flag("--out").unwrap_or_else(|| "/tmp/dumped_pols.json".to_string());
            run_dump_policies(&dex, &team_map, &which, &cfg_tag, threads, &out);
        }
        "cert-shard" => {
            let pol_a = flag("--pol-a").unwrap();
            let pol_b = flag("--pol-b").unwrap();
            let battle_iters: u32 = flag("--iters").map_or(9000, |v| v.parse().unwrap());
            let expl_games: usize = flag("--expl-games").map_or(32, |v| v.parse().unwrap());
            let include_mw = args.iter().any(|a| a == "--include-mw");
            let shard: usize = flag("--shard").map_or(0, |v| v.parse().unwrap());
            let shards: usize = flag("--shards").map_or(1, |v| v.parse().unwrap());
            let out = flag("--out").unwrap_or_else(|| "/tmp/cert_shard.jsonl".to_string());
            run_cert_shard(
                &dex,
                &team_map,
                &pol_a,
                &pol_b,
                battle_iters,
                h2h_pairs,
                expl_games,
                include_mw,
                mw_pairs,
                shard,
                shards,
                threads,
                &out,
            );
        }
        other => panic!("unknown --mode {other}"),
    }
}
