//! External Equity Oracle Ceiling / Headroom Test (`equity_oracle_test`).
//!
//! Subcommands (all executed from a single compiled `--release` binary):
//! - `seam`      : Phase 0 — 0.0-ULP cache invariance, zero PRNG leakage, exact
//!                 side-swap antisymmetry, CRN null control (`EchoBase` vs `Base`
//!                 in `Open` & `Blind`), and multi-thread determinism check.
//! - `calibrate` : Phase 1 — Equity accuracy on the 72 certified M17e exact
//!                 anchors (`tmp/m17e-merged-v4.json`) + stratified midgame
//!                 positions (`standard_3v3`, `boosted_bp`, `recovery_stall`,
//!                 `endgame_2v2`) against both deep full-engine `SkuctSearch`
//!                 and terminal full-engine `OpenAgent` game outcomes.
//! - `duel`      : Phase 2 — CRN-paired side-swapped win-rate duels (`run_duel`
//!                 and `bp_gauntlet`) across `open` and `blind` (`c = 0.4`,
//!                 `team-pool-v2` + `belief-pool-v3`), outer budgets `N`, and
//!                 strata (`full`, `stall` `--heal-min 4`, `bp_h2h`, `bp_gauntlet`).

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::accounting::{
    acct_stats, clear_acct_thread_caches, compact_accounting_equity, evaluate_settled_leaf,
    extract_acct_components, get_active_acct_model, load_acct_model_from_file, reset_acct_stats,
    sample_1ply_settled_child, LeafSettlementMode, ACCT_BASIS_DIM, ACCT_FEAT_DIM,
};
use nc2000_bot::agent::Agent;
use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, reconstruct};
use nc2000_bot::duel::{run_duel, DuelSpec};
use nc2000_bot::eval::{compact_eval01, compact_eval_leaf, EvalWeights};
use nc2000_bot::m17e_artifact::MergedArtifact;
use nc2000_bot::mcts::Playout;
use nc2000_bot::oracle::{
    canonical_state_key_128, clear_oracle_thread_caches, oracle_equity, oracle_stats,
    reset_oracle_stats, swap_compact_sides, OracleMode,
};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_bot::smmcts::{
    dominated_actions, dominated_actions_with, MaskRules, RmAgent, RmConfig, SelRule, SkuctSearch,
};
use nc2000_bot::{Belief, BlindAgent, BlindSearch, Observer, OpenAgent};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::compact::{CompactBattle, MatchTable};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, RequestKind};
use serde_json::json;

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(10)
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn flag_usize(args: &[String], name: &str, default: usize) -> usize {
    flag(args, name)
        .map(|v| v.parse().unwrap_or_else(|e| panic!("bad {name}={v}: {e}")))
        .unwrap_or(default)
}

fn flag_u32(args: &[String], name: &str, default: u32) -> u32 {
    flag(args, name)
        .map(|v| v.parse().unwrap_or_else(|e| panic!("bad {name}={v}: {e}")))
        .unwrap_or(default)
}

fn flag_u64(args: &[String], name: &str, default: u64) -> u64 {
    flag(args, name)
        .map(|v| v.parse().unwrap_or_else(|e| panic!("bad {name}={v}: {e}")))
        .unwrap_or(default)
}

fn flag_f64(args: &[String], name: &str, default: f64) -> f64 {
    flag(args, name)
        .map(|v| v.parse().unwrap_or_else(|e| panic!("bad {name}={v}: {e}")))
        .unwrap_or(default)
}

fn flag_bool(args: &[String], name: &str, default: bool) -> bool {
    match flag(args, name).as_deref() {
        None => default,
        Some("true" | "1" | "yes") => true,
        Some("false" | "0" | "no") => false,
        Some(other) => panic!("bad bool {name}={other}"),
    }
}

#[derive(Clone, Copy, Debug)]
struct ArmSpec {
    oracle: OracleMode,
    settlement: LeafSettlementMode,
    iters_override: Option<u32>,
    extended_noops: bool,
}

fn parse_arm_spec(raw: &str) -> ArmSpec {
    let (spec, iters_override) = match raw.rsplit_once('@') {
        Some((lhs, rhs)) => {
            let n: u32 = rhs.parse().unwrap_or_else(|e| panic!("bad @iters in `{raw}`: {e}"));
            (lhs, Some(n.max(1)))
        }
        None => (raw, None),
    };
    if matches!(spec, "off" | "base" | "legacy") {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::LegacyRollout,
            iters_override,
            extended_noops: false,
        };
    }
    if spec == "oldmask" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: RmConfig::default().settlement,
            iters_override,
            extended_noops: false,
        };
    }
    if spec == "shipped" || spec == "default" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: RmConfig::default().settlement,
            iters_override,
            extended_noops: true,
        };
    }
    if spec == "echo" {
        return ArmSpec {
            oracle: OracleMode::EchoBase,
            settlement: LeafSettlementMode::LegacyRollout,
            iters_override,
            extended_noops: true,
        };
    }
    if spec == "rollacct0" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::RolloutAcct0,
            iters_override,
            extended_noops: true,
        };
    }
    if spec == "bsleaf" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::BlindSpotLeaf,
            iters_override,
            extended_noops: true,
        };
    }
    if spec == "bsroll" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::BlindSpotRollout,
            iters_override,
            extended_noops: true,
        };
    }
    if spec == "bsboth" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::BlindSpotBoth,
            iters_override,
            extended_noops: true,
        };
    }
    if spec == "acct0" {
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::Accounting0,
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("mm1:") {
        let max_k: u8 = rest.parse().expect("mm1:<max_k>");
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::Minimax1 { max_k },
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("amm1:") {
        let max_k: u8 = rest.parse().expect("amm1:<max_k>");
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::AdaptiveMinimax1 { max_k },
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("qh1:") {
        let max_k: u8 = rest.parse().expect("qh1:<max_k>");
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::QuiesceHot1 { max_k },
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("rmm:") {
        let (sk, st) = rest.split_once(',').unwrap_or((rest, "2"));
        let max_k: u8 = sk.parse().expect("rmm:<max_k>,<turns>");
        let rollout_turns: u8 = st.parse().expect("rmm:<max_k>,<turns>");
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::RolloutMinimax {
                max_k,
                rollout_turns,
            },
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("saroll:") {
        let turns: u8 = rest.parse().expect("saroll:<turns>");
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::StochAcctRollout { turns },
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("q2:") {
        let (s1, s2) = rest.split_once(',').unwrap_or((rest, "2"));
        let k1: u8 = s1.parse().expect("q2:<k1>,<k2>");
        let k2: u8 = s2.parse().expect("q2:<k1>,<k2>");
        return ArmSpec {
            oracle: OracleMode::Off,
            settlement: LeafSettlementMode::Quiesce2 { k1, k2 },
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("cutoff:") {
        let leaf_iters: u32 = rest.parse().expect("cutoff:<leaf_iters>");
        return ArmSpec {
            oracle: OracleMode::Cutoff { leaf_iters },
            settlement: LeafSettlementMode::LegacyRollout,
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("treeleaf:") {
        let leaf_iters: u32 = rest.parse().expect("treeleaf:<leaf_iters>");
        return ArmSpec {
            oracle: OracleMode::TreeLeaf { leaf_iters },
            settlement: LeafSettlementMode::LegacyRollout,
            iters_override,
            extended_noops: true,
        };
    }
    if let Some(rest) = spec.strip_prefix("coupled:") {
        let (r_str, l_str) = rest.split_once(',').unwrap_or((rest, rest));
        let rollout_iters: u32 = r_str.parse().expect("coupled:<rollout_iters>,<leaf_iters>");
        let leaf_iters: u32 = l_str.parse().expect("coupled:<rollout_iters>,<leaf_iters>");
        return ArmSpec {
            oracle: OracleMode::Coupled {
                rollout_iters,
                leaf_iters,
            },
            settlement: LeafSettlementMode::LegacyRollout,
            iters_override,
            extended_noops: true,
        };
    }
    panic!(
        "unknown arm spec `{raw}` (expected legacy|oldmask|shipped|off|echo|acct0|mm1:K|q2:K1,K2|cutoff:M|treeleaf:M|coupled:Mr,Ml[@iters])"
    );
}

fn maybe_load_acct_json(args: &[String]) {
    if let Some(path) = flag(args, "--acct-json") {
        load_acct_model_from_file(Path::new(&path))
            .unwrap_or_else(|e| panic!("failed to load --acct-json `{path}`: {e}"));
        let m = get_active_acct_model();
        println!(
            "Loaded AcctModel from `{path}` (residual_active={}, residual_scale={:.3}, w_acct[0..4]=[{:.3},{:.3},{:.3},{:.3}])",
            m.residual_active, m.residual_scale, m.w_acct[0], m.w_acct[1], m.w_acct[2], m.w_acct[3]
        );
    }
}

fn norm_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn team_signature(team: &[PokemonSet]) -> String {
    let mut parts: Vec<String> = team
        .iter()
        .map(|p| {
            let mut mvs: Vec<String> = p.moves.iter().map(|m| norm_id(m)).collect();
            mvs.sort();
            format!(
                "{}:{}:{}:{}",
                norm_id(&p.species),
                p.level,
                norm_id(&p.item),
                mvs.join(",")
            )
        })
        .collect();
    parts.sort();
    parts.join("|")
}

fn carries_count(sets: &[PokemonSet], keys: &[&str]) -> usize {
    sets.iter()
        .filter(|s| {
            s.moves
                .iter()
                .any(|m| keys.contains(&norm_id(m).as_str()))
        })
        .count()
}

fn is_baton_pass_mon(p: &PokemonSet) -> bool {
    p.moves.iter().any(|m| norm_id(m) == "batonpass")
}

fn is_setup_mon(p: &PokemonSet) -> bool {
    p.moves.iter().any(|m| {
        matches!(
            norm_id(m).as_str(),
            "batonpass" | "swordsdance" | "bellydrum" | "agility" | "amnesia" | "growth" | "curse"
        )
    })
}

fn setup_lead_slot_1based(team: &[PokemonSet]) -> Option<u8> {
    if let Some(i) = team.iter().position(is_baton_pass_mon) {
        return Some(i as u8 + 1);
    }
    team.iter().position(is_setup_mon).map(|i| i as u8 + 1)
}

fn build_rm_config(iters: u32, c: f64, oracle: OracleMode) -> RmConfig {
    let playout = Playout::Heavy {
        eps: 0.2,
        turns: 8,
        weights: EvalWeights {
            residual_scale: 0.0,
            ..EvalWeights::default()
        },
    };
    RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c,
        hp_buckets: 16,
        playout,
        rollout_perish: false,
        rollout_combo: true,
        rollout_status: false,
        rollout_switch: false,
        rollout_baton_pass: true,
        oracle,
        settlement: LeafSettlementMode::LegacyRollout,
        ..Default::default()
    }
}

fn build_rm_config_arm(default_iters: u32, c: f64, spec: ArmSpec) -> RmConfig {
    let mut cfg = build_rm_config(spec.iters_override.unwrap_or(default_iters), c, spec.oracle);
    cfg.settlement = spec.settlement;
    cfg.mask_rules.extended_noops = spec.extended_noops;
    cfg
}

fn build_agent(
    is_blind: bool,
    iters: u32,
    c: f64,
    oracle: OracleMode,
    belief_pool: &Arc<MetaPool>,
    seed: u64,
) -> Box<dyn Agent> {
    let cfg = build_rm_config(iters, c, oracle);
    if is_blind {
        Box::new(BlindAgent::new(cfg, belief_pool.clone(), None, seed))
    } else {
        Box::new(OpenAgent::new(cfg, None, seed))
    }
}

fn build_agent_arm(
    is_blind: bool,
    default_iters: u32,
    c: f64,
    spec: ArmSpec,
    belief_pool: &Arc<MetaPool>,
    seed: u64,
) -> Box<dyn Agent> {
    let cfg = build_rm_config_arm(default_iters, c, spec);
    if is_blind {
        Box::new(BlindAgent::new(cfg, belief_pool.clone(), None, seed))
    } else {
        Box::new(OpenAgent::new(cfg, None, seed))
    }
}

fn load_dedup_teams(root: &Path, rel_paths: &[&str]) -> Vec<Vec<PokemonSet>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for rel in rel_paths {
        let pool = load_meta_pool(&root.join(rel));
        for t in pool.teams {
            let sig = team_signature(&t.sets);
            if seen.insert(sig) {
                out.push(t.sets);
            }
        }
    }
    out
}

// =========================================================================
// Phase 0: Seam, Cache Bit-Exactness, Antisymmetry & Determinism Gate
// =========================================================================

fn run_seam(args: &[String]) {
    let root = repo_root();
    let dex = load_dex();
    let threads = flag_usize(args, "--threads", default_threads());
    let games = flag_usize(args, "--games", 20);
    let iters = flag_u32(args, "--iters", 300);

    let teams = load_dedup_teams(&root, &["data/team-pool-v2/team-pool.json"]);
    let belief_pool = Arc::new(load_meta_pool(
        &root.join("data/belief-pool-v3/belief-pool.json"),
    ));

    println!("=== Phase 0A: 60-State Cache Bit-Exactness (0.0 ULP), Zero PRNG Leakage & Side-Swap Antisymmetry ===");
    let mut rng = SplitMix64::new(0x5EA4_2026_0001);
    let mut tested = 0usize;
    let mut max_antisym_err = 0.0f64;
    let mut g_idx = 0usize;
    while tested < 60 {
        let i0 = rng.below(teams.len());
        let mut i1 = rng.below(teams.len());
        if i1 == i0 {
            i1 = (i0 + 1) % teams.len();
        }
        let mut b = Battle::from_fixture(&dex, &rng.battle_seed(), &teams[i0], &teams[i1]).unwrap();
        b.set_log_enabled(false);
        let mut a0 = MaxPreviewRand::new(0x1000 + g_idx as u64);
        let mut a1 = MaxPreviewRand::new(0x2000 + g_idx as u64);
        let target_turn = 2 + (rng.below(10) as u16);
        while b.outcome().is_none() && b.turn < target_turn {
            let l0 = b.legal_choices(&dex, 0);
            let l1 = b.legal_choices(&dex, 1);
            let c0 = (!l0.is_empty()).then(|| a0.pick(&b, &dex, 0, &l0));
            let c1 = (!l1.is_empty()).then(|| a1.pick(&b, &dex, 1, &l1));
            if b.apply_choices(&dex, [c0, c1]).is_err() {
                break;
            }
        }
        g_idx += 1;
        if b.outcome().is_some()
            || b.sides[0].request_state() == Some(RequestKind::TeamPreview)
            || b.sides[1].request_state() == Some(RequestKind::TeamPreview)
        {
            continue;
        }
        let (table, cb) = CompactBattle::from_battle(&b, &dex);
        clear_oracle_thread_caches();
        let cold = oracle_equity(&cb, &table, 64);
        let warm = oracle_equity(&cb, &table, 64);
        assert_eq!(
            cold.to_bits(),
            warm.to_bits(),
            "Cache hit != cache miss at state {tested}"
        );

        let mut cb_rng = cb;
        cb_rng.prng_seed = cb_rng.prng_seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        cb_rng.quick_claw_roll ^= 1;
        clear_oracle_thread_caches();
        let diff_rng = oracle_equity(&cb_rng, &table, 64);
        assert_eq!(
            cold.to_bits(),
            diff_rng.to_bits(),
            "PRNG leakage detected at state {tested}"
        );

        clear_oracle_thread_caches();
        let (scb, st) = swap_compact_sides(&cb, &table);
        let swapped_eq = oracle_equity(&scb, &st, 64);
        let err = (cold + swapped_eq - 1.0).abs();
        if err > max_antisym_err {
            max_antisym_err = err;
        }
        assert!(
            err < 1e-6,
            "Side-swap antisymmetry violated at state {tested}: {cold} + {swapped_eq} - 1 = {err}"
        );
        tested += 1;
    }
    println!(
        "  PASS: {tested} states verified | Cold==Warm: 0.0 ULP | PRNG leak: 0.0 ULP | max |E(s)+E(swap(s))-1| = {max_antisym_err:.2e}"
    );

    println!("\n=== Phase 0B: CRN Null-Control Duels (`EchoBase` vs `Base`, crn_agent_seeds=true) ===");
    for (env_name, is_blind, c_val) in [("open", false, 1.0), ("blind", true, 0.4)] {
        reset_oracle_stats();
        let mut spec = DuelSpec::new(games, 20261029);
        spec.threads = threads;
        spec.log_on = is_blind;
        spec.crn_agent_seeds = true;

        let stats = run_duel(
            &dex,
            &teams,
            &|seed| build_agent(is_blind, iters, c_val, OracleMode::EchoBase, &belief_pool, seed),
            &|seed| build_agent(is_blind, iters, c_val, OracleMode::Off, &belief_pool, seed),
            spec,
        );
        let ostats = oracle_stats();
        let split_pairs = stats
            .pair_scores
            .iter()
            .filter(|&&p| (p - 0.5).abs() > 1e-12)
            .count();
        println!(
            "  [{env_name:5} N={iters}] score = {:.6} +/- {:.6} | pairs={} split_pairs={} | echo_eval_calls={} | {:.2}s",
            stats.score, stats.ci95, stats.pairs, split_pairs, ostats.eval_calls, stats.secs
        );
        assert!(
            (stats.score - 0.5).abs() < 1e-12 && split_pairs == 0,
            "Null control failed for {env_name}: score={}, split_pairs={split_pairs}",
            stats.score
        );
        assert!(
            ostats.eval_calls > 0,
            "EchoBase seam did not fire in {env_name}!"
        );
    }

    println!("\n=== Phase 0C: Multi-Thread Determinism Gate (`threads=1` vs `threads={threads}`) ===");
    let mut scores_by_threads = Vec::new();
    for th in [1usize, threads.max(2)] {
        let mut spec = DuelSpec::new(8, 77777);
        spec.threads = th;
        spec.log_on = true;
        spec.crn_agent_seeds = true;
        let st = run_duel(
            &dex,
            &teams,
            &|seed| {
                build_agent(
                    true,
                    120,
                    0.4,
                    OracleMode::Coupled {
                        rollout_iters: 32,
                        leaf_iters: 32,
                    },
                    &belief_pool,
                    seed,
                )
            },
            &|seed| build_agent(true, 120, 0.4, OracleMode::Off, &belief_pool, seed),
            spec,
        );
        scores_by_threads.push(st.game_scores);
    }
    assert_eq!(
        scores_by_threads[0], scores_by_threads[1],
        "Multi-thread scores diverged from single-thread scores!"
    );
    println!(
        "  PASS: `Oracle-Coupled` game_scores identical across 1 thread and {} threads: {:?}",
        threads.max(2),
        scores_by_threads[0]
    );
}

struct MaxPreviewRand {
    rng: SplitMix64,
}

impl MaxPreviewRand {
    fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }

    fn pick(&mut self, b: &Battle, dex: &Dex, side: usize, cs: &[SearchChoice]) -> SearchChoice {
        if matches!(cs[0], SearchChoice::Team(_)) {
            return cs[self.rng.below(cs.len())];
        }
        if self.rng.next_f64() < 0.35 {
            return cs[self.rng.below(cs.len())];
        }
        let (table, cb) = CompactBattle::from_battle(b, dex);
        let att_slot = cb.sides[side].active_idx.min(2) as usize;
        let def_side = 1 - side;
        let def_slot = cb.sides[def_side].active_idx.min(2) as usize;
        let mut best = (cs[0], -1.0f64);
        for &c in cs {
            if let SearchChoice::Move(mid) = c {
                let num_m = table.mons[side * 3 + att_slot].num_moves as usize;
                if let Some(mslot) =
                    (0..num_m).find(|&m| table.mons[side * 3 + att_slot].moves[m] == mid)
                {
                    let d = cb.expected_hit_fraction(
                        &table, side, att_slot, def_side, def_slot, mslot, true,
                    );
                    if d > best.1 {
                        best = (c, d);
                    }
                }
            }
        }
        if best.1 > 0.0 {
            best.0
        } else {
            cs[self.rng.below(cs.len())]
        }
    }
}

// =========================================================================
// Phase 1: Equity Accuracy Calibration Gate (M17e Exact + Midgame Slices)
// =========================================================================

#[derive(Clone, Copy)]
enum PosSlice {
    Standard3v3,
    BoostedBp,
    RecoveryStall,
    Endgame2v2,
}

impl PosSlice {
    fn name(self) -> &'static str {
        match self {
            PosSlice::Standard3v3 => "standard_3v3",
            PosSlice::BoostedBp => "boosted_bp",
            PosSlice::RecoveryStall => "recovery_stall",
            PosSlice::Endgame2v2 => "endgame_2v2",
        }
    }
}

struct CalibPos {
    battle: Battle,
    slice: PosSlice,
    v_skuct: f64,
    v_term: f64,
}

fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    if xs.len() < 2 {
        return 0.0;
    }
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        cov += (x - mx) * (y - my);
        vx += (x - mx).powi(2);
        vy += (y - my).powi(2);
    }
    if vx <= 1e-18 || vy <= 1e-18 {
        0.0
    } else {
        cov / (vx * vy).sqrt()
    }
}

fn rmse_mae_bias(preds: &[f64], targets: &[f64]) -> (f64, f64, f64) {
    let n = preds.len().max(1) as f64;
    let mut sq = 0.0;
    let mut ab = 0.0;
    let mut bi = 0.0;
    for (&p, &t) in preds.iter().zip(targets.iter()) {
        let d = p - t;
        sq += d * d;
        ab += d.abs();
        bi += d;
    }
    ((sq / n).sqrt(), ab / n, bi / n)
}

fn skuct_root_value(dex: &Dex, battle: &Battle, iters: u32, seed: u64) -> f64 {
    let cfg = build_rm_config(iters, 1.0, OracleMode::Off);
    let mut s = SkuctSearch::new(battle, dex, cfg, seed);
    s.step(dex, iters);
    let v0 = s.visits(0);
    let m0 = s.means(0);
    let v1 = s.visits(1);
    let m1 = s.means(1);
    if !v0.is_empty() {
        let tot: u32 = v0.iter().sum();
        if tot > 0 {
            return v0
                .iter()
                .zip(m0.iter())
                .map(|(&n, &m)| (n as f64) * m)
                .sum::<f64>()
                / (tot as f64);
        }
    }
    if !v1.is_empty() {
        let tot: u32 = v1.iter().sum();
        if tot > 0 {
            return 1.0
                - v1.iter()
                    .zip(m1.iter())
                    .map(|(&n, &m)| (n as f64) * m)
                    .sum::<f64>()
                    / (tot as f64);
        }
    }
    let (table, cb) = CompactBattle::from_battle(battle, dex);
    compact_eval01(&cb, &table, &EvalWeights::default())
}

fn terminal_playoff_value(
    dex: &Dex,
    battle: &Battle,
    playoff_iters: u32,
    n_games: usize,
    base_seed: u64,
) -> f64 {
    let mut sum = 0.0f64;
    for k in 0..n_games {
        let mut sim = battle.clone();
        sim.set_log_enabled(false);
        let mut sr = SplitMix64::new(base_seed ^ (k as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        sim.reseed(sr.next());
        let mut p0 = RmAgent::new(
            build_rm_config(playoff_iters, 1.0, OracleMode::Off),
            sr.next(),
        );
        let mut p1 = RmAgent::new(
            build_rm_config(playoff_iters, 1.0, OracleMode::Off),
            sr.next(),
        );
        let mut pair: [&mut dyn Agent; 2] = [&mut p0, &mut p1];
        let res = play_game(dex, &mut sim, &mut pair, 300).unwrap();
        sum += match res {
            GameResult::Outcome(Outcome::P1Win) => 1.0,
            GameResult::Outcome(Outcome::P2Win) => 0.0,
            GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
        };
    }
    sum / (n_games.max(1) as f64)
}

fn classify_slice(cb: &CompactBattle, table: &nc2000_engine::compact::MatchTable) -> PosSlice {
    let a0 = cb.sides[0].alive_mask.count_ones();
    let a1 = cb.sides[1].alive_mask.count_ones();
    let mut has_boost_or_bp = false;
    let mut has_damaged_healer = false;

    for side in 0..2usize {
        let act = cb.sides[side].active_idx.min(2) as usize;
        let act_idx = side * 3 + act;
        if (cb.sides[side].alive_mask & (1 << act)) != 0 && cb.mons[act_idx].hp() > 0 {
            let m = &cb.mons[act_idx];
            let any_pos_boost =
                m.boosts[0] > 0 || m.boosts[1] > 0 || m.boosts[2] > 0 || m.boosts[4] > 0;
            let has_bp = (0..table.mons[act_idx].num_moves as usize)
                .any(|k| table.move_specs[act_idx][k].special_kind == 18);
            if any_pos_boost || (has_bp && cb.sides[side].alive_mask.count_ones() >= 2) {
                has_boost_or_bp = true;
            }
        }
        for slot in 0..3usize {
            if (cb.sides[side].alive_mask & (1 << slot)) == 0 {
                continue;
            }
            let idx = side * 3 + slot;
            let m = &cb.mons[idx];
            let maxhp = table.mons[idx].maxhp.max(1);
            let has_heal = (0..table.mons[idx].num_moves as usize)
                .any(|k| table.move_specs[idx][k].is_heal_cycle);
            if has_heal && m.hp() > 0 && (m.hp() < (maxhp * 4) / 5 || m.status_raw() != 0) {
                has_damaged_healer = true;
            }
        }
    }

    if has_boost_or_bp {
        PosSlice::BoostedBp
    } else if a0 <= 2 && a1 <= 2 {
        PosSlice::Endgame2v2
    } else if has_damaged_healer {
        PosSlice::RecoveryStall
    } else {
        PosSlice::Standard3v3
    }
}

#[derive(Clone, Copy)]
enum CalibEvalKind {
    Base01,
    BaseLeaf,
    Acct0,
    Settled(LeafSettlementMode),
    Oracle(u32),
}

fn eval_calib_kind(
    kind: CalibEvalKind,
    cb: &CompactBattle,
    table: &MatchTable,
    base_w: &EvalWeights,
) -> f64 {
    match kind {
        CalibEvalKind::Base01 => compact_eval01(cb, table, base_w),
        CalibEvalKind::BaseLeaf => compact_eval_leaf(cb, table, base_w),
        CalibEvalKind::Acct0 => compact_accounting_equity(cb, table),
        CalibEvalKind::Settled(mode) => evaluate_settled_leaf(cb, table, mode),
        CalibEvalKind::Oracle(m) => oracle_equity(cb, table, m),
    }
}

fn run_calibrate(args: &[String]) {
    maybe_load_acct_json(args);
    let root = repo_root();
    let dex = load_dex();
    let threads = flag_usize(args, "--threads", default_threads());
    let target_per_slice = flag_usize(args, "--per-slice", 40);
    let teacher_iters = flag_u32(args, "--teacher-iters", 3000);
    let playoff_iters = flag_u32(args, "--playoff-iters", 150);
    let playoff_games = flag_usize(args, "--playoff-games", 8);

    let base_w = EvalWeights {
        residual_scale: 0.0,
        ..EvalWeights::default()
    };

    let eval_suite: Vec<(&'static str, CalibEvalKind)> = vec![
        ("Base-eval01", CalibEvalKind::Base01),
        ("Base-eval_leaf", CalibEvalKind::BaseLeaf),
        ("Acct0", CalibEvalKind::Acct0),
        (
            "QuiesceHot1(2)",
            CalibEvalKind::Settled(LeafSettlementMode::QuiesceHot1 { max_k: 2 }),
        ),
        (
            "AdaptiveMM1(2)",
            CalibEvalKind::Settled(LeafSettlementMode::AdaptiveMinimax1 { max_k: 2 }),
        ),
        (
            "RollMM(2,2)",
            CalibEvalKind::Settled(LeafSettlementMode::RolloutMinimax {
                max_k: 2,
                rollout_turns: 2,
            }),
        ),
        (
            "Minimax1(K=2)",
            CalibEvalKind::Settled(LeafSettlementMode::Minimax1 { max_k: 2 }),
        ),
        (
            "Minimax1(K=3)",
            CalibEvalKind::Settled(LeafSettlementMode::Minimax1 { max_k: 3 }),
        ),
        (
            "Minimax1(K=4)",
            CalibEvalKind::Settled(LeafSettlementMode::Minimax1 { max_k: 4 }),
        ),
        (
            "Quiesce2(3,2)",
            CalibEvalKind::Settled(LeafSettlementMode::Quiesce2 { k1: 3, k2: 2 }),
        ),
        ("Oracle(M=64)", CalibEvalKind::Oracle(64)),
        ("Oracle(M=128)", CalibEvalKind::Oracle(128)),
        ("Oracle(M=256)", CalibEvalKind::Oracle(256)),
    ];

    // ---- Part 1A: Certified M17e Exact Endgame Anchors
    let anchor_path = root.join("tmp/m17e-merged-v4.json");
    let corpus_dir = root.join("tmp/corpus-spectator");
    if anchor_path.exists() && corpus_dir.exists() {
        println!("=== Phase 1A: 72 Certified M17e Exact Endgame Anchors (`tmp/m17e-merged-v4.json`) ===");
        let artifact: MergedArtifact =
            serde_json::from_str(&std::fs::read_to_string(&anchor_path).unwrap()).unwrap();
        let src = load_sources(&dex, &root);
        let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
        let files = corpus_files(&corpus_dir);

        let mut by_battle: std::collections::HashMap<usize, Vec<&nc2000_bot::m17e_artifact::Row>> =
            std::collections::HashMap::new();
        for row in &artifact.rows {
            by_battle.entry(row.battle).or_default().push(row);
        }
        let mut b_ids: Vec<usize> = by_battle.keys().copied().collect();
        b_ids.sort();

        let mut anchors: Vec<(MatchTable, CompactBattle, f64, f64)> = Vec::new();
        for bi in b_ids {
            let cb_file = load_battle(&files[bi]);
            for a in &by_battle[&bi] {
                let d = &cb_file.decisions[a.decision];
                let b = reconstruct(
                    &dex,
                    &src,
                    &pool_path,
                    &cb_file.lines,
                    &cb_file.evidence,
                    d,
                    artifact.run.selection.reconstruction_seed,
                )
                .unwrap();
                let (table, cb) = CompactBattle::from_battle(&b, &dex);
                anchors.push((table, cb, a.exact, a.width));
            }
        }

        let exact_targets: Vec<f64> = anchors.iter().map(|x| x.2).collect();
        for &(label, kind) in &eval_suite {
            clear_oracle_thread_caches();
            clear_acct_thread_caches();
            let t0 = Instant::now();
            let mut preds = Vec::with_capacity(anchors.len());
            let mut viols = 0usize;
            let mut viol_margin = 0.0f64;
            for (table, cb, exact, width) in &anchors {
                let p = eval_calib_kind(kind, cb, table, &base_w);
                preds.push(p);
                let (lo, hi) = (exact - width / 2.0, exact + width / 2.0);
                let v = (p - hi).max(lo - p).max(0.0);
                if v > 0.02 {
                    viols += 1;
                    viol_margin += v;
                }
            }
            let us_per_eval = t0.elapsed().as_secs_f64() * 1e6 / (anchors.len().max(1) as f64);
            let (rmse, mae, bias) = rmse_mae_bias(&preds, &exact_targets);
            let r = pearson_r(&preds, &exact_targets);
            println!(
                "  {label:15} | viols={viols:2}/{} margin={viol_margin:.4} | RMSE={rmse:.4} MAE={mae:.4} bias={bias:+.4} r={r:.4} ({us_per_eval:6.2} us/eval)",
                anchors.len(),
            );
        }
    }

    // ---- Part 1B: Stratified Midgame Calibration Corpus
    println!(
        "\n=== Phase 1B: Stratified Midgame Calibration ({} per slice = {} positions) ===",
        target_per_slice,
        target_per_slice * 4
    );
    let teams = load_dedup_teams(
        &root,
        &[
            "data/team-pool-v2/team-pool.json",
            "data/belief-pool-v3/belief-pool.json",
            "data/eco-pool/master-pool.json",
        ],
    );
    let bp_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| teams[i].iter().any(is_setup_mon))
        .collect();

    let mut buckets: [Vec<Battle>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut gen_rng = SplitMix64::new(0xCA11_B202_6102_9001);
    let mut games_sim = 0usize;
    while buckets.iter().any(|b| b.len() < target_per_slice) && games_sim < 3000 {
        let want_bp = buckets[1].len() < target_per_slice && !bp_teams.is_empty() && games_sim % 2 == 0;
        let t0_idx = if want_bp {
            bp_teams[gen_rng.below(bp_teams.len())]
        } else {
            gen_rng.below(teams.len())
        };
        let mut t1_idx = gen_rng.below(teams.len());
        if t1_idx == t0_idx {
            t1_idx = (t0_idx + 1) % teams.len();
        }
        let mut b =
            Battle::from_fixture(&dex, &gen_rng.battle_seed(), &teams[t0_idx], &teams[t1_idx])
                .unwrap();
        b.set_log_enabled(false);

        // Advance TeamPreview (forcing setup lead when want_bp is true).
        let l0 = b.legal_choices(&dex, 0);
        let l1 = b.legal_choices(&dex, 1);
        let p0 = if want_bp {
            if let Some(lead) = setup_lead_slot_1based(&teams[t0_idx]) {
                l0.iter()
                    .copied()
                    .find(|c| matches!(c, SearchChoice::Team(u) if u[0] == lead))
                    .unwrap_or(l0[gen_rng.below(l0.len())])
            } else {
                l0[gen_rng.below(l0.len())]
            }
        } else {
            l0[gen_rng.below(l0.len())]
        };
        let p1 = l1[gen_rng.below(l1.len())];
        let _ = b.apply_choices(&dex, [Some(p0), Some(p1)]);

        let mut agent0 = RmAgent::new(build_rm_config(80, 1.0, OracleMode::Off), gen_rng.next());
        let mut agent1 = RmAgent::new(build_rm_config(80, 1.0, OracleMode::Off), gen_rng.next());
        let mut step_n = 0usize;
        while b.outcome().is_none() && b.turn <= 45 && step_n < 70 {
            let cs0 = b.legal_choices(&dex, 0);
            let cs1 = b.legal_choices(&dex, 1);
            // Inject occasional setup/Baton-Pass move on turn 1-4 when want_bp is true to sample boosted boards.
            let c0 = if cs0.is_empty() {
                None
            } else if want_bp && step_n < 3 && gen_rng.next_f64() < 0.65 {
                let tac = cs0.iter().copied().find(|c| match c {
                    SearchChoice::Move(mid) => matches!(
                        dex.moves.key(*mid),
                        "swordsdance" | "bellydrum" | "agility" | "curse" | "growth" | "amnesia"
                    ),
                    _ => false,
                });
                Some(tac.unwrap_or_else(|| agent0.choose(&b, &dex, 0, &cs0)))
            } else {
                Some(agent0.choose(&b, &dex, 0, &cs0))
            };
            let c1 = (!cs1.is_empty()).then(|| agent1.choose(&b, &dex, 1, &cs1));
            if b.apply_choices(&dex, [c0, c1]).is_err() {
                break;
            }
            step_n += 1;
            if b.outcome().is_none()
                && b.sides[0].request_state() == Some(RequestKind::Move)
                && b.sides[1].request_state() == Some(RequestKind::Move)
                && gen_rng.next_f64() < 0.35
            {
                let (table, cb) = CompactBattle::from_battle(&b, &dex);
                let sl = classify_slice(&cb, &table);
                let b_idx = match sl {
                    PosSlice::Standard3v3 => 0,
                    PosSlice::BoostedBp => 1,
                    PosSlice::RecoveryStall => 2,
                    PosSlice::Endgame2v2 => 3,
                };
                if buckets[b_idx].len() < target_per_slice {
                    buckets[b_idx].push(b.clone());
                }
            }
        }
        games_sim += 1;
    }

    let mut raw_list: Vec<(Battle, PosSlice)> = Vec::new();
    for (idx, sl) in [
        (0, PosSlice::Standard3v3),
        (1, PosSlice::BoostedBp),
        (2, PosSlice::RecoveryStall),
        (3, PosSlice::Endgame2v2),
    ] {
        println!("  Slice {:15}: {} positions", sl.name(), buckets[idx].len());
        for b in buckets[idx].drain(..) {
            raw_list.push((b, sl));
        }
    }

    let t_gt = Instant::now();
    let cursor = AtomicUsize::new(0);
    let calib_out: Mutex<Vec<(usize, CalibPos)>> = Mutex::new(Vec::with_capacity(raw_list.len()));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let i = cursor.fetch_add(1, Ordering::Relaxed);
                if i >= raw_list.len() {
                    break;
                }
                let (ref battle, slice) = raw_list[i];
                let s0 = 0x7EA_C4E2_0000_0000u64 ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let v_sk0 = skuct_root_value(&dex, battle, teacher_iters, s0);
                let v_sk1 = skuct_root_value(&dex, battle, teacher_iters, s0 ^ 0x5555_AAAA_5555_AAAA);
                let v_skuct = 0.5 * (v_sk0 + v_sk1);
                let v_term =
                    terminal_playoff_value(&dex, battle, playoff_iters, playoff_games, s0 ^ 0x1234_5678);
                calib_out.lock().unwrap().push((
                    i,
                    CalibPos {
                        battle: battle.clone(),
                        slice,
                        v_skuct,
                        v_term,
                    },
                ));
            });
        }
    });
    let mut indexed = calib_out.into_inner().unwrap();
    indexed.sort_by_key(|x| x.0);
    let positions: Vec<CalibPos> = indexed.into_iter().map(|x| x.1).collect();
    println!(
        "  Ground truth computed in {:.2}s (SkuctSearch({teacher_iters})x2 + OpenAgent({playoff_iters})x{playoff_games})",
        t_gt.elapsed().as_secs_f64()
    );

    let targets_sk: Vec<f64> = positions.iter().map(|p| p.v_skuct).collect();
    let targets_tm: Vec<f64> = positions.iter().map(|p| p.v_term).collect();
    let prebuilt: Vec<(MatchTable, CompactBattle)> = positions
        .iter()
        .map(|pos| CompactBattle::from_battle(&pos.battle, &dex))
        .collect();

    println!("\n  --- Overall Accuracy vs SkuctSearch({teacher_iters})x2 AND Terminal Playoff ---");
    for &(label, kind) in &eval_suite {
        clear_oracle_thread_caches();
        clear_acct_thread_caches();
        let t0 = Instant::now();
        let preds: Vec<f64> = prebuilt
            .iter()
            .map(|(table, cb)| eval_calib_kind(kind, cb, table, &base_w))
            .collect();
        let us_per_eval = t0.elapsed().as_secs_f64() * 1e6 / (prebuilt.len().max(1) as f64);
        let (rmse_sk, mae_sk, bias_sk) = rmse_mae_bias(&preds, &targets_sk);
        let r_sk = pearson_r(&preds, &targets_sk);
        let (rmse_tm, _mae_tm, _bias_tm) = rmse_mae_bias(&preds, &targets_tm);
        let r_tm = pearson_r(&preds, &targets_tm);

        let mut slice_rmses = [0.0f64; 4];
        for (s_i, sl_kind) in [
            PosSlice::Standard3v3,
            PosSlice::BoostedBp,
            PosSlice::RecoveryStall,
            PosSlice::Endgame2v2,
        ]
        .iter()
        .enumerate()
        {
            let mut sp = Vec::new();
            let mut st = Vec::new();
            for (pos, &pr) in positions.iter().zip(preds.iter()) {
                if pos.slice.name() == sl_kind.name() {
                    sp.push(pr);
                    st.push(pos.v_skuct);
                }
            }
            slice_rmses[s_i] = rmse_mae_bias(&sp, &st).0;
        }

        println!(
            "  {label:15} | vs Skuct: RMSE={rmse_sk:.4} MAE={mae_sk:.4} bias={bias_sk:+.4} r={r_sk:.4} | vs Term: RMSE={rmse_tm:.4} r={r_tm:.4} | slices[3v3={:.4}, bp={:.4}, stall={:.4}, 2v2={:.4}] ({us_per_eval:6.2} us/eval)",
            slice_rmses[0],
            slice_rmses[1],
            slice_rmses[2],
            slice_rmses[3],
        );
    }
}

// =========================================================================
// Strict-Search Training Dataset Generator (`gendata`)
// =========================================================================

fn collect_heldout_canonical_keys(root: &Path, dex: &Dex) -> HashSet<(u64, u64)> {
    let mut keys = HashSet::new();
    let anchor_path = root.join("tmp/m17e-merged-v4.json");
    let corpus_dir = root.join("tmp/corpus-spectator");
    if anchor_path.exists() && corpus_dir.exists() {
        let artifact: MergedArtifact =
            serde_json::from_str(&std::fs::read_to_string(&anchor_path).unwrap()).unwrap();
        let src = load_sources(dex, root);
        let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
        let files = corpus_files(&corpus_dir);
        let mut by_battle: std::collections::HashMap<usize, Vec<&nc2000_bot::m17e_artifact::Row>> =
            std::collections::HashMap::new();
        for row in &artifact.rows {
            by_battle.entry(row.battle).or_default().push(row);
        }
        for (bi, rows) in by_battle {
            let cb_file = load_battle(&files[bi]);
            for a in rows {
                let d = &cb_file.decisions[a.decision];
                if let Some(b) = reconstruct(
                    dex,
                    &src,
                    &pool_path,
                    &cb_file.lines,
                    &cb_file.evidence,
                    d,
                    artifact.run.selection.reconstruction_seed,
                ) {
                    let (table, cb) = CompactBattle::from_battle(&b, dex);
                    let (_, _, klo, khi) = canonical_state_key_128(&cb, &table, 0);
                    keys.insert((klo, khi));
                }
            }
        }
    }
    keys
}

fn run_gendata(args: &[String]) {
    maybe_load_acct_json(args);
    let root = repo_root();
    let dex = load_dex();
    let threads = flag_usize(args, "--threads", default_threads());
    let target_count = flag_usize(args, "--count", 50_000);
    let oracle_iters = flag_u32(args, "--oracle-iters", 384);
    let base_seed = flag_u64(args, "--seed", 0x57A1_C75E_A2C4_0001);
    let out_bin = flag(args, "--out").unwrap_or_else(|| "acct_strict_train.bin".to_string());
    let verify_json = flag(args, "--verify-out");

    println!(
        "=== Generating Strict-Search Accounting Dataset: count={target_count}, oracle_iters={oracle_iters}, threads={threads} ==="
    );
    let heldout_keys = collect_heldout_canonical_keys(&root, &dex);
    println!(
        "  Loaded {} held-out M17e canonical state keys for strict 0-leakage exclusion",
        heldout_keys.len()
    );

    let teams = load_dedup_teams(
        &root,
        &[
            "data/team-pool-v2/team-pool.json",
            "data/belief-pool-v3/belief-pool.json",
            "data/eco-pool/master-pool.json",
        ],
    );
    let bp_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| teams[i].iter().any(is_setup_mon))
        .collect();
    let stall_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| carries_count(&teams[i], &["rest", "recover", "softboiled", "milkdrink"]) >= 3)
        .collect();

    struct RawState {
        battle_id: u32,
        slice_id: u32,
        table: MatchTable,
        cb: CompactBattle,
    }

    let t_gen = Instant::now();
    let seen_keys: Mutex<HashSet<(u64, u64)>> = Mutex::new(heldout_keys);
    let raw_states: Mutex<Vec<RawState>> = Mutex::new(Vec::with_capacity(target_count + 256));
    let next_battle_id = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for tid in 0..threads {
            let teams_ref = &teams;
            let bp_ref = &bp_teams;
            let stall_ref = &stall_teams;
            let seen_ref = &seen_keys;
            let out_ref = &raw_states;
            let bid_ref = &next_battle_id;
            let dex_ref = &dex;
            scope.spawn(move || {
                let mut rng = SplitMix64::new(
                    base_seed ^ (tid as u64).wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                );
                loop {
                    if out_ref.lock().unwrap().len() >= target_count {
                        break;
                    }
                    let bid = bid_ref.fetch_add(1, Ordering::Relaxed) as u32;
                    let archetype_roll = bid % 5;
                    let t0_idx = if archetype_roll == 1 && !bp_ref.is_empty() {
                        bp_ref[rng.below(bp_ref.len())]
                    } else if archetype_roll == 2 && !stall_ref.is_empty() {
                        stall_ref[rng.below(stall_ref.len())]
                    } else {
                        rng.below(teams_ref.len())
                    };
                    let mut t1_idx = if archetype_roll == 3 && !bp_ref.is_empty() {
                        bp_ref[rng.below(bp_ref.len())]
                    } else {
                        rng.below(teams_ref.len())
                    };
                    if t1_idx == t0_idx {
                        t1_idx = (t0_idx + 1) % teams_ref.len();
                    }

                    let mut b = Battle::from_fixture(
                        dex_ref,
                        &rng.battle_seed(),
                        &teams_ref[t0_idx],
                        &teams_ref[t1_idx],
                    )
                    .unwrap();
                    b.set_log_enabled(false);

                    let l0 = b.legal_choices(dex_ref, 0);
                    let l1 = b.legal_choices(dex_ref, 1);
                    let p0 = if archetype_roll == 1 {
                        setup_lead_slot_1based(&teams_ref[t0_idx])
                            .and_then(|ld| {
                                l0.iter()
                                    .copied()
                                    .find(|c| matches!(c, SearchChoice::Team(u) if u[0] == ld))
                            })
                            .unwrap_or(l0[rng.below(l0.len())])
                    } else {
                        l0[rng.below(l0.len())]
                    };
                    let p1 = l1[rng.below(l1.len())];
                    let _ = b.apply_choices(dex_ref, [Some(p0), Some(p1)]);

                    let (table, mut cb) = CompactBattle::from_battle(&b, dex_ref);
                    let mut step_n = 0u32;
                    let mut local_batch: Vec<RawState> = Vec::new();

                    while cb.outcome == 0 && cb.turn <= 70 && step_n < 110 {
                        if cb.sides[0].request == 1 && cb.sides[1].request == 1 {
                            // Sample main-line state with probability 0.45
                            if rng.next_f64() < 0.45 {
                                let (_, _, klo, khi) = canonical_state_key_128(&cb, &table, 0);
                                let is_new = seen_ref.lock().unwrap().insert((klo, khi));
                                if is_new {
                                    let sl = match classify_slice(&cb, &table) {
                                        PosSlice::Standard3v3 => 0,
                                        PosSlice::BoostedBp => 1,
                                        PosSlice::RecoveryStall => 2,
                                        PosSlice::Endgame2v2 => 3,
                                    };
                                    local_batch.push(RawState {
                                        battle_id: bid,
                                        slice_id: sl,
                                        table: table.clone(),
                                        cb,
                                    });
                                }
                            }
                            // Also sample a 1-ply settled child state (`Minimax1` leaf distribution!)
                            if rng.next_f64() < 0.45 {
                                if let Some(settled_child) =
                                    sample_1ply_settled_child(&cb, &table, &mut rng)
                                {
                                    let (_, _, klo, khi) =
                                        canonical_state_key_128(&settled_child, &table, 0);
                                    let is_new = seen_ref.lock().unwrap().insert((klo, khi));
                                    if is_new {
                                        local_batch.push(RawState {
                                            battle_id: bid,
                                            slice_id: 4,
                                            table: table.clone(),
                                            cb: settled_child,
                                        });
                                    }
                                }
                            }
                        }

                        // Advance trajectory on CompactBattle with diverse mix of accounting-top
                        // moves and 22% exploratory perturbations (off-policy blunders/switches/status).
                        let pick_act = |st: &CompactBattle, side: usize, r: &mut SplitMix64| -> u8 {
                            if st.sides[side].request == 0 {
                                return 0xFF;
                            }
                            let mut buf = [0u8; 7];
                            let len = st.legal_actions(&table, side, &mut buf);
                            if len == 0 {
                                return 0xFF;
                            }
                            if st.sides[side].request == 1 && r.next_f64() < 0.78 {
                                let mut cands = [0u8; 4];
                                let clen =
                                    nc2000_bot::accounting::select_accounting_candidates(
                                        st, &table, side, 3, &mut cands,
                                    );
                                if clen > 0 {
                                    // Bias toward #1 candidate (60%), #2 (25%), #3 (15%)
                                    let u = r.next_f64();
                                    let idx = if u < 0.60 || clen == 1 {
                                        0
                                    } else if u < 0.85 || clen == 2 {
                                        1
                                    } else {
                                        2
                                    };
                                    return cands[idx];
                                }
                            }
                            buf[r.below(len)]
                        };

                        let a0 = pick_act(&cb, 0, &mut rng);
                        let a1 = pick_act(&cb, 1, &mut rng);
                        if a0 == 0xFF && a1 == 0xFF {
                            break;
                        }
                        cb.prng_seed = rng.next() | 1;
                        cb.quick_claw_roll = if (rng.next() & 0xFF) < 60 { 1 } else { 0 };
                        cb.apply_actions(&table, [a0, a1]);
                        step_n += 1;
                    }

                    if !local_batch.is_empty() {
                        let mut guard = out_ref.lock().unwrap();
                        for item in local_batch {
                            if guard.len() < target_count {
                                guard.push(item);
                            } else {
                                break;
                            }
                        }
                    }
                }
            });
        }
    });

    let mut states = raw_states.into_inner().unwrap();
    states.truncate(target_count);
    // Deterministic canonical ordering by (battle_id, turn)
    states.sort_by_key(|s| (s.battle_id, s.cb.turn, s.slice_id));
    println!(
        "  Sampled {} unique non-overlapping CompactBattle states across {} battles in {:.2}s",
        states.len(),
        next_battle_id.load(Ordering::Relaxed),
        t_gen.elapsed().as_secs_f64()
    );

    // Label all states with strict search `oracle_equity(cb, table, oracle_iters)` on `threads`
    let t_lbl = Instant::now();
    let model = get_active_acct_model();
    let cursor = AtomicUsize::new(0);
    #[allow(clippy::type_complexity)]
    let labeled_mtx: Mutex<
        Vec<(
            usize,
            u32,
            u32,
            u32,
            f32,
            [f32; ACCT_BASIS_DIM],
            [f32; ACCT_FEAT_DIM],
            [f32; ACCT_FEAT_DIM],
        )>,
    > = Mutex::new(Vec::with_capacity(states.len()));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            let st_ref = &states;
            let cur_ref = &cursor;
            let out_ref = &labeled_mtx;
            scope.spawn(move || {
                clear_oracle_thread_caches();
                let mut local = Vec::new();
                loop {
                    let i = cur_ref.fetch_add(1, Ordering::Relaxed);
                    if i >= st_ref.len() {
                        break;
                    }
                    let rs = &st_ref[i];
                    let y = oracle_equity(&rs.cb, &rs.table, oracle_iters) as f32;
                    let (bd, f0, f1) =
                        extract_acct_components(&rs.cb, &rs.table, &model.w_acct);
                    let n0 = rs.cb.sides[0].alive_mask.count_ones();
                    let n1 = rs.cb.sides[1].alive_mask.count_ones();
                    let alive_pack = (n0 << 8) | n1;
                    local.push((i, rs.battle_id, rs.slice_id, alive_pack, y, bd, f0, f1));
                }
                out_ref.lock().unwrap().extend(local);
            });
        }
    });

    let mut labeled = labeled_mtx.into_inner().unwrap();
    labeled.sort_by_key(|r| r.0);
    let lbl_secs = t_lbl.elapsed().as_secs_f64();
    println!(
        "  Labeled {} states with Oracle(M={oracle_iters}) in {:.2}s ({:.0} states/sec)",
        labeled.len(),
        lbl_secs,
        (labeled.len() as f64) / lbl_secs.max(1e-6)
    );

    // Write fixed-width binary dataset (120 x 4-byte words = 480 bytes/sample)
    if let Some(parent) = Path::new(&out_bin).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    let mut bytes = Vec::with_capacity(labeled.len() * 480);
    let mut slice_counts = [0usize; 5];
    for &(_, bid, sid, apack, y, ref bd, ref f0, ref f1) in &labeled {
        if (sid as usize) < 5 {
            slice_counts[sid as usize] += 1;
        }
        bytes.extend_from_slice(&bid.to_le_bytes());
        bytes.extend_from_slice(&sid.to_le_bytes());
        bytes.extend_from_slice(&apack.to_le_bytes());
        bytes.extend_from_slice(&y.to_le_bytes());
        for &v in bd {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in f0 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for &v in f1 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    std::fs::write(&out_bin, &bytes).unwrap();
    println!(
        "  Wrote {} bytes ({} rows) to `{out_bin}` | slices: 3v3={}, bp={}, stall={}, 2v2={}, settled_1ply={}",
        bytes.len(),
        labeled.len(),
        slice_counts[0],
        slice_counts[1],
        slice_counts[2],
        slice_counts[3],
        slice_counts[4],
    );

    if let Some(vpath) = verify_json {
        let mut samples = Vec::new();
        for &(idx, bid, sid, _apack, y, ref bd, ref f0, ref f1) in labeled.iter().take(64) {
            let rs = &states[idx];
            let rust_eq = compact_accounting_equity(&rs.cb, &rs.table);
            samples.push(json!({
                "battle_id": bid,
                "slice_id": sid,
                "target": y,
                "basis_diff": bd.as_slice(),
                "feat0": f0.as_slice(),
                "feat1": f1.as_slice(),
                "rust_analytical_eq": rust_eq,
            }));
        }
        std::fs::write(&vpath, serde_json::to_string_pretty(&samples).unwrap()).unwrap();
        println!("  Wrote 64 parity verification samples to `{vpath}`");
    }
}

// =========================================================================
// Phase 2: CRN-Paired Head-to-Head Duels & Baton Pass / Setup Gauntlet
// =========================================================================

struct ForcedSetupLead {
    inner: Box<dyn Agent>,
    lead: Option<u8>,
}

impl Agent for ForcedSetupLead {
    fn name(&self) -> String {
        format!("forced:{}", self.inner.name())
    }

    fn choose(
        &mut self,
        b: &Battle,
        dex: &Dex,
        side: usize,
        choices: &[SearchChoice],
    ) -> SearchChoice {
        let pick = self.inner.choose(b, dex, side, choices);
        if let (SearchChoice::Team(t), Some(lead)) = (pick, self.lead) {
            let shared = |c: &SearchChoice| match c {
                SearchChoice::Team(u) => u.iter().filter(|p| t.contains(p)).count(),
                _ => 0,
            };
            if let Some(forced_pick) = choices
                .iter()
                .copied()
                .filter(|c| matches!(c, SearchChoice::Team(u) if u[0] == lead))
                .max_by_key(|c| shared(c))
            {
                return forced_pick;
            }
        }
        pick
    }
}

fn run_bp_gauntlet_duel(
    dex: &Dex,
    root: &Path,
    is_blind: bool,
    iters: u32,
    c_val: f64,
    arm_spec: ArmSpec,
    vs_spec: ArmSpec,
    games: usize,
    base_seed: u64,
    threads: usize,
    max_turns: u16,
    belief_pool: &Arc<MetaPool>,
) -> serde_json::Value {
    let teams = load_dedup_teams(
        root,
        &[
            "data/meta-pool-v0/meta-pool.json",
            "data/team-pool-v2/team-pool.json",
            "data/belief-pool-v3/belief-pool.json",
            "data/eco-pool/master-pool.json",
        ],
    );
    let bp_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| teams[i].iter().any(is_baton_pass_mon))
        .collect();
    let setup_teams: Vec<usize> = (0..teams.len())
        .filter(|&i| setup_lead_slot_1based(&teams[i]).is_some())
        .collect();

    struct GauntletGame {
        arm_team: usize,
        opp_team: usize,
        seed: String,
        arm_is_p1: bool,
        defense: bool,
        arm_seed: u64,
        opp_seed: u64,
    }

    let mut sched = SplitMix64::new(base_seed ^ 0x6A09_E667_F3BC_C908);
    let schedule: Vec<GauntletGame> = (0..games)
        .map(|g| {
            let defense = (g / 2) % 2 == 1;
            let tac_pool = if !bp_teams.is_empty() && g % 5 < 3 {
                &bp_teams
            } else {
                &setup_teams
            };
            let (arm_team, opp_team) = if defense {
                (
                    sched.below(teams.len()),
                    tac_pool[sched.below(tac_pool.len())],
                )
            } else {
                (
                    tac_pool[sched.below(tac_pool.len())],
                    sched.below(teams.len()),
                )
            };
            GauntletGame {
                arm_team,
                opp_team,
                seed: sched.battle_seed(),
                arm_is_p1: g % 2 == 0,
                defense,
                arm_seed: base_seed ^ (g as u64).wrapping_mul(0xA24B_AED4_963E_E407),
                opp_seed: base_seed ^ (g as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25),
            }
        })
        .collect();

    reset_oracle_stats();
    reset_acct_stats();
    let t0 = Instant::now();
    let cursor = AtomicUsize::new(0);
    let rows_mtx: Mutex<Vec<(usize, f64, u16)>> = Mutex::new(Vec::with_capacity(2 * games));

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let job = cursor.fetch_add(1, Ordering::Relaxed);
                    if job >= 2 * schedule.len() {
                        break;
                    }
                    let g = &schedule[job / 2];
                    let spec = if job % 2 == 0 { arm_spec } else { vs_spec };
                    let arm_team = &teams[g.arm_team];
                    let opp_team = &teams[g.opp_team];

                    let mut arm_ag = ForcedSetupLead {
                        inner: build_agent_arm(is_blind, iters, c_val, spec, belief_pool, g.arm_seed),
                        lead: if g.defense {
                            None
                        } else {
                            setup_lead_slot_1based(arm_team)
                        },
                    };
                    let mut foe_ag = ForcedSetupLead {
                        inner: build_agent_arm(
                            is_blind,
                            iters,
                            c_val,
                            vs_spec,
                            belief_pool,
                            g.opp_seed,
                        ),
                        lead: if g.defense {
                            setup_lead_slot_1based(opp_team)
                        } else {
                            None
                        },
                    };

                    let (t1, t2) = if g.arm_is_p1 {
                        (arm_team, opp_team)
                    } else {
                        (opp_team, arm_team)
                    };
                    let mut battle = Battle::from_fixture(dex, &g.seed, t1, t2).unwrap();
                    battle.set_log_enabled(is_blind);
                    let res = if g.arm_is_p1 {
                        let mut pair: [&mut dyn Agent; 2] = [&mut arm_ag, &mut foe_ag];
                        play_game(dex, &mut battle, &mut pair, max_turns).unwrap()
                    } else {
                        let mut pair: [&mut dyn Agent; 2] = [&mut foe_ag, &mut arm_ag];
                        play_game(dex, &mut battle, &mut pair, max_turns).unwrap()
                    };
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let arm_score = if g.arm_is_p1 {
                        p1_score
                    } else {
                        1.0 - p1_score
                    };
                    local.push((job, arm_score, battle.turn));
                }
                rows_mtx.lock().unwrap().extend(local);
            });
        }
    });

    let mut rows = rows_mtx.into_inner().unwrap();
    rows.sort_by_key(|r| r.0);
    let secs = t0.elapsed().as_secs_f64();
    let ostats = oracle_stats();
    let astats = acct_stats();

    let arm_scores: Vec<f64> = rows.iter().step_by(2).map(|r| r.1).collect();
    let ref_scores: Vec<f64> = rows.iter().skip(1).step_by(2).map(|r| r.1).collect();
    let diffs: Vec<f64> = arm_scores
        .iter()
        .zip(&ref_scores)
        .map(|(a, b)| a - b)
        .collect();
    let n = diffs.len() as f64;
    let arm_mean = arm_scores.iter().sum::<f64>() / n;
    let ref_mean = ref_scores.iter().sum::<f64>() / n;
    let diff_mean = diffs.iter().sum::<f64>() / n;
    let sd = if diffs.len() > 1 {
        (diffs.iter().map(|d| (d - diff_mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    let ci95 = 1.96 * sd / n.sqrt();
    let better = diffs.iter().filter(|&&d| d > 1e-9).count();
    let worse = diffs.iter().filter(|&&d| d < -1e-9).count();

    println!(
        "  [bp_gauntlet] arm={arm_mean:.4} vs ref={ref_mean:.4} | delta={diff_mean:+.4} +/- {ci95:.4} (better={better}, worse={worse}, n={games}) | acct_hit={:.1}% | {secs:.2}s",
        if astats.leaf_calls > 0 {
            100.0 * (astats.cache_hits as f64) / (astats.leaf_calls as f64)
        } else {
            0.0
        }
    );

    json!({
        "kind": "bp_gauntlet",
        "arm_mean": arm_mean,
        "ref_mean": ref_mean,
        "delta": diff_mean,
        "ci95": ci95,
        "better_games": better,
        "worse_games": worse,
        "games": games,
        "secs": secs,
        "oracle_eval_calls": ostats.eval_calls,
        "oracle_eval_hits": ostats.eval_hits,
        "acct_leaf_calls": astats.leaf_calls,
        "acct_cache_hits": astats.cache_hits,
    })
}

fn run_duel_cmd(args: &[String]) {
    maybe_load_acct_json(args);
    let root = repo_root();
    let dex = load_dex();

    let env_name = flag(args, "--env").unwrap_or_else(|| "blind".to_string());
    let is_blind = match env_name.as_str() {
        "blind" => true,
        "open" => false,
        other => panic!("--env open|blind, got {other}"),
    };
    let default_c = if is_blind { 0.4 } else { 1.0 };
    let c_val = flag_f64(args, "--c", default_c);
    let iters = flag_u32(args, "--iters", 300);
    let arm_str = flag(args, "--arm").unwrap_or_else(|| "mm1:3".to_string());
    let vs_str = flag(args, "--vs").unwrap_or_else(|| "legacy".to_string());
    let mut arm_spec = parse_arm_spec(&arm_str);
    let vs_spec = parse_arm_spec(&vs_str);
    let iso_time = flag_bool(args, "--iso-time", false);
    let stratum = flag(args, "--stratum").unwrap_or_else(|| "full".to_string());
    let games = flag_usize(args, "--games", 160);
    let base_seed = flag_u64(args, "--seed", 20261030);
    let threads = flag_usize(args, "--threads", default_threads());
    let max_turns = flag_usize(args, "--max-turns", 500) as u16;
    let crn = flag_bool(args, "--crn", true);
    let jsonl_out = flag(args, "--jsonl-out");

    let belief_pool = Arc::new(load_meta_pool(
        &root.join("data/belief-pool-v3/belief-pool.json"),
    ));

    if stratum == "bp_gauntlet" {
        println!(
            "=== Duel [env={env_name} c={c_val} N={iters} stratum=bp_gauntlet]: {arm_str} vs {vs_str} ({games} paired games, threads={threads}) ==="
        );
        let mut res = run_bp_gauntlet_duel(
            &dex,
            &root,
            is_blind,
            iters,
            c_val,
            arm_spec,
            vs_spec,
            games,
            base_seed,
            threads,
            max_turns,
            &belief_pool,
        );
        if let Some(obj) = res.as_object_mut() {
            obj.insert("env".to_string(), json!(env_name));
            obj.insert("c".to_string(), json!(c_val));
            obj.insert("iters".to_string(), json!(iters));
            obj.insert("stratum".to_string(), json!(stratum));
            obj.insert("arm".to_string(), json!(arm_str));
            obj.insert("vs".to_string(), json!(vs_str));
            obj.insert("seed".to_string(), json!(base_seed));
        }
        if let Some(path) = jsonl_out {
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            writeln!(f, "{}", serde_json::to_string(&res).unwrap()).unwrap();
        }
        return;
    }

    let teams: Vec<Vec<PokemonSet>> = match stratum.as_str() {
        "full" => {
            if let Some(custom) = flag(args, "--teams") {
                load_meta_pool(Path::new(&custom))
                    .teams
                    .into_iter()
                    .map(|t| t.sets)
                    .collect()
            } else {
                load_dedup_teams(&root, &["data/team-pool-v2/team-pool.json"])
            }
        }
        "wide" => load_dedup_teams(
            &root,
            &[
                "data/team-pool-v2/team-pool.json",
                "data/meta-pool-v0/meta-pool.json",
            ],
        ),
        "stall" => {
            let all = load_dedup_teams(
                &root,
                &[
                    "data/team-pool-v2/team-pool.json",
                    "data/meta-pool-v0/meta-pool.json",
                    "data/belief-pool-v3/belief-pool.json",
                ],
            );
            let filtered: Vec<Vec<PokemonSet>> = all
                .into_iter()
                .filter(|sets| {
                    carries_count(sets, &["rest", "recover", "softboiled", "milkdrink"]) >= 4
                })
                .collect();
            assert!(
                filtered.len() >= 2,
                "stall filter left fewer than 2 teams ({})",
                filtered.len()
            );
            filtered
        }
        "bp_h2h" => {
            let all = load_dedup_teams(
                &root,
                &[
                    "data/team-pool-v2/team-pool.json",
                    "data/meta-pool-v0/meta-pool.json",
                    "data/belief-pool-v3/belief-pool.json",
                    "data/eco-pool/master-pool.json",
                ],
            );
            let filtered: Vec<Vec<PokemonSet>> = all
                .into_iter()
                .filter(|sets| sets.iter().any(is_baton_pass_mon))
                .collect();
            assert!(
                filtered.len() >= 2,
                "bp_h2h filter left fewer than 2 teams ({})",
                filtered.len()
            );
            filtered
        }
        other => panic!("unknown --stratum `{other}` (expected full|wide|stall|bp_h2h|bp_gauntlet)"),
    };

    if iso_time && arm_spec.iters_override.is_none() {
        let mut warm_spec = DuelSpec::new(24, base_seed ^ 0x1507_14EC_A11B_2A70);
        warm_spec.threads = threads;
        warm_spec.max_turns = max_turns;
        warm_spec.log_on = is_blind;
        warm_spec.crn_agent_seeds = crn;
        let warm = run_duel(
            &dex,
            &teams,
            &|seed| build_agent_arm(is_blind, iters, c_val, arm_spec, &belief_pool, seed),
            &|seed| build_agent_arm(is_blind, iters, c_val, vs_spec, &belief_pool, seed),
            warm_spec,
        );
        let ratio = (warm.b_ms_per_move / warm.a_ms_per_move.max(0.05)) * 0.97;
        let calibrated_iters = ((iters as f64) * ratio).round().clamp(30.0, 20_000.0) as u32;
        arm_spec.iters_override = Some(calibrated_iters);
        println!(
            "  [Iso-Time Warmup] arm@{iters}={:.2}ms/mv vs ref@{iters}={:.2}ms/mv -> calibrated arm_iters={calibrated_iters}",
            warm.a_ms_per_move, warm.b_ms_per_move
        );
    }

    let eff_arm_iters = arm_spec.iters_override.unwrap_or(iters);
    let eff_vs_iters = vs_spec.iters_override.unwrap_or(iters);
    println!(
        "=== Duel [env={env_name} c={c_val} N_A={eff_arm_iters} vs N_B={eff_vs_iters} iso_time={iso_time} stratum={stratum} ({} teams) crn={crn}]: {arm_str} vs {vs_str} ({games} games, threads={threads}) ===",
        teams.len()
    );

    reset_oracle_stats();
    reset_acct_stats();
    let mut spec = DuelSpec::new(games, base_seed);
    spec.threads = threads;
    spec.max_turns = max_turns;
    spec.log_on = is_blind;
    spec.crn_agent_seeds = crn;

    let stats = run_duel(
        &dex,
        &teams,
        &|seed| build_agent_arm(is_blind, iters, c_val, arm_spec, &belief_pool, seed),
        &|seed| build_agent_arm(is_blind, iters, c_val, vs_spec, &belief_pool, seed),
        spec,
    );
    let ostats = oracle_stats();
    let astats = acct_stats();

    let pair_wins = stats.pair_scores.iter().filter(|&&p| p > 0.5 + 1e-9).count();
    let pair_losses = stats.pair_scores.iter().filter(|&&p| p < 0.5 - 1e-9).count();
    let pair_ties = stats.pairs - pair_wins - pair_losses;
    let acct_hit_pct = if astats.leaf_calls > 0 {
        100.0 * (astats.cache_hits as f64) / (astats.leaf_calls as f64)
    } else {
        0.0
    };

    println!(
        "  SCORE = {:.4} +/- {:.4} (95% pair CI) | W-L-T = {}-{}-{} (turn_caps={}) | pairs W-L-T = {}-{}-{}",
        stats.score,
        stats.ci95,
        stats.wins,
        stats.losses,
        stats.ties,
        stats.turn_caps,
        pair_wins,
        pair_losses,
        pair_ties
    );
    println!(
        "  avg_turns={:.1} | A={:.2}ms/mv (N={eff_arm_iters}) B={:.2}ms/mv (N={eff_vs_iters}) | acct_calls={} ({acct_hit_pct:.1}% hit) misses={} q_ext={} | {:.2}s",
        stats.avg_turns,
        stats.a_ms_per_move,
        stats.b_ms_per_move,
        astats.leaf_calls,
        astats.cache_misses,
        astats.quiesce_ext,
        stats.secs
    );

    let rec = json!({
        "kind": "h2h",
        "env": env_name,
        "c": c_val,
        "iters": iters,
        "eff_arm_iters": eff_arm_iters,
        "eff_vs_iters": eff_vs_iters,
        "iso_time": iso_time,
        "stratum": stratum,
        "num_teams": teams.len(),
        "crn": crn,
        "arm": arm_str,
        "vs": vs_str,
        "games": stats.games,
        "pairs": stats.pairs,
        "score": stats.score,
        "ci95": stats.ci95,
        "wins": stats.wins,
        "losses": stats.losses,
        "ties": stats.ties,
        "turn_caps": stats.turn_caps,
        "pair_wins": pair_wins,
        "pair_losses": pair_losses,
        "pair_ties": pair_ties,
        "avg_turns": stats.avg_turns,
        "a_ms_per_move": stats.a_ms_per_move,
        "b_ms_per_move": stats.b_ms_per_move,
        "secs": stats.secs,
        "seed": base_seed,
        "oracle_eval_calls": ostats.eval_calls,
        "acct_leaf_calls": astats.leaf_calls,
        "acct_cache_hits": astats.cache_hits,
        "acct_cache_misses": astats.cache_misses,
        "acct_quiesce_ext": astats.quiesce_ext,
    });

    if let Some(path) = jsonl_out {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap();
        writeln!(f, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
    }
}

// =========================================================================
// Phase 3: Production-Budget Decision Divergence & Terminal Counterfactuals
// =========================================================================

#[derive(Clone)]
struct BlindCheckpoint {
    id: usize,
    is_off_pool: bool,
    phase: &'static str,
    battle: Battle,
    obs0: Observer,
    obs1: Observer,
}

fn format_choice_readable(battle: &Battle, dex: &Dex, side: usize, c: SearchChoice) -> String {
    match c {
        SearchChoice::Move(id) => format!("mv:{}", dex.moves.key(id)),
        SearchChoice::Switch(pos) => {
            let sp = battle.sides[side]
                .party
                .get((pos as usize).saturating_sub(1))
                .and_then(|&slot| battle.sides[side].roster.get(slot as usize))
                .map(|m| dex.species.key(m.species))
                .unwrap_or("?");
            format!("sw:{sp}(#{pos})")
        }
        SearchChoice::Team(t) => format!("team:{}{}{}", t[0], t[1], t[2]),
        SearchChoice::Pass => "pass".to_string(),
    }
}

fn format_board_summary(battle: &Battle, dex: &Dex, belief_cands: usize) -> String {
    let (table, cb) = CompactBattle::from_battle(battle, dex);
    let side_str = |s: usize| -> String {
        let act_slot = (cb.sides[s].active_idx as usize).min(2);
        let mut parts = Vec::new();
        for slot in 0..3usize {
            if slot >= battle.sides[s].party.len() {
                continue;
            }
            let r_idx = battle.sides[s].party[slot] as usize;
            let sp = dex.species.key(battle.sides[s].roster[r_idx].species);
            let idx = s * 3 + slot;
            let m = &cb.mons[idx];
            let maxhp = table.mons[idx].maxhp.max(1);
            if (cb.sides[s].alive_mask & (1 << slot)) == 0 || m.hp() <= 0 {
                parts.push(format!("{sp}:KO"));
                continue;
            }
            let pct = ((m.hp() as f64) * 100.0 / (maxhp as f64)).round() as i32;
            let st = match m.status_packed & 0x07 {
                1 => " Par",
                2 => " Slp",
                3 => " Frz",
                4 => " Brn",
                5 => " Psn",
                6 => " Tox",
                _ => "",
            };
            let mut b_tags = String::new();
            let b_names = ["A", "D", "SA", "SD", "S", "Ac", "Ev"];
            for (bi, bn) in b_names.iter().enumerate() {
                let bv = m.boosts[bi];
                if bv != 0 {
                    b_tags.push_str(&format!("{bn}{bv:+}"));
                }
            }
            if m.sub_hp() > 0 {
                b_tags.push_str(&format!(" Sub({})", m.sub_hp()));
            }
            let mark = if slot == act_slot { "*" } else { "" };
            if b_tags.is_empty() {
                parts.push(format!("{mark}{sp}:{pct}%{st}"));
            } else {
                parts.push(format!("{mark}{sp}:{pct}%{st}[{b_tags}]"));
            }
        }
        let mut flags = Vec::new();
        if (cb.sides[s].side_flags & 0x01) != 0 {
            flags.push("Spikes");
        }
        if cb.sides[s].reflect_turns > 0 {
            flags.push("Refl");
        }
        if cb.sides[s].lightscreen_turns > 0 {
            flags.push("LScr");
        }
        if flags.is_empty() {
            parts.join(" / ")
        } else {
            format!("{} {{{}}}", parts.join(" / "), flags.join(","))
        }
    };
    let bel_tag = if belief_cands == 0 {
        "fallback".to_string()
    } else {
        format!("pool:{belief_cands}")
    };
    format!(
        "T{} ({bel_tag}) | P1: [{}] || P2: [{}]",
        battle.turn,
        side_str(0),
        side_str(1)
    )
}

fn collect_blind_checkpoints(
    dex: &Dex,
    cat_teams: &[Vec<PokemonSet>],
    off_teams: &[Vec<PokemonSet>],
    belief_pool: &MetaPool,
    target_count: usize,
    base_seed: u64,
    opportunity_only: bool,
) -> Vec<BlindCheckpoint> {
    let mut rng = SplitMix64::new(base_seed ^ 0xD17E_26E4_CE00_CAFE);
    let mut out = Vec::new();
    let mut game_idx = 0usize;
    let max_games = if opportunity_only {
        target_count * 40
    } else {
        target_count * 6
    };

    while out.len() < target_count && game_idx < max_games {
        let is_off_pool = (game_idx % 3) == 2 && !off_teams.is_empty();
        let t1 = &cat_teams[rng.below(cat_teams.len())];
        let t2 = if is_off_pool {
            let mut pick = &off_teams[rng.below(off_teams.len())];
            for _ in 0..8 {
                let cand = &off_teams[rng.below(off_teams.len())];
                let sig = team_signature(cand);
                if !cat_teams.iter().any(|ct| team_signature(ct) == sig) {
                    pick = cand;
                    break;
                }
            }
            pick
        } else {
            &cat_teams[rng.below(cat_teams.len())]
        };

        let b_seed = rng.battle_seed();
        let mut battle = Battle::from_fixture(dex, &b_seed, t1, t2).unwrap();
        battle.set_log_enabled(true);

        let mut obs0 = Observer::new(&battle, 0);
        let mut obs1 = Observer::new(&battle, 1);
        let mut bel0 = Belief::new(dex, belief_pool, &obs0);
        let mut bel1 = Belief::new(dex, belief_pool, &obs1);

        // Alternate trajectory generator between Legacy and AdaptiveMinimax1(2) so states are balanced.
        let gen_arm = if game_idx % 2 == 0 {
            parse_arm_spec("legacy")
        } else {
            parse_arm_spec("amm1:2")
        };
        let gen_cfg = build_rm_config_arm(400, 0.4, gen_arm);

        let mut took_early = false;
        let mut took_mid = false;
        let mut took_late = false;

        for _step in 0..220 {
            if battle.outcome().is_some() {
                break;
            }
            obs0.observe(&battle, dex);
            obs1.observe(&battle, dex);
            bel0.sync(dex, &obs0);
            bel1.sync(dex, &obs1);

            let is_prev = battle.sides[0].request_state() == Some(RequestKind::TeamPreview)
                || battle.sides[1].request_state() == Some(RequestKind::TeamPreview);

            if !is_prev && out.len() < target_count {
                let c0 = battle.legal_choices(dex, 0);
                if c0.len() >= 2 {
                    let dom = dominated_actions(&battle, dex, 0);
                    let eligible = c0
                        .iter()
                        .filter(|&&ch| !dom.iter().any(|(dc, _)| *dc == ch))
                        .count();
                    if eligible >= 2 {
                        let (table, cb) = CompactBattle::from_battle(&battle, dex);
                        let sl = classify_slice(&cb, &table);
                        let t = battle.turn;
                        let phase_opt = if opportunity_only {
                            let old_rules = MaskRules {
                                extended_noops: false,
                                ..MaskRules::default()
                            };
                            let old_dom = dominated_actions_with(&battle, dex, 0, old_rules);
                            if dom != old_dom {
                                took_mid = true;
                                Some("mask_opp")
                            } else {
                                None
                            }
                        } else if !took_early && (2..=6).contains(&t) && rng.below(2) == 0 {
                            took_early = true;
                            Some("early_3v3")
                        } else if !took_mid
                            && ((7..=15).contains(&t) || matches!(sl, PosSlice::BoostedBp))
                            && rng.below(2) == 0
                        {
                            took_mid = true;
                            Some(match sl {
                                PosSlice::BoostedBp => "boost_bp",
                                PosSlice::RecoveryStall => "stall",
                                _ => "midgame",
                            })
                        } else if !took_late
                            && (t >= 16
                                || matches!(sl, PosSlice::Endgame2v2 | PosSlice::RecoveryStall))
                            && rng.below(2) == 0
                        {
                            took_late = true;
                            Some(match sl {
                                PosSlice::Endgame2v2 => "endgame",
                                PosSlice::RecoveryStall => "stall",
                                PosSlice::BoostedBp => "boost_bp",
                                _ => "lategame",
                            })
                        } else {
                            None
                        };

                        if let Some(phase) = phase_opt {
                            out.push(BlindCheckpoint {
                                id: out.len(),
                                is_off_pool: bel0.candidate_count() == 0,
                                phase,
                                battle: battle.clone(),
                                obs0: obs0.clone(),
                                obs1: obs1.clone(),
                            });
                            if out.len() >= target_count {
                                break;
                            }
                        }
                    }
                }
            }

            let mut picks = [None, None];
            for (s, (obs, bel)) in [(&obs0, &bel0), (&obs1, &bel1)].into_iter().enumerate() {
                let cs = battle.legal_choices(dex, s);
                if cs.is_empty() {
                    continue;
                }
                if cs.len() == 1 {
                    picks[s] = Some(cs[0]);
                    continue;
                }
                let iters = if is_prev { 250 } else { 400 };
                let mut cfg = gen_cfg.clone();
                cfg.iterations = iters;
                let mut bs = BlindSearch::new(&battle, dex, cfg, s, rng.next());
                bs.prune_preview_aliases(dex);
                bs.step(dex, bel, obs, iters);
                picks[s] = bs.best();
            }
            battle.apply_choices(dex, picks).unwrap();
        }
        game_idx += 1;
    }
    out
}

#[derive(Clone)]
struct SearchSnap {
    acts: Vec<SearchChoice>,
    dominated: Vec<bool>,
    visits: Vec<u32>,
    wins: Vec<f64>,
    best: SearchChoice,
    elapsed_ms: f64,
}

impl SearchSnap {
    fn total_visits(&self) -> u32 {
        self.visits.iter().sum()
    }
    fn top1_share(&self) -> f64 {
        let tot = self.total_visits().max(1) as f64;
        let idx = self.acts.iter().position(|&a| a == self.best).unwrap_or(0);
        (self.visits[idx] as f64) / tot
    }
    fn action_stats(&self, c: SearchChoice) -> (f64, f64) {
        let tot = self.total_visits().max(1) as f64;
        if let Some(i) = self.acts.iter().position(|&a| a == c) {
            let v = self.visits[i];
            let share = (v as f64) / tot;
            let mean = if v > 0 { self.wins[i] / (v as f64) } else { 0.5 };
            (share, mean)
        } else {
            (0.0, 0.5)
        }
    }
    fn merge(&self, other: &SearchSnap) -> SearchSnap {
        assert_eq!(self.acts, other.acts);
        let visits: Vec<u32> = self
            .visits
            .iter()
            .zip(&other.visits)
            .map(|(&a, &b)| a + b)
            .collect();
        let wins: Vec<f64> = self
            .wins
            .iter()
            .zip(&other.wins)
            .map(|(&a, &b)| a + b)
            .collect();
        let any_safe = self.dominated.iter().any(|&d| !d);
        let best_idx = (0..self.acts.len())
            .filter(|&i| !any_safe || !self.dominated[i])
            .max_by(|&a, &b| {
                visits[a].cmp(&visits[b]).then_with(|| {
                    let ma = if visits[a] > 0 {
                        wins[a] / (visits[a] as f64)
                    } else {
                        0.5
                    };
                    let mb = if visits[b] > 0 {
                        wins[b] / (visits[b] as f64)
                    } else {
                        0.5
                    };
                    ma.total_cmp(&mb)
                })
            })
            .unwrap_or(0);
        SearchSnap {
            acts: self.acts.clone(),
            dominated: self.dominated.clone(),
            visits,
            wins,
            best: self.acts[best_idx],
            elapsed_ms: self.elapsed_ms + other.elapsed_ms,
        }
    }
}

fn run_blind_snap(
    cp: &BlindCheckpoint,
    dex: &Dex,
    belief: &Belief,
    spec: ArmSpec,
    iters: u32,
    seed: u64,
) -> SearchSnap {
    let cfg = build_rm_config_arm(iters, 0.4, spec);
    let t0 = Instant::now();
    let mut bs = BlindSearch::new(&cp.battle, dex, cfg, 0, seed);
    bs.step(dex, belief, &cp.obs0, iters);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let acts = bs.actions().to_vec();
    let dominated = bs.dominated().to_vec();
    let visits = bs.visits().to_vec();
    let means = bs.means();
    let wins: Vec<f64> = visits
        .iter()
        .zip(&means)
        .map(|(&v, &m)| (v as f64) * m)
        .collect();
    let best = bs.best().unwrap();
    SearchSnap {
        acts,
        dominated,
        visits,
        wins,
        best,
        elapsed_ms,
    }
}

fn play_counterfactual_branch(
    cp: &BlindCheckpoint,
    dex: &Dex,
    belief_pool: &MetaPool,
    forced_p1: SearchChoice,
    forced_p2: Option<SearchChoice>,
    battle_seed: u64,
    cont_spec: ArmSpec,
    cont_iters: u32,
    cont_seed: u64,
    record_trace: bool,
) -> (f64, u16, Vec<String>) {
    let mut sim = cp.battle.clone();
    sim.set_log_enabled(true);
    sim.reseed(battle_seed);

    let mut obs0 = cp.obs0.clone();
    let mut obs1 = cp.obs1.clone();
    let mut bel0 = Belief::new(dex, belief_pool, &obs0);
    let mut bel1 = Belief::new(dex, belief_pool, &obs1);
    bel0.sync(dex, &obs0);
    bel1.sync(dex, &obs1);

    let mut rng = SplitMix64::new(cont_seed);
    let mut trace = Vec::new();
    let start_turn = sim.turn;

    if record_trace {
        let p2_str = forced_p2
            .map(|c| format_choice_readable(&sim, dex, 1, c))
            .unwrap_or_else(|| "-".to_string());
        trace.push(format!(
            "T{}[FORK]: P1={} vs P2={}",
            sim.turn,
            format_choice_readable(&sim, dex, 0, forced_p1),
            p2_str
        ));
    }
    sim.apply_choices(dex, [Some(forced_p1), forced_p2])
        .unwrap();

    let cont_cfg = build_rm_config_arm(cont_iters, 0.4, cont_spec);
    for _ply in 0..260 {
        if sim.outcome().is_some() || sim.turn > start_turn.saturating_add(160) {
            break;
        }
        obs0.observe(&sim, dex);
        obs1.observe(&sim, dex);
        bel0.sync(dex, &obs0);
        bel1.sync(dex, &obs1);

        let mut picks = [None, None];
        for (s, (obs, bel)) in [(&obs0, &bel0), (&obs1, &bel1)].into_iter().enumerate() {
            let cs = sim.legal_choices(dex, s);
            if cs.is_empty() {
                continue;
            }
            if cs.len() == 1 {
                picks[s] = Some(cs[0]);
                continue;
            }
            let mut bs = BlindSearch::new(&sim, dex, cont_cfg.clone(), s, rng.next());
            bs.step(dex, bel, obs, cont_iters);
            picks[s] = bs.best();
        }
        if record_trace && trace.len() < 12 {
            let s0 = picks[0]
                .map(|c| format_choice_readable(&sim, dex, 0, c))
                .unwrap_or_else(|| "-".into());
            let s1 = picks[1]
                .map(|c| format_choice_readable(&sim, dex, 1, c))
                .unwrap_or_else(|| "-".into());
            let (_tab, cb) = CompactBattle::from_battle(&sim, dex);
            trace.push(format!(
                "T{}(alive {}v{}): {} vs {}",
                sim.turn,
                cb.sides[0].alive_mask.count_ones(),
                cb.sides[1].alive_mask.count_ones(),
                s0,
                s1
            ));
        }
        sim.apply_choices(dex, picks).unwrap();
    }

    let score = match sim.outcome() {
        Some(Outcome::P1Win) => 1.0,
        Some(Outcome::P2Win) => 0.0,
        Some(Outcome::Tie) | None => 0.5,
    };
    if record_trace {
        let out_tag = match sim.outcome() {
            Some(Outcome::P1Win) => "P1_WIN(1.0)",
            Some(Outcome::P2Win) => "P2_WIN(0.0)",
            Some(Outcome::Tie) => "TIE(0.5)",
            None => "TURN_CAP(0.5)",
        };
        trace.push(format!("-> END T{}: {out_tag}", sim.turn));
    }
    (score, sim.turn.saturating_sub(start_turn), trace)
}

fn run_divergence(args: &[String]) {
    maybe_load_acct_json(args);
    let root = repo_root();
    let dex = load_dex();

    let n_checkpoints = flag_usize(args, "--checkpoints", 60);
    let iters = flag_u32(args, "--iters", 27_000);
    let extra_seeds = flag_usize(args, "--seeds", 3).max(2);
    let cont_trials = flag_usize(args, "--cont-trials", 24).max(4);
    let cont_iters = flag_u32(args, "--cont-iters", 1_500);
    let arm_str = flag(args, "--arm").unwrap_or_else(|| "shipped".to_string());
    let vs_str = flag(args, "--vs").unwrap_or_else(|| "legacy".to_string());
    let arm_spec = parse_arm_spec(&arm_str);
    let vs_spec = parse_arm_spec(&vs_str);
    let threads = flag_usize(args, "--threads", default_threads());
    let base_seed = flag_u64(args, "--seed", 20261030);
    let opportunity_only = flag_bool(args, "--opportunity", false);

    let belief_pool = Arc::new(load_meta_pool(
        &root.join("data/belief-pool-v3/belief-pool.json"),
    ));
    let cat_teams = load_dedup_teams(&root, &["data/team-pool-v2/team-pool.json"]);
    let off_teams = load_dedup_teams(
        &root,
        &[
            "data/meta-pool-v0/meta-pool.json",
            "data/eco-pool/master-pool.json",
            "data/kodoku-r3/conclusion-equilibrium-deme.json",
        ],
    );

    println!(
        "=== Step 1: Collecting {n_checkpoints} Nontrivial Blind Decision Checkpoints (catalog + off-pool fallback, opportunity_only={opportunity_only}) ==="
    );
    let t_col = Instant::now();
    let checkpoints = collect_blind_checkpoints(
        &dex,
        &cat_teams,
        &off_teams,
        &belief_pool,
        n_checkpoints,
        base_seed,
        opportunity_only,
    );
    let off_cnt = checkpoints.iter().filter(|c| c.is_off_pool).count();
    println!(
        "  Collected {} checkpoints ({} pool-identified, {} fallback-imputed) in {:.2}s\n",
        checkpoints.len(),
        checkpoints.len() - off_cnt,
        off_cnt,
        t_col.elapsed().as_secs_f64()
    );

    println!(
        "=== Step 2: Production-Budget Screen (Blind c=0.4, standard N={iters}, strong N={} = {extra_seeds}x{iters}, threads={threads}) ===",
        (iters as usize) * extra_seeds
    );

    struct ScreenRow {
        cp_idx: usize,
        belief_cands: usize,
        vs_s1: SearchSnap,
        vs_s2: SearchSnap,
        arm_s1: SearchSnap,
        arm_s2: SearchSnap,
        vs_full: SearchSnap,
        arm_full: SearchSnap,
    }

    let t_scr = Instant::now();
    let cursor = AtomicUsize::new(0);
    let rows_mutex: Mutex<Vec<ScreenRow>> = Mutex::new(Vec::with_capacity(checkpoints.len()));

    std::thread::scope(|s| {
        for _ in 0..threads {
            let dex = &dex;
            let cps = &checkpoints;
            let bpool = &belief_pool;
            let cur = &cursor;
            let out_m = &rows_mutex;
            s.spawn(move || {
                clear_acct_thread_caches();
                clear_oracle_thread_caches();
                let mut local = Vec::new();
                loop {
                    let i = cur.fetch_add(1, Ordering::Relaxed);
                    if i >= cps.len() {
                        break;
                    }
                    let cp = &cps[i];
                    let mut bel = Belief::new(dex, bpool, &cp.obs0);
                    bel.sync(dex, &cp.obs0);
                    let belief_cands = bel.candidate_count();

                    let s1 = base_seed ^ ((i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let s2 = base_seed ^ ((i as u64 + 1).wrapping_mul(0xBF58_476D_1CE4_E5B9));

                    let vs_s1 = run_blind_snap(cp, dex, &bel, vs_spec, iters, s1);
                    let vs_s2 = run_blind_snap(cp, dex, &bel, vs_spec, iters, s2);
                    let arm_s1 = run_blind_snap(cp, dex, &bel, arm_spec, iters, s1);
                    let arm_s2 = run_blind_snap(cp, dex, &bel, arm_spec, iters, s2);

                    let mut vs_full = vs_s1.merge(&vs_s2);
                    let mut arm_full = arm_s1.merge(&arm_s2);

                    let any_diff = arm_s1.best != vs_s1.best
                        || arm_s2.best != vs_s2.best
                        || arm_full.best != vs_full.best;

                    if any_diff && extra_seeds > 2 {
                        for k in 2..extra_seeds {
                            let sk = s1.wrapping_add((k as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                            let vs_sk = run_blind_snap(cp, dex, &bel, vs_spec, iters, sk);
                            let arm_sk = run_blind_snap(cp, dex, &bel, arm_spec, iters, sk);
                            vs_full = vs_full.merge(&vs_sk);
                            arm_full = arm_full.merge(&arm_sk);
                        }
                    }

                    local.push(ScreenRow {
                        cp_idx: i,
                        belief_cands,
                        vs_s1,
                        vs_s2,
                        arm_s1,
                        arm_s2,
                        vs_full,
                        arm_full,
                    });
                }
                out_m.lock().unwrap().extend(local);
            });
        }
    });

    let mut rows = rows_mutex.into_inner().unwrap();
    rows.sort_by_key(|r| r.cp_idx);

    let total = rows.len().max(1) as f64;
    let vs_seed_flips = rows
        .iter()
        .filter(|r| r.vs_s1.best != r.vs_s2.best)
        .count();
    let arm_seed_flips = rows
        .iter()
        .filter(|r| r.arm_s1.best != r.arm_s2.best)
        .count();
    let s1_divergences = rows
        .iter()
        .filter(|r| r.arm_s1.best != r.vs_s1.best)
        .count();
    let s1_class_divs = rows
        .iter()
        .filter(|r| {
            r.arm_s1.best != r.vs_s1.best
                && matches!(r.arm_s1.best, SearchChoice::Switch(_))
                    != matches!(r.vs_s1.best, SearchChoice::Switch(_))
        })
        .count();
    let full_divergences: Vec<&ScreenRow> = rows
        .iter()
        .filter(|r| r.arm_full.best != r.vs_full.best)
        .collect();

    let mean_vs_ms = rows.iter().map(|r| r.vs_s1.elapsed_ms).sum::<f64>() / total;
    let mean_arm_ms = rows.iter().map(|r| r.arm_s1.elapsed_ms).sum::<f64>() / total;

    println!(
        "  Screen finished in {:.2}s | Single-decision latency @{iters} (under {threads}-thread load): {vs_str}={mean_vs_ms:.1}ms, {arm_str}={mean_arm_ms:.1}ms ({:.2}x)",
        t_scr.elapsed().as_secs_f64(),
        mean_arm_ms / mean_vs_ms.max(0.1)
    );
    println!(
        "  [{vs_str} @{iters}] Seed-flip floor (s1 != s2):       {:2}/{} ({:.1}%)",
        vs_seed_flips,
        rows.len(),
        100.0 * (vs_seed_flips as f64) / total
    );
    println!(
        "  [{arm_str} @{iters}] Seed-flip floor (s1 != s2):       {:2}/{} ({:.1}%)",
        arm_seed_flips,
        rows.len(),
        100.0 * (arm_seed_flips as f64) / total
    );
    println!(
        "  [{arm_str} vs {vs_str} @{iters}] Top-1 divergence:     {:2}/{} ({:.1}%)  [action-class move<->switch: {}/{} ({:.1}%)]",
        s1_divergences,
        rows.len(),
        100.0 * (s1_divergences as f64) / total,
        s1_class_divs,
        rows.len(),
        100.0 * (s1_class_divs as f64) / total
    );
    println!(
        "  [STRONG @{} = {extra_seeds}x{iters}] Confirmed divergences: {:2}/{} ({:.1}%)",
        (iters as usize) * extra_seeds,
        full_divergences.len(),
        rows.len(),
        100.0 * (full_divergences.len() as f64) / total
    );

    // Confidence-band breakdown
    for (label, lo, hi) in [
        ("Low-conf   (<0.55)", 0.0, 0.55),
        ("Mid-conf (0.55-0.75)", 0.55, 0.75),
        ("High-conf  (>=0.75)", 0.75, 1.01),
    ] {
        let band_rows: Vec<&ScreenRow> = rows
            .iter()
            .filter(|r| {
                let sh = r.vs_s1.top1_share();
                sh >= lo && sh < hi
            })
            .collect();
        let n = band_rows.len();
        if n > 0 {
            let sf = band_rows
                .iter()
                .filter(|r| r.vs_s1.best != r.vs_s2.best)
                .count();
            let dv = band_rows
                .iter()
                .filter(|r| r.arm_s1.best != r.vs_s1.best)
                .count();
            let dv81 = band_rows
                .iter()
                .filter(|r| r.arm_full.best != r.vs_full.best)
                .count();
            println!(
                "    Band {label}: n={n:2} | {vs_str} seed-flip={sf:2} ({:4.1}%) | @{iters} div={dv:2} ({:4.1}%) | @strong div={dv81:2} ({:4.1}%)",
                100.0 * (sf as f64) / (n as f64),
                100.0 * (dv as f64) / (n as f64),
                100.0 * (dv81 as f64) / (n as f64),
            );
        }
    }

    if full_divergences.is_empty() {
        println!("\nNo confirmed strong-budget divergences found.");
        return;
    }

    println!(
        "\n=== Step 3: Counterfactual Play-Out to Terminal Outcome on All {} Confirmed Strong-Budget Divergences ===",
        full_divergences.len()
    );
    println!(
        "  Each divergence played out for {cont_trials} CRN-paired games to terminal outcome (half {vs_str}@{cont_iters} cont, half {arm_str}@{cont_iters} cont).\n"
    );

    let mut all_deltas: Vec<f64> = Vec::new();
    let mut pos_wins = 0usize;
    let mut pos_losses = 0usize;
    let mut pos_ties = 0usize;

    for (rank, row) in full_divergences.iter().enumerate() {
        let cp = &checkpoints[row.cp_idx];
        let a_vs = row.vs_full.best;
        let a_arm = row.arm_full.best;

        // Also check True-Board SkuctSearch(iters) for both arms to separate hidden-info vs true-board!
        let true_vs_cfg = build_rm_config_arm(iters, 1.0, vs_spec);
        let mut sk_vs = SkuctSearch::new(&cp.battle, &dex, true_vs_cfg, base_seed ^ 0x1111_2222);
        sk_vs.step(&dex, iters);
        let true_vs_pick = sk_vs.best(0).unwrap();

        let true_arm_cfg = build_rm_config_arm(iters, 1.0, arm_spec);
        let mut sk_arm = SkuctSearch::new(&cp.battle, &dex, true_arm_cfg, base_seed ^ 0x1111_2222);
        sk_arm.step(&dex, iters);
        let true_arm_pick = sk_arm.best(0).unwrap();

        // Run CRN-paired terminal continuations across threads
        let trial_cursor = AtomicUsize::new(0);
        struct TrialOut {
            t_idx: usize,
            is_arm_cont: bool,
            s_vs: f64,
            s_arm: f64,
            turns_vs: u16,
            turns_arm: u16,
            trace_vs: Vec<String>,
            trace_arm: Vec<String>,
        }
        let tout_m: Mutex<Vec<TrialOut>> = Mutex::new(Vec::with_capacity(cont_trials));

        std::thread::scope(|s| {
            for _ in 0..threads.min(cont_trials) {
                let dex = &dex;
                let bpool = &belief_pool;
                let tcur = &trial_cursor;
                let out_v = &tout_m;
                s.spawn(move || {
                    clear_acct_thread_caches();
                    loop {
                        let t = tcur.fetch_add(1, Ordering::Relaxed);
                        if t >= cont_trials {
                            break;
                        }
                        let is_arm_cont = (t % 2) == 1;
                        let c_spec = if is_arm_cont { arm_spec } else { vs_spec };
                        let b_seed = base_seed
                            ^ ((row.cp_idx as u64 + 100).wrapping_mul(0xD1B5_4A32_D192_ED03))
                            ^ ((t as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let opp_seed = b_seed ^ 0xA5A5_5A5A_F0F0_0F0F;
                        let cont_seed = b_seed ^ 0x3C3C_C3C3_6969_9696;

                        // Neutral simultaneous opponent move at the fork turn
                        let opp_legal = cp.battle.clone().legal_choices(dex, 1);
                        let forced_p2 = if opp_legal.is_empty() {
                            None
                        } else if opp_legal.len() == 1 {
                            Some(opp_legal[0])
                        } else {
                            let mut bel1 = Belief::new(dex, bpool, &cp.obs1);
                            bel1.sync(dex, &cp.obs1);
                            let opp_cfg = build_rm_config_arm(3_000, 0.4, c_spec);
                            let mut bs1 = BlindSearch::new(&cp.battle, dex, opp_cfg, 1, opp_seed);
                            bs1.step(dex, &bel1, &cp.obs1, 3_000);
                            bs1.best()
                        };

                        let rec_trace = t < 4;
                        let (s_vs, turns_vs, trace_vs) = play_counterfactual_branch(
                            cp,
                            dex,
                            bpool,
                            a_vs,
                            forced_p2,
                            b_seed,
                            c_spec,
                            cont_iters,
                            cont_seed,
                            rec_trace,
                        );
                        let (s_arm, turns_arm, trace_arm) = play_counterfactual_branch(
                            cp,
                            dex,
                            bpool,
                            a_arm,
                            forced_p2,
                            b_seed,
                            c_spec,
                            cont_iters,
                            cont_seed,
                            rec_trace,
                        );
                        out_v.lock().unwrap().push(TrialOut {
                            t_idx: t,
                            is_arm_cont,
                            s_vs,
                            s_arm,
                            turns_vs,
                            turns_arm,
                            trace_vs,
                            trace_arm,
                        });
                    }
                });
            }
        });

        let mut touts = tout_m.into_inner().unwrap();
        touts.sort_by_key(|x| x.t_idx);

        let n_t = touts.len() as f64;
        let mean_vs = touts.iter().map(|x| x.s_vs).sum::<f64>() / n_t;
        let mean_arm = touts.iter().map(|x| x.s_arm).sum::<f64>() / n_t;
        let delta = mean_arm - mean_vs;
        let var = touts
            .iter()
            .map(|x| ((x.s_arm - x.s_vs) - delta).powi(2))
            .sum::<f64>()
            / (n_t - 1.0).max(1.0);
        let se = (var / n_t).sqrt();

        let leg_c: Vec<&TrialOut> = touts.iter().filter(|x| !x.is_arm_cont).collect();
        let arm_c: Vec<&TrialOut> = touts.iter().filter(|x| x.is_arm_cont).collect();
        let delta_leg_c = leg_c.iter().map(|x| x.s_arm - x.s_vs).sum::<f64>()
            / (leg_c.len().max(1) as f64);
        let delta_arm_c = arm_c.iter().map(|x| x.s_arm - x.s_vs).sum::<f64>()
            / (arm_c.len().max(1) as f64);
        let avg_rem_turns = touts
            .iter()
            .map(|x| 0.5 * (x.turns_vs as f64 + x.turns_arm as f64))
            .sum::<f64>()
            / n_t;

        if delta > 0.02 {
            pos_wins += 1;
        } else if delta < -0.02 {
            pos_losses += 1;
        } else {
            pos_ties += 1;
        }
        all_deltas.push(delta);

        let (vs_sh_on_vs, vs_q_on_vs) = row.vs_full.action_stats(a_vs);
        let (vs_sh_on_arm, vs_q_on_arm) = row.vs_full.action_stats(a_arm);
        let (arm_sh_on_arm, arm_q_on_arm) = row.arm_full.action_stats(a_arm);
        let (arm_sh_on_vs, arm_q_on_vs) = row.arm_full.action_stats(a_vs);

        println!(
            "--------------------------------------------------------------------------------"
        );
        println!(
            "[Div #{:02} | cp={} | phase={}] {}",
            rank + 1,
            cp.id,
            cp.phase,
            format_board_summary(&cp.battle, &dex, row.belief_cands)
        );
        println!(
            "  {vs_str}@strong pick : {:18} (vis {:5.1}%, Q={:.3}) | alt {:18} (vis {:5.1}%, Q={:.3}) | TrueSkuct: {}",
            format_choice_readable(&cp.battle, &dex, 0, a_vs),
            100.0 * vs_sh_on_vs,
            vs_q_on_vs,
            format_choice_readable(&cp.battle, &dex, 0, a_arm),
            100.0 * vs_sh_on_arm,
            vs_q_on_arm,
            format_choice_readable(&cp.battle, &dex, 0, true_vs_pick),
        );
        println!(
            "  {arm_str}@strong pick: {:18} (vis {:5.1}%, Q={:.3}) | alt {:18} (vis {:5.1}%, Q={:.3}) | TrueSkuct: {}",
            format_choice_readable(&cp.battle, &dex, 0, a_arm),
            100.0 * arm_sh_on_arm,
            arm_q_on_arm,
            format_choice_readable(&cp.battle, &dex, 0, a_vs),
            100.0 * arm_sh_on_vs,
            arm_q_on_vs,
            format_choice_readable(&cp.battle, &dex, 0, true_arm_pick),
        );
        println!(
            "  TERMINAL OUTCOME ({cont_trials} CRN games, avg +{avg_rem_turns:.1} turns to end): W({arm_str})={mean_arm:.3} vs W({vs_str})={mean_vs:.3} => dEquity = {delta:+.3} +/- {se:.3} (under {vs_str}-cont: {delta_leg_c:+.3}, {arm_str}-cont: {delta_arm_c:+.3})"
        );

        // Pick a trace where the outcome differed (or t=0 if all tied)
        if let Some(ex) = touts
            .iter()
            .filter(|x| !x.trace_vs.is_empty())
            .find(|x| (x.s_arm - x.s_vs).abs() > 0.1)
            .or_else(|| touts.iter().find(|x| !x.trace_vs.is_empty()))
        {
            println!(
                "  Sample Terminal Line ({vs_str} branch -> {:.1}): {}",
                ex.s_vs,
                ex.trace_vs.join(" -> ")
            );
            println!(
                "  Sample Terminal Line ({arm_str} branch -> {:.1}): {}",
                ex.s_arm,
                ex.trace_arm.join(" -> ")
            );
        }
    }

    let n_d = all_deltas.len().max(1) as f64;
    let mean_delta = all_deltas.iter().sum::<f64>() / n_d;
    let d_var = all_deltas
        .iter()
        .map(|&d| (d - mean_delta).powi(2))
        .sum::<f64>()
        / (n_d - 1.0).max(1.0);
    let d_se = (d_var / n_d).sqrt();
    let implied_per_decision = mean_delta * (full_divergences.len() as f64) / total;

    println!("================================================================================");
    println!(
        "SUMMARY @ Production Strong Budget ({} iters, {} checkpoints):",
        (iters as usize) * extra_seeds,
        rows.len()
    );
    println!(
        "  Confirmed Divergence Rate : {}/{} ({:.1}%)",
        full_divergences.len(),
        rows.len(),
        100.0 * (full_divergences.len() as f64) / total
    );
    println!(
        "  Divergence Record         : {pos_wins} improved (>+2%), {pos_losses} regressed (<-2%), {pos_ties} neutral (|d|<=2%)"
    );
    println!(
        "  Mean dEquity on Diverged  : {mean_delta:+.4} +/- {d_se:.4} (1 SE)"
    );
    println!(
        "  Implied Per-Decision Lift : {implied_per_decision:+.4} equity/turn"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("seam");
    match cmd {
        "seam" => run_seam(&args[1..]),
        "calibrate" => run_calibrate(&args[1..]),
        "gendata" => run_gendata(&args[1..]),
        "duel" => run_duel_cmd(&args[1..]),
        "divergence" => run_divergence(&args[1..]),
        other => panic!(
            "unknown subcommand `{other}` (expected seam | calibrate | gendata | duel | divergence)"
        ),
    }
}
