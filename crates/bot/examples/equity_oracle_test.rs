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
use nc2000_bot::agent::Agent;
use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, reconstruct};
use nc2000_bot::duel::{run_duel, DuelSpec};
use nc2000_bot::eval::{compact_eval01, compact_eval_leaf, EvalWeights};
use nc2000_bot::m17e_artifact::MergedArtifact;
use nc2000_bot::mcts::Playout;
use nc2000_bot::oracle::{
    clear_oracle_thread_caches, oracle_equity, oracle_stats, reset_oracle_stats,
    swap_compact_sides, OracleMode,
};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::runner::{play_game, GameResult};
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule, SkuctSearch};
use nc2000_bot::{BlindAgent, OpenAgent};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::compact::CompactBattle;
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

fn parse_oracle_mode(spec: &str) -> OracleMode {
    if spec == "off" || spec == "base" {
        return OracleMode::Off;
    }
    if spec == "echo" {
        return OracleMode::EchoBase;
    }
    if let Some(rest) = spec.strip_prefix("cutoff:") {
        let leaf_iters: u32 = rest.parse().expect("cutoff:<leaf_iters>");
        return OracleMode::Cutoff { leaf_iters };
    }
    if let Some(rest) = spec.strip_prefix("treeleaf:") {
        let leaf_iters: u32 = rest.parse().expect("treeleaf:<leaf_iters>");
        return OracleMode::TreeLeaf { leaf_iters };
    }
    if let Some(rest) = spec.strip_prefix("coupled:") {
        let (r_str, l_str) = rest
            .split_once(',')
            .unwrap_or((rest, rest));
        let rollout_iters: u32 = r_str.parse().expect("coupled:<rollout_iters>,<leaf_iters>");
        let leaf_iters: u32 = l_str.parse().expect("coupled:<rollout_iters>,<leaf_iters>");
        return OracleMode::Coupled {
            rollout_iters,
            leaf_iters,
        };
    }
    panic!("unknown oracle mode `{spec}` (expected off|echo|cutoff:M|treeleaf:M|coupled:Mr,Ml)");
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
        ..Default::default()
    }
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

fn run_calibrate(args: &[String]) {
    let root = repo_root();
    let dex = load_dex();
    let threads = flag_usize(args, "--threads", default_threads());
    let target_per_slice = flag_usize(args, "--per-slice", 40);
    let teacher_iters = flag_u32(args, "--teacher-iters", 3000);
    let playoff_iters = flag_u32(args, "--playoff-iters", 150);
    let playoff_games = flag_usize(args, "--playoff-games", 8);
    let budgets: [u32; 4] = [32, 64, 128, 256];

    let base_w = EvalWeights {
        residual_scale: 0.0,
        ..EvalWeights::default()
    };

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

        let mut anchors: Vec<(Battle, f64, f64)> = Vec::new();
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
                anchors.push((b, a.exact, a.width));
            }
        }

        let exact_targets: Vec<f64> = anchors.iter().map(|x| x.1).collect();
        let eval_models: Vec<(String, Option<u32>, bool)> = vec![
            ("Base-eval01".to_string(), None, false),
            ("Base-eval_leaf".to_string(), None, true),
            ("Oracle(M=32)".to_string(), Some(32), false),
            ("Oracle(M=64)".to_string(), Some(64), false),
            ("Oracle(M=128)".to_string(), Some(128), false),
            ("Oracle(M=256)".to_string(), Some(256), false),
        ];

        for (label, m_opt, use_leaf) in eval_models {
            clear_oracle_thread_caches();
            let t0 = Instant::now();
            let mut preds = Vec::with_capacity(anchors.len());
            let mut viols = 0usize;
            let mut viol_margin = 0.0f64;
            for (b, exact, width) in &anchors {
                let (table, cb) = CompactBattle::from_battle(b, &dex);
                let p = match m_opt {
                    None if !use_leaf => compact_eval01(&cb, &table, &base_w),
                    None => compact_eval_leaf(&cb, &table, &base_w),
                    Some(m) => oracle_equity(&cb, &table, m),
                };
                preds.push(p);
                let (lo, hi) = (exact - width / 2.0, exact + width / 2.0);
                let v = (p - hi).max(lo - p).max(0.0);
                if v > 0.02 {
                    viols += 1;
                    viol_margin += v;
                }
            }
            let (rmse, mae, bias) = rmse_mae_bias(&preds, &exact_targets);
            let r = pearson_r(&preds, &exact_targets);
            println!(
                "  {label:15} | viols={viols:2}/{} margin={viol_margin:.4} | RMSE={rmse:.4} MAE={mae:.4} bias={bias:+.4} r={r:.4} ({:.2}s)",
                anchors.len(),
                t0.elapsed().as_secs_f64()
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

    let mut model_specs: Vec<(String, Option<u32>)> = vec![("Base-eval01".to_string(), None)];
    for &m in &budgets {
        model_specs.push((format!("Oracle(M={m})"), Some(m)));
    }

    println!("\n  --- Overall Accuracy vs SkuctSearch({teacher_iters})x2 AND Terminal Playoff ---");
    for (label, m_opt) in &model_specs {
        clear_oracle_thread_caches();
        let t0 = Instant::now();
        let preds: Vec<f64> = positions
            .iter()
            .map(|pos| {
                let (table, cb) = CompactBattle::from_battle(&pos.battle, &dex);
                match m_opt {
                    None => compact_eval01(&cb, &table, &base_w),
                    Some(m) => oracle_equity(&cb, &table, *m),
                }
            })
            .collect();
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
            "  {label:14} | vs Skuct: RMSE={rmse_sk:.4} MAE={mae_sk:.4} bias={bias_sk:+.4} r={r_sk:.4} | vs Term: RMSE={rmse_tm:.4} r={r_tm:.4} | slices[3v3={:.4}, bp={:.4}, stall={:.4}, 2v2={:.4}] ({:.2}s)",
            slice_rmses[0],
            slice_rmses[1],
            slice_rmses[2],
            slice_rmses[3],
            t0.elapsed().as_secs_f64()
        );
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
    arm_oracle: OracleMode,
    vs_oracle: OracleMode,
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
                    let mode = if job % 2 == 0 { arm_oracle } else { vs_oracle };
                    let arm_team = &teams[g.arm_team];
                    let opp_team = &teams[g.opp_team];

                    let mut arm_ag = ForcedSetupLead {
                        inner: build_agent(is_blind, iters, c_val, mode, belief_pool, g.arm_seed),
                        lead: if g.defense {
                            None
                        } else {
                            setup_lead_slot_1based(arm_team)
                        },
                    };
                    let mut foe_ag = ForcedSetupLead {
                        inner: build_agent(
                            is_blind,
                            iters,
                            c_val,
                            OracleMode::Off,
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
        "  [bp_gauntlet] arm={arm_mean:.4} vs ref={ref_mean:.4} | delta={diff_mean:+.4} +/- {ci95:.4} (better={better}, worse={worse}, n={games}) | eval_hit_rate={:.1}% | {secs:.2}s",
        if ostats.eval_calls > 0 {
            100.0 * (ostats.eval_hits as f64) / (ostats.eval_calls as f64)
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
        "oracle_pick_calls": ostats.pick_calls,
        "oracle_pick_hits": ostats.pick_hits,
    })
}

fn run_duel_cmd(args: &[String]) {
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
    let arm_str = flag(args, "--arm").unwrap_or_else(|| "cutoff:128".to_string());
    let vs_str = flag(args, "--vs").unwrap_or_else(|| "off".to_string());
    let arm_oracle = parse_oracle_mode(&arm_str);
    let vs_oracle = parse_oracle_mode(&vs_str);
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
            arm_oracle,
            vs_oracle,
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

    println!(
        "=== Duel [env={env_name} c={c_val} N={iters} stratum={stratum} ({} teams) crn={crn}]: {arm_str} vs {vs_str} ({games} games, threads={threads}) ===",
        teams.len()
    );

    reset_oracle_stats();
    let mut spec = DuelSpec::new(games, base_seed);
    spec.threads = threads;
    spec.max_turns = max_turns;
    spec.log_on = is_blind;
    spec.crn_agent_seeds = crn;

    let stats = run_duel(
        &dex,
        &teams,
        &|seed| build_agent(is_blind, iters, c_val, arm_oracle, &belief_pool, seed),
        &|seed| build_agent(is_blind, iters, c_val, vs_oracle, &belief_pool, seed),
        spec,
    );
    let ostats = oracle_stats();

    let pair_wins = stats.pair_scores.iter().filter(|&&p| p > 0.5 + 1e-9).count();
    let pair_losses = stats.pair_scores.iter().filter(|&&p| p < 0.5 - 1e-9).count();
    let pair_ties = stats.pairs - pair_wins - pair_losses;
    let eval_hit_pct = if ostats.eval_calls > 0 {
        100.0 * (ostats.eval_hits as f64) / (ostats.eval_calls as f64)
    } else {
        0.0
    };
    let pick_hit_pct = if ostats.pick_calls > 0 {
        100.0 * (ostats.pick_hits as f64) / (ostats.pick_calls as f64)
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
        "  avg_turns={:.1} | A={:.2}ms/mv B={:.2}ms/mv | eval_calls={} ({eval_hit_pct:.1}% hit) pick_calls={} ({pick_hit_pct:.1}% hit) | {:.2}s",
        stats.avg_turns,
        stats.a_ms_per_move,
        stats.b_ms_per_move,
        ostats.eval_calls,
        ostats.pick_calls,
        stats.secs
    );

    let rec = json!({
        "kind": "h2h",
        "env": env_name,
        "c": c_val,
        "iters": iters,
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
        "oracle_eval_hits": ostats.eval_hits,
        "oracle_pick_calls": ostats.pick_calls,
        "oracle_pick_hits": ostats.pick_hits,
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("seam");
    match cmd {
        "seam" => run_seam(&args[1..]),
        "calibrate" => run_calibrate(&args[1..]),
        "duel" => run_duel_cmd(&args[1..]),
        other => panic!("unknown subcommand `{other}` (expected seam | calibrate | duel)"),
    }
}
