//! Dataset generator & M17e anchor evaluator for the residual NN (`EvalResidualWeights`
//! and `RolloutResidualWeights`).
//!
//! Generates stratified positions from:
//! - `src = 0`: Meta-pool self-play (`data/meta-pool-v0/meta-pool.json` + `data/eco-pool/master-pool.json`)
//! - `src = 1`: Tactical / setup / Baton Pass / Substitute / phazing / Spikes / Perish forced-exposure branches
//! - `src = 2`: Reconstructed human spectator corpus (`tmp/corpus-spectator`)
//!
//! For each position, records:
//! - `[f32; 32]` perspective feature vectors `x0`, `x1`
//! - `base_logit` from `compact_base_logit_and_eff`
//! - `v_mcts`: root value from a deep `SkuctSearch` (`--teacher-iters`, default 800)
//! - `v_td`: multi-step continuation value over `--playouts` (default 4) independent
//!   6-turn `skuct(--cont-iters)` trajectories bootstrapped with `SkuctSearch`
//! - Rollout action features `(phi_state, act_feats, base_scores, mcts_visits, mcts_q)`
//!   for both sides when `request == 1`
//!
//! Also supports `--check-anchors tmp/m17e-merged-v4.json` to evaluate the 72 certified
//! M17e endgame anchors against the active `ResidualBundle` (`NC2000_RESIDUAL_JSON`).

use std::io::{BufWriter, Write as _};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::agent::Agent;
use nc2000_bot::corpus::{
    corpus_files, load_battle, load_sources, reconstruct, reconstruct_context_with_pool,
};
use nc2000_bot::eval::{
    compact_base_logit_and_eff, compact_eval01, eval01, eval_leaf, extract_eval_features,
    extract_rollout_state_and_actions, EvalWeights,
};
use nc2000_bot::m17e_artifact::MergedArtifact;
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::rng::SplitMix64;
use nc2000_bot::smmcts::{RmAgent, RmConfig, SelRule, SkuctSearch};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::compact::{vol_bit, CompactBattle};
use nc2000_engine::state::Battle;
use serde::Serialize;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn skuct_cfg(iters: u32, c: f64) -> RmConfig {
    let mut cfg = RmConfig {
        iterations: iters,
        rule: SelRule::Ucb,
        c,
        hp_buckets: 16,
        rollout_residual: false,
        ..Default::default()
    };
    if let nc2000_bot::mcts::Playout::Heavy { ref mut weights, .. } = cfg.playout {
        weights.residual_scale = 0.0;
    }
    cfg
}

fn skuct(iters: u32, c: f64, seed: u64) -> RmAgent {
    RmAgent::new(skuct_cfg(iters, c), seed)
}

#[derive(Serialize)]
struct ActionSideRecord {
    side: u8,
    phi: [f32; 12],
    n_legal: usize,
    act_feats: Vec<[f32; 6]>,
    base_scores: Vec<f32>,
    visits: Vec<u32>,
    q_values: Vec<f32>,
}

#[derive(Serialize)]
struct DatasetRow {
    id: usize,
    src: u8,
    game: usize,
    turn: u16,
    tags: Vec<String>,
    base_logit: f32,
    base_eval: f32,
    v_mcts: f32,
    v_td: f32,
    x0: [f32; 32],
    x1: [f32; 32],
    actions: Vec<ActionSideRecord>,
}

struct RawPos {
    battle: Battle,
    src: u8,
    game: usize,
    turn: u16,
}

fn norm_id(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn is_tactical_move_id(id: &str) -> bool {
    matches!(
        id,
        "batonpass"
            | "swordsdance"
            | "bellydrum"
            | "curse"
            | "agility"
            | "amnesia"
            | "growth"
            | "meditate"
            | "sharpen"
            | "substitute"
            | "spikes"
            | "whirlwind"
            | "roar"
            | "encore"
            | "disable"
            | "perishsong"
            | "meanlook"
            | "spiderweb"
            | "leechseed"
            | "Counter"
            | "mirrorcoat"
            | "destinybond"
            | "psychup"
            | "haze"
            | "reflect"
            | "lightscreen"
            | "safeguard"
    )
}

fn has_tactical_move(team: &[PokemonSet]) -> bool {
    team.iter().any(|p| {
        p.moves
            .iter()
            .any(|m| is_tactical_move_id(&norm_id(m)))
    })
}

fn has_baton_pass_or_setup(team: &[PokemonSet]) -> bool {
    team.iter().any(|p| {
        let mut has_bp = false;
        let mut has_setup = false;
        for m in &p.moves {
            let id = norm_id(m);
            if id == "batonpass" {
                has_bp = true;
            }
            if matches!(
                id.as_str(),
                "swordsdance" | "bellydrum" | "curse" | "agility" | "amnesia" | "growth" | "substitute"
            ) {
                has_setup = true;
            }
        }
        has_bp || has_setup
    })
}

fn compute_tags(cb: &CompactBattle, table: &nc2000_engine::compact::MatchTable) -> Vec<String> {
    let mut tags = Vec::new();
    let mut any_boost = false;
    let mut any_sub = false;
    let mut any_bp = false;
    let mut any_spikes = false;
    let mut any_perish_or_trap = false;

    for s in 0..2usize {
        let act_slot = cb.sides[s].active_idx.min(2) as usize;
        let idx = s * 3 + act_slot;
        let mon = &cb.mons[idx];
        if mon.boosts.iter().any(|&b| b != 0) {
            any_boost = true;
        }
        if (mon.vol_flags & vol_bit::SUBSTITUTE) != 0 && mon.sub_hp() > 0 {
            any_sub = true;
        }
        if (mon.vol_flags & (vol_bit::PERISH_SONG | vol_bit::TRAPPED_OR_LOCKED)) != 0 {
            any_perish_or_trap = true;
        }
        if (cb.sides[s].side_flags & 0x03) > 0 {
            any_spikes = true;
        }
        let nm = table.mons[idx].num_moves as usize;
        for m in 0..nm {
            if mon.pp[m] > 0 && table.move_specs[idx][m].self_switch {
                any_bp = true;
            }
        }
    }
    if any_boost {
        tags.push("boosted".to_string());
    } else {
        tags.push("unboosted".to_string());
    }
    if any_sub {
        tags.push("has_substitute".to_string());
    }
    if any_bp {
        tags.push("has_self_switch".to_string());
    }
    if any_boost && any_bp {
        tags.push("boosted_self_switch".to_string());
    }
    if any_spikes {
        tags.push("has_spikes".to_string());
    }
    if any_perish_or_trap {
        tags.push("perish_or_trap".to_string());
    }
    let a0 = cb.sides[0].alive_mask.count_ones();
    let a1 = cb.sides[1].alive_mask.count_ones();
    if a0 != a1 {
        tags.push("alive_diff".to_string());
    }
    if a0 == 1 && a1 == 1 {
        tags.push("1v1".to_string());
    }
    tags
}

fn check_anchors(artifact_path: &str, corpus: &str, eps: f64, res_scale: f64) {
    let text = std::fs::read_to_string(artifact_path)
        .unwrap_or_else(|e| panic!("{artifact_path}: {e}"));
    let artifact: MergedArtifact =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{artifact_path}: {e}"));
    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let files = corpus_files(&root.join(corpus));

    let w_base = EvalWeights {
        residual_scale: 0.0,
        ..EvalWeights::default()
    };
    let w_cand = EvalWeights {
        residual_scale: res_scale,
        ..EvalWeights::default()
    };

    let mut by_battle: std::collections::HashMap<usize, Vec<&nc2000_bot::m17e_artifact::Row>> =
        std::collections::HashMap::new();
    for a in &artifact.rows {
        by_battle.entry(a.battle).or_default().push(a);
    }
    let mut battles: Vec<usize> = by_battle.keys().copied().collect();
    battles.sort();

    for (label, w) in [("baseline (res=0)", &w_base), ("candidate", &w_cand)] {
        let mut viols = 0usize;
        let mut total_margin = 0.0f64;
        let mut tight_absdev = 0.0f64;
        let mut tight_n = 0usize;
        let mut leaf_viols = 0usize;
        let mut leaf_margin = 0.0f64;

        for &bi in &battles {
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
                let (lo, hi) = (a.exact - a.width / 2.0, a.exact + a.width / 2.0);

                let e_raw = eval01(&b, &dex, w);
                let v_raw = (e_raw - hi).max(lo - e_raw).max(0.0);
                if v_raw > eps {
                    viols += 1;
                    total_margin += v_raw;
                }
                if a.width <= 0.05 {
                    tight_absdev += (e_raw - a.exact).abs();
                    tight_n += 1;
                }

                let e_leaf = eval_leaf(&b, &dex, w);
                let v_leaf = (e_leaf - hi).max(lo - e_leaf).max(0.0);
                if v_leaf > eps {
                    leaf_viols += 1;
                    leaf_margin += v_leaf;
                }
            }
        }
        println!(
            "{label:18} | raw: viols {viols:2} margin {total_margin:.4} tight_MAE {:.4} (n={tight_n}) | leaf: viols {leaf_viols:2} margin {leaf_margin:.4}",
            tight_absdev / tight_n.max(1) as f64
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(pos) = args.iter().position(|a| a == "--check-anchors") {
        let path = args
            .get(pos + 1)
            .map(|s| s.as_str())
            .unwrap_or("tmp/m17e-merged-v4.json");
        let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
        let eps: f64 = arg_s(&args, "--eps", "0.02").parse().unwrap();
        let res_scale: f64 = arg_s(&args, "--residual-scale", "1.0").parse().unwrap();
        let _ = nc2000_bot::eval::active_residual_bundle();
        check_anchors(path, &corpus, eps, res_scale);
        return;
    }

    let meta_games = arg(&args, "--meta-games", 350);
    let tac_games = arg(&args, "--tac-games", 450);
    let per_game = arg(&args, "--per-game", 6);
    let corpus_battles = arg(&args, "--corpus-battles", 300);
    let gen_iters = arg(&args, "--gen-iters", 150) as u32;
    let teacher_iters = arg(&args, "--teacher-iters", 800) as u32;
    let cont_iters = arg(&args, "--cont-iters", 150) as u32;
    let cont_turns = arg(&args, "--cont-turns", 6) as u16;
    let playouts = arg(&args, "--playouts", 4);
    let seed = arg(&args, "--seed", 20261029) as u64;
    let out_path = arg_s(&args, "--out", "tmp/residual-dataset.jsonl");
    let threads = arg(
        &args,
        "--threads",
        std::thread::available_parallelism()
            .map(|n| n.get().max(1))
            .unwrap_or(8),
    );

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let meta_pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let master_pool = load_meta_pool(&root.join("data/eco-pool/master-pool.json"));

    let mut all_teams: Vec<Vec<PokemonSet>> =
        meta_pool.teams.iter().map(|t| t.sets.clone()).collect();
    for t in &master_pool.teams {
        all_teams.push(t.sets.clone());
    }
    let tac_indices: Vec<usize> = (0..all_teams.len())
        .filter(|&i| has_baton_pass_or_setup(&all_teams[i]) || has_tactical_move(&all_teams[i]))
        .collect();
    let bp_indices: Vec<usize> = (0..all_teams.len())
        .filter(|&i| {
            all_teams[i]
                .iter()
                .any(|p| p.moves.iter().any(|m| norm_id(m) == "batonpass"))
        })
        .collect();

    eprintln!(
        "pools: {} total teams ({} tactical, {} batonpass) | meta_games={meta_games} tac_games={tac_games} corpus_battles={corpus_battles} teacher_iters={teacher_iters} playouts={playouts}x{cont_turns}t@{cont_iters} threads={threads}",
        all_teams.len(),
        tac_indices.len(),
        bp_indices.len()
    );

    let raw_positions: Mutex<Vec<RawPos>> = Mutex::new(Vec::new());

    // ---- Phase 1a: Standard meta self-play (src = 0)
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let g = cursor.fetch_add(1, Ordering::Relaxed);
                if g >= meta_games {
                    return;
                }
                let mut rng = SplitMix64::new(seed ^ (g as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let a = rng.below(all_teams.len());
                let mut bteam = rng.below(all_teams.len() - 1);
                if bteam >= a {
                    bteam += 1;
                }
                let bs = rng.battle_seed();
                let Ok(mut b) =
                    Battle::from_fixture(&dex, &bs, &all_teams[a], &all_teams[bteam])
                else {
                    continue;
                };
                b.set_log_enabled(false);
                let mut agents = [
                    skuct(gen_iters, 1.0, rng.next()),
                    skuct(gen_iters, 1.0, rng.next()),
                ];
                let mut reservoir: Vec<(Battle, u16)> = Vec::new();
                let mut seen = 0usize;
                loop {
                    if b.outcome().is_some() || b.turn > 80 {
                        break;
                    }
                    if b.turn >= 1 && b.active_id(0).is_some() && b.active_id(1).is_some() {
                        seen += 1;
                        if reservoir.len() < per_game {
                            reservoir.push((b.clone(), b.turn));
                        } else if rng.next_f64() < per_game as f64 / seen as f64 {
                            let slot = rng.below(per_game);
                            reservoir[slot] = (b.clone(), b.turn);
                        }
                    }
                    let mut picks = [None, None];
                    for s in 0..2 {
                        let cs = b.legal_choices(&dex, s);
                        if !cs.is_empty() {
                            picks[s] = Some(agents[s].choose(&b, &dex, s, &cs));
                        }
                    }
                    if picks == [None, None] || b.apply_choices(&dex, picks).is_err() {
                        break;
                    }
                }
                let mut out = raw_positions.lock().unwrap();
                for (battle, turn) in reservoir {
                    out.push(RawPos {
                        battle,
                        src: 0,
                        game: g,
                        turn,
                    });
                }
            });
        }
    });

    // ---- Phase 1b: Tactical / Setup / Baton Pass / Substitute / Phazing exposure (src = 1)
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let g = cursor.fetch_add(1, Ordering::Relaxed);
                if g >= tac_games {
                    return;
                }
                let mut rng = SplitMix64::new(
                    seed ^ 0xBF58_476D_1CE4_E5B9 ^ (g as u64).wrapping_mul(0x94D0_49BB_1331_11EB),
                );
                let a = if !bp_indices.is_empty() && g % 2 == 0 {
                    bp_indices[rng.below(bp_indices.len())]
                } else {
                    tac_indices[rng.below(tac_indices.len())]
                };
                let bteam = rng.below(all_teams.len());
                if a == bteam {
                    continue;
                }
                let (t0, t1) = if g % 2 == 0 {
                    (&all_teams[a], &all_teams[bteam])
                } else {
                    (&all_teams[bteam], &all_teams[a])
                };
                let bs = rng.battle_seed();
                let Ok(mut b) = Battle::from_fixture(&dex, &bs, t0, t1) else {
                    continue;
                };
                b.set_log_enabled(false);
                let mut agents = [
                    skuct(gen_iters, 1.0, rng.next()),
                    skuct(gen_iters, 1.0, rng.next()),
                ];
                let teams_pair = [t0, t1];
                let mut collected: Vec<(Battle, u16)> = Vec::new();
                let cap = per_game + 2;
                loop {
                    if b.outcome().is_some() || b.turn > 70 {
                        break;
                    }
                    let mut picks = [None, None];
                    let mut legal_per_side = [Vec::new(), Vec::new()];
                    for s in 0..2 {
                        let cs = b.legal_choices(&dex, s);
                        if !cs.is_empty() {
                            if matches!(cs.first(), Some(SearchChoice::Team(_))) {
                                let bp_lead = teams_pair[s]
                                    .iter()
                                    .position(|p| p.moves.iter().any(|m| norm_id(m) == "batonpass"))
                                    .or_else(|| {
                                        teams_pair[s].iter().position(|p| {
                                            p.moves.iter().any(|m| {
                                                matches!(
                                                    norm_id(m).as_str(),
                                                    "swordsdance"
                                                        | "bellydrum"
                                                        | "agility"
                                                        | "amnesia"
                                                        | "substitute"
                                                )
                                            })
                                        })
                                    })
                                    .map(|i| i as u8 + 1);
                                if let Some(lead) = bp_lead {
                                    let sub: Vec<SearchChoice> = cs
                                        .iter()
                                        .copied()
                                        .filter(|c| {
                                            matches!(c, SearchChoice::Team(t) if t.contains(&lead))
                                        })
                                        .collect();
                                    if !sub.is_empty() {
                                        picks[s] = Some(agents[s].choose(&b, &dex, s, &sub));
                                        legal_per_side[s] = cs;
                                        continue;
                                    }
                                }
                            }
                            picks[s] = Some(agents[s].choose(&b, &dex, s, &cs));
                        }
                        legal_per_side[s] = cs;
                    }
                    if picks == [None, None] {
                        break;
                    }

                    // If we are in an active battle turn, check for tactical moves to branch/expose
                    if b.turn >= 1 && b.active_id(0).is_some() && b.active_id(1).is_some() {
                        for s in 0..2usize {
                            let Some(att_id) = b.active_id(s) else {
                                continue;
                            };
                            let att = b.poke(att_id);
                            if att.hp * 10 <= att.maxhp * 6 {
                                continue;
                            }
                            let has_boost = att.boosts.iter().any(|&x| x > 0);
                            let tac_choices: Vec<SearchChoice> = legal_per_side[s]
                                .iter()
                                .copied()
                                .filter(|c| match c {
                                    SearchChoice::Move(mid) => {
                                        let k = dex.moves.key(*mid);
                                        if k == "batonpass" {
                                            has_boost
                                        } else {
                                            is_tactical_move_id(k)
                                        }
                                    }
                                    _ => false,
                                })
                                .collect();
                            if !tac_choices.is_empty() && rng.next_f64() < 0.35 {
                                // Snapshot the decision state before the tactical move
                                if collected.len() < cap {
                                    collected.push((b.clone(), b.turn));
                                }
                                // Also branch by executing one of the tactical moves so post-setup /
                                // post-Substitute / post-Baton-Pass states are well represented!
                                let chosen_tac = tac_choices[rng.below(tac_choices.len())];
                                let mut b_branch = b.clone();
                                let mut branch_picks = picks;
                                branch_picks[s] = Some(chosen_tac);
                                if b_branch.apply_choices(&dex, branch_picks).is_ok() {
                                    // Resolve any mid-turn switch request (e.g. Baton Pass)
                                    while b_branch.outcome().is_none()
                                        && (b_branch.sides[0].request_state()
                                            == Some(nc2000_engine::state::RequestKind::Switch)
                                            || b_branch.sides[1].request_state()
                                                == Some(nc2000_engine::state::RequestKind::Switch))
                                    {
                                        let mut sw_picks = [None, None];
                                        for sw_s in 0..2 {
                                            let cs = b_branch.legal_choices(&dex, sw_s);
                                            if !cs.is_empty() {
                                                sw_picks[sw_s] = Some(cs[rng.below(cs.len())]);
                                            }
                                        }
                                        if sw_picks == [None, None]
                                            || b_branch.apply_choices(&dex, sw_picks).is_err()
                                        {
                                            break;
                                        }
                                    }
                                    if b_branch.outcome().is_none()
                                        && b_branch.active_id(0).is_some()
                                        && b_branch.active_id(1).is_some()
                                    {
                                        if collected.len() < cap {
                                            collected.push((b_branch.clone(), b_branch.turn));
                                        }
                                        // With probability 0.4, continue from the tactical branch!
                                        if rng.next_f64() < 0.4 {
                                            b = b_branch;
                                            continue;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if b.apply_choices(&dex, picks).is_err() {
                        break;
                    }
                }
                let mut out = raw_positions.lock().unwrap();
                for (battle, turn) in collected.into_iter().take(cap) {
                    out.push(RawPos {
                        battle,
                        src: 1,
                        game: g,
                        turn,
                    });
                }
            });
        }
    });

    // ---- Phase 1c: Human spectator corpus positions (src = 2)
    let corpus_dir = root.join("tmp/corpus-spectator");
    if corpus_dir.exists() && corpus_battles > 0 {
        let sets_src = load_sources(&dex, &root);
        let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&corpus_dir)
            .into_iter()
            .enumerate()
            .take(corpus_battles)
            .collect();
        let ccursor = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| loop {
                    let j = ccursor.fetch_add(1, Ordering::Relaxed);
                    if j >= files.len() {
                        return;
                    }
                    let (battle_idx, path) = &files[j];
                    let cb = load_battle(path);
                    let mut rng =
                        SplitMix64::new(seed ^ (*battle_idx as u64).wrapping_mul(0xD1B5_4A32));
                    let mut reservoir: Vec<(Battle, u16)> = Vec::new();
                    let mut seen = 0usize;
                    let cap = 4usize;
                    for (di, d) in cb.decisions.iter().enumerate() {
                        let dseed = seed
                            ^ (*battle_idx as u64).wrapping_mul(0x9E37_79B9_7F4A)
                            ^ (di as u64).wrapping_mul(0xBF58_476D)
                            ^ d.side as u64;
                        let Some(rec) = reconstruct_context_with_pool(
                            &dex,
                            &sets_src,
                            meta_pool.clone(),
                            &cb.lines,
                            &cb.evidence,
                            d,
                            dseed,
                        ) else {
                            continue;
                        };
                        let Some(battle) = rec.agent.battle().cloned() else {
                            continue;
                        };
                        if battle.active_id(0).is_none() || battle.active_id(1).is_none() {
                            continue;
                        }
                        seen += 1;
                        let turn = battle.turn;
                        if reservoir.len() < cap {
                            reservoir.push((battle, turn));
                        } else if rng.next_f64() < cap as f64 / seen as f64 {
                            let slot = rng.below(cap);
                            reservoir[slot] = (battle, turn);
                        }
                    }
                    let mut out = raw_positions.lock().unwrap();
                    for (battle, turn) in reservoir {
                        out.push(RawPos {
                            battle,
                            src: 2,
                            game: *battle_idx,
                            turn,
                        });
                    }
                });
            }
        });
    }

    let mut positions = raw_positions.into_inner().unwrap();
    positions.sort_by_key(|p| (p.src, p.game, p.turn));
    let n = positions.len();
    eprintln!("Phase 1 done: {n} positions collected. Running Phase 2 (MCTS teacher + TD continuations)...");

    let w_zero = EvalWeights {
        residual_scale: 0.0,
        ..EvalWeights::default()
    };
    let rows: Mutex<Vec<DatasetRow>> = Mutex::new(Vec::with_capacity(n));
    let pcursor = AtomicUsize::new(0);
    let pdone = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let idx = pcursor.fetch_add(1, Ordering::Relaxed);
                if idx >= n {
                    return;
                }
                let pos = &positions[idx];
                let (table, cb) = CompactBattle::from_battle(&pos.battle, &dex);
                if cb.outcome != 0 {
                    continue;
                }
                if cb.sides[0].request == 3 || cb.sides[1].request == 3 {
                    continue;
                }

                let (base_logit_f64, _act_eff) = compact_base_logit_and_eff(&cb, &table, &w_zero);
                let base_eval_f64 = compact_eval01(&cb, &table, &w_zero);
                let x0 = extract_eval_features(&cb, &table, 0);
                let x1 = extract_eval_features(&cb, &table, 1);
                let tags = compute_tags(&cb, &table);

                // 1. Deep MCTS teacher search at root
                let t_seed = seed ^ 0xA5A5_5A5A_F0F0_0F0F ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let mut ts = SkuctSearch::new(&pos.battle, &dex, skuct_cfg(teacher_iters, 1.0), t_seed);
                ts.step(&dex, teacher_iters);

                let visits0 = ts.visits(0);
                let means0 = ts.means(0);
                let visits1 = ts.visits(1);
                let means1 = ts.means(1);

                let v_mcts = if !visits0.is_empty() {
                    let tot: u32 = visits0.iter().sum();
                    if tot > 0 {
                        visits0
                            .iter()
                            .zip(means0.iter())
                            .map(|(&v, &m)| (v as f64) * m)
                            .sum::<f64>()
                            / (tot as f64)
                    } else {
                        base_eval_f64
                    }
                } else if !visits1.is_empty() {
                    let tot: u32 = visits1.iter().sum();
                    if tot > 0 {
                        1.0 - visits1
                            .iter()
                            .zip(means1.iter())
                            .map(|(&v, &m)| (v as f64) * m)
                            .sum::<f64>()
                            / (tot as f64)
                    } else {
                        base_eval_f64
                    }
                } else {
                    base_eval_f64
                };

                // Extract rollout action records for sides with request == 1
                let mut action_records = Vec::new();
                for s in 0..2usize {
                    if cb.sides[s].request != 1 {
                        continue;
                    }
                    let foe_s = s ^ 1;
                    let act_slot = cb.sides[s].active_idx.min(2) as usize;
                    let att_idx = s * 3 + act_slot;
                    let def_slot = cb.sides[foe_s].active_idx.min(2) as usize;

                    let mut c_actions = [0u8; 7];
                    let n_legal = cb.legal_actions(&table, s, &mut c_actions);
                    if n_legal <= 1 {
                        continue;
                    }

                    let mut base_scores = Vec::with_capacity(n_legal);
                    let mut stay_best_dmg = 0.0f32;
                    for i in 0..n_legal {
                        let code = c_actions[i];
                        let b_score = if code < 4 {
                            let m_idx = code as usize;
                            let ms = &table.move_specs[att_idx][m_idx];
                            if ms.is_status || ms.base_power == 0 {
                                0.0
                            } else {
                                cb.expected_hit_fraction(
                                    &table, s, act_slot, foe_s, def_slot, m_idx, false,
                                ) as f32
                            }
                        } else {
                            0.0
                        };
                        if b_score > stay_best_dmg {
                            stay_best_dmg = b_score;
                        }
                        base_scores.push(b_score);
                    }

                    let Some((phi, act_feats_arr)) = extract_rollout_state_and_actions(
                        &cb,
                        &table,
                        s,
                        &c_actions[..n_legal],
                        stay_best_dmg,
                    ) else {
                        continue;
                    };

                    let ts_acts = ts.actions(s);
                    let ts_vis = ts.visits(s);
                    let ts_m = ts.means(s);

                    let mut act_feats = Vec::with_capacity(n_legal);
                    let mut visits = Vec::with_capacity(n_legal);
                    let mut q_values = Vec::with_capacity(n_legal);

                    for i in 0..n_legal {
                        act_feats.push(act_feats_arr[i]);
                        let code = c_actions[i];
                        let expected_sc = if code < 4 {
                            SearchChoice::Move(table.mons[att_idx].moves[code as usize])
                        } else {
                            SearchChoice::Switch(code - 4 + 1)
                        };
                        if let Some(pos_a) = ts_acts.iter().position(|&x| x == expected_sc) {
                            visits.push(ts_vis[pos_a]);
                            q_values.push(ts_m[pos_a] as f32);
                        } else {
                            visits.push(0);
                            q_values.push(0.5);
                        }
                    }

                    let tot_v: u32 = visits.iter().sum();
                    if tot_v > 0 {
                        action_records.push(ActionSideRecord {
                            side: s as u8,
                            phi,
                            n_legal,
                            act_feats,
                            base_scores,
                            visits,
                            q_values,
                        });
                    }
                }

                // 2. Multi-step TD continuation rollouts (playouts x cont_turns of skuct + search bootstrap)
                let mut td_sum = 0.0f64;
                for k in 0..playouts {
                    let mut rng = SplitMix64::new(
                        seed
                            ^ 0xC0FF_EE00_1234_5678
                            ^ ((idx as u64) << 16)
                            ^ (k as u64).wrapping_mul(0x94D0_49BB_1331_11EB),
                    );
                    let mut b = pos.battle.clone();
                    b.reseed(rng.next());
                    let mut a0 = skuct(cont_iters, 1.0, rng.next());
                    let mut a1 = skuct(cont_iters, 1.0, rng.next());
                    let start_turn = b.turn;
                    while b.outcome().is_none() && b.turn < start_turn.saturating_add(cont_turns) {
                        let mut picks = [None, None];
                        for s in 0..2 {
                            let cs = b.legal_choices(&dex, s);
                            if !cs.is_empty() {
                                let chosen = if s == 0 {
                                    a0.choose(&b, &dex, 0, &cs)
                                } else {
                                    a1.choose(&b, &dex, 1, &cs)
                                };
                                picks[s] = Some(chosen);
                            }
                        }
                        if picks == [None, None] || b.apply_choices(&dex, picks).is_err() {
                            break;
                        }
                    }
                    let val = match b.outcome() {
                        Some(Outcome::P1Win) => 1.0,
                        Some(Outcome::P2Win) => 0.0,
                        Some(Outcome::Tie) => 0.5,
                        None => {
                            let mut leaf_ts = SkuctSearch::new(
                                &b,
                                &dex,
                                skuct_cfg(cont_iters * 2, 1.0),
                                rng.next(),
                            );
                            leaf_ts.step(&dex, cont_iters * 2);
                            let v0 = leaf_ts.visits(0);
                            let m0 = leaf_ts.means(0);
                            let tot: u32 = v0.iter().sum();
                            if tot > 0 {
                                v0.iter()
                                    .zip(m0.iter())
                                    .map(|(&v, &m)| (v as f64) * m)
                                    .sum::<f64>()
                                    / (tot as f64)
                            } else {
                                eval01(&b, &dex, &w_zero)
                            }
                        }
                    };
                    td_sum += val;
                }
                let v_td = if playouts > 0 {
                    td_sum / (playouts as f64)
                } else {
                    v_mcts
                };

                rows.lock().unwrap().push(DatasetRow {
                    id: idx,
                    src: pos.src,
                    game: pos.game,
                    turn: pos.turn,
                    tags,
                    base_logit: base_logit_f64 as f32,
                    base_eval: base_eval_f64 as f32,
                    v_mcts: v_mcts as f32,
                    v_td: v_td as f32,
                    x0,
                    x1,
                    actions: action_records,
                });

                let d = pdone.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 250 == 0 || d == n {
                    eprintln!("  labeled {d}/{n} states");
                }
            });
        }
    });

    let mut rows = rows.into_inner().unwrap();
    rows.sort_by_key(|r| r.id);

    let file = std::fs::File::create(&out_path).unwrap();
    let mut writer = BufWriter::new(file);
    for r in &rows {
        serde_json::to_writer(&mut writer, r).unwrap();
        writer.write_all(b"\n").unwrap();
    }
    writer.flush().unwrap();
    eprintln!("Saved {} labeled rows to {out_path}", rows.len());
}
