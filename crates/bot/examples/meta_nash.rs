//! META-NASH v1 driver (docs/META-NASH-V1.md) — evolve candidate parties,
//! estimate the team-vs-team matchup matrix, solve the Nash mixture, and
//! measure the preregistered OR gate (diversity route A / strength route B).
//!
//! Modes (all resumable; artifacts under data/meta-nash-v1/):
//!   --init                        candidates.json from meta-pool-v0 (curated)
//!   --matrix                      fill missing pairwise cells (gauntlet_eval,
//!                                 skuct both sides, seed-paired; one JSON per
//!                                 unordered pair per --iters; --cell-games)
//!   --solve --tag T               Nash over all candidates; --restrict-evolved
//!                                 = maximin of evolved rows vs all columns;
//!                                 --smooth E blends toward uniform rows
//!   --br --target SOL --lineage L (1+λ) hill-climb vs SOL's weighted support
//!                                 (--seed-team ID | --random); CRN per
//!                                 iteration + holdout; stateless-seed resume
//!   --harvest                     accepted lineage teams -> candidates.json
//!                                 (signature dedupe vs curated + each other)
//!   --duel --x SPEC --y SPEC      pool-vs-pool seed-paired duel; SPEC =
//!                                 sol:FILE | uniform-curated | lineage:FILE
//!                                 (single-team pool from the checkpoint best)
//!   --export --pool SOL           final {teams, weights} product artifact
//!
//! Agents everywhere are the M11a fitness convention: RmAgent skuct
//! (`RmConfig { iterations, rule: Ucb, ..Default }`), both sides equal.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::smmcts::{solve_rm_plus, SelRule};
use nc2000_bot::teamgen::{gauntlet_eval, to_sets, EvalCfg, TeamGen};
use nc2000_bot::{play_game, Agent, GameResult, RmAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::{json, Value};

// ------------------------------------------------------------- candidates

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct CandTeam {
    id: String,
    origin: String, // "curated" | "evolved:<lineage>"
    sets: Vec<PokemonSet>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Candidates {
    teams: Vec<CandTeam>,
}

fn candidates_path(root: &Path) -> PathBuf {
    root.join("data/meta-nash-v1/candidates.json")
}

fn load_candidates(root: &Path) -> Candidates {
    let p = candidates_path(root);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!("read {} (run --init first): {e}", p.display())
    }))
    .expect("candidates.json parse")
}

fn save_candidates(root: &Path, c: &Candidates) {
    let p = candidates_path(root);
    atomic_write(&p, &serde_json::to_string_pretty(c).unwrap());
}

fn atomic_write(path: &Path, content: &str) {
    let tmp = path.with_extension("tmp");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&tmp, content).unwrap();
    std::fs::rename(&tmp, path).unwrap();
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Canonical team signature for template/dedupe checks: mons sorted by
/// species, moves sorted, the identity-bearing set fields only.
fn team_signature(sets: &[PokemonSet]) -> String {
    let mut mons: Vec<String> = sets
        .iter()
        .map(|s| {
            let mut moves = s.moves.clone();
            moves.sort();
            format!(
                "{}|{}|{}|{}|{}|{:?}|{:?}|{:?}",
                s.species,
                s.level,
                s.item,
                s.gender.clone().unwrap_or_default(),
                moves.join(","),
                s.ivs,
                s.evs,
                s.happiness
            )
        })
        .collect();
    mons.sort();
    mons.join("||")
}

// ------------------------------------------------------------------ cells

fn cell_path(root: &Path, a: &str, b: &str, iters: u32) -> PathBuf {
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    root.join(format!("data/meta-nash-v1/cells/cell-{x}__vs__{y}-i{iters}.json"))
}

fn matrix_mode(root: &Path, dex: &Dex, args: &[String]) {
    let iters: u32 = flag(args, "--iters").map(|v| v.parse().unwrap()).unwrap_or(300);
    let cell_games: u32 =
        flag(args, "--cell-games").map(|v| v.parse().unwrap()).unwrap_or(64);
    let base_seed: u64 = flag(args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(20260812);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let cands = load_candidates(root);
    let k = cands.teams.len();
    // pairs missing a cell, or holding fewer games than requested (refine)
    let mut todo: Vec<(usize, usize)> = Vec::new();
    for i in 0..k {
        for j in i + 1..k {
            let p = cell_path(root, &cands.teams[i].id, &cands.teams[j].id, iters);
            let have = std::fs::read_to_string(&p)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(&t).ok())
                .and_then(|v| v["games"].as_u64())
                .unwrap_or(0);
            if (have as u32) < cell_games {
                todo.push((i, j));
            }
        }
    }
    eprintln!(
        "matrix: {k} teams, {} pairs, {} to run (iters {iters}, {cell_games} games/cell)",
        k * (k - 1) / 2,
        todo.len()
    );
    let t0 = Instant::now();
    for (n, (i, j)) in todo.iter().enumerate() {
        let (a, b) = (&cands.teams[*i], &cands.teams[*j]);
        let (lo, hi) = if a.id <= b.id { (&a.id, &b.id) } else { (&b.id, &a.id) };
        let seed = base_seed ^ fnv1a64(&format!("{lo}|{hi}"));
        let res = gauntlet_eval(
            dex,
            &a.sets,
            &[b.sets.clone()],
            &EvalCfg {
                games_per_opponent: cell_games,
                agent_iters: iters,
                max_turns: 500,
                threads,
                seed,
            },
        );
        let path = cell_path(root, &a.id, &b.id, iters);
        // stored orientation: score of the lexicographically SMALLER id
        let (sa, sb) = if a.id <= b.id { (a.id.clone(), b.id.clone()) } else { (b.id.clone(), a.id.clone()) };
        let score_small = if a.id <= b.id { res.score } else { 1.0 - res.score };
        atomic_write(
            &path,
            &serde_json::to_string(&json!({
                "a": sa, "b": sb, "score_a": score_small,
                "games": res.games, "iters": iters, "seed": seed,
            }))
            .unwrap(),
        );
        if (n + 1) % 10 == 0 || n + 1 == todo.len() {
            eprintln!(
                "  {}/{} cells ({:.0}s, {:.1}s/cell)",
                n + 1,
                todo.len(),
                t0.elapsed().as_secs_f64(),
                t0.elapsed().as_secs_f64() / (n + 1) as f64
            );
        }
    }
    eprintln!("matrix complete");
}

/// P row-major k x k: P[i*k+j] = score of team i vs team j (0.5 diagonal).
fn build_matrix(root: &Path, cands: &Candidates, iters: u32) -> Vec<f64> {
    let k = cands.teams.len();
    let mut m = vec![0.5; k * k];
    for i in 0..k {
        for j in i + 1..k {
            let p = cell_path(root, &cands.teams[i].id, &cands.teams[j].id, iters);
            let v: Value = serde_json::from_str(
                &std::fs::read_to_string(&p)
                    .unwrap_or_else(|e| panic!("missing cell {} : {e}", p.display())),
            )
            .unwrap();
            let score_small = v["score_a"].as_f64().unwrap();
            let (small_is_i, _) = if cands.teams[i].id <= cands.teams[j].id {
                (true, ())
            } else {
                (false, ())
            };
            let s_ij = if small_is_i { score_small } else { 1.0 - score_small };
            m[i * k + j] = s_ij;
            m[j * k + i] = 1.0 - s_ij;
        }
    }
    m
}

// ------------------------------------------------------------------ solve

fn solve_mode(root: &Path, args: &[String]) {
    let iters: u32 = flag(args, "--iters").map(|v| v.parse().unwrap()).unwrap_or(300);
    let tag = flag(args, "--tag").expect("--solve needs --tag");
    let sweeps: u32 = flag(args, "--sweeps").map(|v| v.parse().unwrap()).unwrap_or(200_000);
    let restrict_evolved = has(args, "--restrict-evolved");
    let eps: f64 = flag(args, "--smooth").map(|v| v.parse().unwrap()).unwrap_or(0.0);
    let cands = load_candidates(root);
    let k = cands.teams.len();
    let m = build_matrix(root, &cands, iters);
    // orientation sanity: a dominant row must get the weight
    {
        let test = [0.9, 0.9, 0.1, 0.1];
        let (p, _) = solve_rm_plus(&test, [2, 2], 10_000);
        assert!(p[0] > 0.95, "solve_rm_plus orientation drifted");
    }
    let rows: Vec<usize> = if restrict_evolved {
        (0..k).filter(|&i| cands.teams[i].origin.starts_with("evolved")).collect()
    } else {
        (0..k).collect()
    };
    assert!(!rows.is_empty(), "no rows to solve (no evolved teams yet?)");
    let sub: Vec<f64> = rows
        .iter()
        .flat_map(|&i| (0..k).map(move |j| (i, j)))
        .map(|(i, j)| m[i * k + j])
        .collect();
    let (mut w, _) = solve_rm_plus(&sub, [rows.len(), k], sweeps);
    if eps > 0.0 {
        let u = 1.0 / rows.len() as f64;
        for x in w.iter_mut() {
            *x = (1.0 - eps) * *x + eps * u;
        }
    }
    // guaranteed value = worst column against the row mixture
    let col_val = |j: usize| -> f64 {
        rows.iter().zip(&w).map(|(&i, wi)| wi * m[i * k + j]).sum()
    };
    let worst = (0..k).map(col_val).fold(f64::INFINITY, f64::min);
    let worst_j = (0..k).min_by(|&a, &b| col_val(a).total_cmp(&col_val(b))).unwrap();
    let active = w.iter().filter(|&&x| x >= 0.01).count();
    let mut support: Vec<(String, f64)> = rows
        .iter()
        .zip(&w)
        .filter(|(_, &wi)| wi > 1e-4)
        .map(|(&i, &wi)| (cands.teams[i].id.clone(), wi))
        .collect();
    support.sort_by(|a, b| b.1.total_cmp(&a.1));
    let sol = json!({
        "tag": tag, "iters": iters, "sweeps": sweeps,
        "restrict_evolved": restrict_evolved, "smooth": eps,
        "ids": rows.iter().map(|&i| cands.teams[i].id.clone()).collect::<Vec<_>>(),
        "weights": w,
        "support": support.iter().map(|(id, wi)| json!({"id": id, "w": wi})).collect::<Vec<_>>(),
        "value_worst_column": worst,
        "worst_column": cands.teams[worst_j].id,
        "br_margin_internal": 0.5 - worst,
        "active_teams_ge_1pct": active,
        "engine_commit": commit(root),
    });
    let path = root.join(format!("data/meta-nash-v1/solutions/{tag}.json"));
    atomic_write(&path, &serde_json::to_string_pretty(&sol).unwrap());
    eprintln!(
        "solved {tag}: {} rows, support {} (>=1% {}), worst column {} at {:.3} (margin {:.3})",
        rows.len(),
        support.len(),
        active,
        cands.teams[worst_j].id,
        worst,
        0.5 - worst
    );
    eprintln!("wrote {}", path.display());
}

// ------------------------------------------------------- weighted support

struct WeightedPool {
    ids: Vec<String>,
    teams: Vec<Vec<PokemonSet>>,
    weights: Vec<f64>,
}

/// Load a duel-able pool from a SPEC: sol:FILE | uniform-curated |
/// lineage:FILE (the checkpoint's current best as a single-team pool).
fn load_pool(_root: &Path, cands: &Candidates, spec: &str) -> WeightedPool {
    if spec == "uniform-curated" {
        let teams: Vec<&CandTeam> =
            cands.teams.iter().filter(|t| t.origin == "curated").collect();
        let n = teams.len();
        return WeightedPool {
            ids: teams.iter().map(|t| t.id.clone()).collect(),
            teams: teams.iter().map(|t| t.sets.clone()).collect(),
            weights: vec![1.0 / n as f64; n],
        };
    }
    if let Some(f) = spec.strip_prefix("lineage:") {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
        let team: Vec<Value> = v["parent"].as_array().unwrap().clone();
        let sets = to_sets(&team).unwrap();
        return WeightedPool {
            ids: vec![format!("lineage:{}", v["lineage"].as_str().unwrap_or("?"))],
            teams: vec![sets],
            weights: vec![1.0],
        };
    }
    let f = spec.strip_prefix("sol:").unwrap_or(spec);
    let v: Value = serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap();
    let by_id: HashMap<&str, &CandTeam> =
        cands.teams.iter().map(|t| (t.id.as_str(), t)).collect();
    let ids: Vec<String> =
        v["ids"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();
    let weights: Vec<f64> =
        v["weights"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let teams = ids
        .iter()
        .map(|id| by_id.get(id.as_str()).unwrap_or_else(|| panic!("unknown id {id}")).sets.clone())
        .collect();
    WeightedPool { ids, teams, weights }
}

/// Renormalized top support (w >= floor, cap n) — the BR fitness gauntlet.
fn support_of(pool: &WeightedPool, floor: f64, cap: usize) -> (Vec<Vec<PokemonSet>>, Vec<f64>) {
    let mut idx: Vec<usize> = (0..pool.weights.len()).filter(|&i| pool.weights[i] >= floor).collect();
    idx.sort_by(|&a, &b| pool.weights[b].total_cmp(&pool.weights[a]));
    idx.truncate(cap.max(1));
    let total: f64 = idx.iter().map(|&i| pool.weights[i]).sum();
    (
        idx.iter().map(|&i| pool.teams[i].clone()).collect(),
        idx.iter().map(|&i| pool.weights[i] / total).collect(),
    )
}

fn weighted_fitness(
    dex: &Dex,
    team: &[PokemonSet],
    gauntlet: &[Vec<PokemonSet>],
    gweights: &[f64],
    games_per_opponent: u32,
    agent_iters: u32,
    threads: usize,
    seed: u64,
) -> f64 {
    let res = gauntlet_eval(
        dex,
        team,
        gauntlet,
        &EvalCfg { games_per_opponent, agent_iters, max_turns: 500, threads, seed },
    );
    res.per_opponent.iter().zip(gweights).map(|(s, w)| s * w).sum()
}

// --------------------------------------------------------------------- br

#[allow(clippy::too_many_arguments)]
fn br_mode(root: &Path, dex: &Dex, args: &[String]) {
    let target = flag(args, "--target").expect("--br needs --target SOLFILE");
    let lineage = flag(args, "--lineage").expect("--br needs --lineage NAME");
    let iters: usize = flag(args, "--br-iters").map(|v| v.parse().unwrap()).unwrap_or(24);
    let lambda: usize = flag(args, "--lambda").map(|v| v.parse().unwrap()).unwrap_or(4);
    let games: u32 = flag(args, "--fit-games").map(|v| v.parse().unwrap()).unwrap_or(8);
    let agent_iters: u32 = flag(args, "--iters").map(|v| v.parse().unwrap()).unwrap_or(300);
    let base_seed: u64 = flag(args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(20260812);
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let cands = load_candidates(root);
    let pool = load_pool(root, &cands, &target);
    let (gauntlet, gweights) = support_of(&pool, 0.02, 12);
    eprintln!(
        "br {lineage}: target {target}, support {} teams, {games} games/opp, skuct:{agent_iters}",
        gauntlet.len()
    );
    let tg = TeamGen::new(
        dex,
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
        &std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap(),
    )
    .unwrap();
    let ck_path = root.join(format!("data/meta-nash-v1/lineages/{lineage}.json"));
    // resume or seed
    let (mut parent, mut history, start_iter, seed_desc): (Vec<Value>, Vec<Value>, usize, String) =
        if let Ok(t) = std::fs::read_to_string(&ck_path) {
            let v: Value = serde_json::from_str(&t).unwrap();
            (
                v["parent"].as_array().unwrap().clone(),
                v["history"].as_array().unwrap().clone(),
                v["iters_done"].as_u64().unwrap() as usize,
                v["seed_desc"].as_str().unwrap().to_string(),
            )
        } else if let Some(id) = flag(args, "--seed-team") {
            let t = cands.teams.iter().find(|t| t.id == id).expect("unknown --seed-team");
            let team: Vec<Value> =
                t.sets.iter().map(|s| serde_json::to_value(s).unwrap()).collect();
            (team, Vec::new(), 0, format!("candidate:{id}"))
        } else {
            let mut rng = SplitMix64::new(base_seed ^ fnv1a64(&lineage));
            let team = tg.random_team_valid(dex, &mut rng, 500).expect("random team");
            (team, Vec::new(), 0, "random".to_string())
        };
    for it in start_iter..iters {
        let eval_seed = base_seed ^ fnv1a64(&format!("{lineage}|eval|{it}"));
        let mut prop_rng = SplitMix64::new(base_seed ^ fnv1a64(&format!("{lineage}|prop|{it}")));
        let parent_sets = to_sets(&parent).unwrap();
        let parent_fit = weighted_fitness(
            dex, &parent_sets, &gauntlet, &gweights, games, agent_iters, threads, eval_seed,
        );
        let mut best_child: Option<(f64, Vec<Value>, String)> = None;
        for _ in 0..lambda {
            let Some(prop) = tg.propose_valid(dex, &parent, &mut prop_rng, 200) else { continue };
            let sets = to_sets(&prop.team).unwrap();
            let fit = weighted_fitness(
                dex, &sets, &gauntlet, &gweights, games, agent_iters, threads, eval_seed,
            );
            if best_child.as_ref().is_none_or(|(bf, _, _)| fit > *bf) {
                best_child = Some((fit, prop.team, format!("{:?}", prop.op)));
            }
        }
        let (accepted, child_fit, op) = match best_child {
            Some((fit, team, op)) if fit > parent_fit => {
                parent = team;
                (true, fit, op)
            }
            Some((fit, _, op)) => (false, fit, op),
            None => (false, f64::NAN, "none".into()),
        };
        history.push(json!({
            "iter": it, "parent_fit": parent_fit, "best_child_fit": child_fit,
            "accepted": accepted, "op": op,
            "team_after": if accepted { Some(parent.clone()) } else { None },
        }));
        atomic_write(
            &ck_path,
            &serde_json::to_string(&json!({
                "lineage": lineage, "target": target, "seed_desc": seed_desc,
                "iters_done": it + 1, "iters_requested": iters,
                "lambda": lambda, "fit_games": games, "agent_iters": agent_iters,
                "base_seed": base_seed,
                "parent": parent, "history": history,
            }))
            .unwrap(),
        );
        eprintln!(
            "  [{lineage} {it}] parent {parent_fit:.3} best-child {child_fit:.3} {}",
            if accepted { "ACCEPT" } else { "keep" }
        );
    }
    // holdout: parent at 3x games on a shifted seed
    let holdout_seed = base_seed ^ fnv1a64(&format!("{lineage}|holdout"));
    let parent_sets = to_sets(&parent).unwrap();
    let holdout = weighted_fitness(
        dex, &parent_sets, &gauntlet, &gweights, games * 3, agent_iters, threads, holdout_seed,
    );
    let t = std::fs::read_to_string(&ck_path).unwrap();
    let mut v: Value = serde_json::from_str(&t).unwrap();
    v["holdout"] = json!({"seed": holdout_seed, "games_per_opponent": games * 3, "fit": holdout});
    atomic_write(&ck_path, &serde_json::to_string(&v).unwrap());
    eprintln!("br {lineage} done: holdout {holdout:.3} -> {}", ck_path.display());
}

// ---------------------------------------------------------------- harvest

fn harvest_mode(root: &Path, args: &[String]) {
    let cap: usize = flag(args, "--cap").map(|v| v.parse().unwrap()).unwrap_or(4);
    let mut cands = load_candidates(root);
    let mut seen: HashMap<String, String> = cands
        .teams
        .iter()
        .map(|t| (team_signature(&t.sets), t.id.clone()))
        .collect();
    let dir = root.join("data/meta-nash-v1/lineages");
    let mut added = 0;
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|d| d.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    paths.sort();
    for p in paths {
        if p.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
        let lineage = v["lineage"].as_str().unwrap().to_string();
        // newest accepted teams first (strongest lineage points), then cap
        let mut teams: Vec<Vec<Value>> = v["history"]
            .as_array()
            .unwrap()
            .iter()
            .rev()
            .filter(|h| h["accepted"].as_bool() == Some(true))
            .filter_map(|h| h["team_after"].as_array().cloned())
            .collect();
        teams.truncate(cap);
        let mut k = 0;
        for team in teams {
            let sets = to_sets(&team).unwrap();
            let sig = team_signature(&sets);
            if seen.contains_key(&sig) {
                continue;
            }
            let id = format!("ev-{lineage}-{k}");
            seen.insert(sig, id.clone());
            cands.teams.push(CandTeam { id, origin: format!("evolved:{lineage}"), sets });
            k += 1;
            added += 1;
        }
    }
    save_candidates(root, &cands);
    let evolved = cands.teams.iter().filter(|t| t.origin.starts_with("evolved")).count();
    eprintln!("harvest: +{added} teams ({evolved} evolved / {} total)", cands.teams.len());
}

// ------------------------------------------------------------------- duel

fn duel_mode(root: &Path, dex: &Dex, args: &[String]) {
    let x_spec = flag(args, "--x").expect("--duel needs --x");
    let y_spec = flag(args, "--y").expect("--duel needs --y");
    let games: usize = flag(args, "--games").map(|v| v.parse().unwrap()).unwrap_or(800);
    let agent_iters: u32 = flag(args, "--iters").map(|v| v.parse().unwrap()).unwrap_or(300);
    let base_seed: u64 = flag(args, "--seed").map(|v| v.parse().unwrap()).unwrap_or(20260812);
    let label = flag(args, "--label").unwrap_or_else(|| "duel".into());
    let threads: usize = flag(args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let cands = load_candidates(root);
    let x = load_pool(root, &cands, &x_spec);
    let y = load_pool(root, &cands, &y_spec);
    let blocks = games.div_ceil(2);
    // block schedule: teams + battle seed, shared by both orientations
    let mut sched = Vec::with_capacity(blocks);
    for b in 0..blocks {
        let mut r = SplitMix64::new(base_seed ^ (b as u64 + 1).wrapping_mul(0xD1B5_4A32_D192_ED03));
        let xi = sample(&x.weights, &mut r);
        let yi = sample(&y.weights, &mut r);
        sched.push((xi, yi, r.battle_seed()));
    }
    let cursor = AtomicUsize::new(0);
    let t0 = Instant::now();
    let mut results: Vec<(usize, f64, u16)> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (sched, cursor, x, y) = (&sched, &cursor, &x, &y);
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let g = cursor.fetch_add(1, Ordering::Relaxed);
                    if g >= blocks * 2 {
                        break;
                    }
                    let (xi, yi, ref bseed) = sched[g / 2];
                    let x_is_p1 = g % 2 == 0;
                    let skuct = |seed: u64| {
                        RmAgent::new(
                            RmConfig {
                                iterations: agent_iters,
                                rule: SelRule::Ucb,
                                ..Default::default()
                            },
                            seed,
                        )
                    };
                    let mut ax = skuct(base_seed ^ (g as u64).wrapping_mul(0xA24B_AED4_963E_E407));
                    let mut ay = skuct(base_seed ^ (g as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25));
                    let (t1, t2) = if x_is_p1 {
                        (&x.teams[xi], &y.teams[yi])
                    } else {
                        (&y.teams[yi], &x.teams[xi])
                    };
                    let mut battle = Battle::from_fixture(dex, bseed, t1, t2).unwrap();
                    battle.set_log_enabled(false);
                    let res = {
                        let (p1, p2): (&mut dyn Agent, &mut dyn Agent) =
                            if x_is_p1 { (&mut ax, &mut ay) } else { (&mut ay, &mut ax) };
                        play_game(dex, &mut battle, &mut [p1, p2], 500).unwrap()
                    };
                    let p1s = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        _ => 0.5,
                    };
                    out.push((g, if x_is_p1 { p1s } else { 1.0 - p1s }, battle.turn));
                }
                out
            }));
        }
        let mut all = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap().into_iter().map(|(g, s, t)| (g, s, t)));
        }
        all.sort_by_key(|r| r.0);
        results = all;
    });
    let scores: Vec<f64> = results.iter().map(|r| r.1).collect();
    let block_means: Vec<f64> = scores.chunks_exact(2).map(|p| (p[0] + p[1]) / 2.0).collect();
    let n = block_means.len() as f64;
    let mean = block_means.iter().sum::<f64>() / n;
    let sd = (block_means.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let ci = 1.96 * sd / n.sqrt();
    let out = json!({
        "label": label, "x": x_spec, "y": y_spec, "games": scores.len(),
        "agent_iters": agent_iters, "base_seed": base_seed,
        "x_score": mean, "ci95": ci, "blocks": block_means.len(),
        "x_ids_used": x.ids.len(), "y_ids_used": y.ids.len(),
        "secs": t0.elapsed().as_secs_f64(),
        "game_scores": scores,
        "engine_commit": commit(root),
    });
    let path = root.join(format!("data/meta-nash-v1/gates/{label}.json"));
    atomic_write(&path, &serde_json::to_string(&out).unwrap());
    eprintln!(
        "duel {label}: X {mean:.4} ± {ci:.4} ({} games, {:.0}s) -> {}",
        scores.len(),
        t0.elapsed().as_secs_f64(),
        path.display()
    );
}

fn sample(w: &[f64], rng: &mut SplitMix64) -> usize {
    let u = rng.next_f64() * w.iter().sum::<f64>();
    let mut acc = 0.0;
    for (i, &wi) in w.iter().enumerate() {
        acc += wi;
        if u < acc {
            return i;
        }
    }
    w.len() - 1
}

// ----------------------------------------------------------------- export

fn export_mode(root: &Path, args: &[String]) {
    let sol_file = flag(args, "--pool").expect("--export needs --pool SOLFILE");
    let cands = load_candidates(root);
    let pool = load_pool(root, &cands, &sol_file);
    let by_id: HashMap<&str, &CandTeam> =
        cands.teams.iter().map(|t| (t.id.as_str(), t)).collect();
    let mut rows: Vec<(usize, f64)> =
        pool.weights.iter().copied().enumerate().filter(|(_, w)| *w > 1e-4).collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let teams: Vec<Value> = rows
        .iter()
        .map(|&(i, w)| {
            let t = by_id[pool.ids[i].as_str()];
            json!({"id": t.id, "origin": t.origin, "weight": w, "sets": t.sets})
        })
        .collect();
    let out = json!({
        "format": "meta-nash-pool-v1",
        "source_solution": sol_file,
        "engine_commit": commit(root),
        "teams": teams,
    });
    let path = root.join("data/meta-nash-v1/pool-artifact.json");
    atomic_write(&path, &serde_json::to_string_pretty(&out).unwrap());
    eprintln!("exported {} teams -> {}", teams.len(), path.display());
}

// ------------------------------------------------------------------- main

fn commit(root: &Path) -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| args[i + 1].clone())
}

fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = repo_root();
    let dex = load_dex();
    if has(&args, "--init") {
        let meta: Value = serde_json::from_str(
            &std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap(),
        )
        .unwrap();
        let teams: Vec<CandTeam> = meta["teams"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| CandTeam {
                id: t["id"].as_str().unwrap().to_string(),
                origin: "curated".into(),
                sets: serde_json::from_value(t["sets"].clone()).unwrap(),
            })
            .collect();
        eprintln!("init: {} curated teams", teams.len());
        save_candidates(&root, &Candidates { teams });
    } else if has(&args, "--matrix") {
        matrix_mode(&root, &dex, &args);
    } else if has(&args, "--solve") {
        solve_mode(&root, &args);
    } else if has(&args, "--br") {
        br_mode(&root, &dex, &args);
    } else if has(&args, "--harvest") {
        harvest_mode(&root, &args);
    } else if has(&args, "--duel") {
        duel_mode(&root, &dex, &args);
    } else if has(&args, "--export") {
        export_mode(&root, &args);
    } else {
        eprintln!("modes: --init | --matrix | --solve | --br | --harvest | --duel | --export");
        std::process::exit(2);
    }
}
