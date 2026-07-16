//! M8 preview baker: per meta-pool matchup, estimate the 60×60 team-preview
//! payoff matrix and ship its RM+-solved mixed equilibrium.
//!
//! Per pair (row team i, column team j, i ≤ j; the (j,i) matrix is the
//! transpose complement so only one orientation is baked):
//!
//! 1. **Screen** — every joint preview cell (60×60; mirror pairs bake the
//!    upper triangle and reflect) estimated with cheap ε-greedy max-damage
//!    self-play games. Fast enough for full width, biased by the policy's
//!    no-switch play — which is why it only *ranks*.
//! 2. **Support** — top actions per side by best-response payoff against the
//!    screen equilibrium, unioned with picks from a real skuct search at the
//!    preview root (the advisor — insurance against systematic screen bias).
//! 3. **Refine** — support×support cells re-estimated with skuct self-play
//!    games (P1/P2 alternated per game; battle seeds fresh per game). This
//!    matrix is the matrix of record.
//! 4. **Solve** — full-width RM+ (`smmcts::solve_rm_plus`) on the refined
//!    matrix; ship mixed + argmax policies and their exact counter-picking
//!    guarantees (the M8 gate numbers).
//!
//! Resumable: one JSON per pair in --out; existing files are skipped unless
//! --force. Deterministic for a given --seed at any thread count.
//!
//!   cargo run --release -p nc2000-bot --example bake_preview -- \
//!       [--teams 0-7 | --pairs 0:1,0:2] [--screen-games 16] \
//!       [--refine-games 24] [--support 8] [--skuct-iters 200] \
//!       [--advisor-iters 2000] [--advisor-runs 2] [--max-turns 300] \
//!       [--seed 1] [--threads N] [--out data/preview-tables-v0] [--force]
//!   cargo run --release -p nc2000-bot --example bake_preview -- --summarize

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::preview::{
    action_index, canonical_triple, load_meta_pool, preview_actions, solve_pair, BakeCfg,
    MatrixEst, MetaPool, PairTable, RolloutAgent,
};
use nc2000_bot::smmcts::{solve_rm_plus, SelRule};
use nc2000_bot::{play_game, Agent, GameResult, RmAgent, RmConfig, SplitMix64};
use nc2000_engine::battle::{Outcome, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T
where
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    flag(args, name).map(|v| v.parse().expect(name)).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pool_path = flag(&args, "--pool")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("data/meta-pool-v0/meta-pool.json"));
    let out_dir = flag(&args, "--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_root().join("data/preview-tables-v0"));

    if args.iter().any(|a| a == "--summarize") {
        summarize(&out_dir);
        return;
    }

    let cfg = BakeCfg {
        screen_games: num(&args, "--screen-games", 16),
        refine_games: num(&args, "--refine-games", 24),
        support: num(&args, "--support", 8),
        skuct_iters: num(&args, "--skuct-iters", 200),
        advisor_iters: num(&args, "--advisor-iters", 2000),
        advisor_runs: num(&args, "--advisor-runs", 2),
        eps: num(&args, "--eps", 0.2),
        max_turns: num(&args, "--max-turns", 300),
        seed: num(&args, "--seed", 1),
    };
    let threads: usize = flag(&args, "--threads")
        .map(|v| v.parse().unwrap())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
    let force = args.iter().any(|a| a == "--force");

    let dex = load_dex();
    let pool = load_meta_pool(&pool_path);
    std::fs::create_dir_all(&out_dir).unwrap();

    let pairs = select_pairs(&args, pool.teams.len());
    eprintln!(
        "baking {} pairs over {} teams, {} threads (screen {} g/cell, refine {} g/cell @ skuct:{}, support {})",
        pairs.len(),
        pool.teams.len(),
        threads,
        cfg.screen_games,
        cfg.refine_games,
        cfg.skuct_iters,
        cfg.support
    );

    for &(i, j) in &pairs {
        let path = out_dir.join(format!("pair-{i:02}-{j:02}.json"));
        if path.exists() && !force {
            eprintln!("pair {i:02}-{j:02}: exists, skipping");
            continue;
        }
        let t0 = Instant::now();
        let table = bake_pair(&dex, &pool, i, j, &cfg, threads);
        std::fs::write(&path, serde_json::to_string(&table).unwrap()).unwrap();
        let s = &table.sol;
        println!(
            "pair {i:02}-{j:02} {} vs {}: value {:.3}  mix |a|={} |b|={}  gate margins a {:+.3} b {:+.3}  ({:.0}s)",
            table.team_a,
            table.team_b,
            s.value,
            s.p_a.iter().filter(|&&p| p > 0.0).count(),
            s.p_b.iter().filter(|&&p| p > 0.0).count(),
            s.guarantee_mixed_a - s.guarantee_argmax_a,
            s.guarantee_mixed_b - s.guarantee_argmax_b,
            t0.elapsed().as_secs_f64(),
        );
    }
}

/// Pair list: --pairs i:j,... or all i ≤ j within --teams lo-hi (default all).
fn select_pairs(args: &[String], n_teams: usize) -> Vec<(usize, usize)> {
    if let Some(spec) = flag(args, "--pairs") {
        return spec
            .split(',')
            .map(|p| {
                let (i, j) = p.split_once(':').expect("--pairs wants i:j,i:j");
                let (i, j): (usize, usize) = (i.parse().unwrap(), j.parse().unwrap());
                assert!(i < n_teams && j < n_teams, "pair {i}:{j} out of range");
                (i.min(j), i.max(j))
            })
            .collect();
    }
    let (lo, hi) = match flag(args, "--teams") {
        Some(spec) => {
            let (lo, hi) = spec.split_once('-').expect("--teams wants lo-hi");
            (lo.parse().unwrap(), hi.parse::<usize>().unwrap().min(n_teams - 1))
        }
        None => (0, n_teams - 1),
    };
    let mut out = Vec::new();
    for i in lo..=hi {
        for j in i..=hi {
            out.push((i, j));
        }
    }
    out
}

// ------------------------------------------------------------- one pair

fn bake_pair(
    dex: &Dex,
    pool: &MetaPool,
    i: usize,
    j: usize,
    cfg: &BakeCfg,
    threads: usize,
) -> PairTable {
    let actions = preview_actions();
    let row = &pool.teams[i].sets;
    let col = &pool.teams[j].sets;
    let mirror = i == j;
    let pair_seed = cfg.seed ^ ((i as u64) << 32) ^ ((j as u64) << 16);
    let t0 = Instant::now();

    // ---- 1. screen: full-width cheap games
    let mut jobs = Vec::new();
    for a in 0..60usize {
        for b in 0..60usize {
            if mirror && b <= a {
                continue; // reflected below; diagonal is 0.5 by symmetry
            }
            push_cell_jobs(&mut jobs, a * 60 + b, actions[a], actions[b], cfg.screen_games, pair_seed ^ 0x5C4E);
        }
    }
    let eps = cfg.eps;
    let results = run_games(dex, row, col, &jobs, threads, cfg.max_turns, &|seed| {
        Box::new(RolloutAgent::new(eps, seed))
    });
    let mut screen = MatrixEst::new(60, 60);
    for (cell, score) in results {
        screen.record(cell, score);
    }
    if mirror {
        reflect(&mut screen, cfg.screen_games);
    }
    let screen_secs = t0.elapsed().as_secs_f64();

    // ---- 2. support: screen-equilibrium BR ranking ∪ skuct advisor picks
    let (e0, e1) = solve_rm_plus(&screen.v, [60, 60], 20_000);
    let u_a: Vec<f64> =
        (0..60).map(|a| (0..60).map(|b| screen.at(a, b) * e1[b]).sum()).collect();
    let u_b: Vec<f64> =
        (0..60).map(|b| (0..60).map(|a| (1.0 - screen.at(a, b)) * e0[a]).sum()).collect();
    let adv = advisor_picks(dex, row, col, &actions, cfg, pair_seed, mirror);
    let s0 = build_support(&u_a, &adv[0], cfg.support);
    let s1 = if mirror { s0.clone() } else { build_support(&u_b, &adv[1], cfg.support) };

    // ---- 3. refine: support×support skuct self-play
    let (k0, k1) = (s0.len(), s1.len());
    let mut jobs = Vec::new();
    for (ri, &a) in s0.iter().enumerate() {
        for (ci, &b) in s1.iter().enumerate() {
            if mirror && b <= a {
                continue; // reflected below; diagonal is 0.5 by symmetry
            }
            push_cell_jobs(&mut jobs, ri * k1 + ci, actions[a], actions[b], cfg.refine_games, pair_seed ^ 0x4EF1);
        }
    }
    let iters = cfg.skuct_iters;
    let results = run_games(dex, row, col, &jobs, threads, cfg.max_turns, &|seed| {
        Box::new(RmAgent::new(
            RmConfig { iterations: iters, rule: SelRule::Ucb, ..Default::default() },
            seed,
        ))
    });
    let mut refine = MatrixEst::new(k0, k1);
    for (cell, score) in results {
        refine.record(cell, score);
    }
    if mirror {
        // reflect via the support (support is shared, so index math is direct)
        for ri in 0..k0 {
            for ci in 0..k1 {
                if s1[ci] < s0[ri] {
                    refine.v[ri * k1 + ci] = 1.0 - refine.at(ci, ri);
                    refine.n[ri * k1 + ci] = refine.n[ci * k1 + ri];
                } else if s1[ci] == s0[ri] {
                    refine.v[ri * k1 + ci] = 0.5;
                    refine.n[ri * k1 + ci] = cfg.refine_games;
                }
            }
        }
    }

    // ---- 4. solve + gate numbers
    let sol = solve_pair(&refine, &[s0.clone(), s1.clone()], 0.05, 50_000);
    screen.compact();
    refine.compact();
    eprintln!(
        "  pair {i:02}-{j:02}: screen {:.0}s, total {:.0}s, {} refine games",
        screen_secs,
        t0.elapsed().as_secs_f64(),
        jobs.len(),
    );
    PairTable {
        team_a: pool.teams[i].id.clone(),
        team_b: pool.teams[j].id.clone(),
        actions,
        screen,
        support: [s0, s1],
        refine,
        sol,
        cfg: cfg.clone(),
        secs: t0.elapsed().as_secs_f64(),
    }
}

/// Reflect a mirror pair's upper triangle: M[b][a] = 1 − M[a][b], diagonal 0.5.
fn reflect(m: &mut MatrixEst, diag_n: u32) {
    let d = m.rows;
    for a in 0..d {
        m.v[a * d + a] = 0.5;
        m.n[a * d + a] = diag_n;
        for b in a + 1..d {
            m.v[b * d + a] = 1.0 - m.v[a * d + b];
            m.n[b * d + a] = m.n[a * d + b];
        }
    }
}

/// skuct picks at the real preview root, canonicalized — per side, one per
/// advisor run. Mirror pairs share side-0 picks.
fn advisor_picks(
    dex: &Dex,
    row: &[PokemonSet],
    col: &[PokemonSet],
    actions: &[[u8; 3]],
    cfg: &BakeCfg,
    pair_seed: u64,
    mirror: bool,
) -> [Vec<usize>; 2] {
    let mut out = [Vec::new(), Vec::new()];
    if cfg.advisor_runs == 0 {
        return out;
    }
    let mut battle = Battle::from_fixture(dex, "1,2,3,4", row, col).unwrap();
    battle.set_log_enabled(false);
    for side in 0..2 {
        if mirror && side == 1 {
            out[1] = out[0].clone();
            break;
        }
        let choices = battle.legal_choices(dex, side);
        for run in 0..cfg.advisor_runs {
            let mut agent = RmAgent::new(
                RmConfig {
                    iterations: cfg.advisor_iters,
                    rule: SelRule::Ucb,
                    ..Default::default()
                },
                pair_seed ^ 0xAD_115E ^ ((side as u64) << 8) ^ run as u64,
            );
            if let SearchChoice::Team(t) = agent.choose(&battle, dex, side, &choices) {
                let idx = action_index(actions, canonical_triple(t)).unwrap();
                if !out[side].contains(&idx) {
                    out[side].push(idx);
                }
            }
        }
    }
    out
}

/// Advisor picks first, then the best-response ranking fills up to `k`.
fn build_support(u: &[f64], advisors: &[usize], k: usize) -> Vec<usize> {
    let mut sup: Vec<usize> = advisors.to_vec();
    let mut rank: Vec<usize> = (0..u.len()).collect();
    rank.sort_by(|&a, &b| u[b].total_cmp(&u[a]).then(a.cmp(&b)));
    for a in rank {
        if sup.len() >= k {
            break;
        }
        if !sup.contains(&a) {
            sup.push(a);
        }
    }
    sup.sort_unstable();
    sup
}

// --------------------------------------------------------- game execution

struct GameJob {
    cell: usize,
    ta: [u8; 3],
    tb: [u8; 3],
    /// Odd games swap which team is P1 (kills any engine side bias).
    swap: bool,
    battle_seed: String,
    agent_seed: u64,
}

fn push_cell_jobs(
    jobs: &mut Vec<GameJob>,
    cell: usize,
    ta: [u8; 3],
    tb: [u8; 3],
    games: u32,
    seed: u64,
) {
    let mut rng = SplitMix64::new(seed ^ (cell as u64).wrapping_mul(0x9FB2_1C65_1E98_DF25));
    for g in 0..games {
        jobs.push(GameJob {
            cell,
            ta,
            tb,
            swap: g % 2 == 1,
            battle_seed: rng.battle_seed(),
            agent_seed: rng.next(),
        });
    }
}

/// Play every job (thread pool, deterministic at any thread count) and
/// return (cell, row-player score) per game, job-ordered.
fn run_games(
    dex: &Dex,
    row: &[PokemonSet],
    col: &[PokemonSet],
    jobs: &[GameJob],
    threads: usize,
    max_turns: u16,
    build: &(dyn Fn(u64) -> Box<dyn Agent> + Sync),
) -> Vec<(usize, f64)> {
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let t0 = Instant::now();
    let mut results: Vec<(usize, (usize, f64))> = Vec::with_capacity(jobs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..threads {
            let (cursor, done) = (&cursor, &done);
            handles.push(scope.spawn(move || {
                let mut out = Vec::new();
                loop {
                    let n = cursor.fetch_add(1, Ordering::Relaxed);
                    if n >= jobs.len() {
                        break;
                    }
                    let job = &jobs[n];
                    let mut p1 = build(job.agent_seed ^ 0xA24B_AED4_963E_E407);
                    let mut p2 = build(job.agent_seed ^ 0x9FB2_1C65_1E98_DF25);
                    let (t1, t2, team1, team2) = if job.swap {
                        (job.tb, job.ta, col, row)
                    } else {
                        (job.ta, job.tb, row, col)
                    };
                    let mut b =
                        Battle::from_fixture(dex, &job.battle_seed, team1, team2).unwrap();
                    b.set_log_enabled(false);
                    b.apply_choices(
                        dex,
                        [Some(SearchChoice::Team(t1)), Some(SearchChoice::Team(t2))],
                    )
                    .unwrap();
                    let res =
                        play_game(dex, &mut b, &mut [&mut *p1, &mut *p2], max_turns).unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.swap { 1.0 - p1_score } else { p1_score };
                    out.push((n, (job.cell, score)));
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 512 == 0 {
                        let dt = t0.elapsed().as_secs_f64();
                        eprintln!(
                            "    {d}/{} games ({:.0}s, {:.1} games/s)",
                            jobs.len(),
                            dt,
                            d as f64 / dt
                        );
                    }
                }
                out
            }));
        }
        let mut all: Vec<(usize, (usize, f64))> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all;
    });
    results.into_iter().map(|(_, r)| r).collect()
}

// ------------------------------------------------------------- summarize

/// Gate A report over every baked pair: the mixed equilibrium must lose no
/// more to a counter-picking best response than the argmax policy does
/// (guarantee_mixed ≥ guarantee_argmax), exact on the refined matrix.
fn summarize(dir: &std::path::Path) {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("json"))
        .collect();
    files.sort();
    let mut margins = Vec::new();
    let mut mixed_pairs = 0usize;
    let mut violations = 0usize;
    println!(
        "{:<24} {:<24} {:>6} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8}",
        "team_a", "team_b", "value", "|mixA|", "|mixB|", "gA_mix", "gA_arg", "gB_mix", "gB_arg"
    );
    for p in &files {
        let tab: PairTable =
            serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let s = &tab.sol;
        let (ma, mb) = (
            s.guarantee_mixed_a - s.guarantee_argmax_a,
            s.guarantee_mixed_b - s.guarantee_argmax_b,
        );
        margins.push(ma);
        margins.push(mb);
        let mixes = |p: &[f64]| p.iter().filter(|&&x| x > 0.0).count();
        if mixes(&s.p_a) > 1 || mixes(&s.p_b) > 1 {
            mixed_pairs += 1;
        }
        if ma < -1e-6 || mb < -1e-6 {
            violations += 1;
        }
        println!(
            "{:<24} {:<24} {:>6.3} {:>5} {:>5} {:>8.3} {:>8.3} {:>8.3} {:>8.3}",
            trunc(&tab.team_a),
            trunc(&tab.team_b),
            s.value,
            mixes(&s.p_a),
            mixes(&s.p_b),
            s.guarantee_mixed_a,
            s.guarantee_argmax_a,
            s.guarantee_mixed_b,
            s.guarantee_argmax_b,
        );
    }
    let n = margins.len().max(1) as f64;
    println!(
        "\n{} pairs; gate margins (mixed − argmax counter-pick guarantee): mean {:+.4}, max {:+.4}; {} sides at 0 (pure optimum); mixing in {} pairs; {} violations",
        files.len(),
        margins.iter().sum::<f64>() / n,
        margins.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        margins.iter().filter(|&&m| m.abs() <= 1e-6).count(),
        mixed_pairs,
        violations
    );
}

fn trunc(s: &str) -> String {
    s.chars().take(24).collect()
}
