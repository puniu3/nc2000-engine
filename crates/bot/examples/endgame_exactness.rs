//! M17e step 3 — eval vs EXACT endgame equity.
//!
//! Samples endgame positions (≤2 alive per side) from random-legal
//! meta-pool self-play, solves each exactly (`bot::exact`, zero seed
//! noise, certified matrix brackets), and reports where `eval01` diverges
//! from the true value — sliced by the exact value's region, so the
//! loss-region tail (the M17c target) is measured directly against truth
//! instead of playout estimates.
//!
//! Position-source caveat: random-legal play reaches real states but not
//! realistic ones; treat slice signs as leads and re-check the top
//! divergences by hand before acting.
//!
//! Usage: endgame_exactness [games] [state_budget] [leaf_cap] [work_budget]
//!   games        — self-play games to drive (default 40)
//!   state_budget — max new states expanded per solve (default 30000)
//!   leaf_cap     — chance-leaf cap per step (default 20000)
//!   work_budget  — max chance leaves enumerated per solve (default 300000)
//!
//! CSV: tmp/endgame-exactness.csv (exact, eval, turn, alive0, alive1, hpfrac).

use std::collections::HashSet;
use std::io::Write as _;
use std::path::Path;
use std::time::Instant;

use conformance::load_dex;
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::exact::{ExactConfig, ExactSolver};
use nc2000_bot::preview::load_meta_pool;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::prng::Prng;
use nc2000_engine::state::Battle;

const MAX_TURNS: u16 = 200;

fn pick_choices(b: &mut Battle, dex: &Dex, mrng: &mut Prng) -> Option<[Option<SearchChoice>; 2]> {
    let needs = b.needs_choice();
    if !needs[0] && !needs[1] {
        return None;
    }
    let mut choices = [None, None];
    for side in 0..2 {
        if needs[side] {
            let ls = b.legal_choices(dex, side);
            if ls.is_empty() {
                return None;
            }
            choices[side] = Some(ls[mrng.sample_index(ls.len())]);
        }
    }
    Some(choices)
}

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side].party.iter().filter(|&&s| !b.sides[side].roster[s as usize].fainted).count()
}

fn hp_frac(b: &Battle) -> f64 {
    let (mut hp, mut max) = (0u64, 0u64);
    for side in &b.sides {
        for &slot in side.party.iter() {
            let p = &side.roster[slot as usize];
            hp += p.hp as u64;
            max += p.maxhp as u64;
        }
    }
    hp as f64 / max.max(1) as f64
}

struct Row {
    exact: f64,
    width: f64,
    horizon: u16,
    eval: f64,
    turn: u16,
    alive0: usize,
    alive1: usize,
    hpfrac: f64,
    desc: String,
}

/// Bracket width under which a position counts as solved for comparison.
const SOLVED_W: f64 = 0.05;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40);
    let budget: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let leaf_cap: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let work: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2_000_000);

    let dex = load_dex();
    let pool = load_meta_pool(Path::new("data/meta-pool-v0/meta-pool.json"));
    let weights = EvalWeights::default();
    println!(
        "pool: {} teams; {games} games, budget {budget}, leaf_cap {leaf_cap}, work {work}",
        pool.teams.len()
    );

    let mut mrng = Prng::new(0xfeed_beef_cafe_0001);
    let cfg = ExactConfig {
        state_budget: budget,
        leaf_cap,
        work_budget: work,
        ..ExactConfig::default()
    };
    let mut solver = ExactSolver::new(&dex, cfg);
    let mut attempted_keys: HashSet<u64> = HashSet::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut attempted = 0usize;
    let mut unsolved = 0usize;
    let mut solve_secs = 0.0f64;
    let t0 = Instant::now();

    for game in 0..games {
        let n = pool.teams.len();
        let (i, j) = loop {
            let (i, j) = (mrng.sample_index(n), mrng.sample_index(n));
            if i != j {
                break (i, j);
            }
        };
        let mut b = match Battle::from_fixture(&dex, "1,2,3,4", &pool.teams[i].sets, &pool.teams[j].sets)
        {
            Ok(b) => b,
            Err(_) => continue,
        };
        b.set_log_enabled(false);
        b.reseed(((mrng.next_u32() as u64) << 32) | mrng.next_u32() as u64);

        let mut attempts_left = 6usize;
        let mut retry_below = u64::MAX; // after an unsolved attempt, wait for real progress
        while !b.ended && b.turn < MAX_TURNS {
            let choices = match pick_choices(&mut b, &dex, &mut mrng) {
                Some(c) => c,
                None => break,
            };

            let (a0, a1) = (alive(&b, 0), alive(&b, 1));
            let total_hp: u64 = b
                .sides
                .iter()
                .flat_map(|s| s.party.iter().map(|&sl| s.roster[sl as usize].hp as u64))
                .sum();
            // Solve only where the end is plausibly near: the 39-way
            // damage-roll fan is real state splitting, so closures shrink
            // only where overkill merges it — low total HP. Interval
            // deepening bails out of stall positions cheaply (stall_gain),
            // so mild over-inclusion is affordable.
            // Measured 2026-07-21 (12-game diagnostic): at 1v1 with
            // healthy HP (169-283) even horizon 1 costs ~2M runs — the
            // two-sided 39-roll joint fan. Only lethal-range positions
            // (fans collapsed by overkill) can certify; filter there.
            let near_end = a0 <= 2 && a1 <= 2 && total_hp <= 100;
            if near_end
                && attempts_left > 0
                && total_hp < retry_below
                && attempted_keys.insert(b.state_key())
            {
                attempted += 1;
                attempts_left -= 1;
                let runs0 = solver.stats.chance_runs;
                let ts = Instant::now();
                let solved = solver.solve(&b);
                let dt = ts.elapsed().as_secs_f64();
                solve_secs += dt;
                println!(
                    "  attempt g{game} T{} hp{total_hp} {a0}v{a1}: {} ({} runs, {dt:.0}s)",
                    b.turn,
                    match &solved {
                        None => "ABORT (budget mid-first-horizon)".to_string(),
                        Some(c) =>
                            format!("[{:.3},{:.3}] w{:.3} h{}", c.lo, c.hi, c.width(), c.horizon),
                    },
                    solver.stats.chance_runs - runs0
                );
                match solved {
                    None => {
                        unsolved += 1;
                        // don't retry until ~a hit of HP has actually gone
                        retry_below = total_hp.saturating_sub(30);
                    }
                    Some(cert) => {
                        if cert.width() <= SOLVED_W {
                            retry_below = u64::MAX;
                        } else {
                            unsolved += 1;
                            retry_below = total_hp.saturating_sub(30);
                        }
                        let ev = eval01(&b, &dex, &weights);
                        let desc = {
                            let name = |side: usize| {
                                let s = &b.sides[side];
                                s.party
                                    .iter()
                                    .filter(|&&sl| !s.roster[sl as usize].fainted)
                                    .map(|&sl| {
                                        let p = &s.roster[sl as usize];
                                        format!(
                                            "{}({}/{})",
                                            dex.species.get(p.species).name,
                                            p.hp,
                                            p.maxhp
                                        )
                                    })
                                    .collect::<Vec<_>>()
                                    .join("+")
                            };
                            format!("g{game} T{} {} vs {}", b.turn, name(0), name(1))
                        };
                        rows.push(Row {
                            exact: cert.mid(),
                            width: cert.width(),
                            horizon: cert.horizon,
                            eval: ev,
                            turn: b.turn,
                            alive0: a0,
                            alive1: a1,
                            hpfrac: hp_frac(&b),
                            desc,
                        });
                    }
                }
            }

            b.apply_choices(&dex, choices).expect("legal choices");
        }
        println!(
            "game {game}: attempted {attempted} bracketed {} wide/aborted {unsolved} ({:.0}s solve)",
            rows.len(),
            solve_secs
        );
    }

    // ---- report (comparison uses certified-tight rows only)
    let solved: Vec<&Row> = rows.iter().filter(|r| r.width <= SOLVED_W).collect();
    let n = solved.len();
    println!(
        "\nattempted {attempted}: certified-tight {n} (width ≤ {SOLVED_W}), wide/aborted {unsolved}; \
         solver exact-memo {} chance-runs {} worst-gap {:.2e} max-matrix {:?}",
        solver.stats.states, solver.stats.chance_runs, solver.stats.worst_gap, solver.stats.max_matrix
    );
    println!("solve wall {solve_secs:.1}s / total {:.1}s", t0.elapsed().as_secs_f64());

    // Bracket-violation test over ALL bracketed attempts (wide included):
    // eval outside [lo,hi] is a PROVEN eval error of at least that margin,
    // regardless of bracket width.
    let mut viols: Vec<(f64, &Row)> = rows
        .iter()
        .map(|r| {
            let (lo, hi) = (r.exact - r.width / 2.0, r.exact + r.width / 2.0);
            ((r.eval - hi).max(lo - r.eval).max(0.0), r)
        })
        .filter(|(v, _)| *v > 0.02)
        .collect();
    viols.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("proven bracket violations (eval outside certified range by >0.02): {}", viols.len());
    for (v, r) in viols.iter().take(10) {
        println!(
            "  margin {v:.3}: eval {:.3} vs [{:.3},{:.3}]  {}",
            r.eval,
            r.exact - r.width / 2.0,
            r.exact + r.width / 2.0,
            r.desc
        );
    }

    if n == 0 {
        println!("no certified-tight positions — no correlation stats");
        // CSV still written below is skipped by the early return; fine for
        // a diagnostic run.
        return;
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let ex: Vec<f64> = solved.iter().map(|r| r.exact).collect();
    let ev: Vec<f64> = solved.iter().map(|r| r.eval).collect();
    let (me, mv) = (mean(&ex), mean(&ev));
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    let mut mse = 0.0;
    for r in &solved {
        cov += (r.exact - me) * (r.eval - mv);
        vx += (r.exact - me).powi(2);
        vy += (r.eval - mv).powi(2);
        mse += (r.eval - r.exact).powi(2);
    }
    let r_corr = cov / (vx.sqrt() * vy.sqrt()).max(1e-12);
    println!(
        "\nn {n}: r {r_corr:.3}, MSE {:.4}, mean bias (eval−exact) {:+.4}",
        mse / n as f64,
        mv - me
    );

    println!("\nby exact-value region:");
    println!("{:>12} {:>4} {:>10} {:>10} {:>10}", "region", "n", "bias", "MAE", "MSE");
    for (lo, hi) in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0001)] {
        let sel: Vec<&&Row> = solved.iter().filter(|r| r.exact >= lo && r.exact < hi).collect();
        if sel.is_empty() {
            continue;
        }
        let k = sel.len() as f64;
        let bias = sel.iter().map(|r| r.eval - r.exact).sum::<f64>() / k;
        let mae = sel.iter().map(|r| (r.eval - r.exact).abs()).sum::<f64>() / k;
        let msq = sel.iter().map(|r| (r.eval - r.exact).powi(2)).sum::<f64>() / k;
        println!("[{lo:.1},{hi:.1}) {:>4} {bias:>+10.4} {mae:>10.4} {msq:>10.4}", sel.len());
    }

    let mut by_div: Vec<&Row> = solved.to_vec();
    by_div.sort_by(|a, b| {
        (b.eval - b.exact).abs().partial_cmp(&(a.eval - a.exact).abs()).unwrap()
    });
    println!("\nworst divergences:");
    for r in by_div.iter().take(10) {
        println!(
            "  exact {:.3}±{:.3} eval {:.3}  {}",
            r.exact,
            r.width / 2.0,
            r.eval,
            r.desc
        );
    }

    std::fs::create_dir_all("tmp").ok();
    let mut f = std::fs::File::create("tmp/endgame-exactness.csv").expect("csv");
    writeln!(f, "exact,width,horizon,eval,turn,alive0,alive1,hpfrac,desc").unwrap();
    for r in &rows {
        writeln!(
            f,
            "{:.6},{:.6},{},{:.6},{},{},{},{:.4},\"{}\"",
            r.exact, r.width, r.horizon, r.eval, r.turn, r.alive0, r.alive1, r.hpfrac, r.desc
        )
        .unwrap();
    }
    println!("\ncsv: tmp/endgame-exactness.csv (all attempts incl. wide brackets)");
}
