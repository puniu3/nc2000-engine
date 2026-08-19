//! One-ply-deeper certified bounds (round 2, battle 4069 turn 26).
//!
//! `verdict_bounds` certifies the value of a recorded decision ROOT — a
//! simultaneous-move node. This example certifies the value of the position
//! that a CHOSEN joint action leads to, which is strictly smaller:
//!
//!   1. reconstruct the recorded p2 decision at `--turn` (same corpus path
//!      `verdict_bounds` uses),
//!   2. take one decision step under `[--p1 CHOICE, --p2 CHOICE]` with the
//!      engine's exact chance enumerator (`enumerate_step`), which returns
//!      every successor with its exact probability,
//!   3. run `bounds::BoundSolver` on each successor and combine:
//!         lo = sum_i p_i * lo_i,  hi = sum_i p_i * hi_i
//!      which is sound because the value is linear in the chance mixture.
//!
//! With `--p1-all` this is done for every legal p1 reply, which brackets the
//! SECURITY LEVEL of the p2 action: side 0 (the human) maximizes, so p2's
//! worst case is `max_b V0(a,b)`, bounded by `[max_b lo_b, max_b hi_b]`.
//!
//! Polarity: `BoundSolver` bounds SIDE 0 = p1. p2's interval is the
//! complement; both are printed. Nothing here caps or fabricates a bound —
//! a successor that does not converge is reported with its own Stop reason
//! and its full width flows into the combination.
//!
//! Usage:
//!   verdict_bounds_ply --log tmp/corpus-4069/battle-4069.raw.log --turn 26 \
//!     [--list] [--p2 "move earthquake"] [--p1 "move meanlook"|--p1-all] \
//!     [--work N] [--nodes N] [--eps E] [--calls N] [--fresh-solver]

use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::bounds::{BoundConfig, BoundSolver, Bounds, Stop};
use nc2000_bot::corpus::{
    complete_active_moves_from_future, load_battle, load_sources, reconstruct, HumanAction,
};
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_engine::battle::enumerate::enumerate_step;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::state::{Battle, Status};

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}
fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}
fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side].party.iter().filter(|&&s| !b.sides[side].roster[s as usize].fainted).count()
}

fn label(c: SearchChoice, dex: &nc2000_engine::dex::Dex) -> String {
    match c {
        SearchChoice::Move(m) => format!("move {}", dex.moves.key(m)),
        SearchChoice::Switch(p) => format!("switch {p}"),
        SearchChoice::Pass => "pass".to_string(),
        SearchChoice::Team(t) => format!("team {t:?}"),
    }
}

fn pick(cands: &[SearchChoice], want: &str, dex: &nc2000_engine::dex::Dex) -> SearchChoice {
    let want_l = want.to_lowercase();
    let hits: Vec<&SearchChoice> =
        cands.iter().filter(|c| label(**c, dex).to_lowercase().contains(&want_l)).collect();
    match hits.len() {
        1 => *hits[0],
        0 => panic!(
            "no legal choice matches {want:?}; legal = {:?}",
            cands.iter().map(|c| label(*c, dex)).collect::<Vec<_>>()
        ),
        _ => panic!(
            "{want:?} is ambiguous: {:?}",
            hits.iter().map(|c| label(**c, dex)).collect::<Vec<_>>()
        ),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let work: usize = arg(&args, "--work").unwrap_or("200000").parse().unwrap();
    let node_budget: usize = arg(&args, "--nodes").unwrap_or("120000").parse().unwrap();
    let eps: f64 = arg(&args, "--eps").unwrap_or("0.02").parse().unwrap();
    let calls: usize = arg(&args, "--calls").unwrap_or("1").parse().unwrap();
    let cell_cap: usize = arg(&args, "--cell-cap").unwrap_or("4096").parse().unwrap();
    let trial_depth: usize = arg(&args, "--trial-depth").unwrap_or("24").parse().unwrap();
    let descend_floor: f64 = arg(&args, "--descend-floor").unwrap_or("0.1").parse().unwrap();
    let recon_seed: u64 = arg(&args, "--recon-seed").unwrap_or("1").parse().unwrap();
    let step_cap: usize = arg(&args, "--step-cap").unwrap_or("2000000").parse().unwrap();
    let oracle_moves = !flag(&args, "--no-oracle-moves");
    let fresh_solver = flag(&args, "--fresh-solver");
    let tau: Option<(f64, f64)> = arg(&args, "--tau").map(|s| {
        let (lo, hi) = s.split_once(',').expect("--tau lo,hi");
        (lo.parse().unwrap(), hi.parse().unwrap())
    });

    let root_dir = repo_root();
    let log = root_dir.join(arg(&args, "--log").unwrap_or("tmp/corpus-4069/battle-4069.raw.log"));
    let dex = load_dex();
    let src = load_sources(&dex, &root_dir);
    let pool_path = root_dir.join("data/meta-pool-v0/meta-pool.json");
    let battle_log = load_battle(&log);
    let turn: u16 = arg(&args, "--turn").unwrap_or("26").parse().unwrap();

    let d = battle_log
        .decisions
        .iter()
        .find(|d| d.side == 1 && d.turn == turn)
        .unwrap_or_else(|| panic!("no p2 decision at turn {turn}"));
    let mut b = reconstruct(
        &dex,
        &src,
        &pool_path,
        &battle_log.lines,
        &battle_log.evidence,
        d,
        recon_seed,
    )
    .expect("reconstruction");
    if oracle_moves {
        let got = complete_active_moves_from_future(&dex, &mut b, &battle_log.lines);
        println!("oracle-completed active moves: {got:?}");
    }
    let human = match &d.action {
        HumanAction::Move(k) => format!("move {k}"),
        HumanAction::Switch(sp) => format!("switch {sp}"),
    };
    println!("{} | p2 turn {turn} | played {human}", log.display());
    for s in 0..2 {
        let mons: Vec<String> = b.sides[s]
            .party
            .iter()
            .map(|&slot| {
                let p = &b.sides[s].roster[slot as usize];
                format!(
                    "{}({}/{}{})",
                    dex.species.key(p.species),
                    p.hp,
                    p.maxhp,
                    if p.status == Status::None { String::new() } else { format!(" {:?}", p.status) }
                )
            })
            .collect();
        println!("  side {s}: {} alive {}", mons.join(" "), alive(&b, s));
    }
    println!("  state_key128 {:032x}", b.state_key128());
    let ev = eval01(&b, &dex, &EvalWeights::default());
    println!("  eval01(side0) {ev:.4}   eval01(p2) {:.4}", 1.0 - ev);

    let c0 = b.legal_choices(&dex, 0);
    let c1 = b.legal_choices(&dex, 1);
    println!(
        "  p1 legal: {:?}",
        c0.iter().map(|c| label(*c, &dex)).collect::<Vec<_>>()
    );
    println!(
        "  p2 legal: {:?}",
        c1.iter().map(|c| label(*c, &dex)).collect::<Vec<_>>()
    );
    if flag(&args, "--list") {
        return;
    }

    let p2c = pick(&c1, arg(&args, "--p2").expect("--p2 CHOICE (or --list)"), &dex);
    let p1_set: Vec<SearchChoice> = if flag(&args, "--p1-all") {
        c0.clone()
    } else {
        vec![pick(&c0, arg(&args, "--p1").expect("--p1 CHOICE or --p1-all"), &dex)]
    };

    let cfg = BoundConfig {
        work_budget: work,
        node_budget,
        cell_cap,
        eps,
        trial_depth,
        descend_floor,
        ..BoundConfig::default()
    };
    println!(
        "\np2 action = {}   p1 replies = {:?}\nwork/call {work} nodes {node_budget} eps {eps} \
         calls {calls} step-cap {step_cap} fresh-solver {fresh_solver}\n",
        label(p2c, &dex),
        p1_set.iter().map(|c| label(*c, &dex)).collect::<Vec<_>>()
    );

    let mut solver = BoundSolver::new(&dex, cfg.clone());
    let mut per_reply: Vec<(String, Bounds, usize, usize)> = Vec::new();
    let mut total_runs = 0usize;
    let t_all = Instant::now();

    for p1c in &p1_set {
        let t0 = Instant::now();
        let step = enumerate_step(&dex, &b, [Some(*p1c), Some(p2c)], step_cap);
        let Some(step) = step else {
            println!(
                "p1 {:<22} STEP ENUMERATION OVERFLOWED --step-cap {step_cap} — no bound",
                label(*p1c, &dex)
            );
            continue;
        };
        let n_leaves = step.leaves.len();
        let mut enum_runs = step.runs;
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        let mut worst_stop = Stop::WidthMet;
        let mut solver_runs = 0usize;
        if fresh_solver {
            solver = BoundSolver::new(&dex, cfg.clone());
        }
        for leaf in &step.leaves {
            // `--calls N` resumes the SAME persistent graph N times; the
            // interval only tightens, so stopping early on a proof is safe.
            let mut rep = solver.solve(&leaf.battle, tau);
            solver_runs += rep.runs;
            for _ in 1..calls {
                if matches!(
                    rep.stop,
                    Stop::WidthMet | Stop::ProvenAbove | Stop::ProvenBelow | Stop::Contained
                ) {
                    break;
                }
                let again = solver.solve(&leaf.battle, tau);
                solver_runs += again.runs;
                let stalled = again.runs == 0;
                rep = again;
                if stalled {
                    break;
                }
            }
            lo += leaf.prob * rep.bounds.lo;
            hi += leaf.prob * rep.bounds.hi;
            if !matches!(rep.stop, Stop::WidthMet) {
                worst_stop = rep.stop;
            }
        }
        enum_runs += solver_runs;
        total_runs += enum_runs;
        let bd = Bounds { lo, hi };
        println!(
            "p1 {:<22} leaves {:<5} side0 [{:.4},{:.4}] w{:.4}  p2 [{:.4},{:.4}]  worst-stop \
             {:?}  runs {} (enum {})  peak-nodes {}  {:.1}s",
            label(*p1c, &dex),
            n_leaves,
            bd.lo,
            bd.hi,
            bd.width(),
            1.0 - bd.hi,
            1.0 - bd.lo,
            worst_stop,
            enum_runs,
            step.runs,
            solver.stats.peak_nodes,
            t0.elapsed().as_secs_f64()
        );
        per_reply.push((label(*p1c, &dex), bd, n_leaves, enum_runs));
    }

    println!("\ntotal engine runs {total_runs}  peak nodes {}  {:.1}s", solver.stats.peak_nodes, t_all.elapsed().as_secs_f64());
    if per_reply.is_empty() {
        println!("NO BOUND: every reply overflowed enumeration.");
        return;
    }
    // Side 0 maximizes, so p2's security level under this action is max_b V0.
    let sec_lo = per_reply.iter().map(|r| r.1.lo).fold(f64::MIN, f64::max);
    let sec_hi = per_reply.iter().map(|r| r.1.hi).fold(f64::MIN, f64::max);
    if p1_set.len() > 1 {
        println!(
            "SECURITY LEVEL of {} (p1 best-responds): side0 in [{sec_lo:.4},{sec_hi:.4}], \
             p2 in [{:.4},{:.4}]  (sound only if EVERY reply above produced a bound)",
            label(p2c, &dex),
            1.0 - sec_hi,
            1.0 - sec_lo
        );
    }
    let (n, w) = (per_reply.len(), per_reply.iter().map(|r| r.1.width()).fold(0.0, f64::max));
    println!("{n} reply root(s); worst width {w:.4}");
    if w >= 0.999 {
        println!(
            "NOT CONVERGED: at least one reply root is still the trivial [0,1]. The bound is \
             SOUND but carries no information; do NOT read a midpoint as a value."
        );
    }
}
