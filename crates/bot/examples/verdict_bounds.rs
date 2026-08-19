//! A certified-bounds attempt on the battle-4070 endgame (V3/V4).
//!
//! Reconstructs one recorded p2 decision and runs `bounds::BoundSolver` on it
//! — the same lazy interval solver `anchor_gate` and `endgame_exactness_corpus`
//! use, with their config, not an invented one. Work limits stop expansion but
//! never discard it, so `--calls N` resumes the same graph N times and the
//! interval only tightens.
//!
//! This position is the two-sided-heal fiber (both sides holding Rest), which
//! `memory/heal-stall-structure.md` records as NEVER MEASURED. It may not
//! converge. When it does not, this prints NOT CONVERGED with the width it
//! actually reached — it never silently caps, and it never reports a midpoint
//! as if it were a value.
//!
//! Note the polarity: `BoundSolver` bounds SIDE 0's win probability. Side 0 is
//! the human here; the bot is p2 = side 1, so the bot's interval is the
//! complement and both are printed.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example verdict_bounds -- \
//!     [--turn 50] [--work 200000] [--nodes 120000] [--eps 0.02] \
//!     [--calls 5] [--tau 0.4,0.6] [--log PATH] [--no-oracle-moves]

use std::time::Instant;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::bounds::{BoundConfig, BoundSolver, Stop};
use nc2000_bot::corpus::{
    complete_active_moves_from_future, load_battle, load_sources, reconstruct, HumanAction,
};
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::stall::{classify_one_sided_heal, classify_two_sided_heal};
use nc2000_engine::state::{Battle, Status};

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side]
        .party
        .iter()
        .filter(|&&slot| !b.sides[side].roster[slot as usize].fainted)
        .count()
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
    let oracle_moves = !flag(&args, "--no-oracle-moves");
    let tau: Option<(f64, f64)> = arg(&args, "--tau").map(|s| {
        let (lo, hi) = s.split_once(',').expect("--tau lo,hi");
        (lo.parse().expect("--tau lo"), hi.parse().expect("--tau hi"))
    });

    let root = repo_root();
    let log = root.join(arg(&args, "--log").unwrap_or("tmp/corpus-4070/battle-4070.raw.log"));
    let dex = load_dex();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let battle_log = load_battle(&log);

    let turn: u16 = match arg(&args, "--turn") {
        Some(t) => t.parse().unwrap(),
        None => 50,
    };
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
        if let Some(id) = b.active_id(s) {
            let pp: Vec<String> = b
                .poke(id)
                .move_slots
                .iter()
                .map(|m| format!("{}({})", dex.moves.key(m.id), m.pp))
                .collect();
            println!("    active {}", pp.join(" "));
        }
    }
    println!("  state_key128 {:032x}", b.state_key128());
    let ev = eval01(&b, &dex, &EvalWeights::default());
    println!("  eval01(side0) {ev:.4}   eval01(p2) {:.4}", 1.0 - ev);

    // The stall classifiers gate the resource schedulers. Both REQUIRE a
    // last-mon-vs-last-mon root, so a 1-vs-2 endgame classifies as neither;
    // print the exact error rather than a boolean.
    println!("  one-sided-heal: {:?}", classify_one_sided_heal(&b, &dex));
    println!("  two-sided-heal: {:?}", classify_two_sided_heal(&b, &dex));
    match tau {
        Some((lo, hi)) => println!("\nthreshold mode, tau = [{lo:.4}, {hi:.4}] on SIDE 0"),
        None => println!("\nwidth mode, eps = {eps:.4}"),
    }
    println!(
        "work/call {work}  nodes {node_budget}  cell-cap {cell_cap}  trial-depth {trial_depth}  \
         calls {calls}\n"
    );

    let mut solver = BoundSolver::new(
        &dex,
        BoundConfig {
            work_budget: work,
            node_budget,
            cell_cap,
            eps,
            trial_depth,
            descend_floor,
            ..BoundConfig::default()
        },
    );

    let mut cumulative = 0usize;
    let mut last_stop = Stop::WorkExhausted;
    let mut last_bounds = None;
    for call in 1..=calls {
        let t0 = Instant::now();
        let rep = solver.solve(&b, tau);
        cumulative += rep.runs;
        last_stop = rep.stop;
        last_bounds = Some(rep.bounds);
        println!(
            "call {call:>3}: side0 [{lo:.4},{hi:.4}] w{w:.4}  p2 [{plo:.4},{phi:.4}]  {stop:?}  \
             runs {runs} (cum {cumulative})  nodes {nodes} closed {closed}  {dt:.1}s",
            lo = rep.bounds.lo,
            hi = rep.bounds.hi,
            w = rep.bounds.width(),
            plo = 1.0 - rep.bounds.hi,
            phi = 1.0 - rep.bounds.lo,
            stop = rep.stop,
            runs = rep.runs,
            nodes = solver.node_count(),
            closed = solver.closed_count(),
            dt = t0.elapsed().as_secs_f64(),
        );
        let s = &solver.stats;
        println!(
            "          expansions {} trials {} lp {} peak-nodes {} closed-folds {} worst-gap \
             {:.4}",
            s.expansions, s.trials, s.lp_solves, s.peak_nodes, s.closed_folds, s.worst_gap
        );
        println!(
            "          stall: one-sided roots {} inval {} handoffs {} | two-sided roots {} inval \
             {} | min-heal-pp {:?} min-res-pp {:?}",
            s.monotone_roots,
            s.monotone_invalidations,
            s.one_sided_handoffs,
            s.two_sided_roots,
            s.two_sided_invalidations,
            s.min_healing_pp,
            s.min_resource_pp
        );
        println!(
            "          pruning: rows {} cols {} checks {} avoided-cells {} | picks br-lo {} \
             br-hi {} legacy {} fair {}",
            s.dominated_rows,
            s.dominated_cols,
            s.dominance_checks,
            s.avoided_cells,
            s.lower_br_picks,
            s.upper_br_picks,
            s.legacy_support_picks,
            s.fair_cell_picks
        );
        if matches!(
            rep.stop,
            Stop::WidthMet | Stop::ProvenAbove | Stop::ProvenBelow | Stop::Contained
        ) {
            break;
        }
    }

    let bounds = last_bounds.expect("at least one call");
    println!();
    match last_stop {
        Stop::WidthMet => println!(
            "CONVERGED (width): side0 in [{:.4},{:.4}], p2 in [{:.4},{:.4}] after {cumulative} runs",
            bounds.lo,
            bounds.hi,
            1.0 - bounds.hi,
            1.0 - bounds.lo
        ),
        Stop::ProvenAbove => println!(
            "PROVEN ABOVE tau: side0 lo {:.4} > tau_hi; p2 hi {:.4}. {cumulative} runs",
            bounds.lo,
            1.0 - bounds.lo
        ),
        Stop::ProvenBelow => println!(
            "PROVEN BELOW tau: side0 hi {:.4} < tau_lo; p2 lo {:.4}. {cumulative} runs",
            bounds.hi,
            1.0 - bounds.hi
        ),
        Stop::Contained => println!(
            "CONTAINED in tau: side0 [{:.4},{:.4}] fits inside the band. {cumulative} runs",
            bounds.lo, bounds.hi
        ),
        Stop::WorkExhausted | Stop::NodeBudget => println!(
            "NOT CONVERGED ({last_stop:?}): after {calls} call(s) and {cumulative} engine runs \
             the root interval is still side0 [{:.4},{:.4}] (width {:.4}), p2 [{:.4},{:.4}]. \
             The bound is SOUND but uninformative; do NOT read the midpoint as a value. \
             Re-run with more --calls/--work, or accept that this fiber does not close.",
            bounds.lo,
            bounds.hi,
            bounds.width(),
            1.0 - bounds.hi,
            1.0 - bounds.lo
        ),
    }
}
