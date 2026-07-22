//! M17e step 3 — eval vs certified endgame brackets on HUMAN-GAME positions.
//!
//! Same measurement as `endgame_exactness`, but positions come from the
//! 570-battle spectator corpus (real endgames reached by real play) instead
//! of random-legal self-play — owner decision 2026-07-21: similar positions
//! recur in live games, so anchor the eval where it will actually be asked.
//!
//! Position reconstruction is `human_agreement`'s fabrication path (tracker
//! over the protocol prefix → set fabrication from rentals/pool/learnsets →
//! synthesized full battle via ProtocolAgent::on_request), WITHOUT running
//! the search. The exact value is therefore exact FOR THE IMPUTED
//! DETERMINIZATION — the same full-info state family the eval scores, so
//! the comparison is apples-to-apples; it is not the true hidden-set game.
//! (Fabrication helpers are copied from examples/human_agreement.rs —
//! examples cannot import each other; dedup into bot::corpus when a third
//! user appears.)
//!
//! Reports certified-tight comparisons plus PROVEN bracket violations
//! (eval outside a certified interval, any width — zero playouts involved).
//!
//! Usage: endgame_exactness_corpus [--corpus DIR] [--battles LO-HI]
//!        [--side N] [--turn N] [--hp-cap N] [--work N] [--nodes N]
//!        [--per-battle N] [--no-dead-damage-quotient]
//!        [--no-fold-terminal-nodes] [--no-fold-closed-nodes]
//!        [--no-monotone-stall] [--no-two-sided-resource]
//!        [--no-action-pruning] [--no-support-br] [--diagnose-stall] [--out CSV]

use std::collections::HashSet;
use std::io::Write as _;
use std::time::Instant;

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Stop};
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::stall::{classify_one_sided_heal, classify_two_sided_heal};
use nc2000_engine::state::Battle;

use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, reconstruct, HumanAction};

// ------------------------------------------------------------------- main

fn alive(b: &Battle, side: usize) -> usize {
    b.sides[side]
        .party
        .iter()
        .filter(|&&s| !b.sides[side].roster[s as usize].fainted)
        .count()
}

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

struct Row {
    battle: usize,
    side: usize,
    turn: u16,
    human: String,
    exact: f64,
    width: f64,
    stop: u16,
    eval: f64,
    alive0: usize,
    alive1: usize,
    total_hp: u64,
    one_sided_heal: bool,
    two_sided_heal: bool,
    desc: String,
}

const SOLVED_W: f64 = 0.05;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-49");
    let hp_cap: u64 = arg_s(&args, "--hp-cap", "150").parse().unwrap();
    let work: usize = arg_s(&args, "--work", "1000000").parse().unwrap();
    let node_budget: usize = arg_s(&args, "--nodes", "120000").parse().unwrap();
    let per_battle: usize = arg_s(&args, "--per-battle", "2").parse().unwrap();
    let out_path = arg_s(&args, "--out", "tmp/endgame-exactness-corpus.csv");
    let side_filter = args
        .iter()
        .position(|a| a == "--side")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse::<usize>().unwrap());
    let turn_filter = args
        .iter()
        .position(|a| a == "--turn")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse::<u16>().unwrap());
    let dead_damage_quotient = !args.iter().any(|a| a == "--no-dead-damage-quotient");
    let fold_terminal_nodes = !args.iter().any(|a| a == "--no-fold-terminal-nodes");
    let fold_closed_nodes = !args.iter().any(|a| a == "--no-fold-closed-nodes");
    let monotone_stall_scheduling = !args.iter().any(|a| a == "--no-monotone-stall");
    let two_sided_resource_scheduling = !args.iter().any(|a| a == "--no-two-sided-resource");
    let certified_action_pruning = !args.iter().any(|a| a == "--no-action-pruning");
    let support_br_scheduling = !args.iter().any(|a| a == "--no-support-br");
    let diagnose_stall = args.iter().any(|a| a == "--diagnose-stall");
    let (lo, hi) = {
        let mut it = range.split('-');
        (
            it.next().unwrap_or("0").parse::<usize>().unwrap_or(0),
            it.next().unwrap_or("49").parse::<usize>().unwrap_or(49),
        )
    };

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let weights = EvalWeights::default();

    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    println!(
        "corpus battles {} (index {lo}-{hi}), hp-cap {hp_cap}, work {work}, nodes {node_budget}, per-battle {per_battle}, dead-damage-quotient {dead_damage_quotient}, fold-terminal {fold_terminal_nodes}, fold-closed {fold_closed_nodes}, monotone-stall {monotone_stall_scheduling}, two-sided-resource {two_sided_resource_scheduling}, action-pruning {certified_action_pruning}, support-br {support_br_scheduling}",
        files.len()
    );

    let mut seen: HashSet<u64> = HashSet::new();
    let mut rows: Vec<Row> = Vec::new();
    let mut reconstructed = 0usize;
    let mut attempted = 0usize;
    let mut aborted = 0usize;
    let t0 = Instant::now();

    let mut total_runs = 0usize;
    let mut total_expansions = 0usize;
    let mut total_nodes = 0usize;
    let mut total_closed = 0usize;
    let mut total_peak_nodes = 0usize;
    let mut total_closed_folds = 0usize;
    let mut monotone_roots = 0usize;
    let mut monotone_invalidations = 0usize;
    let mut two_sided_roots = 0usize;
    let mut two_sided_invalidations = 0usize;
    let mut one_sided_handoffs = 0usize;
    let mut min_healing_pp: Option<i64> = None;
    let mut min_resource_pp: Option<i64> = None;
    let mut dominated_rows = 0usize;
    let mut dominated_cols = 0usize;
    let mut dominance_checks = 0usize;
    let mut avoided_cells = 0usize;
    let mut lower_br_picks = 0usize;
    let mut upper_br_picks = 0usize;
    let mut legacy_support_picks = 0usize;
    let mut fair_cell_picks = 0usize;
    let mut worst_gap = 0.0f64;
    for (bi, path) in &files {
        let cb = load_battle(path);
        // one graph per battle: positions inside a battle share subgames
        let mut solver = BoundSolver::new(
            &dex,
            BoundConfig {
                work_budget: work,
                node_budget,
                dead_damage_quotient,
                fold_terminal_nodes,
                fold_closed_nodes,
                monotone_stall_scheduling,
                two_sided_resource_scheduling,
                certified_action_pruning,
                support_br_scheduling,
                ..BoundConfig::default()
            },
        );
        let mut battle_attempts = 0usize;
        // walk decisions from the END: endgames live there
        for d in cb.decisions.iter().rev() {
            if side_filter.is_some_and(|side| d.side != side)
                || turn_filter.is_some_and(|turn| d.turn != turn)
            {
                continue;
            }
            if battle_attempts >= per_battle {
                break;
            }
            let Some(b) = reconstruct(&dex, &src, &pool_path, &cb.lines, &cb.eaten, d, 1) else {
                continue;
            };
            reconstructed += 1;
            let (a0, a1) = (alive(&b, 0), alive(&b, 1));
            let total_hp: u64 = b
                .sides
                .iter()
                .flat_map(|s| s.party.iter().map(|&sl| s.roster[sl as usize].hp as u64))
                .sum();
            if !(a0 <= 2 && a1 <= 2 && total_hp <= hp_cap) {
                continue;
            }
            if !seen.insert(b.state_key()) {
                continue;
            }
            attempted += 1;
            battle_attempts += 1;
            let one_sided_heal = classify_one_sided_heal(&b, &dex).is_ok();
            let two_sided_heal = classify_two_sided_heal(&b, &dex).is_ok();
            let heal_class = if one_sided_heal {
                "one"
            } else if two_sided_heal {
                "two"
            } else {
                "none"
            };
            if diagnose_stall {
                println!(
                    "  b{bi} T{} stall-class one={:?} two={:?}; active {:?}; leechseed {:?}",
                    d.turn,
                    classify_one_sided_heal(&b, &dex),
                    classify_two_sided_heal(&b, &dex),
                    [b.active_id(0), b.active_id(1)],
                    dex.conds_id("leechseed").map(|id| [
                        b.active_id(0).and_then(|p| b.poke(p).volatile(id).copied()),
                        b.active_id(1).and_then(|p| b.poke(p).volatile(id).copied()),
                    ])
                );
                for s in 0..2 {
                    let p = b.poke(b.active_id(s).unwrap());
                    println!(
                        "    s{s} item {:?} moves {:?}",
                        p.item.map(|id| dex.items.key(id)),
                        p.move_slots
                            .iter()
                            .map(|m| (dex.moves.key(m.id), m.pp))
                            .collect::<Vec<_>>()
                    );
                }
            }
            let ev = eval01(&b, &dex, &weights);
            let ts = Instant::now();
            let rep = solver.solve(&b, Some((ev - 0.02, ev + 0.02)));
            let dt = ts.elapsed().as_secs_f64();
            let human = match &d.action {
                HumanAction::Move(k) => format!("move {k}"),
                HumanAction::Switch(sp) => format!("switch {sp}"),
            };
            let desc = {
                let name = |side: usize| {
                    let s = &b.sides[side];
                    s.party
                        .iter()
                        .filter(|&&sl| !s.roster[sl as usize].fainted)
                        .map(|&sl| {
                            let p = &s.roster[sl as usize];
                            format!("{}({}/{})", dex.species.get(p.species).name, p.hp, p.maxhp)
                        })
                        .collect::<Vec<_>>()
                        .join("+")
                };
                format!("b{bi} T{} s{} {} vs {}", d.turn, d.side, name(0), name(1))
            };
            println!(
                "  b{bi} T{turn} hp{total_hp} {a0}v{a1} heal={heal_class}: [{lo:.3},{hi:.3}] w{w:.3} {stop:?} ({runs} runs, {dt:.0}s)",
                turn = d.turn,
                lo = rep.bounds.lo,
                hi = rep.bounds.hi,
                w = rep.bounds.width(),
                stop = rep.stop,
                runs = rep.runs,
            );
            if rep.stop == Stop::WorkExhausted || rep.stop == Stop::NodeBudget {
                aborted += 1; // still bracketed — the row keeps the partial bounds
            }
            rows.push(Row {
                battle: *bi,
                side: d.side,
                turn: d.turn,
                human,
                exact: rep.bounds.mid(),
                width: rep.bounds.width(),
                stop: rep.stop as u16,
                eval: ev,
                alive0: a0,
                alive1: a1,
                total_hp,
                one_sided_heal,
                two_sided_heal,
                desc,
            });
        }
        total_runs += solver.stats.runs;
        total_expansions += solver.stats.expansions;
        total_nodes += solver.node_count();
        total_closed += solver.closed_count();
        total_peak_nodes += solver.stats.peak_nodes;
        total_closed_folds += solver.stats.closed_folds;
        monotone_roots += solver.stats.monotone_roots;
        monotone_invalidations += solver.stats.monotone_invalidations;
        two_sided_roots += solver.stats.two_sided_roots;
        two_sided_invalidations += solver.stats.two_sided_invalidations;
        one_sided_handoffs += solver.stats.one_sided_handoffs;
        if let Some(value) = solver.stats.min_healing_pp {
            min_healing_pp = Some(min_healing_pp.map_or(value, |old| old.min(value)));
        }
        if let Some(value) = solver.stats.min_resource_pp {
            min_resource_pp = Some(min_resource_pp.map_or(value, |old| old.min(value)));
        }
        dominated_rows += solver.stats.dominated_rows;
        dominated_cols += solver.stats.dominated_cols;
        dominance_checks += solver.stats.dominance_checks;
        avoided_cells += solver.stats.avoided_cells;
        lower_br_picks += solver.stats.lower_br_picks;
        upper_br_picks += solver.stats.upper_br_picks;
        legacy_support_picks += solver.stats.legacy_support_picks;
        fair_cell_picks += solver.stats.fair_cell_picks;
        worst_gap = worst_gap.max(solver.stats.worst_gap);
    }

    // ---- report
    let tight: Vec<&Row> = rows.iter().filter(|r| r.width <= SOLVED_W).collect();
    println!(
        "\nreconstructed {reconstructed}, attempted {attempted}: bracketed {} (tight {}), aborted {aborted}",
        rows.len(),
        tight.len()
    );
    println!(
        "engine runs {total_runs} expansions {total_expansions} live {total_nodes} peak {total_peak_nodes} closed {total_closed}/{total_closed_folds} one-sided {monotone_roots} roots/{monotone_invalidations} invalid two-sided {two_sided_roots} roots/{two_sided_invalidations} invalid/{one_sided_handoffs} handoffs min-pp {min_healing_pp:?}/{min_resource_pp:?} dominance {dominated_rows}r/{dominated_cols}c/{avoided_cells} cells ({dominance_checks} checks) schedule {lower_br_picks}lo/{upper_br_picks}hi/{legacy_support_picks}legacy/{fair_cell_picks}fair worst-gap {worst_gap:.2e}; wall {:.0}s",
        t0.elapsed().as_secs_f64()
    );

    let mut viols: Vec<(f64, &Row)> = rows
        .iter()
        .map(|r| {
            let (lo, hi) = (r.exact - r.width / 2.0, r.exact + r.width / 2.0);
            ((r.eval - hi).max(lo - r.eval).max(0.0), r)
        })
        .filter(|(v, _)| *v > 0.02)
        .collect();
    viols.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!("\nproven bracket violations (>0.02): {}", viols.len());
    for (v, r) in viols.iter().take(15) {
        println!(
            "  margin {v:.3}: eval {:.3} vs [{:.3},{:.3}]  {} (human: {})",
            r.eval,
            r.exact - r.width / 2.0,
            r.exact + r.width / 2.0,
            r.desc,
            r.human
        );
    }

    if !tight.is_empty() {
        let k = tight.len() as f64;
        let bias = tight.iter().map(|r| r.eval - r.exact).sum::<f64>() / k;
        let mae = tight.iter().map(|r| (r.eval - r.exact).abs()).sum::<f64>() / k;
        println!(
            "\ncertified-tight n {}: bias {bias:+.4} MAE {mae:.4}",
            tight.len()
        );
        for r in tight.iter().take(10) {
            println!(
                "  exact {:.3}±{:.3} eval {:.3}  {}",
                r.exact,
                r.width / 2.0,
                r.eval,
                r.desc
            );
        }
    }

    std::fs::create_dir_all("tmp").ok();
    let mut f = std::fs::File::create(&out_path).expect("csv");
    writeln!(
        f,
        "battle,side,turn,human,exact,width,stop,eval,alive0,alive1,total_hp,one_sided_heal,two_sided_heal,desc"
    )
    .unwrap();
    for r in &rows {
        writeln!(
            f,
            "{},{},{},\"{}\",{:.6},{:.6},{},{:.6},{},{},{},{},{},\"{}\"",
            r.battle,
            r.side,
            r.turn,
            r.human,
            r.exact,
            r.width,
            r.stop,
            r.eval,
            r.alive0,
            r.alive1,
            r.total_hp,
            r.one_sided_heal,
            r.two_sided_heal,
            r.desc
        )
        .unwrap();
    }
    println!("\ncsv: {out_path}");
}
