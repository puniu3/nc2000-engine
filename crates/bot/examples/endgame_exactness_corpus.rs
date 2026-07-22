//! M17e step 3 — eval vs certified endgame brackets on HUMAN-GAME positions.
//!
//! Same measurement as `endgame_exactness`, but positions come from the
//! 570-battle spectator corpus (real endgames reached by real play) instead
//! of random-legal self-play — owner decision 2026-07-21: similar positions
//! recur in live games, so anchor the eval where it will actually be asked.
//!
//! Position reconstruction is the shared `bot::corpus` path: battle state
//! and opponent information come from the protocol prefix, while full-log
//! own-side move/item reveals stand in for the submitted team a live bot
//! would know; remaining own-set fields come from rentals/pool/learnsets.
//! The exact value is therefore exact FOR THE IMPUTED DETERMINIZATION — the
//! same full-info state family the eval scores, not the true hidden-set game.
//!
//! Reports certified-tight comparisons plus PROVEN bracket violations
//! (eval outside a certified interval, any width — zero playouts involved).
//!
//! Formal three-shard sweep (each command must use the same built binary):
//!   cargo run --release -p nc2000-bot --example endgame_exactness_corpus -- \
//!     --shard 0/3 --out tmp/m17e-0.json
//!   # repeat with 1/3 and 2/3, then:
//!   python3 tools/merge-m17e.py --out tmp/m17e-merged-v3.json tmp/m17e-{0,1,2}.json
//!   cargo run --release -p nc2000-bot --example anchor_gate -- \
//!     --artifact tmp/m17e-merged-v3.json
//!
//! Use `--battles LO-HI` only for custom/manual partitioning. A merge is
//! accepted only when its ranges cover the entire content-bound corpus.
//!
//! Solver/selection overrides: [--side N] [--turn N] [--hp-cap N]
//!        [--work N] [--nodes N] [--cell-cap N] [--solver-eps F]
//!        [--trial-depth N] [--descend-floor F] [--threshold-radius F]
//!        [--per-battle N] [--reconstruction-seed N] [--no-dead-damage-quotient]
//!        [--no-fold-terminal-nodes] [--no-fold-closed-nodes]
//!        [--no-monotone-stall] [--no-two-sided-resource]
//!        [--no-action-pruning] [--no-support-br] [--diagnose-stall] [--out JSON]

use std::collections::HashSet;
use std::time::Instant;

use nc2000_bot::bounds::{BoundConfig, BoundSolver, Stop};
use nc2000_bot::eval::{eval01, EvalWeights};
use nc2000_bot::stall::{classify_one_sided_heal, classify_two_sided_heal};
use nc2000_engine::state::Battle;

use nc2000_bot::corpus::{
    corpus_files, corpus_fingerprint, load_battle, load_sources, reconstruct, HumanAction,
};
use nc2000_bot::m17e_artifact::{
    generator_executable_fingerprint, runtime_data_fingerprints, solver_build_fingerprint,
    summarize_rows, validate_shard, Row, RunIdentity, SelectionConfig, ShardArtifact,
    ShardDescriptor, SolverConfig, CUSTOM_PROFILE, FORMAL_PROFILE, SHARD_SCHEMA,
};

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

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|value| value == key)
        .and_then(|index| args.get(index + 1))
        .cloned()
}

fn validate_args(args: &[String]) {
    const VALUES: &[&str] = &[
        "--corpus",
        "--battles",
        "--shard",
        "--side",
        "--turn",
        "--hp-cap",
        "--work",
        "--nodes",
        "--cell-cap",
        "--solver-eps",
        "--trial-depth",
        "--descend-floor",
        "--threshold-radius",
        "--per-battle",
        "--reconstruction-seed",
        "--out",
    ];
    const FLAGS: &[&str] = &[
        "--no-dead-damage-quotient",
        "--no-fold-terminal-nodes",
        "--no-fold-closed-nodes",
        "--no-monotone-stall",
        "--no-two-sided-resource",
        "--no-action-pruning",
        "--no-support-br",
        "--diagnose-stall",
        "--help",
    ];
    let mut index = 1;
    while index < args.len() {
        let key = args[index].as_str();
        if VALUES.contains(&key) {
            assert!(index + 1 < args.len(), "{key} requires a value");
            index += 2;
        } else if FLAGS.contains(&key) {
            index += 1;
        } else {
            panic!("unknown argument {key}; use --help");
        }
    }
}

fn battle_range(args: &[String], corpus_count: usize) -> (usize, usize) {
    assert!(corpus_count > 0, "corpus is empty");
    let explicit = arg(args, "--battles");
    let shard = arg(args, "--shard");
    assert!(
        explicit.is_none() || shard.is_none(),
        "use exactly one of --battles and --shard"
    );
    let (lo, hi) = if let Some(range) = explicit {
        let (lo, hi) = range
            .split_once('-')
            .unwrap_or_else(|| panic!("--battles must be LO-HI"));
        (
            lo.parse().expect("battle LO"),
            hi.parse().expect("battle HI"),
        )
    } else if let Some(shard) = shard {
        let (index, total) = shard
            .split_once('/')
            .unwrap_or_else(|| panic!("--shard must be INDEX/TOTAL"));
        let index: usize = index.parse().expect("shard INDEX");
        let total: usize = total.parse().expect("shard TOTAL");
        assert!(
            total > 0 && index < total,
            "--shard requires 0 <= INDEX < TOTAL"
        );
        let lo = index * corpus_count / total;
        let end = (index + 1) * corpus_count / total;
        assert!(lo < end, "--shard TOTAL exceeds corpus size");
        (lo, end - 1)
    } else {
        (0, corpus_count - 1)
    };
    assert!(
        lo <= hi && hi < corpus_count,
        "battle range {lo}-{hi} is outside 0-{}",
        corpus_count - 1
    );
    (lo, hi)
}

const SOLVED_W: f64 = 0.05;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    validate_args(&args);
    if args.iter().any(|arg| arg == "--help") {
        println!("formal: endgame_exactness_corpus --shard INDEX/TOTAL --out SHARD.json\nmerge: python3 tools/merge-m17e.py --out MERGED.json SHARD.json...");
        return;
    }
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let hp_cap: u64 = arg_s(&args, "--hp-cap", "150").parse().unwrap();
    let work: usize = arg_s(&args, "--work", "1000000").parse().unwrap();
    let node_budget: usize = arg_s(&args, "--nodes", "120000").parse().unwrap();
    let cell_cap: usize = arg_s(&args, "--cell-cap", "4096").parse().unwrap();
    let solver_eps: f64 = arg_s(&args, "--solver-eps", "0.02").parse().unwrap();
    let trial_depth: usize = arg_s(&args, "--trial-depth", "24").parse().unwrap();
    let descend_floor: f64 = arg_s(&args, "--descend-floor", "0.1").parse().unwrap();
    let threshold_radius: f64 = arg_s(&args, "--threshold-radius", "0.02").parse().unwrap();
    let per_battle: usize = arg_s(&args, "--per-battle", "2").parse().unwrap();
    let reconstruction_seed: u64 = arg_s(&args, "--reconstruction-seed", "1").parse().unwrap();
    let out_path = arg_s(&args, "--out", "tmp/m17e-shard-v3.json");
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
    assert!(per_battle > 0 && work > 0 && node_budget > 0 && cell_cap > 0 && trial_depth > 0);
    assert!(solver_eps.is_finite() && solver_eps > 0.0);
    assert!(descend_floor.is_finite() && descend_floor >= 0.0);
    assert!(threshold_radius.is_finite() && threshold_radius >= 0.0);
    assert!(
        side_filter.is_none_or(|side| side <= 1),
        "--side must be 0 or 1"
    );

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let weights = EvalWeights::default();

    let all_files = corpus_files(&root.join(&corpus));
    let corpus_id = corpus_fingerprint(&all_files);
    let corpus_count = all_files.len();
    let (lo, hi) = battle_range(&args, corpus_count);
    let solver_config = SolverConfig {
        work_budget: work,
        node_budget,
        cell_cap,
        eps: solver_eps,
        trial_depth,
        descend_floor,
        dead_damage_quotient,
        fold_terminal_nodes,
        fold_closed_nodes,
        monotone_stall_scheduling,
        two_sided_resource_scheduling,
        certified_action_pruning,
        support_br_scheduling,
        threshold_radius,
    };
    let selection = SelectionConfig {
        hp_cap,
        max_alive_per_side: 2,
        per_battle,
        side_filter,
        turn_filter,
        decision_order: "reverse".to_string(),
        reconstruction_seed,
    };
    let profile =
        if solver_config == SolverConfig::formal() && selection == SelectionConfig::formal() {
            FORMAL_PROFILE
        } else {
            CUSTOM_PROFILE
        };
    let run = RunIdentity {
        profile: profile.to_string(),
        solver_build_fingerprint: solver_build_fingerprint().to_string(),
        generator_executable_fingerprint: generator_executable_fingerprint()
            .unwrap_or_else(|error| panic!("{error}")),
        runtime_data: runtime_data_fingerprints(&root).unwrap_or_else(|error| panic!("{error}")),
        corpus_fingerprint: corpus_id,
        corpus_count,
        solver: solver_config.clone(),
        selection: selection.clone(),
    };
    let files: Vec<(usize, std::path::PathBuf)> = all_files
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    println!(
        "profile {profile}; corpus battles {} (index {lo}-{hi}), hp-cap {hp_cap}, work {work}, nodes {node_budget}, per-battle {per_battle}, dead-damage-quotient {dead_damage_quotient}, fold-terminal {fold_terminal_nodes}, fold-closed {fold_closed_nodes}, monotone-stall {monotone_stall_scheduling}, two-sided-resource {two_sided_resource_scheduling}, action-pruning {certified_action_pruning}, support-br {support_br_scheduling}",
        files.len()
    );

    let mut seen: HashSet<u128> = HashSet::new();
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
                cell_cap,
                eps: solver_eps,
                trial_depth,
                descend_floor,
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
        for (decision, d) in cb.decisions.iter().enumerate().rev() {
            if side_filter.is_some_and(|side| d.side != side)
                || turn_filter.is_some_and(|turn| d.turn != turn)
            {
                continue;
            }
            if battle_attempts >= per_battle {
                break;
            }
            let Some(b) = reconstruct(
                &dex,
                &src,
                &pool_path,
                &cb.lines,
                &cb.evidence,
                d,
                reconstruction_seed,
            ) else {
                continue;
            };
            reconstructed += 1;
            let (a0, a1) = (alive(&b, 0), alive(&b, 1));
            let total_hp: u64 = b
                .sides
                .iter()
                .flat_map(|s| s.party.iter().map(|&sl| s.roster[sl as usize].hp as u64))
                .sum();
            if !(a0 <= selection.max_alive_per_side
                && a1 <= selection.max_alive_per_side
                && total_hp <= hp_cap)
            {
                continue;
            }
            let state_key128 = b.state_key128();
            if !seen.insert(state_key128) {
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
            let rep = solver.solve(&b, Some((ev - threshold_radius, ev + threshold_radius)));
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
                decision,
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
                state_key128: format!("{state_key128:032x}"),
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

    rows.sort_by_key(Row::coordinate);
    let artifact = ShardArtifact {
        schema: SHARD_SCHEMA.to_string(),
        run,
        shard: ShardDescriptor {
            battle_lo: lo,
            battle_hi: hi,
            summary: summarize_rows(&rows),
        },
        rows,
    };
    validate_shard(&artifact).unwrap_or_else(|error| panic!("invalid generated shard: {error}"));
    let out = std::path::Path::new(&out_path);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!("create output directory {}: {error}", parent.display())
        });
    }
    let file = std::fs::File::create(out)
        .unwrap_or_else(|error| panic!("create {}: {error}", out.display()));
    serde_json::to_writer_pretty(file, &artifact).expect("serialize M17e v3 shard");
    println!("\nshard artifact: {out_path}");
}
