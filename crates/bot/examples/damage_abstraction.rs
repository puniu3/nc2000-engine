//! Matched benchmark for exact vs semantically bucketed damage rolls.
//!
//! Every arm uses the same reconstructed position, legal actions, full-turn
//! horizon, static leaf evaluator, and matrix solver. Only the damage-roll
//! partition changes. The exact arm supplies the reference root matrix for
//! value error and policy exploitability. Optional certified corpus anchors
//! measure all arms against previously solved true-value intervals.
//!
//! cargo run --release -p nc2000-bot --example damage_abstraction -- \
//!   --corpus /path/to/corpus-spectator --battles 0-99 --positions 40 \
//!   --horizon 0 --work 2000000 --out /path/to/results.csv

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use nc2000_bot::corpus::{corpus_files, load_battle, load_sources, reconstruct, HumanAction};
use nc2000_bot::damage_search::{
    policy_regret, DamageSearch, DamageSearchConfig, DamageSearchReport, DamageSearchStats,
};
use nc2000_engine::prng::DamageRollMode;
use nc2000_engine::state::Battle;

fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|item| item == key)
        .and_then(|index| args.get(index + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn flag(args: &[String], key: &str) -> bool {
    args.iter().any(|item| item == key)
}

fn range(value: &str) -> (usize, usize) {
    let mut parts = value.split('-');
    let lo = parts.next().unwrap_or("0").parse().unwrap_or(0);
    let hi = parts.next().unwrap_or("0").parse().unwrap_or(lo);
    (lo, hi)
}

fn repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("NC2000_REPO_ROOT") {
        return PathBuf::from(root);
    }
    let current = std::env::current_dir().unwrap();
    if current.join("data/gen2stadium2.json").is_file() {
        return current;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn alive(battle: &Battle, side: usize) -> usize {
    battle.sides[side]
        .party
        .iter()
        .filter(|&&slot| {
            let pokemon = &battle.sides[side].roster[slot as usize];
            !pokemon.fainted && pokemon.hp > 0
        })
        .count()
}

fn total_hp(battle: &Battle) -> u64 {
    battle
        .sides
        .iter()
        .flat_map(|side| {
            side.party
                .iter()
                .map(|&slot| side.roster[slot as usize].hp.max(0) as u64)
        })
        .sum()
}

fn mode_name(mode: DamageRollMode) -> &'static str {
    match mode {
        DamageRollMode::Exact => "exact",
        DamageRollMode::Mean => "mean",
        DamageRollMode::Threshold1 => "threshold1",
        DamageRollMode::Threshold2 => "threshold2",
    }
}

fn parse_modes(value: &str) -> Vec<DamageRollMode> {
    value
        .split(',')
        .filter_map(|name| match name {
            "exact" => Some(DamageRollMode::Exact),
            "mean" => Some(DamageRollMode::Mean),
            "threshold1" | "t1" => Some(DamageRollMode::Threshold1),
            "threshold2" | "t2" => Some(DamageRollMode::Threshold2),
            _ => None,
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Anchor {
    exact: f64,
    width: f64,
}

fn load_anchors(path: &str) -> HashMap<(usize, usize, u16), Anchor> {
    if path.is_empty() {
        return HashMap::new();
    }
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|error| panic!("anchors {path}: {error}"));
    text.lines()
        .skip(1)
        .filter_map(|line| {
            // The first seven fields contain no commas except the quoted
            // human-action field, whose current corpus values contain none.
            let fields: Vec<&str> = line.split(',').collect();
            Some((
                (
                    fields.first()?.parse().ok()?,
                    fields.get(1)?.parse().ok()?,
                    fields.get(2)?.parse().ok()?,
                ),
                Anchor {
                    exact: fields.get(4)?.parse().ok()?,
                    width: fields.get(5)?.parse().ok()?,
                },
            ))
        })
        .collect()
}

struct Position {
    battle_index: usize,
    side: usize,
    turn: u16,
    action: String,
    alive: [usize; 2],
    total_hp: u64,
    battle: Battle,
    anchor: Option<Anchor>,
}

struct TimedReport {
    report: Option<DamageSearchReport>,
    stats: DamageSearchStats,
    seconds: f64,
}

#[derive(Default)]
struct Summary {
    attempted: usize,
    self_completed: usize,
    completed: usize,
    value_abs: f64,
    value_worst: f64,
    policy_regret: f64,
    policy_worst: f64,
    runs: u128,
    exact_runs: u128,
    all_runs: u128,
    all_exact_runs: u128,
    matched_seconds: f64,
    matched_exact_seconds: f64,
    all_seconds: f64,
    all_exact_seconds: f64,
    anchor_n: usize,
    anchor_abs: f64,
    anchor_outside: f64,
}

fn run_search(
    dex: &nc2000_engine::dex::Dex,
    battle: &Battle,
    mode: DamageRollMode,
    horizon: u16,
    state_budget: usize,
    work_budget: usize,
    leaf_cap: usize,
) -> TimedReport {
    let mut search = DamageSearch::new(
        dex,
        DamageSearchConfig {
            horizon,
            damage_mode: mode,
            state_budget,
            work_budget,
            leaf_cap,
            ..Default::default()
        },
    );
    let start = Instant::now();
    let report = search.solve(battle);
    let seconds = start.elapsed().as_secs_f64();
    let stats = search.stats().clone();
    TimedReport {
        report,
        stats,
        seconds,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = PathBuf::from(arg(&args, "--corpus", "tmp/corpus-spectator"));
    let battle_range = range(&arg(&args, "--battles", "0-99"));
    let positions_limit: usize = arg(&args, "--positions", "40").parse().unwrap();
    let per_battle: usize = arg(&args, "--per-battle", "2").parse().unwrap();
    let alive_max: usize = arg(&args, "--alive-max", "2").parse().unwrap();
    let hp_cap: u64 = arg(&args, "--hp-cap", "500").parse().unwrap();
    let horizon: u16 = arg(&args, "--horizon", "0").parse().unwrap();
    let state_budget: usize = arg(&args, "--states", "100000").parse().unwrap();
    let work_budget: usize = arg(&args, "--work", "2000000").parse().unwrap();
    let leaf_cap: usize = arg(&args, "--leaf-cap", "100000").parse().unwrap();
    let out_path = arg(&args, "--out", "tmp/damage-abstraction.csv");
    let anchor_path = arg(&args, "--anchors", "");
    let anchor_only = flag(&args, "--anchor-only");
    let verbose = flag(&args, "--verbose");
    let mut modes = parse_modes(&arg(&args, "--modes", "exact,mean,threshold1,threshold2"));
    if !modes.contains(&DamageRollMode::Exact) {
        modes.insert(0, DamageRollMode::Exact);
    }

    let root = repo_root();
    let dex_json = std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap();
    let dex = nc2000_engine::dex::Dex::from_json(&dex_json).unwrap();
    let corpus = if corpus.is_absolute() {
        corpus
    } else {
        root.join(corpus)
    };
    let sources = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let anchors = load_anchors(&anchor_path);

    let files: Vec<(usize, PathBuf)> = corpus_files(Path::new(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(index, _)| *index >= battle_range.0 && *index <= battle_range.1)
        .collect();
    let mut positions = Vec::new();
    let mut seen = HashSet::new();
    for (battle_index, path) in files {
        if positions.len() >= positions_limit {
            break;
        }
        let corpus_battle = load_battle(&path);
        let mut accepted = 0;
        for decision in corpus_battle.decisions.iter().rev() {
            if positions.len() >= positions_limit || accepted >= per_battle {
                break;
            }
            let key = (battle_index, decision.side, decision.turn);
            let anchor = anchors.get(&key).copied();
            if anchor_only && anchor.is_none() {
                continue;
            }
            let Some(battle) = reconstruct(
                &dex,
                &sources,
                &pool_path,
                &corpus_battle.lines,
                &corpus_battle.eaten,
                decision,
                1,
            ) else {
                continue;
            };
            let alive_now = [alive(&battle, 0), alive(&battle, 1)];
            let hp = total_hp(&battle);
            if alive_now[0] > alive_max || alive_now[1] > alive_max || hp > hp_cap {
                continue;
            }
            if !seen.insert(battle.state_key128()) {
                continue;
            }
            let action = match &decision.action {
                HumanAction::Move(name) => format!("move {name}"),
                HumanAction::Switch(name) => format!("switch {name}"),
            };
            positions.push(Position {
                battle_index,
                side: decision.side,
                turn: decision.turn,
                action,
                alive: alive_now,
                total_hp: hp,
                battle,
                anchor,
            });
            accepted += 1;
        }
    }

    println!(
        "positions {} battles {}-{} horizon {} work {} modes {} anchors {}",
        positions.len(),
        battle_range.0,
        battle_range.1,
        horizon,
        work_budget,
        modes
            .iter()
            .map(|&mode| mode_name(mode))
            .collect::<Vec<_>>()
            .join(","),
        anchors.len(),
    );

    if let Some(parent) = Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut output = std::fs::File::create(&out_path).expect("output csv");
    writeln!(
        output,
        "battle,side,turn,action,alive0,alive1,total_hp,mode,complete,value,value_abs_exact,row_regret,col_regret,policy_regret,runs,leaves,states,cells,wall_s,exact_runs,exact_wall_s,anchor,anchor_width,anchor_abs,anchor_outside"
    )
    .unwrap();

    let mut summaries: HashMap<&'static str, Summary> = HashMap::new();
    for (index, position) in positions.iter().enumerate() {
        let exact = run_search(
            &dex,
            &position.battle,
            DamageRollMode::Exact,
            horizon,
            state_budget,
            work_budget,
            leaf_cap,
        );
        let exact_report = exact.report.as_ref();
        println!(
            "[{}/{}] b{} T{} {}v{} hp{} exact {} {:.3}s{}",
            index + 1,
            positions.len(),
            position.battle_index,
            position.turn,
            position.alive[0],
            position.alive[1],
            position.total_hp,
            exact.stats.chance_runs,
            exact.seconds,
            if exact_report.is_some() { "" } else { " ABORT" },
        );
        if verbose {
            let needs = position.battle.needs_choice();
            let mut probe = position.battle.clone();
            let choices = |probe: &mut Battle, side: usize| {
                if needs[side] {
                    probe.legal_choices(&dex, side)
                } else {
                    Vec::new()
                }
            };
            let row = choices(&mut probe, 0);
            let col = choices(&mut probe, 1);
            println!("    actions row {:?} col {:?}", row, col);
        }

        for &mode in &modes {
            let timed = if mode == DamageRollMode::Exact {
                TimedReport {
                    report: exact.report.clone(),
                    stats: exact.stats.clone(),
                    seconds: exact.seconds,
                }
            } else {
                run_search(
                    &dex,
                    &position.battle,
                    mode,
                    horizon,
                    state_budget,
                    work_budget,
                    leaf_cap,
                )
            };
            if verbose {
                match timed.report.as_ref() {
                    Some(report) => println!(
                        "    {} value {:.9} dims {:?} row {:?} col {:?} matrix {:?}",
                        mode_name(mode),
                        report.value,
                        report.dims,
                        report.row_policy,
                        report.col_policy,
                        report.matrix,
                    ),
                    None => println!("    {} ABORT", mode_name(mode)),
                }
            }
            let name = mode_name(mode);
            let summary = summaries.entry(name).or_default();
            summary.attempted += 1;
            summary.self_completed += usize::from(timed.report.is_some());
            summary.all_seconds += timed.seconds;
            summary.all_exact_seconds += exact.seconds;
            summary.all_runs += timed.stats.chance_runs as u128;
            summary.all_exact_runs += exact.stats.chance_runs as u128;

            let (
                complete,
                value,
                value_abs,
                row_regret,
                col_regret,
                regret,
                runs,
                leaves,
                states,
                cells,
            ) = match (&timed.report, exact_report) {
                (Some(report), Some(reference)) if report.dims == reference.dims => {
                    let (row, col, sum) = policy_regret(
                        &reference.matrix,
                        reference.dims,
                        reference.value,
                        &report.row_policy,
                        &report.col_policy,
                    );
                    let error = (report.value - reference.value).abs();
                    summary.completed += 1;
                    summary.value_abs += error;
                    summary.value_worst = summary.value_worst.max(error);
                    summary.policy_regret += sum;
                    summary.policy_worst = summary.policy_worst.max(sum);
                    summary.runs += report.stats.chance_runs as u128;
                    summary.exact_runs += reference.stats.chance_runs as u128;
                    summary.matched_seconds += timed.seconds;
                    summary.matched_exact_seconds += exact.seconds;
                    (
                        1,
                        report.value,
                        error,
                        row,
                        col,
                        sum,
                        report.stats.chance_runs,
                        report.stats.leaves,
                        report.stats.states,
                        report.stats.matrix_cells,
                    )
                }
                (Some(report), _) => (
                    1,
                    report.value,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    report.stats.chance_runs,
                    report.stats.leaves,
                    report.stats.states,
                    report.stats.matrix_cells,
                ),
                (None, _) => (
                    0,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    f64::NAN,
                    timed.stats.chance_runs,
                    timed.stats.leaves,
                    timed.stats.states,
                    timed.stats.matrix_cells,
                ),
            };

            let (anchor_value, anchor_width, anchor_abs, anchor_outside) =
                if let (Some(anchor), Some(report)) = (position.anchor, timed.report.as_ref()) {
                    let error = (report.value - anchor.exact).abs();
                    let outside = (error - anchor.width * 0.5).max(0.0);
                    summary.anchor_n += 1;
                    summary.anchor_abs += error;
                    summary.anchor_outside += outside;
                    (anchor.exact, anchor.width, error, outside)
                } else {
                    (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
                };

            writeln!(
                output,
                "{},{},{},\"{}\",{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{:.6},{},{:.6},{:.9},{:.9},{:.9},{:.9}",
                position.battle_index,
                position.side,
                position.turn,
                position.action,
                position.alive[0],
                position.alive[1],
                position.total_hp,
                name,
                complete,
                value,
                value_abs,
                row_regret,
                col_regret,
                regret,
                runs,
                leaves,
                states,
                cells,
                timed.seconds,
                exact.stats.chance_runs,
                exact.seconds,
                anchor_value,
                anchor_width,
                anchor_abs,
                anchor_outside,
            )
            .unwrap();
        }
    }

    println!("\nmode summary (matched completed positions):");
    for &mode in &modes {
        let name = mode_name(mode);
        let summary = summaries.get(name).unwrap();
        let n = summary.completed.max(1) as f64;
        let run_ratio = summary.runs as f64 / summary.exact_runs.max(1) as f64;
        let wall_ratio = summary.matched_seconds / summary.matched_exact_seconds.max(f64::EPSILON);
        let all_run_ratio = summary.all_runs as f64 / summary.all_exact_runs.max(1) as f64;
        let all_wall_ratio = summary.all_seconds / summary.all_exact_seconds.max(f64::EPSILON);
        println!(
            "  {name:10} self {}/{} matched {} value-MAE {:.5} worst {:.5} policy-regret {:.5} worst {:.5}",
            summary.self_completed,
            summary.attempted,
            summary.completed,
            summary.value_abs / n,
            summary.value_worst,
            summary.policy_regret / n,
            summary.policy_worst,
        );
        println!(
            "             matched runs {:.3}x wall {:.3}x; all-attempt work {:.3}x wall {:.3}x",
            run_ratio, wall_ratio, all_run_ratio, all_wall_ratio,
        );
        if summary.anchor_n > 0 {
            println!(
                "             anchors {} mid-MAE {:.5} interval-outside {:.5}",
                summary.anchor_n,
                summary.anchor_abs / summary.anchor_n as f64,
                summary.anchor_outside / summary.anchor_n as f64,
            );
        }
    }
    println!("csv: {out_path}");
}
