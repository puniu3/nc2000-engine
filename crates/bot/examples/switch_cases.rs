//! Show me the positions: human switched, bot stays.
//!
//! M16b measures the disagreement (switch top-1 24.9% vs move 42.8%) and the
//! identifying experiment — equilibrium switch rate in an exactly solved
//! position — has not produced an admissible row after four CX submissions.
//! This is the cheap instrument that should have run first: dump the actual
//! positions where the human retreated and the bot's search puts its top visit
//! on staying in, with enough context to read them by eye.
//!
//! A case is only listed when the human's switch is INSIDE the bot's action set
//! (`in_set`), so nothing here is an artifact of set imputation dropping the
//! bench mon the human used.
//!
//! Usage: cargo run --release -p nc2000-bot --example switch_cases -- \
//!          [--corpus tmp/corpus-spectator] [--battles 0-39] [--iters 3000] \
//!          [--max 12] [--per-battle 2] [--threads 4] [--out tmp/switch-cases.jsonl]

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::corpus::{
    cfg, corpus_files, load_battle, load_sources, plain, reconstruct_context_with_cfg, HumanAction,
    ReconstructedDecision, SetSources,
};
use nc2000_bot::preview::MetaPool;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

const BOOST_NAMES: [&str; 7] = ["atk", "def", "spa", "spd", "spe", "acc", "eva"];

struct Case {
    battle_idx: usize,
    turn: u16,
    side: usize,
    text: String,
    json: String,
}

/// `Snorlax 62% par atk-1` — one live mon, everything that fits on a line.
fn mon_line(dex: &Dex, b: &Battle, side: usize, slot: usize) -> String {
    let p = &b.sides[side].roster[slot];
    let pct = if p.maxhp > 0 { p.hp * 100 / p.maxhp } else { 0 };
    let mut s = format!("{} {}%", dex.species.key(p.species), pct);
    if !p.status.as_str().is_empty() {
        s.push(' ');
        s.push_str(p.status.as_str());
    }
    for (i, &v) in p.boosts.iter().enumerate() {
        if v != 0 {
            s.push_str(&format!(" {}{v:+}", BOOST_NAMES[i]));
        }
    }
    s
}

/// The active's own moves with PP, so "why not just attack" is visible.
fn moves_line(dex: &Dex, b: &Battle, side: usize, slot: usize) -> String {
    b.sides[side].roster[slot]
        .move_slots
        .iter()
        .map(|m| format!("{}({}/{})", plain(dex.moves.key(m.id)), m.pp, m.maxpp))
        .collect::<Vec<_>>()
        .join(" ")
}

fn bench_line(dex: &Dex, b: &Battle, side: usize) -> String {
    let active = b.sides[side].active;
    let mut out = Vec::new();
    for &slot in b.sides[side].party.iter() {
        if Some(slot) == active {
            continue;
        }
        let p = &b.sides[side].roster[slot as usize];
        if p.fainted {
            continue;
        }
        out.push(mon_line(dex, b, side, slot as usize));
    }
    if out.is_empty() {
        "-".into()
    } else {
        out.join(", ")
    }
}

#[allow(clippy::too_many_arguments)]
fn scan_battle(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    battle_file: &std::path::Path,
    battle_idx: usize,
    iters: u32,
    base_seed: u64,
    per_battle: usize,
    agent_cfg: &nc2000_bot::smmcts::RmConfig,
    mode: &str,
) -> Vec<Case> {
    let corpus_battle = load_battle(battle_file);
    let mut cases = Vec::new();

    for (di, d) in corpus_battle.decisions.iter().enumerate() {
        if cases.len() >= per_battle {
            break;
        }
        let want_status = mode == "status";
        let human_label = match (&d.action, want_status) {
            (HumanAction::Switch(t), false) => format!("switch {t}"),
            (HumanAction::Move(k), true) => {
                // Cluster 2: the human set something up, the bot swung.
                let cat = dex
                    .moves
                    .id(&plain(k))
                    .map(|id| dex.moves.get(id).category.clone())
                    .unwrap_or_default();
                if cat != "Status" {
                    continue;
                }
                format!("move {k}")
            }
            _ => continue,
        };
        let seed = base_seed
            ^ (battle_idx as u64).wrapping_mul(0x9E37_79B9_7F4A)
            ^ (di as u64).wrapping_mul(0xBF58_476D)
            ^ d.side as u64;
        let Some(reconstructed) = reconstruct_context_with_cfg(
            dex,
            src,
            pool.clone(),
            &corpus_battle.lines,
            &corpus_battle.evidence,
            d,
            seed,
            agent_cfg.clone(),
        ) else {
            continue;
        };
        let ReconstructedDecision {
            mut agent,
            active_slot,
            revealed_moves,
            ..
        } = reconstructed;
        if agent.step(dex, iters).is_err() {
            continue;
        }
        let (Some(battle), Some(search)) = (agent.battle().cloned(), agent.search()) else {
            continue;
        };

        let norm = |choice: SearchChoice| -> String {
            match choice {
                SearchChoice::Move(id) => format!("move {}", plain(dex.moves.key(id))),
                SearchChoice::Switch(pos) => {
                    let slot = battle.sides[d.side]
                        .party
                        .get(pos as usize - 1)
                        .copied()
                        .unwrap_or(0);
                    format!("switch {}", dex.species.key(battle.sides[d.side].roster[slot as usize].species))
                }
                other => other.to_input(dex),
            }
        };
        let actions: Vec<String> = search.actions().iter().map(|&c| norm(c)).collect();
        let visits = search.visits();
        let total: u32 = visits.iter().sum();
        if total == 0 || actions.is_empty() {
            continue;
        }
        let human = human_label.clone();
        let mut order: Vec<usize> = (0..actions.len()).collect();
        order.sort_by(|&a, &z| visits[z].cmp(&visits[a]));
        let best = order[0];
        // The case we are after: the human played the plan, the bot did not.
        let best_is_damage = actions[best]
            .strip_prefix("move ")
            .and_then(|k| dex.moves.id(k))
            .map(|id| dex.moves.get(id).category != "Status")
            .unwrap_or(false);
        if want_status {
            if !best_is_damage {
                continue;
            }
        } else if actions[best].starts_with("switch") {
            continue;
        }
        // Only score it if the human's bench mon actually exists in the bot's
        // action set; otherwise this is imputation noise, not a judgement gap.
        let Some(rank) = order.iter().position(|&i| actions[i] == human) else {
            continue;
        };
        let switch_mass: u32 = actions
            .iter()
            .zip(visits)
            .filter(|(a, _)| a.starts_with("switch"))
            .map(|(_, v)| *v)
            .sum();

        let foe = 1 - d.side;
        let foe_slot = battle.sides[foe].active.map(|s| s as usize);
        let top: Vec<String> = order
            .iter()
            .take(3)
            .map(|&i| format!("{} {:.0}%", actions[i], 100.0 * visits[i] as f64 / total as f64))
            .collect();
        let ctx: Vec<&str> = corpus_battle.lines[..d.cut.min(corpus_battle.lines.len())]
            .iter()
            .rev()
            .filter(|l| !l.trim().is_empty() && !l.starts_with("|upkeep"))
            .take(6)
            .map(|s| s.as_str())
            .collect();

        let mut text = String::new();
        text.push_str(&format!(
            "battle {battle_idx}  turn {}  p{}  ({} own moves revealed)\n",
            d.turn,
            d.side + 1,
            revealed_moves
        ));
        text.push_str(&format!(
            "  self  {}\n        moves {}\n",
            mon_line(dex, &battle, d.side, active_slot),
            moves_line(dex, &battle, d.side, active_slot)
        ));
        text.push_str(&format!("  bench {}\n", bench_line(dex, &battle, d.side)));
        if let Some(fs) = foe_slot {
            text.push_str(&format!(
                "  foe   {}\n        moves {} (imputed where unrevealed)\n",
                mon_line(dex, &battle, foe, fs),
                moves_line(dex, &battle, foe, fs)
            ));
            text.push_str(&format!("  foe bench {}\n", bench_line(dex, &battle, foe)));
        }
        text.push_str(&format!(
            "  HUMAN {human}\n  BOT   {}   [switch mass {:.2}, human ranked {}/{}]\n",
            top.join("  |  "),
            switch_mass as f64 / total as f64,
            rank + 1,
            actions.len()
        ));
        text.push_str("  log   ");
        text.push_str(
            &ctx.iter()
                .rev()
                .copied()
                .collect::<Vec<_>>()
                .join("\n        "),
        );
        text.push('\n');

        cases.push(Case {
            battle_idx,
            turn: d.turn,
            side: d.side,
            text,
            json: serde_json::json!({
                "battle": battle_idx, "turn": d.turn, "side": d.side,
                "human": human, "bot_top": actions[best],
                "switch_mass": switch_mass as f64 / total as f64,
                "human_rank": rank + 1, "n_actions": actions.len(),
                "self": mon_line(dex, &battle, d.side, active_slot),
                "foe": foe_slot.map(|fs| mon_line(dex, &battle, foe, fs)),
                "bench": bench_line(dex, &battle, d.side),
                "revealed": revealed_moves,
            })
            .to_string(),
        });
    }
    cases
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-39");
    let iters = arg(&args, "--iters", 3000) as u32;
    let seed = arg(&args, "--seed", 1) as u64;
    let max = arg(&args, "--max", 12);
    let per_battle = arg(&args, "--per-battle", 2);
    let threads = arg(&args, "--threads", 4);
    let out_path = arg_s(&args, "--out", "tmp/switch-cases.jsonl");
    // "switch" (default): human retreated, bot stayed. "status": human played a
    // status move, bot played a damaging one -- M16b cluster 2.
    let mode = arg_s(&args, "--human-class", "switch");

    let (lo, hi) = {
        let mut it = range.split('-');
        let a: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let b: usize = it.next().unwrap_or("39").parse().unwrap_or(39);
        (a, b)
    };

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool = nc2000_bot::preview::load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let agent_cfg = cfg();
    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    eprintln!(
        "scanning {} battles (index {lo}-{hi}) at {iters} iters on {threads} threads",
        files.len()
    );

    let found = Mutex::new(Vec::new());
    let cursor = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (battle_idx, path) = &files[j];
                let cases = scan_battle(
                    &dex,
                    &src,
                    &pool,
                    path,
                    *battle_idx,
                    iters,
                    seed,
                    per_battle,
                    &agent_cfg,
                    &mode,
                );
                if !cases.is_empty() {
                    found.lock().unwrap().extend(cases);
                }
            });
        }
    });

    let mut cases = found.into_inner().unwrap();
    cases.sort_by_key(|c| (c.battle_idx, c.turn, c.side));
    eprintln!("{} cases (human switched, bot stays, switch in action set)", cases.len());

    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    for c in &cases {
        writeln!(out, "{}", c.json).unwrap();
    }
    for c in cases.iter().take(max) {
        println!("{}", c.text);
    }
    eprintln!("full list -> {out_path}");
}
