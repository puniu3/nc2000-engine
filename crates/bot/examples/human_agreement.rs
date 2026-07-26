//! M16b — human-agreement measurement over the 570-battle spectator corpus.
//!
//! For every observable human decision point, drop the product bot
//! (ProtocolAgent = tracker → synthesize → BlindSearch, skuct config) into
//! the same position and compare its choice with what the human played.
//!
//! The live bot knows its submitted team, but a spectator replay does not.
//! Corpus reconstruction therefore combines prefix-only public battle state
//! with full-log evidence about the acting side's own moves/items, then fills
//! unknown own-set fields from the rentals/meta pool. Opponent information
//! stays prefix-only and runs through the live belief machinery unchanged.
//!
//! Scored only where the human's action is inside the bot's action set
//! (exclusion rate reported per revelation stratum — it is itself a
//! coverage measurement of the imputation).
//!
//! Output: one JSON line per decision point (agreement, bot ranking of the
//! human action, revelation stratum, feature tags, fabrication provenance)
//! → aggregate with tools/aggregate-human-agreement.py.
//!
//! Smoke: cargo run --release -p nc2000-bot --example human_agreement -- \
//!          --corpus tmp/corpus-spectator --battles 0-3 --iters 1000
//! Full (cx): --battles 0-569 --iters 3000 --threads 56

use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::corpus::{
    cfg, corpus_files, load_battle, load_sources, plain, reconstruct_context_with_cfg, HumanAction,
    ReconstructedDecision, SetSources,
};
use nc2000_bot::preview::MetaPool;
use nc2000_engine::dex::Dex;

struct BattleReport {
    lines_out: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn process_battle(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    battle_file: &std::path::Path,
    battle_idx: usize,
    iters: u32,
    base_seed: u64,
    agent_cfg: &nc2000_bot::smmcts::RmConfig,
) -> BattleReport {
    let corpus_battle = load_battle(battle_file);
    let mut out = Vec::new();

    for (di, d) in corpus_battle.decisions.iter().enumerate() {
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
            out.push(
                serde_json::json!({"battle": battle_idx, "side": d.side, "turn": d.turn,
                    "skip": "reconstruct"})
                .to_string(),
            );
            continue;
        };
        let ReconstructedDecision {
            mut agent,
            active_slot,
            revealed_moves,
            provenance,
            imputed_pick,
        } = reconstructed;
        if agent.step(dex, iters).is_err() {
            continue;
        }
        let Some(battle) = agent.battle().cloned() else {
            continue;
        };
        let Some(search) = agent.search() else {
            continue;
        };

        let norm = |choice: nc2000_engine::battle::SearchChoice| -> String {
            match choice {
                nc2000_engine::battle::SearchChoice::Move(id) => {
                    format!("move {}", plain(dex.moves.key(id)))
                }
                nc2000_engine::battle::SearchChoice::Switch(pos) => {
                    let slot = battle.sides[d.side]
                        .party
                        .get(pos as usize - 1)
                        .copied()
                        .unwrap_or(0);
                    let species = battle.sides[d.side].roster[slot as usize].species;
                    format!("switch {}", dex.species.key(species))
                }
                other => other.to_input(dex),
            }
        };
        let actions: Vec<String> = search
            .actions()
            .iter()
            .map(|&choice| norm(choice))
            .collect();
        let visits = search.visits();
        let human = match &d.action {
            HumanAction::Move(key) => format!("move {key}"),
            HumanAction::Switch(species) => format!("switch {species}"),
        };
        let mut order: Vec<usize> = (0..actions.len()).collect();
        order.sort_by(|&a, &z| visits[z].cmp(&visits[a]));
        let bot_best = order
            .first()
            .map(|&i| actions[i].clone())
            .unwrap_or_default();
        let human_rank = order
            .iter()
            .position(|&i| actions[i] == human)
            .map(|rank| rank + 1);
        // M17 switching probe. Two things the agreement number alone cannot
        // distinguish, both visible here for free:
        //   * `top1` — the bot's confidence. A 6-action root whose best action
        //     holds 1/6 of the visits has no opinion, and comparing its argmax
        //     to a human's choice measures a coin flip, not a disagreement.
        //   * the tags — the structural threats a per-mon eval cannot see
        //     (Encore lock, Mean Look trap, a Belly Drum/Curse sweeper), read
        //     off engine volatiles and boosts rather than imputed movesets, so
        //     nothing here depends on fabrication.
        let total_visits: u32 = visits.iter().sum::<u32>().max(1);
        let top1 = visits.iter().copied().max().unwrap_or(0) as f64 / total_visits as f64;
        let switch_mass: u32 = actions
            .iter()
            .zip(visits)
            .filter(|(a, _)| a.starts_with("switch"))
            .map(|(_, v)| *v)
            .sum();
        let foe_side = 1 - d.side;
        let vols = |side: usize, slot: usize| -> Vec<String> {
            battle.sides[side].roster[slot]
                .volatiles
                .iter()
                .map(|(id, _)| dex.conds_key(*id).to_string())
                .filter(|k| {
                    matches!(
                        k.as_str(),
                        "encore"
                            | "trapped"
                            | "partiallytrapped"
                            | "perishsong"
                            | "confusion"
                            | "leechseed"
                            | "substitute"
                            | "lockedmove"
                    )
                })
                .collect()
        };
        let boost_sum = |side: usize, slot: usize| -> i32 {
            battle.sides[side].roster[slot].boosts[..5]
                .iter()
                .map(|&b| b as i32)
                .sum()
        };
        let hp_pct = |side: usize, slot: usize| -> i32 {
            let p = &battle.sides[side].roster[slot];
            if p.maxhp > 0 {
                p.hp * 100 / p.maxhp
            } else {
                0
            }
        };
        let foe_slot = battle.sides[foe_side].active.map(|s| s as usize);

        let class_of = |action: &str| -> &'static str {
            if action.starts_with("switch") {
                "switch"
            } else {
                action
                    .strip_prefix("move ")
                    .and_then(|key| dex.moves.id(key))
                    .map(|id| match dex.moves.get(id).category.as_str() {
                        "Physical" => "Physical",
                        "Special" => "Special",
                        _ => "Status",
                    })
                    .unwrap_or("Status")
            }
        };

        out.push(
            serde_json::json!({
                "battle": battle_idx, "side": d.side, "turn": d.turn,
                "kind": if matches!(d.action, HumanAction::Move(_)) {"move"} else {"switch"},
                "human": human, "bot": bot_best,
                "human_class": class_of(&human), "bot_class": class_of(&bot_best),
                "agree1": !bot_best.is_empty() && bot_best == human,
                "rank": human_rank, "n_actions": actions.len(),
                "in_set": human_rank.is_some(),
                "revealed": revealed_moves,
                "own_prov": provenance[active_slot],
                "imputed_pick": imputed_pick,
                "belief": agent.belief_info(),
                "iters": iters,
                "top1": top1,
                "switch_mass": switch_mass as f64 / total_visits as f64,
                "self_hp": hp_pct(d.side, active_slot),
                "self_status": battle.sides[d.side].roster[active_slot].status.as_str(),
                "self_boost": boost_sum(d.side, active_slot),
                "self_vols": vols(d.side, active_slot),
                "foe_hp": foe_slot.map(|s| hp_pct(foe_side, s)),
                "foe_status": foe_slot.map(|s| battle.sides[foe_side].roster[s].status.as_str()),
                "foe_boost": foe_slot.map(|s| boost_sum(foe_side, s)),
                "foe_vols": foe_slot.map(|s| vols(foe_side, s)),
            })
            .to_string(),
        );
    }
    BattleReport { lines_out: out }
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
    let range = arg_s(&args, "--battles", "0-569");
    let iters = arg(&args, "--iters", 3000) as u32;
    let seed = arg(&args, "--seed", 1) as u64;
    let threads = arg(
        &args,
        "--threads",
        std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(4),
    );
    let out_path = arg_s(&args, "--out", "tmp/human-agreement.jsonl");
    // M17 tail: override the Spikes eval weight for this arm. Without it the
    // harness silently takes EvalWeights::default(), so an A/B over an eval
    // feature cannot be expressed at all — which is how the M16c null result
    // ended up being the only switching evidence on record.
    let spikes: Option<f64> = args
        .iter()
        .position(|a| a == "--spikes")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok());
    let mut agent_cfg = cfg();
    if let Some(w) = spikes {
        if let nc2000_bot::mcts::Playout::Heavy { weights, .. } = &mut agent_cfg.playout {
            weights.spikes = w;
        }
        eprintln!("eval override: spikes = {w}");
    }

    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("569").parse().unwrap_or(569);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool = nc2000_bot::preview::load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    eprintln!(
        "battles {} (index {lo}-{hi})  iters {iters}  threads {threads}  sources: {} species",
        files.len(),
        src.by_species.len()
    );

    let reports = Mutex::new(Vec::with_capacity(files.len()));
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (battle_idx, path) = &files[j];
                let report =
                    process_battle(&dex, &src, &pool, path, *battle_idx, iters, seed, &agent_cfg);
                reports.lock().unwrap().push((*battle_idx, report));
                let completed = done.fetch_add(1, Ordering::Relaxed) + 1;
                if completed.is_multiple_of(10) {
                    eprintln!("  {completed}/{} battles", files.len());
                }
            });
        }
    });

    let mut reports = reports.into_inner().unwrap();
    reports.sort_by_key(|(battle_idx, _)| *battle_idx);
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    for (_, report) in reports {
        for line in report.lines_out {
            writeln!(out, "{line}").unwrap();
        }
    }
    eprintln!("done -> {out_path}");
}
