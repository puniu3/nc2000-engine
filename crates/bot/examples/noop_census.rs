//! Guaranteed-fail pruning: how often it fires on real human positions, and
//! — the part that matters — whether it is ever WRONG.
//!
//! `smmcts::certain_noop` removes actions from the root argmax. A rule that
//! hides a *useful* move is a strength bug that no agreement or duel number
//! would attribute correctly, so every rule has to be checked against the
//! engine rather than against its author's reading of the engine.
//!
//! Method, per corpus decision point (positions come from the same importer
//! `human_agreement` drives, so these are human positions, not self-play):
//!
//! 1. list the acting side's legal actions and ask the mask which it refuses;
//! 2. for each refused MOVE, actually play it — the mask's own turn, the foe
//!    on its first legal action, logging on — and read the protocol back;
//! 3. classify: the engine agrees if the log shows the move failing
//!    (`-fail` / `-immune` / `-miss` / a Substitute `[block]` / `cant`) and
//!    shows no effect landing (`-status`, `-start`, `-boost`, `-unboost`,
//!    `-sidestart`, `-heal`, or damage to the target).
//!
//! Anything the engine does not agree with is printed in full. Zero
//! disagreements is the ship bar; the firing histogram sizes the fix.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example noop_census -- \
//!     [--corpus tmp/corpus-spectator] [--battles 0-99] [--seed 1] [--show N]

use std::collections::BTreeMap;

use nc2000_bot::corpus::{
    corpus_files, extract_decisions, load_battle, load_sources, reconstruct,
};
use nc2000_bot::smmcts::dominated_actions;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Did the masked move do anything?
///
/// Read only the protocol window the move itself owns: from its own `|move|`
/// line to the next `|move|` / `|upkeep|` / `|turn|`. Everything outside is
/// the foe's action or end-of-turn residual — poison ticking, a Substitute
/// the foe put up, a natural wake-up — none of which this move caused. Lines
/// carrying a `[from]` tag are indirect by construction and are skipped even
/// inside the window.
fn read_outcome(log: &[String], actor: &str, subject: &str) -> (bool, Option<String>) {
    let start = log
        .iter()
        .position(|l| l.starts_with("|move|") && l[6..].starts_with(actor));
    let Some(start) = start else {
        // The move never resolved (flinched, frozen, fainted first): no
        // evidence either way.
        return (false, None);
    };
    let mut failed = false;
    for line in log.iter().skip(start + 1) {
        if line.starts_with("|move|") || line.starts_with("|upkeep") || line.starts_with("|turn|") {
            break;
        }
        if line.contains("[from]") {
            continue;
        }
        let f: Vec<&str> = line.split('|').filter(|s| !s.is_empty()).collect();
        let Some(&tag) = f.first() else { continue };
        match tag {
            "-fail" | "-immune" | "-miss" | "cant" => failed = true,
            "-activate" if line.contains("[block]") => failed = true,
            // Perish counters tick on their own schedule, and a screen can
            // expire, inside the window when the turn has no upkeep line.
            "-start" if line.contains("perish") => {}
            "-sideend" => {}
            // Weather ticks and the foe's own Rest can land in the window
            // when the turn logs no `|upkeep|`; an effect is only this move's
            // if it lands on the mon this move could touch.
            "-weather" => {}
            // `[silent]` heals are Leech Seed drain and Rest only (`dmg.rs`
            // heal: the two effect ids that log silently) — residual and
            // foe-move traffic, never something the masked move could have
            // caused. They were the entire disagreement list (5 of 5 before
            // this line existed), which is how a 4,000-refusal audit sat at
            // "5 disagreements" while every rule was in fact clean.
            "-heal" if line.contains("[silent]") => {}
            "-status" | "-start" | "-boost" | "-unboost" | "-heal" | "-sidestart" | "-damage" => {
                let on = f.get(1).copied().unwrap_or("");
                if on.starts_with(subject) || tag == "-sidestart" {
                    return (false, Some(line.clone()));
                }
            }
            _ => {}
        }
    }
    (failed, None)
}

/// `human_agreement` JSONL rows keyed by (battle, side, turn) -> chosen action.
fn load_agreement(path: &str) -> BTreeMap<(usize, usize, usize), String> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else { continue };
        let key = (
            v["battle"].as_u64().unwrap_or(0) as usize,
            v["side"].as_u64().unwrap_or(0) as usize,
            v["turn"].as_u64().unwrap_or(0) as usize,
        );
        if let Some(bot) = v["bot"].as_str() {
            out.insert(key, bot.to_string());
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-99");
    let seed: u64 = arg_s(&args, "--seed", "1").parse().unwrap();
    let show: usize = arg_s(&args, "--show", "12").parse().unwrap();
    // Optional: a `human_agreement` artifact whose recorded argmax was taken
    // over RAW visits. Every row whose chosen action the mask refuses is a
    // decision where the pruning changes the shipped bot's move.
    let agreement = arg_s(&args, "--agreement", "");
    let recorded = (!agreement.is_empty()).then(|| load_agreement(&agreement));
    let mut rows_matched = 0usize;
    let mut rows_changed = 0usize;
    let mut changed_by_rule: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("99").parse().unwrap_or(99);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");
    let files = corpus_files(std::path::Path::new(&corpus));
    assert!(!files.is_empty(), "no corpus battles under {corpus}");

    let mut decisions = 0usize;
    let mut with_any = 0usize;
    let mut flagged = 0usize;
    let mut checked = 0usize;
    let mut agreed = 0usize;
    let mut unproven = 0usize;
    let mut by_rule: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut bad_rule: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut disagreements: Vec<String> = Vec::new();

    for (bi, path) in files.iter().enumerate() {
        if bi < lo || bi > hi {
            continue;
        }
        let cb = load_battle(path);
        for d in extract_decisions(&cb.lines) {
            let Some(mut battle) =
                reconstruct(&dex, &src, &pool_path, &cb.lines, &cb.evidence, &d, seed)
            else {
                continue;
            };
            decisions += 1;
            let dom = dominated_actions(&battle, &dex, d.side);
            if let Some(rec) = recorded.as_ref() {
                if let Some(bot) = rec.get(&(bi, d.side, d.turn as usize)) {
                    rows_matched += 1;
                    for (choice, why) in dom.iter() {
                        let name = match choice {
                            SearchChoice::Move(id) => {
                                format!("move {}", nc2000_bot::corpus::plain(dex.moves.key(*id)))
                            }
                            SearchChoice::Switch(pos) => {
                                let slot = battle.sides[d.side]
                                    .party
                                    .get(*pos as usize - 1)
                                    .copied()
                                    .unwrap_or(0);
                                let sp = battle.sides[d.side].roster[slot as usize].species;
                                format!("switch {}", dex.species.key(sp))
                            }
                            other => other.to_input(&dex),
                        };
                        if &name == bot {
                            rows_changed += 1;
                            *changed_by_rule.entry(why).or_default() += 1;
                            break;
                        }
                    }
                }
            }
            if dom.is_empty() {
                continue;
            }
            with_any += 1;
            flagged += dom.len();
            for (choice, why) in dom {
                *by_rule.entry(why).or_default() += 1;
                // A self-KO is a certain LOSS, not a no-op: it resolves
                // normally and there is nothing for this audit to check.
                if why.starts_with("self-KO") {
                    continue;
                }
                let SearchChoice::Move(id) = choice else { continue };
                // Only a real, resolvable turn can be replayed: both sides
                // must owe a normal move choice.
                let Some(other) = battle.legal_choices(&dex, 1 - d.side).first().copied() else {
                    continue;
                };
                let mut play: Battle = battle.clone();
                play.set_log_enabled(true);
                play.log.clear();
                let actor = format!("p{}a", d.side + 1);
                // Which mon this move could have touched: itself, or the foe.
                let tgt = dex.move_static(id).target;
                let subject = if matches!(tgt, "normal" | "allAdjacentFoes" | "allAdjacent") {
                    format!("p{}a", 2 - d.side)
                } else {
                    actor.clone()
                };
                let mine = SearchChoice::Move(id).to_input(&dex);
                if play.choose(&dex, d.side, &mine).is_err() {
                    continue;
                }
                if play.choose(&dex, 1 - d.side, &other.to_input(&dex)).is_err() {
                    continue;
                }
                let log: Vec<String> = play.log.clone();
                checked += 1;
                let (failed, effect) = read_outcome(&log, &actor, &subject);
                match effect {
                    Some(line) => {
                        *bad_rule.entry(why).or_default() += 1;
                        if disagreements.len() < show {
                            disagreements.push(format!(
                                "  b{bi} T{} side {} masked `{}` ({why}) but the engine logged: {line}",
                                d.turn,
                                d.side,
                                dex.moves.key(id)
                            ));
                        }
                    }
                    None if failed => agreed += 1,
                    // No effect landed and no explicit failure marker — the
                    // engine agrees in substance but silently, so it neither
                    // confirms nor contradicts the rule.
                    None => unproven += 1,
                }
            }
        }
    }

    let disagreed = checked - agreed - unproven;
    println!("corpus battles {}-{}  decisions {decisions}", lo, hi);
    println!(
        "  decisions with at least one refusable action: {with_any} ({:.1}%)",
        100.0 * with_any as f64 / decisions.max(1) as f64
    );
    println!("  actions refused: {flagged}   replayed through the engine: {checked}");
    println!("  engine confirmed the failure: {agreed}");
    println!("  no effect, no explicit marker (silent agreement): {unproven}");
    println!("  ENGINE DISAGREED: {disagreed}");
    if let Some(_) = recorded.as_ref() {
        println!(
            "\nagainst {agreement}: {rows_changed} of {rows_matched} recorded raw-visit argmaxes are refused ({:.2}%)",
            100.0 * rows_changed as f64 / rows_matched.max(1) as f64
        );
        for (why, n) in changed_by_rule.iter() {
            println!("  {n:6}  {why}");
        }
    }
    println!("\nby rule (refusals / of which the engine let through):");
    for (why, n) in by_rule.iter() {
        let bad = bad_rule.get(why).copied().unwrap_or(0);
        println!("  {n:6}  {bad:5}  {why}");
    }
    if !disagreements.is_empty() {
        println!("\ndisagreements (first {show}):");
        for line in &disagreements {
            println!("{line}");
        }
    }
    std::process::exit(if disagreed > 0 { 1 } else { 0 });
}
