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
//!    (`-fail` / `-immune` / `-miss` / a Substitute `[block]` / `cant`) ON A
//!    LINE THAT IS OURS, and shows no effect landing (`-status`, `-start`,
//!    `-boost`, `-unboost`, `-sidestart`, `-heal`, or damage to the target)
//!    anywhere in the window the move owns — INCLUDING inside a move it
//!    called. See `read_outcome` for what those two capitals cost before
//!    they were there, and `crates/bot/tests/mask_sleep_talk.rs` for the two
//!    cases that measured it.
//!
//! Anything the engine does not agree with is printed in full. Zero
//! disagreements is the ship bar; the firing histogram sizes the fix.
//!
//! Because the classification is itself a ship gate, every replay is scored
//! TWICE — by the current reader and by `read_outcome_legacy`, the one it
//! replaced — and every row on which the two differ is printed with the
//! protocol line that decided it, grouped by rule. A confirmation count that
//! falls is a claim that needs its own evidence, not a free win.
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
use nc2000_engine::dex::{Dex, MoveId};
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// PS `toID`: the key form of a display name ("Sleep Talk" -> "sleeptalk").
fn to_id(name: &str) -> String {
    name.chars().filter(char::is_ascii_alphanumeric).collect::<String>().to_lowercase()
}

/// Which mon a move can touch: the foe for foe-directed targets, the user
/// otherwise. (Was inline in `main`; it is now needed per NESTED call too.)
fn move_subject(dex: &Dex, id: MoveId, side: usize) -> String {
    let tgt = dex.move_static(id).target;
    if matches!(tgt, "normal" | "allAdjacentFoes" | "allAdjacent") {
        format!("p{}a", 2 - side)
    } else {
        format!("p{}a", side + 1)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Verdict {
    /// the engine logged this move failing, on a line that is ours
    Confirmed,
    /// nothing landed and nothing said so out loud
    Silent,
    /// something landed: the rule refused a move that WORKS
    Disagreed,
}

fn verdict_of(marker: &Option<String>, effect: &Option<String>) -> Verdict {
    if effect.is_some() {
        Verdict::Disagreed
    } else if marker.is_some() {
        Verdict::Confirmed
    } else {
        Verdict::Silent
    }
}

/// Whose slot a protocol line names, relative to this move.
fn slot_role(line: &str, actor: &str, subject: &str) -> &'static str {
    let on = line.split('|').filter(|s| !s.is_empty()).nth(1).unwrap_or("");
    if on.starts_with(actor) {
        "actor"
    } else if on.starts_with(subject) {
        "subject"
    } else {
        "neither"
    }
}

/// Did the masked move do anything?  Returns (failure marker, effect line).
///
/// Read the protocol window the move itself owns: from its own `|move|` line
/// to the next `|move|` / `|upkeep|` / `|turn|`. Everything outside is the
/// foe's action or end-of-turn residual — poison ticking, a Substitute the
/// foe put up, a natural wake-up — none of which this move caused. Lines
/// carrying a `[from]` tag are indirect by construction and are skipped even
/// inside the window.
///
/// Two corrections over the reader this replaced, both pinned as measured
/// blind spots in `crates/bot/tests/mask_sleep_talk.rs`:
///
///   * a `|move|` line that the ACTOR emits carrying `[from] <this move>` is
///     the move this one CALLED (Sleep Talk / Metronome / Mirror Move), not
///     the next action. Breaking on it closed the window before the call had
///     done anything, so a working Sleep Talk could never be scored
///     DISAGREED — the census was structurally blind to false positives of
///     any rule whose move nests another. The window now follows the call,
///     and re-aims `subject` at the CALLED move's target so its damage is
///     visible.
///   * the failure markers were subject-blind, so an asleep or paralysed foe
///     emitting its own `|cant|` inside our window counted as the engine
///     confirming OUR move failed (245 rows of the 570-battle corpus, over
///     two rules). `cant` is the only marker the engine emits with no
///     `|move|` line in front of it, so it is the only one a foe can smuggle
///     into our window; it and `-miss` are ours only when they name the
///     actor. `-fail`, `-immune` and the Substitute `[block]` name the
///     move's TARGET and are left unfiltered — see the arms below for the
///     Spikes case that proves filtering them is wrong.
fn read_outcome(
    dex: &Dex,
    log: &[String],
    side: usize,
    masked: MoveId,
) -> (Option<String>, Option<String>) {
    let actor = format!("p{}a", side + 1);
    let masked_name = dex.move_static(masked).name.clone();
    let mut subject = move_subject(dex, masked, side);
    // Set while inside a call this move made: the `[from] move: X` tag that
    // marks the called move's own lines as ours rather than as residual.
    let mut nested: Option<String> = None;
    let start = log
        .iter()
        .position(|l| l.starts_with("|move|") && l[6..].starts_with(&actor));
    let Some(start) = start else {
        // The move never resolved (flinched, frozen, fainted first): no
        // evidence either way.
        return (None, None);
    };
    let mut marker: Option<String> = None;
    for line in log.iter().skip(start + 1) {
        let f: Vec<&str> = line.split('|').filter(|s| !s.is_empty()).collect();
        let Some(&tag) = f.first() else { continue };
        if tag == "move" {
            let ours = f.get(1).is_some_and(|on| on.starts_with(&actor));
            if ours && line.contains(&format!("[from] {masked_name}")) {
                if let Some(cid) = f.get(2).and_then(|n| dex.moves.id(&to_id(n))) {
                    subject = move_subject(dex, cid, side);
                    nested = Some(format!("move: {}", dex.move_static(cid).name));
                }
                continue;
            }
            break;
        }
        if tag == "upkeep" || tag == "turn" {
            break;
        }
        if line.contains("[from]") && !nested.as_deref().is_some_and(|n| line.contains(n)) {
            continue;
        }
        let on = f.get(1).copied().unwrap_or("");
        match tag {
            // `cant` is the ONE failure marker the engine emits without a
            // `|move|` line in front of it, which is exactly how a paralysed
            // or sleeping foe's `|cant|` lands inside our window and gets
            // counted as our move failing. It names the mon that could not
            // move, so it is ours only when it names the actor -- a foe
            // `|cant|...|flinch` is in fact evidence our move WORKED.
            // `-miss` names the attacker (`moveexec.rs:1827/1945`,
            // `conditions.rs:1083`, and Bide's own user at 1202) and is only
            // ever emitted inside the attacker's own execution, so the same
            // test is a guard rather than a correction.
            "cant" | "-miss" => {
                if on.starts_with(&actor) && marker.is_none() {
                    marker = Some(line.clone());
                }
            }
            // NOT slot-filtered, deliberately. These three name the move's
            // TARGET pokemon, and which mon that is depends on the move's
            // target kind, not on where its effect would land: a `foeSide`
            // move (Spikes) fails on the FOE's active (`moveexec.rs:2347`,
            // `-fail` on `t`) while `subject` for it is the user's own slot.
            // Filtering them to `actor`/`subject` silently downgraded all 292
            // "that side condition is already up" confirmations in the
            // 570-battle corpus to "silent" -- measured, not argued. Neither
            // can appear in our window unless the mon that owns it also
            // emitted a `|move|` line, which ends the window.
            "-fail" | "-immune" => {
                if marker.is_none() {
                    marker = Some(line.clone());
                }
            }
            "-activate" if line.contains("[block]") => {
                if marker.is_none() {
                    marker = Some(line.clone());
                }
            }
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
                if on.starts_with(&subject) || tag == "-sidestart" {
                    return (marker, Some(line.clone()));
                }
            }
            _ => {}
        }
    }
    (marker, None)
}

/// The reader as it stood before the two corrections above, kept so that one
/// run scores both and every row that moved can be attributed to a rule and
/// to the exact line that used to decide it. Deleting it costs the audit
/// nothing except the ability to say WHY a confirmation count fell.
fn read_outcome_legacy(
    log: &[String],
    actor: &str,
    subject: &str,
) -> (Option<String>, Option<String>) {
    let start = log
        .iter()
        .position(|l| l.starts_with("|move|") && l[6..].starts_with(actor));
    let Some(start) = start else {
        return (None, None);
    };
    let mut marker: Option<String> = None;
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
            "-fail" | "-immune" | "-miss" | "cant" => {
                if marker.is_none() {
                    marker = Some(line.clone());
                }
            }
            "-activate" if line.contains("[block]") => {
                if marker.is_none() {
                    marker = Some(line.clone());
                }
            }
            "-start" if line.contains("perish") => {}
            "-sideend" => {}
            "-weather" => {}
            "-heal" if line.contains("[silent]") => {}
            "-status" | "-start" | "-boost" | "-unboost" | "-heal" | "-sidestart" | "-damage" => {
                let on = f.get(1).copied().unwrap_or("");
                if on.starts_with(subject) || tag == "-sidestart" {
                    return (marker, Some(line.clone()));
                }
            }
            _ => {}
        }
    }
    (marker, None)
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
    // Same replay, scored twice: once by the fixed reader and once by the one
    // it replaces. The gate runs off the fixed column; the legacy column is
    // what makes every moved row attributable to a rule and to the line that
    // used to decide it.
    let mut by_verdict: BTreeMap<(&'static str, Verdict), usize> = BTreeMap::new();
    let mut legacy_by_verdict: BTreeMap<(&'static str, Verdict), usize> = BTreeMap::new();
    let mut moved: BTreeMap<String, usize> = BTreeMap::new();
    let mut moved_examples: Vec<String> = Vec::new();
    let (mut legacy_agreed, mut legacy_unproven, mut legacy_disagreed) = (0usize, 0usize, 0usize);

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
                let subject = move_subject(&dex, id, d.side);
                let mine = SearchChoice::Move(id).to_input(&dex);
                if play.choose(&dex, d.side, &mine).is_err() {
                    continue;
                }
                if play.choose(&dex, 1 - d.side, &other.to_input(&dex)).is_err() {
                    continue;
                }
                let log: Vec<String> = play.log.clone();
                checked += 1;
                let (marker, effect) = read_outcome(&dex, &log, d.side, id);
                let (lmarker, leffect) = read_outcome_legacy(&log, &actor, &subject);
                let v = verdict_of(&marker, &effect);
                let lv = verdict_of(&lmarker, &leffect);
                *by_verdict.entry((why, v)).or_default() += 1;
                *legacy_by_verdict.entry((why, lv)).or_default() += 1;
                match v {
                    Verdict::Disagreed => {
                        let line = effect.clone().unwrap_or_default();
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
                    Verdict::Confirmed => agreed += 1,
                    // No effect landed and no explicit failure marker — the
                    // engine agrees in substance but silently, so it neither
                    // confirms nor contradicts the rule.
                    Verdict::Silent => unproven += 1,
                }
                match lv {
                    Verdict::Disagreed => legacy_disagreed += 1,
                    Verdict::Confirmed => legacy_agreed += 1,
                    Verdict::Silent => legacy_unproven += 1,
                }
                if v != lv {
                    let tag_of = |l: &str| -> String {
                        l.split('|').filter(|x| !x.is_empty()).next().unwrap_or("").to_string()
                    };
                    let (reason, evidence) = match (lv, v) {
                        (_, Verdict::Disagreed) => {
                            let l = effect.clone().unwrap_or_default();
                            (
                                format!(
                                    "the fixed window reaches an effect `{}` on the {} slot",
                                    tag_of(&l),
                                    slot_role(&l, &actor, &subject)
                                ),
                                l,
                            )
                        }
                        (Verdict::Confirmed, _) => {
                            let l = lmarker.clone().unwrap_or_default();
                            (
                                format!(
                                    "legacy credited `{}` naming the {} slot",
                                    tag_of(&l),
                                    slot_role(&l, &actor, &subject)
                                ),
                                l,
                            )
                        }
                        (_, Verdict::Confirmed) => {
                            let l = marker.clone().unwrap_or_default();
                            (
                                format!(
                                    "the fixed window reaches a marker `{}` on the {} slot",
                                    tag_of(&l),
                                    slot_role(&l, &actor, &subject)
                                ),
                                l,
                            )
                        }
                        _ => (String::from("?"), String::new()),
                    };
                    *moved.entry(format!("{lv:?} -> {v:?}  {reason}   [{why}]")).or_default() += 1;
                    if moved_examples.len() < show {
                        moved_examples.push(format!(
                            "  b{bi} T{} side {} `{}` ({why}) {lv:?} -> {v:?}: {evidence}",
                            d.turn,
                            d.side,
                            dex.moves.key(id)
                        ));
                    }
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
    println!("  engine confirmed the failure: {agreed}   (legacy reader: {legacy_agreed})");
    println!(
        "  no effect, no explicit marker (silent agreement): {unproven}   (legacy reader: {legacy_unproven})"
    );
    println!("  ENGINE DISAGREED: {disagreed}   (legacy reader: {legacy_disagreed})");
    if let Some(_) = recorded.as_ref() {
        println!(
            "\nagainst {agreement}: {rows_changed} of {rows_matched} recorded raw-visit argmaxes are refused ({:.2}%)",
            100.0 * rows_changed as f64 / rows_matched.max(1) as f64
        );
        for (why, n) in changed_by_rule.iter() {
            println!("  {n:6}  {why}");
        }
    }
    println!(
        "\nby rule: refused / replayed / engine-confirmed / silent / DISAGREED  \
         (legacy reader's confirmed+silent in brackets)"
    );
    for (why, n) in by_rule.iter() {
        let get = |m: &BTreeMap<(&'static str, Verdict), usize>, v: Verdict| {
            m.get(&(*why, v)).copied().unwrap_or(0)
        };
        let ok = get(&by_verdict, Verdict::Confirmed);
        let sil = get(&by_verdict, Verdict::Silent);
        let bad = get(&by_verdict, Verdict::Disagreed);
        let lok = get(&legacy_by_verdict, Verdict::Confirmed);
        let lsil = get(&legacy_by_verdict, Verdict::Silent);
        println!(
            "  {n:6}  {:6}  {ok:6}  {sil:6}  {bad:5}   [{lok:6} {lsil:6}]  {why}",
            ok + sil + bad
        );
    }
    let moved_rows: usize = moved.values().sum();
    println!("\nrows whose verdict the reader fix moved: {moved_rows} of {checked}");
    for (k, n) in moved.iter() {
        println!("  {n:6}  {k}");
    }
    if !moved_examples.is_empty() {
        println!("\nmoved rows (first {show}):");
        for line in &moved_examples {
            println!("{line}");
        }
    }
    if !disagreements.is_empty() {
        println!("\ndisagreements (first {show}):");
        for line in &disagreements {
            println!("{line}");
        }
    }
    std::process::exit(if disagreed > 0 { 1 } else { 0 });
}
