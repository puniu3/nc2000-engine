#![recursion_limit = "512"]
//! Frequency census for the two decision classes battles 4069/4070 turned up.
//!
//! **Class A — a provably dead move aimed at the mon actually in front, which
//! the mask deliberately does not refuse.** Every foe-reading rule in
//! `smmcts::noop_reason` is gated on `foe_can_switch` (smmcts.rs:756): a foe
//! that can leave makes the move a legitimate switch read, so the proof is
//! dropped. Battle 4069 played `return` and then `bodyslam` into a Ghost with
//! the mask silent on both. Nobody has ever measured what that gate costs.
//!
//! The counterfactual mask is obtained WITHOUT touching shipped code: clone
//! the position, set the foe active's `trapped` flag, and ask
//! `dominated_actions` again. `foe_can_switch` reports false for a trapped
//! foe, so every foe-reading rule fires against the mon in front; nothing
//! else in the mask reads `trapped` (the phaze rule calls the engine's own
//! `can_switch`, which does not), so the difference between the two masks is
//! exactly the set the gate suppresses.
//!
//! **Class B — facing a Mean Look + Perish Song trapper while still able to
//! leave.** Battle 4069 turn 26: Snorlax vs Misdreavus, switching to the
//! Whirlwind Zapdos measured +0.85. Counted at three evidence levels —
//! `truth` (the foe uses both moves somewhere in the log), `known` (both
//! already revealed at the decision) and `belief` (the reconstructed foe set
//! the search actually sees).
//!
//! Both classes are read off the same reconstruction every other corpus
//! instrument uses, so the denominators are comparable with `noop_census`
//! and `human_agreement`.
//!
//! Usage:
//!   perish_switch_census [--corpus tmp/corpus-spectator] [--battles 0-569]
//!     [--seed 1] [--threads 12] [--out rows.jsonl]
//!     [--search --iters 30000]     (searches ONLY class-A / class-B rows)

use std::collections::{BTreeMap, HashMap};
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::corpus::{
    corpus_files, load_battle, load_sources, plain, reconstruct_context_with_pool, HumanAction,
    SetSources,
};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_bot::smmcts::dominated_actions;
use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{toid, Category, Dex, SpeciesId};
use nc2000_engine::state::Battle;

fn arg_s(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Transcribed from `smmcts::foe_can_switch` (private). Switches resolve
/// before moves, so this is the flag every foe-reading mask rule is gated on.
fn foe_can_switch(b: &Battle, side: usize) -> bool {
    let opp = 1 - side;
    let Some(active) = b.active_id(opp) else { return false };
    if b.poke(active).trapped {
        return false;
    }
    let s = &b.sides[opp];
    s.party.iter().skip(1).any(|&slot| !s.roster[slot as usize].fainted)
}

/// Every move each side's species is seen using, with the log index of the
/// first use — so the same table answers both "did it ever" and "was it
/// already public at the cut".
type RevealTable = HashMap<(usize, String), Vec<(String, usize)>>;

fn reveals(dex: &Dex, lines: &[String]) -> RevealTable {
    let mut out: RevealTable = HashMap::new();
    let mut active: [Option<String>; 2] = [None, None];
    for (li, ln) in lines.iter().enumerate() {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() < 4 {
            continue;
        }
        let side = usize::from(p[2].as_bytes().get(1) == Some(&b'2'));
        match p[1] {
            "switch" | "drag" | "replace" => {
                active[side] = Some(toid(p[3].split(',').next().unwrap_or("")));
            }
            "move" if !ln.contains("[from]") => {
                if let Some(sp) = active[side].clone() {
                    let key = plain(&toid(p[3]));
                    if dex.moves.id(&key).is_some() {
                        let e = out.entry((side, sp)).or_default();
                        if !e.iter().any(|(k, _)| k == &key) {
                            e.push((key, li));
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

fn has_move(tbl: &RevealTable, side: usize, sp: &str, mv: &str, before: Option<usize>) -> bool {
    tbl.get(&(side, sp.to_string())).is_some_and(|v| {
        v.iter().any(|(k, li)| k == mv && before.map(|c| *li <= c).unwrap_or(true))
    })
}

fn winner(lines: &[String]) -> Option<usize> {
    let mut names: [String; 2] = [String::new(), String::new()];
    for ln in lines {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() >= 4 && p[1] == "player" {
            names[usize::from(p[2].as_bytes().get(1) == Some(&b'2'))] = p[3].to_string();
        }
        if p.len() >= 3 && p[1] == "win" {
            return (0..2).find(|&s| names[s] == p[2]);
        }
    }
    None
}

/// What each side did on each turn, straight off the protocol: a voluntary
/// `switch` (the thing that would vindicate a suppressed move), a `drag`
/// (a phaze — not a choice, but it still changes who is in front), a `move`,
/// or nothing. Post-faint replacement switches are attributed to the faint,
/// not to a choice, which is why `owes` is tracked.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TurnAct {
    Switch,
    Drag,
    Move,
    None,
}

fn turn_acts(lines: &[String]) -> HashMap<(u16, usize), (TurnAct, String)> {
    let mut out = HashMap::new();
    let mut turn = 0u16;
    let mut owes = [false; 2];
    let mut done = [false; 2];
    for ln in lines {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() < 2 {
            continue;
        }
        let sid = |s: &str| usize::from(s.as_bytes().get(1) == Some(&b'2'));
        match p[1] {
            "turn" => {
                turn = p[2].parse().unwrap_or(0);
                owes = [false; 2];
                done = [false; 2];
            }
            "faint" if p.len() >= 3 => owes[sid(p[2])] = true,
            "switch" | "drag" if p.len() >= 4 => {
                let s = sid(p[2]);
                let sp = toid(p[3].split(',').next().unwrap_or(""));
                let kind = if p[1] == "drag" {
                    TurnAct::Drag
                } else if owes[s] {
                    owes[s] = false;
                    TurnAct::None
                } else {
                    TurnAct::Switch
                };
                if !done[s] && kind != TurnAct::None {
                    done[s] = true;
                    out.insert((turn, s), (kind, sp));
                }
            }
            "move" if p.len() >= 3 && !ln.contains("[from]") => {
                let s = sid(p[2]);
                if !done[s] {
                    done[s] = true;
                    out.insert((turn, s), (TurnAct::Move, plain(&toid(p[3]))));
                }
            }
            _ => {}
        }
    }
    out
}

/// Did the server itself log the move as immune / doing nothing to the foe?
/// Ground truth for the detector: read the window the played move owns.
fn server_immune(lines: &[String], cut: usize, side: usize, mv: &str) -> Option<bool> {
    let actor = format!("p{}a", side + 1);
    let subject = format!("p{}a", 2 - side);
    let mut start = None;
    for (i, ln) in lines.iter().enumerate().skip(cut) {
        if ln.starts_with("|turn|") && i > cut {
            break;
        }
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() >= 4 && p[1] == "move" && p[2].starts_with(&actor) && !ln.contains("[from]") {
            if plain(&toid(p[3])) == mv {
                start = Some(i);
                break;
            }
        }
    }
    let start = start?;
    for ln in lines.iter().skip(start + 1) {
        if ln.starts_with("|move|") || ln.starts_with("|upkeep") || ln.starts_with("|turn|") {
            break;
        }
        let p: Vec<&str> = ln.split('|').collect();
        let Some(&tag) = p.get(1) else { continue };
        if tag == "-immune" && p.get(2).is_some_and(|s| s.starts_with(&subject)) {
            return Some(true);
        }
        if matches!(tag, "-damage" | "-status" | "-start" | "-unboost" | "-heal")
            && p.get(2).is_some_and(|s| s.starts_with(&subject))
            && !ln.contains("[from]")
        {
            return Some(false);
        }
    }
    Some(false)
}

const TRAP_MOVES: [&str; 2] = ["meanlook", "spiderweb"];
const PHAZE_MOVES: [&str; 2] = ["roar", "whirlwind"];

/// The mask reasons that read the MON IN FRONT — exactly the proofs
/// `foe_can_switch` suppresses. Everything else in `noop_reason` reads our
/// own side or a side condition and fires identically in both masks.
/// Transcribed from `smmcts::noop_reason`; a rule added there without an
/// entry here shows up as an unclassified suppression and is printed.
const FOE_READ: [&str; 11] = [
    "the target is immune to the move's type",
    "Dream Eater needs a sleeping, unsubstituted target",
    "the target already carries a major status",
    "a Substitute blocks foe-inflicted status",
    "the target's type cannot take that status",
    "the target already has that volatile",
    "a Substitute blocks confusion",
    "a Substitute rejects this move outright",
    "Leech Seed does not take on a Grass type",
    "Attract needs opposite, known genders",
    "a Substitute blocks foe-directed stat drops",
];

struct Row {
    json: String,
    class_a: bool,
    class_b: bool,
}

#[derive(Default)]
struct Counters {
    decisions: usize,
    reconstructed: usize,
    foe_switchable: usize,
    foe_switched: usize,
    battles: usize,
    /// per-battle presence
    battles_a: usize,
    battles_b: usize,
    /// every suppressed refusal over EVERY decision, by rule
    by_reason: BTreeMap<String, usize>,
}

#[allow(clippy::too_many_arguments)]
fn process_battle(
    dex: &Dex,
    src: &SetSources,
    pool: &MetaPool,
    path: &std::path::Path,
    bi: usize,
    seed: u64,
    search_iters: Option<u32>,
) -> (Vec<Row>, Counters) {
    let cb = load_battle(path);
    let tbl = reveals(dex, &cb.lines);
    let acts = turn_acts(&cb.lines);
    let won = winner(&cb.lines);
    let mut rows = Vec::new();
    let mut c = Counters { battles: 1, ..Default::default() };

    for (di, d) in cb.decisions.iter().enumerate() {
        c.decisions += 1;
        let s = seed
            ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A)
            ^ (di as u64).wrapping_mul(0xBF58_476D)
            ^ d.side as u64;
        let Some(mut rec) =
            reconstruct_context_with_pool(dex, src, pool.clone(), &cb.lines, &cb.evidence, d, s)
        else {
            continue;
        };
        let Some(b) = rec.agent.battle().cloned() else { continue };
        c.reconstructed += 1;
        let side = d.side;
        let opp = 1 - side;

        let fcs = foe_can_switch(&b, side);
        if fcs {
            c.foe_switchable += 1;
        }
        let foe_act = acts.get(&(d.turn, opp)).cloned().unwrap_or((TurnAct::None, String::new()));
        if foe_act.0 == TurnAct::Switch {
            c.foe_switched += 1;
        }

        // ---- masks -------------------------------------------------------
        let shipped = dominated_actions(&b, dex, side);
        let mut bt = b.clone();
        if let Some(id) = bt.active_id(opp) {
            bt.poke_mut(id).trapped = true;
        }
        let trapped_mask = dominated_actions(&bt, dex, side);
        let shipped_set: Vec<SearchChoice> = shipped.iter().map(|(ch, _)| *ch).collect();

        let name_of = |ch: SearchChoice| -> String {
            match ch {
                SearchChoice::Move(id) => format!("move {}", plain(dex.moves.key(id))),
                SearchChoice::Switch(pos) => {
                    let slot = b.sides[side].party.get(pos as usize - 1).copied().unwrap_or(0);
                    format!("switch {}", dex.species.key(b.sides[side].roster[slot as usize].species))
                }
                other => other.to_input(dex),
            }
        };

        // "dead vs the mon in front": everything the foe-reading rules prove,
        // whether or not the shipped mask is allowed to say so.
        let mut dead_attack: Vec<(SearchChoice, &'static str)> = Vec::new();
        let mut dead_status: Vec<(SearchChoice, &'static str)> = Vec::new();
        for (ch, why) in &trapped_mask {
            let SearchChoice::Move(id) = ch else { continue };
            if !FOE_READ.contains(why) {
                continue; // not a proof about the mon in front
            }
            if dex.move_static(*id).category == Category::Status {
                dead_status.push((*ch, why));
            } else {
                dead_attack.push((*ch, why));
            }
        }
        // suppressed = proven dead, and the shipped mask stays silent.
        let suppressed: Vec<(SearchChoice, &'static str)> = trapped_mask
            .iter()
            .filter(|(ch, _)| !shipped_set.contains(ch))
            .map(|(ch, why)| (*ch, *why))
            .collect();
        for (_, why) in &suppressed {
            *c.by_reason.entry((*why).to_string()).or_default() += 1;
        }
        let unclassified: Vec<&'static str> =
            suppressed.iter().map(|(_, w)| *w).filter(|w| !FOE_READ.contains(w)).collect();
        let supp_attack: Vec<(SearchChoice, &'static str)> = suppressed
            .iter()
            .filter(|(ch, _)| {
                matches!(ch, SearchChoice::Move(id) if dex.move_static(*id).category != Category::Status)
            })
            .copied()
            .collect();

        // How blank is the sheet? `n_attacks` = damaging legal moves;
        // `all_attacks_dead` is the Miltank-vs-Misdreavus shape, where the
        // whole attacking kit is a no-op against the mon in front.
        // `live_alt` = legal actions no rule proves dead — if that is zero the
        // mask MUST stay silent whatever the gate does.
        let legal = b.clone().legal_choices(dex, side);
        let n_actions = legal.len();
        let n_attacks = legal
            .iter()
            .filter(|c| match c {
                SearchChoice::Move(id) => dex.move_static(*id).category != Category::Status,
                _ => false,
            })
            .count();
        let all_attacks_dead = n_attacks > 0 && dead_attack.len() == n_attacks;
        let live_alt =
            legal.iter().filter(|c| !trapped_mask.iter().any(|(t, _)| t == *c)).count();

        let human = match &d.action {
            HumanAction::Move(k) => format!("move {k}"),
            HumanAction::Switch(sp) => format!("switch {sp}"),
        };
        let played_dead_attack = dead_attack.iter().any(|(ch, _)| name_of(*ch) == human);
        let played_dead_status = dead_status.iter().any(|(ch, _)| name_of(*ch) == human);
        let played_supp_attack = supp_attack.iter().any(|(ch, _)| name_of(*ch) == human);
        let played_supp_any = suppressed.iter().any(|(ch, _)| name_of(*ch) == human);

        // ---- class B ------------------------------------------------------
        let foe_sp = b
            .active_id(opp)
            .map(|id| dex.species.key(b.poke(id).species).to_string())
            .unwrap_or_default();
        let me_id = b.active_id(side);
        let me_sp = me_id
            .map(|id| dex.species.key(b.poke(id).species).to_string())
            .unwrap_or_default();
        let trap_truth = TRAP_MOVES.iter().any(|m| has_move(&tbl, opp, &foe_sp, m, None));
        let ps_truth = has_move(&tbl, opp, &foe_sp, "perishsong", None);
        let trap_known =
            TRAP_MOVES.iter().any(|m| has_move(&tbl, opp, &foe_sp, m, Some(d.cut)));
        let ps_known = has_move(&tbl, opp, &foe_sp, "perishsong", Some(d.cut));
        let foe_slots: Vec<String> = b
            .active_id(opp)
            .map(|id| {
                b.poke(id).move_slots.iter().map(|m| plain(dex.moves.key(m.id)).to_string()).collect()
            })
            .unwrap_or_default();
        let trap_belief = TRAP_MOVES.iter().any(|m| foe_slots.iter().any(|k| k == m));
        let ps_belief = foe_slots.iter().any(|k| k == "perishsong");

        let me_trapped = me_id.is_some_and(|id| b.poke(id).trapped);
        let can_leave = !me_trapped && b.can_switch(side as u8);
        // bench: alive, non-active party members and what they carry
        let mut bench: Vec<(String, bool, bool)> = Vec::new(); // (species, phazer_belief, phazer_truth)
        for &slot in b.sides[side].party.iter().skip(1) {
            let p = &b.sides[side].roster[slot as usize];
            if p.fainted {
                continue;
            }
            let sp = dex.species.key(p.species).to_string();
            let pb = p
                .move_slots
                .iter()
                .any(|m| PHAZE_MOVES.contains(&plain(dex.moves.key(m.id)).as_str()));
            let pt = PHAZE_MOVES.iter().any(|m| has_move(&tbl, side, &sp, m, None));
            bench.push((sp, pb, pt));
        }
        let bench_phazer_belief = bench.iter().filter(|(_, pb, _)| *pb).count();
        let bench_phazer_truth = bench.iter().filter(|(_, _, pt)| *pt).count();

        let class_b_truth = trap_truth && ps_truth && can_leave;
        let class_b_known = trap_known && ps_known && can_leave;
        let class_b_belief = trap_belief && ps_belief && can_leave;

        let class_a = !dead_attack.is_empty();
        let class_a_wide = class_a || !dead_status.is_empty();
        let class_b = class_b_truth || class_b_known || class_b_belief || (trap_truth && ps_truth);
        if !class_a_wide && !class_b {
            continue;
        }

        // ---- what the foe's switch actually brought in --------------------
        let mut still_dead_after_switch: Option<bool> = None;
        if foe_act.0 == TurnAct::Switch && !dead_attack.is_empty() {
            if let Some(sp_id) = dex.species.id(&foe_act.1) {
                still_dead_after_switch = Some(dead_attack.iter().all(|(ch, _)| {
                    let SearchChoice::Move(id) = ch else { return true };
                    immune_vs_species(dex, &b, side, *id, sp_id)
                }));
            }
        }
        let srv = if played_dead_attack {
            match &d.action {
                HumanAction::Move(k) => server_immune(&cb.lines, d.cut, side, k),
                _ => None,
            }
        } else {
            None
        };

        // ---- optional search ---------------------------------------------
        let mut bot_best = String::new();
        let mut bot_dead = false;
        let mut bot_switch = false;
        let mut bot_phazer = false;
        if let Some(iters) = search_iters {
            if (class_a || class_b) && rec.agent.step(dex, iters).is_ok() {
                if let Some(sr) = rec.agent.search() {
                    let actions: Vec<String> = sr.actions().iter().map(|&ch| name_of(ch)).collect();
                    let visits = sr.visits();
                    let dominated = sr.dominated();
                    let mut order: Vec<usize> = (0..actions.len()).collect();
                    order.sort_by(|&a, &z| {
                        let (da, dz) =
                            (dominated.get(a) == Some(&true), dominated.get(z) == Some(&true));
                        da.cmp(&dz).then_with(|| visits[z].cmp(&visits[a]))
                    });
                    if let Some(&i) = order.first() {
                        bot_best = actions[i].clone();
                    }
                    bot_dead = dead_attack.iter().any(|(ch, _)| name_of(*ch) == bot_best);
                    bot_switch = bot_best.starts_with("switch ");
                    if let Some(sp) = bot_best.strip_prefix("switch ") {
                        bot_phazer = bench.iter().any(|(b_sp, pb, pt)| b_sp == sp && (*pb || *pt));
                    }
                }
            }
        }

        let reasons: Vec<String> =
            suppressed.iter().map(|(ch, why)| format!("{} :: {why}", name_of(*ch))).collect();
        let dead_names: Vec<String> = dead_attack.iter().map(|(ch, _)| name_of(*ch)).collect();

        let json = serde_json::json!({
            "battle": bi, "side": side, "turn": d.turn, "cut": d.cut,
            "me": me_sp, "foe": foe_sp, "human": human, "won": won,
            "actor_won": won.map(|w| w == side),
            "foe_can_switch": fcs,
            "foe_act": format!("{:?}", foe_act.0), "foe_act_target": foe_act.1,
            // class A
            "class_a": class_a, "class_a_wide": class_a_wide, "class_b": class_b,
            "n_actions": n_actions, "n_attacks": n_attacks,
            "a_all_attacks_dead": all_attacks_dead, "a_live_alt": live_alt,
            "a_dead_attack": dead_names,
            "a_dead_attack_n": dead_attack.len(),
            "a_dead_status_n": dead_status.len(),
            "a_supp_attack_n": supp_attack.len(),
            "a_supp_any_n": suppressed.len(),
            "a_reasons": reasons,
            "a_unclassified": unclassified,
            "a_played_dead": played_dead_attack,
            "a_played_dead_status": played_dead_status,
            "a_dead_status": dead_status.iter().map(|(ch, _)| name_of(*ch)).collect::<Vec<_>>(),
            "a_played_supp_attack": played_supp_attack,
            "a_played_supp_any": played_supp_any,
            "a_server_immune": srv,
            "a_still_dead_after_switch": still_dead_after_switch,
            // class B
            "b_trap_truth": trap_truth, "b_ps_truth": ps_truth,
            "b_trap_known": trap_known, "b_ps_known": ps_known,
            "b_trap_belief": trap_belief, "b_ps_belief": ps_belief,
            "b_can_leave": can_leave, "b_me_trapped": me_trapped,
            "b_truth": class_b_truth, "b_known": class_b_known, "b_belief": class_b_belief,
            "b_bench": bench.iter().map(|(s, _, _)| s.clone()).collect::<Vec<_>>(),
            "b_bench_alive": bench.len(),
            "b_phazer_belief": bench_phazer_belief,
            "b_phazer_truth": bench_phazer_truth,
            // search
            "bot": bot_best, "bot_dead": bot_dead,
            "bot_switch": bot_switch, "bot_phazer": bot_phazer,
        })
        .to_string();
        rows.push(Row { json, class_a, class_b });
    }
    c.battles_a = usize::from(rows.iter().any(|r| r.class_a));
    c.battles_b = usize::from(rows.iter().any(|r| r.class_b));
    (rows, c)
}

/// Would `mv`, cast by `side`'s active, be type-immune against a bare species?
/// Ground is resolved by the Flying type exactly as the mask's rule does.
fn immune_vs_species(dex: &Dex, b: &Battle, side: usize, mv: nc2000_engine::dex::MoveId, sp: SpeciesId) -> bool {
    let ms = dex.move_static(mv);
    let key = dex.moves.key(mv);
    let mt = if key == "hiddenpower" {
        b.active_id(side).map(|id| b.poke(id).hp_type).unwrap_or(ms.move_type)
    } else {
        ms.move_type
    };
    let types = dex.species_types(sp);
    if mt == dex.known_types.ground {
        return (0..types.n as usize).any(|i| types.t[i] == dex.known_types.flying);
    }
    (0..types.n as usize).any(|i| dex.type_immune(mt, types.t[i]))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let corpus = arg_s(&args, "--corpus", "tmp/corpus-spectator");
    let range = arg_s(&args, "--battles", "0-569");
    let seed: u64 = arg_s(&args, "--seed", "1").parse().unwrap();
    let threads: usize = arg_s(&args, "--threads", "12").parse().unwrap();
    let out_path = arg_s(&args, "--out", "tmp/perish-switch-census.jsonl");
    let search_iters = args
        .iter()
        .any(|a| a == "--search")
        .then(|| arg_s(&args, "--iters", "30000").parse::<u32>().unwrap());

    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("569").parse().unwrap_or(569);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = conformance::fixture::repo_root();
    let src = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let files: Vec<(usize, std::path::PathBuf)> = corpus_files(&root.join(&corpus))
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i >= lo && *i <= hi)
        .collect();
    eprintln!(
        "battles {} (index {lo}-{hi}) seed {seed} threads {threads} search {:?}",
        files.len(),
        search_iters
    );

    let sink: Mutex<(Vec<(usize, Vec<Row>)>, Counters)> =
        Mutex::new((Vec::new(), Counters::default()));
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads.max(1) {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (bi, path) = &files[j];
                let (rows, c) = process_battle(&dex, &src, &pool, path, *bi, seed, search_iters);
                {
                    let mut g = sink.lock().unwrap();
                    g.1.decisions += c.decisions;
                    g.1.reconstructed += c.reconstructed;
                    g.1.foe_switchable += c.foe_switchable;
                    g.1.foe_switched += c.foe_switched;
                    g.1.battles += c.battles;
                    g.1.battles_a += c.battles_a;
                    g.1.battles_b += c.battles_b;
                    for (k, v) in &c.by_reason {
                        *g.1.by_reason.entry(k.clone()).or_default() += v;
                    }
                    g.0.push((*bi, rows));
                }
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(50) {
                    eprintln!("  {n}/{} battles", files.len());
                }
            });
        }
    });

    let (mut per_battle, c) = sink.into_inner().unwrap();
    per_battle.sort_by_key(|(bi, _)| *bi);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap());
    let mut n_a = 0usize;
    let mut n_b = 0usize;
    for (_, rows) in &per_battle {
        for r in rows {
            writeln!(f, "{}", r.json).unwrap();
            n_a += usize::from(r.class_a);
            n_b += usize::from(r.class_b);
        }
    }
    let summary = serde_json::json!({
        "battles": c.battles,
        "decisions": c.decisions,
        "reconstructed": c.reconstructed,
        "foe_switchable": c.foe_switchable,
        "foe_switched_that_turn": c.foe_switched,
        "rows_class_a": n_a,
        "rows_class_b": n_b,
        "battles_with_a": c.battles_a,
        "battles_with_b": c.battles_b,
        "suppressed_by_reason": c.by_reason,
    });
    eprintln!("{}", serde_json::to_string_pretty(&summary).unwrap());
    eprintln!("rows -> {out_path}");
}
