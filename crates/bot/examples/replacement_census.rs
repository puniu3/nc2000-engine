//! The decision class no harness has ever looked at: which Pokemon to send
//! in after one faints.
//!
//! `corpus::extract_decisions` drops every post-faint replacement (its
//! `if !fainted[s]` guard), so `human_agreement`, `eval_calibration`,
//! `regret_mining`, `anchor_gate` and `noop_census` have all been measuring a
//! population with this class removed. That is exactly the class the
//! battle-4040 report turned on: a Perish Song Gengar's kit is 3/4 dead once
//! it is the last mon, and the only lever that decided whether it *would* be
//! last was a replacement choice on turn 6.
//!
//! This is a PROTOTYPE census, deliberately standalone: it re-derives the
//! decision points and synthesizes its own `forceSwitch` request rather than
//! touching `corpus.rs`, so no fingerprint-bound artifact (`m17e_artifact`,
//! `reconstruction_schema_fingerprint`, `anchor_gate`) moves while the
//! question is still "is there anything here".
//!
//! Reports, per replacement decision: how many live options there were, what
//! the human picked, and — with `--search` — what the bot's root policy says,
//! including the value gap between the two candidates, which is the number
//! that decides whether the ordering blind spot is even expressible.
//!
//! Usage:
//!   cargo run --release -p nc2000-bot --example replacement_census -- \
//!     [--corpus tmp/corpus-spectator] [--battles 0-569] [--search] \
//!     [--iters 30000] [--seed 1] [--show N]

use std::collections::BTreeMap;

use conformance::fixture::repo_root;
use conformance::load_dex;
use nc2000_bot::corpus::{
    cfg, corpus_files, details, fabricate_set, load_battle, load_sources, plain, SetSources,
};
use nc2000_bot::import::{ProtocolAgent, ProtocolTracker};
use nc2000_bot::preview::{load_meta_pool, MetaPool};
use nc2000_engine::battle::{PokemonSet, SearchChoice};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::{Battle, Status};

fn arg<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).map(String::as_str)
}

/// Moves whose value is bench-dependent. HARD = the move itself fails when
/// the user has no bench (`moveexec.rs:499` for the Stadium 2 pair; Baton
/// Pass is gated on `can_switch`), so a kit made of them is dead weight on a
/// last mon. SOFT = the move still resolves but is a certain loss
/// (`smmcts::certain_self_loss`), and the mask already refuses it, so only
/// one slot dies rather than the kit.
const HARD: [&str; 3] = ["perishsong", "destinybond", "batonpass"];
const SOFT: [&str; 2] = ["explosion", "selfdestruct"];

fn bench_dependent(dex: &Dex, b: &Battle, side: usize, slot: u8) -> (Vec<String>, Vec<String>) {
    let p = &b.sides[side].roster[slot as usize];
    let keys: Vec<String> = p.move_slots.iter().map(|m| dex.moves.key(m.id).to_string()).collect();
    (
        keys.iter().filter(|k| HARD.contains(&k.as_str())).cloned().collect(),
        keys.iter().filter(|k| SOFT.contains(&k.as_str())).cloned().collect(),
    )
}

/// Every move each side's species is seen using anywhere in the log, keyed
/// by (side, species). Offline evidence about the ACTING side only — never
/// read for the opponent, whose sets stay belief-imputed.
fn revealed_by_side(
    dex: &Dex,
    lines: &[String],
    at: &mut Vec<(usize, usize, String, String)>,
) -> std::collections::HashMap<(usize, String), Vec<String>> {
    let mut out: std::collections::HashMap<(usize, String), Vec<String>> = Default::default();
    let mut active: [Option<String>; 2] = [None, None];
    for (li, ln) in lines.iter().enumerate() {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() < 4 {
            continue;
        }
        let Some(subject) = p.get(2) else { continue };
        let side = usize::from(subject.as_bytes().get(1) == Some(&b'2'));
        match p[1] {
            "switch" | "drag" | "replace" => {
                active[side] = Some(toid(p[3].split(',').next().unwrap_or("")));
            }
            "move" if !ln.contains("[from]") => {
                if let Some(sp) = active[side].clone() {
                    let key = plain(&toid(p[3]));
                    if dex.moves.id(&key).is_some() {
                        at.push((li, side, sp.clone(), key.clone()));
                        let e = out.entry((side, sp)).or_default();
                        if !e.contains(&key) {
                            e.push(key);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// Which side won, from `|player|` + `|win|`. `None` = tie or no result.
/// The repo's standing rule for corpus evidence is that agreement with the
/// human is not the signal — agreement with the WINNING side is — so a
/// behavioural rate is only interpretable next to this.
fn winner(lines: &[String]) -> Option<usize> {
    let mut names: [String; 2] = [String::new(), String::new()];
    for ln in lines {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() >= 4 && p[1] == "player" {
            let s = usize::from(p[2].as_bytes().get(1) == Some(&b'2'));
            names[s] = p[3].to_string();
        }
        if p.len() >= 3 && p[1] == "win" {
            return (0..2).find(|&s| names[s] == p[2]);
        }
    }
    None
}

/// One post-faint replacement: the side that owes it, the log index to cut
/// at, and the species the human actually sent in.
struct Replacement {
    side: usize,
    turn: u16,
    cut: usize,
    picked: String,
}

/// Post-faint replacements, read straight off the protocol. A `|faint|` puts
/// that side in debt; the next `|switch|` for the same side pays it. Leads
/// are excluded (nothing has fainted yet), as are `|drag|`s (a phaze is not
/// a choice).
fn replacements(lines: &[String]) -> Vec<Replacement> {
    let mut out = Vec::new();
    let mut owes = [false; 2];
    let mut turn = 0u16;
    for (i, ln) in lines.iter().enumerate() {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() < 2 {
            continue;
        }
        let side_of = |s: &str| usize::from(s.as_bytes().get(1) == Some(&b'2'));
        match p[1] {
            "turn" => turn = p[2].parse().unwrap_or(0),
            "faint" => owes[side_of(p[2])] = true,
            "switch" => {
                let s = side_of(p[2]);
                if owes[s] {
                    owes[s] = false;
                    out.push(Replacement {
                        side: s,
                        turn,
                        cut: i.saturating_sub(1),
                        picked: toid(p[3].split(',').next().unwrap_or("")),
                    });
                }
            }
            _ => {}
        }
    }
    out
}

/// The reconstruction, minus the move-request half: same tracker, same
/// fabrication, but a `forceSwitch` request. `synthesize` already routes that
/// to `RequestState::Switch` (import.rs:1519), so nothing downstream needs to
/// know this came from a different emitter.
#[allow(clippy::too_many_arguments)]
fn reconstruct_replacement(
    dex: &Dex,
    src: &SetSources,
    pool: MetaPool,
    lines: &[String],
    revealed: &std::collections::HashMap<(usize, String), Vec<String>>,
    r: &Replacement,
    seed: u64,
) -> Option<ProtocolAgent> {
    let mut tr = ProtocolTracker::new(r.side);
    for ln in &lines[..=r.cut] {
        tr.push_line(dex, ln);
    }
    let (mons, _active) = tr.snapshot(r.side);

    let mut own_sets_json = Vec::new();
    for m in &mons {
        // Own-set evidence, re-derived here rather than through
        // `corpus::SetEvidence` (whose accessors are private): every move
        // this side's mon is seen using ANYWHERE in the log. Legitimate for
        // the acting side — a live bot knows the team it submitted — and it
        // is what makes a Perish Song visible before it is first cast.
        // Sketch, Mimic and Metronome can put more than four distinct move
        // names on one mon across a whole log, so the union has to be clipped
        // to a legal four — already-used slots first, since those are the ones
        // the position actually depends on.
        let key = (r.side, dex.species.key(m.species).to_string());
        let used: Vec<String> =
            m.uses.iter().map(|(id, _)| plain(dex.moves.key(*id)).to_string()).collect();
        let mut future: Vec<String> = Vec::new();
        for k in revealed.get(&key).map(Vec::as_slice).unwrap_or(&[]) {
            if used.len() + future.len() >= 4 {
                break;
            }
            if !used.contains(k) && !future.contains(k) {
                future.push(k.clone());
            }
        }
        let (set, _prov) = fabricate_set(dex, src, m, &future, None, false);
        own_sets_json.push(set);
    }
    let own_sets: Vec<PokemonSet> =
        serde_json::from_value(serde_json::json!(own_sets_json.clone())).ok()?;
    let maxhps: Vec<i32> = Battle::from_fixture(dex, "1,2,3,4", &own_sets, &own_sets)
        .ok()?
        .sides[0]
        .roster
        .iter()
        .map(|p| p.maxhp)
        .collect();

    // The picked three, ACTIVE FIRST — that is PS's own ordering for
    // `side.pokemon`, and `reconstruct_context_inner` pushes `active_slot`
    // before anything else for the same reason. Building the list in roster
    // order instead makes the importer read party position 1 as the active,
    // which silently hides one of the two switch options (measured: 528 of
    // the 993 real decisions collapsed to one option).
    let active_slot = mons.iter().position(|m| m.active && m.fainted)?;
    let mut picked: Vec<usize> = vec![active_slot];
    for i in 0..mons.len() {
        if i != active_slot && mons[i].appeared {
            picked.push(i);
        }
    }
    for i in 0..mons.len() {
        if picked.len() >= 3 {
            break;
        }
        if i != active_slot && !mons[i].appeared {
            picked.push(i);
        }
    }
    if picked.len() < 3 {
        return None;
    }
    picked.truncate(3);

    let req_mons: Vec<serde_json::Value> = picked
        .iter()
        .map(|&i| {
            let m = &mons[i];
            let cond = if m.fainted {
                "0 fnt".to_string()
            } else {
                let hp = ((m.hp_frac * maxhps[i] as f64).round() as i32).clamp(1, maxhps[i]);
                let st = m.status.as_str();
                if st.is_empty() || m.status == Status::Fnt {
                    format!("{}/{}", hp, maxhps[i])
                } else {
                    format!("{}/{} {}", hp, maxhps[i], st)
                }
            };
            let nick = if m.name.is_empty() {
                dex.species.get(m.species).name.clone()
            } else {
                m.name.clone()
            };
            serde_json::json!({
                "ident": format!("p{}: {}", r.side + 1, nick),
                "details": details(dex, m),
                "condition": cond,
                "active": m.active,
                "item": own_sets_json[i]["item"].as_str().map(toid).unwrap_or_default(),
            })
        })
        .collect();
    let req = serde_json::json!({
        "forceSwitch": [true],
        "side": {"name": format!("p{}", r.side + 1), "id": format!("p{}", r.side + 1),
                 "pokemon": req_mons},
        "rqid": r.cut as u64,
    })
    .to_string();

    let mut agent = ProtocolAgent::new(dex, r.side, pool, cfg(), seed);
    agent.set_own_team(own_sets);
    for ln in &lines[..=r.cut] {
        agent.push_line(dex, ln);
    }
    agent.on_request(dex, &req).ok()?;
    Some(agent)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let corpus = arg(&args, "--corpus").unwrap_or("tmp/corpus-spectator").to_string();
    let range = arg(&args, "--battles").unwrap_or("0-569").to_string();
    let do_search = args.iter().any(|a| a == "--search");
    let only_contrasts = args.iter().any(|a| a == "--only-contrasts");
    let iters: u32 = arg(&args, "--iters").unwrap_or("30000").parse().unwrap();
    let seed: u64 = arg(&args, "--seed").unwrap_or("1").parse().unwrap();
    let show: usize = arg(&args, "--show").unwrap_or("25").parse().unwrap();

    let (lo, hi) = range.split_once('-').expect("--battles LO-HI");
    let (lo, hi): (usize, usize) = (lo.parse().unwrap(), hi.parse().unwrap());

    let dex = load_dex();
    let root = repo_root();
    let src = load_sources(&dex, &root);
    let pool = load_meta_pool(&root.join("data/meta-pool-v0/meta-pool.json"));
    let files = corpus_files(&std::path::Path::new(&corpus).to_path_buf());

    let mut total = 0usize;
    let mut real = 0usize;
    let mut rebuilt = 0usize;
    let mut refused = 0usize;
    let mut opts_hist: BTreeMap<usize, usize> = BTreeMap::new();
    let mut dep_cases = 0usize;
    let mut agree = 0usize;
    let mut searched = 0usize;
    let mut gaps: Vec<f64> = Vec::new();
    let mut shown = 0usize;
    // "Stranding": choosing the OTHER candidate, so the bench-dependent kit
    // stays in reserve and risks being the last mon. Counted only where the
    // choice is a clean binary (two candidates, exactly one dependent), and
    // reported for bot and human on the SAME decisions so the comparison is
    // paired.
    let (mut hard_n, mut hard_human_strand, mut hard_bot_strand) = (0usize, 0usize, 0usize);
    let (mut soft_n, mut soft_human_strand, mut soft_bot_strand) = (0usize, 0usize, 0usize);
    let mut hard_searched = 0usize;
    let mut soft_searched = 0usize;
    let mut hard_gaps: Vec<f64> = Vec::new();
    // Outcome of the games in which the human made each choice — the
    // directional signal, since the human rate alone cannot say who is right.
    let (mut hard_pub_n, mut hard_pub_strand) = (0usize, 0usize);
    let (mut hard_imp_n, mut hard_imp_strand) = (0usize, 0usize);
    let (mut hard_strand_games, mut hard_strand_wins) = (0usize, 0usize);
    let (mut hard_deploy_games, mut hard_deploy_wins) = (0usize, 0usize);

    for (bi, f) in files.iter().enumerate() {
        if bi < lo || bi > hi {
            continue;
        }
        let cb = load_battle(f);
        let won = winner(&cb.lines);
        let mut reveal_at: Vec<(usize, usize, String, String)> = Vec::new();
        let revealed = revealed_by_side(&dex, &cb.lines, &mut reveal_at);
        for r in replacements(&cb.lines) {
            total += 1;
            let Some(mut agent) = reconstruct_replacement(&dex, &src, pool.clone(), &cb.lines, &revealed, &r, seed)
            else {
                refused += 1;
                continue;
            };
            rebuilt += 1;
            let b = agent.battle().cloned().expect("battle");
            let acts = b.clone().legal_choices(&dex, r.side);
            *opts_hist.entry(acts.len()).or_default() += 1;
            if acts.len() < 2 {
                continue;
            }
            real += 1;

            // Which candidates carry a bench-dependent kit?
            let mut dep: Vec<(String, Vec<String>)> = Vec::new();
            let mut cand: Vec<(String, bool, bool)> = Vec::new(); // species, hard, soft
            for &c in &acts {
                if let SearchChoice::Switch(pos) = c {
                    let slot = b.sides[r.side].party[pos as usize - 1];
                    let (hard, soft) = bench_dependent(&dex, &b, r.side, slot);
                    let sp = dex.species.key(b.sides[r.side].roster[slot as usize].species).to_string();
                    cand.push((sp.clone(), !hard.is_empty(), !soft.is_empty()));
                    let mut all = hard.clone();
                    all.extend(soft);
                    if !all.is_empty() {
                        dep.push((sp, all));
                    }
                }
            }
            if !dep.is_empty() {
                dep_cases += 1;
            }
            // The clean contrast: exactly one candidate is bench-dependent,
            // so "who gets stranded" is a single binary choice. Sending in
            // the OTHER one strands the dependent kit for later.
            let hard_one: Option<&(String, bool, bool)> =
                if cand.iter().filter(|c| c.1).count() == 1 && cand.len() == 2 {
                    cand.iter().find(|c| c.1)
                } else {
                    None
                };
            let soft_one: Option<&(String, bool, bool)> =
                if cand.iter().filter(|c| c.2 && !c.1).count() == 1 && cand.len() == 2 {
                    cand.iter().find(|c| c.2 && !c.1)
                } else {
                    None
                };
            if let Some(d) = hard_one {
                // Was the dead kit PUBLIC at this point? A revealed-only
                // labelling can only see a mon that has already been on the
                // field, which conditions the sample on deployment — the
                // exact bias that would manufacture "humans deploy them".
                let public = reveal_at.iter().any(|(li, s2, sp, mv)| {
                    *li <= r.cut && *s2 == r.side && *sp == d.0 && HARD.contains(&mv.as_str())
                });
                hard_n += 1;
                let stranded = r.picked != d.0;
                if public {
                    hard_pub_n += 1;
                    if stranded { hard_pub_strand += 1; }
                } else {
                    hard_imp_n += 1;
                    if stranded { hard_imp_strand += 1; }
                }
                if stranded {
                    hard_human_strand += 1;
                }
                if let Some(w) = won {
                    let win = w == r.side;
                    if stranded {
                        hard_strand_games += 1;
                        hard_strand_wins += usize::from(win);
                    } else {
                        hard_deploy_games += 1;
                        hard_deploy_wins += usize::from(win);
                    }
                }
            }
            if let Some(d) = soft_one {
                soft_n += 1;
                if r.picked != d.0 {
                    soft_human_strand += 1;
                }
            }

            // `--only-contrasts` searches only the clean binary contrasts —
            // the 97 HARD + 204 SOFT rows the ordering question turns on —
            // instead of all 993. Same numbers for the stranding rates at a
            // twentieth of the wall clock.
            if !do_search || (only_contrasts && hard_one.is_none() && soft_one.is_none()) {
                continue;
            }
            agent.step(&dex, iters).expect("search");
            let policy: serde_json::Value = serde_json::from_str(&agent.root_policy(&dex)).unwrap();
            let rows = policy["actions"].as_array().cloned().unwrap_or_default();
            if rows.len() < 2 {
                continue;
            }
            searched += 1;
            let gap = rows[0]["mean"].as_f64().unwrap_or(0.0) - rows[1]["mean"].as_f64().unwrap_or(0.0);
            gaps.push(gap.abs());
            // Which species did the bot's best() land on?
            let best = agent.best(&dex).unwrap_or_default();
            let bot_species = best
                .strip_prefix("switch ")
                .and_then(|p| p.parse::<usize>().ok())
                .and_then(|pos| b.sides[r.side].party.get(pos - 1).copied())
                .map(|slot| dex.species.key(b.sides[r.side].roster[slot as usize].species).to_string())
                .unwrap_or_default();
            if bot_species == r.picked {
                agree += 1;
            }
            if let Some(d) = hard_one {
                hard_searched += 1;
                hard_gaps.push(gap.abs());
                if bot_species != d.0 {
                    hard_bot_strand += 1;
                }
            }
            if let Some(d) = soft_one {
                soft_searched += 1;
                if bot_species != d.0 {
                    soft_bot_strand += 1;
                }
            }
            if shown < show && !dep.is_empty() {
                shown += 1;
                let deps: Vec<String> =
                    dep.iter().map(|(s, m)| format!("{s}:{}", m.join("+"))).collect();
                println!(
                    "  b{bi} t{:<3} side {} human={:<12} bot={:<12} gap={:.4}  bench-dep candidates: {}",
                    r.turn, r.side, r.picked, bot_species, gap.abs(), deps.join(" ")
                );
            }
        }
    }

    println!();
    println!("corpus battles {lo}-{hi}");
    println!("  post-faint replacements found      : {total}");
    println!("  reconstructed                      : {rebuilt}   (refused {refused})");
    println!("  REAL decisions (>=2 live options)  : {real}");
    println!("  ...with a bench-dependent candidate: {dep_cases}");
    println!("  option-count histogram             : {opts_hist:?}");
    if do_search {
        let mut g = gaps.clone();
        g.sort_by(f64::total_cmp);
        let med = if g.is_empty() { 0.0 } else { g[g.len() / 2] };
        println!("  searched                           : {searched} at {iters} iters");
        println!("  top-1 agreement with the human     : {agree}/{searched} = {:.1}%",
                 100.0 * agree as f64 / searched.max(1) as f64);
        println!("  |mean gap| between top two         : median {med:.4}, \
                  under 0.01 on {:.0}%",
                 100.0 * g.iter().filter(|&&x| x < 0.01).count() as f64 / g.len().max(1) as f64);
        let mut hg = hard_gaps.clone();
        hg.sort_by(f64::total_cmp);
        if !hg.is_empty() {
            println!("  |mean gap| on HARD contrasts       : median {:.4}", hg[hg.len() / 2]);
        }
    }
    println!();
    println!("  clean binary contrasts (2 candidates, exactly one bench-dependent):");
    println!("    HARD (perish/destiny/batonpass): n={hard_n}   human stranded it {hard_human_strand} ({:.0}%)   \
              bot stranded it {hard_bot_strand}/{hard_searched} ({:.0}%)",
             100.0 * hard_human_strand as f64 / hard_n.max(1) as f64,
             100.0 * hard_bot_strand as f64 / hard_searched.max(1) as f64);
    println!("      outcome for the side that made the call: stranded it -> {hard_strand_wins}/{hard_strand_games} won ({:.0}%),  \
              deployed it -> {hard_deploy_wins}/{hard_deploy_games} won ({:.0}%)",
             100.0 * hard_strand_wins as f64 / hard_strand_games.max(1) as f64,
             100.0 * hard_deploy_wins as f64 / hard_deploy_games.max(1) as f64);
    println!("      label already PUBLIC at the decision: n={hard_pub_n}, human stranded {hard_pub_strand} ({:.0}%)   \
              label IMPUTED from the pool: n={hard_imp_n}, human stranded {hard_imp_strand} ({:.0}%)",
             100.0 * hard_pub_strand as f64 / hard_pub_n.max(1) as f64,
             100.0 * hard_imp_strand as f64 / hard_imp_n.max(1) as f64);
    println!("    SOFT (explosion/selfdestruct)  : n={soft_n}   human stranded it {soft_human_strand} ({:.0}%)   \
              bot stranded it {soft_bot_strand}/{soft_searched} ({:.0}%)",
             100.0 * soft_human_strand as f64 / soft_n.max(1) as f64,
             100.0 * soft_bot_strand as f64 / soft_searched.max(1) as f64);
}
