//! M16b — human-agreement measurement over the 570-battle spectator corpus.
//!
//! For every observable human decision point, drop the product bot
//! (ProtocolAgent = tracker → synthesize → BlindSearch, skuct config) into
//! the same position built from PUBLIC information only, and compare its
//! choice with what the human actually played.
//!
//! The acting side's sets are NOT in a spectator stream, so they are
//! fabricated from public facts + imputation (the feasibility scan sized
//! this: only 17.1% of move decisions have the acting active fully
//! revealed, and 43% of human picks are first uses):
//!   - per preview mon, the best-matching set from the community-rentals DB
//!     (same regulation) then the meta pool — max overlap with the mon's
//!     revealed moves, full containment preferred; revealed moves are
//!     ALWAYS retained, non-contained candidates only donate fills;
//!   - species with no candidate get revealed moves + top-BP learnset fills
//!     (typed Hidden Power only via a candidate or a revealed plain HP,
//!     which falls back to the Ice DV spread);
//!   - unappeared picks are imputed in preview order (pokemon_left must be
//!     right — the 694efb1 lesson);
//!   - eaten berries (|-enditem| [eat]) blank the fabricated item.
//! The opponent side runs the live belief machinery untouched.
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

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use nc2000_bot::import::{MonSnapshot, ProtocolAgent, ProtocolTracker};
use nc2000_bot::preview::load_meta_pool;
use nc2000_bot::smmcts::{RmConfig, SelRule};
use nc2000_engine::dex::{toid, Dex, MoveId, SpeciesId};
use nc2000_engine::state::{Battle, Status};

fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

fn plain(key: &str) -> String {
    if key.starts_with("hiddenpower") { "hiddenpower".into() } else { key.into() }
}

// ------------------------------------------------------------ set sources

struct SetSources {
    /// species id -> candidate set JSONs (rentals first, then pool).
    by_species: HashMap<SpeciesId, Vec<serde_json::Value>>,
    /// species key -> legal move keys.
    learnsets: HashMap<String, Vec<String>>,
    /// hidden-power type -> (atk DV, def DV).
    hp_dvs: HashMap<String, (i64, i64)>,
}

fn load_sources(dex: &Dex, root: &std::path::Path) -> SetSources {
    let mut by_species: HashMap<SpeciesId, Vec<serde_json::Value>> = HashMap::new();
    let mut add_sets = |sets: &[serde_json::Value]| {
        for s in sets {
            if let Some(sp) = s["species"].as_str() {
                if let Some(id) = dex.species.id(&toid(sp)) {
                    by_species.entry(id).or_default().push(s.clone());
                }
            }
        }
    };
    // rentals (same regulation) take precedence by insertion order
    let rentals: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/community-rentals-v0/teams.json")).unwrap(),
    )
    .unwrap();
    if let Some(teams) = rentals.as_array() {
        for t in teams {
            if let Some(sets) = t["sets"].as_array() {
                add_sets(sets);
            }
        }
    }
    let pool: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/meta-pool-v0/meta-pool.json")).unwrap(),
    )
    .unwrap();
    if let Some(teams) = pool["teams"].as_array() {
        for t in teams {
            if let Some(sets) = t["sets"].as_array() {
                add_sets(sets);
            }
        }
    }
    let ls: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap(),
    )
    .unwrap();
    let mut learnsets = HashMap::new();
    if let Some(sp) = ls["species"].as_object() {
        for (k, v) in sp {
            let moves: Vec<String> = v["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect())
                .unwrap_or_default();
            learnsets.insert(k.clone(), moves);
        }
    }
    let mut hp_dvs = HashMap::new();
    if let Some(o) = ls["hpDvs"].as_object() {
        for (k, v) in o {
            hp_dvs.insert(k.clone(), (v["atk"].as_i64().unwrap_or(15), v["def"].as_i64().unwrap_or(15)));
        }
    }
    SetSources { by_species, learnsets, hp_dvs }
}

// ------------------------------------------------------------ decisions

#[derive(Clone, Debug)]
enum HumanAction {
    Move(String),   // plain move key
    Switch(String), // species key
}

#[derive(Clone, Debug)]
struct Decision {
    side: usize,
    turn: u16,
    /// Line index to feed through (inclusive of the |turn|N line).
    cut: usize,
    action: HumanAction,
}

/// Extract observable voluntary decisions (mirror of the feasibility scan):
/// first non-[from] |move| or pre-faint |switch| per side per turn; |cant|,
/// replacement switches and |drag| are not observable decisions.
fn extract_decisions(lines: &[String]) -> Vec<Decision> {
    let mut out = Vec::new();
    let mut turn_open = false;
    let mut turn = 0u16;
    let mut cut = 0usize;
    let mut fainted = [false; 2];
    let mut decided = [false; 2];
    for (i, ln) in lines.iter().enumerate() {
        let p: Vec<&str> = ln.split('|').collect();
        if p.len() < 2 {
            continue;
        }
        let side_of = |s: &str| -> usize { if s.as_bytes().get(1) == Some(&b'2') { 1 } else { 0 } };
        match p[1] {
            "turn" => {
                turn_open = true;
                turn = p[2].parse().unwrap_or(0);
                cut = i;
                fainted = [false; 2];
                decided = [false; 2];
            }
            "faint" => fainted[side_of(p[2])] = true,
            "cant" => {
                let s = side_of(p[2]);
                if turn_open && !decided[s] {
                    decided[s] = true;
                }
            }
            "switch" => {
                let s = side_of(p[2]);
                if turn_open && !decided[s] {
                    if !fainted[s] {
                        let species = p[3].split(',').next().unwrap_or("").trim();
                        out.push(Decision {
                            side: s,
                            turn,
                            cut,
                            action: HumanAction::Switch(toid(species)),
                        });
                    }
                    decided[s] = true;
                }
            }
            "move" => {
                let s = side_of(p[2]);
                let from = ln.contains("[from]");
                if turn_open && !decided[s] && !from {
                    out.push(Decision {
                        side: s,
                        turn,
                        cut,
                        action: HumanAction::Move(plain(&toid(p[3]))),
                    });
                    decided[s] = true;
                }
            }
            _ => {}
        }
    }
    out
}

// ------------------------------------------------------------ fabrication

/// Build one mon's set JSON from public facts + imputation.
/// Returns (set, provenance) — provenance: "cand-full" (candidate contains
/// all revealed), "cand-fill" (candidate donates fills), "learnset-pad".
fn fabricate_set(
    dex: &Dex,
    src: &SetSources,
    m: &MonSnapshot,
    eaten: bool,
) -> (serde_json::Value, &'static str) {
    let sp_key = dex.species.key(m.species).to_string();
    let sp_name = dex.species.get(m.species).name.clone();
    let revealed: Vec<String> = m.uses.iter().map(|(id, _)| plain(dex.moves.key(*id))).collect();
    let nick = if m.name.is_empty() { sp_name.clone() } else { m.name.clone() };
    let gender = match format!("{:?}", m.gender).as_str() {
        "M" => "M",
        "F" => "F",
        _ => "",
    };

    // candidate scoring: normalized-move overlap with revealed
    let cands = src.by_species.get(&m.species).map(|v| v.as_slice()).unwrap_or(&[]);
    let norm_moves = |set: &serde_json::Value| -> Vec<String> {
        set["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| plain(&toid(s)))).collect())
            .unwrap_or_default()
    };
    let mut best: Option<(usize, bool, usize)> = None; // (overlap, full, idx)
    for (i, c) in cands.iter().enumerate() {
        let cm = norm_moves(c);
        let overlap = revealed.iter().filter(|r| cm.contains(r)).count();
        let full = overlap == revealed.len();
        let key = (overlap, full, usize::MAX - i); // earlier source wins ties
        if best.map_or(true, |(o, f, bi)| key > (o, f, usize::MAX - (usize::MAX - bi))) {
            best = Some((overlap, full, i));
        }
    }

    let mut set;
    let provenance;
    match best {
        Some((_, true, i)) if !revealed.is_empty() || !cands.is_empty() => {
            set = cands[i].clone();
            provenance = "cand-full";
        }
        Some((_, false, i)) => {
            // keep revealed, donate candidate fills
            set = cands[i].clone();
            let cm: Vec<String> = set["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let mut moves: Vec<String> = Vec::new();
            for r in &revealed {
                // display name via dex when possible
                let disp = dex
                    .moves
                    .id(r)
                    .map(|id| dex.moves.get(id).name.clone())
                    .unwrap_or_else(|| r.clone());
                moves.push(disp);
            }
            for c in cm {
                if moves.len() >= 4 {
                    break;
                }
                if !moves.iter().any(|x| plain(&toid(x)) == plain(&toid(&c))) {
                    moves.push(c);
                }
            }
            set["moves"] = serde_json::json!(moves);
            provenance = "cand-fill";
        }
        _ => {
            // learnset pad: revealed + top-BP legal moves, one per type
            let legal = src.learnsets.get(&sp_key).cloned().unwrap_or_default();
            let mut moves: Vec<String> = Vec::new();
            let mut ivs = serde_json::json!({});
            for r in &revealed {
                if r == "hiddenpower" {
                    let (a, d) = *src.hp_dvs.get("ice").unwrap_or(&(15, 13));
                    ivs = serde_json::json!({"atk": a * 2, "def": d * 2});
                    moves.push("Hidden Power".into());
                } else if let Some(id) = dex.moves.id(r) {
                    moves.push(dex.moves.get(id).name.clone());
                }
            }
            let mut scored: Vec<(u16, String, String)> = legal
                .iter()
                .filter_map(|k| {
                    let id = dex.moves.id(k)?;
                    let md = dex.moves.get(id);
                    if md.category == "Status" || k.starts_with("hiddenpower") {
                        return None;
                    }
                    Some((md.base_power, md.move_type.clone(), md.name.clone()))
                })
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0));
            let mut seen_types: Vec<String> = Vec::new();
            for (_, ty, name) in scored {
                if moves.len() >= 4 {
                    break;
                }
                if moves.iter().any(|x| plain(&toid(x)) == plain(&toid(&name))) {
                    continue;
                }
                if seen_types.contains(&ty) {
                    continue;
                }
                seen_types.push(ty.clone());
                moves.push(name);
            }
            set = serde_json::json!({
                "species": sp_name, "item": "Leftovers", "ability": "No Ability",
                "moves": moves, "nature": "Serious",
                "evs": {"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255},
                "ivs": ivs,
            });
            provenance = "learnset-pad";
        }
    }
    set["name"] = serde_json::json!(nick);
    set["level"] = serde_json::json!(m.level);
    set["gender"] = serde_json::json!(gender);
    if eaten {
        set["item"] = serde_json::json!("");
    }
    (set, provenance)
}

/// PS `details` string for a fabricated mon.
fn details(dex: &Dex, m: &MonSnapshot) -> String {
    let name = &dex.species.get(m.species).name;
    let mut d = format!("{}, L{}", name, m.level);
    match format!("{:?}", m.gender).as_str() {
        "M" => d.push_str(", M"),
        "F" => d.push_str(", F"),
        _ => {}
    }
    d
}

// ------------------------------------------------------------ per battle

struct BattleReport {
    lines_out: Vec<String>,
}

#[allow(clippy::too_many_arguments)]
fn process_battle(
    dex: &Dex,
    src: &SetSources,
    pool_path: &std::path::Path,
    battle_file: &std::path::Path,
    battle_idx: usize,
    iters: u32,
    base_seed: u64,
) -> BattleReport {
    let text = std::fs::read_to_string(battle_file).unwrap_or_default();
    let lines: Vec<String> = text
        .lines()
        .filter(|l| {
            let t = l.split('|').nth(1).unwrap_or("");
            !matches!(t, "j" | "l" | "t:" | "init" | "title" | "" )
        })
        .map(String::from)
        .collect();
    let decisions = extract_decisions(&lines);
    let mut out = Vec::new();

    // eaten berries per (side, nick)
    let mut eaten: Vec<(usize, String)> = Vec::new();
    for ln in &lines {
        if ln.contains("|-enditem|") && ln.contains("[eat]") {
            let p: Vec<&str> = ln.split('|').collect();
            if let Some(subj) = p.get(2) {
                let side = if subj.as_bytes().get(1) == Some(&b'2') { 1 } else { 0 };
                if let Some(nick) = subj.split(':').nth(1) {
                    eaten.push((side, nick.trim().to_string()));
                }
            }
        }
    }

    for (di, d) in decisions.iter().enumerate() {
        let seed = base_seed
            ^ (battle_idx as u64).wrapping_mul(0x9E37_79B9_7F4A)
            ^ (di as u64).wrapping_mul(0xBF58_476D)
            ^ d.side as u64;
        // tracker over the prefix for fabrication facts
        let mut tr = ProtocolTracker::new(d.side);
        for ln in &lines[..=d.cut] {
            tr.push_line(dex, ln);
        }
        let (mons, active_slot) = tr.snapshot(d.side);
        let Some(active_slot) = active_slot else { continue };
        if mons[active_slot].choiceless {
            out.push(
                serde_json::json!({"battle": battle_idx, "side": d.side, "turn": d.turn,
                    "skip": "choiceless"})
                .to_string(),
            );
            continue;
        }

        // ---- fabricate the own team (all 6, preview order)
        let mut own_sets_json = Vec::new();
        let mut provenance = Vec::new();
        for m in &mons {
            let ate = eaten.iter().any(|(s, n)| *s == d.side && !m.name.is_empty() && *n == m.name);
            let (set, prov) = fabricate_set(dex, src, m, ate);
            own_sets_json.push(set);
            provenance.push(prov);
        }
        let own_sets: Vec<nc2000_engine::battle::PokemonSet> =
            match serde_json::from_value(serde_json::json!(own_sets_json.clone())) {
                Ok(s) => s,
                Err(e) => {
                    out.push(
                        serde_json::json!({"battle": battle_idx, "side": d.side, "turn": d.turn,
                            "skip": "set-parse", "err": e.to_string()})
                        .to_string(),
                    );
                    continue;
                }
            };
        // maxhp probe for absolute HP in the fabricated request
        let maxhps: Vec<i32> = match Battle::from_fixture(dex, "1,2,3,4", &own_sets, &own_sets) {
            Ok(b) => b.sides[0].roster.iter().map(|p| p.maxhp).collect(),
            Err(e) => {
                out.push(
                    serde_json::json!({"battle": battle_idx, "side": d.side, "turn": d.turn,
                        "skip": "probe", "err": format!("{e:?}")})
                    .to_string(),
                );
                continue;
            }
        };

        // ---- picked 3: appeared mons (active first), imputed fills
        let mut picked: Vec<usize> = Vec::new();
        picked.push(active_slot);
        for (i, m) in mons.iter().enumerate() {
            if i != active_slot && m.appeared {
                picked.push(i);
            }
        }
        let mut imputed_pick = false;
        for (i, m) in mons.iter().enumerate() {
            if picked.len() >= 3 {
                break;
            }
            if !m.appeared {
                picked.push(i);
                imputed_pick = true;
            }
        }
        if picked.len() < 3 {
            continue;
        }
        picked.truncate(3);

        // ---- request JSON
        let act = &mons[active_slot];
        let act_set_moves: Vec<String> = own_sets_json[active_slot]["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
            .unwrap_or_default();
        let encore = act.vols.iter().find(|(k, _)| k == "encore").and_then(|(_, m)| *m);
        let disable = act.vols.iter().find(|(k, _)| k == "disable").and_then(|(_, m)| *m);
        let trapped =
            act.vols.iter().any(|(k, _)| k == "meanlook" || k == "partiallytrapped");
        let uses_of = |key: &str| -> i32 {
            act.uses
                .iter()
                .find(|(id, _)| plain(dex.moves.key(*id)) == plain(key))
                .map(|(_, n)| *n)
                .unwrap_or(0)
        };
        let req_moves: Vec<serde_json::Value> = act_set_moves
            .iter()
            .map(|name| {
                let key = toid(name);
                let pk = plain(&key);
                let (pp, maxpp) = dex
                    .moves
                    .id(&key)
                    .map(|id| {
                        let mx = dex.moves.get(id).pp as i32 * 8 / 5;
                        ((mx - uses_of(&key)).max(0), mx)
                    })
                    .unwrap_or((5, 5));
                let disabled = match (encore, disable) {
                    (Some(em), _) => plain(dex.moves.key(em)) != pk,
                    (None, Some(dm)) => plain(dex.moves.key(dm)) == pk,
                    _ => false,
                };
                serde_json::json!({"id": pk, "move": name, "pp": pp, "maxpp": maxpp,
                    "target": "normal", "disabled": disabled})
            })
            .collect();
        let req_mons: Vec<serde_json::Value> = picked
            .iter()
            .map(|&i| {
                let m = &mons[i];
                let cond = if m.fainted {
                    "0 fnt".to_string()
                } else {
                    let hp = ((m.hp_frac * maxhps[i] as f64).round() as i32)
                        .clamp(1, maxhps[i]);
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
                let ident = format!("p{}: {}", d.side + 1, nick);
                serde_json::json!({"ident": ident, "details": details(dex, m),
                    "condition": cond, "active": i == active_slot,
                    "item": own_sets_json[i]["item"].as_str().map(|s| toid(s)).unwrap_or_default()})
            })
            .collect();
        let req = serde_json::json!({
            "active": [{"moves": req_moves, "trapped": trapped}],
            "side": {"name": format!("p{}", d.side + 1), "id": format!("p{}", d.side + 1),
                     "pokemon": req_mons},
            "rqid": di as u64,
        })
        .to_string();

        // ---- run the agent
        let mut agent = ProtocolAgent::new(dex, d.side, load_meta_pool(pool_path), cfg(), seed);
        agent.set_own_team(own_sets);
        for ln in &lines[..=d.cut] {
            agent.push_line(dex, ln);
        }
        if let Err(e) = agent.on_request(dex, &req) {
            out.push(
                serde_json::json!({"battle": battle_idx, "side": d.side, "turn": d.turn,
                    "skip": "on-request", "err": e})
                .to_string(),
            );
            continue;
        }
        if agent.step(dex, iters).is_err() {
            continue;
        }
        let b = agent.battle().unwrap().clone();
        let Some(search) = agent.search() else { continue };

        // normalized root action strings aligned with visits
        let norm = |c: nc2000_engine::battle::SearchChoice| -> String {
            match c {
                nc2000_engine::battle::SearchChoice::Move(id) => {
                    format!("move {}", plain(dex.moves.key(id)))
                }
                nc2000_engine::battle::SearchChoice::Switch(pos) => {
                    let slot = b.sides[d.side].party.get(pos as usize - 1).copied().unwrap_or(0);
                    let sp = b.sides[d.side].roster[slot as usize].species;
                    format!("switch {}", dex.species.key(sp))
                }
                other => other.to_input(dex),
            }
        };
        let acts: Vec<String> = search.actions().iter().map(|&c| norm(c)).collect();
        let visits = search.visits();
        let human = match &d.action {
            HumanAction::Move(k) => format!("move {k}"),
            HumanAction::Switch(sp) => format!("switch {sp}"),
        };
        let mut order: Vec<usize> = (0..acts.len()).collect();
        order.sort_by(|&a, &z| visits[z].cmp(&visits[a]));
        let bot_best = order.first().map(|&i| acts[i].clone()).unwrap_or_default();
        let human_rank = order.iter().position(|&i| acts[i] == human).map(|r| r + 1);
        let revealed_cnt = act.uses.len().min(4);
        // action class for offline clustering: switch | Physical | Special | Status
        let class_of = |a: &str| -> &'static str {
            if a.starts_with("switch") {
                "switch"
            } else {
                a.strip_prefix("move ")
                    .and_then(|k| dex.moves.id(k))
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
                "rank": human_rank, "n_actions": acts.len(),
                "in_set": human_rank.is_some(),
                "revealed": revealed_cnt,
                "own_prov": provenance[active_slot],
                "imputed_pick": imputed_pick,
                "belief": agent.belief_info(),
                "iters": iters,
            })
            .to_string(),
        );
    }
    BattleReport { lines_out: out }
}

// ------------------------------------------------------------ main

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
        std::thread::available_parallelism().map(|n| n.get().saturating_sub(1).max(1)).unwrap_or(4),
    );
    let out_path = arg_s(&args, "--out", "tmp/human-agreement.jsonl");

    let (lo, hi) = {
        let mut it = range.split('-');
        let lo: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
        let hi: usize = it.next().unwrap_or("569").parse().unwrap_or(569);
        (lo, hi)
    };

    let dex = conformance::load_dex();
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src = load_sources(&dex, &root);
    let pool_path = root.join("data/meta-pool-v0/meta-pool.json");

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(root.join(&corpus))
        .unwrap_or_else(|e| panic!("corpus dir {corpus}: {e}"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".raw.log"))
        .collect();
    files.sort();
    let files: Vec<(usize, std::path::PathBuf)> =
        files.into_iter().enumerate().filter(|(i, _)| *i >= lo && *i <= hi).collect();
    eprintln!(
        "battles {} (index {lo}-{hi})  iters {iters}  threads {threads}  sources: {} species",
        files.len(),
        src.by_species.len()
    );

    let out = Mutex::new(std::io::BufWriter::new(std::fs::File::create(&out_path).unwrap()));
    let cursor = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let j = cursor.fetch_add(1, Ordering::Relaxed);
                if j >= files.len() {
                    return;
                }
                let (bi, path) = &files[j];
                let rep = process_battle(&dex, &src, &pool_path, path, *bi, iters, seed);
                {
                    let mut o = out.lock().unwrap();
                    for l in &rep.lines_out {
                        writeln!(o, "{l}").unwrap();
                    }
                    o.flush().unwrap();
                }
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 10 == 0 {
                    eprintln!("  {d}/{} battles", files.len());
                }
            });
        }
    });
    eprintln!("done -> {out_path}");
}
