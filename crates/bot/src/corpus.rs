//! Corpus-position reconstruction shared by the M17e harnesses
//! (endgame_exactness_corpus, anchor_gate). Fabrication logic mirrors
//! examples/human_agreement.rs (which keeps its own copy — migrate it here
//! on its next substantive edit): tracker over the protocol prefix, set
//! fabrication from rentals/pool/learnsets, synthesized full battle via
//! ProtocolAgent::on_request, no search step.

use std::collections::HashMap;

use crate::import::{MonSnapshot, ProtocolAgent, ProtocolTracker};
use crate::preview::load_meta_pool;
use crate::smmcts::{RmConfig, SelRule};
use nc2000_engine::dex::{toid, Dex, SpeciesId};
use nc2000_engine::state::{Battle, Status};

/// One loaded corpus battle: filtered protocol lines, eaten-berry facts,
/// and the extracted observable decisions.
pub struct CorpusBattle {
    pub lines: Vec<String>,
    pub eaten: Vec<(usize, String)>,
    pub decisions: Vec<Decision>,
}

/// Sorted .raw.log paths of the spectator corpus.
pub fn corpus_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("corpus dir {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".raw.log"))
        .collect();
    files.sort();
    files
}

pub fn load_battle(path: &std::path::Path) -> CorpusBattle {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<String> = text
        .lines()
        .filter(|l| {
            let t = l.split('|').nth(1).unwrap_or("");
            !matches!(t, "j" | "l" | "t:" | "init" | "title" | "")
        })
        .map(String::from)
        .collect();
    let decisions = extract_decisions(&lines);
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
    CorpusBattle { lines, eaten, decisions }
}

pub fn cfg() -> RmConfig {
    RmConfig { rule: SelRule::Ucb, c: 1.0, hp_buckets: 16, ..RmConfig::default() }
}

pub fn plain(key: &str) -> String {
    if key.starts_with("hiddenpower") { "hiddenpower".into() } else { key.into() }
}

// -- fabrication machinery: verbatim from examples/human_agreement.rs ------

pub struct SetSources {
    #[allow(dead_code)]
    pub by_species: HashMap<SpeciesId, Vec<serde_json::Value>>,
    pub learnsets: HashMap<String, Vec<String>>,
    pub hp_dvs: HashMap<String, (i64, i64)>,
}

pub fn load_sources(dex: &Dex, root: &std::path::Path) -> SetSources {
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
            hp_dvs
                .insert(k.clone(), (v["atk"].as_i64().unwrap_or(15), v["def"].as_i64().unwrap_or(15)));
        }
    }
    SetSources { by_species, learnsets, hp_dvs }
}

#[derive(Clone, Debug)]
pub enum HumanAction {
    Move(String),
    Switch(String),
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub side: usize,
    pub turn: u16,
    pub cut: usize,
    pub action: HumanAction,
}

pub fn extract_decisions(lines: &[String]) -> Vec<Decision> {
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

pub fn fabricate_set(
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

    let cands = src.by_species.get(&m.species).map(|v| v.as_slice()).unwrap_or(&[]);
    let norm_moves = |set: &serde_json::Value| -> Vec<String> {
        set["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(|s| plain(&toid(s)))).collect())
            .unwrap_or_default()
    };
    let mut best: Option<(usize, bool, usize)> = None;
    for (i, c) in cands.iter().enumerate() {
        let cm = norm_moves(c);
        let overlap = revealed.iter().filter(|r| cm.contains(r)).count();
        let full = overlap == revealed.len();
        let key = (overlap, full, usize::MAX - i);
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
            set = cands[i].clone();
            let cm: Vec<String> = set["moves"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let mut moves: Vec<String> = Vec::new();
            for r in &revealed {
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

pub fn details(dex: &Dex, m: &MonSnapshot) -> String {
    let name = &dex.species.get(m.species).name;
    let mut d = format!("{}, L{}", name, m.level);
    match format!("{:?}", m.gender).as_str() {
        "M" => d.push_str(", M"),
        "F" => d.push_str(", F"),
        _ => {}
    }
    d
}

// -- position reconstruction (human_agreement's per-decision body through
// on_request; no search step) ---------------------------------------------

pub fn reconstruct(
    dex: &Dex,
    src: &SetSources,
    pool_path: &std::path::Path,
    lines: &[String],
    eaten: &[(usize, String)],
    d: &Decision,
    seed: u64,
) -> Option<Battle> {
    let mut tr = ProtocolTracker::new(d.side);
    for ln in &lines[..=d.cut] {
        tr.push_line(dex, ln);
    }
    let (mons, active_slot) = tr.snapshot(d.side);
    let active_slot = active_slot?;
    if mons[active_slot].choiceless {
        return None;
    }

    let mut own_sets_json = Vec::new();
    for m in &mons {
        let ate = eaten.iter().any(|(s, n)| *s == d.side && !m.name.is_empty() && *n == m.name);
        let (set, _prov) = fabricate_set(dex, src, m, ate);
        own_sets_json.push(set);
    }
    let own_sets: Vec<nc2000_engine::battle::PokemonSet> =
        serde_json::from_value(serde_json::json!(own_sets_json.clone())).ok()?;
    let maxhps: Vec<i32> = Battle::from_fixture(dex, "1,2,3,4", &own_sets, &own_sets)
        .ok()?
        .sides[0]
        .roster
        .iter()
        .map(|p| p.maxhp)
        .collect();

    let mut picked: Vec<usize> = Vec::new();
    picked.push(active_slot);
    for (i, m) in mons.iter().enumerate() {
        if i != active_slot && m.appeared {
            picked.push(i);
        }
    }
    for (i, m) in mons.iter().enumerate() {
        if picked.len() >= 3 {
            break;
        }
        if !m.appeared {
            picked.push(i);
        }
    }
    if picked.len() < 3 {
        return None;
    }
    picked.truncate(3);

    let act = &mons[active_slot];
    let act_set_moves: Vec<String> = own_sets_json[active_slot]["moves"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let encore = act.vols.iter().find(|(k, _)| k == "encore").and_then(|(_, m)| *m);
    let disable = act.vols.iter().find(|(k, _)| k == "disable").and_then(|(_, m)| *m);
    let trapped = act.vols.iter().any(|(k, _)| k == "meanlook" || k == "partiallytrapped");
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
            let ident = format!("p{}: {}", d.side + 1, nick);
            serde_json::json!({"ident": ident, "details": details(dex, m),
                "condition": cond, "active": i == active_slot,
                "item": own_sets_json[i]["item"].as_str().map(toid).unwrap_or_default()})
        })
        .collect();
    let req = serde_json::json!({
        "active": [{"moves": req_moves, "trapped": trapped}],
        "side": {"name": format!("p{}", d.side + 1), "id": format!("p{}", d.side + 1),
                 "pokemon": req_mons},
        "rqid": d.cut as u64,
    })
    .to_string();

    let mut agent = ProtocolAgent::new(dex, d.side, load_meta_pool(pool_path), cfg(), seed);
    agent.set_own_team(own_sets);
    for ln in &lines[..=d.cut] {
        agent.push_line(dex, ln);
    }
    agent.on_request(dex, &req).ok()?;
    agent.battle().cloned()
}

