//! Corpus-position reconstruction shared by the M16b/M17 harnesses:
//! tracker state from the protocol prefix, full-log evidence for the acting
//! side's own submitted set, remaining-set fabrication from
//! rentals/pool/learnsets, then `ProtocolAgent::on_request` synthesis.

use std::collections::{HashMap, HashSet};

use crate::import::{MonSnapshot, ProtocolAgent, ProtocolTracker};
use crate::preview::{load_meta_pool, MetaPool};
use crate::smmcts::{RmConfig, SelRule};
use nc2000_engine::dex::{toid, Dex, SpeciesId};
use nc2000_engine::state::{Battle, MoveSlot, Status};

/// Full-log facts about the acting side's own set. These are legitimate in
/// an offline reconstruction: a live bot knows its submitted team even when
/// the spectator prefix has not revealed it yet. Battle state and opponent
/// information still come strictly from the decision prefix.
#[derive(Clone, Debug, Default)]
pub struct SetEvidence {
    moves: HashMap<(usize, String), Vec<String>>,
    items: HashMap<(usize, String), ItemEvent>,
    species_names: HashMap<(usize, String), String>,
}

#[derive(Clone, Debug)]
struct ItemEvent {
    cut: usize,
    item: String,
}

impl SetEvidence {
    fn resolved_name<'a>(&'a self, side: usize, name: &'a str, species: &str) -> &'a str {
        if !name.is_empty() {
            name
        } else {
            self.species_names
                .get(&(side, species.to_string()))
                .map(String::as_str)
                .unwrap_or(name)
        }
    }

    fn moves(&self, side: usize, name: &str, species: &str) -> &[String] {
        let name = self.resolved_name(side, name, species);
        self.moves
            .get(&(side, name.to_string()))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    fn item(&self, side: usize, name: &str, species: &str) -> Option<&ItemEvent> {
        let name = self.resolved_name(side, name, species);
        self.items.get(&(side, name.to_string()))
    }
}

/// One loaded corpus battle: filtered protocol lines, full-log own-set
/// evidence, and the extracted observable decisions.
pub struct CorpusBattle {
    pub lines: Vec<String>,
    pub evidence: SetEvidence,
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
    let evidence = collect_set_evidence(&lines);
    CorpusBattle {
        lines,
        evidence,
        decisions,
    }
}

fn collect_set_evidence(lines: &[String]) -> SetEvidence {
    let mut evidence = SetEvidence::default();
    let mut active_names: [Option<String>; 2] = std::array::from_fn(|_| None);
    let mut transformed: HashSet<(usize, String)> = HashSet::new();
    for (cut, line) in lines.iter().enumerate() {
        let parts: Vec<&str> = line.split('|').collect();
        let Some(subject) = parts.get(2) else {
            continue;
        };
        let side = if subject.as_bytes().get(1) == Some(&b'2') {
            1
        } else {
            0
        };
        let name = subject.split(':').nth(1).unwrap_or("").trim();
        if name.is_empty() {
            continue;
        }
        match parts.get(1).copied().unwrap_or("") {
            "switch" | "drag" | "replace" => {
                if let Some(outgoing) = active_names[side].replace(name.to_string()) {
                    transformed.remove(&(side, outgoing));
                }
                // A transformed mon returns to its base set on re-entry.
                // `replace` is the protocol's identity-replacement analogue
                // of a switch and likewise must not retain copied moves.
                transformed.remove(&(side, name.to_string()));
                let species = parts
                    .get(3)
                    .map(|details| toid(details.split(',').next().unwrap_or("")))
                    .unwrap_or_default();
                if !species.is_empty() {
                    evidence
                        .species_names
                        .entry((side, species))
                        .or_insert_with(|| name.to_string());
                }
            }
            "-transform" => {
                transformed.insert((side, name.to_string()));
            }
            "-end"
                if parts
                    .get(3)
                    .is_some_and(|effect| toid(effect) == "transform") =>
            {
                transformed.remove(&(side, name.to_string()));
            }
            "faint" => {
                transformed.remove(&(side, name.to_string()));
                if active_names[side].as_deref() == Some(name) {
                    active_names[side] = None;
                }
            }
            "move" => {
                // A move executed through Transform belongs to the copied
                // set, not the submitted one.
                if transformed.contains(&(side, name.to_string())) {
                    continue;
                }
                let Some(move_name) = parts.get(3) else {
                    continue;
                };
                let key = plain(&toid(move_name));
                // `[from] <other move>` is a called move. Sleep Talk is the
                // deliberate exception: its result is one of the submitted
                // slots. `[from] Pursuit` names the executing move itself,
                // so it is an ordinary submitted-move reveal.
                let from_key = parts
                    .iter()
                    .find_map(|part| part.strip_prefix("[from] "))
                    .map(|from| plain(&toid(from.strip_prefix("move: ").unwrap_or(from))));
                if from_key.is_some_and(|from| from != key && from != "sleeptalk") {
                    continue;
                }
                let moves = evidence.moves.entry((side, name.to_string())).or_default();
                if !moves.contains(&key) {
                    moves.push(key);
                }
            }
            "-enditem" if line.contains("[eat]") => {
                let Some(item) = parts.get(3) else { continue };
                evidence
                    .items
                    .entry((side, name.to_string()))
                    .or_insert(ItemEvent {
                        cut,
                        item: (*item).to_string(),
                    });
            }
            _ => {}
        }
    }
    evidence
}

pub fn cfg() -> RmConfig {
    RmConfig {
        rule: SelRule::Ucb,
        c: 1.0,
        hp_buckets: 16,
        ..RmConfig::default()
    }
}

pub fn plain(key: &str) -> String {
    if key.starts_with("hiddenpower") {
        "hiddenpower".into()
    } else {
        key.into()
    }
}

// -- fabrication machinery -------------------------------------------------

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
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            learnsets.insert(k.clone(), moves);
        }
    }
    let mut hp_dvs = HashMap::new();
    if let Some(o) = ls["hpDvs"].as_object() {
        for (k, v) in o {
            hp_dvs.insert(
                k.clone(),
                (
                    v["atk"].as_i64().unwrap_or(15),
                    v["def"].as_i64().unwrap_or(15),
                ),
            );
        }
    }
    SetSources {
        by_species,
        learnsets,
        hp_dvs,
    }
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
        let side_of = |s: &str| -> usize {
            if s.as_bytes().get(1) == Some(&b'2') {
                1
            } else {
                0
            }
        };
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
    future_moves: &[String],
    known_item: Option<&str>,
    item_consumed: bool,
) -> (serde_json::Value, &'static str) {
    let sp_key = dex.species.key(m.species).to_string();
    let sp_name = dex.species.get(m.species).name.clone();
    let mut revealed: Vec<String> = m
        .uses
        .iter()
        .map(|(id, _)| plain(dex.moves.key(*id)))
        .collect();
    for key in future_moves {
        if !revealed.contains(key) {
            revealed.push(key.clone());
        }
    }
    let nick = if m.name.is_empty() {
        sp_name.clone()
    } else {
        m.name.clone()
    };
    let gender = match format!("{:?}", m.gender).as_str() {
        "M" => "M",
        "F" => "F",
        _ => "",
    };

    let cands = src
        .by_species
        .get(&m.species)
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let norm_moves = |set: &serde_json::Value| -> Vec<String> {
        set["moves"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| plain(&toid(s))))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut best: Option<(usize, bool, usize)> = None;
    for (i, c) in cands.iter().enumerate() {
        let cm = norm_moves(c);
        let overlap = revealed.iter().filter(|r| cm.contains(r)).count();
        let full = overlap == revealed.len();
        let better = best.map_or(true, |(o, f, bi)| {
            overlap > o || (overlap == o && (full > f || (full == f && i < bi)))
        });
        if better {
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
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
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
    if let Some(item) = known_item {
        set["item"] = serde_json::json!(item);
    }
    if item_consumed {
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

/// Reconstruct one public-information decision point and retain the
/// protocol agent's belief/observer alongside the synthesized battle.  The
/// M17a regret miner needs all three to repeat blind searches from exactly
/// the same information set.
pub fn reconstruct_agent(
    dex: &Dex,
    src: &SetSources,
    pool_path: &std::path::Path,
    lines: &[String],
    evidence: &SetEvidence,
    d: &Decision,
    seed: u64,
) -> Option<ProtocolAgent> {
    reconstruct_agent_with_pool(
        dex,
        src,
        load_meta_pool(pool_path),
        lines,
        evidence,
        d,
        seed,
    )
}

/// Reconstruction plus the fabrication metadata used by the standing
/// human-agreement coverage report.
pub struct ReconstructedDecision {
    pub agent: ProtocolAgent,
    pub active_slot: usize,
    pub revealed_moves: usize,
    pub provenance: Vec<&'static str>,
    pub imputed_pick: bool,
}

/// [`reconstruct_agent`] with a caller-supplied pool. Corpus-scale tools
/// load the JSON once and clone the parsed pool per independent seed.
pub fn reconstruct_agent_with_pool(
    dex: &Dex,
    src: &SetSources,
    pool: MetaPool,
    lines: &[String],
    evidence: &SetEvidence,
    d: &Decision,
    seed: u64,
) -> Option<ProtocolAgent> {
    reconstruct_context_with_pool(dex, src, pool, lines, evidence, d, seed)
        .map(|reconstructed| reconstructed.agent)
}

/// [`reconstruct_agent_with_pool`] retaining fabrication metadata.
pub fn reconstruct_context_with_pool(
    dex: &Dex,
    src: &SetSources,
    pool: MetaPool,
    lines: &[String],
    evidence: &SetEvidence,
    d: &Decision,
    seed: u64,
) -> Option<ReconstructedDecision> {
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
    let mut provenance = Vec::new();
    for m in &mons {
        let species = dex.species.key(m.species);
        let item = evidence.item(d.side, &m.name, species);
        let (set, prov) = fabricate_set(
            dex,
            src,
            m,
            evidence.moves(d.side, &m.name, species),
            item.map(|event| event.item.as_str()),
            item.is_some_and(|event| event.cut <= d.cut),
        );
        own_sets_json.push(set);
        provenance.push(prov);
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
        return None;
    }
    picked.truncate(3);

    let act = &mons[active_slot];
    let act_set_moves: Vec<String> = own_sets_json[active_slot]["moves"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let encore = act
        .vols
        .iter()
        .find(|(k, _)| k == "encore")
        .and_then(|(_, m)| *m);
    let disable = act
        .vols
        .iter()
        .find(|(k, _)| k == "disable")
        .and_then(|(_, m)| *m);
    let trapped = act
        .vols
        .iter()
        .any(|(k, _)| k == "meanlook" || k == "partiallytrapped");
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

    let mut agent = ProtocolAgent::new(dex, d.side, pool, cfg(), seed);
    agent.set_own_team(own_sets);
    for ln in &lines[..=d.cut] {
        agent.push_line(dex, ln);
    }
    agent.on_request(dex, &req).ok()?;
    Some(ReconstructedDecision {
        agent,
        active_slot,
        revealed_moves: act.uses.len().min(4),
        provenance,
        imputed_pick,
    })
}

/// Reconstruct only the synthesized battle.  Kept as the compact API used
/// by the M17e exact-equity harnesses.
pub fn reconstruct(
    dex: &Dex,
    src: &SetSources,
    pool_path: &std::path::Path,
    lines: &[String],
    evidence: &SetEvidence,
    d: &Decision,
    seed: u64,
) -> Option<Battle> {
    reconstruct_agent(dex, src, pool_path, lines, evidence, d, seed)?
        .battle()
        .cloned()
}

/// Moves publicly used by each currently-active Pokemon anywhere in the
/// complete log. This deliberately looks past `cut`: it is an offline corpus
/// label, never information a live agent may consume.
pub fn active_future_revealed_moves(
    dex: &Dex,
    battle: &Battle,
    lines: &[String],
) -> [Vec<String>; 2] {
    let evidence = collect_set_evidence(lines);
    let mut out: [Vec<String>; 2] = std::array::from_fn(|_| Vec::new());
    for (side, moves) in out.iter_mut().enumerate() {
        let Some(id) = battle.active_id(side) else {
            continue;
        };
        let name = battle.poke(id).name.to_string();
        let species = dex.species.key(battle.poke(id).species);
        moves.extend(
            evidence
                .moves(side, &name, species)
                .iter()
                .filter(|key| dex.moves.id(key).is_some())
                .cloned(),
        );
    }
    out
}

/// Research-only oracle completion for the currently active pair. Future
/// public move reveals replace only candidate slots that had not been used at
/// the reconstruction cut. State, HP, status, already-observed PP, and every
/// opponent-hidden field remain exactly as reconstructed.
///
/// This is intentionally separate from `reconstruct`: callers must opt into
/// future leakage explicitly, and product agents cannot reach this API.
pub fn complete_active_moves_from_future(
    dex: &Dex,
    battle: &mut Battle,
    lines: &[String],
) -> [Vec<String>; 2] {
    let revealed = active_future_revealed_moves(dex, battle, lines);
    for side in 0..2 {
        let Some(id) = battle.active_id(side) else {
            continue;
        };
        if battle.poke(id).transformed {
            continue;
        }
        let revealed_ids: Vec<_> = revealed[side]
            .iter()
            .filter_map(|key| dex.moves.id(key))
            .collect();
        for move_id in revealed_ids.iter().copied() {
            if battle
                .poke(id)
                .move_slots
                .iter()
                .any(|slot| slot.id == move_id)
            {
                continue;
            }
            let replacement = battle
                .poke(id)
                .move_slots
                .iter()
                .enumerate()
                .rev()
                .find(|(_, slot)| {
                    !slot.used
                        && !revealed_ids.contains(&slot.id)
                        && battle.poke(id).last_move != Some(slot.id)
                        && battle.poke(id).last_move_used != Some(slot.id)
                })
                .map(|(index, _)| index);
            let maxpp = dex.moves.get(move_id).pp as i32 * 8 / 5;
            let slot = MoveSlot {
                id: move_id,
                pp: maxpp,
                maxpp,
                disabled: false,
                used: false,
                shared: true,
            };
            let pokemon = battle.poke_mut(id);
            match replacement {
                Some(index) => {
                    pokemon.move_slots[index] = slot;
                    if index < pokemon.base_move_slots.len() {
                        pokemon.base_move_slots[index] = slot;
                    }
                }
                None if pokemon.move_slots.len() < 4 => {
                    pokemon.move_slots.push(slot);
                    pokemon.base_move_slots.push(slot);
                }
                None => {}
            }
        }
    }
    revealed
}

#[cfg(test)]
mod tests {
    use super::*;
    use nc2000_engine::battle::PokemonSet;
    use nc2000_engine::state::Gender;

    fn splash_set() -> PokemonSet {
        PokemonSet {
            name: "Pikachu".into(),
            species: "Pikachu".into(),
            item: String::new(),
            ability: String::new(),
            moves: vec!["Splash".into()],
            level: 50,
            evs: None,
            ivs: None,
            happiness: None,
            gender: None,
        }
    }

    fn transform_set() -> PokemonSet {
        PokemonSet {
            name: "Ditto".into(),
            species: "Ditto".into(),
            item: String::new(),
            ability: String::new(),
            moves: vec!["Transform".into()],
            level: 50,
            evs: None,
            ivs: None,
            happiness: None,
            gender: None,
        }
    }

    fn snapshot(dex: &Dex, species: &str, moves: &[&str]) -> MonSnapshot {
        MonSnapshot {
            species: dex.species.id(species).unwrap(),
            level: 50,
            gender: Gender::N,
            name: species.to_string(),
            appeared: true,
            active: true,
            fainted: false,
            hp_frac: 1.0,
            status: Status::None,
            rest: false,
            boosts: [0; 7],
            vols: Vec::new(),
            uses: moves
                .iter()
                .map(|key| (dex.moves.id(key).unwrap(), 1))
                .collect(),
            choiceless: false,
        }
    }

    #[test]
    fn full_log_own_moves_select_the_matching_set_and_item_is_time_aware() {
        let dex = conformance::load_dex();
        let snorlax = dex.species.id("snorlax").unwrap();
        let mut by_species = HashMap::new();
        by_species.insert(
            snorlax,
            vec![
                serde_json::json!({"species":"Snorlax","item":"Leftovers",
                    "moves":["Body Slam","Earthquake","Self-Destruct","Belly Drum"]}),
                serde_json::json!({"species":"Snorlax","item":"Mint Berry",
                    "moves":["Body Slam","Earthquake","Rest","Curse"]}),
            ],
        );
        let src = SetSources {
            by_species,
            learnsets: HashMap::new(),
            hp_dvs: HashMap::new(),
        };
        let mon = snapshot(&dex, "snorlax", &["bodyslam", "earthquake"]);

        let (before, provenance) = fabricate_set(
            &dex,
            &src,
            &mon,
            &["rest".into()],
            Some("Mint Berry"),
            false,
        );
        assert_eq!(provenance, "cand-full");
        assert_eq!(before["item"], "Mint Berry");
        assert!(before["moves"]
            .as_array()
            .unwrap()
            .iter()
            .any(|m| m == "Rest"));

        let (after, _) =
            fabricate_set(&dex, &src, &mon, &["rest".into()], Some("Mint Berry"), true);
        assert_eq!(after["item"], "");
    }

    #[test]
    fn equal_candidate_scores_keep_the_earlier_source() {
        let dex = conformance::load_dex();
        let pikachu = dex.species.id("pikachu").unwrap();
        let mut by_species = HashMap::new();
        by_species.insert(
            pikachu,
            vec![
                serde_json::json!({"species":"Pikachu","item":"Mint Berry","moves":["Splash"]}),
                serde_json::json!({"species":"Pikachu","item":"Gold Berry","moves":["Splash"]}),
            ],
        );
        let src = SetSources {
            by_species,
            learnsets: HashMap::new(),
            hp_dvs: HashMap::new(),
        };
        let mon = snapshot(&dex, "pikachu", &["splash"]);
        let (set, _) = fabricate_set(&dex, &src, &mon, &[], None, false);
        assert_eq!(set["item"], "Mint Berry");
    }

    #[test]
    fn evidence_excludes_generic_called_moves_but_keeps_sleep_talk_calls() {
        let lines = vec![
            "|switch|p1a: Snorlax|Snorlax, L50, M|100/100".into(),
            "|move|p1a: Snorlax|Body Slam|p2a: Pikachu".into(),
            "|move|p1a: Snorlax|Thunder|p2a: Pikachu|[from] Metronome".into(),
            "|move|p1a: Snorlax|Rest|p1a: Snorlax|[from] Sleep Talk".into(),
            "|move|p1a: Snorlax|Pursuit|p2a: Pikachu|[from] Pursuit".into(),
            "|-enditem|p1a: Snorlax|Mint Berry|[eat]".into(),
        ];
        let evidence = collect_set_evidence(&lines);
        assert_eq!(
            evidence.moves(0, "", "snorlax"),
            ["bodyslam", "rest", "pursuit"]
        );
        let item = evidence.item(0, "", "snorlax").unwrap();
        assert_eq!(item.cut, 5);
        assert_eq!(item.item, "Mint Berry");
    }

    #[test]
    fn transformed_moves_are_not_submitted_set_evidence_and_switch_or_replace_clears_it() {
        let switched = vec![
            "|switch|p1a: Mew|Mew, L50|100/100".into(),
            "|move|p1a: Mew|Transform|p2a: Snorlax".into(),
            "|-transform|p1a: Mew|p2a: Snorlax".into(),
            "|move|p1a: Mew|Body Slam|p2a: Snorlax".into(),
            "|switch|p1a: Pikachu|Pikachu, L50|100/100".into(),
            "|switch|p1a: Mew|Mew, L50|100/100".into(),
            "|move|p1a: Mew|Psychic|p2a: Snorlax".into(),
        ];
        let evidence = collect_set_evidence(&switched);
        assert_eq!(evidence.moves(0, "Mew", "mew"), ["transform", "psychic"]);

        let replaced = vec![
            "|switch|p1a: Mew|Mew, L50|100/100".into(),
            "|move|p1a: Mew|Transform|p2a: Snorlax".into(),
            "|-transform|p1a: Mew|p2a: Snorlax".into(),
            "|move|p1a: Mew|Body Slam|p2a: Snorlax".into(),
            "|replace|p1a: Mew|Mew, L50|100/100".into(),
            "|move|p1a: Mew|Psychic|p2a: Snorlax".into(),
        ];
        let evidence = collect_set_evidence(&replaced);
        assert_eq!(evidence.moves(0, "Mew", "mew"), ["transform", "psychic"]);
    }

    #[test]
    fn ditto_transform_corpus_excerpt_does_not_leak_copied_move_to_future_oracle() {
        // battle-gen2...pbs-F-20260401-121508-7c046d6831be.raw.log
        // lines 51-66: Ditto transforms into Snorlax, then uses Body Slam.
        let dex = conformance::load_dex();
        let team = vec![transform_set()];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
        let p1 = battle.legal_choices(&dex, 0)[0];
        let p2 = battle.legal_choices(&dex, 1)[0];
        battle.apply_choices(&dex, [Some(p1), Some(p2)]).unwrap();
        let lines = vec![
            "|switch|p1a: Ditto|Ditto, L50|100/100".into(),
            "|move|p1a: Ditto|Transform|p2a: Snorlax".into(),
            "|-transform|p1a: Ditto|p2a: Snorlax".into(),
            "|move|p1a: Ditto|Body Slam|p2a: Snorlax".into(),
        ];

        let evidence = collect_set_evidence(&lines);
        assert_eq!(evidence.moves(0, "Ditto", "ditto"), ["transform"]);
        assert_eq!(
            active_future_revealed_moves(&dex, &battle, &lines)[0],
            ["transform"]
        );
    }

    #[test]
    fn tracker_does_not_retain_transformed_move_as_base_set_evidence() {
        let dex = conformance::load_dex();
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Ditto, L50|",
            "|poke|p1|Pikachu, L50|",
            "|poke|p2|Snorlax, L50, M|",
            "|switch|p1a: Ditto|Ditto, L50|100/100",
            "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
            "|turn|2",
            "|move|p1a: Ditto|Transform|p2a: Snorlax",
            "|-transform|p1a: Ditto|p2a: Snorlax",
            "|turn|3",
            "|move|p1a: Ditto|Body Slam|p2a: Snorlax",
            "|switch|p1a: Pikachu|Pikachu, L50|100/100",
            "|switch|p1a: Ditto|Ditto, L50|100/100",
        ] {
            tracker.push_line(&dex, line);
        }
        let (mons, _) = tracker.snapshot(0);
        let moves: Vec<_> = mons[0]
            .uses
            .iter()
            .map(|(id, _)| dex.moves.key(*id))
            .collect();
        assert_eq!(moves, ["transform"]);
    }

    #[test]
    fn future_move_oracle_counts_sleep_talk_calls_but_not_other_called_moves() {
        let dex = conformance::load_dex();
        let team = vec![splash_set()];
        let mut battle = Battle::from_fixture(&dex, "1,2,3,4", &team, &team).unwrap();
        let p1 = battle.legal_choices(&dex, 0)[0];
        let p2 = battle.legal_choices(&dex, 1)[0];
        battle.apply_choices(&dex, [Some(p1), Some(p2)]).unwrap();
        let lines = vec![
            "|move|p1a: Pikachu|Rest|p1a: Pikachu".into(),
            "|move|p1a: Pikachu|Thunderbolt|p2a: Pikachu|[from] Metronome".into(),
            "|move|p2a: Pikachu|Rest|p2a: Pikachu|[from] Sleep Talk".into(),
        ];

        let revealed = active_future_revealed_moves(&dex, &battle, &lines);
        assert_eq!(
            revealed,
            [vec!["rest".to_string()], vec!["rest".to_string()]]
        );
        complete_active_moves_from_future(&dex, &mut battle, &lines);
        let rest = dex.moves.id("rest").unwrap();
        for side in 0..2 {
            let id = battle.active_id(side).unwrap();
            assert!(battle
                .poke(id)
                .move_slots
                .iter()
                .any(|slot| slot.id == rest));
        }
    }
}
