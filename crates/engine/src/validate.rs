//! M14a team validator + canonicalizer for gen2nc2000 — a data-driven mirror
//! of what PS's TeamValidator enforces for this format (probed and verified
//! against the real validator by `tools/validate-oracle.js`).
//!
//! Inputs: the same team JSON the `Battle` constructor takes (fixture
//! `p1team` / meta-pool `sets` shape), plus `data/learnsets-gen2.json`
//! (exported by `tools/export-learnsets.js` through the format's full lens).
//!
//! Scope contract (documented superset): move legality is checked against
//! FLAT per-species acceptance sets — cross-move compatibility constraints
//! (incompatible egg-move parents, event-only move combinations) are NOT
//! encoded, so this validator accepts a small superset of true PS legality.
//! It must never reject a PS-legal team (the oracle cross-check certifies
//! zero reverse disagreements).
//!
//! Findings are machine-readable (`code` + params; M14b localizes the
//! messages) with two severities:
//! - `"error"`: PS's validator rejects the team (our verdict: illegal).
//! - `"fix"`: PS silently canonicalizes (ability/nature/gender fills, level
//!   default, clamps); legal, but not in canonical engine form.
//!
//! `canonicalize_team` applies every `fix` plus the *derivable* errors (HP DV
//! from the other DVs, SpD:=SpA mirrors, gender/shiny from DVs, typed-Hidden-
//! Power DV spreads, EV clamps, nickname repairs, duplicate-move dedupe).
//! Deliberately NOT auto-fixed (user intent, M14b surfaces them): species
//! identity/legality, levels, move choices, species/item clause conflicts,
//! team size, level-sum violations.

use std::collections::BTreeMap;

use serde_json::{json, Map, Value};

use crate::dex::{toid, Dex, SpeciesData};

/// Gen-2 Hidden Power type order: index = 4*(atkDV%4) + defDV%4.
const HP_TYPES: [&str; 16] = [
    "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
];

const STAT_KEYS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];
const MIN_LEVEL: i64 = 50;
const MAX_LEVEL: i64 = 55;
const DEFAULT_LEVEL: i64 = 55;
const MAX_TOTAL_LEVEL: i64 = crate::battle::MAX_TOTAL_LEVEL as i64;
const PICKED_TEAM_SIZE: usize = 3;
const MIN_TEAM_SIZE: usize = 3;
const MAX_TEAM_SIZE: usize = 6;
const MAX_MOVE_COUNT: usize = 4;
const MAX_NICKNAME_LEN: usize = 18; // UTF-16 units, PS's limit

// ------------------------------------------------------------- learnsets

/// `data/learnsets-gen2.json` — per-species flat move-acceptance sets plus
/// the species/move level floors and the canonical typed-Hidden-Power DV
/// spreads.
pub struct Learnsets {
    species: BTreeMap<String, SpeciesLearnset>,
    /// typeid -> canonical DV overrides (PS typechart `HPdvs`, DV units).
    hp_dvs: BTreeMap<String, BTreeMap<String, i64>>,
}

pub struct SpeciesLearnset {
    /// Smallest legal level in 50..=55 (evolution floors: dragonite 55, ...).
    pub min_level: i64,
    /// Move ids legal at level 55 (single-move acceptance — flat superset).
    pub moves: Vec<String>,
    /// Moves only legal from a level above `min_level` (level-up moves
    /// learned inside the 51..=55 window).
    pub move_min_level: BTreeMap<String, i64>,
}

impl SpeciesLearnset {
    pub fn allows(&self, move_id: &str) -> bool {
        self.moves.binary_search_by(|m| m.as_str().cmp(move_id)).is_ok()
    }
}

impl Learnsets {
    pub fn from_json(text: &str) -> Result<Learnsets, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let mut species = BTreeMap::new();
        for (id, entry) in v["species"].as_object().ok_or("missing species table")? {
            let mut moves: Vec<String> = entry["moves"]
                .as_array()
                .ok_or_else(|| format!("{id}: missing moves"))?
                .iter()
                .filter_map(|m| m.as_str().map(str::to_string))
                .collect();
            moves.sort();
            let move_min_level = entry["moveMinLevel"]
                .as_object()
                .map(|m| {
                    m.iter()
                        .filter_map(|(k, v)| v.as_i64().map(|n| (k.clone(), n)))
                        .collect()
                })
                .unwrap_or_default();
            species.insert(
                id.clone(),
                SpeciesLearnset {
                    min_level: entry["minLevel"].as_i64().unwrap_or(MIN_LEVEL),
                    moves,
                    move_min_level,
                },
            );
        }
        let mut hp_dvs = BTreeMap::new();
        if let Some(table) = v["hpDvs"].as_object() {
            for (t, dvs) in table {
                let m: BTreeMap<String, i64> = dvs
                    .as_object()
                    .map(|o| {
                        o.iter().filter_map(|(k, v)| v.as_i64().map(|n| (k.clone(), n))).collect()
                    })
                    .unwrap_or_default();
                hp_dvs.insert(t.clone(), m);
            }
        }
        Ok(Learnsets { species, hp_dvs })
    }

    /// The species' acceptance entry (`None` = not legal in this format:
    /// unknown or Uber-banned — the banned five carry no entry).
    pub fn species(&self, species_id: &str) -> Option<&SpeciesLearnset> {
        self.species.get(species_id)
    }
}

// -------------------------------------------------------------- findings

fn finding(severity: &str, code: &str, params: Value) -> Value {
    let mut obj = Map::new();
    obj.insert("severity".into(), json!(severity));
    obj.insert("code".into(), json!(code));
    if let Value::Object(p) = params {
        obj.extend(p);
    }
    Value::Object(obj)
}

fn err(code: &str, params: Value) -> Value {
    finding("error", code, params)
}

fn fix(code: &str, params: Value) -> Value {
    finding("fix", code, params)
}

// ------------------------------------------------------------ set analysis

/// One set's fields, tolerantly extracted (bad JSON shapes become findings,
/// not parse failures).
struct RawSet {
    name: String,
    species: String,
    item: String,
    ability: String,
    nature: Option<String>,
    gender: Option<String>,
    shiny: bool,
    level: Option<i64>,
    happiness: Option<i64>,
    moves: Vec<String>,
    evs: Option<BTreeMap<String, i64>>,
    ivs: Option<BTreeMap<String, i64>>,
}

fn stat_map(v: &Value) -> Option<BTreeMap<String, i64>> {
    v.as_object().map(|o| {
        o.iter()
            .filter_map(|(k, x)| x.as_f64().map(|n| (k.clone(), n as i64)))
            .collect()
    })
}

fn raw_set(v: &Value) -> RawSet {
    let s = |k: &str| v[k].as_str().unwrap_or("").to_string();
    RawSet {
        name: s("name"),
        species: s("species"),
        item: s("item"),
        ability: s("ability"),
        nature: v["nature"].as_str().map(str::to_string),
        gender: v["gender"].as_str().map(str::to_string),
        shiny: v["shiny"].as_bool().unwrap_or(false),
        level: v["level"].as_i64(),
        happiness: v["happiness"].as_i64(),
        moves: v["moves"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m.as_str().map(str::to_string)).collect())
            .unwrap_or_default(),
        evs: stat_map(&v["evs"]),
        ivs: stat_map(&v["ivs"]),
    }
}

/// Six raw stat values in hp/atk/def/spa/spd/spe order, missing keys
/// defaulted (PS fillStats semantics), values NOT clamped (PS checks the raw
/// numbers — e.g. iv 60 fails the HP-DV check with DV 30, exactly like PS).
fn six(map: &Option<BTreeMap<String, i64>>, default: i64) -> [i64; 6] {
    let mut out = [default; 6];
    if let Some(m) = map {
        for (i, k) in STAT_KEYS.iter().enumerate() {
            if let Some(&v) = m.get(*k) {
                out[i] = v;
            }
        }
    }
    out
}

/// JS `Math.floor(iv / 2)` (negative-safe mirror).
fn dv(iv: i64) -> i64 {
    (iv as f64 / 2.0).floor() as i64
}

/// Gen-2 expected HP DV from the other DVs' low bits (JS `%` semantics).
fn expected_hp_dv(dvs: [i64; 4]) -> i64 {
    let [atk, def, spe, spc] = dvs;
    (atk % 2) * 8 + (def % 2) * 4 + (spe % 2) * 2 + (spc % 2)
}

fn hp_type_from_dvs(atk_dv: i64, def_dv: i64) -> &'static str {
    let idx = (4 * (atk_dv % 4) + (def_dv % 4)).rem_euclid(16);
    HP_TYPES[idx as usize]
}

/// Gen-2 shiny is DV-determined.
fn expected_shiny(atk_dv: i64, def_dv: i64, spe_dv: i64, spc_dv: i64) -> bool {
    def_dv == 10 && spe_dv == 10 && spc_dv == 10 && atk_dv % 4 >= 2
}

/// PS's Unown letter formula, mirrored on the raw IVs: bits 3..2 of each
/// 5-bit IV (atk, def, spe, spa order) concatenated, /10. Base Unown = "A".
fn unown_letter(ivs: &[i64; 6]) -> char {
    let mut bits = String::new();
    for &i in &[1usize, 2, 5, 3] {
        let iv = ivs[i].clamp(0, i64::MAX); // negative IVs: JS would produce garbage; clamp low
        let b = format!("{:05b}", iv.min(31)); // >31 keeps high bits in JS; 5-bit mirror is enough for 0..=31
        bits.push_str(&b[1..3]);
    }
    let n = i64::from_str_radix(&bits, 2).unwrap_or(0) / 10;
    (b'A' + n as u8) as char
}

/// The typed-Hidden-Power type carried by a move id, if any
/// ("hiddenpowerice" -> "ice"; plain "hiddenpower" -> None).
fn hidden_power_type(move_id: &str) -> Option<&str> {
    move_id
        .strip_prefix("hiddenpower")
        .filter(|rest| !rest.is_empty())
}

/// The gen-2 gender expected from the Atk DV, `None` for fixed-gender
/// species (PS silently overwrites those).
fn expected_gender(species: &SpeciesData, atk_dv: i64) -> Option<&'static str> {
    if species.gender.is_some() {
        return None;
    }
    let f = species.extra.get("genderRatio").and_then(|g| g["F"].as_f64()).unwrap_or(0.5);
    Some(if atk_dv as f64 >= f * 16.0 { "M" } else { "F" })
}

/// Effective IVs for the stat checks: PS rewrites a typed-Hidden-Power set's
/// DVs to the canonical spread when every IV is maxed (silent auto-fill).
/// Returns the rewritten IVs, or `None` when no rewrite applies.
fn hp_autofill(ls: &Learnsets, hp_type: &str, ivs: &[i64; 6]) -> Option<[i64; 6]> {
    if !ivs.iter().all(|&v| v == 31) {
        return None;
    }
    let mut out = [30i64; 6];
    if let Some(dvs) = ls.hp_dvs.get(hp_type) {
        for (i, k) in STAT_KEYS.iter().enumerate() {
            if let Some(&d) = dvs.get(*k) {
                out[i] = d * 2;
            }
        }
    }
    out[0] = expected_hp_dv([dv(out[1]), dv(out[2]), dv(out[5]), dv(out[3])]) * 2;
    Some(out)
}

/// Validate one set. `mon` params carry the species id + display slot.
fn validate_set(dex: &Dex, ls: &Learnsets, set: &RawSet, slot: usize, out: &mut Vec<Value>) {
    let sid = toid(&set.species);
    let mon = json!({ "mon": sid, "slot": slot });
    let mon_with = |extra: Value| -> Value {
        let mut m = mon.as_object().unwrap().clone();
        if let Value::Object(e) = extra {
            m.extend(e);
        }
        Value::Object(m)
    };

    // species existence: PS hard-returns ("does not exist in Gen 2")
    let Some(species_id) = dex.species.id(&sid) else {
        out.push(err("species-unknown", json!({ "mon": sid, "slot": slot, "species": set.species })));
        return;
    };
    let species = dex.species.get(species_id);

    // Uber tag ban (the format's whole banlist)
    if species.tier.as_deref() == Some("Uber") {
        out.push(err("species-banned", mon_with(json!({ "tier": "Uber" }))));
    }

    // levels: missing/0 -> PS defaults to 55 (silent); 50..=55 enforced
    let level = match set.level {
        Some(l) if l != 0 => l,
        _ => {
            out.push(fix("level-default", mon_with(json!({ "level": DEFAULT_LEVEL }))));
            DEFAULT_LEVEL
        }
    };
    if level < MIN_LEVEL {
        out.push(err("level-min", mon_with(json!({ "level": level, "min": MIN_LEVEL }))));
    }
    if level > MAX_LEVEL {
        out.push(err("level-max", mon_with(json!({ "level": level, "max": MAX_LEVEL }))));
    }

    // species level floor ("must be at least level N to be evolved")
    let learnset = ls.species(&sid);
    if let Some(l) = learnset {
        if level < l.min_level {
            out.push(err(
                "species-underleveled",
                mon_with(json!({ "level": level, "min": l.min_level })),
            ));
        }
    } else if species.tier.as_deref() != Some("Uber") {
        // dex-known species missing from the learnset table — data drift
        out.push(err("species-unknown", mon_with(json!({ "species": set.species }))));
    }

    // moves: 1..=4, known, no duplicates, inside the flat acceptance set
    let move_ids: Vec<String> = set.moves.iter().filter(|m| !m.is_empty()).map(|m| toid(m)).collect();
    if move_ids.is_empty() {
        out.push(err("move-none", mon.clone()));
    }
    if move_ids.len() > MAX_MOVE_COUNT {
        out.push(err(
            "move-count",
            mon_with(json!({ "count": move_ids.len(), "max": MAX_MOVE_COUNT })),
        ));
    }
    let mut seen_moves: Vec<&str> = Vec::new();
    let mut hp_move_type: Option<&str> = None;
    let mut hp_conflict = false;
    for id in &move_ids {
        if dex.moves.id(id).is_none() {
            out.push(err("move-unknown", mon_with(json!({ "move": id }))));
            continue;
        }
        if seen_moves.contains(&id.as_str()) {
            out.push(err("move-duplicate", mon_with(json!({ "move": id }))));
        }
        seen_moves.push(id);
        // learnset membership (typed Hidden Power collapses onto the base id)
        let base = if hidden_power_type(id).is_some() { "hiddenpower" } else { id.as_str() };
        if let Some(t) = hidden_power_type(id) {
            match hp_move_type {
                Some(prev) if prev != t && !hp_conflict => {
                    out.push(err(
                        "hp-type-conflict",
                        mon_with(json!({ "a": prev, "b": t })),
                    ));
                    hp_conflict = true;
                }
                None => hp_move_type = Some(t),
                _ => {}
            }
        }
        if let Some(l) = learnset {
            if !l.allows(base) {
                out.push(err("move-illegal", mon_with(json!({ "move": base }))));
            } else if let Some(&min) = l.move_min_level.get(base) {
                if level < min {
                    out.push(err(
                        "move-level",
                        mon_with(json!({ "move": base, "level": level, "min": min })),
                    ));
                }
            }
        }
    }

    // item: empty ok; otherwise must exist in the gen-2 dex
    if !set.item.is_empty() && dex.items.id(&toid(&set.item)).is_none() {
        out.push(err("item-unknown", mon_with(json!({ "item": set.item }))));
    }

    // ---- gen-2 stat consistency (PS validateStats, mirrored on raw values)
    let given_ivs = six(&set.ivs, 31);
    // typed Hidden Power: with maxed IVs PS silently rewrites the DVs to the
    // canonical spread; otherwise the derived type must match the move.
    let mut ivs = given_ivs;
    if let (Some(t), false) = (hp_move_type, hp_conflict) {
        if let Some(rewritten) = hp_autofill(ls, t, &given_ivs) {
            ivs = rewritten;
        } else {
            let derived = hp_type_from_dvs(dv(given_ivs[1]), dv(given_ivs[2]));
            if derived != t {
                out.push(err(
                    "hp-type-mismatch",
                    mon_with(json!({ "want": t, "derived": derived })),
                ));
            }
        }
    }
    let (atk_dv, def_dv, spe_dv, spc_dv) = (dv(ivs[1]), dv(ivs[2]), dv(ivs[5]), dv(ivs[3]));
    if ivs[3] != ivs[4] {
        out.push(err("dv-spc", mon_with(json!({ "spa": ivs[3], "spd": ivs[4] }))));
    }
    let want_hp = expected_hp_dv([atk_dv, def_dv, spe_dv, spc_dv]);
    if dv(ivs[0]) != want_hp {
        out.push(err("dv-hp", mon_with(json!({ "hp": dv(ivs[0]), "expected": want_hp }))));
    }
    match (expected_gender(species, atk_dv), set.gender.as_deref()) {
        (Some(exp), Some(g @ ("M" | "F"))) if g != exp => {
            out.push(err(
                "dv-gender",
                mon_with(json!({ "gender": g, "expected": exp, "atkDv": atk_dv })),
            ));
        }
        (Some(exp), Some("M" | "F")) => {
            let _ = exp; // matches — fine
        }
        (Some(exp), _) => {
            // missing/invalid on a dual-gender species: PS fills silently
            out.push(fix("gender-fill", mon_with(json!({ "expected": exp }))));
        }
        (None, g) => {
            // fixed-gender species: PS silently overwrites a wrong value
            let fixed = species.gender.as_deref().unwrap_or("N");
            if g != Some(fixed) && g.is_some() {
                out.push(fix("gender-species", mon_with(json!({ "expected": fixed }))));
            }
        }
    }
    let shiny = expected_shiny(atk_dv, def_dv, spe_dv, spc_dv);
    if shiny != set.shiny {
        out.push(err("dv-shiny", mon_with(json!({ "expected": shiny }))));
    }
    // Unown's letter forme is DV-derived; the engine's base Unown is forme A
    if sid == "unown" {
        let letter = unown_letter(&ivs);
        if letter != 'A' {
            out.push(err(
                "unown-forme",
                mon_with(json!({ "letter": letter.to_string(), "expected": "A" })),
            ));
        }
    }

    // EVs (gen-2 stat exp): 0..=255 each, SpA==SpD, not all zero
    if set.evs.is_none() {
        out.push(fix("evs-missing", mon.clone()));
    } else {
        let evs = six(&set.evs, 0);
        for (i, k) in STAT_KEYS.iter().enumerate() {
            if !(0..=255).contains(&evs[i]) {
                out.push(err("ev-range", mon_with(json!({ "stat": k, "value": evs[i] }))));
            }
        }
        if evs[3] != evs[4] {
            out.push(err("ev-spc", mon_with(json!({ "spa": evs[3], "spd": evs[4] }))));
        }
        if evs.iter().sum::<i64>() == 0 {
            out.push(err("ev-zero", mon.clone()));
        }
    }

    // canonical-form fills (PS silently normalizes; the engine expects the
    // canonical 'No Ability' / 'Serious' forms)
    if set.ability != "No Ability" {
        out.push(fix("ability-canonical", mon.clone()));
    }
    if set.nature.as_deref() != Some("Serious") {
        out.push(fix("nature-canonical", mon.clone()));
    }
    if let Some(h) = set.happiness {
        if !(0..=255).contains(&h) {
            out.push(fix("happiness-range", mon_with(json!({ "happiness": h }))));
        }
    }

    // nickname sanity: PS's 18-UTF-16-unit limit (a name equal to the species
    // name is silently reset, never an error) and species impersonation
    if !set.name.is_empty() && set.name != species.name {
        let len = set.name.encode_utf16().count();
        if len > MAX_NICKNAME_LEN {
            out.push(err(
                "nickname-length",
                mon_with(json!({ "name": set.name, "len": len, "max": MAX_NICKNAME_LEN })),
            ));
        }
        if let Some(other) = dex.species.id(&toid(&set.name)) {
            let other = dex.species.get(other);
            if other.name.to_lowercase() == set.name.to_lowercase() && other.name != species.name {
                out.push(err("nickname-species", mon_with(json!({ "name": set.name }))));
            }
        }
    }
}

// ------------------------------------------------------------ team analysis

fn analyze(dex: &Dex, ls: &Learnsets, sets: &[RawSet]) -> Vec<Value> {
    let mut out = Vec::new();

    if sets.len() < MIN_TEAM_SIZE {
        out.push(err("team-size", json!({ "size": sets.len(), "min": MIN_TEAM_SIZE })));
    }
    if sets.len() > MAX_TEAM_SIZE {
        out.push(err("team-size", json!({ "size": sets.len(), "max": MAX_TEAM_SIZE })));
    }

    // Species Clause (by species identity) + Item Clause (limit 1) +
    // Nickname Clause (dupes; a name matching the mon's own species is exempt)
    let mut species_seen: Vec<String> = Vec::new();
    let mut items_seen: Vec<String> = Vec::new();
    let mut names_seen: Vec<String> = Vec::new();
    for (i, set) in sets.iter().enumerate() {
        let sid = toid(&set.species);
        if let Some(id) = dex.species.id(&sid) {
            if species_seen.contains(&sid) {
                out.push(err("species-clause", json!({ "mon": sid, "slot": i })));
            }
            species_seen.push(sid.clone());
            let item = toid(&set.item);
            if !item.is_empty() {
                if items_seen.contains(&item) {
                    out.push(err("item-clause", json!({ "mon": sid, "slot": i, "item": item })));
                }
                items_seen.push(item);
            }
            let species_name = &dex.species.get(id).name;
            if !set.name.is_empty() && &set.name != species_name {
                if names_seen.contains(&set.name) {
                    out.push(err(
                        "nickname-clause",
                        json!({ "mon": sid, "slot": i, "name": set.name }),
                    ));
                }
                names_seen.push(set.name.clone());
            }
        }
    }

    // Max Total Level = 155 over the picked 3 (PS's exact two checks,
    // including the index-existence semantics for short teams)
    let mut levels: Vec<i64> = sets
        .iter()
        .map(|s| match s.level {
            Some(l) if l != 0 => l,
            _ => DEFAULT_LEVEL,
        })
        .collect();
    levels.sort_unstable();
    if levels.len() >= PICKED_TEAM_SIZE {
        let sum: i64 = levels[..PICKED_TEAM_SIZE].iter().sum();
        if sum > MAX_TOTAL_LEVEL {
            out.push(err("level-sum", json!({ "sum": sum, "limit": MAX_TOTAL_LEVEL })));
        }
    }
    if levels.len() >= PICKED_TEAM_SIZE - 1 && !levels.is_empty() {
        let sum: i64 =
            levels[levels.len() - 1] + levels[..PICKED_TEAM_SIZE - 1].iter().sum::<i64>();
        if sum > MAX_TOTAL_LEVEL {
            out.push(err(
                "level-sum-highest",
                json!({ "sum": sum, "limit": MAX_TOTAL_LEVEL, "level": levels[levels.len() - 1] }),
            ));
        }
    }

    for (i, set) in sets.iter().enumerate() {
        validate_set(dex, ls, set, i, &mut out);
    }
    out
}

/// Validate a team (the `Battle` constructor's JSON shape). Returns findings
/// JSON: `{ok, errors, findings: [{severity, code, ...params}]}` — `ok` means
/// zero `error`-severity findings (`fix` findings alone keep a team legal).
pub fn validate_team(dex: &Dex, ls: &Learnsets, team_json: &str) -> Value {
    let sets = match parse_team(team_json) {
        Ok(sets) => sets,
        Err(e) => return verdict(vec![err("json-invalid", json!({ "detail": e }))]),
    };
    verdict(analyze(dex, ls, &sets))
}

fn parse_team(team_json: &str) -> Result<Vec<RawSet>, String> {
    let v: Value = serde_json::from_str(team_json).map_err(|e| e.to_string())?;
    let arr = v.as_array().ok_or("team JSON must be an array of sets")?;
    Ok(arr.iter().map(raw_set).collect())
}

fn verdict(findings: Vec<Value>) -> Value {
    let errors = findings.iter().filter(|f| f["severity"] == "error").count();
    json!({ "ok": errors == 0, "errors": errors, "findings": findings })
}

// ------------------------------------------------------------ canonicalize

/// Canonical Unown-A DV spreads per Hidden Power type (only Fighting /
/// Flying / Rock / Bug coexist with forme A): (atkDV, defDV); Spe DV 9,
/// Spc DV 15, HP DV derived.
fn unown_a_dvs(hp_type: Option<&str>) -> Option<(i64, i64)> {
    match hp_type {
        None | Some("bug") => Some((9, 9)),
        Some("fighting") => Some((8, 8)),
        Some("flying") => Some((8, 9)),
        Some("rock") => Some((9, 8)),
        _ => None,
    }
}

/// Canonicalize a team: apply every `fix`-severity finding plus the
/// derivable errors (HP DV, SpD:=SpA mirrors, gender/shiny from DVs, typed
/// Hidden Power DV spreads, EV clamps/fills, nickname repairs, duplicate
/// moves). Returns `{ok, team, applied, errors}`: `applied` lists what was
/// changed, `errors` what remains (species/level/move legality, clauses —
/// not auto-fixable). `ok` = parseable and no remaining errors.
pub fn canonicalize_team(dex: &Dex, ls: &Learnsets, team_json: &str) -> Value {
    let v: Value = match serde_json::from_str(team_json) {
        Ok(v) => v,
        Err(e) => {
            return json!({
                "ok": false, "team": Value::Null, "applied": [],
                "errors": [err("json-invalid", json!({ "detail": e.to_string() }))],
            })
        }
    };
    let Some(arr) = v.as_array() else {
        return json!({
            "ok": false, "team": Value::Null, "applied": [],
            "errors": [err("json-invalid", json!({ "detail": "team JSON must be an array of sets" }))],
        });
    };

    let mut applied: Vec<Value> = Vec::new();
    let mut fixed_sets: Vec<Value> = Vec::new();
    for (slot, raw) in arr.iter().enumerate() {
        fixed_sets.push(canonicalize_set(dex, ls, &raw_set(raw), slot, &mut applied));
    }

    // Nickname Clause repair: later duplicates reset to the species name
    let mut names_seen: Vec<String> = Vec::new();
    for (slot, set) in fixed_sets.iter_mut().enumerate() {
        let name = set["name"].as_str().unwrap_or("").to_string();
        let species = set["species"].as_str().unwrap_or("").to_string();
        if name.is_empty() || name == species {
            continue;
        }
        if names_seen.contains(&name) {
            set["name"] = json!(species);
            applied.push(fix("nickname-clause", json!({ "mon": toid(&species), "slot": slot, "name": name })));
        } else {
            names_seen.push(name);
        }
    }

    // Remaining problems on the fixed team (errors only — canonicalization
    // leaves no fix-severity findings behind by construction).
    let fixed: Vec<RawSet> = fixed_sets.iter().map(raw_set).collect();
    let errors: Vec<Value> = analyze(dex, ls, &fixed)
        .into_iter()
        .filter(|f| f["severity"] == "error")
        .collect();
    json!({
        "ok": errors.is_empty(),
        "team": fixed_sets,
        "applied": applied,
        "errors": errors,
    })
}

fn canonicalize_set(
    dex: &Dex,
    ls: &Learnsets,
    set: &RawSet,
    slot: usize,
    applied: &mut Vec<Value>,
) -> Value {
    let sid = toid(&set.species);
    let mon_with = |extra: Value| -> Value {
        let mut m = json!({ "mon": sid, "slot": slot }).as_object().unwrap().clone();
        if let Value::Object(e) = extra {
            m.extend(e);
        }
        Value::Object(m)
    };
    let mut note = |f: Value| applied.push(f);

    let Some(species_id) = dex.species.id(&sid) else {
        // unknown species: nothing to canonicalize; re-validation reports it
        return set_to_value(set, &set.species, &set.name, set.gender.clone(), set.shiny);
    };
    let species = dex.species.get(species_id);
    let species_name = species.name.clone();

    // name: empty -> species name; impersonation -> species name; length cap
    let mut name = if set.name.is_empty() { species_name.clone() } else { set.name.clone() };
    if name != species_name {
        if let Some(other) = dex.species.id(&toid(&name)) {
            let other = dex.species.get(other);
            if other.name.to_lowercase() == name.to_lowercase() && other.name != species_name {
                note(fix("nickname-species", mon_with(json!({ "name": name }))));
                name = species_name.clone();
            }
        }
    }
    if name != species_name && name.encode_utf16().count() > MAX_NICKNAME_LEN {
        note(fix("nickname-length", mon_with(json!({ "name": name }))));
        name = {
            let units: Vec<u16> = name.encode_utf16().take(MAX_NICKNAME_LEN).collect();
            String::from_utf16_lossy(&units)
        };
    }

    // moves: drop empties, dedupe by id, normalize to dex display names
    let mut move_names: Vec<String> = Vec::new();
    let mut move_ids: Vec<String> = Vec::new();
    let mut hp_move_type: Option<String> = None;
    for m in set.moves.iter().filter(|m| !m.is_empty()) {
        let id = toid(m);
        if move_ids.contains(&id) {
            note(fix("move-duplicate", mon_with(json!({ "move": id }))));
            continue;
        }
        if let Some(t) = hidden_power_type(&id) {
            if hp_move_type.is_none() {
                hp_move_type = Some(t.to_string());
            }
        }
        move_names.push(match dex.moves.id(&id) {
            Some(mid) => dex.moves.get(mid).name.clone(),
            None => m.clone(),
        });
        move_ids.push(id);
    }

    // IVs: materialize + clamp, then the derivable DV fixes
    let mut ivs = six(&set.ivs, 31);
    let clamped: Vec<usize> = (0..6).filter(|&i| !(0..=31).contains(&ivs[i])).collect();
    if !clamped.is_empty() {
        note(fix("iv-range", mon_with(json!({ "stats": clamped.iter().map(|&i| STAT_KEYS[i]).collect::<Vec<_>>() }))));
        for i in clamped {
            ivs[i] = ivs[i].clamp(0, 31);
        }
    }
    let hp_t = hp_move_type.as_deref();
    if sid == "unown" {
        // joint forme-A + Hidden-Power spread when one exists; otherwise the
        // conflict stays a validation error
        if unown_letter(&ivs) != 'A' {
            if let Some((atk, def)) = unown_a_dvs(hp_t) {
                note(fix("unown-forme", mon_with(json!({}))));
                ivs = [0, atk * 2, def * 2, 30, 30, 18];
                ivs[0] = expected_hp_dv([atk, def, 9, 15]) * 2;
            }
        }
    } else if let Some(t) = hp_t {
        // typed Hidden Power: rewrite to the canonical spread when maxed
        // (PS's silent auto-fill) or when the derived type mismatches
        let maxed = ivs.iter().all(|&v| v == 31);
        let derived = hp_type_from_dvs(dv(ivs[1]), dv(ivs[2]));
        if maxed || derived != t {
            note(fix("hp-type-dvs", mon_with(json!({ "type": t }))));
            let mut out = [30i64; 6];
            if let Some(dvs) = ls.hp_dvs.get(t) {
                for (i, k) in STAT_KEYS.iter().enumerate() {
                    if let Some(&d) = dvs.get(*k) {
                        out[i] = d * 2;
                    }
                }
            }
            out[0] = expected_hp_dv([dv(out[1]), dv(out[2]), dv(out[5]), dv(out[3])]) * 2;
            ivs = out;
        }
    }
    if ivs[3] != ivs[4] {
        note(fix("dv-spc", mon_with(json!({}))));
        ivs[4] = ivs[3];
    }
    let want_hp = expected_hp_dv([dv(ivs[1]), dv(ivs[2]), dv(ivs[5]), dv(ivs[3])]);
    if dv(ivs[0]) != want_hp {
        note(fix("dv-hp", mon_with(json!({ "expected": want_hp }))));
        ivs[0] = want_hp * 2;
    }

    // gender from the (possibly rewritten) Atk DV / the species' fixed gender
    let atk_dv = dv(ivs[1]);
    let gender = match expected_gender(species, atk_dv) {
        Some(exp) => {
            match set.gender.as_deref() {
                Some(g) if g == exp => {}
                Some(_) => note(fix("dv-gender", mon_with(json!({ "expected": exp })))),
                None => note(fix("gender-fill", mon_with(json!({ "expected": exp })))),
            }
            exp.to_string()
        }
        None => {
            let fixed_g = species.gender.clone().unwrap_or_else(|| "N".into());
            if let Some(g) = set.gender.as_deref() {
                if g != fixed_g {
                    note(fix("gender-species", mon_with(json!({ "expected": fixed_g }))));
                }
            }
            fixed_g
        }
    };

    // shiny strictly follows the DVs
    let shiny = expected_shiny(atk_dv, dv(ivs[2]), dv(ivs[5]), dv(ivs[3]));
    if shiny != set.shiny {
        note(fix("dv-shiny", mon_with(json!({ "expected": shiny }))));
    }

    // EVs: fill missing at 255 (the format convention), clamp, mirror SpD
    let mut evs = match &set.evs {
        None => {
            note(fix("evs-missing", mon_with(json!({}))));
            [255i64; 6]
        }
        Some(_) => six(&set.evs, 0),
    };
    let clamped: Vec<usize> = (0..6).filter(|&i| !(0..=255).contains(&evs[i])).collect();
    if !clamped.is_empty() {
        note(fix("ev-range", mon_with(json!({ "stats": clamped.iter().map(|&i| STAT_KEYS[i]).collect::<Vec<_>>() }))));
        for i in clamped {
            evs[i] = evs[i].clamp(0, 255);
        }
    }
    if evs[3] != evs[4] {
        note(fix("ev-spc", mon_with(json!({}))));
        evs[4] = evs[3];
    }
    if evs.iter().sum::<i64>() == 0 {
        note(fix("ev-zero", mon_with(json!({}))));
        evs = [255; 6];
    }

    // scalar canonical forms
    if set.ability != "No Ability" {
        note(fix("ability-canonical", mon_with(json!({}))));
    }
    if set.nature.as_deref() != Some("Serious") {
        note(fix("nature-canonical", mon_with(json!({}))));
    }
    let level = match set.level {
        Some(l) if l != 0 => l,
        _ => {
            note(fix("level-default", mon_with(json!({ "level": DEFAULT_LEVEL }))));
            DEFAULT_LEVEL
        }
    };
    let happiness = match set.happiness {
        Some(h) if !(0..=255).contains(&h) => {
            note(fix("happiness-range", mon_with(json!({ "happiness": h }))));
            h.clamp(0, 255)
        }
        Some(h) => h,
        None => 255,
    };

    // item: normalize to the dex display name when known
    let item = match dex.items.id(&toid(&set.item)) {
        Some(iid) if !set.item.is_empty() => dex.items.get(iid).name.clone(),
        _ => set.item.clone(),
    };

    let stat_obj = |vals: [i64; 6]| -> Value {
        let mut m = Map::new();
        for (i, k) in STAT_KEYS.iter().enumerate() {
            m.insert((*k).into(), json!(vals[i]));
        }
        Value::Object(m)
    };
    let mut out = Map::new();
    out.insert("name".into(), json!(name));
    out.insert("species".into(), json!(species_name));
    out.insert("item".into(), json!(item));
    out.insert("ability".into(), json!("No Ability"));
    out.insert("moves".into(), json!(move_names));
    out.insert("nature".into(), json!("Serious"));
    out.insert("evs".into(), stat_obj(evs));
    out.insert("gender".into(), json!(gender));
    out.insert("ivs".into(), stat_obj(ivs));
    out.insert("level".into(), json!(level));
    out.insert("happiness".into(), json!(happiness));
    if shiny {
        out.insert("shiny".into(), json!(true));
    }
    Value::Object(out)
}

/// Untouched passthrough for a set whose species is unknown (nothing can be
/// derived; re-validation reports it).
fn set_to_value(set: &RawSet, species: &str, name: &str, gender: Option<String>, shiny: bool) -> Value {
    let mut out = Map::new();
    out.insert("name".into(), json!(name));
    out.insert("species".into(), json!(species));
    out.insert("item".into(), json!(set.item));
    out.insert("ability".into(), json!(set.ability));
    out.insert("moves".into(), json!(set.moves));
    if let Some(n) = &set.nature {
        out.insert("nature".into(), json!(n));
    }
    if let Some(evs) = &set.evs {
        out.insert("evs".into(), json!(evs));
    }
    if let Some(g) = gender {
        out.insert("gender".into(), json!(g));
    }
    if let Some(ivs) = &set.ivs {
        out.insert("ivs".into(), json!(ivs));
    }
    if let Some(l) = set.level {
        out.insert("level".into(), json!(l));
    }
    if let Some(h) = set.happiness {
        out.insert("happiness".into(), json!(h));
    }
    if shiny {
        out.insert("shiny".into(), json!(true));
    }
    Value::Object(out)
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
    }

    fn load() -> (Dex, Learnsets) {
        let root = repo_root();
        let dex_text = std::fs::read_to_string(root.join("data/gen2stadium2.json")).unwrap();
        let ls_text = std::fs::read_to_string(root.join("data/learnsets-gen2.json")).unwrap();
        (Dex::from_json(&dex_text).unwrap(), Learnsets::from_json(&ls_text).unwrap())
    }

    fn team(sets: Vec<Value>) -> String {
        serde_json::to_string(&sets).unwrap()
    }

    fn snorlax() -> Value {
        json!({
            "name": "Snorlax", "species": "Snorlax", "item": "Leftovers",
            "ability": "No Ability", "moves": ["Body Slam", "Rest", "Curse", "Earthquake"],
            "nature": "Serious",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "gender": "M",
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "level": 55
        })
    }

    fn cloyster() -> Value {
        json!({
            "name": "Cloyster", "species": "Cloyster", "item": "Gold Berry",
            "ability": "No Ability", "moves": ["Surf", "Explosion"],
            "nature": "Serious",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "gender": "M",
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "level": 50
        })
    }

    fn suicune() -> Value {
        json!({
            "name": "Suicune", "species": "Suicune", "item": "",
            "ability": "No Ability", "moves": ["Surf", "Rest"],
            "nature": "Serious",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "gender": "N",
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "level": 50
        })
    }

    fn codes(v: &Value) -> Vec<(String, String)> {
        v["findings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                (f["severity"].as_str().unwrap().to_string(), f["code"].as_str().unwrap().to_string())
            })
            .collect()
    }

    #[test]
    fn clean_team_passes_with_no_findings() {
        let (dex, ls) = load();
        let r = validate_team(&dex, &ls, &team(vec![snorlax(), cloyster(), suicune()]));
        assert_eq!(r["ok"], true, "{r}");
        assert!(codes(&r).is_empty(), "{r}");
    }

    #[test]
    fn error_catalogue() {
        let (dex, ls) = load();
        let case = |mutate: fn(&mut Value), code: &str| {
            let mut s = snorlax();
            mutate(&mut s);
            let r = validate_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
            assert_eq!(r["ok"], false, "{code}: {r}");
            assert!(
                codes(&r).iter().any(|(sev, c)| sev == "error" && c == code),
                "expected {code}: {r}"
            );
        };
        case(|s| s["species"] = json!("Blaziken"), "species-unknown");
        case(|s| s["species"] = json!("Mewtwo"), "species-banned");
        case(|s| s["level"] = json!(49), "level-min");
        case(|s| s["level"] = json!(56), "level-max");
        case(|s| s["moves"] = json!(["Spikes"]), "move-illegal");
        case(|s| s["moves"] = json!(["Body Slam", "Body Slam"]), "move-duplicate");
        case(|s| s["moves"] = json!([]), "move-none");
        case(|s| s["moves"] = json!(["Body Slam", "Rest", "Curse", "Earthquake", "Surf"]), "move-count");
        case(|s| s["moves"] = json!(["Slime Wave"]), "move-unknown");
        case(|s| s["item"] = json!("Choice Band"), "item-unknown");
        case(|s| s["ivs"]["spd"] = json!(28), "dv-spc");
        case(|s| s["ivs"]["hp"] = json!(10), "dv-hp");
        case(|s| s["ivs"] = json!({"hp":22,"atk":2,"def":30,"spa":30,"spd":30,"spe":30}), "dv-gender");
        case(
            |s| s["ivs"] = json!({"hp":30,"atk":31,"def":21,"spa":21,"spd":21,"spe":21}),
            "dv-shiny",
        );
        case(|s| s["evs"]["hp"] = json!(300), "ev-range");
        case(|s| s["evs"]["spd"] = json!(4), "ev-spc");
        case(|s| s["evs"] = json!({"hp":0,"atk":0,"def":0,"spa":0,"spd":0,"spe":0}), "ev-zero");
        case(|s| s["name"] = json!("AAAAAAAAAAAAAAAAAAA"), "nickname-length");
        case(|s| s["name"] = json!("Pikachu"), "nickname-species");
    }

    #[test]
    fn dragonite_evolution_floor() {
        let (dex, ls) = load();
        let mut s = snorlax();
        s["species"] = json!("Dragonite");
        s["name"] = json!("Dragonite");
        s["moves"] = json!(["Thunder Wave"]);
        s["level"] = json!(50);
        let r = validate_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert!(
            codes(&r).iter().any(|(_, c)| c == "species-underleveled"),
            "{r}"
        );
    }

    #[test]
    fn team_level_clauses() {
        let (dex, ls) = load();
        // 52+52+52 = 156 > 155
        let mut a = snorlax();
        let mut b = cloyster();
        let mut c = suicune();
        a["level"] = json!(52);
        b["level"] = json!(52);
        c["level"] = json!(52);
        let r = validate_team(&dex, &ls, &team(vec![a, b, c]));
        assert!(codes(&r).iter().any(|(_, c)| c == "level-sum"), "{r}");
        // 55 highest + two 50s = fine; 55 + 51 + 51 = 157 -> highest unusable
        let mut a = snorlax();
        let mut b = cloyster();
        let mut c = suicune();
        a["level"] = json!(55);
        b["level"] = json!(51);
        c["level"] = json!(51);
        let r = validate_team(&dex, &ls, &team(vec![a, b, c]));
        assert!(codes(&r).iter().any(|(_, c)| c == "level-sum-highest"), "{r}");
    }

    #[test]
    fn clause_dupes() {
        let (dex, ls) = load();
        let mut b = snorlax();
        b["item"] = json!("Gold Berry");
        let r = validate_team(&dex, &ls, &team(vec![snorlax(), b, suicune()]));
        assert!(codes(&r).iter().any(|(_, c)| c == "species-clause"), "{r}");
        let mut b = cloyster();
        b["item"] = json!("Leftovers");
        let r = validate_team(&dex, &ls, &team(vec![snorlax(), b, suicune()]));
        assert!(codes(&r).iter().any(|(_, c)| c == "item-clause"), "{r}");
        let mut a = snorlax();
        let mut b = cloyster();
        a["name"] = json!("Blob");
        b["name"] = json!("Blob");
        let r = validate_team(&dex, &ls, &team(vec![a, b, suicune()]));
        assert!(codes(&r).iter().any(|(_, c)| c == "nickname-clause"), "{r}");
    }

    #[test]
    fn hidden_power_typing() {
        let (dex, ls) = load();
        // typed HP matching the DVs (HP Ice canonical spread: def DV 13)
        let mut s = snorlax();
        s["moves"] = json!(["Hidden Power Ice"]);
        s["ivs"] = json!({"hp":30,"atk":30,"def":26,"spa":30,"spd":30,"spe":30});
        let r = validate_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert_eq!(r["ok"], true, "{r}");
        // typed HP with maxed IVs: PS auto-fills, we accept
        let mut s = snorlax();
        s["moves"] = json!(["Hidden Power Ice"]);
        s["ivs"] = json!({"hp":31,"atk":31,"def":31,"spa":31,"spd":31,"spe":31});
        s["gender"] = Value::Null;
        let r = validate_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert_eq!(r["ok"], true, "{r}");
        // typed HP contradicting the DVs
        let mut s = snorlax();
        s["moves"] = json!(["Hidden Power Ice"]);
        let r = validate_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert!(codes(&r).iter().any(|(_, c)| c == "hp-type-mismatch"), "{r}");
    }

    #[test]
    fn unown_forme() {
        let (dex, ls) = load();
        let mut s = snorlax();
        s["name"] = json!("Unown");
        s["species"] = json!("Unown");
        s["moves"] = json!(["Hidden Power"]);
        s["gender"] = json!("N");
        // max DVs = forme Z -> error
        let r = validate_team(&dex, &ls, &team(vec![s.clone(), cloyster(), suicune()]));
        assert!(codes(&r).iter().any(|(_, c)| c == "unown-forme"), "{r}");
        // forme-A spread passes
        s["ivs"] = json!({"hp":30,"atk":18,"def":18,"spa":30,"spd":30,"spe":18});
        let r = validate_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert_eq!(r["ok"], true, "{r}");
    }

    #[test]
    fn canonicalize_fixes_derivables() {
        let (dex, ls) = load();
        // broken hp DV + missing gender + wrong ability/nature + dup move
        let mut s = snorlax();
        s["ivs"]["hp"] = json!(10);
        s["gender"] = Value::Null;
        s["ability"] = json!("");
        s["nature"] = json!("Adamant");
        s["moves"] = json!(["Body Slam", "Body Slam", "Rest"]);
        let r = canonicalize_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert_eq!(r["ok"], true, "{r}");
        let fixed = serde_json::to_string(&r["team"]).unwrap();
        let v = validate_team(&dex, &ls, &fixed);
        assert_eq!(v["ok"], true, "{v}");
        assert!(codes(&v).is_empty(), "canonical team must have zero findings: {v}");
        assert_eq!(r["team"][0]["ivs"]["hp"], json!(30));
        assert_eq!(r["team"][0]["gender"], json!("M"));
        assert_eq!(r["team"][0]["ability"], json!("No Ability"));
        assert_eq!(r["team"][0]["nature"], json!("Serious"));
        assert_eq!(r["team"][0]["moves"], json!(["Body Slam", "Rest"]));
        // unfixable errors survive canonicalization
        let mut s = snorlax();
        s["moves"] = json!(["Spikes"]);
        let r = canonicalize_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert_eq!(r["ok"], false, "{r}");
        assert!(r["errors"].as_array().unwrap().iter().any(|f| f["code"] == "move-illegal"));
    }

    #[test]
    fn canonicalized_team_constructs_a_battle() {
        let (dex, ls) = load();
        let mut s = snorlax();
        s["gender"] = Value::Null;
        s["ability"] = json!("");
        let r = canonicalize_team(&dex, &ls, &team(vec![s, cloyster(), suicune()]));
        assert_eq!(r["ok"], true, "{r}");
        let fixed = serde_json::to_string(&r["team"]).unwrap();
        let sets: Vec<crate::battle::PokemonSet> = serde_json::from_str(&fixed).unwrap();
        let b = crate::state::Battle::from_fixture(&dex, "1,2,3,4", &sets, &sets.clone());
        assert!(b.is_ok(), "{:?}", b.err());
    }
}
