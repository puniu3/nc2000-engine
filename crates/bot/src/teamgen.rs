//! M11a metagame-research machinery: mutation operators over the *legal*
//! set space plus gauntlet fitness evaluation.
//!
//! Mutation operators propose neighbors of a parent team (6 sets) inside the
//! format-legal space defined by the M14a validator: move swaps from the
//! species' learnset, item swaps from the format's item lens, level/DV/
//! stat-exp tweaks honoring the gen-2 landmines, species swaps over the 246
//! legal species (baseline = a meta-pool set of that species when one
//! exists, else a heuristic default), and whole-mon replacement from the
//! pool's 204 sets. Every proposal is repaired through
//! `canonicalize_team` (the derivable fixes: HP DV, SpD mirrors, gender/
//! shiny from DVs, typed-Hidden-Power spreads, EV clamps) and then must
//! re-validate with ZERO findings — illegal or unrepairable drafts are
//! rejected, never emitted. Deterministic given the caller's `SplitMix64`.
//!
//! Teams travel as canonical set JSON (`Vec<serde_json::Value>`, the
//! validator's shape) because the canonical form carries fields
//! (`nature`, materialized IVs) that `PokemonSet` does not; `to_sets`
//! converts for battle construction.
//!
//! Fitness: `gauntlet_eval` — seed-paired duels (both orientations on the
//! same battle seed) against each gauntlet team at a configurable skuct
//! budget, preview included (live search — candidates have no baked
//! tables). Deterministic for a given seed at any thread count (the
//! arena/bake pattern: precomputed job list, atomic cursor, results sorted
//! by job index before aggregation).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{json, Value};

use nc2000_engine::battle::{Outcome, PokemonSet};
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::Battle;
use nc2000_engine::validate::{canonicalize_team, validate_team, Learnsets};

use crate::rng::SplitMix64;
use crate::runner::{play_game, GameResult};
use crate::smmcts::{RmAgent, RmConfig, SelRule};

/// Gen-2 Hidden Power types (typed move ids exist for each).
const HP_TYPES: [&str; 16] = [
    "fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
    "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark",
];

/// Utility-move preference order for heuristic default sets (first legal
/// entry fills the fourth slot).
const UTILITY_MOVES: [&str; 22] = [
    "rest", "recover", "softboiled", "milkdrink", "moonlight", "morningsun",
    "synthesis", "sleeppowder", "spore", "lovelykiss", "hypnosis",
    "thunderwave", "stunspore", "toxic", "substitute", "reflect",
    "lightscreen", "curse", "amnesia", "swordsdance", "agility", "encore",
];

// -------------------------------------------------------------- operators

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutOp {
    /// Replace (or, when fewer than 4, sometimes add) one move with another
    /// from the species' learnset; typed Hidden Power resolves to a random
    /// type (the canonicalizer applies the DV spread).
    MoveSwap,
    /// New held item from the format's item lens (meta-pool items weighted
    /// 3:1 over the full dex list), honoring Item Clause.
    ItemSwap,
    /// New level within `[max(50, species floor), 55]`; team-level-sum rules
    /// are enforced by the validator (illegal draws are rejected).
    LevelTweak,
    /// One of Atk/Def/Spe/Spc DVs redrawn (Spc mirrors SpA=SpD); HP DV,
    /// gender, shiny, and typed-HP consistency re-derived by the
    /// canonicalizer.
    DvTweak,
    /// One stat-exp value redrawn from {0,64,128,192,255} (Spc mirrored).
    EvTweak,
    /// New species from the 246 legal (Species Clause honored); set
    /// re-derived: meta-pool baseline when the species has one, else the
    /// heuristic default.
    SpeciesSwap,
    /// Whole-mon replacement with one of the meta pool's 204 sets.
    MonReplace,
}

impl MutOp {
    pub const ALL: [MutOp; 7] = [
        MutOp::MoveSwap,
        MutOp::ItemSwap,
        MutOp::LevelTweak,
        MutOp::DvTweak,
        MutOp::EvTweak,
        MutOp::SpeciesSwap,
        MutOp::MonReplace,
    ];

    pub fn name(self) -> &'static str {
        match self {
            MutOp::MoveSwap => "move-swap",
            MutOp::ItemSwap => "item-swap",
            MutOp::LevelTweak => "level-tweak",
            MutOp::DvTweak => "dv-tweak",
            MutOp::EvTweak => "ev-tweak",
            MutOp::SpeciesSwap => "species-swap",
            MutOp::MonReplace => "mon-replace",
        }
    }
}

/// A validator-clean neighbor of the parent team.
pub struct Proposal {
    pub op: MutOp,
    pub slot: usize,
    /// Canonical team JSON (six set objects).
    pub team: Vec<Value>,
}

// -------------------------------------------------------------- pool teams

pub struct PoolTeamRaw {
    pub id: String,
    pub tier: String,
    /// Raw set objects exactly as the pool file carries them.
    pub sets: Vec<Value>,
}

// ---------------------------------------------------------------- TeamGen

pub struct TeamGen {
    ls: Learnsets,
    /// The format-legal species ids (sorted; 246 under the current export).
    species_ids: Vec<String>,
    /// Every item display name in the format lens (sorted by id).
    all_items: Vec<String>,
    /// Meta-pool-observed item display names + "" (no item), deduped sorted.
    pool_items: Vec<String>,
    teams: Vec<PoolTeamRaw>,
    /// species id -> (team, slot) of pool sets of that species.
    by_species: BTreeMap<String, Vec<(usize, usize)>>,
    /// Flat (team, slot) list over all pool sets (mon-replacement draws).
    all_sets: Vec<(usize, usize)>,
}

impl TeamGen {
    /// `learnsets_json` = data/learnsets-gen2.json, `pool_json` =
    /// data/meta-pool-v0/meta-pool.json (raw text — the pool's set objects
    /// are kept as JSON so no fields are lost in typed round-trips).
    pub fn new(dex: &Dex, learnsets_json: &str, pool_json: &str) -> Result<TeamGen, String> {
        let ls = Learnsets::from_json(learnsets_json)?;
        let v: Value = serde_json::from_str(learnsets_json).map_err(|e| e.to_string())?;
        let mut species_ids: Vec<String> = v["species"]
            .as_object()
            .ok_or("learnsets: missing species table")?
            .keys()
            .filter(|id| dex.species.id(id).is_some())
            .cloned()
            .collect();
        species_ids.sort();

        let all_items: Vec<String> = dex.items.values.iter().map(|d| d.name.clone()).collect();

        let pv: Value = serde_json::from_str(pool_json).map_err(|e| e.to_string())?;
        let mut teams = Vec::new();
        let mut by_species: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
        let mut all_sets = Vec::new();
        let mut pool_items: Vec<String> = Vec::new();
        for (ti, t) in pv["teams"].as_array().ok_or("pool: missing teams")?.iter().enumerate() {
            let id = t["id"].as_str().ok_or("pool team without id")?.to_string();
            let tier = t["tier"].as_str().unwrap_or("").to_string();
            let sets = t["sets"].as_array().ok_or("pool team without sets")?.clone();
            for (si, s) in sets.iter().enumerate() {
                let sid = toid(s["species"].as_str().unwrap_or(""));
                by_species.entry(sid).or_default().push((ti, si));
                all_sets.push((ti, si));
                let item = s["item"].as_str().unwrap_or("").to_string();
                if !pool_items.contains(&item) {
                    pool_items.push(item);
                }
            }
            teams.push(PoolTeamRaw { id, tier, sets });
        }
        if !pool_items.contains(&String::new()) {
            pool_items.push(String::new());
        }
        pool_items.sort();

        Ok(TeamGen { ls, species_ids, all_items, pool_items, teams, by_species, all_sets })
    }

    pub fn teams(&self) -> &[PoolTeamRaw] {
        &self.teams
    }

    /// A pool team's raw set JSON (canonize before mutating from it).
    pub fn team_json(&self, i: usize) -> Vec<Value> {
        self.teams[i].sets.clone()
    }

    /// Canonicalize + strict-validate a raw team. `None` = illegal (the
    /// canonicalizer could not repair it) — never emit such a team.
    pub fn canonize(&self, dex: &Dex, raw: &[Value]) -> Option<Vec<Value>> {
        let text = serde_json::to_string(raw).ok()?;
        let r = canonicalize_team(dex, &self.ls, &text);
        if r["ok"] != json!(true) {
            return None;
        }
        let team = r["team"].as_array()?.clone();
        // Strict re-validation: canonical output must carry zero findings.
        let v = validate_team(dex, &self.ls, &serde_json::to_string(&team).ok()?);
        if v["ok"] != json!(true) || !v["findings"].as_array()?.is_empty() {
            return None;
        }
        Some(team)
    }

    /// One mutation attempt from a CANONICAL parent (6 sets). `None` = the
    /// draw produced an illegal/unchanged team; the caller retries
    /// (`propose_valid`). Deterministic given `rng`.
    pub fn propose(&self, dex: &Dex, parent: &[Value], rng: &mut SplitMix64) -> Option<Proposal> {
        assert_eq!(parent.len(), 6, "mutation parents must be 6-mon teams");
        let op = MutOp::ALL[rng.below(MutOp::ALL.len())];
        let slot = rng.below(6);
        let mut raw = parent.to_vec();
        match op {
            MutOp::MoveSwap => self.mut_move_swap(dex, &mut raw, slot, rng)?,
            MutOp::ItemSwap => self.mut_item_swap(&mut raw, slot, rng)?,
            MutOp::LevelTweak => self.mut_level_tweak(&mut raw, slot, rng)?,
            MutOp::DvTweak => mut_dv_tweak(&mut raw, slot, rng),
            MutOp::EvTweak => mut_ev_tweak(&mut raw, slot, rng),
            MutOp::SpeciesSwap => self.mut_species_swap(dex, &mut raw, slot, rng)?,
            MutOp::MonReplace => self.mut_mon_replace(&mut raw, slot, rng)?,
        }
        let team = self.canonize(dex, &raw)?;
        if team_key(&team) == team_key(parent) {
            return None; // no-op (e.g. a DV tweak the canonicalizer reverted)
        }
        Some(Proposal { op, slot, team })
    }

    /// `propose` with up to `tries` fresh draws.
    pub fn propose_valid(
        &self,
        dex: &Dex,
        parent: &[Value],
        rng: &mut SplitMix64,
        tries: usize,
    ) -> Option<Proposal> {
        (0..tries).find_map(|_| self.propose(dex, parent, rng))
    }

    // ------------------------------------------------------ op internals

    fn mut_move_swap(
        &self,
        dex: &Dex,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) -> Option<()> {
        let sid = species_id(&raw[slot]);
        let l = self.ls.species(&sid)?;
        let level = set_level(&raw[slot]);
        let cur_bases: Vec<String> = move_ids(&raw[slot]).iter().map(|m| base_id(m)).collect();
        let cands: Vec<&String> = l
            .moves
            .iter()
            .filter(|m| {
                !cur_bases.contains(m)
                    && dex.moves.id(m).is_some()
                    && l.move_min_level.get(m.as_str()).is_none_or(|&mn| level >= mn)
            })
            .collect();
        if cands.is_empty() {
            return None;
        }
        let pick = cands[rng.below(cands.len())].clone();
        let display = if pick == "hiddenpower" {
            // Typed variant; the canonicalizer applies the DV spread.
            let typed: Vec<String> = HP_TYPES
                .iter()
                .map(|t| format!("hiddenpower{t}"))
                .filter(|id| dex.moves.id(id).is_some())
                .collect();
            if typed.is_empty() {
                return None;
            }
            move_name(dex, &typed[rng.below(typed.len())])
        } else {
            move_name(dex, &pick)
        };
        let moves = raw[slot]["moves"].as_array_mut()?;
        if moves.is_empty() {
            return None;
        }
        if moves.len() < 4 && rng.below(3) == 0 {
            moves.push(json!(display));
        } else {
            let p = rng.below(moves.len());
            moves[p] = json!(display);
        }
        Some(())
    }

    fn mut_item_swap(&self, raw: &mut [Value], slot: usize, rng: &mut SplitMix64) -> Option<()> {
        let current = toid(raw[slot]["item"].as_str().unwrap_or(""));
        let held = held_items(raw, slot);
        let list = if rng.below(4) < 3 { &self.pool_items } else { &self.all_items };
        let cands: Vec<&String> = list
            .iter()
            .filter(|it| {
                let id = toid(it);
                id != current && (id.is_empty() || !held.contains(&id))
            })
            .collect();
        if cands.is_empty() {
            return None;
        }
        raw[slot]["item"] = json!(cands[rng.below(cands.len())]);
        Some(())
    }

    fn mut_level_tweak(&self, raw: &mut [Value], slot: usize, rng: &mut SplitMix64) -> Option<()> {
        let sid = species_id(&raw[slot]);
        let lo = self.ls.species(&sid)?.min_level.max(50);
        let cur = set_level(&raw[slot]);
        let new = lo + rng.below((55 - lo + 1) as usize) as i64;
        if new == cur {
            return None;
        }
        raw[slot]["level"] = json!(new);
        Some(())
    }

    fn mut_species_swap(
        &self,
        dex: &Dex,
        raw: &mut [Value],
        slot: usize,
        rng: &mut SplitMix64,
    ) -> Option<()> {
        let cur = species_id(&raw[slot]);
        let teammates: Vec<String> =
            (0..raw.len()).filter(|&i| i != slot).map(|i| species_id(&raw[i])).collect();
        let sid = &self.species_ids[rng.below(self.species_ids.len())];
        if *sid == cur || teammates.contains(sid) {
            return None;
        }
        let held = held_items(raw, slot);
        raw[slot] = self.derive_set(dex, sid, set_level(&raw[slot]), &held, rng)?;
        Some(())
    }

    fn mut_mon_replace(&self, raw: &mut [Value], slot: usize, rng: &mut SplitMix64) -> Option<()> {
        let (ti, si) = self.all_sets[rng.below(self.all_sets.len())];
        let mut set = self.teams[ti].sets[si].clone();
        let sid = species_id(&set);
        let teammates: Vec<String> =
            (0..raw.len()).filter(|&i| i != slot).map(|i| species_id(&raw[i])).collect();
        if teammates.contains(&sid) {
            return None; // Species Clause
        }
        self.repair_item_clause(&mut set, &held_items(raw, slot), rng);
        raw[slot] = set;
        Some(())
    }

    /// Set for `sid`: a meta-pool set of that species when one exists
    /// (uniform draw, kept at its own level), else the heuristic default at
    /// `level_hint` (clamped to the species floor).
    fn derive_set(
        &self,
        dex: &Dex,
        sid: &str,
        level_hint: i64,
        held: &[String],
        rng: &mut SplitMix64,
    ) -> Option<Value> {
        if let Some(entries) = self.by_species.get(sid) {
            let (ti, si) = entries[rng.below(entries.len())];
            let mut set = self.teams[ti].sets[si].clone();
            self.repair_item_clause(&mut set, held, rng);
            return Some(set);
        }
        let l = self.ls.species(sid)?;
        let level = level_hint.max(l.min_level).clamp(50, 55);
        let moves = self.heuristic_moves(dex, sid, level)?;
        let species_name = dex.species.get(dex.species.id(sid)?).name.clone();
        let item_cands: Vec<&String> =
            self.pool_items.iter().filter(|it| !held.contains(&toid(it))).collect();
        let item = if item_cands.is_empty() {
            String::new()
        } else {
            item_cands[rng.below(item_cands.len())].clone()
        };
        Some(json!({
            "name": species_name,
            "species": species_name,
            "item": item,
            "ability": "No Ability",
            "moves": moves,
            "nature": "Serious",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "level": level,
        }))
    }

    /// Item Clause repair: when a derived set's item is already held by a
    /// teammate, redraw from the pool items (or go itemless).
    fn repair_item_clause(&self, set: &mut Value, held: &[String], rng: &mut SplitMix64) {
        let item = toid(set["item"].as_str().unwrap_or(""));
        if item.is_empty() || !held.contains(&item) {
            return;
        }
        let cands: Vec<&String> =
            self.pool_items.iter().filter(|it| !held.contains(&toid(it))).collect();
        set["item"] = if cands.is_empty() {
            json!("")
        } else {
            json!(cands[rng.below(cands.len())])
        };
    }

    /// Deterministic default moveset: up to 3 damaging moves greedy by
    /// `base_power × STAB` with type dedupe, one utility move from the
    /// preference list, remaining slots filled by the next-best damaging
    /// moves. Falls back to the first learnset moves when the species has
    /// no plain damaging move (Unown, Ditto).
    fn heuristic_moves(&self, dex: &Dex, sid: &str, level: i64) -> Option<Vec<String>> {
        let l = self.ls.species(sid)?;
        let sp_types = &dex.species.get(dex.species.id(sid)?).types;
        let usable = |m: &String| {
            dex.moves.id(m).is_some()
                && l.move_min_level.get(m.as_str()).is_none_or(|&mn| level >= mn)
        };
        let mut dmg: Vec<(f64, &String)> = l
            .moves
            .iter()
            .filter(|m| usable(m))
            .filter_map(|m| {
                let md = dex.moves.get(dex.moves.id(m)?);
                if md.category == "Status" || md.base_power == 0 {
                    return None;
                }
                let stab = if sp_types.contains(&md.move_type) { 1.5 } else { 1.0 };
                Some((md.base_power as f64 * stab, m))
            })
            .collect();
        dmg.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(b.1)));

        let mut picked: Vec<String> = Vec::new();
        let mut used_types: Vec<&str> = Vec::new();
        for (_, m) in &dmg {
            if picked.len() >= 3 {
                break;
            }
            let t = dex.moves.get(dex.moves.id(m).unwrap()).move_type.as_str();
            if !used_types.contains(&t) {
                used_types.push(t);
                picked.push((*m).clone());
            }
        }
        for u in UTILITY_MOVES {
            if picked.len() >= 4 {
                break;
            }
            let id = u.to_string();
            if l.allows(u) && usable(&id) && !picked.contains(&id) {
                picked.push(id);
                break;
            }
        }
        for (_, m) in &dmg {
            if picked.len() >= 4 {
                break;
            }
            if !picked.contains(m) {
                picked.push((*m).clone());
            }
        }
        if picked.is_empty() {
            picked = l.moves.iter().filter(|m| usable(m)).take(4).cloned().collect();
        }
        if picked.is_empty() {
            return None;
        }
        Some(picked.iter().map(|m| move_name(dex, m)).collect())
    }

    // --------------------------------------------------- research helpers

    /// Deliberate weakening for smoke validation (proves the hill-climb has
    /// signal, not strength): strip every item, then downgrade the strongest
    /// damaging move of `downgrades` random slots to the weakest legal
    /// damaging move — all through the same canonize/validate pipeline as
    /// the mutation operators.
    pub fn weaken(
        &self,
        dex: &Dex,
        team: &[Value],
        rng: &mut SplitMix64,
        downgrades: usize,
    ) -> Option<Vec<Value>> {
        let mut raw = team.to_vec();
        for set in raw.iter_mut() {
            set["item"] = json!("");
        }
        for _ in 0..downgrades {
            let slot = rng.below(6);
            let sid = species_id(&raw[slot]);
            let Some(l) = self.ls.species(&sid) else { continue };
            let level = set_level(&raw[slot]);
            let sp_types = &dex.species.get(dex.species.id(&sid)?).types;
            let score = |m: &str| -> Option<f64> {
                let md = dex.moves.get(dex.moves.id(m).or_else(|| dex.moves.id(&base_id(m)))?);
                if md.category == "Status" || md.base_power == 0 {
                    return None;
                }
                let stab = if sp_types.contains(&md.move_type) { 1.5 } else { 1.0 };
                Some(md.base_power as f64 * stab)
            };
            let cur = move_ids(&raw[slot]);
            let cur_bases: Vec<String> = cur.iter().map(|m| base_id(m)).collect();
            // strongest current damaging move
            let Some((target, _)) = cur
                .iter()
                .enumerate()
                .filter_map(|(i, m)| score(m).map(|s| (i, s)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            else {
                continue;
            };
            // weakest legal damaging replacement not already carried
            let mut weak: Vec<(f64, &String)> = l
                .moves
                .iter()
                .filter(|m| {
                    !cur_bases.contains(m)
                        && dex.moves.id(m).is_some()
                        && l.move_min_level.get(m.as_str()).is_none_or(|&mn| level >= mn)
                })
                .filter_map(|m| score(m).map(|s| (s, m)))
                .collect();
            weak.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)));
            if let Some((_, m)) = weak.first() {
                raw[slot]["moves"][target] = json!(move_name(dex, m));
            }
        }
        self.canonize(dex, &raw)
    }

    /// One attempt at a random legal team: 6 distinct species on the
    /// 55/55/50/50/50/50 level template (species floors > 50 claim the 55
    /// slots), sets derived like `SpeciesSwap`. `None` = draw failed;
    /// retry with `random_team_valid`.
    pub fn random_team(&self, dex: &Dex, rng: &mut SplitMix64) -> Option<Vec<Value>> {
        let mut sids: Vec<String> = Vec::new();
        for _ in 0..40 {
            if sids.len() == 6 {
                break;
            }
            let s = self.species_ids[rng.below(self.species_ids.len())].clone();
            if !sids.contains(&s) {
                sids.push(s);
            }
        }
        if sids.len() != 6 {
            return None;
        }
        // species floors descending onto the 55,55,50,50,50,50 template
        sids.sort_by(|a, b| {
            let fa = self.ls.species(a).map_or(50, |l| l.min_level);
            let fb = self.ls.species(b).map_or(50, |l| l.min_level);
            fb.cmp(&fa).then(a.cmp(b))
        });
        const LEVELS: [i64; 6] = [55, 55, 50, 50, 50, 50];
        let mut raw: Vec<Value> = Vec::new();
        let mut held: Vec<String> = Vec::new();
        for (i, sid) in sids.iter().enumerate() {
            if self.ls.species(sid).map_or(50, |l| l.min_level) > LEVELS[i] {
                return None; // more than two floor-55 species drawn
            }
            let mut set = self.derive_set(dex, sid, LEVELS[i], &held, rng)?;
            set["level"] = json!(LEVELS[i]);
            let item = toid(set["item"].as_str().unwrap_or(""));
            if !item.is_empty() {
                held.push(item);
            }
            raw.push(set);
        }
        self.canonize(dex, &raw)
    }

    pub fn random_team_valid(
        &self,
        dex: &Dex,
        rng: &mut SplitMix64,
        tries: usize,
    ) -> Option<Vec<Value>> {
        (0..tries).find_map(|_| self.random_team(dex, rng))
    }
}

// -------------------------------------------------------- JSON set helpers

fn species_id(set: &Value) -> String {
    toid(set["species"].as_str().unwrap_or(""))
}

fn set_level(set: &Value) -> i64 {
    match set["level"].as_i64() {
        Some(l) if l != 0 => l,
        _ => 55,
    }
}

fn move_ids(set: &Value) -> Vec<String> {
    set["moves"]
        .as_array()
        .map(|a| a.iter().filter_map(|m| m.as_str()).map(toid).collect())
        .unwrap_or_default()
}

/// Typed Hidden Power collapses onto the base id for learnset checks.
fn base_id(move_id: &str) -> String {
    if move_id.starts_with("hiddenpower") && move_id != "hiddenpower" {
        "hiddenpower".into()
    } else {
        move_id.into()
    }
}

fn move_name(dex: &Dex, id: &str) -> String {
    match dex.moves.id(id) {
        Some(mid) => dex.moves.get(mid).name.clone(),
        None => id.to_string(),
    }
}

/// Item ids held by every slot except `slot` (Item Clause universe).
fn held_items(raw: &[Value], slot: usize) -> Vec<String> {
    (0..raw.len())
        .filter(|&i| i != slot)
        .map(|i| toid(raw[i]["item"].as_str().unwrap_or("")))
        .filter(|it| !it.is_empty())
        .collect()
}

fn stat6(set: &Value, key: &str, default: i64) -> [i64; 6] {
    const KEYS: [&str; 6] = ["hp", "atk", "def", "spa", "spd", "spe"];
    let mut out = [default; 6];
    if let Some(m) = set[key].as_object() {
        for (i, k) in KEYS.iter().enumerate() {
            if let Some(v) = m.get(*k).and_then(|v| v.as_i64()) {
                out[i] = v;
            }
        }
    }
    out
}

fn stat_obj(v: [i64; 6]) -> Value {
    json!({"hp": v[0], "atk": v[1], "def": v[2], "spa": v[3], "spd": v[4], "spe": v[5]})
}

fn mut_dv_tweak(raw: &mut [Value], slot: usize, rng: &mut SplitMix64) {
    let mut ivs = stat6(&raw[slot], "ivs", 31);
    let dv = rng.below(16) as i64;
    match rng.below(4) {
        0 => ivs[1] = dv * 2,                       // Atk
        1 => ivs[2] = dv * 2,                       // Def
        2 => ivs[5] = dv * 2,                       // Spe
        _ => (ivs[3], ivs[4]) = (dv * 2, dv * 2),   // Spc (SpA=SpD mirror)
    }
    // HP DV / gender / shiny re-derived by the canonicalizer.
    raw[slot]["ivs"] = stat_obj(ivs);
}

fn mut_ev_tweak(raw: &mut [Value], slot: usize, rng: &mut SplitMix64) {
    let mut evs = stat6(&raw[slot], "evs", 255);
    let val = [0i64, 64, 128, 192, 255][rng.below(5)];
    match rng.below(5) {
        0 => evs[0] = val,                       // HP
        1 => evs[1] = val,                       // Atk
        2 => evs[2] = val,                       // Def
        3 => (evs[3], evs[4]) = (val, val),      // Spc
        _ => evs[5] = val,                       // Spe
    }
    raw[slot]["evs"] = stat_obj(evs);
}

/// Stable identity of a canonical team (both sides always come from
/// `canonize`, so key order is consistent).
pub fn team_key(team: &[Value]) -> String {
    serde_json::to_string(team).unwrap()
}

/// Canonical team JSON -> engine sets (extra canonical-only fields like
/// `nature` are ignored by `PokemonSet`).
pub fn to_sets(team: &[Value]) -> Result<Vec<PokemonSet>, String> {
    serde_json::from_value(Value::Array(team.to_vec())).map_err(|e| e.to_string())
}

// ------------------------------------------------------------------ fitness

#[derive(Clone, Debug)]
pub struct EvalCfg {
    /// Seed-paired games per gauntlet member (rounded up to even).
    pub games_per_opponent: u32,
    /// skuct iterations per decision, both sides (preview = live search).
    pub agent_iters: u32,
    pub max_turns: u16,
    pub threads: usize,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct EvalResult {
    /// Candidate mean score over every game (win 1 / tie 0.5 / loss 0).
    pub score: f64,
    pub per_opponent: Vec<f64>,
    pub games: usize,
}

struct EvalJob {
    opp: usize,
    battle_seed: String,
    cand_seed: u64,
    opp_seed: u64,
    cand_is_p1: bool,
}

/// Candidate fitness vs a gauntlet: seed-paired skuct-vs-skuct duels (both
/// orientations on the same battle seed), full games from team preview.
/// Deterministic for a given `cfg.seed` at any `cfg.threads`.
pub fn gauntlet_eval(
    dex: &Dex,
    candidate: &[PokemonSet],
    gauntlet: &[Vec<PokemonSet>],
    cfg: &EvalCfg,
) -> EvalResult {
    let games = (cfg.games_per_opponent + cfg.games_per_opponent % 2) as usize;
    let mut jobs = Vec::with_capacity(gauntlet.len() * games);
    for opp in 0..gauntlet.len() {
        let mut rng =
            SplitMix64::new(cfg.seed ^ (opp as u64 + 1).wrapping_mul(0x9FB2_1C65_1E98_DF25));
        let mut battle_seed = String::new();
        for g in 0..games {
            if g % 2 == 0 {
                battle_seed = rng.battle_seed();
            }
            jobs.push(EvalJob {
                opp,
                battle_seed: battle_seed.clone(),
                cand_seed: rng.next(),
                opp_seed: rng.next(),
                cand_is_p1: g % 2 == 0,
            });
        }
    }

    let cursor = AtomicUsize::new(0);
    let mut results: Vec<(usize, f64)> = Vec::with_capacity(jobs.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..cfg.threads.max(1) {
            let (jobs, cursor) = (&jobs, &cursor);
            handles.push(scope.spawn(move || {
                let mut out: Vec<(usize, (usize, f64))> = Vec::new();
                loop {
                    let i = cursor.fetch_add(1, Ordering::Relaxed);
                    if i >= jobs.len() {
                        break;
                    }
                    let job = &jobs[i];
                    let skuct = |seed: u64| {
                        RmAgent::new(
                            RmConfig {
                                iterations: cfg.agent_iters,
                                rule: SelRule::Ucb,
                                ..Default::default()
                            },
                            seed,
                        )
                    };
                    let mut cand_agent = skuct(job.cand_seed);
                    let mut opp_agent = skuct(job.opp_seed);
                    let (t1, t2) = if job.cand_is_p1 {
                        (candidate, gauntlet[job.opp].as_slice())
                    } else {
                        (gauntlet[job.opp].as_slice(), candidate)
                    };
                    let mut b = Battle::from_fixture(dex, &job.battle_seed, t1, t2).unwrap();
                    b.set_log_enabled(false);
                    let res = if job.cand_is_p1 {
                        play_game(dex, &mut b, &mut [&mut cand_agent, &mut opp_agent], cfg.max_turns)
                    } else {
                        play_game(dex, &mut b, &mut [&mut opp_agent, &mut cand_agent], cfg.max_turns)
                    }
                    .unwrap();
                    let p1_score = match res {
                        GameResult::Outcome(Outcome::P1Win) => 1.0,
                        GameResult::Outcome(Outcome::P2Win) => 0.0,
                        GameResult::Outcome(Outcome::Tie) | GameResult::TurnCapped => 0.5,
                    };
                    let score = if job.cand_is_p1 { p1_score } else { 1.0 - p1_score };
                    out.push((i, (job.opp, score)));
                }
                out
            }));
        }
        let mut all: Vec<(usize, (usize, f64))> = Vec::new();
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all.sort_by_key(|r| r.0);
        results = all.into_iter().map(|(i, (_, s))| (jobs[i].opp, s)).collect();
    });

    // job-ordered summation -> bit-stable means at any thread count
    let mut per: Vec<(f64, usize)> = vec![(0.0, 0); gauntlet.len()];
    let mut total = 0.0;
    for &(opp, s) in &results {
        per[opp].0 += s;
        per[opp].1 += 1;
        total += s;
    }
    EvalResult {
        score: total / results.len() as f64,
        per_opponent: per.iter().map(|&(s, n)| s / n.max(1) as f64).collect(),
        games: results.len(),
    }
}
