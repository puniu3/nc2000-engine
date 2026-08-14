//! Hand-entered positions: a complete, human-writable serialization of the
//! M15 importer's public information set.
//!
//! The solver screen lets a player describe ONE decision point — their own
//! six sets exactly, the opponent by public facts only — and asks the bot to
//! score it. That is precisely the information `ProtocolTracker` accumulates
//! from a live protocol stream, so this module does not build a second
//! reconstruction path: it serializes the tracker (plus the `Observer`
//! reveal channel, which the tracker deliberately does not carry) and hands
//! the result to the same `synthesize` -> `BlindSearch` pipeline the ladder
//! bot runs on every move.
//!
//! Two consequences worth stating, because they are the reason this shape
//! was chosen over compiling synthetic protocol lines in the browser:
//!
//! - **Round-trippable.** `ProtocolTracker::to_spec` / `from_spec` are
//!   inverses over every field, so a real ladder position can be exported to
//!   a spec and re-imported, and the two synthesized battles must be
//!   identical. `crates/bot/tests/position.rs` proves exactly that over the
//!   conformance corpus — an assertion no line-fabricating path could make.
//! - **No protocol vocabulary outside `import.rs`.** `push_line` handles ~30
//!   line kinds with rules (PP deduction on charge turns, Baton Pass boost
//!   carry, companion-volatile lifecycles) that exist in one place; a form
//!   that emitted protocol would be a second, weaker implementation of them.
//!
//! Ids cross the boundary as **keys, not indices** (`"thunderbolt"`, not a
//! `MoveId`): the JSON is meant to be hand-edited, stored, and shared, and a
//! dex reordering must not silently reinterpret a saved position.

use nc2000_engine::battle::PokemonSet;
use nc2000_engine::dex::{toid, Dex};
use nc2000_engine::state::{Battle, Status};
use serde::{Deserialize, Serialize};

use crate::belief::Belief;
use crate::import::{ProtocolTracker, Request};
use crate::observe::Observer;
use crate::preview::MetaPool;
use crate::rng::SplitMix64;

/// Schema tag stamped on every spec this build writes, and the only one it
/// accepts. Bump on any incompatible field change — a saved position that
/// silently reinterprets is worse than one that refuses to load.
pub const SCHEMA: &str = "nc2000-position-v1";

/// Fixed battle seed for the throwaway `from_fixture` used to derive own-side
/// max HP and stats. Nothing about the analyzed position depends on it: the
/// synthesized battle is reseeded by `synthesize`, and genders are stamped
/// from the spec afterwards.
const STAT_SEED: &str = "1,2,3,4";

/// One decision point, described entirely in public terms plus the analyzing
/// side's own (private, exact) sets.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PositionSpec {
    /// Must equal [`SCHEMA`].
    pub schema: String,
    /// The side being solved for (0 = p1). Its sets are known exactly.
    pub side: usize,
    pub turn: u16,
    /// The turn's residual phase has already run (suppresses the synthetic
    /// `Residual` queue entry on a mid-turn switch request).
    #[serde(default)]
    pub upkeep_this_turn: bool,
    /// The analyzing side's six sets, in team-preview (`|poke|`) order.
    pub own_sets: Vec<PokemonSet>,
    /// Absolute side order: `sides[0]` is p1 whatever `side` is.
    pub sides: [SideSpec; 2],
    #[serde(default)]
    pub weather: Option<WeatherSpec>,
    /// The decision point is team preview: no board yet, the whole question
    /// is which three to bring and in what order.
    #[serde(default)]
    pub team_preview: bool,
    /// The request is a forced switch (the active fainted, or Baton Pass).
    #[serde(default)]
    pub force_switch: bool,
    /// The active cannot switch out (Mean Look / Wrap family).
    #[serde(default)]
    pub trapped: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeatherSpec {
    /// Condition key: `raindance` / `sunnyday` / `sandstorm` / `hail`.
    pub key: String,
    /// Upkeeps seen since it was set (the duration is back-dated by this).
    #[serde(default)]
    pub upkeeps: u16,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SideSpec {
    /// Roster order = team-preview order. Six entries in a normal position.
    pub mons: Vec<MonSpec>,
    /// Roster slot of the active mon.
    #[serde(default)]
    pub active: Option<usize>,
    /// Display order of the picked party as roster slots — for the analyzing
    /// side this is what its request shows (PS keeps the active at position
    /// 0), and `switch N` is indexed against it, so it is part of the
    /// position rather than something to re-derive. Empty = derive it
    /// (active first, then the mons that have appeared, then the rest).
    #[serde(default)]
    pub party: Vec<usize>,
    #[serde(default)]
    pub conditions: Vec<CondSpec>,
    /// A Baton Pass switch is pending on this side.
    #[serde(default)]
    pub pending_bp: bool,
    /// This side's action for the turn is already spent.
    #[serde(default)]
    pub acted_this_turn: bool,
    #[serde(default)]
    pub fainted_this_turn: Option<usize>,
    #[serde(default)]
    pub fainted_last_turn: Option<usize>,
    /// Stadium self-KO clause bookkeeping (`side.lastMove`).
    #[serde(default)]
    pub last_move: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CondSpec {
    /// `spikes` / `reflect` / `lightscreen` / `safeguard` / `mist`.
    pub key: String,
    /// Turn it was set (durations are back-dated from `turn`).
    #[serde(default)]
    pub start_turn: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolSpec {
    /// Engine condition key (`confusion`, `substitute`, `leechseed`, ...).
    pub key: String,
    #[serde(default)]
    pub start_turn: u16,
    /// encore/disable: the locked move; partiallytrapped: the binding move.
    #[serde(default, rename = "move")]
    pub move_key: Option<String>,
    /// `[side, slot]` of the volatile's source, when it matters.
    #[serde(default)]
    pub source: Option<[usize; 2]>,
    /// perishsong: remaining count.
    #[serde(default)]
    pub counter: Option<i64>,
}

/// A move and a count — PP marks (`uses`) and the thrash-lock progress.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UseSpec {
    #[serde(rename = "move")]
    pub move_key: String,
    pub n: i32,
}

/// The analyzing side's authoritative per-move PP, exactly as the request
/// states it. Kept apart from `uses` because the two are different truths:
/// `uses` is the public PP-mark channel (what an opponent can count), while
/// this is what the player reads off their own screen — and the two diverge
/// whenever PP moves without a visible use (Mystery Berry, Sleep Talk's
/// call, Mimic).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PpSpec {
    #[serde(rename = "move")]
    pub move_key: String,
    pub pp: i32,
    pub maxpp: i32,
    #[serde(default)]
    pub disabled: bool,
}

/// What is publicly known about one opponent mon's item. `known == false`
/// is "no idea"; `known == true` with `item: null` is "known to hold
/// nothing" (consumed, or stolen from). Mirrors `observe::ItemObs`, which
/// keeps the two apart because set filtering and determinization read them
/// differently.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ItemKnowledge {
    #[serde(default)]
    pub known: bool,
    #[serde(default)]
    pub item: Option<String>,
}

impl ItemKnowledge {
    pub fn unknown() -> ItemKnowledge {
        ItemKnowledge { known: false, item: None }
    }
    pub fn from_obs(v: Option<Option<&str>>) -> ItemKnowledge {
        match v {
            None => ItemKnowledge { known: false, item: None },
            Some(None) => ItemKnowledge { known: true, item: None },
            Some(Some(k)) => ItemKnowledge { known: true, item: Some(k.to_string()) },
        }
    }
    pub fn to_obs(&self) -> Option<Option<&str>> {
        if !self.known {
            None
        } else {
            Some(self.item.as_deref())
        }
    }
}

/// One roster mon. Every field is either publicly announced or (own side)
/// exactly known to the player typing it in; nothing here is imputed.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MonSpec {
    pub species: String,
    pub level: u8,
    /// `"M"` / `"F"` / `""` (genderless).
    #[serde(default)]
    pub gender: String,
    /// Nickname; `""` until the mon has appeared. Also the log-subject key.
    #[serde(default)]
    pub name: String,
    /// The `|poke|` preview line's item flag: it brought *an* item.
    #[serde(default)]
    pub item_flag: bool,
    #[serde(default)]
    pub appeared: bool,
    #[serde(default)]
    pub appear_count: i32,
    #[serde(default)]
    pub switch_in_turn: u16,
    #[serde(default)]
    pub active: bool,
    #[serde(default)]
    pub fainted: bool,
    /// Announced HP as `hp_num` / `hp_den` (100 = HP Percentage Mod, this
    /// format's stream; 48 = legacy pixel bar). The opponent's true HP is
    /// imputed to the midpoint of the bucket this admits.
    #[serde(default = "full_hp")]
    pub hp_num: i32,
    #[serde(default = "hp_den_default")]
    pub hp_den: i32,
    /// Own side only: exact current HP. The request is authoritative for the
    /// analyzing side, so this is what it carries; `None` falls back to the
    /// announced fraction.
    #[serde(default)]
    pub hp_exact: Option<i32>,
    /// `""` / `brn` / `par` / `slp` / `frz` / `psn` / `tox` / `fnt`.
    #[serde(default)]
    pub status: String,
    /// The sleep came from Rest (a public two-turn clock).
    #[serde(default)]
    pub rest: bool,
    /// Turns publicly slept through (`|cant|...|slp` count).
    #[serde(default)]
    pub slept: i32,
    /// Toxic counter (the Stadium `residualdmg` companion volatile).
    #[serde(default)]
    pub tox_counter: Option<i32>,
    /// Burn / paralysis stat-drop companions outlive their status here.
    #[serde(default)]
    pub comp_brn: bool,
    #[serde(default)]
    pub comp_par: bool,
    /// atk, def, spa, spd, spe, accuracy, evasion.
    #[serde(default)]
    pub boosts: [i8; 7],
    #[serde(default)]
    pub volatiles: Vec<VolSpec>,
    /// PP marks: how often each move has been publicly used.
    #[serde(default)]
    pub uses: Vec<UseSpec>,
    /// Own side: the request's per-move PP. Empty = derive from the set and
    /// the public use counts (the hand-entry default: "as printed, minus
    /// what has been seen").
    #[serde(default)]
    pub pp: Vec<PpSpec>,
    /// Own side: the item held *now*, `Some("")` for an empty hand. `None`
    /// falls back to the set's item — right until something consumes it.
    #[serde(default)]
    pub item_now: Option<String>,
    /// Thrash-class lock: (move, continuation turns spent).
    #[serde(default)]
    pub locked: Option<UseSpec>,
    /// Two-turn move charged and pending release.
    #[serde(default)]
    pub charging: Option<String>,
    #[serde(default)]
    pub must_recharge: bool,
    /// `[side, slot]` this mon is Transformed into.
    #[serde(default)]
    pub transformed_into: Option<[usize; 2]>,
    /// The move Mimic wrote into its slot.
    #[serde(default)]
    pub mimic_overlay: Option<String>,
    #[serde(default)]
    pub last_move: Option<String>,
    /// Protect/Endure consecutive-use counter.
    #[serde(default)]
    pub stall_streak: i32,
    #[serde(default)]
    pub last_protect_turn: u16,
    #[serde(default)]
    pub protected_this_turn: bool,
    // ---- Observer channel (opponent side only; ignored for `side`)
    /// Moves publicly known to be in this mon's set.
    #[serde(default)]
    pub revealed_moves: Vec<String>,
    /// The item it *brought* (what set filtering matches against).
    #[serde(default)]
    pub item_original: ItemKnowledge,
    /// The item it holds *now* (the determinizer pins the true field when
    /// this is known).
    #[serde(default)]
    pub item_current: ItemKnowledge,
    /// A Thief steal was seen, so later reveals no longer identify the
    /// original item.
    #[serde(default)]
    pub item_gained: bool,
}

fn full_hp() -> i32 {
    100
}
fn hp_den_default() -> i32 {
    100
}

impl Default for MonSpec {
    fn default() -> MonSpec {
        MonSpec {
            species: String::new(),
            level: 50,
            gender: String::new(),
            name: String::new(),
            item_flag: false,
            appeared: false,
            appear_count: 0,
            switch_in_turn: 0,
            active: false,
            fainted: false,
            hp_num: 100,
            hp_den: 100,
            hp_exact: None,
            status: String::new(),
            rest: false,
            slept: 0,
            tox_counter: None,
            comp_brn: false,
            comp_par: false,
            boosts: [0; 7],
            volatiles: Vec::new(),
            uses: Vec::new(),
            pp: Vec::new(),
            item_now: None,
            locked: None,
            charging: None,
            must_recharge: false,
            transformed_into: None,
            mimic_overlay: None,
            last_move: None,
            stall_streak: 0,
            last_protect_turn: 0,
            protected_this_turn: false,
            revealed_moves: Vec::new(),
            item_original: ItemKnowledge::unknown(),
            item_current: ItemKnowledge::unknown(),
            item_gained: false,
        }
    }
}

impl PositionSpec {
    pub fn parse(json: &str) -> Result<PositionSpec, String> {
        let spec: PositionSpec =
            serde_json::from_str(json).map_err(|e| format!("position parse: {e}"))?;
        spec.check()?;
        Ok(spec)
    }

    /// Structural checks that must hold before any of this reaches the
    /// engine. Deliberately fail-closed and specific: this is the one place
    /// a hand-typed position is rejected, and "it silently analyzed a
    /// different board" is the failure mode worth spending errors on.
    pub fn check(&self) -> Result<(), String> {
        if self.schema != SCHEMA {
            return Err(format!("schema: expected {SCHEMA}, got {}", self.schema));
        }
        if self.side > 1 {
            return Err(format!("side: {} (must be 0 or 1)", self.side));
        }
        if self.own_sets.len() != self.sides[self.side].mons.len() {
            return Err(format!(
                "own_sets ({}) must align with sides[{}].mons ({})",
                self.own_sets.len(),
                self.side,
                self.sides[self.side].mons.len()
            ));
        }
        for (s, side) in self.sides.iter().enumerate() {
            if side.mons.is_empty() {
                return Err(format!("sides[{s}].mons is empty"));
            }
            if let Some(a) = side.active {
                if a >= side.mons.len() {
                    return Err(format!("sides[{s}].active {a} out of range"));
                }
            }
            let actives = side.mons.iter().filter(|m| m.active).count();
            if actives > 1 {
                return Err(format!("sides[{s}] has {actives} active mons"));
            }
            match (side.active, side.mons.iter().position(|m| m.active)) {
                (Some(a), Some(b)) if a != b => {
                    return Err(format!("sides[{s}].active {a} disagrees with mons[{b}].active"))
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(format!("sides[{s}].active disagrees with the mons' flags"))
                }
                _ => {}
            }
            for (i, m) in side.mons.iter().enumerate() {
                if m.hp_den <= 0 {
                    return Err(format!("sides[{s}].mons[{i}].hp_den must be positive"));
                }
                if m.hp_num < 0 || m.hp_num > m.hp_den {
                    return Err(format!(
                        "sides[{s}].mons[{i}].hp_num {} out of 0..={}",
                        m.hp_num, m.hp_den
                    ));
                }
                if m.fainted && m.hp_num != 0 {
                    return Err(format!("sides[{s}].mons[{i}] is fainted with HP left"));
                }
                if !m.fainted && m.hp_num == 0 {
                    return Err(format!("sides[{s}].mons[{i}] is at 0 HP but not fainted"));
                }
                if m.fainted && !m.appeared {
                    return Err(format!(
                        "sides[{s}].mons[{i}] has fainted without ever appearing"
                    ));
                }
                if m.active && m.fainted && !self.force_switch && !self.team_preview {
                    return Err(format!(
                        "sides[{s}].mons[{i}] is the fainted active outside a switch request"
                    ));
                }
            }
        }
        if self.sides[self.side].active.is_none() && !self.force_switch && !self.team_preview {
            return Err("the analyzing side has no active mon and no switch request".to_string());
        }
        Ok(())
    }

    /// The opponent's side index.
    pub fn foe(&self) -> usize {
        1 - self.side
    }

    /// Max HP of each of the analyzing side's mons, in roster order — the
    /// numbers the request's `condition` strings are written against.
    /// Derived by constructing the sets through the real engine, so DVs,
    /// stat exp and the level all land exactly as they will in the battle.
    pub fn own_maxhps(&self, dex: &Dex) -> Result<Vec<i32>, String> {
        let b = Battle::from_fixture(dex, STAT_SEED, &self.own_sets, &self.own_sets)
            .map_err(|e| format!("own sets: {e:?}"))?;
        Ok(b.sides[0].roster.iter().map(|p| p.maxhp).collect())
    }

    /// The PS-shaped `|request|` JSON this position implies. Same schema the
    /// live client receives from the server and the corpus harness fabricates
    /// (`corpus.rs`), so `Request::parse` needs no special case.
    pub fn request_json(&self, dex: &Dex) -> Result<String, String> {
        let me = &self.sides[self.side];
        let maxhps = self.own_maxhps(dex)?;
        let picked = self.own_party()?;
        let active_slot = me.active;

        let mut req_moves: Vec<serde_json::Value> = Vec::new();
        if !self.force_switch && !self.team_preview {
            let slot = active_slot.ok_or("no active mon for a move request")?;
            let mon = &me.mons[slot];
            for row in self.own_move_rows(dex, slot)? {
                req_moves.push(serde_json::json!({
                    "id": plain(&toid(&row.move_key)),
                    "move": row.move_key,
                    "pp": row.pp,
                    "maxpp": row.maxpp,
                    "target": "normal",
                    "disabled": row.disabled,
                }));
            }
            let _ = mon;
        }

        let req_mons: Vec<serde_json::Value> = picked
            .iter()
            .map(|&i| {
                let m = &me.mons[i];
                let maxhp = maxhps[i];
                let cond = if m.fainted {
                    "0 fnt".to_string()
                } else {
                    let hp = m.own_hp(maxhp);
                    match m.status.as_str() {
                        "" | "fnt" => format!("{hp}/{maxhp}"),
                        st => format!("{hp}/{maxhp} {st}"),
                    }
                };
                let nick = if m.name.is_empty() { species_name(dex, &m.species) } else { m.name.clone() };
                serde_json::json!({
                    "ident": format!("p{}: {}", self.side + 1, nick),
                    "details": details_of(dex, m),
                    "condition": cond,
                    "active": Some(i) == active_slot,
                    "item": m
                        .item_now
                        .clone()
                        .unwrap_or_else(|| toid(&self.own_sets[i].item)),
                })
            })
            .collect();

        let mut req = serde_json::json!({
            "side": {
                "name": format!("p{}", self.side + 1),
                "id": format!("p{}", self.side + 1),
                "pokemon": req_mons,
            },
            "rqid": self.turn as u64,
        });
        if self.team_preview {
            req["teamPreview"] = serde_json::json!(true);
            req["maxChosenTeamSize"] = serde_json::json!(PICKS);
        } else if self.force_switch {
            req["forceSwitch"] = serde_json::json!([true]);
        } else {
            req["active"] = serde_json::json!([{"moves": req_moves, "trapped": self.trapped}]);
        }
        Ok(req.to_string())
    }

    /// The analyzing side's active move rows: whatever the position states,
    /// else the set's four moves at full PP minus their public use counts,
    /// with Encore/Disable folded into `disabled`.
    pub fn own_move_rows(&self, dex: &Dex, slot: usize) -> Result<Vec<PpSpec>, String> {
        let mon = &self.sides[self.side].mons[slot];
        if !mon.pp.is_empty() {
            return Ok(mon.pp.clone());
        }
        let set = self
            .own_sets
            .get(slot)
            .ok_or_else(|| format!("no set for own roster slot {slot}"))?;
        let encore = mon.volatile("encore").and_then(|v| v.move_key.clone());
        let disable = mon.volatile("disable").and_then(|v| v.move_key.clone());
        let mut out = Vec::new();
        for name in &set.moves {
            let key = toid(name);
            let id = dex.moves.id(&key).ok_or_else(|| format!("unknown move `{name}`"))?;
            let maxpp = max_pp(dex, id);
            let disabled = match (&encore, &disable) {
                (Some(em), _) => plain(&toid(em)) != plain(&key),
                (None, Some(dm)) => plain(&toid(dm)) == plain(&key),
                _ => false,
            };
            out.push(PpSpec {
                pp: (maxpp - mon.uses_of(dex, &key)).max(0),
                maxpp,
                disabled,
                move_key: name.clone(),
            });
        }
        Ok(out)
    }

    /// The analyzing side's picked party in display order. Stated by the
    /// position when it knows it (a real request does); otherwise derived —
    /// active first, then the mons that have appeared, then never-appeared
    /// picks, the same canonical order `synthesize` arranges the opponent by.
    pub fn own_party(&self) -> Result<Vec<usize>, String> {
        let me = &self.sides[self.side];
        if self.team_preview {
            // A preview request lists the whole roster, in roster order —
            // `team 5, 6, 1` is indexed against that, not against a party.
            return Ok((0..me.mons.len()).collect());
        }
        if !me.party.is_empty() {
            if let Some(&bad) = me.party.iter().find(|&&i| i >= me.mons.len()) {
                return Err(format!("party slot {bad} out of range"));
            }
            return Ok(me.party.clone());
        }
        let mut out: Vec<usize> = Vec::new();
        if let Some(a) = me.active {
            out.push(a);
        }
        for (i, m) in me.mons.iter().enumerate() {
            if m.appeared && Some(i) != me.active {
                out.push(i);
            }
        }
        for (i, m) in me.mons.iter().enumerate() {
            if out.len() >= PICKS {
                break;
            }
            if !m.appeared && !out.contains(&i) {
                out.push(i);
            }
        }
        if out.is_empty() {
            return Err("the analyzing side has no picked mons".to_string());
        }
        out.truncate(PICKS);
        Ok(out)
    }
}

/// The dex display name of a species key, or the key itself when the dex has
/// never heard of it (the caller's own error to report, not this one's).
pub fn species_name(dex: &Dex, key: &str) -> String {
    dex.species
        .id(&toid(key))
        .map(|s| dex.species.get(s).name.clone())
        .unwrap_or_else(|| key.to_string())
}

/// `"Species, L55, M"` — the protocol details string, with the species'
/// display name (what PS puts on the wire).
pub fn details_of(dex: &Dex, m: &MonSpec) -> String {
    let mut s = format!("{}, L{}", species_name(dex, &m.species), m.level);
    if !m.gender.is_empty() {
        s.push_str(", ");
        s.push_str(&m.gender);
    }
    s
}

/// NC2000 brings three of six.
const PICKS: usize = 3;

impl SideSpec {
    pub fn active_mon(&self) -> Option<&MonSpec> {
        self.active.and_then(|a| self.mons.get(a))
    }
}

impl MonSpec {
    pub fn volatile(&self, key: &str) -> Option<&VolSpec> {
        self.volatiles.iter().find(|v| v.key == key)
    }

    /// Public use count of `key`, matching PS's plain-`hiddenpower` naming.
    pub fn uses_of(&self, _dex: &Dex, key: &str) -> i32 {
        let want = plain(key);
        self.uses
            .iter()
            .find(|u| plain(&toid(&u.move_key)) == want)
            .map(|u| u.n)
            .unwrap_or(0)
    }

    /// `"Species, L55, M"` — the protocol details string.
    pub fn details(&self) -> String {
        let mut s = format!("{}, L{}", self.species, self.level);
        if !self.gender.is_empty() {
            s.push_str(", ");
            s.push_str(&self.gender);
        }
        s
    }

    /// The analyzing side's exact current HP for the request's `condition`.
    pub fn own_hp(&self, maxhp: i32) -> i32 {
        match self.hp_exact {
            Some(hp) => hp.clamp(1, maxhp),
            None => {
                let frac = self.hp_num as f64 / self.hp_den.max(1) as f64;
                ((frac * maxhp as f64).round() as i32).clamp(1, maxhp)
            }
        }
    }

    pub fn status_enum(&self) -> Status {
        Status::from_str(&self.status)
    }
}

/// PS normalizes typed hidden powers to the plain id in requests and in
/// `|move|` lines, so any comparison against a set's move must fold them.
pub fn plain(key: &str) -> &str {
    if key.starts_with("hiddenpower") {
        "hiddenpower"
    } else {
        key
    }
}

/// Max PP under this format's fixed 3 PP Ups — the same rule the engine
/// applies when it builds a set (`battle/mod.rs`, `new_pokemon`).
pub fn max_pp(dex: &Dex, id: nc2000_engine::dex::MoveId) -> i32 {
    let ms = dex.move_static(id);
    let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
    let mut maxpp = ms.pp * (5 + pp_ups) / 5;
    if ms.pp == 40 {
        maxpp -= pp_ups;
    }
    maxpp
}

/// Build the battle a position describes — the same synthesis the live path
/// runs, without a searcher on top. Two callers: the "this is what the solver
/// understood" readback a hand-entry screen owes its user, and the round-trip
/// gate, which needs an explicit `seed` because synthesis rolls the hidden
/// durations (sleep counters, bind turns, substitute HP) the public channel
/// never announced.
///
/// `pinned` is open-team-sheet mode: the opponent's true sets, when they are
/// somehow known. Blind (`None`) is the solver's own information structure.
pub fn synthesize_spec(
    dex: &Dex,
    spec: &PositionSpec,
    pool: &MetaPool,
    pinned: Option<&[PokemonSet]>,
    seed: u64,
) -> Result<Battle, String> {
    spec.check()?;
    let tracker = ProtocolTracker::from_spec(dex, spec)?;
    let obs = Observer::from_position(dex, spec)?;
    let mut belief = match pinned {
        Some(sets) => Belief::pinned_checked(dex, "opponent", sets, &obs)?,
        None => Belief::new(dex, pool, &obs),
    };
    belief.sync_checked(dex, &obs)?;
    let req = Request::parse(dex, &spec.request_json(dex)?)?;
    let pick = belief.alive().first().copied();
    let mut rng = SplitMix64::new(seed);
    tracker.synthesize(dex, &spec.own_sets, belief.refs(pick), &obs, &req, &mut rng)
}
