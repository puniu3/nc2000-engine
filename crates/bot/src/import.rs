//! M15a protocol→state importer: reconstruct a `Battle` from PLAYER-VISIBLE
//! information only — the accumulated protocol lines of a PS-hosted battle
//! plus our side's request JSON — with the hidden opponent fields imputed
//! from an M10 belief source.
//!
//! This is the from-scratch cousin of `Belief::determinize`: the determinizer
//! overwrites a *known* true battle; here no true battle exists, so a
//! synthetic one is built that honors every public fact and leaves the
//! hidden fields to the belief. The division of labor maximizes reuse:
//!
//! - `ProtocolTracker` (new): parses the protocol vocabulary this engine
//!   itself emits (bit-exact with PS for this format, so the vocabulary is
//!   closed and known) into per-mon public state: HP (own exact / foe 1/48
//!   pixels), status (+ public counters: sleep turns, Rest, residual ticks),
//!   boosts (Baton Pass keeps them), announced volatiles, side conditions,
//!   weather (+ upkeep count), move usage counts for PP marks (charge-turn
//!   and locked-continuation `|move|` lines deduct no PP — mirrored from
//!   `run_move`'s deduction rule), picks/appearance/faint flow.
//! - `synthesize`: builds `Battle::from_fixture(own sets, belief refs)` and
//!   performs the state surgery: picks/positions (the M10 canonical party
//!   scheme), per-mon public fields, engine-API planting of statuses /
//!   volatiles / side conditions / weather (so companion state — Stadium
//!   residual counters, substitute HP, rolled hidden durations — comes from
//!   the same code the real engine runs), request bookkeeping via the
//!   engine's own `make_request`, and the mid-turn queue (pending foe move +
//!   residual) that `Belief::determinize`'s pending-move scrub expects.
//! - `Observer` (M10a, protocol mode): revealed moves + item knowledge from
//!   the same lines; `Belief` (M10a) filters candidates and provides the
//!   imputation refs; `BlindSearch` (M10c) then runs per-iteration
//!   determinization on the synthesized battle exactly as it does on a true
//!   one.
//!
//! Declared imputations (hidden by protocol, fixed at synthesis): foe HP
//! within its announced 1/48 bucket, hidden status/volatile counters (sleep
//! turns remaining, confusion/encore/disable/bind durations, substitute HP),
//! unrevealed sets/DVs (the belief's job), never-appeared pick identities
//! (resampled per iteration by the determinizer). Thrash/rollout lock
//! *inference* (never announced) is mechanics-based and calibrated against
//! the conformance corpus by `tests/import.rs`.

use nc2000_engine::battle::{EffectHandle, PokemonSet, SearchChoice};
use nc2000_engine::dex::{toid, Dex, MoveId, SpeciesId};
use nc2000_engine::state::{
    Action, ActionKind, Battle, Gender, MoveSlot, PokeId, PokeName, RequestState, Status,
    SwitchFlag, DK,
};

use crate::belief::Belief;
use crate::blind::BlindSearch;
use crate::observe::{move_matches, MonObs, Observer};
use crate::preview::{MetaPool, TableSet};
use crate::rng::SplitMix64;
use crate::smmcts::RmConfig;

/// NC2000 rule: Max Total Level = 155. Since the 2026-07-17 preview-space
/// fix the ENGINE enforces this at team preview (validation + enumeration;
/// certificate on `nc2000_engine::battle::MAX_TOTAL_LEVEL`), so the M15
/// root mask and the filtered table sampling below are redundant belt-and-
/// suspenders at preview — kept because they are harmless and still guard
/// table policies fed from external files.
pub const MAX_TOTAL_LEVEL: i32 = nc2000_engine::battle::MAX_TOTAL_LEVEL as i32;

// ===================================================================== lines

/// One mon's tracked public state.
#[derive(Clone, Debug)]
struct TrackMon {
    species: SpeciesId,
    level: u8,
    gender: Gender,
    preview_item: bool,
    /// Nickname, learned at first appearance ("" until then).
    name: String,
    appeared: bool,
    appear_count: i32,
    switch_in_turn: u16,
    active: bool,
    fainted: bool,
    /// Foe: last announced HP numerator (over `hp_den`). Own side is
    /// authoritative from the request, so pixels are only a fallback.
    pixels: i32,
    /// Denominator of the announced foe HP: 100 under HP Percentage Mod
    /// (this format), 48 on legacy pixel-bar streams.
    hp_den: i32,
    status: Status,
    /// Sleep came from Rest (public 2-turn counter).
    rest: bool,
    /// `|cant|...|slp` count since falling asleep.
    slept: i32,
    /// Stadium companion volatiles (public lifecycles of their own —
    /// created by their status, surviving cures/replacements, removed by
    /// boosting the dropped stat / Haze / switching):
    /// `residualdmg` = the Toxic counter (created by tox only; +1 per
    /// residual tick while brn/psn/tox; retained through Heal Bell/rests).
    comp_res: Option<i32>,
    comp_brn: bool,
    comp_par: bool,
    boosts: [i8; 7],
    vols: Vec<TVol>,
    /// PP-mark deductions by move id.
    uses: Vec<(MoveId, i32)>,
    /// Mechanics-inferred thrash/rollout lock: (move, uses so far).
    locked: Option<(MoveId, i32)>,
    /// Two-turn move charge pending release.
    charging: Option<MoveId>,
    must_recharge: bool,
    transformed_into: Option<(usize, usize)>,
    mimic_overlay: Option<MoveId>,
    last_move: Option<MoveId>,
    stall_streak: i32,
    last_protect_turn: u16,
    protected_this_turn: bool,
}

impl TrackMon {
    fn new(species: SpeciesId, level: u8, gender: Gender, item: bool) -> TrackMon {
        TrackMon {
            species,
            level,
            gender,
            preview_item: item,
            name: String::new(),
            appeared: false,
            appear_count: 0,
            switch_in_turn: 0,
            active: false,
            fainted: false,
            pixels: 48,
            hp_den: 48,
            status: Status::None,
            rest: false,
            slept: 0,
            comp_res: None,
            comp_brn: false,
            comp_par: false,
            boosts: [0; 7],
            vols: Vec::new(),
            uses: Vec::new(),
            locked: None,
            charging: None,
            must_recharge: false,
            transformed_into: None,
            mimic_overlay: None,
            last_move: None,
            stall_streak: 0,
            last_protect_turn: 0,
            protected_this_turn: false,
        }
    }

    fn deduct(&mut self, id: MoveId) {
        match self.uses.iter_mut().find(|(m, _)| *m == id) {
            Some((_, n)) => *n += 1,
            None => self.uses.push((id, 1)),
        }
    }

    /// Everything a switch-out clears (mirrors `clear_volatile`).
    fn clear_on_exit(&mut self) {
        self.boosts = [0; 7];
        self.vols.clear();
        self.locked = None;
        self.charging = None;
        self.must_recharge = false;
        self.transformed_into = None;
        self.mimic_overlay = None;
        self.last_move = None;
        self.stall_streak = 0;
        self.protected_this_turn = false;
        self.comp_res = None;
        self.comp_brn = false;
        self.comp_par = false;
    }
}

/// One announced volatile.
#[derive(Clone, Debug)]
struct TVol {
    /// Engine condition key ("confusion", "substitute", ...).
    key: String,
    start_turn: u16,
    /// encore/disable: the locked move; partiallytrapped: the binding move.
    move_id: Option<MoveId>,
    /// (side, slot) of the volatile's source, when it matters.
    source: Option<(usize, usize)>,
    /// perishsong: remaining count.
    counter: Option<i64>,
}

#[derive(Clone, Debug, Default)]
struct TrackSide {
    mons: Vec<TrackMon>,
    active: Option<usize>,
    /// (condition key, start turn).
    conds: Vec<(String, u16)>,
    pending_bp: bool,
    /// The side's action this turn is spent (move / cant / switch).
    acted_this_turn: bool,
    fainted_this_turn: Option<usize>,
    fainted_last_turn: Option<usize>,
    /// Last move by this side (Stadium self-KO clause bookkeeping).
    side_last_move: Option<MoveId>,
}

/// Player-visible protocol accumulator for one battle.
pub struct ProtocolTracker {
    /// Our side (0 = p1).
    side: usize,
    sides: [TrackSide; 2],
    /// (weather key, upkeeps seen since set).
    weather: Option<(String, u16)>,
    turn: u16,
    upkeep_this_turn: bool,
}

/// Locking (thrash-class) move keys: `lockedmove` continuation turns emit
/// plain `|move|` lines with NO PP deduction.
const THRASH_CLASS: [&str; 3] = ["thrash", "petaldance", "outrage"];
/// Binding move keys (`-activate |move: X|[of]` plants `partiallytrapped`).
const BIND_CLASS: [&str; 5] = ["wrap", "bind", "firespin", "clamp", "whirlpool"];

impl ProtocolTracker {
    pub fn new(side: usize) -> ProtocolTracker {
        ProtocolTracker {
            side,
            sides: [TrackSide::default(), TrackSide::default()],
            weather: None,
            turn: 0,
            upkeep_this_turn: false,
        }
    }

    pub fn turn(&self) -> u16 {
        self.turn
    }

    /// Opponent preview facts as M10a `MonObs` (feeds `Observer::from_mons`).
    pub(crate) fn observer_mons(&self) -> Vec<MonObs> {
        self.sides[1 - self.side]
            .mons
            .iter()
            .map(|m| MonObs {
                species: m.species,
                level: m.level,
                gender: m.gender,
                name: m.name.clone(),
                preview_has_item: m.preview_item,
                revealed_moves: Vec::new(),
                item: Default::default(),
                appeared: m.appeared,
            })
            .collect()
    }

    /// Nicknames currently known for `side`'s roster slots ("" = unknown).
    pub(crate) fn names(&self, side: usize) -> Vec<&str> {
        self.sides[side].mons.iter().map(|m| m.name.as_str()).collect()
    }

    pub fn opp_roster_len(&self) -> usize {
        self.sides[1 - self.side].mons.len()
    }

    // ------------------------------------------------------------ subjects

    /// "p2a: Nick" / "p2: Nick" → (side, roster slot) via known nicknames.
    fn subject(&self, s: &str) -> Option<(usize, usize)> {
        let b = s.as_bytes();
        if b.len() < 4 || b[0] != b'p' || (b[1] != b'1' && b[1] != b'2') {
            return None;
        }
        let side = (b[1] - b'1') as usize;
        let rest = &s[2..];
        let rest = rest.strip_prefix(|c: char| c.is_ascii_lowercase()).unwrap_or(rest);
        let name = rest.strip_prefix(": ")?;
        let slot = self.sides[side].mons.iter().position(|m| m.name == name)?;
        Some((side, slot))
    }

    /// "p1: P1" side-condition subject → side index.
    fn side_subject(s: &str) -> Option<usize> {
        let b = s.as_bytes();
        if b.len() >= 2 && b[0] == b'p' && (b[1] == b'1' || b[1] == b'2') {
            Some((b[1] - b'1') as usize)
        } else {
            None
        }
    }

    /// Parse "Species, L50, M" details.
    fn parse_details(dex: &Dex, details: &str) -> Option<(SpeciesId, u8, Gender)> {
        let mut species_name = details;
        let mut level = 100u8;
        let mut gender = Gender::N;
        for (i, part) in details.split(", ").enumerate() {
            if i == 0 {
                species_name = part;
                continue;
            }
            if let Some(l) = part.strip_prefix('L') {
                if let Ok(v) = l.parse() {
                    level = v;
                }
            } else if part == "M" {
                gender = Gender::M;
            } else if part == "F" {
                gender = Gender::F;
            }
        }
        let sid = dex.species.id(&toid(species_name))?;
        Some((sid, level, gender))
    }

    /// Parse an HP-status field: "137/211 par" / "31/48" / "0 fnt" →
    /// (current, max, status token or ""). Fainted → (0, 0, "fnt").
    fn parse_hp(s: &str) -> (i32, i32, &str) {
        let (hp_part, status) = match s.split_once(' ') {
            Some((h, st)) => (h, st),
            None => (s, ""),
        };
        if hp_part == "0" {
            return (0, 0, status);
        }
        match hp_part.split_once('/') {
            Some((c, m)) => (c.parse().unwrap_or(0), m.parse().unwrap_or(0), status),
            None => (hp_part.parse().unwrap_or(0), 0, status),
        }
    }

    fn apply_hp_status(&mut self, side: usize, slot: usize, field: &str) {
        let (cur, max, status) = Self::parse_hp(field);
        let mon = &mut self.sides[side].mons[slot];
        if status == "fnt" {
            mon.pixels = 0;
        } else {
            mon.pixels = cur; // own-side exact values are unused (request wins)
            if max > 0 {
                mon.hp_den = max;
            }
            let st = Status::from_str(status);
            if st != mon.status && !status.is_empty() {
                mon.status = st;
            } else if status.is_empty() && mon.status != Status::None {
                // HP strings always carry the status token when one exists;
                // its absence on a fresh announcement means it is gone.
                // (Curing is always separately announced too — harmless.)
                mon.status = Status::None;
            }
        }
    }

    // ------------------------------------------------------------ the feed

    /// Feed one player-visible protocol line.
    pub fn push_line(&mut self, dex: &Dex, line: &str) {
        let Some(body) = line.strip_prefix('|') else { return };
        let parts: Vec<&str> = body.split('|').collect();
        let cmd = *parts.first().unwrap_or(&"");
        let arg = |i: usize| parts.get(i).copied().unwrap_or("");
        match cmd {
            "poke" => {
                let Some(side) = Self::side_subject(arg(1)) else { return };
                if let Some((sp, lv, g)) = Self::parse_details(dex, arg(2)) {
                    self.sides[side].mons.push(TrackMon::new(sp, lv, g, arg(3) == "item"));
                }
            }
            "turn" => {
                self.turn = arg(1).parse().unwrap_or(self.turn + 1);
                self.upkeep_this_turn = false;
                for s in self.sides.iter_mut() {
                    s.acted_this_turn = false;
                    s.pending_bp = false;
                    s.fainted_last_turn = s.fainted_this_turn.take();
                    if let Some(a) = s.active {
                        let m = &mut s.mons[a];
                        if m.protected_this_turn {
                            m.stall_streak += 1;
                            m.last_protect_turn = self.turn - 1;
                            m.protected_this_turn = false;
                        } else if m.last_protect_turn + 1 < self.turn.max(1) {
                            m.stall_streak = 0;
                        }
                    }
                }
            }
            "upkeep" => self.upkeep_this_turn = true,
            "switch" | "drag" => self.on_switch(dex, &parts, cmd == "drag"),
            "faint" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    let turn = self.turn;
                    let side = &mut self.sides[s];
                    side.fainted_this_turn = Some(m);
                    let mon = &mut side.mons[m];
                    mon.fainted = true;
                    mon.pixels = 0;
                    mon.clear_on_exit();
                    let _ = turn;
                    self.clear_sourced_vols(s, m);
                }
            }
            "move" => self.on_move(dex, &parts),
            "cant" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.sides[s].acted_this_turn = true;
                    let mon = &mut self.sides[s].mons[m];
                    match arg(2) {
                        "slp" => {
                            mon.slept += 1;
                            mon.locked = None; // lockedmove drops during sleep
                            mon.charging = None;
                        }
                        "recharge" => {
                            mon.must_recharge = false;
                            mon.charging = None;
                        }
                        _ => {
                            mon.charging = None; // twoturnmove aborts on cant
                        }
                    }
                }
            }
            "-damage" | "-heal" | "-sethp" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.apply_hp_status(s, m, arg(2));
                    if parts.iter().any(|p| *p == "[from] psn" || *p == "[from] brn") {
                        let mon = &mut self.sides[s].mons[m];
                        if let Some(n) = mon.comp_res.as_mut() {
                            *n += 1;
                        }
                    }
                    // a confusion self-hit replaces the turn's move: a
                    // pending two-turn charge is lost (its release would
                    // otherwise stay imputed as the only legal choice)
                    if parts.iter().any(|p| *p == "[from] confusion") {
                        self.sides[s].mons[m].charging = None;
                    }
                }
            }
            "-status" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    let silent = parts.iter().any(|p| *p == "[silent]");
                    let mon = &mut self.sides[s].mons[m];
                    mon.status = Status::from_str(arg(2));
                    mon.slept = 0;
                    mon.rest = parts.iter().any(|p| *p == "[from] move: Rest");
                    if !silent {
                        // onStart companion creation (silent = the tox→psn
                        // switch-in rewrite, which runs no onStart)
                        match mon.status {
                            Status::Tox => mon.comp_res = Some(0),
                            Status::Brn => mon.comp_brn = true,
                            Status::Par => mon.comp_par = true,
                            _ => {}
                        }
                    }
                }
            }
            "-curestatus" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    cure(&mut self.sides[s].mons[m]);
                }
            }
            "-cureteam" => {
                if let Some((s, _)) = self.subject(arg(1)) {
                    for mon in self.sides[s].mons.iter_mut() {
                        if !mon.fainted {
                            cure(mon);
                        }
                    }
                }
            }
            "-boost" | "-unboost" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    if let Some(i) = boost_index(arg(2)) {
                        let d: i8 = arg(3).parse().unwrap_or(0);
                        let d = if cmd == "-unboost" { -d } else { d };
                        let mon = &mut self.sides[s].mons[m];
                        let b = &mut mon.boosts[i];
                        *b = (*b + d).clamp(-6, 6);
                        // any spe change while par (atk while brn) publicly
                        // removes the Stadium stat-drop companion volatile
                        if d != 0 && i == 4 && mon.status == Status::Par {
                            mon.comp_par = false;
                        }
                        if d != 0 && i == 0 && mon.status == Status::Brn {
                            mon.comp_brn = false;
                        }
                    }
                }
            }
            "-copyboost" => {
                if let (Some((ts, tm)), Some((ss, sm))) =
                    (self.subject(arg(1)), self.subject(arg(2)))
                {
                    let src = self.sides[ss].mons[sm].boosts;
                    self.sides[ts].mons[tm].boosts = src;
                }
            }
            "-clearallboost" => {
                // Haze: boosts cleared AND the par/brn drop volatiles removed
                for s in self.sides.iter_mut() {
                    if let Some(a) = s.active {
                        s.mons[a].boosts = [0; 7];
                        s.mons[a].comp_par = false;
                        s.mons[a].comp_brn = false;
                    }
                }
            }
            "-start" => self.on_start(dex, &parts),
            "-end" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    let key = toid(arg(2).strip_prefix("move: ").unwrap_or(arg(2)));
                    // binding traps end under their MOVE's name
                    let key = if BIND_CLASS.contains(&key.as_str()) {
                        "partiallytrapped".to_string()
                    } else {
                        key
                    };
                    self.sides[s].mons[m].vols.retain(|v| v.key != key);
                }
            }
            "-activate" => self.on_activate(dex, &parts),
            "-singleturn" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    let what = arg(2).strip_prefix("move: ").unwrap_or(arg(2));
                    if matches!(toid(what).as_str(), "protect" | "endure" | "detect") {
                        self.sides[s].mons[m].protected_this_turn = true;
                    }
                }
            }
            "-singlemove" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    let key = toid(arg(2).strip_prefix("move: ").unwrap_or(arg(2)));
                    if matches!(key.as_str(), "rage" | "destinybond") {
                        let turn = self.turn;
                        let mon = &mut self.sides[s].mons[m];
                        if !mon.vols.iter().any(|v| v.key == key) {
                            mon.vols.push(TVol {
                                key,
                                start_turn: turn,
                                move_id: None,
                                source: None,
                                counter: None,
                            });
                        }
                    }
                }
            }
            "-sidestart" => {
                if let Some(s) = Self::side_subject(arg(1)) {
                    let key = toid(arg(2).strip_prefix("move: ").unwrap_or(arg(2)));
                    if !self.sides[s].conds.iter().any(|(k, _)| *k == key) {
                        let turn = self.turn;
                        self.sides[s].conds.push((key, turn));
                    }
                }
            }
            "-sideend" => {
                if let Some(s) = Self::side_subject(arg(1)) {
                    let key = toid(arg(2).strip_prefix("move: ").unwrap_or(arg(2)));
                    self.sides[s].conds.retain(|(k, _)| *k != key);
                }
            }
            "-weather" => {
                let w = arg(1);
                if w == "none" {
                    self.weather = None;
                } else if parts.iter().any(|p| *p == "[upkeep]") {
                    if let Some((_, upkeeps)) = self.weather.as_mut() {
                        *upkeeps += 1;
                    }
                } else {
                    self.weather = Some((toid(w), 0));
                }
            }
            "-prepare" => {
                // charge turn of a genuine two-turn move (the |move| line
                // with [still] cannot be trusted — failed moves get [still]
                // attributed retroactively too)
                if let Some((s, m)) = self.subject(arg(1)) {
                    if let Some(mid) = dex.moves.id(&toid(arg(2))) {
                        self.sides[s].mons[m].charging = Some(mid);
                    }
                }
            }
            "-anim" => {
                // an animation right after |-prepare| = the charge released
                // in the same turn (Solar Beam in sun): no lock next turn
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.sides[s].mons[m].charging = None;
                }
            }
            "-transform" => {
                if let (Some((s, m)), Some(t)) = (self.subject(arg(1)), self.subject(arg(2))) {
                    // Transform copies the target's stat stages (public: both
                    // sides' boost lines are announced) — seed the tracked
                    // boosts with the copy; later lines mutate them normally.
                    let tb = self.sides[t.0].mons[t.1].boosts;
                    let mon = &mut self.sides[s].mons[m];
                    mon.transformed_into = Some(t);
                    mon.boosts = tb;
                }
            }
            "-mustrecharge" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.sides[s].mons[m].must_recharge = true;
                }
            }
            "-miss" | "-fail" => {
                // rollout's lock and fury cutter's streak end on a whiff
                if let Some((s, m)) = self.subject(arg(1)) {
                    let mon = &mut self.sides[s].mons[m];
                    if let Some((lm, _)) = mon.locked {
                        if dex.moves.key(lm) == "rollout" {
                            mon.locked = None;
                        }
                    }
                    mon.vols.retain(|v| v.key != "furycutter");
                }
            }
            "-immune" => {
                // subject is the TARGET; the attacker's rollout streak ends
                if let Some((s, _)) = self.subject(arg(1)) {
                    let atk = &mut self.sides[1 - s];
                    if let Some(a) = atk.active {
                        let mon = &mut atk.mons[a];
                        if let Some((lm, _)) = mon.locked {
                            if dex.moves.key(lm) == "rollout" {
                                mon.locked = None;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn on_switch(&mut self, dex: &Dex, parts: &[&str], is_drag: bool) {
        let subj = parts.get(1).copied().unwrap_or("");
        let Some(side) = Self::side_subject(subj) else { return };
        let Some((sp, lv, _g)) = Self::parse_details(dex, parts.get(2).copied().unwrap_or(""))
        else {
            return;
        };
        // resolve incoming slot by species (+level; Species Clause ⇒ unique)
        let Some(slot) = self.sides[side]
            .mons
            .iter()
            .position(|m| m.species == sp && m.level == lv)
        else {
            return;
        };
        // learn nickname
        if let Some(name) = subj.split_once(": ").map(|(_, n)| n) {
            self.sides[side].mons[slot].name = name.to_string();
        }
        // outgoing
        let bp = self.sides[side].pending_bp && !is_drag;
        let mut inherit: Option<([i8; 7], Vec<TVol>, Option<i32>, bool, bool)> = None;
        if let Some(old) = self.sides[side].active {
            if old != slot {
                let om = &mut self.sides[side].mons[old];
                om.active = false;
                if bp {
                    let vols: Vec<TVol> = om
                        .vols
                        .iter()
                        .filter(|v| {
                            dex.conds_id(&v.key).map_or(false, |c| !dex.cond(c).no_copy)
                        })
                        .cloned()
                        .collect();
                    inherit = Some((om.boosts, vols, om.comp_res, om.comp_brn, om.comp_par));
                }
                om.clear_on_exit();
                self.clear_sourced_vols(side, old);
            }
        }
        self.sides[side].pending_bp = false;
        self.sides[side].acted_this_turn = true;
        // incoming
        let turn = self.turn;
        {
            let mon = &mut self.sides[side].mons[slot];
            mon.appeared = true;
            mon.appear_count += 1;
            mon.switch_in_turn = turn;
            mon.active = true;
            if let Some((boosts, vols, res, brn, par)) = inherit {
                mon.boosts = boosts;
                mon.vols = vols;
                mon.comp_res = res;
                mon.comp_brn = brn;
                mon.comp_par = par;
            }
        }
        self.sides[side].active = Some(slot);
        self.apply_hp_status(side, slot, parts.get(3).copied().unwrap_or(""));
        // brn/par onSwitchIn re-add their Stadium stat-drop volatiles
        {
            let mon = &mut self.sides[side].mons[slot];
            match mon.status {
                Status::Brn => mon.comp_brn = true,
                Status::Par => mon.comp_par = true,
                _ => {}
            }
        }
    }

    fn on_move(&mut self, dex: &Dex, parts: &[&str]) {
        let Some((s, m)) = self.subject(parts.get(1).copied().unwrap_or("")) else { return };
        self.sides[s].acted_this_turn = true;
        let name = parts.get(2).copied().unwrap_or("");
        let Some(mid) = dex.moves.id(&toid(name)) else { return };
        let key = dex.moves.key(mid);
        // `[from] <other move>` = a called move (Sleep Talk / Mirror Move —
        // no PP of its own). `[from] <the same move>` = a special execution
        // of the mon's own choice (Pursuit intercepting a switch) — PS
        // deducts PP for those exactly like a normal use.
        let from_key = parts
            .iter()
            .find_map(|p| p.strip_prefix("[from] "))
            .map(|n| toid(n.strip_prefix("move: ").unwrap_or(n)));
        let called = match &from_key {
            Some(k) => k.as_str() != key,
            None => false,
        };
        {
            let side = &mut self.sides[s];
            side.side_last_move = Some(mid);
            let mon = &mut side.mons[m];
            mon.last_move = Some(mid);
            if key == "batonpass" {
                side.pending_bp = true;
            }
            // single-move volatiles expire at the user's next move
            mon.vols.retain(|v| v.key != "destinybond");
            if key != "rage" {
                mon.vols.retain(|v| v.key != "rage");
            }
            if key != "furycutter" {
                mon.vols.retain(|v| v.key != "furycutter");
            }
        }
        if called {
            return; // Sleep Talk / Mirror Move calls: no PP, no lock changes
        }
        if matches!(key, "struggle" | "recharge") {
            return;
        }
        let mon = &mut self.sides[s].mons[m];
        if mon.charging == Some(mid) {
            // release turn of a two-turn move (`-prepare` set `charging` on
            // the charge turn, whose |move| line already deducted): no PP
            mon.charging = None;
            return;
        }
        mon.charging = None;
        if let Some((lm, uses)) = mon.locked {
            if lm == mid {
                // lockedmove/rollout continuation: no PP deduction.
                // lockedmove runs 2-3 move actions; rollout up to 5 hits
                // (a miss ends it — handled at the |-miss| line).
                let cap = if key == "rollout" { 5 } else { 3 };
                let uses = uses + 1;
                mon.locked = if uses >= cap { None } else { Some((lm, uses)) };
                return;
            }
            mon.locked = None;
        }
        mon.deduct(mid);
        if THRASH_CLASS.contains(&key) || key == "rollout" {
            mon.locked = Some((mid, 1));
        }
        let turn = self.turn;
        let mon = &mut self.sides[s].mons[m];
        match key {
            // moves whose lasting volatile is implied by the (public) use
            "defensecurl" | "minimize" => {
                if !mon.vols.iter().any(|v| v.key == key) {
                    mon.vols.push(TVol {
                        key: key.to_string(),
                        start_turn: turn,
                        move_id: None,
                        source: None,
                        counter: None,
                    });
                }
            }
            "furycutter" => {
                match mon.vols.iter_mut().find(|v| v.key == "furycutter") {
                    Some(v) => *v.counter.get_or_insert(1) += 1,
                    None => mon.vols.push(TVol {
                        key: "furycutter".to_string(),
                        start_turn: turn,
                        move_id: None,
                        source: None,
                        counter: Some(1),
                    }),
                }
            }
            _ => {}
        }
    }

    fn on_start(&mut self, dex: &Dex, parts: &[&str]) {
        let Some((s, m)) = self.subject(parts.get(1).copied().unwrap_or("")) else { return };
        let what = parts.get(2).copied().unwrap_or("");
        let turn = self.turn;
        // perish counter
        if let Some(n) = what.strip_prefix("perish").and_then(|n| n.parse::<i64>().ok()) {
            let mon = &mut self.sides[s].mons[m];
            match mon.vols.iter_mut().find(|v| v.key == "perishsong") {
                Some(v) => v.counter = Some(n),
                None => mon.vols.push(TVol {
                    key: "perishsong".to_string(),
                    start_turn: turn,
                    move_id: None,
                    source: None,
                    counter: Some(n),
                }),
            }
            return;
        }
        let key = toid(what.strip_prefix("move: ").unwrap_or(what));
        // fatigue confusion ends a thrash lock (indistinguishable from other
        // confusion sources — clearing is the conservative reading for PP)
        if key == "confusion" {
            self.sides[s].mons[m].locked = None;
        }
        if dex.conds_id(&key).is_none() {
            return; // not a runtime condition we can plant (e.g. typechange)
        }
        let move_id = match key.as_str() {
            // encore locks the subject's last move; disable names it in arg 3
            "encore" => self.sides[s].mons[m].last_move,
            "disable" => parts
                .get(3)
                .and_then(|n| dex.moves.id(&toid(n)))
                .or(self.sides[s].mons[m].last_move),
            _ => None,
        };
        let source = self.sides[1 - s].active.map(|a| (1 - s, a));
        let mon = &mut self.sides[s].mons[m];
        if mon.vols.iter().any(|v| v.key == key) {
            return;
        }
        mon.vols.push(TVol { key, start_turn: turn, move_id, source, counter: None });
    }

    fn on_activate(&mut self, dex: &Dex, parts: &[&str]) {
        let Some((s, m)) = self.subject(parts.get(1).copied().unwrap_or("")) else { return };
        let what = parts.get(2).copied().unwrap_or("");
        let turn = self.turn;
        if what == "trapped" {
            let source = self.sides[1 - s].active.map(|a| (1 - s, a));
            let mon = &mut self.sides[s].mons[m];
            if !mon.vols.iter().any(|v| v.key == "trapped") {
                mon.vols.push(TVol {
                    key: "trapped".to_string(),
                    start_turn: turn,
                    move_id: None,
                    source,
                    counter: None,
                });
            }
            return;
        }
        if what == "Protect" || what == "Detect" {
            // the attacker was blocked: its rollout streak ends
            let atk = &mut self.sides[1 - s];
            if let Some(a) = atk.active {
                let mon = &mut atk.mons[a];
                if let Some((lm, _)) = mon.locked {
                    if dex.moves.key(lm) == "rollout" {
                        mon.locked = None;
                    }
                }
            }
            return;
        }
        // Mystery Berry PP restore (+5, capped — the cap is invisible from
        // the protocol but the restored slot was at 0, so uses -= 5 with a
        // floor of maxpp-consistency is exact up to the cap)
        if what == "item: Mystery Berry" {
            if let Some(mid) = parts.get(3).and_then(|n| dex.moves.id(&toid(n))) {
                let mon = &mut self.sides[s].mons[m];
                if let Some((_, n)) = mon.uses.iter_mut().find(|(id, _)| *id == mid) {
                    *n = (*n - 5).max(0);
                }
            }
            return;
        }
        if let Some(name) = what.strip_prefix("move: ") {
            let key = toid(name);
            if key == "spite" {
                // |-activate|target|move: Spite|<movekey>|<n>
                if let (Some(mid), Some(n)) = (
                    parts.get(3).and_then(|n| dex.moves.id(&toid(n))),
                    parts.get(4).and_then(|n| n.parse::<i32>().ok()),
                ) {
                    let mon = &mut self.sides[s].mons[m];
                    match mon.uses.iter_mut().find(|(id, _)| *id == mid) {
                        Some((_, u)) => *u += n,
                        None => mon.uses.push((mid, n)),
                    }
                }
                return;
            }
            if key == "mimic" {
                if let Some(overlay) =
                    parts.get(3).and_then(|n| dex.moves.id(&toid(n)))
                {
                    self.sides[s].mons[m].mimic_overlay = Some(overlay);
                }
                return;
            }
            if BIND_CLASS.contains(&key.as_str()) {
                let mid = dex.moves.id(&key);
                let source = parts
                    .iter()
                    .find_map(|p| p.strip_prefix("[of] "))
                    .and_then(|of| self.subject(of));
                let mon = &mut self.sides[s].mons[m];
                if !mon.vols.iter().any(|v| v.key == "partiallytrapped") {
                    mon.vols.push(TVol {
                        key: "partiallytrapped".to_string(),
                        start_turn: turn,
                        move_id: mid,
                        source,
                        counter: None,
                    });
                }
            }
        }
    }

    /// Defensive: linked volatiles die with their source (mean look /
    /// binding) — remove them when the source leaves the field. Attract's
    /// end is always announced (`-end ... [silent]` is still a line), so it
    /// is NOT cleared here (gen 2 keeps it when the attractor leaves).
    fn clear_sourced_vols(&mut self, src_side: usize, src_slot: usize) {
        for s in 0..2 {
            for mon in self.sides[s].mons.iter_mut() {
                mon.vols.retain(|v| {
                    !(v.source == Some((src_side, src_slot))
                        && matches!(v.key.as_str(), "trapped" | "partiallytrapped"))
                });
            }
        }
    }
}

/// Add/remove a Stadium companion volatile to match its tracked presence
/// (they carry no callbacks with side effects — see conditions.rs).
fn sync_companion(
    dex: &Dex,
    b: &mut Battle,
    id: PokeId,
    key: &str,
    present: bool,
) {
    let has = dex
        .conds_id(key)
        .map(|c| b.poke(id).has_volatile(c))
        .unwrap_or(false);
    if present && !has {
        b.add_volatile(dex, id, key, None, EffectHandle::None);
    } else if !present && has {
        b.remove_volatile(dex, id, key);
    }
}

fn cure(mon: &mut TrackMon) {
    // companion volatiles (residualdmg counter, stat drops) survive cures
    mon.status = Status::None;
    mon.rest = false;
    mon.slept = 0;
}

fn boost_index(name: &str) -> Option<usize> {
    ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        .iter()
        .position(|&n| n == name)
}

// =================================================================== request

/// Parsed PS request JSON (the player-visible own-side truth).
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub team_preview: bool,
    pub force_switch: bool,
    pub wait: bool,
    pub trapped: bool,
    pub active_moves: Vec<ReqMove>,
    pub pokemon: Vec<ReqMon>,
}

#[derive(Clone, Debug)]
pub struct ReqMove {
    pub id: String,
    pub pp: Option<i32>,
    pub maxpp: Option<i32>,
    pub disabled: bool,
}

#[derive(Clone, Debug)]
pub struct ReqMon {
    pub species: SpeciesId,
    pub level: u8,
    pub gender: Gender,
    pub hp: i32,
    pub maxhp: i32,
    pub status: Status,
    pub fainted: bool,
    pub active: bool,
    pub item: String,
}

impl Request {
    pub fn parse(dex: &Dex, json: &str) -> Result<Request, String> {
        let v: serde_json::Value =
            serde_json::from_str(json).map_err(|e| format!("request parse: {e}"))?;
        let mut req = Request {
            team_preview: v["teamPreview"].as_bool().unwrap_or(false),
            wait: v["wait"].as_bool().unwrap_or(false),
            force_switch: v["forceSwitch"]
                .as_array()
                .map(|a| a.first().and_then(|x| x.as_bool()).unwrap_or(false))
                .unwrap_or(false),
            ..Default::default()
        };
        if let Some(active) = v["active"].as_array().and_then(|a| a.first()) {
            req.trapped = active["trapped"].as_bool().unwrap_or(false);
            if let Some(moves) = active["moves"].as_array() {
                for m in moves {
                    req.active_moves.push(ReqMove {
                        id: m["id"].as_str().unwrap_or("").to_string(),
                        pp: m["pp"].as_i64().map(|x| x as i32),
                        maxpp: m["maxpp"].as_i64().map(|x| x as i32),
                        disabled: m["disabled"].as_bool().unwrap_or(false),
                    });
                }
            }
        }
        if let Some(mons) = v["side"]["pokemon"].as_array() {
            for p in mons {
                let details = p["details"].as_str().unwrap_or("");
                let (species, level, gender) = ProtocolTracker::parse_details(dex, details)
                    .ok_or_else(|| format!("request details: {details}"))?;
                let cond = p["condition"].as_str().unwrap_or("");
                let (hp, maxhp, status) = ProtocolTracker::parse_hp(cond);
                req.pokemon.push(ReqMon {
                    species,
                    level,
                    gender,
                    hp,
                    maxhp,
                    status: Status::from_str(status),
                    fainted: status == "fnt",
                    active: p["active"].as_bool().unwrap_or(false),
                    item: p["item"].as_str().unwrap_or("").to_string(),
                });
            }
        }
        Ok(req)
    }

    /// The PS-legal choice inputs this request admits (battle phase only;
    /// team preview legality is handled by the level mask).
    pub fn legal_inputs(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.team_preview || self.wait {
            return out;
        }
        if !self.force_switch {
            for m in &self.active_moves {
                if !m.disabled && m.pp != Some(0) {
                    out.push(format!("move {}", m.id));
                }
            }
        }
        let can_switch = self.force_switch || (!self.trapped && !self.active_moves.is_empty());
        if can_switch {
            for (pos, mon) in self.pokemon.iter().enumerate() {
                if !mon.active && !mon.fainted {
                    out.push(format!("switch {}", pos + 1));
                }
            }
        }
        if self.force_switch && out.is_empty() {
            out.push("pass".to_string());
        }
        out
    }
}

// ================================================================= synthesis

impl ProtocolTracker {
    /// Build the synthetic battle for the current decision point.
    ///
    /// `own_sets` = the exact team we submitted; `refs` = the belief's
    /// imputation roster for the opponent (aligned to `|poke|` order — the
    /// same alignment `Belief::determinize` uses); `obs` supplies public
    /// item knowledge; `req` is our request at this decision point.
    pub fn synthesize(
        &self,
        dex: &Dex,
        own_sets: &[PokemonSet],
        refs: &[nc2000_engine::state::Pokemon],
        obs: &Observer,
        req: &Request,
        rng: &mut SplitMix64,
    ) -> Result<Battle, String> {
        let me = self.side;
        let opp = 1 - me;
        let (p1, p2): (&[PokemonSet], &[PokemonSet]) = (own_sets, own_sets);
        let mut b = Battle::from_fixture(dex, &rng.battle_seed(), p1, p2)
            .map_err(|e| format!("from_fixture: {e:?}"))?;
        b.set_log_enabled(false);

        // opponent roster := belief refs (fresh preview-state mons aligned
        // to observed slots), with public preview facts stamped on
        if refs.len() != b.sides[opp].roster.len() {
            return Err("refs/roster length mismatch".to_string());
        }
        b.sides[opp].roster = refs.to_vec();
        for (slot, tm) in self.sides[opp].mons.iter().enumerate() {
            let p = &mut b.sides[opp].roster[slot];
            p.gender = tm.gender;
            if !tm.name.is_empty() {
                p.name = safe_name(&tm.name);
            }
        }
        for slot in 0..b.sides[opp].roster.len() {
            b.refresh_poke_mask(dex, PokeId { side: opp as u8, slot: slot as u8 });
        }
        // own genders are public (details) and PS sampled them with ITS rng
        for (slot, rm) in req.pokemon.iter().enumerate() {
            if req.team_preview {
                // preview request lists all 6 in roster order
                if let Some(p) = b.sides[me].roster.get_mut(slot) {
                    if p.species == rm.species {
                        p.gender = rm.gender;
                    }
                }
            }
        }

        if req.team_preview {
            // from_fixture state IS the preview decision point (party of 6,
            // TeamPreview request, queue holding the Start action).
            b.battle_mask = b.recompute_battle_mask(dex);
            return Ok(b);
        }

        // ------------------------------------------------- battle phase
        b.queue.clear(); // drop the preview Start action
        b.mid_turn = false;
        b.turn = self.turn;

        // ---- party arrangement
        // own: the request lists the picked mons in display order
        let mut own_party: Vec<u8> = Vec::new();
        for rm in &req.pokemon {
            let slot = b.sides[me]
                .roster
                .iter()
                .position(|p| p.species == rm.species && p.level == rm.level)
                .ok_or_else(|| "request mon not in own roster".to_string())?;
            own_party.push(slot as u8);
            let p = &mut b.sides[me].roster[slot];
            p.gender = rm.gender;
        }
        apply_party(&mut b, me, &own_party);
        // opponent: active first (display position 0 in singles), then the
        // other appeared picks, then imputed hidden picks (the determinizer
        // resamples those identities per iteration anyway)
        let ts = &self.sides[opp];
        let picked = self.sides[me].mons.len().min(req.pokemon.len()).max(1);
        let picked = picked.min(3); // NC2000: 3 picks
        let mut opp_party: Vec<u8> = Vec::new();
        if let Some(a) = ts.active {
            opp_party.push(a as u8);
        }
        for (i, m) in ts.mons.iter().enumerate() {
            if m.appeared && Some(i) != ts.active {
                opp_party.push(i as u8);
            }
        }
        for (i, m) in ts.mons.iter().enumerate() {
            if opp_party.len() >= picked {
                break;
            }
            if !m.appeared {
                let _ = m;
                opp_party.push(i as u8);
            }
        }
        apply_party(&mut b, opp, &opp_party);

        // ---- actives + flow
        for s in 0..2 {
            let active = self.sides[s].active;
            b.sides[s].active = active.map(|a| a as u8);
            b.sides[s].fainted_this_turn = self.sides[s].fainted_this_turn.map(|x| x as u8);
            b.sides[s].fainted_last_turn = self.sides[s].fainted_last_turn.map(|x| x as u8);
            b.sides[s].last_move = self.sides[s].side_last_move;
            for (slot, tm) in self.sides[s].mons.iter().enumerate() {
                let id = PokeId { side: s as u8, slot: slot as u8 };
                let p = b.poke_mut(id);
                p.is_active = tm.active && !tm.fainted;
                p.is_started = tm.active && !tm.fainted;
                p.previously_switched_in = tm.appear_count;
                p.active_turns =
                    (self.turn.saturating_sub(tm.switch_in_turn)) as i32;
                p.newly_switched = false;
            }
        }

        // ---- per-mon public state (status while HP is still full, then HP)
        for s in 0..2 {
            for slot in 0..b.sides[s].roster.len() {
                let id = PokeId { side: s as u8, slot: slot as u8 };
                let tm = &self.sides[s].mons[slot];
                self.apply_status(dex, &mut b, id, tm, rng);
            }
        }
        // own side authoritative overwrite from the request
        for (pos, rm) in req.pokemon.iter().enumerate() {
            let slot = b.sides[me].party[pos];
            let id = PokeId { side: me as u8, slot };
            if rm.fainted {
                let p = b.poke_mut(id);
                p.hp = 0;
                p.fainted = true;
                p.status = Status::None;
            } else {
                let p = b.poke_mut(id);
                p.hp = rm.hp.min(p.maxhp);
                p.status = rm.status;
            }
            // current item
            let item = if rm.item.is_empty() { None } else { dex.items.id(&toid(&rm.item)) };
            set_item_raw(&mut b, id, item);
            b.refresh_poke_mask(dex, id);
        }
        // opponent HP buckets + faints + items
        for (slot, tm) in self.sides[opp].mons.iter().enumerate() {
            let id = PokeId { side: opp as u8, slot: slot as u8 };
            if tm.fainted {
                let p = b.poke_mut(id);
                p.hp = 0;
                p.fainted = true;
                p.status = Status::None;
            } else {
                let maxhp = b.poke(id).maxhp;
                let hp = impute_hp(tm.pixels, tm.hp_den, maxhp);
                b.poke_mut(id).hp = hp;
            }
            // item: mirror impute_mon's contract (Observer knowledge wins)
            let mo = &obs.mons()[slot];
            match mo.item.current {
                Some(known) => set_item_raw(&mut b, id, known),
                None => {} // keep the ref's (candidate's) item
            }
            b.refresh_poke_mask(dex, id);
        }

        // ---- PP marks
        for s in 0..2 {
            for (slot, tm) in self.sides[s].mons.iter().enumerate() {
                let id = PokeId { side: s as u8, slot: slot as u8 };
                let p = b.poke_mut(id);
                for &(mid, n) in &tm.uses {
                    for ms in p.base_move_slots.iter_mut() {
                        // move_matches: PS reveals plain `hiddenpower`
                        if move_matches(dex, ms.id, mid) {
                            ms.pp = (ms.pp - n).max(0);
                            ms.used = true;
                        }
                    }
                }
                p.move_slots = p.base_move_slots;
                // Mimic overlay (announced): replace the Mimic slot
                if let Some(overlay) = tm.mimic_overlay {
                    if let Some(mimic) = dex.moves.id("mimic") {
                        if let Some(i) =
                            (0..p.move_slots.len()).find(|&i| p.move_slots[i].id == mimic)
                        {
                            let ms = dex.move_static(overlay);
                            let pp_ups = if ms.no_pp_boosts { 0 } else { 3 };
                            let mut maxpp = ms.pp * (5 + pp_ups) / 5;
                            if ms.pp == 40 {
                                maxpp -= pp_ups;
                            }
                            p.move_slots[i] = MoveSlot {
                                id: overlay,
                                pp: ms.pp.min(5),
                                maxpp,
                                disabled: false,
                                used: false,
                                shared: false,
                            };
                        }
                    }
                }
            }
        }

        // ---- transforms (needs both mons' base state final)
        for s in 0..2 {
            for (slot, tm) in self.sides[s].mons.iter().enumerate() {
                if let Some((ts_, tslot)) = tm.transformed_into {
                    let id = PokeId { side: s as u8, slot: slot as u8 };
                    let target = PokeId { side: ts_ as u8, slot: tslot as u8 };
                    // A live transform persists after the copy target later
                    // faints, but transform_into refuses fainted targets —
                    // lift the flag for the re-plant, then restore.
                    let (was_fainted, was_hp) = {
                        let t = b.poke(target);
                        (t.fainted, t.hp)
                    };
                    if was_fainted {
                        let t = b.poke_mut(target);
                        t.fainted = false;
                        if t.hp <= 0 {
                            t.hp = 1;
                        }
                    }
                    let _ = b.transform_into(dex, id, target);
                    if was_fainted {
                        let t = b.poke_mut(target);
                        t.fainted = true;
                        t.hp = was_hp;
                    }
                }
            }
        }

        // ---- volatiles (engine-API planting)
        for s in 0..2 {
            for (slot, tm) in self.sides[s].mons.iter().enumerate() {
                let id = PokeId { side: s as u8, slot: slot as u8 };
                if b.poke(id).fainted {
                    continue;
                }
                self.plant_volatiles(dex, &mut b, id, tm, rng);
            }
        }

        // ---- boosts (after transform/plants: tracked values are the truth)
        for s in 0..2 {
            for (slot, tm) in self.sides[s].mons.iter().enumerate() {
                let id = PokeId { side: s as u8, slot: slot as u8 };
                b.poke_mut(id).boosts = tm.boosts;
            }
        }

        // ---- last-move flow (after encore/disable planting)
        for s in 0..2 {
            for (slot, tm) in self.sides[s].mons.iter().enumerate() {
                let id = PokeId { side: s as u8, slot: slot as u8 };
                let p = b.poke_mut(id);
                p.last_move = tm.last_move;
                p.last_move_encore = tm.last_move;
                p.last_move_used = tm.last_move;
            }
        }

        // ---- side conditions + weather
        for s in 0..2 {
            let Some(active) = b.active_id(s) else { continue };
            for (key, start) in &self.sides[s].conds {
                b.add_side_condition(dex, s as u8, key, Some(active), EffectHandle::None);
                if let Some(sc) = b.sides[s]
                    .side_conditions
                    .iter_mut()
                    .find(|(c, _)| dex.conds_key(*c) == key)
                {
                    if let Some(d) = sc.1.duration {
                        let elapsed = self.turn.saturating_sub(*start) as i32;
                        sc.1.duration = Some((d - elapsed).max(1));
                    }
                }
            }
        }
        if let Some((wkey, upkeeps)) = &self.weather {
            let src = b.active_id(me).or_else(|| b.active_id(opp));
            b.set_weather(dex, wkey, src, EffectHandle::None);
            if let Some(d) = b.field.weather_state.duration {
                b.field.weather_state.duration = Some((d - *upkeeps as i32).max(1));
            }
        }

        // ---- own active authoritative move state (move requests)
        if !req.force_switch && !req.active_moves.is_empty() {
            if let Some(active) = b.active_id(me) {
                let p = b.poke_mut(active);
                for rm in &req.active_moves {
                    if let Some(mid) = dex.moves.id(&toid(&rm.id)) {
                        for ms in p.move_slots.iter_mut() {
                            if move_matches(dex, ms.id, mid) {
                                if let Some(pp) = rm.pp {
                                    ms.pp = pp;
                                }
                                if let Some(maxpp) = rm.maxpp {
                                    ms.maxpp = maxpp;
                                }
                                ms.disabled = rm.disabled;
                            }
                        }
                        // mirror into shared base slots (PS shares objects)
                        for ms in p.base_move_slots.iter_mut() {
                            if move_matches(dex, ms.id, mid) {
                                if let Some(pp) = rm.pp {
                                    ms.pp = pp;
                                }
                                if let Some(maxpp) = rm.maxpp {
                                    ms.maxpp = maxpp;
                                }
                            }
                        }
                    }
                }
                p.trapped = req.trapped;
                p.maybe_trapped = false;
            }
        }

        // ---- pokemon_left: PS counts the picked, unfainted team. The
        // fixture constructor counted the full 6-mon roster and the faints
        // planted above never decremented it, so every terminal check
        // (win / Self-KO clause) was unreachable in searched games —
        // recount from the truncated party.
        for s in 0..2 {
            b.sides[s].pokemon_left = b.sides[s]
                .party
                .iter()
                .filter(|&&slot| !b.sides[s].roster[slot as usize].fainted)
                .count() as i32;
        }

        // ---- request bookkeeping + mid-turn queue
        let kind = if req.force_switch { RequestState::Switch } else { RequestState::Move };
        if kind == RequestState::Switch {
            for s in 0..2 {
                if let Some(active) = b.active_id(s) {
                    if b.poke(active).fainted {
                        // mirror check_fainted
                        b.poke_mut(active).status = Status::Fnt;
                        b.poke_mut(active).switch_flag = SwitchFlag::Yes;
                        b.refresh_poke_mask(dex, active);
                    }
                }
            }
            if let Some(active) = b.active_id(me) {
                if !b.poke(active).fainted {
                    // mid-turn self-switch (Baton Pass)
                    let bp = dex.moves.id("batonpass").ok_or("batonpass id")?;
                    b.poke_mut(active).switch_flag = SwitchFlag::Move(bp);
                }
            }
            b.mid_turn = true;
            // pending foe move (only when the foe still owes an action this
            // turn) — the determinizer's pending-move scrub resamples its id
            // per iteration, so any usable placeholder is fine
            let foe_pending = !self.sides[opp].acted_this_turn
                && b.active_id(opp).map(|a| !b.poke(a).fainted).unwrap_or(false)
                && b.active_id(me).map(|a| !b.poke(a).fainted).unwrap_or(false);
            if foe_pending {
                if let Some(fa) = b.active_id(opp) {
                    let usable: Vec<MoveId> = b
                        .poke(fa)
                        .move_slots
                        .iter()
                        .filter(|s| s.pp > 0 && !s.disabled)
                        .map(|s| s.id)
                        .collect();
                    let mid = if usable.is_empty() {
                        dex.moves.id("struggle").ok_or("struggle id")?
                    } else {
                        usable[rng.below(usable.len())]
                    };
                    let speed = b.get_pokemon_action_speed(dex, fa) as f64;
                    b.queue.push(Action {
                        choice: ActionKind::Move {
                            move_id: mid,
                            target_loc: 0,
                            original_target: None,
                            source_effect: None,
                        },
                        order: 200,
                        priority: dex.move_static(mid).priority as f64,
                        fractional_priority: 0.0,
                        speed,
                        pokemon: Some(fa),
                    });
                }
            }
            if !self.upkeep_this_turn {
                b.queue.push(Action {
                    choice: ActionKind::Residual,
                    order: 300,
                    priority: 0.0,
                    fractional_priority: 0.0,
                    speed: 1.0,
                    pokemon: None,
                });
            }
        }
        b.make_request(dex, kind);

        // ---- masks + speeds
        for s in 0..2u8 {
            for slot in 0..b.sides[s as usize].roster.len() {
                b.refresh_poke_mask(dex, PokeId { side: s, slot: slot as u8 });
            }
        }
        b.update_all_speeds(dex);
        b.battle_mask = b.recompute_battle_mask(dex);
        Ok(b)
    }

    fn apply_status(
        &self,
        dex: &Dex,
        b: &mut Battle,
        id: PokeId,
        tm: &TrackMon,
        _rng: &mut SplitMix64,
    ) {
        if tm.fainted {
            return;
        }
        if tm.status != Status::None {
            self.plant_status(dex, b, id, tm);
        }
        // Stadium companion volatiles have public lifecycles of their own
        // (they survive cures and status replacement) — sync them exactly
        sync_companion(dex, b, id, "brnattackdrop", tm.comp_brn);
        sync_companion(dex, b, id, "parspeeddrop", tm.comp_par);
        sync_companion(dex, b, id, "residualdmg", tm.comp_res.is_some());
        if let Some(n) = tm.comp_res {
            if let Some(rd) = dex.conds_id("residualdmg") {
                if let Some(vs) = b.poke_mut(id).volatile_mut(rd) {
                    vs.set_int(DK::Counter, n as i64);
                }
            }
        }
    }

    fn plant_status(&self, dex: &Dex, b: &mut Battle, id: PokeId, tm: &TrackMon) {
        // plant through the engine so companion state (residualdmg counter,
        // brnattackdrop, rolled sleep turns) comes from the real code paths
        let r = b.set_status(dex, id, tm.status.as_str(), None, EffectHandle::None, true);
        if !r.truthy() {
            // defensive: force the enum (e.g. an immunity edge the tracker
            // cannot see) — public status is authoritative
            b.poke_mut(id).status = tm.status;
            b.refresh_poke_mask(dex, id);
        }
        match tm.status {
            Status::Slp => {
                let p = b.poke_mut(id);
                if tm.rest {
                    // Rest sleep is a public 2-turn counter
                    p.status_state.set_int(DK::Time, (3 - tm.slept as i64).max(1));
                    p.status_state.set_int(DK::StartTime, 3);
                } else if let Some(t) = p.status_state.get(DK::Time).map(|v| v.as_int()) {
                    // clamp the rolled counter under the publicly slept turns
                    p.status_state.set_int(DK::Time, (t - tm.slept as i64).max(1));
                }
            }
            _ => {}
        }
    }

    fn plant_volatiles(
        &self,
        dex: &Dex,
        b: &mut Battle,
        id: PokeId,
        tm: &TrackMon,
        rng: &mut SplitMix64,
    ) {
        let src_id = |src: Option<(usize, usize)>| {
            src.map(|(s, m)| PokeId { side: s as u8, slot: m as u8 })
        };
        for v in &tm.vols {
            let elapsed = self.turn.saturating_sub(v.start_turn) as i32;
            match v.key.as_str() {
                "encore" => {
                    let Some(mid) = v.move_id else { continue };
                    b.poke_mut(id).last_move_encore = Some(mid);
                    b.add_volatile(dex, id, "encore", None, EffectHandle::None);
                }
                "disable" => {
                    let Some(mid) = v.move_id else { continue };
                    b.poke_mut(id).last_move = Some(mid);
                    b.add_volatile(dex, id, "disable", None, EffectHandle::None);
                }
                "trapped" => {
                    let Some(src) = src_id(v.source) else { continue };
                    if b.poke(src).is_active && !b.poke(src).fainted {
                        b.add_volatile_linked(
                            dex,
                            id,
                            "trapped",
                            Some(src),
                            EffectHandle::None,
                            "trapper",
                        );
                    }
                }
                "partiallytrapped" => {
                    let Some(mid) = v.move_id else { continue };
                    let src = src_id(v.source);
                    b.add_volatile(dex, id, "partiallytrapped", src, EffectHandle::MoveEff(mid));
                }
                "perishsong" => {
                    b.add_volatile(dex, id, "perishsong", None, EffectHandle::None);
                    if let Some(c) = dex.conds_id("perishsong") {
                        if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                            vs.duration = Some(v.counter.unwrap_or(3) as i32);
                        }
                    }
                }
                "furycutter" => {
                    b.add_volatile(dex, id, "furycutter", None, EffectHandle::None);
                    if let Some(c) = dex.conds_id("furycutter") {
                        if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                            let n = v.counter.unwrap_or(1).clamp(1, 5);
                            vs.set_int(DK::Multiplier, 1i64 << (n - 1).min(4));
                        }
                    }
                }
                key => {
                    let src = src_id(v.source);
                    b.add_volatile(dex, id, key, src, EffectHandle::None);
                }
            }
            // hidden rolled durations: clamp under the publicly elapsed turns
            if let Some(c) = dex.conds_id(&v.key) {
                if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                    if v.key != "perishsong" {
                        if let Some(d) = vs.duration {
                            vs.duration = Some((d - elapsed).max(1));
                        }
                    }
                }
            }
        }
        // mechanics-inferred locks / charge state / recharge
        if tm.must_recharge {
            b.add_volatile(dex, id, "mustrecharge", None, EffectHandle::None);
        }
        if let Some(m) = tm.charging {
            b.add_volatile(dex, id, "twoturnmove", Some(id), EffectHandle::MoveEff(m));
        }
        if let Some((m, uses)) = tm.locked {
            if dex.moves.key(m) == "rollout" {
                b.add_volatile(dex, id, "rollout", Some(id), EffectHandle::MoveEff(m));
                if let Some(c) = dex.conds_id("rollout") {
                    if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                        vs.set_int(DK::HitCount, uses as i64);
                        vs.set_int(DK::ContactHitCount, uses as i64);
                    }
                }
            } else {
                b.add_volatile(dex, id, "lockedmove", Some(id), EffectHandle::MoveEff(m));
                if let Some(c) = dex.conds_id("lockedmove") {
                    if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                        // total roll is 2-3 actions; uses so far are public
                        vs.duration = Some((2 - uses).max(1));
                    }
                }
            }
        }
        if tm.stall_streak > 0 && tm.last_protect_turn + 1 >= self.turn && self.turn > 0 {
            b.add_volatile(dex, id, "stall", None, EffectHandle::None);
            if let Some(c) = dex.conds_id("stall") {
                if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                    let mut counter = 127.0f64;
                    for _ in 1..tm.stall_streak {
                        counter /= 2.0;
                    }
                    if counter.fract() == 0.0 {
                        vs.set(DK::Counter, nc2000_engine::state::Scalar::Int(counter as i64));
                    } else {
                        vs.set(DK::Counter, nc2000_engine::state::Scalar::Float(counter));
                    }
                    vs.duration = Some(if tm.protected_this_turn { 2 } else { 1 });
                }
            }
        }
        let _ = rng;
    }
}

/// party := `slots` (display order); positions canonical (party index for
/// members, then bench parked in roster order — the M10 determinizer scheme).
fn apply_party(b: &mut Battle, s: usize, slots: &[u8]) {
    let side = &mut b.sides[s];
    side.party.clear();
    side.party.extend_from_slice(slots);
    let party_len = side.party.len();
    for pos in 0..party_len {
        let slot = side.party[pos] as usize;
        side.roster[slot].position = pos as u8;
    }
    let mut bench_pos = party_len as u8;
    for slot in 0..side.roster.len() {
        if !side.party.contains(&(slot as u8)) {
            side.roster[slot].position = bench_pos;
            bench_pos += 1;
        }
    }
}

/// Set an item field directly (planting, not an in-battle transition).
fn set_item_raw(b: &mut Battle, id: PokeId, item: Option<nc2000_engine::dex::ItemId>) {
    let p = b.poke_mut(id);
    p.item = item;
    p.item_state = nc2000_engine::state::EffectState {
        id: item.map(nc2000_engine::state::EffId::Item).unwrap_or_default(),
        effect_order: p.item_state.effect_order,
        ..Default::default()
    };
}

/// Impute an HP amount inside the announced 1/48 pixel bucket:
/// px = floor(48·hp/maxhp), floored to 1 while alive.
/// Midpoint of the true-HP range consistent with an announced `cur/den`
/// HP string. The announcement rounding differs by mode: HP Percentage Mod
/// (den = 100) announces `ceil(100*hp/maxhp)` with a not-quite-full 100
/// knocked down to 99; the legacy pixel bar (den = 48) announces
/// `floor(48*hp/maxhp)` clamped to >= 1.
fn impute_hp(cur: i32, den: i32, maxhp: i32) -> i32 {
    let den = if den > 0 { den } else { 48 };
    if cur <= 0 {
        return 0;
    }
    if cur >= den {
        return maxhp;
    }
    let (lo, hi) = if den == 100 {
        let lo = ((cur - 1) * maxhp) / 100 + 1;
        let hi = if cur == 99 { maxhp - 1 } else { (cur * maxhp) / 100 };
        (lo, hi)
    } else {
        let lo = if cur == 1 { 1 } else { (cur * maxhp + den - 1) / den };
        let hi = ((cur + 1) * maxhp + den - 1) / den - 1;
        (lo, hi)
    };
    let hi = hi.min(maxhp - 1).max(lo);
    ((lo + hi + 1) / 2).clamp(1, maxhp)
}

fn safe_name(name: &str) -> PokeName {
    let mut s = String::new();
    for ch in name.chars().take(20) {
        if s.len() + ch.len_utf8() > 24 {
            break;
        }
        s.push(ch);
    }
    PokeName::new(&s)
}

// ==================================================================== agent

/// The M15 protocol-driven imperfect-info agent: `ProtocolTracker` +
/// `Observer`/`Belief` (M10a) + `BlindSearch` (M10c) over synthesized
/// battles. One instance per GAME; drive it with `push_line` for every
/// player-visible battle line and `on_request` at each `|request|`, then
/// pump `step` / read `best` exactly like the wasm `BlindSearcher`.
pub struct ProtocolAgent {
    side: usize,
    cfg: RmConfig,
    pool: MetaPool,
    tables: TableSet,
    own_sets: Vec<PokemonSet>,
    pinned_sets: Option<Vec<PokemonSet>>,
    tracker: ProtocolTracker,
    history: Vec<String>,
    observer: Option<Observer>,
    belief: Option<Belief>,
    rng: SplitMix64,
    battle: Option<Battle>,
    search: Option<BlindSearch>,
    baked: Option<SearchChoice>,
    forced: Option<String>,
    request: Option<Request>,
    /// Decision points where the synthesized own-side legal set differed
    /// from the request-derived one (target 0; the projection still keeps
    /// every submission PS-legal).
    pub legality_drift: u32,
    /// Times `best()` had to be projected onto the request-legal set.
    pub projections: u32,
}

impl ProtocolAgent {
    pub fn new(
        dex: &Dex,
        side: usize,
        pool: MetaPool,
        cfg: RmConfig,
        seed: u64,
    ) -> ProtocolAgent {
        let tables = TableSet::from_pool(dex, &pool);
        ProtocolAgent {
            side,
            cfg,
            pool,
            tables,
            own_sets: Vec::new(),
            pinned_sets: None,
            tracker: ProtocolTracker::new(side),
            history: Vec::new(),
            observer: None,
            belief: None,
            rng: SplitMix64::new(seed),
            battle: None,
            search: None,
            baked: None,
            forced: None,
            request: None,
            legality_drift: 0,
            projections: 0,
        }
    }

    pub fn set_own_team(&mut self, sets: Vec<PokemonSet>) {
        self.own_sets = sets;
    }

    /// Open-team-sheet mode: the opponent's true sets are public.
    pub fn pin_opponent(&mut self, sets: Vec<PokemonSet>) {
        self.pinned_sets = Some(sets);
    }

    pub fn add_pair_json(&mut self, json: &str) -> Result<(), String> {
        self.tables.add_pair_json(json)
    }

    pub fn side(&self) -> usize {
        self.side
    }

    pub fn battle(&self) -> Option<&Battle> {
        self.battle.as_ref()
    }

    pub fn belief(&self) -> Option<&Belief> {
        self.belief.as_ref()
    }

    pub fn search(&self) -> Option<&BlindSearch> {
        self.search.as_ref()
    }

    pub fn observer(&self) -> Option<&Observer> {
        self.observer.as_ref()
    }

    /// Feed one player-visible protocol line (battle lines only, not
    /// `|request|`).
    pub fn push_line(&mut self, dex: &Dex, line: &str) {
        self.tracker.push_line(dex, line);
        match self.observer.as_mut() {
            Some(obs) => {
                sync_names(obs, &self.tracker);
                obs.ingest_line(line, dex);
            }
            None => self.history.push(line.to_string()),
        }
    }

    /// Feed the request JSON at a decision point. Returns false for `wait`
    /// requests (nothing owed).
    pub fn on_request(&mut self, dex: &Dex, request_json: &str) -> Result<bool, String> {
        let req = Request::parse(dex, request_json)?;
        if req.wait {
            self.search = None;
            self.baked = None;
            self.forced = None;
            self.request = None;
            return Ok(false);
        }
        if self.own_sets.is_empty() {
            return Err("own team not set".to_string());
        }
        // lazy observer/belief construction (preview |poke| lines are in by
        // the time the first request arrives)
        if self.observer.is_none() {
            if self.tracker.opp_roster_len() == 0 {
                return Err("no |poke| preview lines seen before first request".to_string());
            }
            let mut obs = Observer::from_mons(self.side, self.tracker.observer_mons());
            sync_names(&mut obs, &self.tracker);
            for line in std::mem::take(&mut self.history) {
                obs.ingest_line(&line, dex);
            }
            let belief = match &self.pinned_sets {
                Some(sets) => Belief::pinned(dex, "opponent", sets, &obs),
                None => Belief::new(dex, &self.pool, &obs),
            };
            self.observer = Some(obs);
            self.belief = Some(belief);
        }
        let obs = self.observer.as_ref().unwrap();
        let belief = self.belief.as_mut().unwrap();
        belief.sync(dex, obs);

        // synthesize
        let pick = belief.alive().first().copied();
        let battle = {
            let refs = belief.refs(pick);
            self.tracker.synthesize(dex, &self.own_sets, refs, obs, &req, &mut self.rng)?
        };

        // search + preview policy
        let mut search =
            BlindSearch::new(&battle, dex, self.cfg.clone(), self.side, self.rng.next());
        self.baked = None;
        self.forced = None;
        if search.is_preview() {
            // PS enforces Max Total Level at preview; the engine's
            // enumeration doesn't — mask overweight picks out of the root
            let levels: Vec<i32> =
                battle.sides[self.side].party.iter().map(|&s| {
                    battle.sides[self.side].roster[s as usize].level as i32
                }).collect();
            let allowed: Vec<bool> = search
                .actions()
                .iter()
                .map(|c| match c {
                    SearchChoice::Team(slots) => team_total_level(&levels, slots)
                        <= MAX_TOTAL_LEVEL,
                    _ => true,
                })
                .collect();
            if allowed.iter().any(|&a| a) {
                search.mask_actions(&allowed);
            }
            // baked pair tables, restricted to the legal picks
            let pick = if self.pinned_sets.is_some() {
                filtered_open_preview(&self.tables, &battle, self.side, &levels, &mut self.rng)
            } else {
                filtered_baked_preview(
                    &self.tables,
                    belief,
                    &battle,
                    self.side,
                    &levels,
                    &mut self.rng,
                )
            };
            if let Some(c) = pick {
                if search.actions().contains(&c) {
                    self.baked = Some(c);
                }
            }
        } else {
            // request-side legality metrics + single-choice bypass
            let legal = req.legal_inputs();
            let synth: Vec<String> = {
                let mut bb = battle.clone();
                bb.legal_choices(dex, self.side)
                    .iter()
                    .map(|&c| ps_input(dex, c))
                    .collect()
            };
            let mut a = legal.clone();
            let mut s2 = synth.clone();
            a.sort();
            s2.sort();
            if a != s2 {
                self.legality_drift += 1;
            }
            if legal.len() == 1 {
                self.forced = Some(legal[0].clone());
            }
        }
        self.battle = Some(battle);
        self.search = Some(search);
        self.request = Some(req);
        Ok(true)
    }

    /// Baked preview pick when one applies (ready-to-submit input string).
    pub fn baked_preview(&self, dex: &Dex) -> Option<String> {
        self.baked.map(|c| c.to_input(dex))
    }

    pub fn step(&mut self, dex: &Dex, n: u32) -> Result<u32, String> {
        let search = self.search.as_mut().ok_or("step before on_request")?;
        if self.baked.is_some() || self.forced.is_some() {
            return Ok(search.iterations());
        }
        let belief = self.belief.as_ref().ok_or("no belief")?;
        let obs = self.observer.as_ref().ok_or("no observer")?;
        Ok(search.step(dex, belief, obs, n))
    }

    pub fn iterations(&self) -> u32 {
        self.search.as_ref().map_or(0, |s| s.iterations())
    }

    /// Current best choice, projected onto the request-legal set (never
    /// submits something PS rejects).
    pub fn best(&mut self, dex: &Dex) -> Option<String> {
        if let Some(f) = &self.forced {
            return Some(f.clone());
        }
        if let Some(c) = self.baked {
            return Some(c.to_input(dex));
        }
        let best = ps_input(dex, self.search.as_ref()?.best()?);
        let req = self.request.as_ref()?;
        if req.team_preview {
            return Some(best); // level mask already applied at the root
        }
        let legal = req.legal_inputs();
        if legal.is_empty() || legal.contains(&best) {
            return Some(best);
        }
        self.projections += 1;
        Some(legal[0].clone())
    }

    /// JSON belief info (mirrors the wasm `BlindSearcher.beliefInfo`).
    pub fn belief_info(&self) -> String {
        match &self.belief {
            Some(b) => {
                let candidates: Vec<&str> =
                    b.alive().iter().map(|&i| b.candidate_id(i)).collect();
                serde_json::json!({
                    "count": b.candidate_count(),
                    "fallback": b.is_fallback(),
                    "candidates": candidates,
                })
                .to_string()
            }
            None => r#"{"count":0,"fallback":false,"candidates":[]}"#.to_string(),
        }
    }

    /// Root policy rows (input/visits/frac), sorted by visits.
    pub fn root_policy(&self, dex: &Dex) -> String {
        let (iterations, rows) = match &self.search {
            Some(s) => {
                let acts = s.actions();
                let visits = s.visits();
                let means = s.means();
                let total: u32 = visits.iter().sum();
                let mut rows: Vec<serde_json::Value> = acts
                    .iter()
                    .zip(visits.iter().zip(means.iter()))
                    .map(|(&a, (&n, &m))| {
                        serde_json::json!({
                            "input": a.to_input(dex),
                            "visits": n,
                            "mean": m,
                            "frac": if total > 0 { n as f64 / total as f64 } else { 0.0 },
                        })
                    })
                    .collect();
                rows.sort_by(|a, b| b["visits"].as_u64().cmp(&a["visits"].as_u64()));
                (s.iterations(), rows)
            }
            None => (0, Vec::new()),
        };
        serde_json::json!({
            "iterations": iterations,
            "baked": self.baked.is_some(),
            "forced": self.forced,
            "actions": rows,
        })
        .to_string()
    }
}

/// PS-canonical choice string: PS normalizes typed hidden powers to the
/// plain `hiddenpower` id (requests and choice parsing both), so a typed
/// engine id must be submitted in the plain form.
fn ps_input(dex: &Dex, c: SearchChoice) -> String {
    if let SearchChoice::Move(id) = c {
        let key = dex.moves.key(id);
        if key.starts_with("hiddenpower") && key != "hiddenpower" {
            return "move hiddenpower".to_string();
        }
    }
    c.to_input(dex)
}

fn sync_names(obs: &mut Observer, tracker: &ProtocolTracker) {
    let opp = 1 - tracker.side;
    for (slot, name) in tracker.names(opp).into_iter().enumerate() {
        if !name.is_empty() {
            obs.set_name(slot, name);
        }
    }
}

fn team_total_level(levels: &[i32], slots: &[u8; 3]) -> i32 {
    slots
        .iter()
        .filter(|&&s| s != 0)
        .map(|&s| levels.get(s as usize - 1).copied().unwrap_or(0))
        .sum()
}

/// `open_preview_pick` restricted to Max-Total-Level-legal actions.
fn filtered_open_preview(
    tables: &TableSet,
    battle: &Battle,
    side: usize,
    levels: &[i32],
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    let (tab, i_am_a) = tables.lookup(battle, side)?;
    let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
    sample_filtered(tables, p, levels, rng)
}

/// `baked_preview_pick` restricted to Max-Total-Level-legal actions.
fn filtered_baked_preview(
    tables: &TableSet,
    belief: &Belief,
    battle: &Battle,
    side: usize,
    levels: &[i32],
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    if belief.candidate_count() != 1 {
        return None;
    }
    let opp = belief.alive()[0];
    let me = tables.side_index(battle, side)?;
    let (tab, i_am_a) = tables.pair_by_idx(me, opp)?;
    let p = if i_am_a { &tab.sol.p_a } else { &tab.sol.p_b };
    sample_filtered(tables, p, levels, rng)
}

/// Sample a mixed preview policy conditioned on the PS-legal action subset
/// (renormalized; degenerate rows fall back to the legal argmax).
fn sample_filtered(
    tables: &TableSet,
    p: &[f64],
    levels: &[i32],
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    let actions = tables.actions();
    let legal: Vec<usize> = (0..p.len())
        .filter(|&i| team_total_level(levels, &actions[i]) <= MAX_TOTAL_LEVEL)
        .collect();
    if legal.is_empty() {
        return None;
    }
    let mass: f64 = legal.iter().map(|&i| p[i]).sum();
    let pick = if mass <= 0.0 {
        legal
            .iter()
            .copied()
            .max_by(|&a, &b| p[a].total_cmp(&p[b]))
            .unwrap()
    } else {
        let u = rng.next_f64() * mass;
        let mut acc = 0.0;
        let mut pick = legal[legal.len() - 1];
        for &i in &legal {
            acc += p[i];
            if u < acc {
                pick = i;
                break;
            }
        }
        pick
    };
    Some(SearchChoice::Team(actions[pick]))
}
