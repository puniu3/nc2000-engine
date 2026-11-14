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

use nc2000_engine::battle::{
    events::{ev, EvTarget},
    EffectHandle, PokemonSet, SearchChoice,
};
use nc2000_engine::dex::{toid, Category, Dex, FixedDamage, MoveId, SpeciesId};
use nc2000_engine::state::{
    Action, ActionKind, Battle, Gender, MoveSlot, MoveSlots, PokeId, PokeName, RequestState,
    Status, SwitchFlag, DK,
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
    /// Upkeeps observed while `locked` is active (for Thrash/Outrage/Petal Dance).
    locked_upkeeps: u8,
    /// Lock initiated by a called move (`Metronome`/`Mirror Move`/`Sleep Talk`).
    called_locked: bool,
    /// Two-turn move charge pending release.
    charging: Option<MoveId>,
    must_recharge: bool,
    transformed_into: Option<(usize, usize)>,
    transformed_uses: Vec<(MoveId, i32)>,
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
            locked_upkeeps: 0,
            called_locked: false,
            charging: None,
            must_recharge: false,
            transformed_into: None,
            transformed_uses: Vec::new(),
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

    fn deduct_transformed(&mut self, id: MoveId) {
        match self.transformed_uses.iter_mut().find(|(m, _)| *m == id) {
            Some((_, n)) => *n += 1,
            None => self.transformed_uses.push((id, 1)),
        }
    }

    /// Everything a switch-out clears (mirrors `clear_volatile`).
    fn clear_on_exit(&mut self) {
        self.boosts = [0; 7];
        self.vols.clear();
        self.locked = None;
        self.locked_upkeeps = 0;
        self.called_locked = false;
        self.charging = None;
        self.must_recharge = false;
        self.transformed_into = None;
        self.transformed_uses.clear();
        self.mimic_overlay = None;
        self.last_move = None;
        self.stall_streak = 0;
        self.protected_this_turn = false;
        self.comp_res = None;
        self.comp_brn = false;
        self.comp_par = false;
    }
}

/// Public per-mon snapshot for harness consumers — see
/// [`ProtocolTracker::snapshot`].
#[derive(Clone, Debug)]
pub struct MonSnapshot {
    pub species: SpeciesId,
    pub level: u8,
    pub gender: Gender,
    pub name: String,
    pub appeared: bool,
    pub active: bool,
    pub fainted: bool,
    /// Announced HP as a fraction of max (percentage-granular under HP% Mod).
    pub hp_frac: f64,
    /// The announced HP exactly as the protocol printed it: numerator over
    /// `hp_den` (100 under HP Percentage Mod, 48 on a legacy pixel-bar
    /// stream). Kept unrounded so a checker can re-announce an imputed HP
    /// amount and prove the rounding rule matches the one the stream used —
    /// the M18 class-A `%`-HP invariant.
    pub hp_num: i32,
    pub hp_den: i32,
    pub status: Status,
    /// Sleep came from Rest (public 2-turn clock).
    pub rest: bool,
    pub boosts: [i8; 7],
    /// Announced volatile keys (+ the locked/binding move where tracked).
    pub vols: Vec<(String, Option<MoveId>)>,
    /// PP-mark deductions by move id = revealed moves + use counts.
    pub uses: Vec<(MoveId, i32)>,
    /// Mid-action state that removes or distorts the choice (thrash lock,
    /// two-turn charge, recharge, transform).
    pub choiceless: bool,
}

/// One announced volatile.
#[derive(Clone, Debug)]
struct TVol {
    /// Engine condition key ("confusion", "substitute", ...).
    key: String,
    start_turn: u16,
    /// Number of `|upkeep|` boundaries elapsed since start.
    upkeeps: u8,
    /// Whether the target had already acted this turn when the volatile started (Disable +1).
    acted_on_start: bool,
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
    /// Transient current move `((side, slot), move_id, crit)` while resolving
    /// a `|move|` block (used to estimate Substitute `[damage]`).
    cur_move: Option<((usize, usize), MoveId, bool)>,
}

/// Locking (thrash-class) move keys: `lockedmove` continuation turns emit
/// plain `|move|` lines with NO PP deduction.
const THRASH_CLASS: [&str; 3] = ["thrash", "petaldance", "outrage"];
/// Binding move keys (`-activate |move: X|[of]` plants `partiallytrapped`).
const BIND_CLASS: [&str; 5] = ["wrap", "bind", "firespin", "clamp", "whirlpool"];

impl ProtocolTracker {
    pub(crate) fn set_hp_observations(&self, belief: &mut Belief, side: usize) {
        belief.set_hp_observations(self.sides[side].mons.iter()
            .map(|m| if m.fainted { (0, m.hp_den) } else { (m.pixels, m.hp_den) }).collect());
    }
    pub fn new(side: usize) -> ProtocolTracker {
        ProtocolTracker {
            side,
            sides: [TrackSide::default(), TrackSide::default()],
            weather: None,
            turn: 0,
            upkeep_this_turn: false,
            cur_move: None,
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
                hp_type_mask: u16::MAX,
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

    /// Read-only public snapshot of `side`'s tracked state + active roster
    /// slot — for harnesses that fabricate requests from a public stream
    /// (M16b human-agreement corpus replay). Facts only; no imputation.
    pub fn snapshot(&self, side: usize) -> (Vec<MonSnapshot>, Option<usize>) {
        let ts = &self.sides[side];
        let mons = ts
            .mons
            .iter()
            .map(|m| MonSnapshot {
                species: m.species,
                level: m.level,
                gender: m.gender,
                name: m.name.clone(),
                appeared: m.appeared,
                active: m.active,
                fainted: m.fainted,
                hp_frac: (m.pixels as f64 / m.hp_den.max(1) as f64).clamp(0.0, 1.0),
                hp_num: m.pixels,
                hp_den: m.hp_den.max(1),
                status: m.status,
                rest: m.rest,
                boosts: m.boosts,
                vols: m.vols.iter().map(|v| (v.key.clone(), v.move_id)).collect(),
                uses: m.uses.clone(),
                choiceless: m.locked.is_some()
                    || m.charging.is_some()
                    || m.must_recharge
                    || m.transformed_into.is_some(),
            })
            .collect();
        (mons, ts.active)
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
        let has_active_subpos = rest.starts_with(|c: char| c.is_ascii_lowercase());
        let rest = rest.strip_prefix(|c: char| c.is_ascii_lowercase()).unwrap_or(rest);
        let name = rest.strip_prefix(": ")?;
        if has_active_subpos {
            if let Some(a) = self.sides[side].active {
                if self.sides[side].mons.get(a).is_some_and(|m| m.name == name) {
                    return Some((side, a));
                }
            }
        }
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
                self.cur_move = None;
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
            "upkeep" => {
                self.upkeep_this_turn = true;
                self.cur_move = None;
                for side in self.sides.iter_mut() {
                    if let Some(a) = side.active {
                        let m = &mut side.mons[a];
                        if !m.fainted {
                            for v in &mut m.vols {
                                v.upkeeps = v.upkeeps.saturating_add(1);
                            }
                            if m.locked.is_some() {
                                m.locked_upkeeps = m.locked_upkeeps.saturating_add(1);
                            }
                        }
                    }
                }
            }
            "switch" | "drag" => {
                self.cur_move = None;
                self.on_switch(dex, &parts, cmd == "drag");
            }
            "faint" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    let side = &mut self.sides[s];
                    side.fainted_this_turn = Some(m);
                    side.pending_bp = false;
                    let mon = &mut side.mons[m];
                    mon.fainted = true;
                    mon.pixels = 0;
                    mon.clear_on_exit();
                    self.clear_sourced_vols(s, m, None);
                }
            }
            "move" => self.on_move(dex, &parts),
            "cant" => {
                self.cur_move = None;
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.sides[s].acted_this_turn = true;
                    self.sides[s].pending_bp = false;
                    let mon = &mut self.sides[s].mons[m];
                    mon.locked = None;
                    mon.called_locked = false;
                    match arg(2) {
                        "slp" => {
                            mon.slept += 1;
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
                    // a confusion self-hit spends the turn's move action and
                    // drops any pending two-turn charge or lock
                    if parts.iter().any(|p| *p == "[from] confusion") {
                        self.sides[s].acted_this_turn = true;
                        self.sides[s].pending_bp = false;
                        let mon = &mut self.sides[s].mons[m];
                        mon.charging = None;
                        mon.locked = None;
                        mon.called_locked = false;
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
                if let Some(s) =
                    self.subject(arg(1)).map(|(s, _)| s).or_else(|| Self::side_subject(arg(1)))
                {
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
                    let mon = &mut self.sides[s].mons[m];
                    if key == "transform" {
                        mon.transformed_into = None;
                        mon.transformed_uses.clear();
                    }
                    mon.vols.retain(|v| v.key != key);
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
                                upkeeps: 0,
                                acted_on_start: false,
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
                    mon.transformed_uses.clear();
                    mon.boosts = tb;
                }
            }
            "-crit" => {
                if let Some((_, _, crit)) = self.cur_move.as_mut() {
                    *crit = true;
                }
            }
            "-mustrecharge" => {
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.sides[s].mons[m].must_recharge = true;
                }
            }
            "-miss" | "-fail" => {
                // rollout's lock, fury cutter's streak, and any pending Baton Pass end on a whiff
                if let Some((s, m)) = self.subject(arg(1)) {
                    self.sides[s].pending_bp = false;
                    let mon = &mut self.sides[s].mons[m];
                    if let Some((lm, _)) = mon.locked {
                        if dex.moves.key(lm) == "rollout" {
                            mon.locked = None;
                            mon.called_locked = false;
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
                                mon.called_locked = false;
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
                self.clear_sourced_vols(side, old, bp.then_some(slot));
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
        self.cur_move = Some(((s, m), mid, false));
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
        // Stadium 2 Self-KO Clause: `side.lastMove` is set ONLY inside
        // `tryMoveHit` when a `selfdestruct` move executes (clearing the
        // target side's `lastMove` and recording this side's). Setting it on
        // ordinary moves would mis-score mutual KOs (Perish Song, Destiny Bond,
        // Sandstorm, Leech Seed) in both `Battle` and `CompactBattle::encode`.
        if !parts.iter().any(|p| *p == "[notarget]") && dex.move_static(mid).selfdestruct {
            self.sides[1 - s].side_last_move = None;
            self.sides[s].side_last_move = Some(mid);
        }
        {
            let side = &mut self.sides[s];
            let mon = &mut side.mons[m];
            let continuing_called_lock =
                mon.called_locked && mon.locked.is_some_and(|(lm, _)| lm == mid);
            if !called && !continuing_called_lock {
                mon.last_move = if matches!(
                    key,
                    "metronome" | "mimic" | "mirrormove" | "sketch" | "sleeptalk" | "transform"
                ) {
                    None
                } else {
                    Some(mid)
                };
            }
            if key == "batonpass" {
                side.pending_bp = true;
            } else if !called {
                side.pending_bp = false;
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
            // Metronome/Mirror Move calling Outrage/Thrash/Petal Dance/Rollout locks into
            // continuation turns on Turn 2+ (Sleep Talk never starts a multi-turn lock in Gen 2).
            if (THRASH_CLASS.contains(&key) || key == "rollout")
                && from_key.as_deref() != Some("sleeptalk")
            {
                let mon = &mut self.sides[s].mons[m];
                mon.locked = Some((mid, 1));
                mon.locked_upkeeps = 0;
                mon.called_locked = true;
            }
            return;
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
                if uses >= cap {
                    mon.locked = None;
                    mon.locked_upkeeps = 0;
                    mon.called_locked = false;
                } else {
                    mon.locked = Some((lm, uses));
                }
                return;
            }
            mon.locked = None;
            mon.locked_upkeeps = 0;
            mon.called_locked = false;
        }
        // A transformed mon is executing the copied set: track PP usage on its
        // temporary transformed slots without polluting submitted base-set PP.
        if mon.transformed_into.is_some() {
            mon.deduct_transformed(mid);
        } else {
            mon.deduct(mid);
        }
        if THRASH_CLASS.contains(&key) || key == "rollout" {
            mon.locked = Some((mid, 1));
            mon.locked_upkeeps = 0;
            mon.called_locked = false;
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
                        upkeeps: 0,
                        acted_on_start: false,
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
                        upkeeps: 0,
                        acted_on_start: false,
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
                    upkeeps: 0,
                    acted_on_start: false,
                    move_id: None,
                    source: None,
                    counter: Some(n),
                }),
            }
            return;
        }
        let key = toid(what.strip_prefix("move: ").unwrap_or(what));
        // fatigue confusion ends a thrash lock (indistinguishable from other
        // confusion sources — clearing is the conservative reading for PP),
        // and silently replaces any prior confusion counter (`lockedmove` onEnd)
        if key == "confusion" {
            let mon = &mut self.sides[s].mons[m];
            mon.locked = None;
            mon.locked_upkeeps = 0;
            mon.called_locked = false;
            mon.vols.retain(|v| v.key != "confusion");
        }
        if dex.conds_id(&key).is_none() {
            return; // not a runtime condition we can plant (e.g. typechange)
        }
        if key == "substitute" {
            let mon = &mut self.sides[s].mons[m];
            mon.vols.retain(|v| v.key != "substitute");
            mon.vols.push(TVol {
                key,
                start_turn: turn,
                upkeeps: 0,
                acted_on_start: false,
                move_id: None,
                source: Some((s, m)),
                counter: Some(0),
            });
            return;
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
        let acted_on_start = self.sides[s].acted_this_turn;
        let source = self.sides[1 - s].active.map(|a| (1 - s, a));
        let mon = &mut self.sides[s].mons[m];
        if mon.vols.iter().any(|v| v.key == key) {
            return;
        }
        mon.vols.push(TVol {
            key,
            start_turn: turn,
            upkeeps: 0,
            acted_on_start,
            move_id,
            source,
            counter: None,
        });
    }

    fn on_activate(&mut self, dex: &Dex, parts: &[&str]) {
        let Some((s, m)) = self.subject(parts.get(1).copied().unwrap_or("")) else { return };
        let what = parts.get(2).copied().unwrap_or("");
        let turn = self.turn;
        if what == "confusion" {
            self.sides[s].acted_this_turn = true;
            let mon = &mut self.sides[s].mons[m];
            match mon.vols.iter_mut().find(|v| v.key == "confusion") {
                Some(v) => *v.counter.get_or_insert(0) += 1,
                None => mon.vols.push(TVol {
                    key: "confusion".to_string(),
                    start_turn: turn,
                    upkeeps: 0,
                    acted_on_start: false,
                    move_id: None,
                    source: None,
                    counter: Some(1),
                }),
            }
            return;
        }
        if what == "Substitute" && parts.iter().any(|p| *p == "[damage]") {
            let dmg = match self.cur_move {
                Some((atk, mid, crit)) => {
                    estimate_sub_damage(dex, &self.sides, &self.weather, atk, (s, m), mid, crit)
                }
                None => 1,
            };
            let mon = &mut self.sides[s].mons[m];
            match mon.vols.iter_mut().find(|v| v.key == "substitute") {
                Some(v) => *v.counter.get_or_insert(0) += dmg,
                None => mon.vols.push(TVol {
                    key: "substitute".to_string(),
                    start_turn: turn,
                    upkeeps: 0,
                    acted_on_start: false,
                    move_id: None,
                    source: Some((s, m)),
                    counter: Some(dmg),
                }),
            }
            return;
        }
        if what == "trapped" {
            let source = self.sides[1 - s].active.map(|a| (1 - s, a));
            let mon = &mut self.sides[s].mons[m];
            if !mon.vols.iter().any(|v| v.key == "trapped") {
                mon.vols.push(TVol {
                    key: "trapped".to_string(),
                    start_turn: turn,
                    upkeeps: 0,
                    acted_on_start: false,
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
                        mon.called_locked = false;
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
                let list = if mon.transformed_into.is_some() {
                    &mut mon.transformed_uses
                } else {
                    &mut mon.uses
                };
                if let Some((_, n)) = list.iter_mut().find(|(id, _)| *id == mid) {
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
                    let list = if mon.transformed_into.is_some() {
                        &mut mon.transformed_uses
                    } else {
                        &mut mon.uses
                    };
                    match list.iter_mut().find(|(id, _)| *id == mid) {
                        Some((_, u)) => *u += n,
                        None => list.push((mid, n)),
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
                        upkeeps: 0,
                        acted_on_start: false,
                        move_id: mid,
                        source,
                        counter: None,
                    });
                }
            }
        }
    }

    /// Defensive: linked volatiles die with their source (mean look /
    /// binding) — remove them when the source leaves the field, except
    /// Trap-Passing via Baton Pass which transfers `"trapped"` ownership
    /// to the incoming recipient (`bp_new_slot`). Attract's end is always
    /// announced (`-end ... [silent]` is still a line), so it is NOT cleared.
    fn clear_sourced_vols(&mut self, src_side: usize, src_slot: usize, bp_new_slot: Option<usize>) {
        for s in 0..2 {
            for mon in self.sides[s].mons.iter_mut() {
                mon.vols.retain_mut(|v| {
                    if v.source != Some((src_side, src_slot)) {
                        return true;
                    }
                    match v.key.as_str() {
                        "trapped" => {
                            if let Some(new_slot) = bp_new_slot {
                                v.source = Some((src_side, new_slot));
                                true
                            } else {
                                false
                            }
                        }
                        "partiallytrapped" => false,
                        _ => true,
                    }
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

/// Conservative Gen 2 damage estimate for `|-activate|...|Substitute|[damage]`
/// (PS hides exact Substitute damage). Uses exact public formula for fixed-damage
/// moves and the conservative min roll (`217/255`) on standard max-StatExp/DV
/// Gen 2 stats so surviving Substitutes lose realistic HP without prematurely
/// reaching 0.
fn estimate_sub_damage(
    dex: &Dex,
    sides: &[TrackSide; 2],
    weather: &Option<(String, u16)>,
    atk: (usize, usize),
    def: (usize, usize),
    mid: MoveId,
    crit: bool,
) -> i64 {
    const POS_BOOST: [f64; 7] = [1.0, 1.33, 1.66, 2.0, 2.33, 2.66, 3.0];
    const NEG_BOOST: [f64; 7] = [1.0, 0.75, 0.6, 0.5, 0.43, 0.36, 0.33];

    let Some(att_mon) = sides.get(atk.0).and_then(|s| s.mons.get(atk.1)) else { return 1 };
    let Some(def_mon) = sides.get(def.0).and_then(|s| s.mons.get(def.1)) else { return 1 };
    let ms = dex.move_static(mid);
    if let Some(fixed) = ms.damage {
        return match fixed {
            FixedDamage::Level => (att_mon.level as i64).max(1),
            FixedDamage::Amount(n) => (n as i64).max(1),
        };
    }
    if ms.category == Category::Status {
        return 1;
    }
    let key = dex.moves.key(mid);
    if key == "superfang" {
        let def_sp = match def_mon.transformed_into {
            Some((ts, tslot)) => sides
                .get(ts)
                .and_then(|s| s.mons.get(tslot))
                .map(|m| m.species)
                .unwrap_or(def_mon.species),
            None => def_mon.species,
        };
        let def_data = dex.species.get(def_sp);
        let def_maxhp =
            (((2 * def_data.base_stats.hp as i32) * def_mon.level as i32) / 100
                + def_mon.level as i32
                + 10)
            .max(1);
        return ((def_maxhp / 2) as i64).max(1);
    }
    let bp = if ms.base_power > 0 {
        ms.base_power
    } else if key == "return" {
        102
    } else if key == "hiddenpower" {
        31
    } else if key.starts_with("hiddenpower") {
        70
    } else {
        return 1;
    };

    let (att_sp, att_lv) = match att_mon.transformed_into {
        Some((ts, tslot)) => sides
            .get(ts)
            .and_then(|s| s.mons.get(tslot))
            .map(|m| (m.species, att_mon.level))
            .unwrap_or((att_mon.species, att_mon.level)),
        None => (att_mon.species, att_mon.level),
    };
    let (def_sp, def_lv) = match def_mon.transformed_into {
        Some((ts, tslot)) => sides
            .get(ts)
            .and_then(|s| s.mons.get(tslot))
            .map(|m| (m.species, def_mon.level))
            .unwrap_or((def_mon.species, def_mon.level)),
        None => (def_mon.species, def_mon.level),
    };
    let att_data = dex.species.get(att_sp);
    let def_data = dex.species.get(def_sp);
    let is_phys = ms.category == Category::Physical;
    let stat_at = |base: u16, lv: u8| -> i32 {
        (((2 * base as i32 + 30 + 63) * lv as i32) / 100 + 5).max(1)
    };
    let raw_atk = stat_at(
        if is_phys { att_data.base_stats.atk } else { att_data.base_stats.spa },
        att_lv,
    );
    let raw_def = stat_at(
        if is_phys { def_data.base_stats.def } else { def_data.base_stats.spd },
        def_lv,
    );
    let atk_stage = att_mon.boosts[if is_phys { 0 } else { 2 }].clamp(-6, 6);
    let def_stage = def_mon.boosts[if is_phys { 1 } else { 3 }].clamp(-6, 6);
    let ignore_boosts = crit && atk_stage <= def_stage;
    let boost_mul = |st: i8| -> f64 {
        if st >= 0 { POS_BOOST[st as usize] } else { NEG_BOOST[(-st) as usize] }
    };
    let mut a = if ignore_boosts {
        raw_atk
    } else {
        ((raw_atk as f64 * boost_mul(atk_stage)).floor() as i32).clamp(1, 999)
    };
    let mut d = if ignore_boosts {
        raw_def
    } else {
        ((raw_def as f64 * boost_mul(def_stage)).floor() as i32).clamp(1, 999)
    };
    if !ignore_boosts && is_phys && att_mon.comp_brn {
        a = (a / 2).max(1);
    }
    if !ignore_boosts {
        let screen = if is_phys { "reflect" } else { "lightscreen" };
        if sides[def.0].conds.iter().any(|(k, _)| k == screen) {
            d *= 2;
        }
    }
    if ms.selfdestruct && is_phys {
        d = (d / 2).max(1);
    }
    if a > 255 || d > 255 {
        a = (a >> 2).max(1);
        d = (d >> 2).max(1);
    }
    let mut base_dmg = (((2 * att_lv as i32 / 5 + 2) * bp * a) / d) / 50;
    if crit {
        base_dmg *= 2;
    }
    let mv_type_name = dex.type_name(ms.move_type);
    let mv_type = toid(mv_type_name);
    if let Some((w, _)) = weather {
        if (w == "raindance" && mv_type == "water") || (w == "sunnyday" && mv_type == "fire") {
            base_dmg = base_dmg * 3 / 2;
        } else if (w == "raindance" && mv_type == "fire")
            || (w == "sunnyday" && mv_type == "water")
        {
            base_dmg /= 2;
        }
    }
    base_dmg = base_dmg.min(997) + 2;
    if key == "hiddenpower" {
        // Un-typed Hidden Power can hit any non-immune type at conservative 0.25x min; never apply STAB.
        base_dmg = (base_dmg / 4).max(1);
    } else {
        if att_data.types.iter().any(|t| toid(t) == mv_type) {
            base_dmg += base_dmg / 2;
        }
        for dt in &def_data.types {
            match dex.get_effectiveness(mv_type_name, dt) {
                1 => base_dmg *= 2,
                -1 => base_dmg /= 2,
                _ => {}
            }
        }
    }
    if base_dmg <= 1 {
        return 1;
    }
    if !ms.no_damage_variance {
        base_dmg = (base_dmg * 217) / 255;
    }
    (base_dmg as i64).max(1)
}

// =================================================================== request

/// Parsed PS request JSON (the player-visible own-side truth).
#[derive(Clone, Debug, Default)]
pub struct Request {
    pub team_preview: bool,
    /// Public local format rules; absent on PS means the standard 155 cap.
    pub preview_level_caps: Option<[Option<u32>; 2]>,
    pub force_switch: bool,
    pub wait: bool,
    pub trapped: bool,
    pub maybe_trapped: bool,
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
        req.preview_level_caps = v["side"].get("previewLevelCaps")
            .map(|caps| serde_json::from_value(caps.clone()).map_err(|e|format!("preview caps: {e}")))
            .transpose()?;
        if let Some(active) = v["active"].as_array().and_then(|a| a.first()) {
            req.trapped = active["trapped"].as_bool().unwrap_or(false);
            req.maybe_trapped = active["maybeTrapped"].as_bool().unwrap_or(false);
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
        b.preview_level_caps=req.preview_level_caps.unwrap_or([Some(MAX_TOTAL_LEVEL as u32);2]);

        // opponent roster := belief refs (fresh preview-state mons aligned
        // to observed slots), with public preview facts stamped on
        if refs.len() != self.sides[opp].mons.len() {
            return Err("refs/public roster length mismatch".to_string());
        }
        b.sides[opp].roster = refs.to_vec();
        b.sides[opp].party=(0..refs.len() as u8).collect();
        b.sides[opp].pokemon_left=refs.len() as i32;
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
                .enumerate()
                .find(|(slot,p)| !own_party.contains(&(*slot as u8)) && p.species == rm.species && p.level == rm.level)
                .map(|(slot,_)|slot)
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
                    if !tm.transformed_uses.is_empty() {
                        let p = b.poke_mut(id);
                        for &(mid, n) in &tm.transformed_uses {
                            for ms in p.move_slots.iter_mut() {
                                if move_matches(dex, ms.id, mid) {
                                    ms.pp = (ms.pp - n).max(0);
                                    ms.used = true;
                                }
                            }
                        }
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
                if p.transformed {
                    let real_req: Vec<&ReqMove> = req
                        .active_moves
                        .iter()
                        .filter(|rm| !matches!(toid(&rm.id).as_str(), "struggle" | "recharge"))
                        .take(4)
                        .collect();
                    if !real_req.is_empty()
                        && !(real_req.len() == 1 && real_req[0].maxpp.is_none())
                    {
                        let old_slots = p.move_slots;
                        let mut rebuilt = MoveSlots::default();
                        for rm in real_req {
                            let Some(mid) = dex.moves.id(&toid(&rm.id)) else { continue };
                            let existing =
                                old_slots.iter().find(|ms| move_matches(dex, ms.id, mid)).copied();
                            let slot_id = existing.map(|ms| ms.id).unwrap_or(mid);
                            let maxpp = rm.maxpp.unwrap_or_else(|| {
                                existing
                                    .map(|ms| ms.maxpp)
                                    .unwrap_or_else(|| dex.move_static(slot_id).pp.min(5))
                            });
                            let pp = rm
                                .pp
                                .unwrap_or_else(|| existing.map(|ms| ms.pp).unwrap_or(maxpp));
                            rebuilt.push(MoveSlot {
                                id: slot_id,
                                pp,
                                maxpp,
                                disabled: rm.disabled,
                                used: pp < maxpp || existing.map_or(false, |ms| ms.used),
                                shared: false,
                            });
                        }
                        if !rebuilt.is_empty() {
                            p.move_slots = rebuilt;
                        }
                    } else {
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
                            }
                        }
                    }
                } else {
                    for rm in &req.active_moves {
                        if let Some(mid) = dex.moves.id(&toid(&rm.id)) {
                            let mut mirror_base = false;
                            for ms in p.move_slots.iter_mut() {
                                if move_matches(dex, ms.id, mid) {
                                    if let Some(pp) = rm.pp {
                                        ms.pp = pp;
                                    }
                                    if let Some(maxpp) = rm.maxpp {
                                        ms.maxpp = maxpp;
                                    }
                                    ms.disabled = rm.disabled;
                                    if ms.shared {
                                        mirror_base = true;
                                    }
                                }
                            }
                            // mirror into shared base slots (PS shares objects, except Transform/Mimic)
                            if mirror_base {
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
                    }
                }
                if req.active_moves.len() == 1 && toid(&req.active_moves[0].id) == "struggle" {
                    for ms in p.move_slots.iter_mut() {
                        if !ms.disabled {
                            ms.pp = 0;
                        }
                    }
                }
                p.trapped = req.trapped;
                p.maybe_trapped = req.maybe_trapped;
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
                    let forced = b
                        .get_locked_move(dex, fa)
                        .or_else(|| b.get_semi_locked_move(dex, fa))
                        .and_then(|k| dex.moves.id(k))
                        .or_else(|| {
                            dex.conds_id("encore")
                                .and_then(|c| b.poke(fa).volatile(c))
                                .and_then(|v| v.get_move())
                                .filter(|&m| {
                                    b.poke(fa).move_slots.iter().any(|s| s.id == m && s.pp > 0)
                                })
                        });
                    let mid = if let Some(mid) = forced {
                        mid
                    } else {
                        let usable: Vec<MoveId> = b
                            .poke(fa)
                            .move_slots
                            .iter()
                            .filter(|s| s.pp > 0 && !s.disabled)
                            .map(|s| s.id)
                            .collect();
                        if usable.is_empty() {
                            dex.moves.id("struggle").ok_or("struggle id")?
                        } else {
                            usable[rng.below(usable.len())]
                        }
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
                    if dex.moves.key(mid) == "pursuit" {
                        if let Some(target) = b.active_id(me) {
                            b.before_turn_callback(dex, mid, fa, target);
                        }
                    }
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
        if kind == RequestState::Move {
            if let Some(active) = b.active_id(opp) {
                b.run_event(
                    dex, &ev::TrapPokemon, EvTarget::Poke(active), None,
                    EffectHandle::None, None, false, false,
                );
            }
        }
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
        let self_inflicted = tm.status == Status::Slp && tm.rest;
        let source = Some(if self_inflicted { id } else { PokeId { side: 1 - id.side, slot: 0 } });
        b.restore_status(dex, id, tm.status, source);
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
                "substitute" => {
                    let src = src_id(v.source);
                    b.add_volatile(dex, id, "substitute", src, EffectHandle::None);
                    let creator = src.unwrap_or(id);
                    let init_hp = (b.poke(creator).base_maxhp / 4).max(1) as i64;
                    let dmg = v.counter.unwrap_or(0).max(0);
                    let rem = (init_hp - dmg).clamp(1, init_hp);
                    if let Some(c) = dex.conds_id("substitute") {
                        if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                            vs.set_int(DK::Hp, rem);
                        }
                    }
                }
                "confusion" => {
                    let src = src_id(v.source);
                    b.add_volatile(dex, id, "confusion", src, EffectHandle::None);
                    let ticks = v.counter.unwrap_or(0);
                    if ticks > 0 {
                        if let Some(c) = dex.conds_id("confusion") {
                            if let Some(vs) = b.poke_mut(id).volatile_mut(c) {
                                if let Some(t) = vs.get(DK::Time).map(|s| s.as_int()) {
                                    vs.set_int(DK::Time, (t - ticks).max(1));
                                }
                            }
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

    pub(crate) fn hidden_timer_specs(
        &self,
        dex: &Dex,
        battle: &Battle,
    ) -> Vec<crate::player_state::HiddenTimerSpec> {
        use crate::player_state::{HiddenTimerKind, HiddenTimerSpec};
        let mut out = Vec::new();
        for s in 0..2 {
            let len = self.sides[s].mons.len().min(battle.sides[s].roster.len());
            for slot in 0..len {
                let p = &battle.sides[s].roster[slot];
                let tm = &self.sides[s].mons[slot];
                if p.hp == 0 || tm.fainted {
                    continue;
                }
                let side = s as u8;
                let slot_u8 = slot as u8;
                if tm.status == Status::Slp && !tm.rest && p.status == Status::Slp {
                    out.push(HiddenTimerSpec {
                        side,
                        slot: slot_u8,
                        kind: HiddenTimerKind::Sleep {
                            slept: tm.slept.min(255) as u8,
                        },
                    });
                }
                if !p.is_active {
                    continue;
                }
                for v in &tm.vols {
                    let Some(cid) = dex.conds_id(&v.key) else {
                        continue;
                    };
                    if p.volatile(cid).is_none() {
                        continue;
                    }
                    let kind = match v.key.as_str() {
                        "confusion" => HiddenTimerKind::Confusion {
                            ticks: v.counter.unwrap_or(0).clamp(0, 255) as u8,
                        },
                        "encore" => HiddenTimerKind::Encore { upkeeps: v.upkeeps },
                        "disable" => HiddenTimerKind::Disable {
                            upkeeps: v.upkeeps,
                            acted_on_start: v.acted_on_start,
                        },
                        "partiallytrapped" => {
                            HiddenTimerKind::PartiallyTrapped { upkeeps: v.upkeeps }
                        }
                        "bide" => HiddenTimerKind::Bide { upkeeps: v.upkeeps },
                        _ => continue,
                    };
                    out.push(HiddenTimerSpec {
                        side,
                        slot: slot_u8,
                        kind,
                    });
                }
                if let Some((m, _)) = tm.locked {
                    if dex.moves.key(m) != "rollout"
                        && dex.conds_id("lockedmove").is_some_and(|c| p.volatile(c).is_some())
                    {
                        out.push(HiddenTimerSpec {
                            side,
                            slot: slot_u8,
                            kind: HiddenTimerKind::LockedMove {
                                upkeeps: tm.locked_upkeeps,
                            },
                        });
                    }
                }
            }
        }
        out
    }
}

/// party := `slots` (display order); positions canonical (party index for
/// members, then bench parked in roster order — the M10 determinizer scheme).
pub(crate) fn apply_party(b: &mut Battle, s: usize, slots: &[u8]) {
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
pub fn impute_hp(cur: i32, den: i32, maxhp: i32) -> i32 {
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

/// Re-announce an HP amount the way the stream that produced `den` would —
/// the forward direction [`impute_hp`] inverts. HP Percentage Mod (den = 100)
/// prints `ceil(100*hp/maxhp)` with a not-quite-full value knocked down to 99;
/// the legacy pixel bar (den = 48) prints `floor(48*hp/maxhp)` floored to 1
/// while alive.
///
/// Exists so the `%`-HP invariant is checkable rather than argued: for every
/// announced `cur`, `announce_hp(impute_hp(cur, den, maxhp), maxhp, den)` must
/// be `cur` again. That round trip is exactly what the 2026-07-21 bug broke —
/// the importer assumed a /48 pixel bar on a /100 stream, inflating every foe
/// HP by ~2.08x — and it is the M18 class-A gate's second assertion.
pub fn announce_hp(hp: i32, maxhp: i32, den: i32) -> i32 {
    let den = if den > 0 { den } else { 48 };
    if hp <= 0 {
        return 0;
    }
    if maxhp <= 0 || hp >= maxhp {
        return den;
    }
    if den == 100 {
        ((100 * hp + maxhp - 1) / maxhp).clamp(1, 99)
    } else {
        (den * hp / maxhp).clamp(1, den - 1)
    }
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
    /// M18 community belief prior for the hidden-team fallback imputation.
    /// `None` (the default) leaves it exactly as shipped.
    prior: Option<std::sync::Arc<crate::prior::BeliefPrior>>,
    fallback_policy: crate::belief::FallbackPolicy,
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
            prior: None,
            fallback_policy: crate::belief::FallbackPolicy::default(),
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

    pub fn set_fallback_policy(&mut self, policy: crate::belief::FallbackPolicy) {
        self.fallback_policy = policy;
    }

    /// Open-team-sheet mode: the opponent's true sets are public.
    pub fn pin_opponent(&mut self, dex: &Dex, sets: Vec<PokemonSet>) -> Result<(), String> {
        if sets.is_empty() {
            return Err("pinned opponent team is empty".to_string());
        }
        if let Some(obs) = self.observer.as_ref() {
            let belief = Belief::pinned_checked(dex, "opponent", &sets, obs)?;
            self.belief = Some(belief);
        }
        self.pinned_sets = Some(sets);
        if let (Some(_), Some(req)) = (self.observer.as_ref(), self.request.clone()) {
            self.install(dex, req)?;
        }
        Ok(())
    }

    /// M18: consume a community belief prior. Governs only the hidden-team
    /// fallback imputation; a pinned (open-sheet) belief ignores it.
    pub fn set_belief_prior(&mut self, prior: std::sync::Arc<crate::prior::BeliefPrior>) {
        self.prior = Some(prior);
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
            let mut belief = match &self.pinned_sets {
                Some(sets) => Belief::pinned_checked(dex, "opponent", sets, &obs)?,
                None => Belief::with_fallback_policy(dex, &self.pool, &obs, self.fallback_policy),
            };
            if let (Some(prior), None) = (self.prior.clone(), &self.pinned_sets) {
                belief.set_prior(prior);
            }
            self.observer = Some(obs);
            self.belief = Some(belief);
        }
        let unlocked_choices = req
            .active_moves
            .iter()
            .filter(|m| !m.disabled && m.pp != Some(0))
            .count();
        if !req.wait && !req.force_switch && !req.team_preview && unlocked_choices > 1 {
            if let Some(slot) = self.tracker.sides[self.side].active {
                if let Some(mon) = self.tracker.sides[self.side].mons.get_mut(slot) {
                    mon.locked = None;
                    mon.called_locked = false;
                }
            }
        }
        self.install(dex, req)
    }

    /// Everything a decision point needs once the observer and belief
    /// exist and the request is parsed: belief sync, synthesis, and the
    /// searcher plus its preview/forced shortcuts. Shared by the live
    /// protocol path (`on_request`) and the hand-entered one
    /// (`set_position`), so the two can never drift apart.
    fn install(&mut self, dex: &Dex, req: Request) -> Result<bool, String> {
        let obs = self.observer.as_ref().unwrap();
        let belief = self.belief.as_mut().unwrap();
        belief.sync_checked(dex, obs)?;

        // Preserve player HP buckets across every candidate determinization.
        self.tracker.set_hp_observations(belief, 1-self.side);
        // synthesize
        let pick = belief.alive().first().copied();
        let battle = {
            let refs = belief.refs(pick);
            self.tracker.synthesize(dex, &self.own_sets, refs, obs, &req, &mut self.rng)?
        };

        let specs = self.tracker.hidden_timer_specs(dex, &battle);
        let pos = crate::player_state::ObservedPosition::reconstructed_with_timers(
            battle.clone(),
            specs,
        );
        // search + preview policy
        let mut search =
            BlindSearch::from_observation(&pos, dex, self.cfg.clone(), self.side, self.rng.next());
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
                    SearchChoice::Team(slots) => battle.preview_level_caps[self.side]
                        .map_or(true, |cap| team_total_level(&levels, slots) <= cap as i32),
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
            if self.baked.is_none() {
                search.prune_preview_aliases(dex);
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

    pub fn step_with_endgame(
        &mut self,
        dex: &Dex,
        n: u32,
        session: &mut crate::endgame::EndgameSession,
        clock: &mut impl FnMut() -> f64,
    ) -> Result<u32, String> {
        let search = self.search.as_mut().ok_or("step before on_request")?;
        if self.baked.is_some() || self.forced.is_some() { return Ok(search.iterations()); }
        let belief = self.belief.as_ref().ok_or("no belief")?;
        let obs = self.observer.as_ref().ok_or("no observer")?;
        Ok(search.step_with_endgame(dex, belief, obs, n, session, clock))
    }

    pub fn step_observed(
        &mut self,
        dex: &Dex,
        n: u32,
        trace: &mut impl FnMut(crate::smmcts::SearchTrace<'_>),
    ) -> Result<u32, String> {
        let search = self.search.as_mut().ok_or("step before on_request")?;
        if self.baked.is_some() || self.forced.is_some() {
            return Ok(search.iterations());
        }
        let belief = self.belief.as_ref().ok_or("no belief")?;
        let obs = self.observer.as_ref().ok_or("no observer")?;
        Ok(search.step_observed(dex, belief, obs, n, trace))
    }

    /// The callback returns a side-0 reward in [0, 1]; the flag marks a newly expanded leaf.
    pub fn step_with_leaf(
        &mut self,
        dex: &Dex,
        n: u32,
        leaf: &mut impl FnMut(&mut Battle, &mut SplitMix64, bool) -> f64,
    ) -> Result<u32, String> {
        let search = self.search.as_mut().ok_or("step before on_request")?;
        if self.baked.is_some() || self.forced.is_some() {
            return Ok(search.iterations());
        }
        let belief = self.belief.as_ref().ok_or("no belief")?;
        let obs = self.observer.as_ref().ok_or("no observer")?;
        Ok(search.step_with_leaf(dex, belief, obs, n, leaf))
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
        let search = self.search.as_ref()?;
        let best = ps_input(dex, search.best()?);
        let req = self.request.as_ref()?;
        if req.team_preview {
            return Some(best); // level mask already applied at the root
        }
        let legal = req.legal_inputs();
        if legal.is_empty() || legal.contains(&best) {
            return Some(best);
        }
        self.projections += 1;
        let acts = search.actions();
        let visits = search.visits();
        let means = search.means();
        let fallback = acts
            .iter()
            .enumerate()
            .filter_map(|(i, &act)| {
                let inp = ps_input(dex, act);
                legal.contains(&inp).then_some((visits[i], means[i], inp))
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)))
            .map(|(_, _, inp)| inp);
        Some(fallback.unwrap_or_else(|| legal[0].clone()))
    }

    /// Execution distribution in request syntax. Project the whole policy
    /// before sampling, merging aliases and displaced mass; never discard
    /// probability after the draw. Preview tables have already sampled in
    /// the legacy lifecycle and are explicitly labelled as such.
    pub fn decision_policy(&self, dex: &Dex, mode: crate::decision::PolicyMode) -> Option<crate::decision::DecisionPolicy<String>> {
        use crate::decision::DecisionPolicy;
        if let Some(input) = &self.forced { return Some(DecisionPolicy::pure("forced", input.clone())); }
        if let Some(action) = self.baked { return Some(DecisionPolicy::pure("preview-sampled", action.to_input(dex))); }
        let search = self.search.as_ref()?;
        let policy = search.decision_policy(mode)?.map(|&action| ps_input(dex, action));
        let req = self.request.as_ref()?;
        if req.team_preview { return Some(policy); }
        let legal = req.legal_inputs();
        if legal.is_empty() { return None; }
        let projected = policy.actions().iter().any(|row| !legal.contains(&row.input));
        if !projected {
            return Some(policy);
        }
        let kept_mass: f64 = policy
            .actions()
            .iter()
            .filter(|row| legal.contains(&row.input))
            .map(|row| row.probability)
            .sum();
        if kept_mass > 0.0 {
            return DecisionPolicy::new(
                "request-projected",
                policy
                    .actions()
                    .iter()
                    .filter(|row| legal.contains(&row.input))
                    .map(|row| (row.input.clone(), row.probability)),
            )
            .ok();
        }
        let acts = search.actions();
        let visits = search.visits();
        let means = search.means();
        let target = acts
            .iter()
            .enumerate()
            .filter_map(|(i, &act)| {
                let inp = ps_input(dex, act);
                legal.contains(&inp).then_some((visits[i], means[i], inp))
            })
            .max_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.total_cmp(&b.1)))
            .map(|(_, _, inp)| inp)
            .unwrap_or_else(|| legal[0].clone());
        Some(DecisionPolicy::pure("request-projected", target))
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
                            "input": ps_input(dex, a),
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
    sample_filtered(tables, p, levels, battle.preview_level_caps[side], rng)
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
    sample_filtered(tables, p, levels, battle.preview_level_caps[side], rng)
}

/// Sample a mixed preview policy conditioned on the PS-legal action subset
/// (renormalized; degenerate rows fall back to the legal argmax).
fn sample_filtered(
    tables: &TableSet,
    p: &[f64],
    levels: &[i32],
    cap: Option<u32>,
    rng: &mut SplitMix64,
) -> Option<SearchChoice> {
    let actions = tables.actions();
    let legal: Vec<usize> = (0..p.len())
        .filter(|&i| cap.is_none_or(|c| team_total_level(levels, &actions[i]) <= c as i32))
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

#[cfg(test)]
mod m17a_tests {
    use super::*;

    #[test]
    fn diagnostic_policy_uses_ps_canonical_hidden_power() {
        let dex = conformance::load_dex();
        let ice = dex.moves.id("hiddenpowerice").unwrap();
        assert_eq!(ps_input(&dex, SearchChoice::Move(ice)), "move hiddenpower");
    }

    #[test]
    fn transformed_moves_keep_charge_and_lock_lifecycle_without_base_pp_evidence() {
        let dex = conformance::load_dex();
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Ditto, L50|",
            "|poke|p2|Snorlax, L50, M|",
            "|switch|p1a: Ditto|Ditto, L50|100/100",
            "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
            "|move|p1a: Ditto|Transform|p2a: Snorlax",
            "|-transform|p1a: Ditto|p2a: Snorlax",
            "|move|p1a: Ditto|Solar Beam|p2a: Snorlax",
            "|-prepare|p1a: Ditto|Solar Beam|p2a: Snorlax",
        ] {
            tracker.push_line(&dex, line);
        }
        let solar_beam = dex.moves.id("solarbeam").unwrap();
        assert_eq!(tracker.sides[0].mons[0].charging, Some(solar_beam));

        tracker.push_line(&dex, "|move|p1a: Ditto|Solar Beam|p2a: Snorlax");
        assert_eq!(tracker.sides[0].mons[0].charging, None);

        let thrash = dex.moves.id("thrash").unwrap();
        tracker.push_line(&dex, "|move|p1a: Ditto|Thrash|p2a: Snorlax");
        assert_eq!(tracker.sides[0].mons[0].locked, Some((thrash, 1)));
        tracker.push_line(&dex, "|move|p1a: Ditto|Thrash|p2a: Snorlax");
        assert_eq!(tracker.sides[0].mons[0].locked, Some((thrash, 2)));
        tracker.push_line(&dex, "|move|p1a: Ditto|Thrash|p2a: Snorlax");
        assert_eq!(tracker.sides[0].mons[0].locked, None);

        let uses: Vec<_> = tracker.sides[0].mons[0]
            .uses
            .iter()
            .map(|(id, _)| dex.moves.key(*id))
            .collect();
        assert_eq!(uses, ["transform"]);
        let t_uses: Vec<_> = tracker.sides[0].mons[0]
            .transformed_uses
            .iter()
            .map(|(id, n)| (dex.moves.key(*id), *n))
            .collect();
        assert_eq!(t_uses, [("solarbeam", 1), ("thrash", 1)]);
    }

    fn test_set(species: &str, level: u8, item: &str, moves: &[&str]) -> PokemonSet {
        serde_json::from_value(serde_json::json!({
            "name": species,
            "species": species,
            "item": item,
            "moves": moves,
            "level": level,
            "ability": "No Ability",
            "nature": "Hardy",
            "evs": {"hp": 255, "atk": 255, "def": 255, "spa": 255, "spd": 255, "spe": 255},
            "ivs": {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
            "happiness": 255,
        }))
        .unwrap()
    }

    fn canonical_team(dex: &Dex, raw: Vec<PokemonSet>) -> Vec<PokemonSet> {
        let ls_text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/learnsets-gen2.json"),
        )
        .unwrap();
        let ls = nc2000_engine::validate::Learnsets::from_json(&ls_text).unwrap();
        let res = nc2000_engine::validate::canonicalize_team(
            dex,
            &ls,
            &serde_json::to_string(&raw).unwrap(),
        );
        assert_eq!(res["ok"], serde_json::json!(true), "bad test team: {res}");
        serde_json::from_value(res["team"].clone()).unwrap()
    }

    fn poke_lines(dex: &Dex, tag: &str, team: &[PokemonSet]) -> Vec<String> {
        let b = Battle::from_fixture(dex, "1,2,3,4", team, team).unwrap();
        b.sides[0]
            .roster
            .iter()
            .map(|p| {
                let sp_name = &dex.species.get(p.species).name;
                let g = match p.gender.as_str() {
                    "" => String::new(),
                    s => format!(", {s}"),
                };
                let item = if p.item.is_some() { "item" } else { "" };
                format!("|poke|{tag}|{sp_name}, L{}{g}|{item}", p.level)
            })
            .collect()
    }

    #[test]
    fn side_last_move_tracks_only_selfdestruct_for_selfko_clause() {
        let dex = conformance::load_dex();
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Gengar, L50, M|item",
            "|poke|p2|Misdreavus, L50, F|item",
            "|switch|p1a: Gengar|Gengar, L50, M|100/100",
            "|switch|p2a: Misdreavus|Misdreavus, L50, F|100/100",
            "|turn|1",
            "|move|p1a: Gengar|Destiny Bond|p1a: Gengar",
            "|move|p2a: Misdreavus|Perish Song|p2a: Misdreavus",
        ] {
            tracker.push_line(&dex, line);
        }
        assert_eq!(tracker.sides[0].side_last_move, None);
        assert_eq!(tracker.sides[1].side_last_move, None);

        let boom = dex.moves.id("explosion").unwrap();
        tracker.push_line(&dex, "|move|p1a: Gengar|Explosion|p2a: Misdreavus");
        assert_eq!(tracker.sides[0].side_last_move, Some(boom));
        assert_eq!(tracker.sides[1].side_last_move, None);

        // Subsequent non-selfdestruct moves leave Self-KO bookkeeping untouched;
        // another selfdestruct clears the target side and records the new user.
        let selfdestruct = dex.moves.id("selfdestruct").unwrap();
        tracker.push_line(
            &dex,
            "|move|p2a: Misdreavus|Self-Destruct|p1a: Gengar|[from] Metronome",
        );
        assert_eq!(tracker.sides[0].side_last_move, None);
        assert_eq!(tracker.sides[1].side_last_move, Some(selfdestruct));
    }

    #[test]
    fn sleep_talk_and_called_moves_clear_encore_disable_last_move() {
        let dex = conformance::load_dex();
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Snorlax, L55, M|item",
            "|poke|p2|Alakazam, L50, M|item",
            "|switch|p1a: Snorlax|Snorlax, L55, M|100/100",
            "|switch|p2a: Alakazam|Alakazam, L50, M|100/100",
            "|turn|1",
            "|move|p1a: Snorlax|Body Slam|p2a: Alakazam",
        ] {
            tracker.push_line(&dex, line);
        }
        assert_eq!(
            tracker.sides[0].mons[0].last_move,
            dex.moves.id("bodyslam")
        );
        for line in [
            "|turn|2",
            "|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax",
            "|move|p1a: Snorlax|Earthquake|p2a: Alakazam|[from] Sleep Talk",
        ] {
            tracker.push_line(&dex, line);
        }
        assert_eq!(tracker.sides[0].mons[0].last_move, None);
    }

    #[test]
    fn confusion_self_hit_and_failed_baton_pass_keep_turn_flow_consistent() {
        let dex = conformance::load_dex();
        let pool = MetaPool { teams: Vec::new() };
        let own = canonical_team(
            &dex,
            vec![
                test_set("Jolteon", 50, "Leftovers", &["Thunderbolt", "Agility", "Baton Pass", "Hidden Power"]),
                test_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
                test_set("Exeggutor", 50, "Mint Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
                test_set("Gengar", 50, "Gold Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Cloyster", 50, "Mystery Berry", &["Surf", "Spikes", "Explosion", "Icy Wind"]),
            ],
        );
        let foe = canonical_team(
            &dex,
            vec![
                test_set("Machamp", 50, "Quick Claw", &["Cross Chop", "Rock Slide", "Curse", "Rest"]),
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Self-Destruct", "Curse"]),
                test_set("Suicune", 50, "Miracle Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Gold Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Forretress", 50, "Pink Bow", &["Spikes", "Rapid Spin", "Explosion", "Hidden Power"]),
            ],
        );
        let mut agent = ProtocolAgent::new(&dex, 0, pool, RmConfig::default(), 42);
        agent.set_own_team(own.clone());
        agent.pin_opponent(&dex, foe.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe))
        {
            agent.push_line(&dex, &line);
        }
        for line in [
            "|switch|p1a: Jolteon|Jolteon, L50, M|100/100",
            "|switch|p2a: Machamp|Machamp, L50, M|100/100",
            "|turn|1",
            "|move|p1a: Jolteon|Agility|p1a: Jolteon",
            "|-boost|p1a: Jolteon|spe|2",
            "|-start|p2a: Machamp|confusion",
            "|turn|2",
            // Faster Machamp under Quick Claw: hits itself in confusion, then Jolteon Baton Passes
            "|-activate|p2a: Machamp|confusion",
            "|-damage|p2a: Machamp|88/100|[from] confusion",
            "|move|p1a: Jolteon|Baton Pass|p1a: Jolteon",
        ] {
            agent.push_line(&dex, line);
        }
        let req = serde_json::json!({
            "forceSwitch": [true],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Jolteon", "details": "Jolteon, L50, M", "condition": "171/171", "active": true, "item": "leftovers"},
                    {"ident": "p1: Marowak", "details": "Marowak, L50, M", "condition": "166/166", "active": false, "item": "thickclub"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 2
        })
        .to_string();
        assert!(agent.on_request(&dex, &req).unwrap());
        let b = agent.battle().unwrap();
        assert!(
            !b.queue.iter().any(|a| matches!(a.choice, ActionKind::Move { .. })),
            "confused foe already spent its move action; queue must not schedule a second move"
        );

        // Verify failed Baton Pass does not pass boosts to faint replacement
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Jolteon, L50, M|item",
            "|poke|p1|Marowak, L50, M|item",
            "|poke|p2|Machamp, L50, M|item",
            "|switch|p1a: Jolteon|Jolteon, L50, M|100/100",
            "|switch|p2a: Machamp|Machamp, L50, M|100/100",
            "|turn|1",
            "|move|p1a: Jolteon|Agility|p1a: Jolteon",
            "|-boost|p1a: Jolteon|spe|2",
            "|move|p1a: Jolteon|Baton Pass|p1a: Jolteon",
            "|-fail|p1a: Jolteon",
            "|faint|p1a: Jolteon",
            "|switch|p1a: Marowak|Marowak, L50, M|100/100",
        ] {
            tracker.push_line(&dex, line);
        }
        assert_eq!(tracker.sides[0].mons[1].boosts, [0; 7]);
    }

    #[test]
    fn substitute_creator_and_damage_persist_through_baton_pass_and_roundtrip() {
        let dex = conformance::load_dex();
        let pool = MetaPool { teams: Vec::new() };
        let own = canonical_team(
            &dex,
            vec![
                test_set("Jolteon", 55, "Leftovers", &["Thunderbolt", "Substitute", "Baton Pass", "Hidden Power"]),
                test_set("Marowak", 50, "Thick Club", &["Earthquake", "Rock Slide", "Swords Dance", "Rest"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
                test_set("Exeggutor", 50, "Mint Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
                test_set("Gengar", 50, "Gold Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Cloyster", 50, "Mystery Berry", &["Surf", "Spikes", "Explosion", "Icy Wind"]),
            ],
        );
        let foe = canonical_team(
            &dex,
            vec![
                test_set("Blissey", 50, "Leftovers", &["Seismic Toss", "Soft-Boiled", "Heal Bell", "Thunder Wave"]),
                test_set("Zapdos", 55, "Mint Berry", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Snorlax", 50, "Miracle Berry", &["Body Slam", "Earthquake", "Self-Destruct", "Curse"]),
                test_set("Suicune", 50, "Gold Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Mystery Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Forretress", 50, "Pink Bow", &["Spikes", "Rapid Spin", "Explosion", "Hidden Power"]),
            ],
        );
        let mut agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 99);
        agent.set_own_team(own.clone());
        agent.pin_opponent(&dex, foe.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe))
        {
            agent.push_line(&dex, &line);
        }
        for line in [
            "|switch|p1a: Jolteon|Jolteon, L55, M|100/100",
            "|switch|p2a: Blissey|Blissey, L50, F|100/100",
            "|turn|1",
            "|move|p1a: Jolteon|Substitute|p1a: Jolteon",
            "|-start|p1a: Jolteon|Substitute",
            "|-damage|p1a: Jolteon|75/100",
            "|move|p2a: Blissey|Seismic Toss|p1a: Jolteon",
            "|-activate|p1a: Jolteon|Substitute|[damage]",
            "|turn|2",
            "|move|p1a: Jolteon|Baton Pass|p1a: Jolteon",
            "|switch|p1a: Marowak|Marowak, L50, M|100/100",
            "|move|p2a: Blissey|Soft-Boiled|p2a: Blissey",
            "|turn|3",
        ] {
            agent.push_line(&dex, line);
        }
        let req = serde_json::json!({
            "active": [{
                "moves": [
                    {"id": "earthquake", "move": "Earthquake", "pp": 16, "maxpp": 16, "disabled": false},
                    {"id": "rockslide", "move": "Rock Slide", "pp": 16, "maxpp": 16, "disabled": false},
                    {"id": "swordsdance", "move": "Swords Dance", "pp": 32, "maxpp": 32, "disabled": false},
                    {"id": "rest", "move": "Rest", "pp": 16, "maxpp": 16, "disabled": false}
                ],
                "trapped": false,
                "maybeTrapped": true
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Marowak", "details": "Marowak, L50, M", "condition": "166/166", "active": true, "item": "thickclub"},
                    {"ident": "p1: Jolteon", "details": "Jolteon, L55, M", "condition": "141/187", "active": false, "item": "leftovers"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 3
        })
        .to_string();
        assert!(agent.on_request(&dex, &req).unwrap());
        let b = agent.battle().unwrap();
        let active = b.active_id(0).unwrap();
        let jolteon_maxhp = b.sides[0].roster[0].base_maxhp;
        let expected_init = (jolteon_maxhp / 4) as i64;
        let sub_cid = dex.conds_id("substitute").unwrap();
        let sub_hp = b.poke(active).volatile(sub_cid).unwrap().get_int(DK::Hp);
        assert_eq!(sub_hp, (expected_init - 50).max(1));
        assert!(b.poke(active).maybe_trapped);

        let spec = agent.to_position_spec(&dex).unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        let parsed = crate::position::PositionSpec::parse(&json).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
        let live = agent.resynthesize(&dex, 777).unwrap();
        let rebuilt = crate::position::synthesize_spec(&dex, &parsed, &pool, Some(&foe), 777).unwrap();
        assert_eq!(live.state_key128(), rebuilt.state_key128());
    }

    #[test]
    fn transform_request_moves_do_not_corrupt_base_slots_and_roundtrip_pp() {
        let dex = conformance::load_dex();
        let pool = MetaPool { teams: Vec::new() };
        let own = canonical_team(
            &dex,
            vec![
                test_set("Ditto", 50, "Leftovers", &["Transform"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Curse", "Rest", "Sleep Talk"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
                test_set("Exeggutor", 50, "Gold Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
                test_set("Gengar", 50, "Mystery Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Cloyster", 50, "Pink Bow", &["Surf", "Spikes", "Explosion", "Icy Wind"]),
            ],
        );
        let foe = canonical_team(
            &dex,
            vec![
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Cloyster", 50, "Mint Berry", &["Surf", "Spikes", "Explosion", "Icy Wind"]),
                test_set("Gengar", 50, "Miracle Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Suicune", 50, "Gold Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Mystery Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Forretress", 50, "Pink Bow", &["Spikes", "Rapid Spin", "Explosion", "Hidden Power"]),
            ],
        );
        let mut agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 17);
        agent.set_own_team(own.clone());
        agent.pin_opponent(&dex, foe.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe))
        {
            agent.push_line(&dex, &line);
        }
        for line in [
            "|switch|p1a: Ditto|Ditto, L50|100/100",
            "|switch|p2a: Zapdos|Zapdos, L55|100/100",
            "|turn|1",
            "|move|p1a: Ditto|Transform|p2a: Zapdos",
            "|-transform|p1a: Ditto|p2a: Zapdos",
            "|move|p2a: Zapdos|Thunderbolt|p1a: Ditto",
            "|turn|2",
            "|move|p1a: Ditto|Thunderbolt|p2a: Zapdos",
            "|move|p2a: Zapdos|Drill Peck|p1a: Ditto",
            "|turn|3",
        ] {
            agent.push_line(&dex, line);
        }
        let req = serde_json::json!({
            "active": [{
                "moves": [
                    {"id": "thunderbolt", "move": "Thunderbolt", "pp": 4, "maxpp": 5, "disabled": false},
                    {"id": "drillpeck", "move": "Drill Peck", "pp": 5, "maxpp": 5, "disabled": false},
                    {"id": "rest", "move": "Rest", "pp": 5, "maxpp": 5, "disabled": false},
                    {"id": "sleeptalk", "move": "Sleep Talk", "pp": 5, "maxpp": 5, "disabled": false}
                ],
                "trapped": false
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Ditto", "details": "Ditto, L50", "condition": "85/154", "active": true, "item": "leftovers"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 3
        })
        .to_string();
        assert!(agent.on_request(&dex, &req).unwrap());
        let b = agent.battle().unwrap();
        let ditto = b.poke(b.active_id(0).unwrap());
        let tbolt = dex.moves.id("thunderbolt").unwrap();
        let transform = dex.moves.id("transform").unwrap();
        // Copied slot has 4/5 PP, while Ditto's base_move_slots stays 1-slot Transform (15/16 PP)!
        let active_tb = ditto.move_slots.iter().find(|s| s.id == tbolt).unwrap();
        assert_eq!((active_tb.pp, active_tb.maxpp, active_tb.shared), (4, 5, false));
        assert_eq!(ditto.base_move_slots.len(), 1);
        assert_eq!((ditto.base_move_slots[0].id, ditto.base_move_slots[0].pp, ditto.base_move_slots[0].maxpp), (transform, 15, 16));

        let spec = agent.to_position_spec(&dex).unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        let parsed = crate::position::PositionSpec::parse(&json).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
        let live = agent.resynthesize(&dex, 55).unwrap();
        let rebuilt = crate::position::synthesize_spec(&dex, &parsed, &pool, Some(&foe), 55).unwrap();
        assert_eq!(live.state_key128(), rebuilt.state_key128());
    }

    #[test]
    fn ppless_locked_and_struggle_requests_round_trip_without_zeroing_pp() {
        let dex = conformance::load_dex();
        let pool = MetaPool { teams: Vec::new() };

        // 1. Non-transformed locked move (Rollout continuation: PS omits pp/maxpp).
        let own = canonical_team(
            &dex,
            vec![
                test_set("Miltank", 55, "Leftovers", &["Rollout", "Milk Drink", "Body Slam", "Heal Bell"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Thunder Wave", "Recover"]),
                test_set("Exeggutor", 50, "Gold Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
                test_set("Gengar", 50, "Mystery Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Cloyster", 50, "Pink Bow", &["Surf", "Spikes", "Explosion", "Icy Wind"]),
            ],
        );
        let foe = canonical_team(
            &dex,
            vec![
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Cloyster", 50, "Focus Band", &["Surf", "Spikes", "Toxic", "Explosion"]),
                test_set("Exeggutor", 50, "Miracle Berry", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
                test_set("Suicune", 50, "Gold Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Mystery Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Forretress", 50, "Pink Bow", &["Spikes", "Rapid Spin", "Explosion", "Hidden Power"]),
            ],
        );
        let mut agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 91);
        agent.set_own_team(own.clone());
        agent.pin_opponent(&dex, foe.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe))
        {
            agent.push_line(&dex, &line);
        }
        for line in [
            "|teampreview|3",
            "|start",
            "|switch|p1a: Miltank|Miltank, L55, F|202/202",
            "|switch|p2a: Zapdos|Zapdos, L55|100/100",
            "|turn|1",
            "|move|p1a: Miltank|Rollout|p2a: Zapdos",
            "|-supereffective|p2a: Zapdos",
            "|-damage|p2a: Zapdos|82/100",
            "|move|p2a: Zapdos|Thunderbolt|p1a: Miltank",
            "|-damage|p1a: Miltank|118/202",
            "|upkeep",
            "|turn|2",
        ] {
            agent.push_line(&dex, line);
        }
        let rollout_req = serde_json::json!({
            "active": [{
                "moves": [{"move": "Rollout", "id": "rollout"}],
                "trapped": true
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Miltank", "details": "Miltank, L55, F", "condition": "130/202", "active": true, "item": "leftovers"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 2
        })
        .to_string();
        assert!(agent.on_request(&dex, &rollout_req).unwrap());
        let spec = agent.to_position_spec(&dex).unwrap();
        let miltank_idx = spec.sides[0].active.unwrap();
        assert_eq!(spec.sides[0].mons[miltank_idx].pp[0].pp, None);
        assert_eq!(spec.sides[0].mons[miltank_idx].pp[0].maxpp, None);
        let json = serde_json::to_string(&spec).unwrap();
        let parsed = crate::position::PositionSpec::parse(&json).unwrap();
        assert_eq!(serde_json::to_string(&parsed).unwrap(), json);
        let round_req = Request::parse(&dex, &parsed.request_json(&dex).unwrap()).unwrap();
        assert_eq!(round_req.legal_inputs(), vec!["move rollout".to_string()]);
        let live = agent.resynthesize(&dex, 99).unwrap();
        let rebuilt = crate::position::synthesize_spec(&dex, &parsed, &pool, Some(&foe), 99).unwrap();
        assert_eq!(live.state_key128(), rebuilt.state_key128());
        let rollout_id = dex.moves.id("rollout").unwrap();
        let rebuilt_slot = rebuilt.poke(rebuilt.active_id(0).unwrap()).get_move_slot(rollout_id).unwrap();
        assert_eq!((rebuilt_slot.pp, rebuilt_slot.maxpp), (31, 32));
        let mut solver = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 91);
        solver.pin_opponent(&dex, foe.clone()).unwrap();
        solver.set_position(&dex, &parsed).unwrap();
        solver.step(&dex, 8).unwrap();
        assert_eq!(solver.best(&dex).as_deref(), Some("move rollout"));
        assert_eq!(solver.legality_drift, 0);

        // 2. Transformed active locked move: preserves all 4 copied move slots across PositionSpec.
        let own_ditto = canonical_team(
            &dex,
            vec![
                test_set("Ditto", 50, "Leftovers", &["Transform"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Thunder Wave", "Recover"]),
                test_set("Exeggutor", 50, "Gold Berry", &["Psychic", "Giga Drain", "Sleep Powder", "Explosion"]),
                test_set("Gengar", 50, "Mystery Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Cloyster", 50, "Pink Bow", &["Surf", "Spikes", "Explosion", "Icy Wind"]),
            ],
        );
        let foe_rollout = canonical_team(
            &dex,
            vec![
                test_set("Miltank", 55, "Leftovers", &["Rollout", "Milk Drink", "Body Slam", "Heal Bell"]),
                test_set("Zapdos", 55, "Mint Berry", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Cloyster", 50, "Miracle Berry", &["Surf", "Spikes", "Toxic", "Explosion"]),
                test_set("Suicune", 50, "Gold Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Mystery Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Forretress", 50, "Pink Bow", &["Spikes", "Rapid Spin", "Explosion", "Hidden Power"]),
            ],
        );
        let mut t_agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 92);
        t_agent.set_own_team(own_ditto.clone());
        t_agent.pin_opponent(&dex, foe_rollout.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own_ditto)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe_rollout))
        {
            t_agent.push_line(&dex, &line);
        }
        for line in [
            "|teampreview|3",
            "|start",
            "|switch|p1a: Ditto|Ditto, L50|154/154",
            "|switch|p2a: Miltank|Miltank, L55, F|100/100",
            "|turn|1",
            "|move|p2a: Miltank|Body Slam|p1a: Ditto",
            "|-damage|p1a: Ditto|90/154",
            "|move|p1a: Ditto|Transform|p2a: Miltank",
            "|-transform|p1a: Ditto|p2a: Miltank",
            "|upkeep",
            "|turn|2",
            "|move|p1a: Ditto|Rollout|p2a: Miltank",
            "|-damage|p2a: Miltank|94/100",
            "|move|p2a: Miltank|Milk Drink|p2a: Miltank",
            "|-heal|p2a: Miltank|100/100",
            "|upkeep",
            "|turn|3",
        ] {
            t_agent.push_line(&dex, line);
        }
        let ditto_locked_req = serde_json::json!({
            "active": [{
                "moves": [{"move": "Rollout", "id": "rollout"}],
                "trapped": true
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Ditto", "details": "Ditto, L50", "condition": "108/154", "active": true, "item": "leftovers"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 3
        })
        .to_string();
        assert!(t_agent.on_request(&dex, &ditto_locked_req).unwrap());
        let t_spec = t_agent.to_position_spec(&dex).unwrap();
        let t_json = serde_json::to_string(&t_spec).unwrap();
        let t_parsed = crate::position::PositionSpec::parse(&t_json).unwrap();
        assert_eq!(serde_json::to_string(&t_parsed).unwrap(), t_json);
        let t_live = t_agent.resynthesize(&dex, 101).unwrap();
        let t_rebuilt = crate::position::synthesize_spec(&dex, &t_parsed, &pool, Some(&foe_rollout), 101).unwrap();
        assert_eq!(t_live.state_key128(), t_rebuilt.state_key128());
        assert_eq!(t_rebuilt.poke(t_rebuilt.active_id(0).unwrap()).move_slots.len(), 4);

        // 3. Struggle request with live bench: keeps "move struggle" legal alongside switches.
        let struggle_req = serde_json::json!({
            "active": [{
                "moves": [{"move": "Struggle", "id": "struggle"}],
                "trapped": false
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Miltank", "details": "Miltank, L55, F", "condition": "130/202", "active": true, "item": "leftovers"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 4
        })
        .to_string();
        let mut s_agent = ProtocolAgent::new(&dex, 0, pool.clone(), RmConfig::default(), 93);
        s_agent.set_own_team(own.clone());
        s_agent.pin_opponent(&dex, foe.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe))
        {
            s_agent.push_line(&dex, &line);
        }
        for line in [
            "|teampreview|3",
            "|start",
            "|switch|p1a: Miltank|Miltank, L55, F|202/202",
            "|switch|p2a: Zapdos|Zapdos, L55|100/100",
            "|turn|1",
        ] {
            s_agent.push_line(&dex, line);
        }
        assert!(s_agent.on_request(&dex, &struggle_req).unwrap());
        let s_spec = s_agent.to_position_spec(&dex).unwrap();
        let s_json = serde_json::to_string(&s_spec).unwrap();
        let s_parsed = crate::position::PositionSpec::parse(&s_json).unwrap();
        let s_round_req = Request::parse(&dex, &s_parsed.request_json(&dex).unwrap()).unwrap();
        assert_eq!(
            s_round_req.legal_inputs(),
            vec![
                "move struggle".to_string(),
                "switch 2".to_string(),
                "switch 3".to_string()
            ]
        );
        let s_live = s_agent.resynthesize(&dex, 103).unwrap();
        let s_rebuilt = crate::position::synthesize_spec(&dex, &s_parsed, &pool, Some(&foe), 103).unwrap();
        assert_eq!(s_live.state_key128(), s_rebuilt.state_key128());
    }

    /// The M18 class-A `%`-HP invariant, as a pure property: whatever the
    /// stream announced, the imputed amount re-announces to the same string.
    /// This is the check the 2026-07-21 /48-on-a-/100-stream bug fails -- it
    /// inflated every foe HP by ~2.08x, and the bot then refused kills it had.
    #[test]
    fn imputed_hp_re_announces_to_the_number_the_stream_printed() {
        let mut checked = 0u32;
        for &den in &[48, 100] {
            for maxhp in [3, 17, 63, 100, 141, 189, 214, 363, 714] {
                for cur in 1..den {
                    let hp = impute_hp(cur, den, maxhp);
                    assert!(
                        (1..=maxhp).contains(&hp),
                        "den {den} maxhp {maxhp} cur {cur}: imputed {hp} out of range"
                    );
                    // Buckets can be empty when maxhp < den (a 63-HP mon has
                    // no HP at all for most of the 100 percentages); those
                    // announce as the nearest reachable value instead.
                    if maxhp >= den {
                        assert_eq!(
                            announce_hp(hp, maxhp, den),
                            cur,
                            "den {den} maxhp {maxhp}: {cur} -> {hp} -> {}",
                            announce_hp(hp, maxhp, den)
                        );
                        checked += 1;
                    }
                }
                assert_eq!(impute_hp(0, den, maxhp), 0);
                assert_eq!(impute_hp(den, den, maxhp), maxhp);
                assert_eq!(announce_hp(maxhp, maxhp, den), den);
                assert_eq!(announce_hp(0, maxhp, den), 0);
            }
        }
        assert!(checked > 900, "only {checked} buckets exercised");
    }

    #[test]
    fn pin_opponent_after_on_request_updates_belief_and_resyncs_searcher() {
        let dex = conformance::load_dex();
        let own = canonical_team(
            &dex,
            vec![
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
                test_set("Gengar", 50, "Gold Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Raikou", 50, "Magnet", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
                test_set("Exeggutor", 50, "Pink Bow", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
            ],
        );
        let foe_a = canonical_team(
            &dex,
            vec![
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Raikou", 50, "Magnet", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Gold Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Cloyster", 50, "Focus Band", &["Surf", "Spikes", "Toxic", "Explosion"]),
                test_set("Suicune", 50, "Miracle Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Exeggutor", 50, "Mystery Berry", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
            ],
        );
        let foe_b = canonical_team(
            &dex,
            vec![
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Hidden Power Ice", "Rest", "Sleep Talk"]),
                test_set("Raikou", 50, "Magnet", &["Thunder", "Roar", "Rest", "Sleep Talk"]),
                test_set("Misdreavus", 50, "Gold Berry", &["Thunder", "Mean Look", "Perish Song", "Protect"]),
                test_set("Cloyster", 50, "Focus Band", &["Surf", "Spikes", "Toxic", "Explosion"]),
                test_set("Suicune", 50, "Miracle Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Exeggutor", 50, "Mystery Berry", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
            ],
        );
        let pool = MetaPool {
            teams: vec![
                crate::preview::MetaTeam { id: "a".into(), sets: foe_a, weight: None },
                crate::preview::MetaTeam { id: "b".into(), sets: foe_b.clone(), weight: None },
            ],
        };
        let mut agent = ProtocolAgent::new(&dex, 0, pool, RmConfig::default(), 77);
        agent.set_own_team(own.clone());
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe_b))
            .chain([
                "|teampreview|3".to_string(),
                "|start".to_string(),
                "|switch|p1a: Zapdos|Zapdos, L55|100/100".to_string(),
                "|switch|p2a: Zapdos|Zapdos, L55|100/100".to_string(),
                "|turn|1".to_string(),
            ])
        {
            agent.push_line(&dex, &line);
        }
        let req = serde_json::json!({
            "active": [{
                "moves": [
                    {"move": "Thunderbolt", "id": "thunderbolt", "pp": 24, "maxpp": 24, "disabled": false},
                    {"move": "Drill Peck", "id": "drillpeck", "pp": 32, "maxpp": 32, "disabled": false},
                    {"move": "Rest", "id": "rest", "pp": 16, "maxpp": 16, "disabled": false},
                    {"move": "Sleep Talk", "id": "sleeptalk", "pp": 16, "maxpp": 16, "disabled": false}
                ],
                "trapped": false
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Zapdos", "details": "Zapdos, L55", "condition": "205/205", "active": true, "item": "leftovers"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 1
        })
        .to_string();
        assert!(agent.on_request(&dex, &req).unwrap());
        assert_eq!(agent.belief().unwrap().candidate_count(), 2);

        agent.pin_opponent(&dex, foe_b).unwrap();
        assert_eq!(agent.belief().unwrap().candidate_count(), 1);
        assert!(agent.search().is_some());
    }

    #[test]
    fn safeguard_outrage_lock_clears_when_multi_move_request_arrives() {
        let dex = conformance::load_dex();
        let own = canonical_team(
            &dex,
            vec![
                test_set("Dragonite", 55, "Leftovers", &["Outrage", "Safeguard", "Rest", "Hyper Beam"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
                test_set("Starmie", 50, "Miracle Berry", &["Surf", "Psychic", "Recover", "Rapid Spin"]),
                test_set("Gengar", 50, "Gold Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Raikou", 50, "Magnet", &["Thunderbolt", "Roar", "Rest", "Sleep Talk"]),
                test_set("Exeggutor", 50, "Pink Bow", &["Psychic", "Giga Drain", "Stun Spore", "Explosion"]),
            ],
        );
        let foe = canonical_team(
            &dex,
            vec![
                test_set("Zapdos", 55, "Leftovers", &["Thunderbolt", "Drill Peck", "Rest", "Sleep Talk"]),
                test_set("Snorlax", 50, "Mint Berry", &["Body Slam", "Earthquake", "Curse", "Rest"]),
                test_set("Gengar", 50, "Gold Berry", &["Thunderbolt", "Ice Punch", "Destiny Bond", "Explosion"]),
                test_set("Cloyster", 50, "Focus Band", &["Surf", "Spikes", "Toxic", "Explosion"]),
                test_set("Suicune", 50, "Miracle Berry", &["Surf", "Ice Beam", "Rest", "Sleep Talk"]),
                test_set("Forretress", 50, "Mystery Berry", &["Spikes", "Rapid Spin", "Explosion", "Hidden Power"]),
            ],
        );
        let pool = MetaPool {
            teams: vec![crate::preview::MetaTeam { id: "f".into(), sets: foe.clone(), weight: None }],
        };
        let mut agent = ProtocolAgent::new(&dex, 0, pool, RmConfig::default(), 88);
        agent.set_own_team(own.clone());
        agent.pin_opponent(&dex, foe.clone()).unwrap();
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &foe))
            .chain([
                "|teampreview|3".to_string(),
                "|start".to_string(),
                "|switch|p1a: Dragonite|Dragonite, L55, M|100/100".to_string(),
                "|switch|p2a: Zapdos|Zapdos, L55|100/100".to_string(),
                "|turn|1".to_string(),
                "|-sidestart|p1: Dragonite|Safeguard".to_string(),
                "|move|p1a: Dragonite|Outrage|p2a: Zapdos".to_string(),
                "|turn|2".to_string(),
                "|move|p1a: Dragonite|Outrage|p2a: Zapdos".to_string(),
                "|turn|3".to_string(),
            ])
        {
            agent.push_line(&dex, &line);
        }
        assert!(agent.tracker().sides[0].mons[0].locked.is_some());
        let req = serde_json::json!({
            "active": [{
                "moves": [
                    {"move": "Outrage", "id": "outrage", "pp": 23, "maxpp": 24, "disabled": false},
                    {"move": "Safeguard", "id": "safeguard", "pp": 40, "maxpp": 40, "disabled": false},
                    {"move": "Rest", "id": "rest", "pp": 16, "maxpp": 16, "disabled": false},
                    {"move": "Hyper Beam", "id": "hyperbeam", "pp": 8, "maxpp": 8, "disabled": false}
                ],
                "trapped": false
            }],
            "side": {
                "id": "p1",
                "name": "p1",
                "pokemon": [
                    {"ident": "p1: Dragonite", "details": "Dragonite, L55, M", "condition": "205/205", "active": true, "item": "leftovers"},
                    {"ident": "p1: Snorlax", "details": "Snorlax, L50, M", "condition": "266/266", "active": false, "item": "mintberry"},
                    {"ident": "p1: Starmie", "details": "Starmie, L50", "condition": "166/166", "active": false, "item": "miracleberry"}
                ]
            },
            "rqid": 3
        })
        .to_string();
        assert!(agent.on_request(&dex, &req).unwrap());
        assert_eq!(agent.tracker().sides[0].mons[0].locked, None);
        assert!(!agent.tracker().sides[0].mons[0].called_locked);
        assert!(agent.battle().unwrap().clone().legal_choices(&dex, 0).len() > 1);
        agent.step(&dex, 4).unwrap();
        let _ = agent.best(&dex);
        assert_eq!(agent.legality_drift, 0);
    }

    #[test]
    fn called_locking_move_sets_called_locked_without_charging_base_pp() {
        let dex = conformance::load_dex();
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Clefable, L50, F|",
            "|poke|p2|Snorlax, L50, M|",
            "|switch|p1a: Clefable|Clefable, L50, F|100/100",
            "|switch|p2a: Snorlax|Snorlax, L50, M|100/100",
            "|turn|1",
            "|move|p1a: Clefable|Metronome|p1a: Clefable",
            "|move|p1a: Clefable|Outrage|p2a: Snorlax|[from] move: Metronome",
        ] {
            tracker.push_line(&dex, line);
        }
        let metro = dex.moves.id("metronome").unwrap();
        let outrage = dex.moves.id("outrage").unwrap();
        let mon = &tracker.sides[0].mons[0];
        assert!(mon.called_locked);
        assert_eq!(mon.locked, Some((outrage, 1)));
        assert_eq!(mon.uses, vec![(metro, 1)]);

        tracker.push_line(&dex, "|turn|2");
        tracker.push_line(&dex, "|move|p1a: Clefable|Outrage|p2a: Snorlax");
        let mon = &tracker.sides[0].mons[0];
        assert_eq!(mon.locked, Some((outrage, 2)));
        assert_eq!(mon.uses, vec![(metro, 1)]);

        tracker.push_line(&dex, "|cant|p1a: Clefable|par");
        let mon = &tracker.sides[0].mons[0];
        assert_eq!(mon.locked, None);
        assert!(!mon.called_locked);
    }

    #[test]
    fn estimate_sub_damage_handles_return_hidden_power_type_and_super_fang() {
        let dex = conformance::load_dex();
        let mut tracker = ProtocolTracker::new(0);
        for line in [
            "|poke|p1|Snorlax, L55, M|item",
            "|poke|p2|Rhydon, L50, M|item",
            "|switch|p1a: Snorlax|Snorlax, L55, M|100/100",
            "|switch|p2a: Rhydon|Rhydon, L50, M|100/100",
            "|turn|1",
        ] {
            tracker.push_line(&dex, line);
        }
        let return_id = dex.moves.id("return").unwrap();
        let superf_id = dex.moves.id("superfang").unwrap();
        let hp_id = dex.moves.id("hiddenpower").unwrap();
        let return_dmg = estimate_sub_damage(&dex, &tracker.sides, &tracker.weather, (0, 0), (1, 0), return_id, false);
        let superf_dmg = estimate_sub_damage(&dex, &tracker.sides, &tracker.weather, (0, 0), (1, 0), superf_id, false);
        let hp_dmg = estimate_sub_damage(&dex, &tracker.sides, &tracker.weather, (0, 0), (1, 0), hp_id, false);
        let rhydon_hp_base = dex.species.get(tracker.sides[1].mons[0].species).base_stats.hp as i32;
        let rhydon_maxhp = (((2 * rhydon_hp_base * 50) / 100 + 50 + 10).max(1)) as i64;
        assert!(return_dmg > 15, "Return should use 102 BP, got {return_dmg}");
        assert_eq!(superf_dmg, (rhydon_maxhp / 2).max(1));
        assert!(hp_dmg >= 1, "plain hiddenpower should stay conservative >= 1, got {hp_dmg}");
    }

    #[test]
    fn hidden_timer_specs_exclude_rest_and_track_disable_acted_and_upkeeps() {
        use crate::player_state::HiddenTimerKind;
        let dex = conformance::load_dex();
        let own = vec![test_set("Snorlax", 55, "leftovers", &["Rest", "Return", "Curse", "Earthquake"])];
        let refs = vec![test_set("Zapdos", 50, "mintberry", &["Thunderbolt", "Rest", "Sleep Talk", "Hidden Power Ice"])];
        let mut tracker = ProtocolTracker::new(0);
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &refs))
            .chain([
                "|switch|p1a: Snorlax|Snorlax, L55, M|100/100".to_string(),
                "|switch|p2a: Zapdos|Zapdos, L50|100/100".to_string(),
                "|turn|1".to_string(),
                "|move|p2a: Zapdos|Rest|p2a: Zapdos".to_string(),
                "|-status|p2a: Zapdos|slp|[from] move: Rest".to_string(),
                "|move|p1a: Snorlax|Disable|p2a: Zapdos".to_string(),
                "|-start|p2a: Zapdos|Disable|Rest".to_string(),
                "|upkeep".to_string(),
                "|turn|2".to_string(),
            ])
        {
            tracker.push_line(&dex, &line);
        }
        let req_json = serde_json::json!({
            "active": [{"moves": [
                {"id": "rest", "pp": 16, "maxpp": 16, "disabled": false},
                {"id": "return", "pp": 32, "maxpp": 32, "disabled": false},
                {"id": "curse", "pp": 16, "maxpp": 16, "disabled": false},
                {"id": "earthquake", "pp": 16, "maxpp": 16, "disabled": false}
            ]}],
            "side": {"pokemon": [{
                "ident": "p1: Snorlax",
                "details": "Snorlax, L55, M",
                "active": true,
                "condition": "240/240",
                "moves": ["rest", "return", "curse", "earthquake"],
                "item": "leftovers"
            }]}
        })
        .to_string();
        let req = Request::parse(&dex, &req_json).unwrap();
        let mut rng = SplitMix64::new(42);
        let obs = Observer::from_mons(0, tracker.observer_mons());
        let ref_mons = Battle::from_fixture(&dex, "1,2,3,4", &own, &refs)
            .unwrap()
            .sides[1]
            .roster
            .clone();
        let battle = tracker.synthesize(&dex, &own, &ref_mons, &obs, &req, &mut rng).unwrap();
        let specs = tracker.hidden_timer_specs(&dex, &battle);
        // Rest sleep must NOT produce a Sleep timer, whereas Disable after Zapdos acted
        // must carry acted_on_start = true and upkeeps = 1.
        assert!(!specs.iter().any(|s| matches!(s.kind, HiddenTimerKind::Sleep { .. })));
        assert!(specs.contains(&crate::player_state::HiddenTimerSpec {
            side: 1,
            slot: 0,
            kind: HiddenTimerKind::Disable {
                upkeeps: 1,
                acted_on_start: true,
            },
        }));

        // Non-Rest sleep (e.g. Lovely Kiss) DOES emit Sleep { slept: 1 }.
        let mut t2 = ProtocolTracker::new(0);
        for line in poke_lines(&dex, "p1", &own)
            .into_iter()
            .chain(poke_lines(&dex, "p2", &refs))
            .chain([
                "|switch|p1a: Snorlax|Snorlax, L55, M|100/100".to_string(),
                "|switch|p2a: Zapdos|Zapdos, L50|100/100".to_string(),
                "|turn|1".to_string(),
                "|move|p1a: Snorlax|Lovely Kiss|p2a: Zapdos".to_string(),
                "|-status|p2a: Zapdos|slp".to_string(),
                "|cant|p2a: Zapdos|slp".to_string(),
                "|upkeep".to_string(),
                "|turn|2".to_string(),
            ])
        {
            t2.push_line(&dex, &line);
        }
        let b2 = t2.synthesize(&dex, &own, &ref_mons, &obs, &req, &mut rng).unwrap();
        let s2 = t2.hidden_timer_specs(&dex, &b2);
        assert!(s2.contains(&crate::player_state::HiddenTimerSpec {
            side: 1,
            slot: 0,
            kind: HiddenTimerKind::Sleep { slept: 1 },
        }));
    }
}

// ================================================== hand-entered positions

use crate::position::{
    CondSpec, ItemKnowledge, MonSpec, PositionSpec, PpSpec, SideSpec, UseSpec, VolSpec,
    WeatherSpec, SCHEMA,
};

fn gender_of(s: &str) -> Gender {
    match s {
        "M" => Gender::M,
        "F" => Gender::F,
        _ => Gender::N,
    }
}

fn move_key(dex: &Dex, id: MoveId) -> String {
    dex.moves.key(id).to_string()
}

fn move_id_of(dex: &Dex, key: &str) -> Result<MoveId, String> {
    dex.moves.id(&toid(key)).ok_or_else(|| format!("unknown move `{key}`"))
}

impl TrackMon {
    fn to_spec(&self, dex: &Dex) -> MonSpec {
        MonSpec {
            species: dex.species.key(self.species).to_string(),
            level: self.level,
            gender: self.gender.as_str().to_string(),
            name: self.name.clone(),
            item_flag: self.preview_item,
            appeared: self.appeared,
            appear_count: self.appear_count,
            switch_in_turn: self.switch_in_turn,
            active: self.active,
            fainted: self.fainted,
            hp_num: self.pixels,
            hp_den: self.hp_den.max(1),
            hp_exact: None,
            status: self.status.as_str().to_string(),
            rest: self.rest,
            slept: self.slept,
            tox_counter: self.comp_res,
            comp_brn: self.comp_brn,
            comp_par: self.comp_par,
            boosts: self.boosts,
            volatiles: self
                .vols
                .iter()
                .map(|v| VolSpec {
                    key: v.key.clone(),
                    start_turn: v.start_turn,
                    move_key: v.move_id.map(|m| move_key(dex, m)),
                    source: v.source.map(|(s, slot)| [s, slot]),
                    counter: v.counter,
                })
                .collect(),
            uses: self
                .uses
                .iter()
                .map(|&(m, n)| UseSpec { move_key: move_key(dex, m), n })
                .collect(),
            locked: self.locked.map(|(m, n)| UseSpec { move_key: move_key(dex, m), n }),
            charging: self.charging.map(|m| move_key(dex, m)),
            must_recharge: self.must_recharge,
            transformed_into: self.transformed_into.map(|(s, slot)| [s, slot]),
            transformed_uses: self
                .transformed_uses
                .iter()
                .map(|&(m, n)| UseSpec { move_key: move_key(dex, m), n })
                .collect(),
            mimic_overlay: self.mimic_overlay.map(|m| move_key(dex, m)),
            last_move: self.last_move.map(|m| move_key(dex, m)),
            stall_streak: self.stall_streak,
            last_protect_turn: self.last_protect_turn,
            protected_this_turn: self.protected_this_turn,
            ..MonSpec::default()
        }
    }

    fn from_spec(dex: &Dex, m: &MonSpec) -> Result<TrackMon, String> {
        let species = dex
            .species
            .id(&toid(&m.species))
            .ok_or_else(|| format!("unknown species `{}`", m.species))?;
        let mut tm = TrackMon::new(species, m.level, gender_of(&m.gender), m.item_flag);
        tm.name = m.name.clone();
        tm.appeared = m.appeared;
        // "It has appeared" and "it has switched in at least once" are the
        // same statement, and the tracker sets them together on every switch
        // line. A hand-written position states only the first — there is no
        // control for a switch-in COUNT and there should not be — so the
        // second is restored here.
        //
        // Not cosmetic: `Belief::determinize` decides which party slots hold
        // unseen picks from `previously_switched_in`, which this feeds. Left
        // at zero, an opponent the user has seen (and possibly watched
        // faint) reads as a slot never revealed, and the determinizer
        // reshuffles a live bench mon into it — the search then answers a
        // question about a team the opponent does not have.
        tm.appear_count = m.appear_count.max(i32::from(m.appeared));
        tm.switch_in_turn = m.switch_in_turn;
        tm.active = m.active;
        tm.fainted = m.fainted;
        tm.pixels = m.hp_num;
        tm.hp_den = m.hp_den.max(1);
        tm.status = m.status_enum();
        tm.rest = m.rest;
        tm.slept = m.slept;
        tm.comp_res = m.tox_counter;
        tm.comp_brn = m.comp_brn;
        tm.comp_par = m.comp_par;
        tm.boosts = m.boosts;
        for v in &m.volatiles {
            tm.vols.push(TVol {
                key: v.key.clone(),
                start_turn: v.start_turn,
                upkeeps: 0,
                acted_on_start: false,
                move_id: v.move_key.as_deref().map(|k| move_id_of(dex, k)).transpose()?,
                source: v.source.map(|[s, slot]| (s, slot)),
                counter: v.counter,
            });
        }
        for u in &m.uses {
            tm.uses.push((move_id_of(dex, &u.move_key)?, u.n));
        }
        tm.locked = match &m.locked {
            Some(u) => Some((move_id_of(dex, &u.move_key)?, u.n)),
            None => None,
        };
        tm.locked_upkeeps = tm.locked.map_or(0, |(_, u)| (u - 1).clamp(0, 2) as u8);
        tm.charging = m.charging.as_deref().map(|k| move_id_of(dex, k)).transpose()?;
        tm.must_recharge = m.must_recharge;
        tm.transformed_into = m.transformed_into.map(|[s, slot]| (s, slot));
        for u in &m.transformed_uses {
            tm.transformed_uses.push((move_id_of(dex, &u.move_key)?, u.n));
        }
        tm.mimic_overlay = m.mimic_overlay.as_deref().map(|k| move_id_of(dex, k)).transpose()?;
        tm.last_move = m.last_move.as_deref().map(|k| move_id_of(dex, k)).transpose()?;
        tm.stall_streak = m.stall_streak;
        tm.last_protect_turn = m.last_protect_turn;
        tm.protected_this_turn = m.protected_this_turn;
        Ok(tm)
    }
}

impl ProtocolTracker {
    /// Serialize the whole tracked information set as a hand-editable
    /// position. `obs` supplies the opponent reveal channel, which lives in
    /// the `Observer` rather than here (the tracker records *what happened*;
    /// the observer records *what that proves about the set*).
    ///
    /// `own_sets` are copied in verbatim — they are the analyzing side's
    /// private truth and have no protocol representation at all.
    pub fn to_spec(
        &self,
        dex: &Dex,
        own_sets: &[PokemonSet],
        obs: &Observer,
        req: &Request,
    ) -> PositionSpec {
        let me = self.side;
        let opp = 1 - me;
        let side_spec = |s: usize| -> SideSpec {
            let ts = &self.sides[s];
            SideSpec {
                mons: ts.mons.iter().map(|m| m.to_spec(dex)).collect(),
                active: ts.active,
                conditions: ts
                    .conds
                    .iter()
                    .map(|(key, start)| CondSpec { key: key.clone(), start_turn: *start })
                    .collect(),
                pending_bp: ts.pending_bp,
                acted_this_turn: ts.acted_this_turn,
                fainted_this_turn: ts.fainted_this_turn,
                fainted_last_turn: ts.fainted_last_turn,
                last_move: ts.side_last_move.map(|m| move_key(dex, m)),
                party: Vec::new(),
            }
        };
        let mut sides = [side_spec(0), side_spec(1)];
        // Own side: the REQUEST is the authority, not the protocol echo —
        // display order, exact HP, the item actually in hand, and the active's
        // per-move PP all come from it, and all four can differ from what the
        // public channel implies (a consumed berry, a Mystery Berry's PP, a
        // party the player reordered by switching).
        for (pos, rm) in req.pokemon.iter().enumerate() {
            let Some(slot) = self.sides[me]
                .mons
                .iter()
                .enumerate()
                .find(|(slot, m)| {
                    !sides[me].party.contains(slot)
                        && m.species == rm.species
                        && m.level == rm.level
                })
                .map(|(slot, _)| slot)
            else {
                continue;
            };
            sides[me].party.push(slot);
            let m = &mut sides[me].mons[slot];
            m.hp_exact = Some(rm.hp);
            m.item_now = Some(rm.item.clone());
            if rm.active && !req.active_moves.is_empty() {
                m.pp = req
                    .active_moves
                    .iter()
                    .map(|am| PpSpec {
                        move_key: am.id.clone(),
                        pp: am.pp,
                        maxpp: am.maxpp,
                        disabled: am.disabled,
                    })
                    .collect();
            }
            let _ = pos;
        }
        // Opponent: fold in the reveal channel, slot-aligned with the
        // observer's roster (both follow `|poke|` order).
        for (slot, mo) in obs.mons().iter().enumerate() {
            let Some(m) = sides[opp].mons.get_mut(slot) else { break };
            m.hp_type_mask = mo.hp_type_mask;
            m.revealed_moves = mo.revealed_moves.iter().map(|&x| move_key(dex, x)).collect();
            m.item_original =
                ItemKnowledge::from_obs(mo.item.original.map(|x| x.map(|i| dex.items.key(i))));
            m.item_current =
                ItemKnowledge::from_obs(mo.item.current.map(|x| x.map(|i| dex.items.key(i))));
            m.item_gained = mo.item.gained;
        }
        PositionSpec {
            schema: SCHEMA.to_string(),
            side: me,
            turn: self.turn,
            upkeep_this_turn: self.upkeep_this_turn,
            own_sets: own_sets.to_vec(),
            sides,
            weather: self
                .weather
                .as_ref()
                .map(|(key, upkeeps)| WeatherSpec { key: key.clone(), upkeeps: *upkeeps }),
            team_preview: req.team_preview,
            force_switch: req.force_switch,
            trapped: req.trapped,
            maybe_trapped: req.maybe_trapped,
            preview_level_caps: req.preview_level_caps,
        }
    }

    /// Rebuild a tracker from a position — the exact inverse of [`to_spec`]
    /// over every tracked field. `crates/bot/tests/import.rs` proves the
    /// round trip at every decision point of the conformance corpus.
    pub fn from_spec(dex: &Dex, spec: &PositionSpec) -> Result<ProtocolTracker, String> {
        let mut tr = ProtocolTracker::new(spec.side);
        tr.turn = spec.turn;
        tr.upkeep_this_turn = spec.upkeep_this_turn;
        tr.weather = spec.weather.as_ref().map(|w| (w.key.clone(), w.upkeeps));
        for s in 0..2 {
            let src = &spec.sides[s];
            let dst = &mut tr.sides[s];
            for m in &src.mons {
                dst.mons.push(TrackMon::from_spec(dex, m)?);
            }
            dst.active = src.active;
            dst.conds = src.conditions.iter().map(|c| (c.key.clone(), c.start_turn)).collect();
            dst.pending_bp = src.pending_bp;
            dst.acted_this_turn = src.acted_this_turn;
            dst.fainted_this_turn = src.fainted_this_turn;
            dst.fainted_last_turn = src.fainted_last_turn;
            dst.side_last_move = src.last_move.as_deref().map(|k| move_id_of(dex, k)).transpose()?;
        }
        Ok(tr)
    }
}

impl ProtocolAgent {
    /// The accumulated public information set (solver export, harness
    /// inspection). Read-only: the tracker only ever advances through
    /// `push_line` or a whole-position install.
    pub fn tracker(&self) -> &ProtocolTracker {
        &self.tracker
    }

    /// Export the current decision point as a hand-editable position.
    /// `None` before the first request — there is no decision point yet, and
    /// a spec without one would not describe anything.
    pub fn to_position_spec(&self, dex: &Dex) -> Option<PositionSpec> {
        let obs = self.observer.as_ref()?;
        let req = self.request.as_ref()?;
        Some(self.tracker.to_spec(dex, &self.own_sets, obs, req))
    }

    /// Re-run this decision point's synthesis with an explicit seed. The
    /// battle `on_request` already built is the one being searched; this is
    /// the same construction made reproducible, so two information sets can
    /// be compared field-for-field without the rolled hidden durations
    /// (sleep counters, bind turns, substitute HP) drifting between them.
    pub fn resynthesize(&self, dex: &Dex, seed: u64) -> Result<Battle, String> {
        let obs = self.observer.as_ref().ok_or("resynthesize before on_request")?;
        let belief = self.belief.as_ref().ok_or("resynthesize before on_request")?;
        let req = self.request.as_ref().ok_or("resynthesize before on_request")?;
        let pick = belief.alive().first().copied();
        let mut rng = SplitMix64::new(seed);
        self.tracker.synthesize(dex, &self.own_sets, belief.refs(pick), obs, req, &mut rng)
    }

    /// Solver mode: analyze a hand-entered position. Replaces the tracked
    /// battle wholesale — own team, public facts, reveal channel and belief
    /// — and lands on the same searcher the live path builds.
    ///
    /// Deliberately NOT a variant of `push_line`: a position is a *state*,
    /// while the protocol is a *history*, and fabricating a history that
    /// happens to produce the desired state is both harder and weaker than
    /// stating the state (see `position.rs`).
    pub fn set_position(&mut self, dex: &Dex, spec: &PositionSpec) -> Result<(), String> {
        spec.check()?;
        if spec.side != self.side {
            return Err(format!(
                "position is for side {} but this agent plays side {}",
                spec.side, self.side
            ));
        }
        let request_json = spec.request_json(dex)?;
        let req = Request::parse(dex, &request_json)?;
        self.own_sets = spec.own_sets.clone();
        self.tracker = ProtocolTracker::from_spec(dex, spec)?;
        self.history.clear();
        let obs = Observer::from_position(dex, spec)?;
        let mut belief = match &self.pinned_sets {
            Some(sets) => Belief::pinned_checked(dex, "opponent", sets, &obs)?,
            None => Belief::with_fallback_policy(dex, &self.pool, &obs, self.fallback_policy),
        };
        if let (Some(prior), None) = (self.prior.clone(), &self.pinned_sets) {
            belief.set_prior(prior);
        }
        self.observer = Some(obs);
        self.belief = Some(belief);
        self.install(dex, req).map(|_| ())
    }
}
