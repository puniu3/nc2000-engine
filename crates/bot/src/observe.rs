//! M10a observation tracker: what one player legitimately knows about the
//! opponent's team, accumulated over a battle.
//!
//! # Tap point
//!
//! Two channels, both driven by `Observer::observe(&Battle)` called at the
//! bot's real decision points (never inside search iterations):
//!
//! 1. **State diff** — reads opponent fields whose *every mutation path in
//!    this engine is publicly announced*, so the diff carries exactly the
//!    information a competent human extracts from the protocol:
//!    - `base_move_slots`: `pp < maxpp || used` ⇔ the move was executed at
//!      least once with a public `|move|` line. Audited mutation paths:
//!      `deduct_pp` is called only from `run_move` (public `|move|`) and
//!      Spite (public `-activate`, targets the already-revealed last move);
//!      Mystery Berry's PP restore is public and only reaches a slot that
//!      hit 0 PP (already revealed). Transform/Mimic overlay slots are
//!      `shared = false`, so foreign moves never touch the base slots.
//!    - `item`: transitions only. `Some(x) → None` (use_item / eat_item /
//!      take_item — all emit `-enditem` or the Thief `-item ... [of]` line),
//!      `None → Some(y)` (Thief steal, public `-item`). The standing value
//!      is stored privately for diffing but never exposed; only transitions
//!      (all public) produce knowledge.
//!    - preview-public facts read once at construction: species, level,
//!      gender (the `|poke|` details) and *item presence* (the `|poke|`
//!      line's item flag).
//!    This channel works with the protocol log disabled.
//!
//! 2. **Log scan** — incremental parse of `battle.log` when the observed
//!    battle happens to run log-on. Adds the reveals that leave no state
//!    trace: Leftovers residual heals (`-heal ... [from] item: Leftovers`),
//!    Focus Band procs (`-activate ... item: Focus Band`), and moves called
//!    by Sleep Talk (`|move| ... [from] Sleep Talk` — in gen 2 the called
//!    move is from the sleeper's own set). Degrades silently to channel 1
//!    when the log is off.
//!
//! Rejected taps: *pure log parsing* (dies in log-off arenas), *instrumenting
//! apply_choices* (would hand the tracker the opponent's submitted choice,
//! which is NOT public when the move is cancelled invisibly — sleep/para
//! `|cant|` hides the selection — i.e. a psychic tap by construction; it
//! would also touch engine code the bit-exactness constraint protects), and
//! a *tracker-owned log-on mirror battle* (needs the applied choices plumbed
//! in — the same psychic tap — and adds 2x real-game stepping for no
//! information beyond the two channels above).
//!
//! M10b should run the *outer* (real) battle log-on where possible — the
//! cost is one game's protocol log; search clones inside `SkuctSearch`
//! always disable the log themselves.
//!
//! # What is deliberately NOT tracked (conservative by design)
//!
//! Quick Claw (silent in this engine's log; humans infer it from impossible
//! ordering), Bright Powder (unattributed misses), King's Rock (unattributed
//! flinches), stat-boost items (silent), Hidden Power type inference from
//! effectiveness lines. Missing a reveal only leaves the belief a superset —
//! it can never false-filter the true team.

use nc2000_engine::dex::{toid, Dex, ItemId, MoveId, SpeciesId};
use nc2000_engine::state::{Battle, Gender};

/// Public knowledge about one opponent roster mon's held item.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ItemObs {
    /// The item the mon *brought* (what set filtering matches against).
    /// `None` = identity unknown; `Some(None)` = known to have started
    /// empty-handed (its Thief succeeded).
    pub original: Option<Option<ItemId>>,
    /// The *currently held* item when publicly known (the determinizer keeps
    /// the true item field exactly when this is `Some`). `Some(None)` =
    /// known consumed/stolen away.
    pub current: Option<Option<ItemId>>,
    /// A gain event (Thief steal) was observed — later reveals no longer
    /// identify the original item.
    gained: bool,
}

impl ItemObs {
    /// `Some(x) → None` transition or `-enditem` line: x left the mon.
    fn lost(&mut self, x: ItemId) -> bool {
        let mut dirty = self.current != Some(None);
        self.current = Some(None);
        if !self.gained && self.original.is_none() {
            self.original = Some(Some(x));
            dirty = true;
        }
        dirty
    }

    /// `None → Some(y)` transition or `-item` line: y arrived (Thief).
    fn gained(&mut self, y: ItemId) -> bool {
        let dirty = self.current != Some(Some(y)) || !self.gained;
        self.current = Some(Some(y));
        if self.original.is_none() {
            // Thief only steals into an empty hand.
            self.original = Some(None);
        }
        self.gained = true;
        dirty
    }

    /// Passive reveal of a held item (Leftovers heal, Focus Band proc).
    /// Only upgrades unknown state — a stale `-activate` right after an
    /// `-enditem` (Mystery Berry) must not resurrect the item.
    fn revealed_held(&mut self, x: ItemId) -> bool {
        let mut dirty = false;
        if self.current.is_none() {
            self.current = Some(Some(x));
            dirty = true;
        }
        if !self.gained && self.original.is_none() {
            self.original = Some(Some(x));
            dirty = true;
        }
        dirty
    }
}

/// Everything legitimately known about one opponent roster mon.
#[derive(Clone, Debug)]
pub struct MonObs {
    // ---- public at team preview
    pub species: SpeciesId,
    pub level: u8,
    pub gender: Gender,
    /// Nickname (public whenever it appears in a log line; used only to
    /// resolve log subjects back to roster slots).
    pub name: String,
    /// The `|poke|` preview line's item flag: the mon brought *an* item.
    pub preview_has_item: bool,
    // ---- accumulated in battle
    /// Moves known to be in the mon's set (each publicly executed at least
    /// once, or named by a Sleep Talk call). Grows monotonically.
    pub revealed_moves: Vec<MoveId>,
    pub item: ItemObs,
    /// The mon has switched in at least once (its pick is public).
    pub appeared: bool,
}

/// Observation tracker for one side of one battle. Construct at battle
/// start (team preview — item *presence* is read then), call `observe`
/// at every real decision point.
pub struct Observer {
    side: usize,
    mons: Vec<MonObs>,
    /// Private diff snapshot of the opponent's item fields — never exposed;
    /// only its (publicly-announced) transitions produce knowledge.
    prev_item: Vec<Option<ItemId>>,
    log_pos: usize,
    /// Per-mon log-channel suppression: while a Transform copy or Mimic
    /// overlay is active (tracked from the log lines themselves, in order),
    /// plain `|move|` lines may name moves that are not the mon's own.
    log_suppress: Vec<bool>,
    revision: u64,
}

impl Observer {
    pub fn new(battle: &Battle, side: usize) -> Observer {
        let opp = 1 - side;
        let mons = battle.sides[opp]
            .roster
            .iter()
            .map(|p| MonObs {
                species: p.species,
                level: p.level,
                gender: p.gender,
                name: p.name.as_str().to_string(),
                preview_has_item: p.item.is_some(),
                revealed_moves: Vec::new(),
                item: ItemObs::default(),
                appeared: false,
            })
            .collect::<Vec<_>>();
        let prev_item = battle.sides[opp].roster.iter().map(|p| p.item).collect();
        let n = mons.len();
        Observer { side, mons, prev_item, log_pos: 0, log_suppress: vec![false; n], revision: 0 }
    }

    pub fn side(&self) -> usize {
        self.side
    }

    pub fn opp(&self) -> usize {
        1 - self.side
    }

    /// Per opponent roster slot, aligned with `battle.sides[opp].roster`.
    pub fn mons(&self) -> &[MonObs] {
        &self.mons
    }

    /// Bumped whenever new information arrives (belief sync trigger).
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Ingest everything that became visible since the last call. Call at
    /// every decision point of the *real* battle (never on search clones).
    pub fn observe(&mut self, battle: &Battle, dex: &Dex) {
        let mut dirty = false;
        let opp = self.opp();

        // ---- channel 1: state diff over public-equivalent fields
        for (slot, p) in battle.sides[opp].roster.iter().enumerate() {
            let mo = &mut self.mons[slot];
            if !mo.appeared && (p.previously_switched_in > 0 || p.is_active) {
                mo.appeared = true;
            }
            // moves: base-slot usage marks (every path is a public |move|)
            for s in p.base_move_slots.iter() {
                if (s.pp < s.maxpp || s.used) && !mo.revealed_moves.contains(&s.id) {
                    mo.revealed_moves.push(s.id);
                    dirty = true;
                }
            }
            // item: transitions only (every path is publicly announced)
            let cur = p.item;
            let prev = self.prev_item[slot];
            if cur != prev {
                match (prev, cur) {
                    (Some(x), None) => dirty |= mo.item.lost(x),
                    (None, Some(y)) => dirty |= mo.item.gained(y),
                    (Some(x), Some(y)) => {
                        // unreachable in gen 2 (no in-place swap); defensive
                        if !mo.item.gained && mo.item.original.is_none() {
                            mo.item.original = Some(Some(x));
                        }
                        dirty |= mo.item.gained(y);
                    }
                    (None, None) => unreachable!(),
                }
                self.prev_item[slot] = cur;
            }
        }

        // ---- channel 2: incremental protocol-log scan (log-on battles)
        if battle.log.len() < self.log_pos {
            // log was cleared (set_log_enabled(false)); lose the tail
            self.log_pos = battle.log.len();
        }
        for i in self.log_pos..battle.log.len() {
            let line = battle.log[i].clone();
            dirty |= self.scan_line(&line, dex);
        }
        self.log_pos = battle.log.len();

        if dirty {
            self.revision += 1;
        }
    }

    // ------------------------------------------------------------ log scan

    fn scan_line(&mut self, line: &str, dex: &Dex) -> bool {
        let Some(body) = line.strip_prefix('|') else { return false };
        let parts: Vec<&str> = body.split('|').collect();
        let cmd = parts.first().copied().unwrap_or("");
        match cmd {
            "move" => self.scan_move(&parts, dex),
            "switch" | "drag" | "faint" => {
                // clear_volatile drops Transform/Mimic overlays on exit; a
                // switch line's subject is the incoming mon, so clear all
                // suppression for the side (only actives carry overlays).
                if self.subject_is_opp_side(parts.get(1).copied().unwrap_or("")) {
                    self.log_suppress.iter_mut().for_each(|s| *s = false);
                }
                false
            }
            "-transform" => {
                if let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) {
                    self.log_suppress[m] = true;
                }
                false
            }
            "-activate" => {
                let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) else {
                    return false;
                };
                let what = parts.get(2).copied().unwrap_or("");
                if what == "move: Mimic" {
                    self.log_suppress[m] = true;
                    return false;
                }
                if let Some(item_name) = what.strip_prefix("item: ") {
                    if let Some(id) = dex.items.id(&toid(item_name)) {
                        return self.mons[m].item.revealed_held(id);
                    }
                }
                false
            }
            "-enditem" => {
                let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) else {
                    return false;
                };
                match dex.items.id(&toid(parts.get(2).copied().unwrap_or(""))) {
                    Some(id) => self.mons[m].item.lost(id),
                    None => false,
                }
            }
            "-item" => {
                let subject = parts.get(1).copied().unwrap_or("");
                let Some(id) = dex.items.id(&toid(parts.get(2).copied().unwrap_or(""))) else {
                    return false;
                };
                if let Some(m) = self.opp_subject(subject) {
                    // opponent gained an item (their Thief stole ours)
                    return self.mons[m].item.gained(id);
                }
                // our Thief stole theirs: `-item <us> <item> [from] move: Thief [of] <them>`
                if parts.iter().any(|p| *p == "[from] move: Thief") {
                    if let Some(of) = parts.iter().find_map(|p| p.strip_prefix("[of] ")) {
                        if let Some(v) = self.opp_subject(of) {
                            return self.mons[v].item.lost(id);
                        }
                    }
                }
                false
            }
            "-heal" => {
                let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) else {
                    return false;
                };
                if let Some(item_name) =
                    parts.iter().find_map(|p| p.strip_prefix("[from] item: "))
                {
                    if let Some(id) = dex.items.id(&toid(item_name)) {
                        return self.mons[m].item.revealed_held(id);
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn scan_move(&mut self, parts: &[&str], dex: &Dex) -> bool {
        let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) else {
            return false;
        };
        if self.log_suppress[m] {
            // Transform copy / Mimic overlay active: plain |move| lines (and
            // even Sleep Talk calls) may name moves outside the mon's set.
            return false;
        }
        let from = parts.iter().find_map(|p| p.strip_prefix("[from] "));
        let named = match from {
            // called moves: only Sleep Talk selects from the user's own set
            // (Metronome / Mirror Move call foreign moves)
            Some("Sleep Talk") => parts.get(2).copied().unwrap_or(""),
            Some(_) => return false,
            None => parts.get(2).copied().unwrap_or(""),
        };
        let Some(id) = dex.moves.id(&toid(named)) else { return false };
        // synthetic non-set moves
        if matches!(dex.moves.key(id), "struggle" | "recharge") {
            return false;
        }
        if !self.mons[m].revealed_moves.contains(&id) {
            self.mons[m].revealed_moves.push(id);
            return true;
        }
        false
    }

    // -------------------------------------------------- subject resolution

    /// `"p2a: Nick"` / `"p2: Nick"` → opponent roster slot (unique-nickname
    /// match; ambiguity or foreign side → `None`, i.e. skip = conservative).
    fn opp_subject(&self, s: &str) -> Option<usize> {
        if !self.subject_is_opp_side(s) {
            return None;
        }
        let rest = &s[2..];
        let rest = rest.strip_prefix(|c: char| c.is_ascii_lowercase()).unwrap_or(rest);
        let name = rest.strip_prefix(": ")?;
        let mut found = None;
        for (i, m) in self.mons.iter().enumerate() {
            if m.name == name {
                if found.is_some() {
                    return None;
                }
                found = Some(i);
            }
        }
        found
    }

    fn subject_is_opp_side(&self, s: &str) -> bool {
        let b = s.as_bytes();
        b.len() >= 4 && b[0] == b'p' && b[1] == b'1' + self.opp() as u8
    }
}
