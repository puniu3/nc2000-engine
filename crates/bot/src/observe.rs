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
//! flinches), and stat-boost items (silent). Missing a reveal only leaves the belief a superset —
//! it can never false-filter the true team.

use nc2000_engine::dex::{toid, Dex, ItemId, MoveId, SpeciesId};
use nc2000_engine::state::{Battle, Gender, Status};

pub const ITEM_LEFTOVERS: u16 = 1 << 0;
pub const ITEM_MINT_BERRY: u16 = 1 << 1;
pub const ITEM_MIRACLE_BERRY: u16 = 1 << 2;
pub const ITEM_PRZCUREBERRY: u16 = 1 << 3;
pub const ITEM_PSNCUREBERRY: u16 = 1 << 4;
pub const ITEM_ICE_BERRY: u16 = 1 << 5;
pub const ITEM_BURNT_BERRY: u16 = 1 << 6;
pub const ITEM_BITTER_BERRY: u16 = 1 << 7;
pub const ITEM_BERRY: u16 = 1 << 8;
pub const ITEM_GOLD_BERRY: u16 = 1 << 9;
pub const ITEM_BERRY_JUICE: u16 = 1 << 10;
pub const ITEM_BERSERK_GENE: u16 = 1 << 11;

pub fn item_exclusion_bit(item_key: &str) -> u16 {
    match item_key {
        "leftovers" => ITEM_LEFTOVERS,
        "mintberry" => ITEM_MINT_BERRY,
        "miracleberry" => ITEM_MIRACLE_BERRY,
        "przcureberry" => ITEM_PRZCUREBERRY,
        "psncureberry" => ITEM_PSNCUREBERRY,
        "iceberry" => ITEM_ICE_BERRY,
        "burntberry" => ITEM_BURNT_BERRY,
        "bitterberry" => ITEM_BITTER_BERRY,
        "berry" => ITEM_BERRY,
        "goldberry" => ITEM_GOLD_BERRY,
        "berryjuice" => ITEM_BERRY_JUICE,
        "berserkgene" => ITEM_BERSERK_GENE,
        _ => 0,
    }
}

pub fn move_matches(dex: &Dex, slot: MoveId, revealed: MoveId) -> bool {
    slot == revealed
        || (dex.moves.key(revealed) == "hiddenpower"
            && dex.moves.key(slot).starts_with("hiddenpower"))
}

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
    /// identify the original item. `pub(crate)` so a hand-entered position
    /// can round-trip the whole channel (`position.rs`), not just the two
    /// halves a reader would guess at.
    pub(crate) gained: bool,
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
    pub hp_type_mask: u16,
    pub item: ItemObs,
    /// Bitmask of original held items deductively excluded by public
    /// Gen 2 Stadium 2 timings (`ITEM_*` constants).
    pub excluded_items: u16,
    /// The mon has switched in at least once (its pick is public).
    pub appeared: bool,
}

/// Observation tracker for one side of one battle. Construct at battle
/// start (team preview — item *presence* is read then), call `observe`
/// at every real decision point.
#[derive(Clone)]
pub struct Observer {
    side: usize,
    mons: Vec<MonObs>,
    /// Private diff snapshot of the opponent's item fields — never exposed;
    /// only its (publicly-announced) transitions produce knowledge.
    prev_item: Vec<Option<ItemId>>,
    log_pos: usize,
    /// Per-mon log-channel suppression: while a Transform copy is active
    /// (tracked from the log lines themselves, in order), `|move|` lines may
    /// name any move of the copied set, so none of them is evidence.
    log_suppress: Vec<bool>,
    /// Per-mon: the single move Mimic overwrote its own slot with. Mimic
    /// replaces exactly one slot, and PS names the copied move on the
    /// `|-activate|<mon>|move: Mimic|<Move>` line, so only that move has to
    /// be dropped — the mon's other three slots are still its own and every
    /// use of them is a genuine reveal. Cleared with the mon's stay, because
    /// `clearVolatile` restores `baseMoveSlots` on switch-out.
    mimic_overlay: Vec<Option<MoveId>>,
    /// Per-mon: a two-turn move a CALLER (Metronome / Mirror Move) put into
    /// its charge turn. The release line is a plain `|move|` with no `[from]`,
    /// so without this the caller's victim move is credited to the mon's own
    /// set — an over-reveal, which is worse than a miss: five "revealed" moves
    /// for four slots is unsatisfiable and makes `Belief::sync_checked` fail
    /// closed on the pinned/open-sheet path. `corpus.rs::collect_set_evidence`
    /// has carried this rule offline since the corpus harness was built; this
    /// is the same rule on the live channel.
    called_charging: Vec<Option<MoveId>>,
    /// Per-mon: a multi-turn locked move (`Outrage`, `Petal Dance`, `Thrash`,
    /// `Rollout`, `Bide`) started by a CALLER (`Metronome` / `Mirror Move`).
    /// Continuation turns emit plain `|move|` lines with no `[from]` until the
    /// lock ends (mon switches/faints or executes a different move).
    called_locked: Vec<Option<MoveId>>,
    /// The last line was a called move announcing its charge (`[from]` +
    /// `[still]`), awaiting the `-prepare` that confirms which move it is.
    pending_call: Option<(usize, MoveId)>,
    /// Tracks `|split|p1`/`p2` in raw `battle.log` so `Observer::observe(&Battle)`
    /// processes the exact same public stream as `PlayerChannel`.
    split_state: Option<(usize, bool)>,
    opp_status: Vec<Status>,
    opp_hp: Vec<Option<(u32, u32)>>,
    opp_active: Option<usize>,
    opp_confused: bool,
    opp_leftovers_healed: bool,
    opp_safeguard: bool,
    own_active_species: Option<SpeciesId>,
    own_active_alive: bool,
    own_type_modified: bool,
    pending_hp: Option<(usize, SpeciesId)>,
    revision: u64,
}

impl ItemObs {
    /// Rebuild the channel from a serialized position (`position.rs`). The
    /// three parts are not independent — `gained` decides whether a later
    /// reveal may still identify `original` — so they are set together
    /// rather than through three public fields.
    pub fn from_parts(
        original: Option<Option<ItemId>>,
        current: Option<Option<ItemId>>,
        gained: bool,
    ) -> ItemObs {
        ItemObs { original, current, gained }
    }
}

impl Observer {
    /// Solver mode: construct the reveal channel directly from a
    /// hand-entered position instead of replaying protocol lines. Same
    /// contract as `from_mons` — the state-diff channel never runs, so
    /// `observe` must not be called on the result.
    pub fn from_position(
        dex: &Dex,
        spec: &crate::position::PositionSpec,
    ) -> Result<Observer, String> {
        let foe = spec.foe();
        let item = |k: &crate::position::ItemKnowledge| -> Result<Option<Option<ItemId>>, String> {
            match k.to_obs() {
                None => Ok(None),
                Some(None) => Ok(Some(None)),
                Some(Some(key)) => dex
                    .items
                    .id(&toid(key))
                    .map(|i| Some(Some(i)))
                    .ok_or_else(|| format!("unknown item `{key}`")),
            }
        };
        let mut mons = Vec::new();
        for m in &spec.sides[foe].mons {
            if m.hp_type_mask == 0 {
                return Err("Hidden Power type constraint is empty".into());
            }
            if m.hp_type_mask != u16::MAX && !m.revealed_moves.iter().any(|m| toid(m).starts_with("hiddenpower")) {
                return Err("Hidden Power type constraint requires a revealed move".into());
            }
            let species = dex
                .species
                .id(&toid(&m.species))
                .ok_or_else(|| format!("unknown species `{}`", m.species))?;
            if m.hp_type_mask != u16::MAX {
                let gender = match m.gender.as_str() { "M" => Gender::M, "F" => Gender::F, _ => Gender::N };
                if m.hp_type_mask & crate::hp_evidence::possible_types(dex, species, gender) == 0 {
                    return Err("Hidden Power type contradicts public gender".into());
                }
            }
            if m.hp_type_mask != u16::MAX && m.revealed_moves.iter().any(|key| {
                let key = toid(key);
                key.strip_prefix("hiddenpower").filter(|suffix| !suffix.is_empty()).and_then(|suffix| {
                    crate::hp_evidence::TYPES.iter().position(|t| t.eq_ignore_ascii_case(suffix))
                }).is_some_and(|index| m.hp_type_mask & (1 << index) == 0)
            }) {
                return Err("Hidden Power type constraint contradicts a revealed move".into());
            }
            let mut revealed = Vec::new();
            for key in &m.revealed_moves {
                let id = dex
                    .moves
                    .id(&toid(key))
                    .ok_or_else(|| format!("unknown move `{key}`"))?;
                if !revealed.contains(&id) {
                    revealed.push(id);
                }
            }
            mons.push(MonObs {
                species,
                level: m.level,
                gender: match m.gender.as_str() {
                    "M" => Gender::M,
                    "F" => Gender::F,
                    _ => Gender::N,
                },
                name: m.name.clone(),
                preview_has_item: m.item_flag,
                revealed_moves: revealed,
                hp_type_mask: m.hp_type_mask,
                item: ItemObs::from_parts(item(&m.item_original)?, item(&m.item_current)?, m.item_gained),
                excluded_items: m.excluded_items,
                appeared: m.appeared,
            });
        }
        let mut obs = Observer::from_mons(spec.side, mons);
        if let Some(a) = spec.sides[spec.side].active {
            if let Some(ms) = spec.sides[spec.side].mons.get(a) {
                obs.own_active_species = dex.species.id(&toid(&ms.species));
                obs.own_active_alive = !ms.fainted;
            }
        }
        let opp = spec.foe();
        if let Some(a) = spec.sides[opp].active {
            if a < obs.mons.len() {
                let ms = &spec.sides[opp].mons[a];
                if !ms.fainted {
                    obs.opp_active = Some(a);
                }
                obs.opp_confused = ms.volatiles.iter().any(|v| v.key == "confusion");
            }
        }
        for (slot, ms) in spec.sides[opp].mons.iter().enumerate() {
            if slot < obs.mons.len() {
                obs.opp_hp[slot] = if ms.fainted {
                    Some((0, 100))
                } else if ms.hp_den > 0 {
                    Some((ms.hp_num.max(0) as u32, ms.hp_den as u32))
                } else {
                    None
                };
                obs.opp_status[slot] = Status::from_str(&ms.status);
            }
        }
        obs.opp_safeguard = spec.sides[opp].conditions.iter().any(|c| c.key == "safeguard");
        Ok(obs)
    }

    /// M15 protocol mode: construct from preview-public facts parsed out of
    /// the protocol (`|poke|` lines) instead of a battle. The state-diff
    /// channel never runs (there is no true battle to diff — `observe` must
    /// not be called); all knowledge arrives via `ingest_line`. Nicknames
    /// are learned from switch lines by the caller (`set_name`) — subject
    /// resolution needs them before the mon's first `|move|` line, and
    /// switch lines always precede them.
    pub(crate) fn from_mons(side: usize, mons: Vec<MonObs>) -> Observer {
        let n = mons.len();
        Observer {
            side,
            mons,
            prev_item: vec![None; n],
            log_pos: 0,
            log_suppress: vec![false; n],
            mimic_overlay: vec![None; n],
            called_charging: vec![None; n],
            called_locked: vec![None; n],
            pending_call: None,
            split_state: None,
            opp_status: vec![Status::None; n],
            opp_hp: vec![None; n],
            opp_active: None,
            opp_confused: false,
            opp_leftovers_healed: false,
            opp_safeguard: false,
            own_active_species: None,
            own_active_alive: false,
            own_type_modified: false,
            pending_hp: None,
            revision: 0,
        }
    }

    /// Protocol mode: feed one player-visible protocol line through the log
    /// channel (the only channel — see `from_mons`).
    pub(crate) fn ingest_line(&mut self, line: &str, dex: &Dex) {
        if self.scan_line(line, dex) {
            self.revision += 1;
        }
    }

    /// Protocol mode: a mon appeared under `name` (resolves log subjects).
    pub(crate) fn set_name(&mut self, slot: usize, name: &str) {
        if self.mons[slot].name != name {
            self.mons[slot].name = name.to_string();
        }
        if !self.mons[slot].appeared {
            self.mons[slot].appeared = true;
            self.revision += 1;
        }
    }

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
                hp_type_mask: u16::MAX,
                item: ItemObs::default(),
                excluded_items: 0,
                appeared: false,
            })
            .collect::<Vec<_>>();
        let prev_item = battle.sides[opp].roster.iter().map(|p| p.item).collect();
        let n = mons.len();
        Observer {
            side,
            mons,
            prev_item,
            log_pos: 0,
            log_suppress: vec![false; n],
            mimic_overlay: vec![None; n],
            called_charging: vec![None; n],
            called_locked: vec![None; n],
            pending_call: None,
            split_state: None,
            opp_status: vec![Status::None; n],
            opp_hp: vec![None; n],
            opp_active: None,
            opp_confused: false,
            opp_leftovers_healed: false,
            opp_safeguard: false,
            own_active_species: None,
            own_active_alive: false,
            own_type_modified: false,
            pending_hp: None,
            revision: 0,
        }
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

    /// True if `item_id` is impossible for opponent slot `slot` given
    /// revealed original item, preview item flag, Item Clause, and deductive
    /// Gen 2 Stadium 2 exclusions (`excluded_items`).
    pub fn is_item_excluded_for_slot(
        dex: &Dex,
        mons: &[MonObs],
        slot: usize,
        item_id: Option<ItemId>,
    ) -> bool {
        let Some(mo) = mons.get(slot) else { return false };
        if let Some(known) = mo.item.original {
            return known != item_id;
        }
        match item_id {
            None => mo.preview_has_item,
            Some(id) => {
                if !mo.preview_has_item {
                    return true;
                }
                if mons
                    .iter()
                    .enumerate()
                    .any(|(j, other)| j != slot && other.item.original == Some(Some(id)))
                {
                    return true;
                }
                let bit = item_exclusion_bit(dex.items.key(id));
                bit != 0 && (mo.excluded_items & bit) != 0
            }
        }
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
                dirty = true;
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
        if let Some((owner, secret)) = self.split_state {
            self.split_state = secret.then_some((owner, false));
            if (owner == self.side) != secret {
                return false;
            }
        }
        let Some(body) = line.strip_prefix('|') else { return false };
        let parts: Vec<&str> = body.split('|').collect();
        let cmd = parts.first().copied().unwrap_or("");
        if cmd == "split" {
            let p = parts.get(1).copied().unwrap_or("");
            let owner = if p == "p1" { Some(0) } else if p == "p2" { Some(1) } else { None };
            if let Some(o) = owner {
                self.split_state = Some((o, true));
            }
            return false;
        }

        // The charge announcement and its `-prepare` are adjacent; anything
        // in between means this was not a charging call after all.
        let pending = self.pending_call.take();

        let mut dirty = false;
        if let Some((hp_slot, def_sp)) = self.pending_hp {
            let subj = parts.get(1).copied().unwrap_or("");
            let own_hit = self.subject_is_own_side(subj);
            let mut consume_hp = false;
            let mut eff = None;
            match cmd {
                "-supereffective" if own_hit => {
                    eff = Some(crate::hp_evidence::Effectiveness::SuperEffective);
                    consume_hp = true;
                }
                "-resisted" if own_hit => {
                    eff = Some(crate::hp_evidence::Effectiveness::Resisted);
                    consume_hp = true;
                }
                "-immune" if own_hit && !parts.iter().any(|p| *p == "[ohko]") => {
                    eff = Some(crate::hp_evidence::Effectiveness::Immune);
                    consume_hp = true;
                }
                "-damage" if own_hit && !parts.iter().any(|p| p.starts_with("[from] ")) => {
                    eff = Some(crate::hp_evidence::Effectiveness::Neutral);
                    consume_hp = true;
                }
                "-activate"
                    if own_hit
                        && parts.get(2) == Some(&"Substitute")
                        && parts.iter().any(|p| *p == "[damage]") =>
                {
                    eff = Some(crate::hp_evidence::Effectiveness::Neutral);
                    consume_hp = true;
                }
                "-activate"
                    if own_hit
                        && matches!(parts.get(2).copied().unwrap_or(""), "Protect" | "Detect") =>
                {
                    consume_hp = true;
                }
                "-end"
                    if own_hit
                        && parts.get(2) == Some(&"Substitute")
                        && !parts.iter().any(|p| p.starts_with("[from] ")) =>
                {
                    eff = Some(crate::hp_evidence::Effectiveness::Neutral);
                    consume_hp = true;
                }
                "move" | "cant" | "switch" | "drag" | "faint" | "turn" | "upkeep" | "-miss"
                | "-fail" => {
                    consume_hp = true;
                }
                _ => {}
            }
            if let Some(effectiveness) = eff {
                dirty |= self.narrow_hp_mask(dex, hp_slot, def_sp, effectiveness);
            }
            if consume_hp {
                self.pending_hp = None;
            }
        }

        match cmd {
            "-prepare" => {
                let subject = parts.get(1).copied().unwrap_or("");
                let named = parts.get(2).copied().unwrap_or("");
                if let (Some(m), Some(id)) = (self.opp_subject(subject), dex.moves.id(&toid(named)))
                {
                    if pending == Some((m, id)) {
                        self.called_charging[m] = Some(id);
                    }
                }
            }
            "-anim" => {
                if let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) {
                    self.called_charging[m] = None;
                }
            }
            "cant" => {
                dirty |= self.check_on_update_exclusions();
                let subj = parts.get(1).copied().unwrap_or("");
                let reason = parts.get(2).copied().unwrap_or("");
                if let Some(m) = self.opp_subject(subj) {
                    self.called_charging[m] = None;
                    if reason == "slp" {
                        dirty |= self.exclude_items(m, ITEM_MINT_BERRY | ITEM_MIRACLE_BERRY);
                    } else if reason == "par" || reason == "frz" {
                        dirty |= self.exclude_items(m, ITEM_MIRACLE_BERRY);
                    }
                }
            }
            "move" => {
                dirty |= self.scan_move(&parts, dex);
            }
            "turn" => {
                self.opp_leftovers_healed = false;
                dirty |= self.check_on_update_exclusions();
            }
            "upkeep" => {
                if let Some(m) = self.opp_active {
                    if let Some((num, den)) = self.opp_hp[m] {
                        if num > 0 && den > 0 {
                            if num < den && !self.opp_leftovers_healed {
                                dirty |= self.exclude_items(m, ITEM_LEFTOVERS);
                            }
                            if (den == 100 && num * 2 < 100)
                                || (den != 100 && (num + 1) * 2 <= den)
                            {
                                dirty |= self.exclude_items(
                                    m,
                                    ITEM_BERRY | ITEM_GOLD_BERRY | ITEM_BERRY_JUICE,
                                );
                            }
                            if self.opp_status[m] == Status::Slp {
                                dirty |=
                                    self.exclude_items(m, ITEM_MINT_BERRY | ITEM_MIRACLE_BERRY);
                            } else if self.opp_status[m] != Status::None || self.opp_confused {
                                dirty |= self.exclude_items(m, ITEM_MIRACLE_BERRY);
                            }
                        }
                    }
                }
                self.opp_leftovers_healed = false;
            }
            "switch" | "drag" => {
                let subj = parts.get(1).copied().unwrap_or("");
                let det = parts.get(2).copied().unwrap_or("");
                let hp_s = parts.get(3).copied().unwrap_or("");
                if self.subject_is_opp_side(subj) {
                    self.log_suppress.iter_mut().for_each(|s| *s = false);
                    self.mimic_overlay.iter_mut().for_each(|o| *o = None);
                    self.called_charging.iter_mut().for_each(|c| *c = None);
                    self.called_locked.iter_mut().for_each(|c| *c = None);
                    self.opp_confused = false;
                    self.opp_leftovers_healed = false;
                    let slot = self.opp_subject(subj).or_else(|| self.opp_switch_slot(det, dex));
                    self.opp_active = slot;
                    if let Some(m) = slot {
                        let (hp, st) = Self::parse_hp_status(hp_s);
                        if let Some(pair) = hp {
                            self.opp_hp[m] = Some(pair);
                        }
                        self.opp_status[m] = st;
                    }
                } else if self.subject_is_own_side(subj) {
                    let sp_name = det.split(',').next().unwrap_or("");
                    self.own_active_species = dex.species.id(&toid(sp_name));
                    let (hp, _) = Self::parse_hp_status(hp_s);
                    self.own_active_alive = hp.map_or(true, |(n, _)| n > 0);
                    self.own_type_modified = false;
                }
            }
            "faint" => {
                let subj = parts.get(1).copied().unwrap_or("");
                if self.subject_is_opp_side(subj) {
                    self.log_suppress.iter_mut().for_each(|s| *s = false);
                    self.mimic_overlay.iter_mut().for_each(|o| *o = None);
                    self.called_charging.iter_mut().for_each(|c| *c = None);
                    self.called_locked.iter_mut().for_each(|c| *c = None);
                    self.opp_confused = false;
                    if let Some(m) = self.opp_subject(subj).or(self.opp_active) {
                        self.opp_hp[m] = Some((0, 100));
                        self.opp_status[m] = Status::None;
                    }
                    self.opp_active = None;
                } else if self.subject_is_own_side(subj) {
                    self.own_active_alive = false;
                    self.own_type_modified = false;
                }
            }
            "-damage" => {
                let subj = parts.get(1).copied().unwrap_or("");
                let hp_s = parts.get(2).copied().unwrap_or("");
                if let Some(m) = self.opp_subject(subj) {
                    let (hp, st) = Self::parse_hp_status(hp_s);
                    if let Some(pair) = hp {
                        self.opp_hp[m] = Some(pair);
                        if pair.0 == 0 {
                            self.opp_status[m] = Status::None;
                        } else if st != Status::None {
                            self.opp_status[m] = st;
                        }
                    }
                } else if self.subject_is_own_side(subj) {
                    let (hp, _) = Self::parse_hp_status(hp_s);
                    if hp.is_some_and(|(n, _)| n == 0) {
                        self.own_active_alive = false;
                    }
                }
            }
            "-sethp" => {
                for pair_idx in [1usize, 3usize] {
                    if let (Some(subj), Some(hp_s)) = (parts.get(pair_idx), parts.get(pair_idx + 1)) {
                        if let Some(m) = self.opp_subject(subj) {
                            let (hp, _) = Self::parse_hp_status(hp_s);
                            if let Some(p) = hp {
                                self.opp_hp[m] = Some(p);
                            }
                        }
                    }
                }
            }
            "-status" => {
                let subj = parts.get(1).copied().unwrap_or("");
                let st_s = parts.get(2).copied().unwrap_or("");
                if let Some(m) = self.opp_subject(subj) {
                    self.opp_status[m] = Status::from_str(st_s);
                }
            }
            "-curestatus" => {
                let subj = parts.get(1).copied().unwrap_or("");
                let st_s = parts.get(2).copied().unwrap_or("");
                if let Some(m) = self.opp_subject(subj) {
                    if st_s == "confusion" {
                        if self.opp_active == Some(m) {
                            self.opp_confused = false;
                        }
                    } else {
                        self.opp_status[m] = Status::None;
                    }
                }
            }
            "-cureteam" => {
                let subj = parts.get(1).copied().unwrap_or("");
                if self.subject_is_opp_side(subj) {
                    self.opp_status.iter_mut().for_each(|s| *s = Status::None);
                }
            }
            "-start" => {
                let subj = parts.get(1).copied().unwrap_or("");
                let what = parts.get(2).copied().unwrap_or("");
                if self.opp_subject(subj).is_some() && what == "confusion" {
                    self.opp_confused = true;
                }
                if self.subject_is_own_side(subj)
                    && (what == "typechange" || what.contains("Foresight"))
                {
                    self.own_type_modified = true;
                }
            }
            "-end" => {
                let subj = parts.get(1).copied().unwrap_or("");
                let what = parts.get(2).copied().unwrap_or("");
                if self.opp_subject(subj).is_some() && what == "confusion" {
                    self.opp_confused = false;
                }
            }
            "-sidestart" | "-sideend" => {
                let side_s = parts.get(1).copied().unwrap_or("");
                let cond = parts.get(2).copied().unwrap_or("");
                let opp_prefix = if self.opp() == 0 { "p1" } else { "p2" };
                if side_s.starts_with(opp_prefix) && cond.contains("Safeguard") {
                    self.opp_safeguard = cmd == "-sidestart";
                }
            }
            "-transform" => {
                let subj = parts.get(1).copied().unwrap_or("");
                if let Some(m) = self.opp_subject(subj) {
                    self.log_suppress[m] = true;
                }
                if self.subject_is_own_side(subj) {
                    self.own_type_modified = true;
                }
            }
            "-activate" => {
                if let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) {
                    let what = parts.get(2).copied().unwrap_or("");
                    if what == "move: Mimic" {
                        match parts.get(3).and_then(|n| dex.moves.id(&toid(n))) {
                            Some(copied) => self.mimic_overlay[m] = Some(copied),
                            None => self.log_suppress[m] = true,
                        }
                    } else if what == "confusion" {
                        dirty |= self.exclude_items(m, ITEM_MIRACLE_BERRY);
                    } else if let Some(item_name) = what.strip_prefix("item: ") {
                        if let Some(id) = dex.items.id(&toid(item_name)) {
                            dirty |= self.mons[m].item.revealed_held(id);
                        }
                    }
                }
            }
            "-enditem" => {
                if let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) {
                    if let Some(id) = dex.items.id(&toid(parts.get(2).copied().unwrap_or(""))) {
                        dirty |= self.mons[m].item.lost(id);
                    }
                }
            }
            "-item" => {
                let subject = parts.get(1).copied().unwrap_or("");
                if let Some(id) = dex.items.id(&toid(parts.get(2).copied().unwrap_or(""))) {
                    if let Some(m) = self.opp_subject(subject) {
                        dirty |= self.mons[m].item.gained(id);
                    } else if parts.iter().any(|p| *p == "[from] move: Thief") {
                        if let Some(of) = parts.iter().find_map(|p| p.strip_prefix("[of] ")) {
                            if let Some(v) = self.opp_subject(of) {
                                dirty |= self.mons[v].item.lost(id);
                            }
                        }
                    }
                }
            }
            "-heal" => {
                if let Some(m) = self.opp_subject(parts.get(1).copied().unwrap_or("")) {
                    let hp_s = parts.get(2).copied().unwrap_or("");
                    let (hp, _) = Self::parse_hp_status(hp_s);
                    if let Some(pair) = hp {
                        self.opp_hp[m] = Some(pair);
                    }
                    if let Some(item_name) =
                        parts.iter().find_map(|p| p.strip_prefix("[from] item: "))
                    {
                        if toid(item_name) == "leftovers" {
                            self.opp_leftovers_healed = true;
                        }
                        if let Some(id) = dex.items.id(&toid(item_name)) {
                            dirty |= self.mons[m].item.revealed_held(id);
                        }
                    }
                }
            }
            _ => {}
        }
        dirty
    }

    fn scan_move(&mut self, parts: &[&str], dex: &Dex) -> bool {
        let subj = parts.get(1).copied().unwrap_or("");
        let is_pursuit_intercept = parts
            .iter()
            .any(|p| *p == "[from] Pursuit" || *p == "[from] move: Pursuit");
        let mut dirty = false;
        if !is_pursuit_intercept {
            dirty |= self.check_on_update_exclusions();
        }
        let Some(m) = self.opp_subject(subj) else {
            return dirty;
        };
        let from = parts
            .iter()
            .find_map(|p| p.strip_prefix("[from] "))
            .map(|f| f.strip_prefix("move: ").unwrap_or(f));
        if from.is_none() {
            if self.opp_status[m] == Status::Slp {
                dirty |= self.exclude_items(m, ITEM_MINT_BERRY | ITEM_MIRACLE_BERRY);
            } else if self.opp_status[m] != Status::None || self.opp_confused {
                dirty |= self.exclude_items(m, ITEM_MIRACLE_BERRY);
            }
        }
        if self.log_suppress[m] {
            // Transform copy active: plain |move| lines (and even Sleep Talk
            // calls) may name any move of the copied set.
            return dirty;
        }
        let named = parts.get(2).copied().unwrap_or("");
        let Some(id) = dex.moves.id(&toid(named)) else { return dirty };
        if self.opp_safeguard && matches!(dex.moves.key(id), "outrage" | "petaldance" | "thrash") {
            self.opp_confused = false;
        }
        if self.mimic_overlay[m] == Some(id) {
            // the one slot Mimic overwrote. Sleep Talk selects from the
            // CURRENT slots, so this must be checked before the `[from]`
            // rules, not only for plain lines.
            return dirty;
        }
        if let Some(from) = from {
            let from_id = dex.moves.id(&toid(from));
            let self_named = from_id.is_some_and(|f| move_matches(dex, id, f));
            let sleep_talk = toid(from) == "sleeptalk";
            if !self_named && !sleep_talk {
                if parts.iter().any(|p| *p == "[still]") {
                    self.pending_call = Some((m, id));
                }
                if matches!(
                    dex.moves.key(id),
                    "outrage" | "petaldance" | "thrash" | "rollout" | "bide"
                ) {
                    self.called_locked[m] = Some(id);
                }
                return dirty;
            }
        }
        if from.is_none() && self.called_charging[m] == Some(id) {
            // the release of the charge a caller started: not a submitted slot
            self.called_charging[m] = None;
            return dirty;
        }
        if from.is_none() && self.called_locked[m] == Some(id) {
            // continuation turn of a caller-initiated multi-turn lock
            return dirty;
        }
        // synthetic non-set moves
        if matches!(dex.moves.key(id), "struggle" | "recharge") {
            return dirty;
        }
        if from.is_none() {
            self.called_locked[m] = None;
        }
        if !self.mons[m].revealed_moves.contains(&id) {
            self.mons[m].revealed_moves.push(id);
            dirty = true;
        }
        if dex.moves.key(id) == "hiddenpower" {
            let tgt = parts.get(3).copied().unwrap_or("");
            if self.subject_is_own_side(tgt) && !self.own_type_modified {
                if let Some(def_sp) = self.own_active_species {
                    self.pending_hp = Some((m, def_sp));
                }
            }
        }
        dirty
    }

    fn exclude_items(&mut self, slot: usize, bits: u16) -> bool {
        if bits == 0 {
            return false;
        }
        let Some(mo) = self.mons.get_mut(slot) else { return false };
        if !mo.preview_has_item
            || mo.item.original.is_some()
            || mo.item.current.is_some()
            || mo.item.gained
        {
            return false;
        }
        let next = mo.excluded_items | bits;
        if next != mo.excluded_items {
            mo.excluded_items = next;
            true
        } else {
            false
        }
    }

    fn check_on_update_exclusions(&mut self) -> bool {
        if !self.own_active_alive {
            return false;
        }
        let Some(m) = self.opp_active else { return false };
        if !self.opp_hp.get(m).copied().flatten().map_or(true, |(n, _)| n > 0) {
            return false;
        }
        let mut bits = ITEM_BERSERK_GENE;
        match self.opp_status[m] {
            Status::Par => bits |= ITEM_PRZCUREBERRY,
            Status::Psn | Status::Tox => bits |= ITEM_PSNCUREBERRY,
            Status::Brn => bits |= ITEM_ICE_BERRY,
            Status::Frz => bits |= ITEM_BURNT_BERRY,
            _ => {}
        }
        if self.opp_confused {
            bits |= ITEM_BITTER_BERRY;
        }
        self.exclude_items(m, bits)
    }

    fn narrow_hp_mask(
        &mut self,
        dex: &Dex,
        slot: usize,
        def_sp: SpeciesId,
        eff: crate::hp_evidence::Effectiveness,
    ) -> bool {
        let Some(mo) = self.mons.get_mut(slot) else { return false };
        let def_types = dex.species_types(def_sp);
        let eff_mask = crate::hp_evidence::types_with_effectiveness(dex, &def_types, eff);
        let legal = crate::hp_evidence::possible_types(dex, mo.species, mo.gender);
        let next = mo.hp_type_mask & eff_mask;
        if next & legal == 0 {
            return false;
        }
        if next != mo.hp_type_mask {
            mo.hp_type_mask = next;
            true
        } else {
            false
        }
    }

    fn parse_hp_status(s: &str) -> (Option<(u32, u32)>, Status) {
        let s = s.trim();
        if s == "0 fnt" || s.starts_with("0 ") {
            return (Some((0, 100)), Status::None);
        }
        let mut iter = s.split_whitespace();
        let hp = iter.next().and_then(|frac| {
            let (a, b) = frac.split_once('/')?;
            Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?))
        });
        let st = iter.next().map(Status::from_str).unwrap_or(Status::None);
        (hp, st)
    }

    fn opp_switch_slot(&self, details: &str, dex: &Dex) -> Option<usize> {
        let mut segs = details.split(',').map(str::trim);
        let sp = dex.species.id(&toid(segs.next().unwrap_or("")))?;
        let mut level = 100u8;
        let mut gender = Gender::N;
        for seg in segs {
            if let Some(lv) = seg.strip_prefix('L').and_then(|v| v.parse::<u8>().ok()) {
                level = lv;
            } else if seg == "M" {
                gender = Gender::M;
            } else if seg == "F" {
                gender = Gender::F;
            }
        }
        self.mons
            .iter()
            .position(|m| m.species == sp && m.level == level && m.gender == gender)
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

    fn subject_is_own_side(&self, s: &str) -> bool {
        let b = s.as_bytes();
        b.len() >= 4 && b[0] == b'p' && b[1] == b'1' + self.side as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use conformance::load_dex;

    fn one_mon_observer(dex: &Dex, species: &str, name: &str) -> Observer {
        let mons = vec![MonObs {
            species: dex.species.id(&toid(species)).expect("species"),
            level: 50,
            gender: Gender::N,
            name: name.to_string(),
            preview_has_item: false,
            revealed_moves: Vec::new(),
            hp_type_mask: u16::MAX,
            item: Default::default(),
            excluded_items: 0,
            appeared: true,
        }];
        // side 0 observes side 1 (`p2a:` subjects).
        Observer::from_mons(0, mons)
    }

    fn revealed(obs: &Observer, dex: &Dex) -> Vec<String> {
        obs.mons[0]
            .revealed_moves
            .iter()
            .map(|&id| dex.moves.key(id).to_string())
            .collect()
    }

    #[test]
    fn live_hp_effectiveness_narrows_and_respects_transforms_and_calls() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Aerodactyl", "A");
        // SE against Snorlax (Normal) -> Fighting only
        for line in [
            "|switch|p1a: N|Snorlax, L50|100/100",
            "|move|p2a: A|Hidden Power|p1a: N",
            "|-supereffective|p1a: N",
            "|-damage|p1a: N|62/100",
        ] {
            obs.ingest_line(line, &dex);
        }
        let fight_bit = crate::hp_evidence::types_with_effectiveness(
            &dex,
            &dex.species_types(dex.species.id("snorlax").unwrap()),
            crate::hp_evidence::Effectiveness::SuperEffective,
        );
        assert_eq!(obs.mons[0].hp_type_mask, fight_bit);

        // Resisted + Immune via Substitute and Slept Sleep Talk
        let mut obs2 = one_mon_observer(&dex, "Zapdos", "Z");
        for line in [
            "|switch|p1a: S|Skarmory, L50|100/100",
            "|move|p2a: Z|Hidden Power|p1a: S|[from] Sleep Talk",
            "|-activate|p1a: S|Substitute|[damage]",
        ] {
            obs2.ingest_line(line, &dex);
        }
        let neutral_skarm = crate::hp_evidence::types_with_effectiveness(
            &dex,
            &dex.species_types(dex.species.id("skarmory").unwrap()),
            crate::hp_evidence::Effectiveness::Neutral,
        );
        assert_eq!(obs2.mons[0].hp_type_mask, neutral_skarm);

        // Metronome-called Hidden Power or Transform/typechange own mon must NOT narrow
        let mut obs3 = one_mon_observer(&dex, "Clefable", "C");
        for line in [
            "|switch|p1a: N|Snorlax, L50|100/100",
            "|move|p2a: C|Hidden Power|p1a: N|[from] Metronome",
            "|-supereffective|p1a: N",
            "|-start|p1a: N|typechange|Ghost",
            "|move|p2a: C|Hidden Power|p1a: N",
            "|-resisted|p1a: N",
        ] {
            obs3.ingest_line(line, &dex);
        }
        assert_eq!(obs3.mons[0].hp_type_mask, u16::MAX);
    }

    #[test]
    fn deductive_item_exclusions_track_timings_and_item_clause() {
        let dex = load_dex();
        let mons = vec![
            MonObs {
                species: dex.species.id("snorlax").unwrap(),
                level: 50,
                gender: Gender::M,
                name: "Lax".into(),
                preview_has_item: true,
                revealed_moves: vec![],
                hp_type_mask: u16::MAX,
                item: Default::default(),
                excluded_items: 0,
                appeared: true,
            },
            MonObs {
                species: dex.species.id("zapdos").unwrap(),
                level: 50,
                gender: Gender::N,
                name: "Zap".into(),
                preview_has_item: true,
                revealed_moves: vec![],
                hp_type_mask: u16::MAX,
                item: Default::default(),
                excluded_items: 0,
                appeared: false,
            },
        ];
        let mut obs = Observer::from_mons(0, mons);
        // Before upkeep: 49/100 mid-turn must NOT exclude Berry / Leftovers yet.
        for line in [
            "|switch|p1a: Own|Raikou, L50|100/100",
            "|switch|p2a: Lax|Snorlax, L50, M|100/100",
            "|turn|1",
            "|-damage|p2a: Lax|50/100",
            "|upkeep",
        ] {
            obs.ingest_line(line, &dex);
        }
        // At 50/100 on upkeep without Leftovers heal: Leftovers & Berserk Gene excluded,
        // heal berries NOT yet excluded!
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("leftovers")));
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("berserkgene")));
        assert!(!Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("goldberry")));
        assert!(!Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("berry")));

        // Mid-turn 49/100 before upkeep still does not exclude berries; upkeep at 49/100 does!
        obs.ingest_line("|-damage|p2a: Lax|49/100", &dex);
        assert!(!Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("goldberry")));
        obs.ingest_line("|upkeep", &dex);
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("goldberry")));
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("berry")));
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("berryjuice")));

        // Sleep + Sleep Talk excludes Mint Berry and Miracle Berry
        for line in [
            "|-status|p2a: Lax|slp",
            "|turn|2",
            "|move|p2a: Lax|Sleep Talk|p2a: Lax",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("mintberry")));
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("miracleberry")));

        // Heal Bell clears status so slot 1 switching in asleep after Heal Bell isn't penalized
        obs.ingest_line("|-cureteam|p2a: Lax|[from] move: Heal Bell", &dex);
        obs.ingest_line("|switch|p2a: Zap|Zapdos, L50|100/100", &dex);
        obs.ingest_line("|turn|3", &dex);
        assert!(!Observer::is_item_excluded_for_slot(&dex, obs.mons(), 1, dex.items.id("mintberry")));

        // Item Clause: revealing Leftovers on slot 1 excludes Leftovers on slot 0 even if un-excluded
        obs.ingest_line("|-heal|p2a: Zap|100/100|[from] item: Leftovers", &dex);
        assert!(Observer::is_item_excluded_for_slot(&dex, obs.mons(), 0, dex.items.id("leftovers")));
    }

    /// A caller charging a two-turn move must not credit the victim move to
    /// the mon's own set. Corpus battle 215 is the observed case: p1 Snorlax
    /// ended with FIVE revealed moves for four slots (substitute, doubleteam,
    /// attract, metronome, razorwind) because Metronome's Razor Wind released
    /// on a plain `|move|` line. Five facts for four slots is unsatisfiable,
    /// so `Belief::sync_checked` fails closed on the pinned/open-sheet path —
    /// the shipped product's own configuration.
    #[test]
    fn called_two_turn_release_is_not_a_reveal() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Snorlax", "Snorlax");
        for line in [
            "|move|p2a: Snorlax|Metronome|p2a: Snorlax",
            "|move|p2a: Snorlax|Razor Wind||[from] Metronome|[still]",
            "|-prepare|p2a: Snorlax|Razor Wind",
            "|move|p2a: Snorlax|Razor Wind|p1a: Target",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(revealed(&obs, &dex), vec!["metronome".to_string()]);
    }

    /// `[from]` naming the executing move itself is a genuine reveal, not a
    /// call. Pursuit's interception of a switching target is the case that
    /// occurs in the corpus (4 lines, all Umbreon): dropping it means the bot
    /// never learns a move it publicly watched land, which is the design's own
    /// class-A example — "ignoring an already-revealed move".
    #[test]
    fn a_self_named_from_tag_is_a_reveal() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Umbreon", "Umbreon");
        obs.ingest_line("|move|p2a: Umbreon|Pursuit|p1a: Rapidash|[from] Pursuit", &dex);
        assert_eq!(revealed(&obs, &dex), vec!["pursuit".to_string()]);
    }

    /// The rule must not eat a genuine reveal: the same move used normally on
    /// a later turn is the mon's own.
    #[test]
    fn an_uncalled_two_turn_move_still_reveals() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Snorlax", "Snorlax");
        for line in [
            "|move|p2a: Snorlax|Razor Wind||[still]",
            "|-prepare|p2a: Snorlax|Razor Wind",
            "|move|p2a: Snorlax|Razor Wind|p1a: Target",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(revealed(&obs, &dex), vec!["razorwind".to_string()]);
    }

    /// Mimic overwrites exactly ONE slot, and PS names the copied move on
    /// the `-activate` line. Suppressing the whole mon instead loses every
    /// later use of its other three slots — an under-reveal of moves the bot
    /// publicly watched land. The copied move itself stays uncredited, and so
    /// does a Sleep Talk call that selects it (Sleep Talk picks from the
    /// CURRENT slots, which include the overwritten one).
    #[test]
    fn mimic_suppresses_only_the_copied_slot() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Zapdos", "Zapdos");
        for line in [
            "|move|p2a: Zapdos|Mimic|p1a: Snorlax",
            "|-activate|p2a: Zapdos|move: Mimic|Body Slam",
            // the copied slot: not the mon's own, by either route
            "|move|p2a: Zapdos|Body Slam|p1a: Snorlax",
            "|move|p2a: Zapdos|Body Slam|p1a: Snorlax|[from] Sleep Talk",
            // its own remaining slots stay reveals
            "|move|p2a: Zapdos|Thunder|p1a: Snorlax",
            "|move|p2a: Zapdos|Rest|p2a: Zapdos",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(
            revealed(&obs, &dex),
            vec!["mimic".to_string(), "thunder".to_string(), "rest".to_string()]
        );
    }

    /// `clearVolatile` puts the Mimic slot back on switch-out, so the copied
    /// move must stop being suppressed — otherwise a genuine later use by a
    /// DIFFERENT mon of the same name, or by this one after it really learns
    /// it, would be dropped forever.
    #[test]
    fn a_mimic_overlay_dies_with_the_mons_stay() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Zapdos", "Zapdos");
        for line in [
            "|move|p2a: Zapdos|Mimic|p1a: Snorlax",
            "|-activate|p2a: Zapdos|move: Mimic|Body Slam",
            "|switch|p2a: Zapdos|Zapdos, L50|100/100",
            "|move|p2a: Zapdos|Body Slam|p1a: Snorlax",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(
            revealed(&obs, &dex),
            vec!["mimic".to_string(), "bodyslam".to_string()]
        );
    }

    /// Transform stays whole-mon suppression: the copy replaces all four
    /// slots, so no `|move|` line from it is evidence about the base set.
    #[test]
    fn transform_still_suppresses_every_slot() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Ditto", "Ditto");
        for line in [
            "|move|p2a: Ditto|Transform|p1a: Snorlax",
            "|-transform|p2a: Ditto|p1a: Snorlax",
            "|move|p2a: Ditto|Body Slam|p1a: Snorlax",
            "|move|p2a: Ditto|Rest|p2a: Ditto",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(revealed(&obs, &dex), vec!["transform".to_string()]);
    }

    /// An interrupted charge releases the tracking, so a later genuine use of
    /// the same move is still credited.
    #[test]
    fn an_interrupted_charge_does_not_suppress_forever() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Snorlax", "Snorlax");
        for line in [
            "|move|p2a: Snorlax|Razor Wind||[from] Metronome|[still]",
            "|-prepare|p2a: Snorlax|Razor Wind",
            "|cant|p2a: Snorlax|par",
            "|move|p2a: Snorlax|Razor Wind|p1a: Target",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(revealed(&obs, &dex), vec!["razorwind".to_string()]);
    }

    #[test]
    fn a_called_multiturn_locked_move_does_not_reveal_slot() {
        let dex = load_dex();
        let mut obs = one_mon_observer(&dex, "Granbull", "Granbull");
        for line in [
            "|move|p2a: Granbull|Metronome|p2a: Granbull",
            "|move|p2a: Granbull|Rollout|p1a: Electrode|[from] move: Metronome",
            "|move|p2a: Granbull|Rollout|p1a: Electrode",
            "|move|p2a: Granbull|Rollout|p1a: Electrode",
            "|move|p2a: Granbull|Mud-Slap|p1a: Electrode",
        ] {
            obs.ingest_line(line, &dex);
        }
        assert_eq!(
            revealed(&obs, &dex),
            vec!["metronome".to_string(), "mudslap".to_string()]
        );
    }
}
