//! M3 search API: legal-choice enumeration + structured stepping for
//! DUCT/MCTS-style self-play.
//!
//! Contract with the rest of the engine: enumeration mirrors the validation
//! rules of `choices.rs` exactly (same helpers, same order of checks), and
//! `apply` funnels through `Battle::choose` with the PS-canonical choice
//! string — one code path shared with fixture replay, so search can never
//! drift from conformance-verified semantics.
//!
//! Search usage: `Battle` is a plain deep-clonable value. Typical loop:
//! ```ignore
//! battle.set_log_enabled(false);          // search mode: skip protocol log
//! while battle.outcome().is_none() {
//!     let picks = [0, 1].map(|s| {
//!         let cs = battle.legal_choices(&dex, s);
//!         if cs.is_empty() { None } else { Some(cs[rng(cs.len())]) }
//!     });
//!     battle.apply_choices(&dex, picks).unwrap();
//! }
//! ```
//! Determinized playouts: clone the battle, then `reseed(...)` the clone so
//! each playout samples fresh chance outcomes.

use crate::dex::{Dex, MoveId};
use crate::prng::Prng;
use crate::state::*;

use super::EngineError;

/// FxHash-style word-folding hasher (rustc-hash algorithm): deterministic,
/// no dependencies, ~5x faster than SipHash on the state-key walk, quality
/// plenty for 64-bit search-tree keys. Not DoS-resistant — never use for
/// attacker-controlled map keys.
#[derive(Default)]
struct FxHasher(u64);

impl FxHasher {
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(Self::K);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rem.len()].copy_from_slice(rem);
            self.add(u64::from_le_bytes(buf) ^ rem.len() as u64);
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
}

/// A single side's choice, compact and `Copy` (fits in search tree nodes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SearchChoice {
    /// Team-preview pick: 1-based display positions in lead order.
    /// Trailing zeros = unused (only when fewer than 3 are picked, which
    /// NC2000 never does).
    Team([u8; 3]),
    /// Use a move by id. Locked/recharge/struggle turns enumerate exactly one.
    Move(MoveId),
    /// Switch to the 1-based display position (bench = 2..=party size).
    Switch(u8),
    /// Forced pass (only legal when a forced switch has no legal target).
    Pass,
}

impl SearchChoice {
    /// The PS-canonical inputLog string for this choice (what fixtures carry
    /// and what `Battle::choose` parses).
    pub fn to_input(self, dex: &Dex) -> String {
        match self {
            SearchChoice::Team(slots) => {
                let parts: Vec<String> = slots
                    .iter()
                    .filter(|&&s| s != 0)
                    .map(|s| s.to_string())
                    .collect();
                format!("team {}", parts.join(", "))
            }
            SearchChoice::Move(id) => format!("move {}", dex.moves.key(id)),
            SearchChoice::Switch(pos) => format!("switch {pos}"),
            SearchChoice::Pass => "pass".to_string(),
        }
    }
}

/// Terminal battle result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    P1Win,
    P2Win,
    Tie,
}

impl Battle {
    /// Which sides must submit a choice before the battle advances.
    /// (`[false, false]` means the battle has ended.)
    pub fn needs_choice(&self) -> [bool; 2] {
        [0, 1].map(|n| !self.ended && self.sides[n].request_state().is_some())
    }

    /// Terminal result, `None` while the battle is still running.
    pub fn outcome(&self) -> Option<Outcome> {
        if !self.ended {
            return None;
        }
        match self.winner.as_deref() {
            Some(name) if !name.is_empty() => {
                if name == self.sides[0].name {
                    Some(Outcome::P1Win)
                } else {
                    Some(Outcome::P2Win)
                }
            }
            _ => Some(Outcome::Tie),
        }
    }

    /// Replace the PRNG (determinized playouts: clone the battle, reseed the
    /// clone). The seed is the raw 64-bit LCG state.
    pub fn reseed(&mut self, seed: u64) {
        self.prng = Prng::new(seed);
    }

    /// Hash of the decision-relevant battle state — everything the game's
    /// future depends on EXCEPT the PRNG (search resamples chance, so two
    /// sims that differ only in seed are the same decision point). Used by
    /// state-keyed search trees (M7): equal keys ⇒ same state ⇒ per-node
    /// statistics and cached legal-action sets stay valid; a 64-bit collision
    /// only mis-aggregates one node's statistics, never produces an illegal
    /// choice submission path (choices are still validated by `choose`).
    ///
    /// Skipped on purpose: `prng`, the protocol log and its bookkeeping
    /// (`log`, `log_enabled`, `sent_log_pos`, `last_move_line`), scratch
    /// buffers (`listener_pool`), and the derived `battle_mask` (a function
    /// of hashed state). Mid-event machinery is quiescent at request points;
    /// its occupancy is hashed as a cheap guard.
    pub fn state_key(&self) -> u64 {
        self.state_key_with(None)
    }

    /// `state_key` with search-tree abstraction: HP is hashed as one of
    /// `buckets` maxhp-relative buckets and pure roll-magnitude bookkeeping
    /// (`last_damage`, `attacked_by` damages, `hurt_this_turn`) in coarse
    /// steps, so chance outcomes that differ only in damage-roll detail map
    /// to the same node. Discrete chance (KOs, status procs, request kinds,
    /// volatile durations) still splits exactly — that is the open-loop
    /// aliasing M7 exists to fix. The bucketing trades a bounded value
    /// blur inside a node for sample density below it.
    pub fn state_key_bucketed(&self, buckets: i64) -> u64 {
        self.state_key_with(Some(buckets))
    }

    fn state_key_with(&self, hp_buckets: Option<i64>) -> u64 {
        use std::hash::{Hash, Hasher};
        // Total destructuring on purpose: adding a `Battle` field breaks
        // this fn until the field is placed (hashed or explicitly skipped).
        let Battle {
            prng: _,           // chance is resampled by search
            turn,
            request_state,
            mid_turn,
            started,
            ended,
            winner,
            field,
            sides,
            queue,
            faint_queue,
            log: _,            // protocol log + bookkeeping: not state
            log_enabled: _,
            effect_order,
            event_depth,
            last_move_line: _, // log bookkeeping
            last_successful_move_this_turn,
            last_damage,
            quick_claw_roll,
            speed_order,
            format_data,
            sent_log_pos: _,   // log bookkeeping
            event_stack,
            effect_stack,
            active_move,
            active_pokemon,
            active_target,
            last_move_id,
            pending_boosts,
            listener_pool: _,  // scratch buffers
            battle_mask: _,    // derived from hashed state
        } = self;
        let mut h = FxHasher::default();
        (turn, request_state, mid_turn, started, ended, winner).hash(&mut h);
        field.hash(&mut h);
        for side in sides.iter() {
            side.hash_with(&mut h, hp_buckets);
        }
        (queue, faint_queue, effect_order, event_depth).hash(&mut h);
        last_successful_move_this_turn.hash(&mut h);
        match hp_buckets {
            None => last_damage.hash(&mut h),
            Some(_) => (last_damage / 16).hash(&mut h),
        }
        (quick_claw_roll, speed_order, format_data).hash(&mut h);
        // mid-event machinery: quiescent at request points, hashed as a guard
        (event_stack.len(), effect_stack.len(), active_move.is_some()).hash(&mut h);
        (active_pokemon, active_target, last_move_id, pending_boosts).hash(&mut h);
        h.finish()
    }

    /// All legal choices for one side at the current request point. Empty ⇔
    /// the side has nothing to submit (waiting side, or battle ended).
    ///
    /// Needs `&mut self` because locked-move detection runs the (side-effect
    /// free, PRNG-free) `LockMove`/`SemiLockMove` priority events, exactly
    /// like PS request generation does.
    pub fn legal_choices(&mut self, dex: &Dex, side_n: usize) -> Vec<SearchChoice> {
        if self.ended {
            return Vec::new();
        }
        let Some(request) = self.sides[side_n].request_state() else {
            return Vec::new();
        };
        match request {
            RequestKind::TeamPreview => self.legal_team_choices(side_n),
            RequestKind::Move => self.legal_move_choices(dex, side_n),
            RequestKind::Switch => self.legal_forced_switch_choices(side_n),
            RequestKind::Wait => Vec::new(),
        }
    }

    /// All ordered picks of `picked_team_size` distinct display positions
    /// whose level sum respects Max Total Level (mirrors `choose_team`;
    /// certificate on `MAX_TOTAL_LEVEL` in choices.rs). Never empty for a
    /// validator-legal team — the validator guarantees a legal triple.
    fn legal_team_choices(&self, side_n: usize) -> Vec<SearchChoice> {
        let n = self.sides[side_n].party.len() as u8;
        let k = self.picked_team_size(side_n).min(3) as u8;
        let mut out = Vec::new();
        let push = |out: &mut Vec<SearchChoice>, slots: [u8; 3]| {
            let positions: Vec<usize> =
                slots.iter().filter(|&&s| s != 0).map(|&s| s as usize - 1).collect();
            if self.picked_total_level(side_n, &positions) <= super::choices::MAX_TOTAL_LEVEL {
                out.push(SearchChoice::Team(slots));
            }
        };
        for a in 1..=n {
            if k <= 1 {
                push(&mut out, [a, 0, 0]);
                continue;
            }
            for b in 1..=n {
                if b == a {
                    continue;
                }
                if k == 2 {
                    push(&mut out, [a, b, 0]);
                    continue;
                }
                for c in 1..=n {
                    if c != a && c != b {
                        push(&mut out, [a, b, c]);
                    }
                }
            }
        }
        out
    }

    /// Mirrors `choose_move` + `choose_switch` legality on a move request.
    fn legal_move_choices(&mut self, dex: &Dex, side_n: usize) -> Vec<SearchChoice> {
        let Some(active) = self.active_id(side_n) else {
            return vec![SearchChoice::Pass];
        };
        if self.poke(active).fainted {
            // only reachable defensively; PS would have made a switch request
            return vec![SearchChoice::Pass];
        }

        // locked moves preempt everything (thrash/rollout/recharge/...)
        let locked = self
            .get_locked_move(dex, active)
            .or_else(|| self.get_semi_locked_move(dex, active));
        if let Some(locked_id) = locked {
            let id = dex
                .moves
                .id(&locked_id)
                .or_else(|| dex.moves.id("recharge"))
                .expect("locked move id interned");
            return vec![SearchChoice::Move(id)];
        }

        let mut out = Vec::new();
        let moves = self.pokemon_choosable_moves(active);
        if moves.is_empty() {
            out.push(SearchChoice::Move(dex.moves.id("struggle").unwrap()));
        } else {
            for (id, disabled) in moves {
                let choice = SearchChoice::Move(id);
                if !disabled && !out.contains(&choice) {
                    out.push(choice);
                }
            }
        }

        // voluntary switches
        if !self.poke(active).trapped {
            let side = &self.sides[side_n];
            for pos in 1..side.party.len() {
                let slot = side.party[pos];
                if !side.roster[slot as usize].fainted {
                    out.push(SearchChoice::Switch(pos as u8 + 1));
                }
            }
        }
        out
    }

    /// Mirrors `choose_switch`/`choose_pass` legality on a forced switch.
    fn legal_forced_switch_choices(&self, side_n: usize) -> Vec<SearchChoice> {
        let side = &self.sides[side_n];
        let mut out = Vec::new();
        if side.choice.forced_switches_left > 0 {
            for pos in 1..side.party.len() {
                let slot = side.party[pos];
                if !side.roster[slot as usize].fainted
                    && !side.choice.switch_ins.contains(&(pos as u8))
                {
                    out.push(SearchChoice::Switch(pos as u8 + 1));
                }
            }
        }
        if out.is_empty() && side.choice.forced_passes_left > 0 {
            out.push(SearchChoice::Pass);
        }
        out
    }

    /// Submit one side's choice (advances the battle when it was the last
    /// side owing one).
    pub fn apply_choice(
        &mut self,
        dex: &Dex,
        side_n: usize,
        choice: SearchChoice,
    ) -> Result<(), EngineError> {
        let input = choice.to_input(dex);
        self.choose(dex, side_n, &input)
    }

    /// Submit both sides' choices for this decision point (`None` for a side
    /// that owes none). Afterwards the battle has advanced to the next
    /// request point or ended.
    pub fn apply_choices(
        &mut self,
        dex: &Dex,
        choices: [Option<SearchChoice>; 2],
    ) -> Result<(), EngineError> {
        for (side_n, choice) in choices.into_iter().enumerate() {
            if let Some(c) = choice {
                self.apply_choice(dex, side_n, c)?;
            }
        }
        Ok(())
    }
}
