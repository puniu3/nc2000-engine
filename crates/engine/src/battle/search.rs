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

    /// All ordered picks of `picked_team_size` distinct display positions.
    fn legal_team_choices(&self, side_n: usize) -> Vec<SearchChoice> {
        let n = self.sides[side_n].party.len() as u8;
        let k = self.picked_team_size(side_n).min(3) as u8;
        let mut out = Vec::new();
        for a in 1..=n {
            if k <= 1 {
                out.push(SearchChoice::Team([a, 0, 0]));
                continue;
            }
            for b in 1..=n {
                if b == a {
                    continue;
                }
                if k == 2 {
                    out.push(SearchChoice::Team([a, b, 0]));
                    continue;
                }
                for c in 1..=n {
                    if c != a && c != b {
                        out.push(SearchChoice::Team([a, b, c]));
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
