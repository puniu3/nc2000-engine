//! Battle driver — the part that is still TO BE PORTED.
//!
//! Architecture decisions (fixed by the project plan, do not re-litigate):
//! - No dynamic event broadcast: the 76 data-facing hooks (`events::Handler`)
//!   dispatch through enum + match; effect behavior is ported function by
//!   function from PS (`PORTING.md` is the checklist).
//! - PRNG consumption order must match PS exactly (speed-tie shuffle first,
//!   then per-action rolls in PS's order) — snapshot parity asserts this via
//!   `prng_seed` at every snapshot point.
//! - Snapshot points = after every input line that produced log output
//!   (see conformance::fixture docs).
//!
//! Porting order (milestone 1): team init + stat calc (gen2 DV/stat-exp
//! formulas from data/mods/gen2/scripts.ts) → team preview → switch-in →
//! damage pipeline for pure-data moves (gen2 getDamage in
//! data/mods/gen2/scripts.ts:744 lineage) → residuals → the `puredata`
//! corpus goes green. Then port conditions, then callback moves, then items.

use crate::choice::Choice;
use crate::dex::Dex;
use crate::state::Battle;

#[derive(Debug)]
pub enum EngineError {
    /// The engine has not been ported far enough to run this battle.
    Unimplemented(&'static str),
    InvalidChoice(String),
}

/// A player's team as delivered by the fixture (canonical validated sets).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PokemonSet {
    pub name: String,
    pub species: String,
    #[serde(default)]
    pub item: String,
    #[serde(default)]
    pub ability: String,
    pub moves: Vec<String>,
    pub level: u8,
    #[serde(default)]
    pub evs: Option<std::collections::BTreeMap<String, u16>>,
    #[serde(default)]
    pub ivs: Option<std::collections::BTreeMap<String, u8>>,
    #[serde(default)]
    pub happiness: Option<u8>,
    #[serde(default)]
    pub gender: Option<String>,
}

impl Battle {
    /// Constructs a battle in team-preview state from a fixture's seed and
    /// canonical teams. Mirrors PS `new Battle({formatid, seed}) + setPlayer`.
    pub fn from_fixture(
        _dex: &Dex,
        _seed: &str,
        _p1: &[PokemonSet],
        _p2: &[PokemonSet],
    ) -> Result<Battle, EngineError> {
        Err(EngineError::Unimplemented("Battle::from_fixture: milestone 1"))
    }

    /// Applies one side's canonical choice line (fixture `choices[i]`).
    /// Returns true if the choice completed a commit (log grew — snapshot
    /// point for conformance comparison).
    pub fn choose(&mut self, _side: u8, _choice: &[Choice]) -> Result<bool, EngineError> {
        Err(EngineError::Unimplemented("Battle::choose: milestone 1"))
    }
}
