//! Drives one battle between two agents through the search API.

use nc2000_engine::battle::{EngineError, Outcome};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;

use crate::agent::Agent;

/// `TurnCapped` = neither side won within `max_turns` (scored as a tie by
/// callers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Outcome(Outcome),
    TurnCapped,
}

pub fn play_game(
    dex: &Dex,
    battle: &mut Battle,
    agents: &mut [&mut dyn Agent; 2],
    max_turns: u16,
) -> Result<GameResult, EngineError> {
    loop {
        if let Some(o) = battle.outcome() {
            return Ok(GameResult::Outcome(o));
        }
        if battle.turn > max_turns {
            return Ok(GameResult::TurnCapped);
        }
        // Both sides pick against the same state (PS request semantics),
        // then the choices are applied together.
        let mut picks = [None, None];
        for s in 0..2 {
            let cs = battle.legal_choices(dex, s);
            if !cs.is_empty() {
                picks[s] = Some(agents[s].choose(battle, dex, s, &cs));
            }
        }
        if picks == [None, None] {
            return Err(EngineError::InvalidChoice(
                "no side owes a choice but the battle has not ended".into(),
            ));
        }
        battle.apply_choices(dex, picks)?;
    }
}
