//! Parser for the canonical choice strings PS writes into its inputLog
//! (side.ts getChoice() output — what our fixtures replay).
//!
//! Observed grammar in NC2000 fixtures:
//!   `team 5, 6, 1`      (team preview pick, 1-based slots)
//!   `move thunderbolt`  (by move id) / `move 2` (by 1-based slot)
//!   `switch 3`          (1-based slot)
//!   `pass`, `default`, `undo`

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Choice {
    Team(Vec<u8>),
    MoveById(String),
    MoveBySlot(u8),
    Switch(u8),
    Pass,
    Default,
    Undo,
}

impl Choice {
    pub fn parse(input: &str) -> Result<Vec<Choice>, String> {
        // A side's full choice can be comma-separated per active slot, but
        // `team 5, 6, 1` is a single team choice containing commas.
        let input = input.trim();
        if let Some(rest) = input.strip_prefix("team ") {
            let slots: Result<Vec<u8>, _> = rest
                .split(',')
                .map(|p| p.trim().parse::<u8>().map_err(|e| e.to_string()))
                .collect();
            return Ok(vec![Choice::Team(slots?)]);
        }
        input.split(',').map(|part| Choice::parse_one(part.trim())).collect()
    }

    fn parse_one(part: &str) -> Result<Choice, String> {
        if part == "pass" {
            return Ok(Choice::Pass);
        }
        if part == "default" {
            return Ok(Choice::Default);
        }
        if part == "undo" {
            return Ok(Choice::Undo);
        }
        if let Some(rest) = part.strip_prefix("move ") {
            let rest = rest.trim();
            return Ok(match rest.parse::<u8>() {
                Ok(slot) => Choice::MoveBySlot(slot),
                Err(_) => Choice::MoveById(rest.to_string()),
            });
        }
        if let Some(rest) = part.strip_prefix("switch ") {
            return rest
                .trim()
                .parse::<u8>()
                .map(Choice::Switch)
                .map_err(|e| format!("bad switch slot: {e}"));
        }
        Err(format!("unrecognized choice: {part:?}"))
    }
}
