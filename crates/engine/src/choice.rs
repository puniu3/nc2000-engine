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
            let slots: Result<Vec<u8>, String> = rest
                .split(',')
                .map(|p| {
                    let s = p.trim().parse::<u8>().map_err(|e| e.to_string())?;
                    if s == 0 {
                        return Err("team slot cannot be 0".to_string());
                    }
                    Ok(s)
                })
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
            return match rest.parse::<u8>() {
                Ok(0) => Err("move slot cannot be 0".to_string()),
                Ok(slot) => Ok(Choice::MoveBySlot(slot)),
                Err(_) => Ok(Choice::MoveById(rest.to_string())),
            };
        }
        if let Some(rest) = part.strip_prefix("switch ") {
            let slot = rest
                .trim()
                .parse::<u8>()
                .map_err(|e| format!("bad switch slot: {e}"))?;
            if slot == 0 {
                return Err("switch slot cannot be 0".to_string());
            }
            return Ok(Choice::Switch(slot));
        }
        Err(format!("unrecognized choice: {part:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::Choice;

    #[test]
    fn zero_slots_are_rejected() {
        assert!(Choice::parse("move 0").is_err());
        assert!(Choice::parse("switch 0").is_err());
        assert!(Choice::parse("team 0, 1, 2").is_err());
        assert!(Choice::parse("team 1, 0, 2").is_err());
        assert_eq!(Choice::parse("move 1").unwrap(), vec![Choice::MoveBySlot(1)]);
        assert_eq!(Choice::parse("switch 2").unwrap(), vec![Choice::Switch(2)]);
    }
}
