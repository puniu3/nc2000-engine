use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::Dex;
use nc2000_engine::state::{Battle, PokeId, RequestState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PlayerFrame {
    pub lines: Vec<String>,
    pub request: Value,
    pub legal_actions: Vec<String>,
}

pub struct PlayerChannel {
    side: usize,
    log_cursor: usize,
    split: Option<(usize, bool)>,
}

impl PlayerChannel {
    pub fn new(side: usize) -> Self {
        assert!(side < 2);
        Self {
            side,
            log_cursor: 0,
            split: None,
        }
    }

    pub fn frame(&mut self, battle: &mut Battle, dex: &Dex) -> Result<PlayerFrame, String> {
        if !battle.log_enabled {
            return Err("player channel requires the public battle log".into());
        }
        if battle.log.len() < self.log_cursor {
            return Err("player channel reused across battles".into());
        }
        let mut lines = Vec::new();
        for line in &battle.log[self.log_cursor..] {
            if let Some((owner, secret)) = self.split {
                if (owner == self.side) == secret {
                    lines.push(line.clone());
                }
                self.split = secret.then_some((owner, false));
            } else if let Some(owner) = line.strip_prefix("|split|") {
                let owner = match owner {
                    "p1" => 0,
                    "p2" => 1,
                    _ => return Err(format!("invalid split owner: {owner}")),
                };
                self.split = Some((owner, true));
            } else {
                lines.push(line.clone());
            }
        }
        self.log_cursor = battle.log.len();
        let choices = battle.legal_choices(dex, self.side);
        let legal_actions = choices.iter().map(|&c| action_input(dex, c)).collect();
        let request = player_request(battle, dex, self.side, &choices);
        Ok(PlayerFrame {
            lines,
            request,
            legal_actions,
        })
    }
}

pub fn action_input(dex: &Dex, choice: SearchChoice) -> String {
    match choice {
        SearchChoice::Move(id) if dex.moves.key(id).starts_with("hiddenpower") => {
            "move hiddenpower".into()
        }
        _ => choice.to_input(dex),
    }
}

fn player_request(battle: &Battle, dex: &Dex, side: usize, choices: &[SearchChoice]) -> Value {
    let sd = &battle.sides[side];
    let pokemon: Vec<Value> = sd
        .party
        .iter()
        .map(|&slot| {
            let p = &sd.roster[slot as usize];
            let id = PokeId {
                side: side as u8,
                slot,
            };
            let species = &dex.species.get(p.base_species).name;
            let details = format!("{species}, L{}, {}", p.level, p.gender.as_str());
            let moves: Vec<&str> = p
                .base_move_slots
                .iter()
                .map(|m| dex.moves.key(m.id))
                .collect();
            json!({
                "ident": format!("p{}: {}", side + 1, p.name.as_str()),
                "details": details,
                "condition": battle.get_health(id).0,
                "active": p.is_active,
                "moves": moves,
                "item": p.item.map(|id| dex.items.key(id)).unwrap_or(""),
            })
        })
        .collect();
    let own = json!({"pokemon": pokemon});
    if choices.is_empty() {
        return json!({"wait": true, "side": own});
    }
    match battle.request_state {
        RequestState::TeamPreview => json!({
            "teamPreview": true, "maxChosenTeamSize": 3, "side": own,
        }),
        RequestState::Switch => json!({"forceSwitch": [true], "side": own}),
        _ => {
            let active = battle.active_id(side).map(|id| battle.poke(id));
            let legal_moves: Vec<_> = choices
                .iter()
                .filter_map(|&c| match c {
                    SearchChoice::Move(id) => Some(id),
                    _ => None,
                })
                .collect();
            let mut moves: Vec<Value> = active
                .map(|p| {
                    p.move_slots.iter().map(|m| {
                json!({
                    "id": action_input(dex, SearchChoice::Move(m.id)).trim_start_matches("move "),
                    "pp": m.pp,
                    "maxpp": m.maxpp,
                    "disabled": !legal_moves.contains(&m.id),
                })
            }).collect()
                })
                .unwrap_or_default();
            for &id in &legal_moves {
                let input = action_input(dex, SearchChoice::Move(id));
                let key = input.trim_start_matches("move ");
                if !moves
                    .iter()
                    .any(|m| m["id"] == key && m["disabled"] == false && m["pp"] != 0)
                {
                    moves.retain(|m| m["id"] != key);
                    moves.push(json!({"id": key, "disabled": false}));
                }
            }
            let trapped = !choices.iter().any(|c| matches!(c, SearchChoice::Switch(_)));
            json!({"active": [{"moves": moves, "trapped": trapped}], "side": own})
        }
    }
}
