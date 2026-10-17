use nc2000_engine::battle::SearchChoice;
use nc2000_engine::dex::{Dex, ItemId, MoveId, SpeciesId};
use nc2000_engine::state::{RequestState, Status};
use serde::{Deserialize, Serialize};

use crate::import::ProtocolAgent;
use crate::observe::Observer;
use crate::player::action_input;

pub const OBSERVATION_SCHEMA: &str = "nc2000-observation-v1";
pub const MON_FEATURES: usize = 72;
pub const GLOBAL_FEATURES: usize = 24;
pub const ACTION_FEATURES: usize = 17;

const VOLATILES: [&str; 16] = [
    "confusion",
    "substitute",
    "trapped",
    "partiallytrapped",
    "perishsong",
    "encore",
    "disable",
    "leechseed",
    "attract",
    "focusenergy",
    "curse",
    "nightmare",
    "protect",
    "mustrecharge",
    "rollout",
    "lockedmove",
];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningMon {
    pub species: usize,
    pub item: usize,
    pub moves: [usize; 4],
    pub features: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningAction {
    pub input: String,
    pub eligible: bool,
    pub move_id: usize,
    pub features: Vec<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningObservation {
    pub schema: String,
    pub global: Vec<f32>,
    pub mons: Vec<LearningMon>,
    pub actions: Vec<LearningAction>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Vocabulary {
    pub species: Vec<String>,
    pub moves: Vec<String>,
    pub items: Vec<String>,
}

impl Vocabulary {
    pub fn from_dex(dex: &Dex) -> Self {
        Self {
            species: std::iter::once(String::new())
                .chain(
                    (0..dex.species.len())
                        .map(|i| dex.species.key(SpeciesId(i as u16)).to_string()),
                )
                .collect(),
            moves: std::iter::once(String::new())
                .chain((0..dex.moves.len()).map(|i| dex.moves.key(MoveId(i as u16)).to_string()))
                .collect(),
            items: std::iter::once(String::new())
                .chain((0..dex.items.len()).map(|i| dex.items.key(ItemId(i as u16)).to_string()))
                .collect(),
        }
    }
}

pub fn observation(agent: &ProtocolAgent, dex: &Dex) -> Result<LearningObservation, String> {
    let b = agent.battle().ok_or("observation before request")?;
    let obs = agent.observer().ok_or("observation before observer")?;
    let me = agent.side();
    let search = agent.search().ok_or("observation without search")?;
    let belief = agent.belief().ok_or("observation without belief")?;
    Ok(state_observation(
        b,
        dex,
        obs,
        me,
        belief.is_fallback(),
        belief.candidate_count(),
        search.actions(),
        search.dominated(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn state_observation(
    b: &nc2000_engine::state::Battle,
    dex: &Dex,
    obs: &Observer,
    me: usize,
    fallback: bool,
    candidate_count: usize,
    choices: &[SearchChoice],
    dominated: &[bool],
) -> LearningObservation {
    assert_eq!(choices.len(), dominated.len());
    let mut global = vec![0.0; GLOBAL_FEATURES];
    global[0] = b.turn as f32 / 1000.0;
    global[1] = f32::from(b.request_state == RequestState::TeamPreview);
    global[2] = f32::from(b.request_state == RequestState::Switch);
    global[3] = f32::from(b.request_state == RequestState::Move);
    global[4] = b.sides[me].pokemon_left as f32 / 6.0;
    global[5] = b.sides[1 - me].pokemon_left as f32 / 6.0;
    for (i, name) in ["raindance", "sunnyday", "sandstorm"].iter().enumerate() {
        global[6 + i] = f32::from(b.field.weather.is_some_and(|id| dex.conds.key(id) == *name));
    }
    for relative in 0..2 {
        let side = me ^ relative;
        for (i, key) in ["spikes", "reflect", "lightscreen", "safeguard", "mist"]
            .iter()
            .enumerate()
        {
            global[9 + relative * 5 + i] = f32::from(
                dex.conds_id(key)
                    .is_some_and(|id| b.sides[side].side_conditions.iter().any(|(c, _)| *c == id)),
            );
        }
    }
    global[19] = f32::from(fallback);
    global[20] = (candidate_count as f32).ln_1p() / 5.0;
    let mut mons = Vec::with_capacity(12);
    for relative in 0..2 {
        let side = me ^ relative;
        for slot in 0..6 {
            let Some(p) = b.sides[side].roster.get(slot) else {
                mons.push(LearningMon {
                    species: 0,
                    item: 0,
                    moves: [0; 4],
                    features: vec![0.0; MON_FEATURES],
                });
                continue;
            };
            let knowledge = if relative == 1 {
                obs.mons().get(slot)
            } else {
                None
            };
            let appeared = knowledge.map_or(p.previously_switched_in > 0, |k| k.appeared);
            let selected = if relative == 0 {
                b.sides[side].party.contains(&(slot as u8))
            } else {
                appeared
            };
            let mut f = vec![
                1.0,
                p.hp as f32 / p.maxhp.max(1) as f32,
                p.maxhp as f32 / 500.0,
                p.level as f32 / 100.0,
                f32::from(appeared),
                f32::from(selected),
                f32::from(p.is_active),
                f32::from(p.fainted),
            ];
            f.extend(p.stored_stats.iter().map(|&x| x as f32 / 500.0));
            for i in 0..18 {
                f.push(f32::from(p.types.iter().any(|t| t.0 as usize == i)));
            }
            for status in [
                Status::None,
                Status::Brn,
                Status::Par,
                Status::Slp,
                Status::Frz,
                Status::Psn,
                Status::Tox,
            ] {
                f.push(f32::from(p.status == status));
            }
            f.extend(p.boosts.iter().map(|&x| x as f32 / 6.0));
            let mut moves = [0; 4];
            let mut pp = [0.0; 4];
            let mut revealed = [0.0; 4];
            for (i, m) in p.move_slots.iter().enumerate() {
                moves[i] = m.id.0 as usize + 1;
                pp[i] = m.pp as f32 / m.maxpp.max(1) as f32;
                revealed[i] = f32::from(
                    relative == 0 || knowledge.is_some_and(|k| k.revealed_moves.contains(&m.id)),
                );
            }
            f.extend(pp);
            for key in VOLATILES {
                f.push(f32::from(
                    dex.conds_id(key).is_some_and(|id| p.has_volatile(id)),
                ));
            }
            f.extend(revealed);
            f.resize(MON_FEATURES, 0.0);
            mons.push(LearningMon {
                species: p.species.0 as usize + 1,
                item: p.item.map_or(0, |id| id.0 as usize + 1),
                moves,
                features: f,
            });
        }
    }
    let any_eligible = dominated.iter().any(|d| !d);
    let actions = choices
        .iter()
        .enumerate()
        .map(|(index, &choice)| {
            let mut f = vec![0.0; ACTION_FEATURES];
            let mut move_id = 0;
            match choice {
                SearchChoice::Move(id) => {
                    f[0] = 1.0;
                    move_id = id.0 as usize + 1;
                    if let (Some(att), Some(def)) = (b.active_id(me), b.active_id(1 - me)) {
                        f[15] = crate::eval::expected_hit_fraction(b, dex, att, def, id, true)
                            .min(4.0) as f32;
                    }
                    f[16] = dex.move_static(id).priority as f32 / 6.0;
                }
                SearchChoice::Switch(pos) => {
                    f[1] = 1.0;
                    if let Some(&slot) = b.sides[me].party.get(pos as usize - 1) {
                        f[3 + slot as usize] = 1.0;
                    }
                }
                SearchChoice::Team(slots) => {
                    f[2] = 1.0;
                    for (i, pos) in slots.iter().copied().filter(|p| *p > 0).enumerate() {
                        if let Some(&slot) = b.sides[me].party.get(pos as usize - 1) {
                            f[3 + slot as usize] = 1.0;
                            if i == 0 {
                                f[9 + slot as usize] = 1.0;
                            }
                        }
                    }
                }
                SearchChoice::Pass => {}
            }
            LearningAction {
                input: action_input(dex, choice),
                eligible: !any_eligible || !dominated[index],
                move_id,
                features: f,
            }
        })
        .collect();
    LearningObservation {
        schema: OBSERVATION_SCHEMA.into(),
        global,
        mons,
        actions,
    }
}
