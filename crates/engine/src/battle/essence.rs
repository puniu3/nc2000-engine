//! Snapshot "essence" extraction — must produce byte-identical JSON structure
//! to tools/gen-fixtures.js `essence()` (scalar keys only, effectOrder
//! excluded, sourceSlot/name/duration included when present).

use crate::dex::Dex;
use crate::state::*;
use serde_json::{json, Map, Value};

fn eff_id_str<'d>(dex: &'d Dex, id: EffId) -> &'d str {
    match id {
        EffId::None => "",
        EffId::Cond(c) => dex.conds_key(c),
        EffId::Item(i) => dex.items.key(i),
        EffId::Format => "gen2nintendocup2000noohkostadium2strict",
    }
}

fn scalar_json(dex: &Dex, v: &Scalar) -> Value {
    match v {
        Scalar::Int(i) => json!(i),
        Scalar::Float(f) => json!(f),
        Scalar::Bool(b) => json!(b),
        Scalar::MoveK(m) => json!(dex.moves.key(*m)),
        Scalar::CondK(c) => json!(dex.conds_key(*c)),
        Scalar::Slot(side, pos) => json!(format!("p{}{}", side + 1, (b'a' + pos) as char)),
    }
}

fn scal(dex: &Dex, state: &EffectState) -> Value {
    let mut out = Map::new();
    out.insert("id".into(), Value::String(eff_id_str(dex, state.id).to_string()));
    if state.has_name {
        // addVolatile stores the condition's display name.
        let name = match state.id {
            EffId::Cond(c) => dex.cond_display_name(c).to_string(),
            other => eff_id_str(dex, other).to_string(),
        };
        out.insert("name".into(), Value::String(name));
    }
    // PS key order: id, name (addVolatile), source*, duration, then data.
    // Key order doesn't matter for comparison (structural), but keep sane.
    if let Some((side, pos)) = state.source_slot {
        out.insert(
            "sourceSlot".into(),
            Value::String(format!("p{}{}", side + 1, (b'a' + pos) as char)),
        );
    }
    if let Some(d) = state.duration {
        out.insert("duration".into(), json!(d));
    }
    for (k, v) in state.data.iter() {
        out.insert(k.as_str().to_string(), scalar_json(dex, v));
    }
    Value::Object(out)
}

fn map_scal<'a>(
    dex: &Dex,
    entries: impl Iterator<Item = (&'a crate::dex::CondId, &'a EffectState)>,
) -> Value {
    let mut out = Map::new();
    for (cond, state) in entries {
        out.insert(dex.conds_key(*cond).to_string(), scal(dex, state));
    }
    Value::Object(out)
}

impl Battle {
    /// The essence snapshot (compare against fixture snapshots).
    pub fn essence(&self, dex: &Dex) -> Value {
        let sides: Vec<Value> = (0..2)
            .map(|side_n| {
                let side = &self.sides[side_n];
                let active: Vec<Value> = vec![match self.active_id(side_n) {
                    Some(a) => Value::String(self.fullname(a)),
                    None => Value::Null,
                }];
                let pokemon: Vec<Value> = side
                    .party
                    .iter()
                    .map(|&slot| {
                        let id = PokeId { side: side_n as u8, slot };
                        self.pokemon_essence(dex, id)
                    })
                    .collect();
                json!({
                    "pokemonLeft": side.pokemon_left,
                    "sideConditions": map_scal(dex, side.side_conditions.iter().map(|(c, s)| (c, s))),
                    "active": active,
                    "pokemon": pokemon,
                })
            })
            .collect();
        json!({
            "turn": self.turn,
            "prngSeed": self.prng.seed_str(),
            "requestState": self.request_state.as_str(),
            "field": {
                "weather": self.field.weather.map(|w| dex.conds_key(w)).unwrap_or(""),
                "weatherState": scal(dex, &self.field.weather_state),
                "pseudoWeather": map_scal(dex, self.field.pseudo_weather.iter().map(|(c, s)| (c, s))),
            },
            "sides": sides,
        })
    }

    fn pokemon_essence(&self, dex: &Dex, id: PokeId) -> Value {
        let p = self.poke(id);
        let moves: Vec<Value> = p
            .move_slots
            .iter()
            .map(|m| {
                json!({
                    "id": dex.moves.key(m.id),
                    "pp": m.pp,
                    "disabled": m.disabled,
                })
            })
            .collect();
        json!({
            "ident": self.fullname(id),
            "species": dex.species.key(p.species),
            "hp": p.hp,
            "maxhp": p.maxhp,
            "fainted": p.fainted,
            "status": p.status.as_str(),
            "statusState": scal(dex, &p.status_state),
            "boosts": {
                "atk": p.boosts[0],
                "def": p.boosts[1],
                "spa": p.boosts[2],
                "spd": p.boosts[3],
                "spe": p.boosts[4],
                "accuracy": p.boosts[5],
                "evasion": p.boosts[6],
            },
            "item": p.item.map(|i| dex.items.key(i).to_string()).unwrap_or_default(),
            "lastItem": p.last_item.map(|i| dex.items.key(i).to_string()).unwrap_or_default(),
            "itemState": scal(dex, &p.item_state),
            "moves": moves,
            "volatiles": map_scal(dex, p.volatiles.iter().map(|(c, s)| (c, s))),
            "types": p.types.iter().map(|t| dex.type_name(t)).collect::<Vec<_>>(),
            "transformed": p.transformed,
            "active": p.is_active,
            "position": p.position,
        })
    }
}
