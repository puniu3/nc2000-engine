//! Identity for precomputed live-policy previews. Opponent fields are public
//! species and levels only. Own fields preserve order, moves, DVs, stats and
//! happiness; the old order-insensitive team signature is insufficient here.
use nc2000_engine::{state::{Battle, RequestState}};

pub fn key(battle: &Battle, side: usize) -> Option<String> {
    if side > 1 || battle.request_state != RequestState::TeamPreview {
        return None;
    }
    let own: Vec<_> = battle.sides[side].roster.iter().map(|p| serde_json::json!([
        p.species.0, p.level, format!("{:?}", p.gender), p.happiness,
        p.set_ivs, p.set_evs, p.base_stored_stats, p.base_maxhp,
        p.item.map(|x| x.0), p.base_hp_type.0, p.base_hp_power,
        p.base_move_slots.iter().map(|m| (m.id.0, m.maxpp)).collect::<Vec<_>>()
    ])).collect();
    let foe: Vec<_> = battle.sides[1-side].roster.iter().map(|p| (p.species.0, p.level)).collect();
    Some(serde_json::json!([own, foe]).to_string())
}

pub fn build_fingerprint() -> &'static str {
    env!("M17E_SOLVER_BUILD_FINGERPRINT")
}
