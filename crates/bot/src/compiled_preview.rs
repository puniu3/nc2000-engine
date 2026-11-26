//! Identity for precomputed live-policy previews. Opponent fields are public
//! species and levels only on PS Stadium 2; on Castle PBS they additionally
//! include seat (`side`), opponent genders, `preview_has_item` flags, and
//! ruleset / turn-limit / overflow metadata. Own fields preserve order, moves,
//! DVs, stats and happiness; the old order-insensitive team signature is
//! insufficient here.
use nc2000_engine::rules::Ruleset;
use nc2000_engine::state::{Battle, RequestState};

pub fn key(battle: &Battle, side: usize) -> Option<String> {
    if side > 1 || battle.request_state != RequestState::TeamPreview {
        return None;
    }
    if battle.is_castle() {
        return castle_key(battle, side, None);
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

/// Castle-aware preview cache key including `side`, opponent genders,
/// `preview_has_item` flags (read from `preview_has_item` override when provided,
/// otherwise `p.item.is_some()`), `ruleset.canonical_id()`, `turn_limit`, and
/// `overflow_mode`.
pub fn castle_key(
    battle: &Battle,
    side: usize,
    preview_has_item: Option<&[bool]>,
) -> Option<String> {
    if side > 1 || battle.request_state != RequestState::TeamPreview {
        return None;
    }
    let own: Vec<_> = battle.sides[side].roster.iter().map(|p| serde_json::json!([
        p.species.0, p.level, format!("{:?}", p.gender), p.happiness,
        p.set_ivs, p.set_evs, p.base_stored_stats, p.base_maxhp,
        p.item.map(|x| x.0), p.base_hp_type.0, p.base_hp_power,
        p.base_move_slots.iter().map(|m| (m.id.0, m.maxpp)).collect::<Vec<_>>()
    ])).collect();
    let foe: Vec<_> = battle.sides[1 - side]
        .roster
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            let has_item = preview_has_item
                .and_then(|flags| flags.get(idx).copied())
                .unwrap_or(p.item.is_some());
            (p.species.0, p.level, format!("{:?}", p.gender), has_item)
        })
        .collect();
    let meta = serde_json::json!({
        "epoch": policy_epoch_for(battle.ruleset),
        "side": side,
        "ruleset": battle.ruleset.canonical_id(),
        "turn_limit": battle.ruleset.turn_limit(),
        "overflow_mode": battle.ruleset.overflow_mode(),
    });
    Some(serde_json::json!([meta, own, foe]).to_string())
}

/// Convenience helper for blind Castle callers passing observed public `preview_has_item` flags
/// rather than reading opponent held items from a full-info `Battle`.
pub fn castle_key_from_flags(
    battle: &Battle,
    side: usize,
    foe_has_item: &[bool],
) -> Option<String> {
    castle_key(battle, side, Some(foe_has_item))
}

/// Bump manually ONLY when PS team-preview / search policy intentionally changes.
pub const POLICY_EPOCH: &str = "policy-v4";

/// Bump manually ONLY when Castle PBS team-preview / search policy intentionally changes.
pub const CASTLE_POLICY_EPOCH: &str = "castle-policy-v4";

#[inline]
pub const fn policy_epoch_for(ruleset: Ruleset) -> &'static str {
    if ruleset.is_castle() {
        CASTLE_POLICY_EPOCH
    } else {
        POLICY_EPOCH
    }
}

pub fn build_fingerprint() -> &'static str {
    POLICY_EPOCH
}
