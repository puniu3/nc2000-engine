use nc2000_engine::dex::{Dex, SpeciesId, TypeList};
use nc2000_engine::state::{Gender, Pokemon};

use crate::observe::MonObs;
use crate::rng::SplitMix64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effectiveness {
    Immune,
    Resisted,
    Neutral,
    SuperEffective,
}

pub(crate) fn types_with_effectiveness(
    dex: &Dex,
    defender_types: &TypeList,
    eff: Effectiveness,
) -> u16 {
    if defender_types.iter().count() == 0 {
        return u16::MAX;
    }
    let mut mask = 0u16;
    for (i, &ty_name) in TYPES.iter().enumerate() {
        let Some(att) = dex.type_id(ty_name) else {
            continue;
        };
        let actual = if defender_types.iter().any(|def| dex.type_immune(att, def)) {
            Effectiveness::Immune
        } else {
            let total_mod: i32 = defender_types.iter().map(|def| dex.eff(att, def)).sum();
            if total_mod > 0 {
                Effectiveness::SuperEffective
            } else if total_mod < 0 {
                Effectiveness::Resisted
            } else {
                Effectiveness::Neutral
            }
        };
        if actual == eff {
            mask |= 1u16 << i;
        }
    }
    mask
}

pub(crate) const TYPES: [&str; 16] = [
    "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", "Fire", "Water",
    "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
];

pub(crate) fn modeled_type(dex: &Dex, p: &Pokemon) -> Option<usize> {
    let slot = p
        .base_move_slots
        .iter()
        .find(|m| dex.moves.key(m.id).starts_with("hiddenpower"))?;
    let name = if dex.moves.key(slot.id) == "hiddenpower" {
        dex.type_name(p.base_hp_type)
    } else {
        dex.moves.get(slot.id).move_type.as_str()
    };
    TYPES.iter().position(|t| *t == name)
}

pub(crate) fn consistent(dex: &Dex, p: &Pokemon, mask: u16) -> bool {
    mask == u16::MAX || modeled_type(dex, p).is_some_and(|i| mask & (1 << i) != 0)
}

fn gender_matches(dex: &Dex, p: &Pokemon, gender: Gender, atk: i32) -> bool {
    let species = dex.species.get(p.species);
    if species.gender.is_some() || gender == Gender::N {
        return true;
    }
    let female = species
        .extra
        .get("genderRatio")
        .and_then(|r| r["F"].as_f64())
        .unwrap_or(0.5);
    (f64::from(atk) < female * 16.0) == (gender == Gender::F)
}

pub(crate) fn possible_types(dex: &Dex, species: SpeciesId, gender: Gender) -> u16 {
    let species = dex.species.get(species);
    if species.gender.is_some() || gender == Gender::N {
        return u16::MAX;
    }
    let female = species
        .extra
        .get("genderRatio")
        .and_then(|r| r["F"].as_f64())
        .unwrap_or(0.5);
    (0..16)
        .filter(|atk| (f64::from(*atk) < female * 16.0) == (gender == Gender::F))
        .fold(0, |mask, atk| mask | (15 << (4 * (atk % 4))))
}

#[inline]
pub(crate) fn calc_maxhp(base_hp: u16, level: u8, hp_dv: i32, effort_hp: i32) -> i32 {
    (2 * i32::from(base_hp) + 2 * hp_dv + effort_hp) * i32::from(level) / 100
        + i32::from(level)
        + 10
}

/// True iff Hidden Power type `index` (fixing the top 2 bits of `hp_dv`) can
/// realize `obs_hp` for `(species, level)` with some legal `(spe_dv&1, spc_dv&1)`
/// and legal HP effort level `0..=63` (§1 Astra C3: never reject an HP type just
/// because max-EV HP exceeds `obs_hp`).
pub(crate) fn hp_type_can_realize_maxhp(
    dex: &Dex,
    species: SpeciesId,
    level: u8,
    index: usize,
    obs_hp: i32,
) -> bool {
    let base_hp = dex.species.get(species).base_stats.hp;
    let hp_dv_hi = ((index as i32 / 4) & 1) * 8 + ((index as i32 % 4) & 1) * 4;
    let lo = calc_maxhp(base_hp, level, hp_dv_hi, 0);
    let hi = calc_maxhp(base_hp, level, hp_dv_hi + 3, 63);
    if obs_hp < lo || obs_hp > hi {
        return false;
    }
    for low in 0..=3 {
        let hp_dv = hp_dv_hi + low;
        if obs_hp < calc_maxhp(base_hp, level, hp_dv, 0)
            || obs_hp > calc_maxhp(base_hp, level, hp_dv, 63)
        {
            continue;
        }
        for eff in (0..=63).rev() {
            if calc_maxhp(base_hp, level, hp_dv, eff) == obs_hp {
                return true;
            }
        }
    }
    false
}

pub(crate) fn possible_types_with_maxhp(
    dex: &Dex,
    species: SpeciesId,
    level: u8,
    gender: Gender,
    observed_maxhp: Option<i32>,
) -> u16 {
    let base = possible_types(dex, species, gender);
    let Some(obs_hp) = observed_maxhp else {
        return base;
    };
    let mut mask = 0u16;
    for i in 0..16 {
        if (base & (1u16 << i)) != 0 && hp_type_can_realize_maxhp(dex, species, level, i, obs_hp) {
            mask |= 1u16 << i;
        }
    }
    if mask != 0 { mask } else { base }
}

fn find_effort_for_maxhp(base_hp: u16, level: u8, hp_dv: i32, pref_ev: i32, obs_hp: i32) -> Option<i32> {
    let pref_eff = (pref_ev / 4).clamp(0, 63);
    if calc_maxhp(base_hp, level, hp_dv, pref_eff) == obs_hp {
        return Some(pref_ev.clamp(0, 255));
    }
    for eff in (0..=63).rev() {
        if calc_maxhp(base_hp, level, hp_dv, eff) == obs_hp {
            return Some(if eff == 63 { 255 } else { eff * 4 });
        }
    }
    None
}

fn recompute_mon_stats(dex: &Dex, result: &mut Pokemon) {
    let atk = result.set_ivs[1] / 2;
    let def = result.set_ivs[2] / 2;
    let spc = result.set_ivs[3] / 2;
    let spe = result.set_ivs[5] / 2;
    result.set_ivs[0] = 2 * ((atk & 1) * 8 + (def & 1) * 4 + (spe & 1) * 2 + (spc & 1));
    result.set_ivs[4] = result.set_ivs[3];
    let hp_idx = ((atk % 4) * 4 + (def % 4)) as usize;
    result.base_hp_type = dex.type_id(TYPES[hp_idx]).unwrap();
    result.hp_type = result.base_hp_type;
    result.base_hp_power =
        (5 * ((spc >> 3) + 2 * (spe >> 3) + 4 * (def >> 3) + 8 * (atk >> 3)) + spc % 4) / 2 + 31;
    result.hp_power = result.base_hp_power;
    let stats = &dex.species.get(result.species).base_stats;
    let bases = [
        stats.hp, stats.atk, stats.def, stats.spa, stats.spd, stats.spe,
    ];
    let computed: [i32; 6] = std::array::from_fn(|i| {
        (2 * i32::from(bases[i]) + result.set_ivs[i] + result.set_evs[i] / 4) * i32::from(result.level)
            / 100
            + if i == 0 { i32::from(result.level) + 10 } else { 5 }
    });
    result.base_maxhp = computed[0];
    result.maxhp = computed[0];
    result.hp = computed[0];
    result.base_stored_stats.copy_from_slice(&computed[1..]);
    result.stored_stats = result.base_stored_stats;
}

/// Adjust a mon's HP EV (and minimally DVs if required, preserving gender and
/// any Hidden Power type) to match `obs_hp` exactly. Returns `None` if `p`
/// already has `base_maxhp == obs_hp` or if `obs_hp` is unrealizable.
pub(crate) fn adjust_maxhp(
    dex: &Dex,
    p: &Pokemon,
    gender: Gender,
    obs_hp: i32,
) -> Option<Pokemon> {
    if p.base_maxhp == obs_hp {
        return None;
    }
    let base_hp = dex.species.get(p.species).base_stats.hp;
    let cur_hp_dv = p.set_ivs[0] / 2;
    if let Some(ev_hp) = find_effort_for_maxhp(base_hp, p.level, cur_hp_dv, p.set_evs[0], obs_hp) {
        let mut result = p.clone();
        result.set_evs[0] = ev_hp;
        recompute_mon_stats(dex, &mut result);
        return Some(result);
    }
    if let Some(hp_idx) = modeled_type(dex, p) {
        return project_with_index_for_maxhp(dex, p, gender, hp_idx, Some(obs_hp));
    }
    let orig_atk = p.set_ivs[1] / 2;
    let orig_def = p.set_ivs[2] / 2;
    let orig_spc = p.set_ivs[3] / 2;
    let orig_spe = p.set_ivs[5] / 2;
    // Search DV LSB combinations (closest to existing DVs first) that preserve gender
    let mut best: Option<(i32, i32, i32, i32, i32, i32)> = None;
    for atk in (0..16).filter(|&a| gender_matches(dex, p, gender, a)) {
        let d_atk = (atk - orig_atk).abs();
        if d_atk > 1 && best.as_ref().is_some_and(|b| b.0 <= d_atk) {
            continue;
        }
        for def_bit in [orig_def & 1, (orig_def & 1) ^ 1] {
            let def = (orig_def & !1) | def_bit;
            for spe_bit in [orig_spe & 1, (orig_spe & 1) ^ 1] {
                let spe = (orig_spe & !1) | spe_bit;
                for spc_bit in [orig_spc & 1, (orig_spc & 1) ^ 1] {
                    let spc = (orig_spc & !1) | spc_bit;
                    let hp_dv = (atk & 1) * 8 + (def & 1) * 4 + (spe & 1) * 2 + (spc & 1);
                    if let Some(ev_hp) = find_effort_for_maxhp(base_hp, p.level, hp_dv, p.set_evs[0], obs_hp) {
                        let cost = d_atk + (def - orig_def).abs() + (spe - orig_spe).abs() + (spc - orig_spc).abs();
                        if best.as_ref().is_none_or(|b| cost < b.0 || (cost == b.0 && ev_hp > b.5)) {
                            best = Some((cost, atk, def, spc, spe, ev_hp));
                        }
                    }
                }
            }
        }
    }
    let (_, atk, def, spc, spe, ev_hp) = best?;
    let mut result = p.clone();
    result.set_ivs[1] = 2 * atk;
    result.set_ivs[2] = 2 * def;
    result.set_ivs[3] = 2 * spc;
    result.set_ivs[5] = 2 * spe;
    result.set_evs[0] = ev_hp;
    recompute_mon_stats(dex, &mut result);
    Some(result)
}

pub(crate) fn project(
    dex: &Dex,
    p: &Pokemon,
    mo: &MonObs,
    rng: &mut SplitMix64,
) -> Option<Pokemon> {
    let Some(cur_idx) = modeled_type(dex, p) else {
        return mo.observed_maxhp.and_then(|obs_hp| adjust_maxhp(dex, p, mo.gender, obs_hp));
    };
    let hp_type_ok = consistent(dex, p, mo.hp_type_mask);
    let maxhp_type_ok = mo
        .observed_maxhp
        .is_none_or(|obs_hp| hp_type_can_realize_maxhp(dex, p.species, p.level, cur_idx, obs_hp));
    if hp_type_ok && maxhp_type_ok {
        return mo.observed_maxhp.and_then(|obs_hp| {
            if p.base_maxhp == obs_hp {
                None
            } else {
                project_with_index_for_maxhp(dex, p, mo.gender, cur_idx, Some(obs_hp))
            }
        });
    }
    let mut choices = Vec::new();
    for i in 0..16 {
        if mo.hp_type_mask & (1 << i) != 0
            && (0..16).any(|atk| atk % 4 == i / 4 && gender_matches(dex, p, mo.gender, atk))
            && mo
                .observed_maxhp
                .is_none_or(|obs_hp| hp_type_can_realize_maxhp(dex, p.species, p.level, i as usize, obs_hp))
        {
            choices.push(i as usize);
        }
    }
    if choices.is_empty() {
        debug_assert!(
            mo.observed_maxhp.is_some(),
            "Hidden Power type contradicts public gender"
        );
        return None;
    }
    let index = *choices.get(if choices.len() > 1 {
        rng.below(choices.len())
    } else {
        0
    })?;
    project_with_index_for_maxhp(dex, p, mo.gender, index, mo.observed_maxhp)
}

#[cfg(test)]
pub(crate) fn project_deterministic(
    dex: &Dex,
    p: &Pokemon,
    gender: Gender,
    mask: u16,
) -> Option<Pokemon> {
    project_deterministic_with_maxhp(dex, p, gender, mask, None)
}

pub(crate) fn project_deterministic_with_maxhp(
    dex: &Dex,
    p: &Pokemon,
    gender: Gender,
    mask: u16,
    observed_maxhp: Option<i32>,
) -> Option<Pokemon> {
    if mask == 0 {
        return None;
    }
    let Some(cur_idx) = modeled_type(dex, p) else {
        return observed_maxhp.and_then(|obs_hp| adjust_maxhp(dex, p, gender, obs_hp));
    };
    let hp_type_ok = consistent(dex, p, mask);
    let maxhp_type_ok = observed_maxhp
        .is_none_or(|obs_hp| hp_type_can_realize_maxhp(dex, p.species, p.level, cur_idx, obs_hp));
    if hp_type_ok && maxhp_type_ok {
        return observed_maxhp.and_then(|obs_hp| {
            if p.base_maxhp == obs_hp {
                None
            } else {
                project_with_index_for_maxhp(dex, p, gender, cur_idx, Some(obs_hp))
            }
        });
    }
    let legal = mask & possible_types_with_maxhp(dex, p.species, p.level, gender, observed_maxhp);
    let eff_mask = if legal != 0 {
        legal
    } else if observed_maxhp.is_some() {
        return None;
    } else {
        mask
    };
    let index = eff_mask.trailing_zeros() as usize;
    if index >= TYPES.len() {
        return None;
    }
    project_with_index_for_maxhp(dex, p, gender, index, observed_maxhp)
}

fn project_with_index_for_maxhp(
    dex: &Dex,
    p: &Pokemon,
    gender: Gender,
    index: usize,
    observed_maxhp: Option<i32>,
) -> Option<Pokemon> {
    let mut result = p.clone();
    let atk = (0..16)
        .filter(|a| a % 4 == index as i32 / 4 && gender_matches(dex, p, gender, *a))
        .min_by_key(|a| ((a - p.set_ivs[1] / 2).abs(), std::cmp::Reverse(*a)))?;
    let def = (p.set_ivs[2] / 2 & !3) | (index as i32 % 4);
    let mut spe = p.set_ivs[5] / 2;
    let mut spc = p.set_ivs[3] / 2;
    if let Some(obs_hp) = observed_maxhp {
        let base_hp = dex.species.get(p.species).base_stats.hp;
        let hp_dv_hi = (atk & 1) * 8 + (def & 1) * 4;
        let cur_low = (spe & 1) * 2 + (spc & 1);
        if let Some(ev_hp) = find_effort_for_maxhp(base_hp, p.level, hp_dv_hi + cur_low, p.set_evs[0], obs_hp) {
            result.set_evs[0] = ev_hp;
        } else {
            // Try flipping spe/spc LSBs (prefer highest spe/spc values) to realize obs_hp
            let candidates = [
                (spe, spc),
                (spe ^ 1, spc),
                (spe, spc ^ 1),
                (spe ^ 1, spc ^ 1),
            ];
            let mut found = None;
            for (cand_spe, cand_spc) in candidates {
                let hp_dv = hp_dv_hi + (cand_spe & 1) * 2 + (cand_spc & 1);
                if let Some(ev_hp) = find_effort_for_maxhp(base_hp, p.level, hp_dv, p.set_evs[0], obs_hp) {
                    found = Some((cand_spe, cand_spc, ev_hp));
                    break;
                }
            }
            let (cand_spe, cand_spc, ev_hp) = found?;
            spe = cand_spe;
            spc = cand_spc;
            result.set_evs[0] = ev_hp;
        }
    }
    result.set_ivs[1] = 2 * atk;
    result.set_ivs[2] = 2 * def;
    result.set_ivs[3] = 2 * spc;
    result.set_ivs[5] = 2 * spe;
    recompute_mon_stats(dex, &mut result);
    for slot in result.base_move_slots.iter_mut() {
        if dex.moves.key(slot.id).starts_with("hiddenpower") {
            slot.id = dex.moves.id("hiddenpower").unwrap();
        }
    }
    result.move_slots = result.base_move_slots;
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::ItemObs;
    use nc2000_engine::battle::PokemonSet;
    use nc2000_engine::state::Battle;
    use serde_json::json;

    fn set(species: &str, gender: &str, level: u8, ivs: [i32; 6]) -> PokemonSet {
        serde_json::from_value(
            json!({"name":"", "species":species,"level":level,"gender":gender,
            "moves":["hiddenpower","rest"],"item":"leftovers",
            "ivs":{"hp":ivs[0],"atk":ivs[1],"def":ivs[2],"spa":ivs[3],"spd":ivs[4],"spe":ivs[5]},
            "evs":{"hp":252,"atk":248,"def":240,"spa":232,"spd":224,"spe":216}}),
        )
        .unwrap()
    }

    #[test]
    fn projected_genotypes_match_engine_stats_power_and_public_gender() {
        let dex = Dex::from_json(include_str!("../../../data/gen2stadium2.json")).unwrap();
        for species in ["exeggutor", "snorlax"] {
            for (label, gender, atk) in [
                ("M", Gender::M, 30),
                ("F", Gender::F, if species == "snorlax" { 2 } else { 14 }),
            ] {
                for level in [50, 55] {
                    let source = set(species, label, level, [30, atk, 30, 30, 30, 30]);
                    let b = Battle::from_fixture(&dex, "1,2,3,4", &[source.clone()], &[source])
                        .unwrap();
                    let p = &b.sides[0].roster[0];
                    for index in 0..16 {
                        if possible_types(&dex, p.species, gender) & (1 << index) == 0 {
                            continue;
                        }
                        let obs = MonObs {
                            species: p.species,
                            level,
                            gender,
                            name: String::new(),
                            preview_has_item: true,
                            revealed_moves: vec![dex.moves.id("hiddenpower").unwrap()],
                            hp_type_mask: 1 << index,
                            excluded_items: 0,
                            item: ItemObs::default(),
                            appeared: true,
                            observed_maxhp: None,
                            sketch_overlay: None,
                        };
                        let corrected = project(&dex, p, &obs, &mut SplitMix64::new(1))
                            .unwrap_or_else(|| p.clone());
                        assert_eq!(modeled_type(&dex, &corrected), Some(index));
                        assert!(gender_matches(
                            &dex,
                            &corrected,
                            gender,
                            corrected.set_ivs[1] / 2
                        ));
                        let fresh = set(species, label, level, corrected.set_ivs);
                        let built =
                            Battle::from_fixture(&dex, "1,2,3,4", &[fresh.clone()], &[fresh])
                                .unwrap();
                        let expected = &built.sides[0].roster[0];
                        assert_eq!(corrected.base_maxhp, expected.base_maxhp);
                        assert_eq!(corrected.base_stored_stats, expected.base_stored_stats);
                        assert_eq!(corrected.base_hp_type, expected.base_hp_type);
                        assert_eq!(corrected.base_hp_power, expected.base_hp_power);
                        assert_eq!(corrected.set_ivs[4], corrected.set_ivs[3]);
                        let iv = corrected.set_ivs;
                        assert_eq!(
                            iv[0] / 2,
                            ((iv[1] / 2 & 1) * 8
                                + (iv[2] / 2 & 1) * 4
                                + (iv[5] / 2 & 1) * 2
                                + (iv[3] / 2 & 1))
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unknown_and_satisfied_constraints_preserve_rng_while_repairs_sample_support() {
        let dex = Dex::from_json(include_str!("../../../data/gen2stadium2.json")).unwrap();
        let source = set("exeggutor", "M", 50, [30; 6]);
        let b = Battle::from_fixture(&dex, "1,2,3,4", &[source.clone()], &[source]).unwrap();
        let p = &b.sides[0].roster[0];
        let mut obs = MonObs {
            species: p.species,
            level: 50,
            gender: Gender::M,
            name: String::new(),
            preview_has_item: true,
            revealed_moves: vec![dex.moves.id("hiddenpower").unwrap()],
            hp_type_mask: u16::MAX,
            excluded_items: 0,
            item: ItemObs::default(),
            appeared: true,
            observed_maxhp: None,
            sketch_overlay: None,
        };
        for mask in [u16::MAX, 1 << 15] {
            obs.hp_type_mask = mask;
            let mut rng = SplitMix64::new(7);
            assert!(project(&dex, p, &obs, &mut rng).is_none());
            assert_eq!(rng.next(), SplitMix64::new(7).next());
        }
        obs.hp_type_mask = (1 << 0) | (1 << 4);
        let mut seen = 0;
        for seed in 0..64 {
            let projected = project(&dex, p, &obs, &mut SplitMix64::new(seed)).unwrap();
            let index = modeled_type(&dex, &projected).unwrap();
            assert!(index == 0 || index == 4);
            seen |= 1 << index;
        }
        assert_eq!(seen, obs.hp_type_mask);
    }

    #[test]
    fn effectiveness_masks_match_gen2_typechart_and_deterministic_projection() {
        let dex = Dex::from_json(include_str!("../../../data/gen2stadium2.json")).unwrap();
        let snorlax_id = dex.species.id("snorlax").unwrap();
        let skarm_id = dex.species.id("skarmory").unwrap();
        let zapdos_id = dex.species.id("zapdos").unwrap();

        let snorlax_types = dex.species_types(snorlax_id);
        let ghost_bit = 1u16 << TYPES.iter().position(|t| *t == "Ghost").unwrap();
        let fighting_bit = 1u16 << TYPES.iter().position(|t| *t == "Fighting").unwrap();
        assert_eq!(
            types_with_effectiveness(&dex, &snorlax_types, Effectiveness::Immune),
            ghost_bit
        );
        assert_eq!(
            types_with_effectiveness(&dex, &snorlax_types, Effectiveness::SuperEffective),
            fighting_bit
        );
        assert_eq!(
            types_with_effectiveness(&dex, &snorlax_types, Effectiveness::Neutral),
            u16::MAX ^ ghost_bit ^ fighting_bit
        );

        let skarm_types = dex.species_types(skarm_id);
        let ground_bit = 1u16 << TYPES.iter().position(|t| *t == "Ground").unwrap();
        let poison_bit = 1u16 << TYPES.iter().position(|t| *t == "Poison").unwrap();
        let fire_bit = 1u16 << TYPES.iter().position(|t| *t == "Fire").unwrap();
        let electric_bit = 1u16 << TYPES.iter().position(|t| *t == "Electric").unwrap();
        assert_eq!(
            types_with_effectiveness(&dex, &skarm_types, Effectiveness::Immune),
            ground_bit | poison_bit
        );
        assert_eq!(
            types_with_effectiveness(&dex, &skarm_types, Effectiveness::SuperEffective),
            fire_bit | electric_bit
        );

        let zapdos_types = dex.species_types(zapdos_id);
        let ice_bit = 1u16 << TYPES.iter().position(|t| *t == "Ice").unwrap();
        let rock_bit = 1u16 << TYPES.iter().position(|t| *t == "Rock").unwrap();
        assert_eq!(
            types_with_effectiveness(&dex, &zapdos_types, Effectiveness::SuperEffective),
            ice_bit | rock_bit
        );

        // Deterministic projection picks lowest legal bit without RNG and keeps Hidden Power consistent.
        let source = set("zapdos", "N", 50, [30; 6]); // Dark
        let b = Battle::from_fixture(&dex, "1,2,3,4", &[source.clone()], &[source]).unwrap();
        let p = &b.sides[0].roster[0];
        let mask = fire_bit | ice_bit;
        let det1 = project_deterministic(&dex, p, Gender::N, mask).unwrap();
        let det2 = project_deterministic(&dex, p, Gender::N, mask).unwrap();
        assert_eq!(det1.set_ivs, det2.set_ivs);
        assert_eq!(modeled_type(&dex, &det1), Some(TYPES.iter().position(|t| *t == "Fire").unwrap()));
        assert!(consistent(&dex, &det1, mask));
    }

    #[test]
    fn project_respects_observed_maxhp_and_astra_c3() {
        let dex = Dex::from_json(include_str!("../../../data/gen2stadium2.json")).unwrap();
        let fire_idx = TYPES.iter().position(|t| *t == "Fire").unwrap();
        let ice_idx = TYPES.iter().position(|t| *t == "Ice").unwrap();
        let fire_bit = 1u16 << fire_idx;
        let ice_bit = 1u16 << ice_idx;

        // Zapdos Lv55 with HP Fire (Atk=14, Def=12, Spe=15, Spc=15 -> HP DV=3, max HP at 255 EV = 201)
        let hp_fire_zap = set("zapdos", "N", 55, [6, 28, 24, 30, 30, 30]);
        let b_fire =
            Battle::from_fixture(&dex, "1,2,3,4", &[hp_fire_zap.clone()], &[hp_fire_zap]).unwrap();
        let p_fire = &b_fire.sides[0].roster[0];
        assert_eq!(modeled_type(&dex, p_fire), Some(fire_idx));
        assert_eq!(p_fire.base_maxhp, 201);

        // At observed_maxhp = 215 (only HP DV == 15 possible): HP Fire cannot realize 215,
        // so project() MUST switch away from Fire to Ice and realize base_maxhp == 215.
        let mut obs = MonObs {
            species: p_fire.species,
            level: 55,
            gender: Gender::N,
            name: "Zapdos".into(),
            preview_has_item: true,
            hp_type_mask: fire_bit | ice_bit,
            revealed_moves: Vec::new(),
            excluded_items: 0,
            item: ItemObs::default(),
            appeared: true,
            observed_maxhp: Some(215),
            sketch_overlay: None,
        };
        let proj_215 = project(&dex, p_fire, &obs, &mut SplitMix64::new(7)).unwrap();
        assert_eq!(modeled_type(&dex, &proj_215), Some(ice_idx));
        assert_eq!(proj_215.base_maxhp, 215);

        // §1 Astra C3: At observed_maxhp = 205 (lower HP EV on HP Ice Zapdos Lv55),
        // HP Fire is still unrealizable (max 201), while HP Ice IS realizable by lowering HP EV!
        obs.observed_maxhp = Some(205);
        let proj_205 = project(&dex, p_fire, &obs, &mut SplitMix64::new(11)).unwrap();
        assert_eq!(modeled_type(&dex, &proj_205), Some(ice_idx));
        assert_eq!(proj_205.base_maxhp, 205);
        assert!(proj_205.set_evs[0] < 255);

        // Starting from HP Ice Zapdos Lv55 (215 HP) when observed_maxhp = 205: keeps HP Ice, adjusts HP EV.
        let hp_ice_zap = set("zapdos", "N", 55, [30, 30, 26, 30, 30, 30]);
        let b_ice =
            Battle::from_fixture(&dex, "1,2,3,4", &[hp_ice_zap.clone()], &[hp_ice_zap]).unwrap();
        let p_ice = &b_ice.sides[0].roster[0];
        assert_eq!(modeled_type(&dex, p_ice), Some(ice_idx));
        assert_eq!(p_ice.base_maxhp, 215);
        let proj_ice_205 = project(&dex, p_ice, &obs, &mut SplitMix64::new(13)).unwrap();
        assert_eq!(modeled_type(&dex, &proj_ice_205), Some(ice_idx));
        assert_eq!(proj_ice_205.base_maxhp, 205);
    }
}
