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

pub(crate) fn project(
    dex: &Dex,
    p: &Pokemon,
    mo: &MonObs,
    rng: &mut SplitMix64,
) -> Option<Pokemon> {
    if consistent(dex, p, mo.hp_type_mask) || modeled_type(dex, p).is_none() {
        return None;
    }
    let mut choices = Vec::new();
    for i in 0..16 {
        if mo.hp_type_mask & (1 << i) != 0
            && (0..16).any(|atk| atk % 4 == i / 4 && gender_matches(dex, p, mo.gender, atk))
        {
            choices.push(i as usize);
        }
    }
    debug_assert!(
        !choices.is_empty(),
        "Hidden Power type contradicts public gender"
    );
    let index = *choices.get(if choices.len() > 1 {
        rng.below(choices.len())
    } else {
        0
    })?;
    project_with_index(dex, p, mo.gender, index)
}

pub(crate) fn project_deterministic(
    dex: &Dex,
    p: &Pokemon,
    gender: Gender,
    mask: u16,
) -> Option<Pokemon> {
    if mask == 0 || mask == u16::MAX || consistent(dex, p, mask) || modeled_type(dex, p).is_none() {
        return None;
    }
    let legal = mask & possible_types(dex, p.species, gender);
    let eff_mask = if legal != 0 { legal } else { mask };
    let index = eff_mask.trailing_zeros() as usize;
    if index >= TYPES.len() {
        return None;
    }
    project_with_index(dex, p, gender, index)
}

fn project_with_index(
    dex: &Dex,
    p: &Pokemon,
    gender: Gender,
    index: usize,
) -> Option<Pokemon> {
    let mut result = p.clone();
    let atk = (0..16)
        .filter(|a| a % 4 == index as i32 / 4 && gender_matches(dex, p, gender, *a))
        .min_by_key(|a| ((a - p.set_ivs[1] / 2).abs(), std::cmp::Reverse(*a)))?;
    let def = (p.set_ivs[2] / 2 & !3) | (index as i32 % 4);
    let spe = p.set_ivs[5] / 2;
    let spc = p.set_ivs[3] / 2;
    result.set_ivs[0] = 2 * ((atk & 1) * 8 + (def & 1) * 4 + (spe & 1) * 2 + (spc & 1));
    result.set_ivs[1] = 2 * atk;
    result.set_ivs[2] = 2 * def;
    result.set_ivs[4] = result.set_ivs[3];
    result.base_hp_type = dex.type_id(TYPES[index]).unwrap();
    result.hp_type = result.base_hp_type;
    result.base_hp_power =
        (5 * ((spc >> 3) + 2 * (spe >> 3) + 4 * (def >> 3) + 8 * (atk >> 3)) + spc % 4) / 2 + 31;
    result.hp_power = result.base_hp_power;
    let stats = &dex.species.get(p.species).base_stats;
    let bases = [
        stats.hp, stats.atk, stats.def, stats.spa, stats.spd, stats.spe,
    ];
    let computed: [i32; 6] = std::array::from_fn(|i| {
        (2 * i32::from(bases[i]) + result.set_ivs[i] + result.set_evs[i] / 4) * i32::from(p.level)
            / 100
            + if i == 0 { i32::from(p.level) + 10 } else { 5 }
    });
    result.base_maxhp = computed[0];
    result.maxhp = computed[0];
    result.hp = computed[0];
    result.base_stored_stats.copy_from_slice(&computed[1..]);
    result.stored_stats = result.base_stored_stats;
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
}
