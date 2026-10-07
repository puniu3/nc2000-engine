use nc2000_bot::eval::expected_hit_fraction;
use nc2000_engine::battle::{enumerate::enumerate_step, PokemonSet, SearchChoice};
use nc2000_engine::dex::Dex;
use nc2000_engine::state::Battle;
use serde_json::json;
use std::collections::BTreeMap;

pub const TYPES: [&str; 16] = [
    "Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel", "Fire", "Water",
    "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark",
];

pub fn fixture(dex: &Dex, index: usize, high: bool) -> Battle {
    let mon = |species: &str, moves: Vec<&str>| {
        serde_json::from_value::<PokemonSet>(json!({"name":species,"species":species,"level":50,"moves":moves,"ivs":{"hp":30,"atk":30,"def":30,"spa":30,"spd":30,"spe":30},"evs":{"hp":255,"atk":255,"def":255,"spa":255,"spd":255,"spe":255}})).unwrap()
    };
    let atk = if high { 12 } else { 0 } + index as i32 / 4;
    let def = if high { 12 } else { 0 } + index as i32 % 4;
    let speed = if high { 15 } else { 1 };
    let hp = (atk & 1) * 8 + (def & 1) * 4 + (speed & 1) * 2 + 1;
    let mut own = vec![
        mon("Alakazam", vec!["Hidden Power"]),
        mon("Snorlax", vec!["Tackle"]),
        mon("Tauros", vec!["Tackle"]),
    ];
    own[0].ivs = Some(
        ["hp", "atk", "def", "spa", "spd", "spe"]
            .into_iter()
            .zip([hp * 2, atk * 2, def * 2, 30, 30, speed * 2])
            .map(|(k, v)| (k.to_string(), v as u8))
            .collect(),
    );
    let foe = vec![
        mon("Mew", vec!["Swords Dance"]),
        mon("Miltank", vec!["Tackle"]),
        mon("Starmie", vec!["Tackle"]),
    ];
    let mut b = Battle::from_fixture(dex, "1,2,3,4", &own, &foe).unwrap();
    let picks = [0, 1].map(|s| Some(b.legal_choices(dex, s)[0]));
    b.apply_choices(dex, picks).unwrap();
    let user = b.poke(b.active_id(0).unwrap());
    assert_eq!(dex.type_name(user.hp_type), TYPES[index]);
    assert_eq!(user.hp_power, if high { 70 } else { 35 });
    b
}

pub fn estimate(dex: &Dex, b: &Battle) -> f64 {
    let att = b.active_id(0).unwrap();
    let def = b.active_id(1).unwrap();
    expected_hit_fraction(b, dex, att, def, dex.moves.id("hiddenpower").unwrap(), true)
        * f64::from(b.poke(def).hp)
}

pub fn damage(dex: &Dex, b: &Battle) -> (BTreeMap<i32, f64>, u64) {
    let def = b.active_id(1).unwrap();
    let hp = b.poke(def).hp;
    let picks = [
        Some(SearchChoice::Move(dex.moves.id("hiddenpower").unwrap())),
        Some(SearchChoice::Move(dex.moves.id("swordsdance").unwrap())),
    ];
    let step = enumerate_step(dex, b, picks, 200_000).unwrap();
    let mut distribution = BTreeMap::new();
    for leaf in step.leaves {
        *distribution
            .entry(hp - leaf.battle.poke(def).hp)
            .or_insert(0.0) += leaf.prob;
    }
    assert!((distribution.values().sum::<f64>() - 1.0).abs() < 1e-10);
    (distribution, step.runs as u64)
}

pub fn perturb(b: &Battle, index: usize, attacker: bool, relevant: bool) -> Battle {
    let mut b = b.clone();
    let special = index >= 8;
    let side = usize::from(!attacker);
    let id = b.active_id(side).unwrap();
    let stat = if special ^ (!relevant) {
        if attacker {
            2
        } else {
            3
        }
    } else {
        if attacker {
            0
        } else {
            1
        }
    };
    b.poke_mut(id).stored_stats[stat] *= 2;
    b.poke_mut(id).base_stored_stats[stat] *= 2;
    b
}
